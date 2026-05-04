from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import orjson
from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.encoding_dsv32 import DS32EncodingError
from sglang.srt.entrypoints.openai.protocol import ErrorResponse, OpenAIServingRequest
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.otel_payload_logger import chatcmpl_rid_from_headers

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        self.allowed_custom_labels = (
            set(
                self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            )
            if isinstance(self.tokenizer_manager.server_args, ServerArgs)
            and self.tokenizer_manager.server_args.tokenizer_metrics_allowed_custom_labels
            else None
        )

    def _parse_model_parameter(self, model: str) -> Tuple[str, Optional[str]]:
        """Parse 'base-model:adapter-name' syntax to extract LoRA adapter.

        Returns (base_model, adapter_name) or (model, None) if no colon present.
        """
        if ":" not in model:
            return model, None

        # Split on first colon only to handle model paths with multiple colons
        parts = model.split(":", 1)
        base_model = parts[0].strip()
        adapter_name = parts[1].strip() or None

        return base_model, adapter_name

    def _resolve_lora_path(
        self,
        request_model: str,
        explicit_lora_path: Optional[Union[str, List[Optional[str]]]],
    ) -> Optional[Union[str, List[Optional[str]]]]:
        """Resolve LoRA adapter with priority: model parameter > explicit lora_path.

        Returns adapter name or None. Supports both single values and lists (batches).
        """
        _, adapter_from_model = self._parse_model_parameter(request_model)

        # Model parameter adapter takes precedence
        if adapter_from_model is not None:
            return adapter_from_model

        # Fall back to explicit lora_path
        return explicit_lora_path

    def _observe_openai_request_metrics(
        self, request: OpenAIServingRequest, payload: Optional[dict]
    ) -> None:
        if not getattr(self.tokenizer_manager, "enable_metrics", False):
            return
        metrics_collector = getattr(self.tokenizer_manager, "metrics_collector", None)
        if metrics_collector is None or payload is None:
            return

        image_count = video_count = audio_count = 0
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "image_url":
                    image_count += 1
                elif part_type == "video_url":
                    video_count += 1
                elif part_type == "audio_url":
                    audio_count += 1

        tools = payload.get("tools")
        tool_count = len(tools) if isinstance(tools, list) else 0
        tool_choice = payload.get("tool_choice")
        if isinstance(tool_choice, dict):
            tool_choice = "function"
        elif tool_choice is None:
            tool_choice = getattr(request, "tool_choice", None)

        structured_output_kind = None
        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            structured_output_kind = response_format.get("type")
        structured_outputs = payload.get("structured_outputs")
        if structured_output_kind is None and isinstance(structured_outputs, dict):
            for key in ("json", "json_object", "regex", "grammar", "choice"):
                if structured_outputs.get(key) is not None:
                    structured_output_kind = key
                    break
        for attr_name, kind in (("regex", "regex"), ("ebnf", "grammar")):
            if structured_output_kind is None and payload.get(attr_name) is not None:
                structured_output_kind = kind

        metrics_collector.observe_openai_request(
            dict(metrics_collector.labels),
            image_count=image_count,
            video_count=video_count,
            audio_count=audio_count,
            tool_count=tool_count,
            tool_choice=str(tool_choice) if tool_choice is not None else None,
            structured_output_kind=structured_output_kind,
            structured_output_backend="xgrammar" if structured_output_kind else None,
        )

    def _error_log_payload(self, response: ORJSONResponse) -> dict:
        payload = json.loads(response.body)
        if isinstance(payload, dict) and "error" not in payload:
            return {"error": payload}
        return payload

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern
        If you want to override this method, you should be careful to record the validation time.
        """
        received_time = monotonic_time()
        request_logger = self.tokenizer_manager.request_logger
        header_rid = chatcmpl_rid_from_headers(raw_request.headers)
        if (
            header_rid
            and hasattr(request, "rid")
            and getattr(request, "rid", None) is None
        ):
            request.rid = header_rid
        raw_payload = None
        try:
            raw_payload = json.loads(await raw_request.body())
        except Exception:
            raw_payload = None

        try:
            served_model_name = self.tokenizer_manager.served_model_name
            served_model_names = getattr(
                self.tokenizer_manager,
                "served_model_names",
                [served_model_name],
            )
            request_model = getattr(request, "model", None)
            if request_model:
                base_model, adapter_name = self._parse_model_parameter(request_model)
                if base_model not in served_model_names:
                    response = self.create_error_response(
                        message=f"Model {request_model!r} is not served.",
                        err_type="NotFound",
                        status_code=404,
                    )
                    request_logger.log_openai_response(
                        self._error_log_payload(response),
                        request=raw_request,
                        rid=getattr(request, "rid", None),
                    )
                    return response
                if base_model != served_model_name:
                    canonical_model = served_model_name
                    if adapter_name is not None:
                        canonical_model = f"{served_model_name}:{adapter_name}"
                    request = request.model_copy(update={"model": canonical_model})

            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                response = self.create_error_response(error_msg)
                request_logger.log_openai_received_request(
                    raw_payload if isinstance(raw_payload, dict) else request,
                    request=raw_request,
                )
                request_logger.log_openai_response(
                    self._error_log_payload(response),
                    request=raw_request,
                    rid=getattr(request, "rid", None),
                )
                return response

            # Log the raw OpenAI request payload before conversion to tokenized form.
            request_logger.log_openai_received_request(
                raw_payload if isinstance(raw_payload, dict) else request,
                request=raw_request,
            )
            if isinstance(raw_payload, dict):
                self._observe_openai_request_metrics(request, raw_payload)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )

            if isinstance(adapted_request, (GenerateReqInput, EmbeddingReqInput)):
                # Only set timing fields if adapted_request supports them
                adapted_request.received_time = received_time

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                response = await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
                if not isinstance(response, StreamingResponse):
                    request_logger.log_openai_response(
                        response,
                        request=raw_request,
                        rid=getattr(request, "rid", None),
                )
                return response
        except HTTPException as e:
            response = self.create_error_response(
                message=e.detail, err_type=str(e.status_code), status_code=e.status_code
            )
            request_logger.log_openai_response(
                self._error_log_payload(response),
                request=raw_request,
                rid=getattr(request, "rid", None),
            )
            return response
        except ValueError as e:
            message = str(e)
            if "grammar" in message.lower() and not message.startswith("Grammar error:"):
                message = f"Grammar error: {message}"
            response = self.create_error_response(
                message=message,
                err_type="BadRequest",
                status_code=400,
            )
            request_logger.log_openai_response(
                self._error_log_payload(response),
                request=raw_request,
                rid=getattr(request, "rid", None),
            )
            return response
        except DS32EncodingError as e:
            logger.info(f"DS32EncodingError: {e}")
            response = self.create_error_response(
                message=str(e),
                err_type="BadRequest",
                status_code=400,
            )
            request_logger.log_openai_response(
                self._error_log_payload(response),
                request=raw_request,
                rid=getattr(request, "rid", None),
            )
            return response
        except Exception as e:
            logger.exception(f"Error in request: {e}")
            response = self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )
            request_logger.log_openai_response(
                self._error_log_payload(response),
                request=raw_request,
                rid=getattr(request, "rid", None),
            )
            return response

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> Optional[str]:
        """Generate request ID based on request type"""
        return None

        # TODO(chang): the rid is used in io_strcut check and often violates `The rid should be a list` AssertionError
        # Temporarily return None in this function until the rid logic is clear.
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    def _compute_extra_key(self, request: OpenAIServingRequest) -> Optional[str]:
        """Compute the final extra_key by concatenating cache_salt and extra_key if both are provided."""
        parts = []
        for key in ["cache_salt", "extra_key"]:
            value = getattr(request, key, None)
            if value:
                if not isinstance(value, str):
                    raise TypeError(
                        f"Value of {key} must be a string, but got {type(value).__name__}"
                    )
                parts.append(value)
        return "".join(parts) if parts else None

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse, ORJSONResponse]:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Union[Any, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> Optional[str]:
        """Validate request"""
        pass

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: Optional[str] = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        # TODO: remove fastapi dependency in openai and move response handling to the entrypoint
        if "grammar" in message.lower() and not message.startswith("Grammar error:"):
            message = f"Grammar error: {message}"
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(
            content={"error": error.model_dump()}, status_code=status_code
        )

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})

    def extract_custom_labels(self, raw_request):
        if (
            not self.allowed_custom_labels
            or not self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        ):
            return None

        custom_labels = None
        header = (
            self.tokenizer_manager.server_args.tokenizer_metrics_custom_labels_header
        )
        try:
            raw_labels = (
                orjson.loads(raw_request.headers.get(header))
                if raw_request and raw_request.headers.get(header)
                else None
            )
        except json.JSONDecodeError as e:
            logger.exception(f"Error in request: {e}")
            raw_labels = None

        if isinstance(raw_labels, dict):
            custom_labels = {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return custom_labels

    def extract_routing_key(self, raw_request):
        if raw_request is None:
            return None
        return raw_request.headers.get("x-smg-routing-key")

    def extract_routed_dp_rank_from_header(
        self, raw_request: Request, body_routed_dp_rank: Optional[int] = None
    ) -> Optional[int]:
        """Extract routed_dp_rank from HTTP header, with higher priority than routed_dp_rank in body.

        Header name: X-Data-Parallel-Rank (case-insensitive in HTTP/1.1/2)
        """
        if raw_request is None:
            return body_routed_dp_rank

        header_value = raw_request.headers.get("x-data-parallel-rank")
        if header_value is not None:
            try:
                header_dp_rank = int(header_value)
                if (
                    body_routed_dp_rank is not None
                    and header_dp_rank != body_routed_dp_rank
                ):
                    logger.debug(
                        f"X-Data-Parallel-Rank header ({header_dp_rank}) overrides "
                        f"body routed_dp_rank ({body_routed_dp_rank})"
                    )
                return header_dp_rank
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid X-Data-Parallel-Rank header: must be an integer, got '{header_value}'",
                )

        return body_routed_dp_rank
