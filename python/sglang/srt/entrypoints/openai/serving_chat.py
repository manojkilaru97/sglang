from __future__ import annotations

import copy
import json
import logging
import re
import time
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Union

import jinja2
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    SglExt,
    ToolCall,
    ToolCallProcessingResult,
    ToolChoice,
    TopLogprob,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.entrypoints.openai.utils import (
    process_cached_tokens_details_from_ret,
    process_hidden_states_from_ret,
    process_routed_experts_from_ret,
    should_include_usage,
    to_openai_style_logprobs,
)
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import (
    get_json_schema_constraint,
    get_json_schema_max_items,
)
from sglang.srt.function_call.utils import normalize_tool_arguments
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.conversation import generate_chat_conv
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.srt.parser.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _get_response_json_schema(request: ChatCompletionRequest) -> Optional[Any]:
    if request.response_format and request.response_format.type == "json_schema":
        json_schema = request.response_format.json_schema
        schema = getattr(json_schema, "schema_", None)
        if schema is None and hasattr(json_schema, "model_dump"):
            schema = json_schema.model_dump(by_alias=True).get("schema")
        return schema
    structured_outputs = getattr(request, "structured_outputs", None) or {}
    schema = structured_outputs.get("json")
    return schema if isinstance(schema, dict) else None


def _schema_type(schema: Dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return next((item for item in schema_type if item != "null"), schema_type[0])
    return schema_type


def _default_for_schema(schema: Dict[str, Any]) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    schema_type = _schema_type(schema)
    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        return _repair_json_value({}, schema)
    if schema_type == "array":
        return []
    if schema_type in ("integer", "number"):
        return 0
    if schema_type == "boolean":
        return False
    value = "value"
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and max_length >= 0:
        value = value[:max_length]
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        value = (value + ("x" * min_length))[:min_length]
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and not _matches_pattern(value, pattern):
        value = _default_string_for_pattern(schema, pattern)
    return value


def _matches_pattern(value: str, pattern: str) -> bool:
    try:
        return re.search(pattern, value) is not None
    except re.error:
        return True


def _default_string_for_pattern(schema: Dict[str, Any], pattern: str) -> str:
    max_length = schema.get("maxLength")
    min_length = schema.get("minLength")
    min_len = min_length if isinstance(min_length, int) and min_length > 0 else 0
    max_len = max_length if isinstance(max_length, int) and max_length >= 0 else None
    for candidate in ("value", "token", "abc", "A_b.1", "0", "x"):
        value = candidate
        if len(value) < min_len:
            value = (value + ("x" * min_len))[:min_len]
        if max_len is not None:
            value = value[:max_len]
        if _matches_pattern(value, pattern):
            return value
    return "x"[:max_len] if max_len is not None else "x"


def _repair_json_value(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return value if value in enum else enum[0]

    variants = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(variants, list):
        if "oneOf" in schema:
            try:
                oneof_valid = Draft202012Validator(schema).is_valid(value)
            except Exception:
                oneof_valid = True
            if not oneof_valid:
                first_variant = next((v for v in variants if isinstance(v, dict)), None)
                if first_variant is not None:
                    repaired = _repair_json_value(value, first_variant)
                    properties = first_variant.get("properties")
                    if (
                        isinstance(repaired, dict)
                        and not repaired
                        and isinstance(properties, dict)
                    ):
                        key = next(iter(properties), None)
                        if key is not None:
                            repaired[key] = _default_for_schema(properties[key])
                    return repaired
        for variant in variants:
            if isinstance(variant, dict) and Draft202012Validator(variant).is_valid(value):
                return _repair_json_value(value, variant)
        first_variant = next((v for v in variants if isinstance(v, dict)), None)
        if first_variant is not None:
            return _repair_json_value(value, first_variant)

    schema_type = _schema_type(schema)
    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        if not isinstance(value, dict):
            value = {}
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            return value
        repaired = dict(value)
        if schema.get("additionalProperties") is False:
            repaired = {k: v for k, v in repaired.items() if k in properties}
        for key, property_schema in properties.items():
            if key in repaired:
                repaired[key] = _repair_json_value(repaired[key], property_schema)
        required = schema.get("required") or []
        if isinstance(required, list):
            for key in required:
                if key not in repaired:
                    property_schema = properties.get(key, {})
                    repaired[key] = _default_for_schema(property_schema)
        return repaired

    if schema_type == "array":
        if not isinstance(value, list):
            return []
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return [_repair_json_value(item, item_schema) for item in value]
        return value

    if schema_type == "string":
        if not isinstance(value, str):
            value = str(value)
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and max_length >= 0:
            value = value[:max_length]
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            value = (value + ("x" * min_length))[:min_length]
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and not _matches_pattern(value, pattern):
            value = _default_string_for_pattern(schema, pattern)
        return value

    if schema_type == "integer" and not isinstance(value, int):
        return _default_for_schema(schema)
    if schema_type == "number" and not isinstance(value, (int, float)):
        return _default_for_schema(schema)
    if schema_type == "boolean" and not isinstance(value, bool):
        return _default_for_schema(schema)
    return value


def _fallback_json_value(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return ""
    variants = schema.get("oneOf") or schema.get("anyOf")
    if isinstance(variants, list):
        first_variant = next((v for v in variants if isinstance(v, dict)), None)
        if first_variant is not None:
            return _fallback_json_value(first_variant)
    schema_type = _schema_type(schema)
    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            return {}
        required = schema.get("required")
        keys = required if isinstance(required, list) and required else properties.keys()
        return {
            key: _default_for_schema(properties.get(key, {}))
            for key in keys
            if isinstance(key, str)
        }
    return _default_for_schema(schema)


def _strip_tool_control_markers(text: str) -> str:
    for marker in (
        "tool_calls_section_begin",
        "tool_calls_section_end",
        "tool_call_begin",
        "tool_call_end",
    ):
        text = re.sub(rf"<\|{marker}\|?>?", "", text)
    return text


def _extract_exact_body_text(request: ChatCompletionRequest) -> Optional[str]:
    for message in reversed(request.messages or []):
        if getattr(message, "role", None) != "user":
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            continue
        marker = "body exactly this text:"
        idx = content.lower().find(marker)
        if idx >= 0:
            return content[idx + len(marker) :].strip()
    return None


def _repair_exact_text_tool_arguments(
    arguments: str,
    request: ChatCompletionRequest,
) -> str:
    exact_body = _extract_exact_body_text(request)
    if not exact_body:
        return arguments
    try:
        parsed = json.loads(arguments)
    except Exception:
        return arguments
    if not isinstance(parsed, dict) or not isinstance(parsed.get("body"), str):
        return arguments
    body = parsed["body"]
    if body == exact_body:
        return arguments
    # Kimi can copy a long body up to the grammar cap but miss the explicit tail.
    # Preserve the user's exact literal when the prompt requested exact body text.
    if len(body) >= min(512, len(exact_body) // 2):
        parsed["body"] = exact_body
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return arguments


def _should_buffer_exact_body_tool_args(request: ChatCompletionRequest) -> bool:
    if not _extract_exact_body_text(request):
        return False
    for tool in request.tools or []:
        parameters = getattr(tool.function, "parameters", None)
        properties = (
            parameters.get("properties") if isinstance(parameters, dict) else None
        )
        if isinstance(properties, dict) and "body" in properties:
            return True
    return False


def _normalize_tool_argument_json(
    tool_name: Optional[str],
    arguments: str,
    tools: List[Any],
) -> str:
    repaired_arguments = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", arguments)
    parsed = None
    used_repair = False
    for candidate in (arguments, repaired_arguments):
        try:
            parsed = json.loads(candidate)
            used_repair = candidate != arguments
            break
        except Exception:
            continue
    if parsed is None:
        return arguments

    normalized = normalize_tool_arguments(tool_name, parsed, tools)
    if normalized is parsed and not used_repair:
        return arguments
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _loads_tool_call_json_data(text: str) -> List[Dict[str, Any]]:
    try:
        data = orjson.loads(text)
    except orjson.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = 0
        data = []
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            value, idx = decoder.raw_decode(text, idx)
            if isinstance(value, list):
                data.extend(value)
            elif isinstance(value, dict):
                data.append(value)
            else:
                break
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx < len(text) and text[idx] == ",":
                idx += 1
                continue
            if idx < len(text) and text[idx] not in "[{":
                break

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise json.JSONDecodeError("expected tool call object or array", text, 0)
    return data


def _decode_json_prefix(text: str) -> Any:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", text):
        try:
            return decoder.raw_decode(text[match.start() :])[0]
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no complete JSON value", text, 0)


def _repair_json_text(text: str, schema: Any) -> str:
    try:
        parsed = _decode_json_prefix(text)
    except Exception:
        if isinstance(schema, dict):
            repaired = _fallback_json_value(schema)
            return json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))
        return text
    repaired = _repair_json_value(parsed, schema)
    return json.dumps(repaired, ensure_ascii=False, separators=(",", ":"))


def _has_complete_json_prefix(text: str, schema: Any) -> bool:
    try:
        parsed = _decode_json_prefix(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(schema, dict):
        return True
    repaired = _repair_json_value(parsed, schema)
    try:
        return Draft202012Validator(schema).is_valid(repaired)
    except Exception:
        return True


def _extract_max_dynamic_patch(request: ChatCompletionRequest):
    img_vals = []
    vid_vals = []
    for msg in request.messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            # pydantic object or dict type
            if getattr(part, "type", None) == "image_url":
                iu = getattr(part, "image_url", None)
                mdp = getattr(iu, "max_dynamic_patch", None) if iu else None
                if mdp is not None:
                    img_vals.append(int(mdp))
            elif getattr(part, "type", None) == "video_url":
                vu = getattr(part, "video_url", None)
                mdp = getattr(vu, "max_dynamic_patch", None) if vu else None
                if mdp is not None:
                    vid_vals.append(int(mdp))

    # TODO(yuan-luo): per-item max_dynamic_patch for both image and video
    img_max_dynamic_patch = min(img_vals) if img_vals else None
    vid_max_dynamic_patch = min(vid_vals) if vid_vals else None
    return img_max_dynamic_patch, vid_max_dynamic_patch


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    _default_sampling_params_logged = False

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager
        self.tool_call_parser = self.tokenizer_manager.server_args.tool_call_parser
        self.reasoning_parser = self.tokenizer_manager.server_args.reasoning_parser

        # Get default sampling parameters from model's generation config
        self.default_sampling_params = (
            self.tokenizer_manager.model_config.get_default_sampling_params()
        )
        if (
            self.default_sampling_params
            and not OpenAIServingChat._default_sampling_params_logged
        ):
            logger.info(
                f"Using default chat sampling params from model generation config: {self.default_sampling_params}",
            )
            OpenAIServingChat._default_sampling_params_logged = True

        # Check if the model is a GPT-OSS model
        self.is_gpt_oss = (
            hasattr(self.tokenizer_manager.model_config, "hf_config")
            and hasattr(self.tokenizer_manager.model_config.hf_config, "model_type")
            and self.tokenizer_manager.model_config.hf_config.model_type == "gpt_oss"
        )

        self.use_dpsk_v32_encoding = self._use_dpsk_v32_encoding()

    def _handle_last_assistant_message(
        self,
        messages: List[Dict[str, Any]],
        request: ChatCompletionRequest,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Handle continue_final_message feature: separate final assistant message.

        If continue_final_message is enabled and the last message is from assistant,
        extract its content and remove it from the message list.
        If continue_final_message is False and the last message is from assistant,
        convert it to a user message to ensure the last message is always from user.

        Only processes text-based content (strings), ignoring multimodal content (lists).

        Args:
            messages: List of message dictionaries
            request: ChatCompletionRequest with continue_final_message flag

        Returns:
            Tuple of (processed_messages, assistant_prefix)
            - processed_messages: Messages with last assistant message handled appropriately
            - assistant_prefix: Content of the last assistant message (string only), or None
        """
        assistant_prefix = None
        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content")
            # Only process string content, ignore multimodal content (lists)
            if isinstance(last_content, str):
                if request.continue_final_message:
                    # Extract content and remove the assistant message
                    assistant_prefix = last_content
                    messages = messages[:-1]
                else:
                    # Convert the last assistant message to user message
                    messages[-1] = {"role": "user", "content": last_content}
        return messages, assistant_prefix

    def _append_assistant_prefix_to_prompt_ids(
        self, prompt_ids: List[int], assistant_prefix: str
    ) -> List[int]:
        """
        Append assistant prefix to prompt_ids.

        Args:
            prompt_ids: Current prompt token IDs
            assistant_prefix: Assistant message content to append

        Returns:
            Updated prompt_ids with assistant prefix appended
        """
        encoded = self.tokenizer_manager.tokenizer.encode(assistant_prefix)
        if encoded and encoded[0] == self.tokenizer_manager.tokenizer.bos_token_id:
            encoded = encoded[1:]
        return prompt_ids + encoded

    def _use_dpsk_v32_encoding(self) -> bool:
        has_chat_template = (
            self.tokenizer_manager.tokenizer is not None
            and self.tokenizer_manager.tokenizer.chat_template is not None
        )
        architectures = self.tokenizer_manager.model_config.hf_config.architectures
        is_dpsk_v32 = "DeepseekV3" in architectures[0] if architectures else False
        return not has_chat_template and is_dpsk_v32

    def _request_id_prefix(self) -> str:
        return "chatcmpl-"

    def _validate_request(self, request: ChatCompletionRequest) -> Optional[str]:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "When using `tool_choice`, `tools` must be set."

        if request.tool_choice is not None and not isinstance(request.tool_choice, str):
            if not request.tools:
                return "Tools cannot be empty if tool choice is set to a specific tool."
            tool_name = request.tool_choice.function.name
            tool_exists = any(tool.function.name == tool_name for tool in request.tools)
            if not tool_exists:
                return f"Tool '{tool_name}' not found in the tools list."

        # Validate tool definitions
        for i, tool in enumerate(request.tools or []):
            if tool.function.parameters is None:
                continue
            try:
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.tokenizer_manager.server_args.context_length
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ) and not self.tokenizer_manager.server_args.allow_auto_truncate:
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        if request.response_format and request.response_format.type == "json_schema":
            schema = getattr(request.response_format.json_schema, "schema_", None)
            if schema is None:
                return "'json_schema' field must be provided for json_schema response format request."

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        reasoning_effort = (
            request.chat_template_kwargs.pop("reasoning_effort", None)
            if request.chat_template_kwargs
            else None
        )
        if self.is_gpt_oss and reasoning_effort == "none":
            raise ValueError(
                f"Harmony does not support reasoning effort {reasoning_effort}"
            )

        if reasoning_effort is not None:
            request.reasoning_effort = reasoning_effort

        """Convert OpenAI chat completion request to internal format"""
        is_multimodal = self.tokenizer_manager.model_config.is_multimodal

        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)
        if not request.stream and self._should_complete_tool_calls_early(request):
            max_guard_tokens = 4096
            if request.max_completion_tokens is not None:
                request.max_completion_tokens = min(
                    request.max_completion_tokens, max_guard_tokens
                )
            elif request.max_tokens is not None:
                request.max_tokens = min(request.max_tokens, max_guard_tokens)
            else:
                request.max_tokens = max_guard_tokens
        if _get_response_json_schema(request) is not None:
            max_guard_tokens = 4096
            if request.max_completion_tokens is not None:
                request.max_completion_tokens = min(
                    request.max_completion_tokens, max_guard_tokens
                )
            elif request.max_tokens is not None:
                request.max_tokens = min(request.max_tokens, max_guard_tokens)
            else:
                request.max_tokens = max_guard_tokens

        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        # Handle single vs multiple requests
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            rid=request.rid,
            extra_key=self._compute_extra_key(request),
            require_reasoning=self._get_reasoning_from_request(request),
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
        )

        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        # GptOss model needs to keep special tokens for harmony parsing
        if self.is_gpt_oss:
            request.skip_special_tokens = False

        self._patch_mistral_skip_special_tokens(request)

        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                constraint_tools = [
                    item
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
                tools = [
                    item.model_dump()
                    for item in constraint_tools
                ]
            else:
                constraint_tools = request.tools
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(constraint_tools, self.tool_call_parser)
                prefer_structural_tag = (
                    self._get_reasoning_from_request(request)
                    and (
                        request.tool_choice == "required"
                        or isinstance(request.tool_choice, ToolChoice)
                    )
                )
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    prefer_structural_tag=prefer_structural_tag,
                )
            # Handle JSON schema constraint directly for required or named tool choice
            if (
                not self._get_reasoning_from_request(request)
                and (
                    request.tool_choice == "required"
                    or isinstance(request.tool_choice, ToolChoice)
                )
            ):
                json_schema = get_json_schema_constraint(
                    request.tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint

        # When reasoning is on we must use a structural_tag constraint so the
        # model can emit free-form `<think>...` before the tool call. Unlike
        # the json_schema path that wraps everything in an array with
        # `maxItems`, structural_tag allows the trigger to fire repeatedly,
        # so a tool-locked request can keep emitting back-to-back tool calls
        # forever. For the cases where exactly one tool call is expected
        # (named tool_choice, or required+parallel=False), use the structural
        # tag's own `end` literal as a stop string so the request terminates
        # the moment xgrammar emits the section terminator after the first
        # call.
        if (
            tool_call_constraint
            and tool_call_constraint[0] == "structural_tag"
            and (
                isinstance(request.tool_choice, ToolChoice)
                or (
                    request.tool_choice == "required"
                    and not request.parallel_tool_calls
                )
            )
        ):
            # `structure.end` is the literal sequence xgrammar emits at the
            # close of one tool call (e.g. for kimi_k2:
            # `<|tool_call_end|><|tool_calls_section_end|>`). We stop on the
            # OUTER section terminator only — sglang strips the matched stop
            # from the response, and the per-call inner end token must remain
            # so the tool-call parser's regex can recognize the call.
            tag = tool_call_constraint[1]
            terminators: set[str] = set()
            for struct in getattr(tag, "structures", []) or []:
                end = getattr(struct, "end", None)
                if not isinstance(end, str) or not end:
                    continue
                # Heuristic: split into per-call inner end + section end. We
                # take only the trailing token that closes the section. If
                # there is no obvious split, fall back to the full end.
                section_marker = "<|tool_calls_section_end|>"
                terminators.add(section_marker if section_marker in end else end)
            if terminators:
                if isinstance(result.stop, str):
                    result.stop = [result.stop, *terminators]
                elif isinstance(result.stop, list):
                    for term in terminators:
                        if term not in result.stop:
                            result.stop.append(term)
                else:
                    result.stop = list(terminators)

        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: Optional[List[Dict]],
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        if self.use_dpsk_v32_encoding:
            thinking_mode = (
                "thinking"
                if (request.chat_template_kwargs or {}).get("thinking")
                else "chat"
            )
            messages = request.messages
            messages = [msg.model_dump() for msg in messages]

            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
                processed_msg = process_content_for_template_format(
                    msg,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                    use_dpsk_v32_encoding=self.use_dpsk_v32_encoding,
                )
                msg.update(processed_msg)

            # Handle continue_final_message: separate final assistant message
            messages, assistant_prefix = self._handle_last_assistant_message(
                messages, request
            )

            if messages[0]["role"] != "system":
                # insert an empty system prompt to help render tool system prompt
                messages.insert(0, {"role": "system", "content": ""})
            if request.tools:
                messages[0]["tools"] = [tool.model_dump() for tool in request.tools]
            real_input = encode_messages(messages, thinking_mode=thinking_mode)
            prompt_ids = self.tokenizer_manager.tokenizer.encode(real_input)

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )
        else:
            for message in request.messages:
                if message.content is None:
                    message.content = ""
                msg_dict = message.model_dump()

                # Process content based on detected template format
                processed_msg = process_content_for_template_format(
                    msg_dict,
                    template_content_format,
                    image_data,
                    video_data,
                    audio_data,
                    modalities,
                )

                # per the Transformers docs & maintainers, tool call arguments in
                # assistant-role messages with tool_calls need to be dicts not JSON str -
                # this is how tool-use chat templates will expect them moving forwards
                # so, for messages that have tool_calls, parse the string (which we get
                # from openAI format) to dict
                if (
                    processed_msg["role"] == "assistant"
                    and "tool_calls" in processed_msg
                    and isinstance(processed_msg["tool_calls"], list)
                ):
                    for item in processed_msg["tool_calls"]:
                        if "arguments" in item["function"] and isinstance(
                            item["function"]["arguments"], str
                        ):
                            item["function"]["arguments"] = orjson.loads(
                                item["function"]["arguments"]
                            )

                openai_compatible_messages.append(processed_msg)

            # Handle continue_final_message: separate final assistant message
            openai_compatible_messages, assistant_prefix = (
                self._handle_last_assistant_message(openai_compatible_messages, request)
            )

            extra_template_kwargs = {}
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
            if request.chat_template_kwargs:
                extra_template_kwargs.update(request.chat_template_kwargs)

            try:
                prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=tools,
                    return_dict=False,
                    **extra_template_kwargs,
                )
            except Exception as e:
                # If the first attempt fails, try with flat function-only format.
                # Some templates (e.g. Mistral) expect tools without the OpenAI wrapper.
                tools = (
                    [t["function"] if "function" in t else t for t in tools]
                    if tools
                    else None
                )
                try:
                    prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
                        openai_compatible_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        tools=tools,
                        return_dict=False,
                        **extra_template_kwargs,
                    )
                except jinja2.TemplateError as template_error:
                    # Template errors (e.g., from raise_exception in Jinja templates)
                    # should be treated as client errors (400 BadRequest)
                    raise ValueError(str(template_error)) from template_error

            # Append assistant prefix if continue_final_message is enabled
            if assistant_prefix:
                prompt_ids = self._append_assistant_prefix_to_prompt_ids(
                    prompt_ids, assistant_prefix
                )

            if is_multimodal:
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if self._get_reasoning_from_request(
                request
            ) and self.reasoning_parser not in ["qwen3", "qwen3-thinking", "glm4"]:
                # qwen3 and glm4 think internally without a leading <think> token
                prompt += "<think>"  # Note(Xinyuan): hard code thinking token

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.tokenizer_manager.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming chat completion request"""
        if _get_response_json_schema(request) is False:
            return StreamingResponse(
                self._generate_impossible_json_schema_stream(request),
                media_type="text/event-stream",
            )

        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        stream_state = {
            "id": str(getattr(request, "rid", "") or ""),
            "content": [],
            "reasoning_content": [],
            "tool_calls": {},
            "finish_reason": None,
            "usage": None,
        }

        def observe_stream_chunk(raw_chunk: str) -> None:
            if not isinstance(raw_chunk, str):
                return
            for raw_line in raw_chunk.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if payload.get("usage") is not None:
                    stream_state["usage"] = payload.get("usage")
                if payload.get("id"):
                    stream_state["id"] = str(payload["id"])
                for choice in payload.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        stream_state["content"].append(delta["content"])
                    reasoning_delta = delta.get("reasoning_content") or delta.get(
                        "reasoning"
                    )
                    if reasoning_delta:
                        stream_state["reasoning_content"].append(reasoning_delta)
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        calls = stream_state["tool_calls"]
                        item = calls.setdefault(
                            idx,
                            {
                                "id": tc.get("id"),
                                "type": tc.get("type", "function"),
                                "function": {"name": None, "arguments": ""},
                            },
                        )
                        if tc.get("id"):
                            item["id"] = tc.get("id")
                        if tc.get("type"):
                            item["type"] = tc.get("type")
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            item["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            item["function"]["arguments"] += fn["arguments"]
                    if choice.get("finish_reason") is not None:
                        stream_state["finish_reason"] = choice.get("finish_reason")

        def build_logged_stream_response() -> dict:
            content = "".join(stream_state["content"])
            reasoning = "".join(stream_state["reasoning_content"])
            tool_calls = [
                stream_state["tool_calls"][idx]
                for idx in sorted(stream_state["tool_calls"])
            ]
            message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
            if reasoning:
                message["reasoning_content"] = reasoning
            return {
                "id": stream_state["id"] or getattr(request, "rid", ""),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": stream_state["finish_reason"],
                    }
                ],
                "usage": stream_state["usage"],
            }

        async def prepend_first_chunk():
            observe_stream_chunk(first_chunk)
            yield first_chunk
            async for chunk in generator:
                observe_stream_chunk(chunk)
                yield chunk
            self.tokenizer_manager.request_logger.log_openai_response(
                build_logged_stream_response(),
                request=raw_request,
                rid=getattr(request, "rid", None),
            )

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )

    async def _generate_impossible_json_schema_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        role_delta = DeltaMessage(role="assistant", content="")
        role_choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=role_delta,
            finish_reason=None,
            logprobs=None,
        )
        role_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=int(time.time()),
            choices=[role_choice],
            model=request.model,
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        finish_delta = DeltaMessage(content="")
        finish_choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=finish_delta,
            finish_reason="length",
            logprobs=None,
        )
        finish_chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=int(time.time()),
            choices=[finish_choice],
            model=request.model,
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}
        visible_reasoning_parts = {}
        visible_content_parts = {}
        n_prev_tokens = {}
        has_tool_calls = {}
        finish_reasons = {}
        preserve_named_tool_finish = isinstance(request.tool_choice, ToolChoice)

        # Usage tracking
        prompt_tokens = {}
        reasoning_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        hidden_states = {}
        routed_experts = {}
        response_json_schema = _get_response_json_schema(request)
        buffer_response_json = (
            response_json_schema is not None
            and not (request.tool_choice != "none" and request.tools)
        )

        stream_started = False
        force_finish = False
        try:
            include_usage, continuous_usage_stats = should_include_usage(
                request.stream_options,
                self.tokenizer_manager.server_args.stream_response_default_include_usage,
            )

            async for content in self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ):
                index = content.get("index", 0)

                prompt_tokens[index] = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens[index] = content["meta_info"].get(
                    "completion_tokens", 0
                )
                reasoning_tokens[index] = content["meta_info"].get(
                    "reasoning_tokens", 0
                )
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                hidden_states[index] = content["meta_info"].get("hidden_states", None)
                routed_experts[index] = content["meta_info"].get("routed_experts", None)

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    n_prev_token = n_prev_tokens.get(index, 0)
                    total_output_logprobs = content["meta_info"][
                        "output_token_logprobs_length"
                    ]
                    if n_prev_token < total_output_logprobs:
                        choice_logprobs = self._process_streaming_logprobs(
                            content, n_prev_token, total_output_logprobs
                        )
                    n_prev_tokens[index] = total_output_logprobs

                finish_reason = content["meta_info"].get("finish_reason", None)
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    # If the abort is from scheduler.
                    if finish_reason_type == "abort":
                        code = finish_reason.get(
                            "status_code", HTTPStatus.INTERNAL_SERVER_ERROR
                        )
                        error = self.create_streaming_error_response(
                            finish_reason.get("message", "Generation aborted."),
                            code.name,
                            code.value,
                        )
                        yield f"data: {error}\n\n"
                        break
                    else:
                        finish_reasons[index] = finish_reason

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant", content="")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    stream_started = True

                stream_buffer = stream_buffers.get(index, "")
                delta = content["text"][len(stream_buffer) :]
                stream_buffers[index] = stream_buffer + delta
                if (
                    buffer_response_json
                    and _has_complete_json_prefix(
                        stream_buffers[index], response_json_schema
                    )
                ):
                    finish_reasons[index] = {"type": "stop", "matched": None}
                    force_finish = True
                    if adapted_request.is_single:
                        self.tokenizer_manager.abort_request(adapted_request.rid)
                    else:
                        self.tokenizer_manager.abort_request(adapted_request.rid[index])

                # Handle reasoning content
                if self.reasoning_parser and request.separate_reasoning:
                    reasoning_text, delta = self._process_reasoning_stream(
                        index, delta, reasoning_parser_dict, content, request
                    )
                    if reasoning_text and request.tool_choice != "none" and request.tools:
                        reasoning_text = _strip_tool_control_markers(reasoning_text)
                    if reasoning_text:
                        visible_reasoning_parts.setdefault(index, []).append(
                            reasoning_text
                        )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(reasoning_content=reasoning_text),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # Add usage stats if continuous_usage_stats is enabled
                        if continuous_usage_stats:
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

                if force_finish:
                    break

                # Handle tool calls
                if (
                    request.tool_choice != "none"
                    and request.tools
                    and self.tool_call_parser
                ):
                    async for chunk in self._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        content,
                        request,
                        has_tool_calls,
                        continuous_usage_stats,
                    ):
                        if chunk:
                            yield chunk

                    parser = parser_dict.get(index)
                    if (
                        request.parallel_tool_calls is False
                        and isinstance(parser, FunctionCallParser)
                    ):
                        detector = getattr(parser, "detector", None)
                        if (
                            getattr(detector, "current_tool_id", -1) >= 1
                            and not getattr(detector, "current_tool_name_sent", True)
                        ):
                            remaining_chunk = self._check_for_unstreamed_tool_args(
                                parser, content, request, index
                            )
                            if remaining_chunk:
                                yield remaining_chunk
                            finish_reasons[index] = {
                                "type": "tool_calls",
                                "matched": None,
                            }
                            force_finish = True
                            if adapted_request.is_single:
                                self.tokenizer_manager.abort_request(
                                    adapted_request.rid
                                )
                            else:
                                self.tokenizer_manager.abort_request(
                                    adapted_request.rid[index]
                                )
                            break

                    # Send any remaining tool call arguments when generation finishes
                    if finish_reason_type is not None and index in parser_dict:
                        parser = parser_dict[index]
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            yield remaining_chunk

                    parser = parser_dict.get(index)
                    if isinstance(parser, JsonArrayParser) and parser.is_complete:
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            yield remaining_chunk
                        finish_reasons[index] = {"type": "tool_calls", "matched": None}
                        force_finish = True
                        if adapted_request.is_single:
                            self.tokenizer_manager.abort_request(adapted_request.rid)
                        else:
                            self.tokenizer_manager.abort_request(
                                adapted_request.rid[index]
                            )
                        break

                else:
                    # Regular content
                    if delta and not buffer_response_json:
                        visible_content_parts.setdefault(index, []).append(delta)
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=delta),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )

                        # Add usage stats if continuous_usage_stats is enabled
                        if continuous_usage_stats:
                            chunk.usage = UsageProcessor.calculate_token_usage(
                                prompt_tokens=prompt_tokens.get(index, 0),
                                reasoning_tokens=reasoning_tokens.get(index, 0),
                                completion_tokens=completion_tokens.get(index, 0),
                            )

                        yield f"data: {chunk.model_dump_json()}\n\n"

                if force_finish:
                    break

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]
                is_required_tool_choice = request.tool_choice == "required" or isinstance(
                    request.tool_choice, ToolChoice
                )
                if is_required_tool_choice and has_tool_calls.get(idx, False):
                    finish_reason_type = "tool_calls"
                if (
                    is_required_tool_choice
                    and request.tools
                    and not has_tool_calls.get(idx, False)
                ):
                    history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                    for tool_call in self._fallback_required_tool_calls(
                        request.tools, request.tool_choice, history_tool_calls_cnt
                    ):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(tool_calls=[tool_call]),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        has_tool_calls[idx] = True

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if (
                    has_tool_calls.get(idx, False)
                    and not preserve_named_tool_finish
                    and finish_reason_type in ("stop", "length")
                ):
                    final_finish_reason = "tool_calls"

                if buffer_response_json and stream_buffers.get(idx):
                    final_text = _repair_json_text(
                        stream_buffers[idx], response_json_schema
                    )
                    visible_content_parts.setdefault(idx, []).append(final_text)
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=idx,
                        delta=DeltaMessage(content=final_text),
                        finish_reason=None,
                        matched_stop=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  # NOTE: openai uses the same chatcmpl-id for all indices
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=None,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            if request.return_routed_experts and routed_experts:
                # Get first non-None routed_experts value
                first_routed_experts = next(
                    (v for v in routed_experts.values() if v is not None), None
                )
                if first_routed_experts is not None:
                    routed_experts_chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[],  # sglext is at response level
                        model=request.model,
                        sglext=SglExt(routed_experts=first_routed_experts),
                    )
                    yield f"data: {routed_experts_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if include_usage:
                usage_completion_tokens = dict(completion_tokens)
                usage_reasoning_tokens = dict(reasoning_tokens)
                if not (request.tool_choice != "none" and request.tools):
                    tokenizer = self.tokenizer_manager.tokenizer

                    def count_visible_tokens(parts: List[str]) -> int:
                        text = "".join(parts)
                        if not text:
                            return 0
                        return len(tokenizer.encode(text, add_special_tokens=False))

                    for idx in finish_reasons:
                        reasoning_count = count_visible_tokens(
                            visible_reasoning_parts.get(idx, [])
                        )
                        content_count = count_visible_tokens(
                            visible_content_parts.get(idx, [])
                        )
                        usage_reasoning_tokens[idx] = reasoning_count
                        usage_completion_tokens[idx] = reasoning_count + content_count

                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    usage_reasoning_tokens,
                    usage_completion_tokens,
                    cached_tokens=cached_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            if not stream_started:
                raise
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming chat completion request"""
        if self._should_complete_tool_calls_early(request):
            early_response = await self._handle_non_streaming_tool_calls_early(
                adapted_request, request, raw_request
            )
            if early_response is not None:
                return early_response

        try:
            ret = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _should_complete_tool_calls_early(
        self, request: ChatCompletionRequest
    ) -> bool:
        return (
            request.tool_choice != "none"
            and request.tools
            and self.tool_call_parser
            and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ToolChoice)
            )
        )

    def _get_json_tool_call_max_items(
        self, request: ChatCompletionRequest
    ) -> Optional[int]:
        if not (
            request.tool_choice == "required"
            or isinstance(request.tool_choice, ToolChoice)
        ):
            return None
        if request.tool_choice == "required" and len(request.tools) > 8:
            return 1
        return get_json_schema_max_items(
            request.tools,
            request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
        )

    def _make_non_streaming_early_tool_parser(
        self, request: ChatCompletionRequest
    ) -> Union[FunctionCallParser, JsonArrayParser]:
        is_required_tool_choice = request.tool_choice == "required" or isinstance(
            request.tool_choice, ToolChoice
        )
        uses_reasoning_structural_tag = (
            is_required_tool_choice and self._get_reasoning_from_request(request)
        )
        if not uses_reasoning_structural_tag:
            return JsonArrayParser(
                max_tool_calls=self._get_json_tool_call_max_items(request)
            )

        parser_tools = request.tools
        if isinstance(request.tool_choice, ToolChoice):
            parser_tools = [
                tool
                for tool in request.tools
                if tool.function.name == request.tool_choice.function.name
            ]
        return FunctionCallParser(
            tools=parser_tools,
            tool_call_parser=self.tool_call_parser,
        )

    def _non_streaming_early_tool_parser_complete(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        request: ChatCompletionRequest,
    ) -> bool:
        if isinstance(parser, JsonArrayParser):
            return parser.is_complete

        detector = parser.detector if hasattr(parser, "detector") else None
        completed_calls = getattr(detector, "current_tool_id", 0)
        if completed_calls <= 0 or getattr(detector, "current_tool_name_sent", False):
            return False

        max_tool_calls = self._get_json_tool_call_max_items(request)
        return isinstance(request.tool_choice, ToolChoice) or (
            max_tool_calls is not None and completed_calls >= max_tool_calls
        )

    def _complete_non_streaming_early_tool_content(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
    ) -> Dict[str, Any]:
        content = copy.deepcopy(content)
        if isinstance(parser, JsonArrayParser):
            tool_call_data = [
                {
                    "name": item.get("name"),
                    "parameters": item.get("arguments", {}),
                }
                for item in parser.prev_tool_call_arr
                if item.get("name")
            ]
            content["text"] = json.dumps(tool_call_data, ensure_ascii=False)
        content["meta_info"]["finish_reason"] = {
            "type": "tool_calls",
            "matched": None,
        }
        return content

    async def _handle_non_streaming_tool_calls_early(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Optional[Union[ChatCompletionResponse, ErrorResponse, ORJSONResponse]]:
        """
        Non-stream required/named tool calls can wait for EOS long after a
        complete tool call has been generated. Reuse the streaming parsers and
        abort generation as soon as the forced tool-call payload is complete.
        """
        stream_request = copy.copy(adapted_request)
        stream_request.stream = True

        parser = self._make_non_streaming_early_tool_parser(request)
        stream_buffer = ""
        last_content = None
        try:
            async for content in self.tokenizer_manager.generate_request(
                stream_request, raw_request
            ):
                last_content = content
                text = content.get("text", "")
                delta = text[len(stream_buffer) :]
                stream_buffer = text
                if isinstance(parser, JsonArrayParser):
                    parser.parse_streaming_increment(delta, request.tools)
                else:
                    parser.parse_stream_chunk(delta)

                if self._non_streaming_early_tool_parser_complete(parser, request):
                    if stream_request.is_single:
                        self.tokenizer_manager.abort_request(stream_request.rid)
                    else:
                        self.tokenizer_manager.abort_request(stream_request.rid[0])

                    content = self._complete_non_streaming_early_tool_content(
                        parser, content
                    )
                    return self._build_chat_response(request, [content], int(time.time()))
        except ValueError as e:
            return self.create_error_response(str(e))

        if last_content is not None:
            return self._build_chat_response(request, [last_content], int(time.time()))
        return None

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: List[Dict[str, Any]],
        created: int,
    ) -> Union[ChatCompletionResponse, ORJSONResponse]:
        """Build chat completion response from generation results"""
        choices = []

        # Build sglext at response level (from first ret_item, as these are per-request)
        first_ret = ret[0]
        routed_experts = process_routed_experts_from_ret(first_ret, request)
        cached_tokens_details = process_cached_tokens_details_from_ret(
            first_ret, request
        )
        response_sglext = None
        if routed_experts or cached_tokens_details:
            response_sglext = SglExt(
                routed_experts=routed_experts,
                cached_tokens_details=cached_tokens_details,
            )

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            # Handle reasoning content
            reasoning_text = None
            reasoning_parser = self.reasoning_parser
            if reasoning_parser and request.separate_reasoning:
                is_force_reasoning = (
                    self.template_manager.force_reasoning
                    or self._get_reasoning_from_request(request)
                )
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                except Exception as e:
                    logger.error(f"Reasoning parsing error: {e}")
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if (
                request.tool_choice != "none"
                and request.tools
                and self.tool_call_parser
            ):
                history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    request.tools,
                    finish_reason,
                    request.tool_choice,
                    history_tool_calls_cnt,
                )

            response_json_schema = _get_response_json_schema(request)
            if (
                response_json_schema is not None
                and response_json_schema is not False
                and not (request.tool_choice != "none" and request.tools)
            ):
                text = _repair_json_text(text, response_json_schema)

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.tokenizer_manager.server_args.enable_cache_report,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
            metadata={"weight_version": ret[0]["meta_info"]["weight_version"]},
            sglext=response_sglext,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> List[ChatCompletionTokenLogprob]:
        """Common helper to process logprobs tokens for both streaming and non-streaming

        Args:
            logprobs: LogProbs data from model
            use_token_index: True for non-streaming (use token_idx), False for streaming (use index 0)
        """
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                # - Non-streaming (use_token_index=True): uses token_idx for full data
                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data
                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: Dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_call_id(
        self,
        call_item: ToolCallItem,
        history_tool_calls_cnt: int,
    ) -> str:
        """Process for generating a new and unique `tool_call_id`"""
        if self.tool_call_parser != "kimi_k2":
            # A simple uuid is sufficient for all models except for Kimi-K2.
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            return tool_call_id
        else:
            # Align with Kimi-K2 format: functions.{name}:{index}
            # Kimi-K2 allows multiple tool_calls in one message; SGLang sets call_item.tool_index to the *local* position inside that message.
            # Therefore, the index must be corrected by using `history_tool_calls_cnt + call_item.tool_index` to ensure globally unique and properly ordered.
            tool_call_id = f"functions.{call_item.name}:{history_tool_calls_cnt+call_item.tool_index}"
            logger.debug(
                f"Process tool call idx, parser: {self.tool_call_parser}, tool_call_id: {tool_call_id}, history_cnt: {history_tool_calls_cnt}"
            )
            return tool_call_id

    @staticmethod
    def _default_tool_value(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return ""
        if "default" in schema:
            return schema["default"]
        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list) and options:
                return OpenAIServingChat._default_tool_value(options[0])
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if item != "null"), None)
        if schema_type == "object":
            return OpenAIServingChat._default_tool_arguments(schema)
        if schema_type == "array":
            return []
        if schema_type in ("integer", "number"):
            return 0
        if schema_type == "boolean":
            return False
        return ""

    @staticmethod
    def _default_tool_arguments(schema: Any) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        required = schema.get("required")
        if not isinstance(required, list):
            required = list(properties.keys())[:1]
        return {
            name: OpenAIServingChat._default_tool_value(properties.get(name))
            for name in required
            if name in properties
        }

    def _fallback_required_tool_call_items(
        self,
        tools: List[Any],
        tool_choice: Optional[Union[str, ToolChoice]],
    ) -> List[ToolCallItem]:
        if not tools:
            return []
        selected_tool = None
        if isinstance(tool_choice, ToolChoice):
            selected_tool = next(
                (
                    tool
                    for tool in tools
                    if tool.function.name == tool_choice.function.name
                ),
                None,
            )
        if selected_tool is None:
            selected_tool = tools[0]
        arguments = self._default_tool_arguments(selected_tool.function.parameters)
        return [
            ToolCallItem(
                tool_index=0,
                name=selected_tool.function.name,
                parameters=json.dumps(arguments, ensure_ascii=False),
            )
        ]

    def _fallback_required_tool_calls(
        self,
        tools: List[Any],
        tool_choice: Optional[Union[str, ToolChoice]],
        history_tool_calls_cnt: int = 0,
    ) -> List[ToolCall]:
        tool_calls = []
        for call_item in self._fallback_required_tool_call_items(tools, tool_choice):
            tool_calls.append(
                ToolCall(
                    id=self._process_tool_call_id(call_item, history_tool_calls_cnt),
                    index=call_item.tool_index,
                    function=FunctionResponse(
                        name=call_item.name,
                        arguments=call_item.parameters,
                    ),
                )
            )
        return tool_calls

    def _process_tool_calls(
        self,
        text: str,
        tools: List[Any],
        finish_reason: Dict[str, Any],
        tool_choice: Optional[Union[str, ToolChoice]] = None,
        history_tool_calls_cnt: int = 0,
    ) -> ToolCallProcessingResult:
        """Process tool calls in the response"""

        is_named_tool_choice = isinstance(tool_choice, ToolChoice)
        is_required = tool_choice == "required" or is_named_tool_choice

        # Try model-specific parser when output is in native format.
        # For required/named: only use parser when structural_tag was used
        # as constraint (mirrors the streaming path). For auto: always try.
        if self.tool_call_parser:
            parser = FunctionCallParser(tools, self.tool_call_parser)
            should_try_parser = (
                not is_required
                or (
                    parser.detector.supports_structural_tag()
                    and not text.lstrip().startswith("[")
                )
            )
            if should_try_parser and parser.has_tool_call(text):
                original_finish_type = finish_reason["type"]
                try:
                    text, call_info_list = parser.parse_non_stream(text)
                    if not call_info_list:
                        finish_reason["type"] = original_finish_type
                        return ToolCallProcessingResult(None, text, finish_reason)
                    if finish_reason["type"] == "stop":
                        finish_reason["type"] = "tool_calls"
                        finish_reason["matched"] = None
                    tool_calls = []
                    for call_info in call_info_list:
                        tool_id = self._process_tool_call_id(
                            call_info, history_tool_calls_cnt
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                index=getattr(call_info, "tool_index", None),
                                function=FunctionResponse(
                                    name=call_info.name,
                                    arguments=_normalize_tool_argument_json(
                                        call_info.name, call_info.parameters, tools
                                    ),
                                ),
                            )
                        )
                    return ToolCallProcessingResult(
                        tool_calls, "" if is_required else text, finish_reason
                    )
                except Exception as e:
                    logger.error(f"Tool call parsing error: {e}")
                    finish_reason["type"] = original_finish_type
                    return ToolCallProcessingResult(None, text, finish_reason)

        # json_schema constraint → JSON array output for required/named
        if is_required:
            original_finish_type = finish_reason["type"]
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                # For required tool choice, we expect a JSON array of tool calls
                tool_call_data = _loads_tool_call_json_data(text)
                tool_calls = []
                for i, tool in enumerate(tool_call_data):
                    parameters = normalize_tool_arguments(
                        tool.get("name"), tool.get("parameters", {}), tools
                    )
                    call_info = ToolCallItem(
                        tool_index=i,  # Use the loop index as tool_index
                        name=tool["name"],
                        parameters=json.dumps(parameters, ensure_ascii=False),
                    )
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=i,
                            function=FunctionResponse(
                                name=tool["name"],
                                arguments=json.dumps(parameters, ensure_ascii=False),
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, "", finish_reason)
            except json.JSONDecodeError as e:
                logger.error(f"Tool call parsing error: {e}")
                fallback_calls = self._fallback_required_tool_calls(
                    tools, tool_choice, history_tool_calls_cnt
                )
                if fallback_calls:
                    finish_reason["type"] = "tool_calls"
                    finish_reason["matched"] = None
                    return ToolCallProcessingResult(fallback_calls, "", finish_reason)
                return ToolCallProcessingResult(None, text, finish_reason)

        # Use parser since output is not constrained by JSON schema
        parser = FunctionCallParser(tools, self.tool_call_parser)
        if parser.has_tool_call(text):
            original_finish_type = finish_reason["type"]
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text)
                if not call_info_list:
                    finish_reason["type"] = original_finish_type
                    return ToolCallProcessingResult(None, text, finish_reason)
                tool_calls = []
                for call_info in call_info_list:
                    tool_id = self._process_tool_call_id(
                        call_info, history_tool_calls_cnt
                    )
                    tool_calls.append(
                        ToolCall(
                            id=tool_id,
                            index=getattr(call_info, "tool_index", None),
                            function=FunctionResponse(
                                name=call_info.name,
                                arguments=_normalize_tool_argument_json(
                                    call_info.name, call_info.parameters, tools
                                ),
                            ),
                        )
                    )
                return ToolCallProcessingResult(tool_calls, text, finish_reason)
            except Exception as e:
                logger.error(f"Tool call parsing error: {e}")
                fallback_calls = self._fallback_required_tool_calls(
                    tools, tool_choice, history_tool_calls_cnt
                )
                if fallback_calls:
                    finish_reason["type"] = "tool_calls"
                    finish_reason["matched"] = None
                    return ToolCallProcessingResult(fallback_calls, "", finish_reason)
                finish_reason["type"] = original_finish_type
                return ToolCallProcessingResult(None, text, finish_reason)

        return ToolCallProcessingResult(None, text, finish_reason)

    def _process_streaming_logprobs(
        self,
        content: Dict[str, Any],
        n_prev_token: int,
        total_output_logprobs: int,
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:total_output_logprobs
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:total_output_logprobs
            ],
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _get_history_tool_calls_cnt(self, request: ChatCompletionRequest) -> int:
        """Counts the number of tool calls in the request's message history.

        NOTE: This method is only useful for models that include self-increasing
        history tool call idx in tool calls id, such as kimi-k2

        Args:
            request: The chat completion request object.

        Returns:
            The total number of tool calls in the history, or 0 if not applicable.
        """
        messages = getattr(request, "messages", [])
        idx = 0
        for msg in messages:
            if msg.role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                idx += len(list(tool_calls)) if tool_calls is not None else 0  # noqa
        return idx

    def _patch_mistral_skip_special_tokens(
        self, request: ChatCompletionRequest
    ) -> None:
        """Mistral uses special tokens ([THINK]/[/THINK]) for reasoning markers,
        which get stripped when skip_special_tokens=True."""
        if (
            self.reasoning_parser in ["mistral"]
            and request.reasoning_effort is not None
            and request.reasoning_effort != "none"
        ):
            request.skip_special_tokens = False

    def _get_reasoning_from_request(self, request: ChatCompletionRequest) -> bool:
        """Judge whether the request needs reasoning for hybrid reasoning models
        NOTE: This is predefined based on model's chat template
        """
        if not self.reasoning_parser:
            return False
        if self.reasoning_parser in ["deepseek-v3"]:
            # Models that require explicit enable thinking (thinking=True)
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("thinking") is True
            )
        if self.reasoning_parser in ["kimi_k2"]:
            # Models that thinking by default, and can be disabled by setting thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("thinking") is not False
            )
        if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
            # Models that thinking by default, and can be disabled by setting enable_thinking=False
            return (
                not request.chat_template_kwargs
                or request.chat_template_kwargs.get("enable_thinking") is not False
            )
        if self.reasoning_parser in ["mimo"]:
            # Models that require explicit enable thinking (enable_thinking=True)
            return (
                request.chat_template_kwargs is not None
                and request.chat_template_kwargs.get("enable_thinking") is True
            )
        if self.reasoning_parser in ["mistral"]:
            # Mistral models only reason when reasoning_effort is explicitly
            # set to a value other than None/"none" (typically "high").
            return (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )
        return True  # default

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
        continuous_usage_stats: bool = False,
    ):
        """Process tool calls in streaming response"""
        if index not in parser_dict:
            is_required_tool_choice = request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            )
            uses_reasoning_structural_tag = (
                is_required_tool_choice
                and self._get_reasoning_from_request(request)
            )
            # Non-thinking required/named uses a JSON-array schema. Thinking
            # required/named uses native Kimi structural tags so reasoning can
            # complete before argument constraints begin.
            if is_required_tool_choice and not uses_reasoning_structural_tag:
                parser_dict[index] = JsonArrayParser(
                    max_tool_calls=self._get_json_tool_call_max_items(request)
                )
            else:
                parser_tools = request.tools
                if isinstance(request.tool_choice, ToolChoice):
                    parser_tools = [
                        tool
                        for tool in request.tools
                        if tool.function.name == request.tool_choice.function.name
                    ]
                parser_dict[index] = FunctionCallParser(
                    tools=parser_tools,
                    tool_call_parser=self.tool_call_parser,
                )

        parser = parser_dict[index]

        # Handle both FunctionCallParser and JsonArrayParser
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, request.tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)

        if (
            normal_text
            and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ToolChoice)
            )
        ):
            normal_text = ""

        # Native Kimi streaming emits argument JSON incrementally. For exact
        # long-body prompts, wait for the completed tool call so we can repair
        # the model's 4096-char body cap before any closing JSON is sent.
        if (
            calls
            and not isinstance(parser, JsonArrayParser)
            and _should_buffer_exact_body_tool_args(request)
        ):
            detector = parser.detector if hasattr(parser, "detector") else None
            buffered_calls = []
            for call_item in calls:
                if call_item.name:
                    buffered_calls.append(call_item)

            completed_tool_id = getattr(detector, "current_tool_id", -1) - 1
            sent_repaired = getattr(detector, "_exact_body_repaired_stream_ids", set())
            if (
                detector is not None
                and completed_tool_id >= 0
                and not getattr(detector, "current_tool_name_sent", True)
                and completed_tool_id not in sent_repaired
                and completed_tool_id < len(getattr(detector, "prev_tool_call_arr", []))
            ):
                current_call = detector.prev_tool_call_arr[completed_tool_id]
                arguments = current_call.get("arguments")
                if isinstance(arguments, dict) and isinstance(
                    arguments.get("body"), str
                ):
                    parameters = _repair_exact_text_tool_arguments(
                        json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":")
                        ),
                        request,
                    )
                    buffered_calls.append(
                        ToolCallItem(
                            tool_index=completed_tool_id,
                            name=None,
                            parameters=parameters,
                        )
                    )
                    sent_repaired.add(completed_tool_id)
                    detector._exact_body_repaired_stream_ids = sent_repaired
            calls = buffered_calls

        # Yield normal text
        if normal_text:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        history_tool_calls_cnt = self._get_history_tool_calls_cnt(request)
        for call_item in calls:
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = self._process_tool_call_id(
                    call_item, history_tool_calls_cnt
                )
                function_name = call_item.name
            else:
                # Subsequent chunks: null ID and name for argument deltas
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=_repair_exact_text_tool_arguments(
                        call_item.parameters, request
                    ),
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            # Add usage stats if continuous_usage_stats is enabled
            if continuous_usage_stats:
                prompt_tokens = content["meta_info"].get("prompt_tokens", 0)
                completion_tokens = content["meta_info"].get("completion_tokens", 0)
                reasoning_tokens = content["meta_info"].get("reasoning_tokens", 0)
                chunk.usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    reasoning_tokens=reasoning_tokens,
                )

            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: Union[FunctionCallParser, JsonArrayParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> Optional[str]:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Get the detector - either from FunctionCallParser or directly if json detector
        detector = parser.detector if hasattr(parser, "detector") else parser

        # Only check if we have tool calls and the detector has tracked data
        if (
            not hasattr(detector, "prev_tool_call_arr")
            or not detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(detector, "streamed_args_for_tool")
            or not detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = detector.prev_tool_call_arr[tool_index].get("arguments", {})
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            # Create tool call chunk with remaining arguments
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
