from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _strip_chatcmpl_prefix(rid: Optional[str]) -> str:
    if not rid:
        return ""
    return rid[len("chatcmpl-") :] if rid.startswith("chatcmpl-") else rid


def request_id_from_headers(headers: Any) -> str:
    if headers is None:
        return ""
    for key in ("x-request-id", "X-Request-Id"):
        value = headers.get(key)
        if value:
            return str(value)
    return ""


def chatcmpl_rid_from_headers(headers: Any) -> str:
    rid = request_id_from_headers(headers)
    if not rid:
        return ""
    return rid if rid.startswith("chatcmpl-") else f"chatcmpl-{rid}"


def _headers_to_dict(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    out: dict[str, str] = {}
    try:
        items = headers.items()
    except Exception:
        return out
    for key, value in items:
        key_s = str(key).lower()
        if key_s == "authorization":
            continue
        out[key_s] = str(value)
    return out


def _model_to_payload(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=False)
    if isinstance(obj, dict):
        return obj
    body = getattr(obj, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            import json

            decoded = json.loads(body)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    return {"value": obj}


def _redact_media_text(value: str) -> str:
    if not any(
        marker in value
        for marker in (
            "data:image/",
            "data:video/",
            "data:audio/",
            "asset_id",
            "<img",
            "<video",
            "<audio",
        )
    ):
        return value
    value = re.sub(
        r'src=(["\'])(?:data:(?:image|video|audio)/[^"\']+|file://[^"\']+)\1',
        'src="[redacted-mm-input]"',
        value,
    )
    value = re.sub(
        r"data:(?:image|video|audio)/[^,\s\"']+,[^\s\"']+",
        "[redacted-mm-input]",
        value,
    )
    return value


def _redact_media_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_media_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_media_text(value)
    if not isinstance(value, dict):
        return value

    part_type = value.get("type")
    redacted = {key: _redact_media_payload(item) for key, item in value.items()}
    if part_type in {"image_url", "video_url", "audio_url"}:
        field = part_type
        field_value = redacted.get(field)
        if isinstance(field_value, dict):
            field_value["url"] = "[redacted-mm-input]"
        elif field in redacted:
            redacted[field] = "[redacted-mm-input]"
    if part_type == "input_audio":
        input_audio = redacted.get("input_audio")
        if isinstance(input_audio, dict) and "data" in input_audio:
            input_audio["data"] = "[redacted-mm-input]"
    return redacted


def _count_content_parts(payload: dict[str, Any], kind: str) -> int:
    count = 0
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == kind:
                count += 1
    return count


def _tool_choice_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        function = value.get("function")
        if isinstance(function, dict) and function.get("name"):
            return "named"
        if value.get("type"):
            return str(value["type"])
    return str(value)


def _structured_output_kind(payload: dict[str, Any]) -> Optional[str]:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type"):
        return str(response_format["type"])
    structured_outputs = payload.get("structured_outputs")
    if isinstance(structured_outputs, dict):
        for key in ("json", "json_schema", "regex", "grammar", "schema"):
            if key in structured_outputs:
                return key
        if structured_outputs:
            return "structured_outputs"
    return None


def _request_attrs(
    payload: dict[str, Any],
    headers: dict[str, str],
    rid: str,
) -> dict[str, Any]:
    image_count = _count_content_parts(payload, "image_url")
    video_count = _count_content_parts(payload, "video_url")
    audio_count = _count_content_parts(payload, "audio_url")
    tools = payload.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    tool_choice = _tool_choice_name(payload.get("tool_choice"))
    structured_kind = _structured_output_kind(payload)

    attrs: dict[str, Any] = {
        "rid": _strip_chatcmpl_prefix(rid),
        "request_id": _strip_chatcmpl_prefix(rid),
        "payload": payload,
        "headers": headers,
        "input_image_count": image_count,
        "input_video_count": video_count,
        "input_audio_count": audio_count,
        "input_tool_count": tool_count,
        "has_images": image_count > 0,
        "has_videos": video_count > 0,
        "has_audios": audio_count > 0,
        "has_tools": tool_count > 0,
        "has_tool_calls_enabled": tool_count > 0 and tool_choice != "none",
        "has_structured_output": structured_kind is not None,
    }
    if tool_choice is not None:
        attrs["tool_choice"] = tool_choice
    if structured_kind is not None:
        attrs["structured_output_kind"] = structured_kind
    return attrs


class OTelPayloadLogger:
    def __init__(self) -> None:
        self.enabled = _env_enabled("SGLANG_OTEL_PAYLOAD_LOGGING")
        self._logger = None
        if not self.enabled:
            return
        try:
            from opentelemetry._logs import set_logger_provider
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter,
            )
            from opentelemetry.sdk._logs import LoggerProvider
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource

            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
                "http://127.0.0.1:4318/v1/logs",
            )
            provider = LoggerProvider(
                resource=Resource.create({SERVICE_NAME: "sglang"})
            )
            provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    OTLPLogExporter(endpoint=endpoint),
                    schedule_delay_millis=envs.SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS.get(),
                    max_export_batch_size=envs.SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE.get(),
                )
            )
            set_logger_provider(provider)
            self._logger = provider.get_logger("sglang.openai.payload")
        except Exception:
            self.enabled = False
            logger.exception("Failed to initialize OpenTelemetry payload logger")

    def emit_request(self, payload_obj: Any, headers_obj: Any, rid: str) -> None:
        if not self.enabled or self._logger is None:
            return
        payload = _redact_media_payload(_model_to_payload(payload_obj))
        headers = _headers_to_dict(headers_obj)
        if not rid:
            rid = chatcmpl_rid_from_headers(headers)
        self._logger.emit(
            severity_text="INFO",
            body="openai.request",
            attributes=_request_attrs(payload, headers, rid),
        )

    def emit_response(self, payload_obj: Any, headers_obj: Any, rid: str) -> None:
        if not self.enabled or self._logger is None:
            return
        payload = _model_to_payload(payload_obj)
        if payload.get("object") == "error" and "error" not in payload:
            payload = {"error": payload}
        headers = _headers_to_dict(headers_obj)
        if not rid:
            rid = str(payload.get("id") or chatcmpl_rid_from_headers(headers))
        self._logger.emit(
            severity_text="INFO",
            body="openai.response",
            attributes={
                "rid": _strip_chatcmpl_prefix(rid),
                "request_id": _strip_chatcmpl_prefix(rid),
                "payload": payload,
                "headers": headers,
            },
        )


@lru_cache(maxsize=1)
def get_otel_payload_logger() -> OTelPayloadLogger:
    return OTelPayloadLogger()
