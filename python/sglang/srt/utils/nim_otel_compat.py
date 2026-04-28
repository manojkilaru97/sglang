import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_LOCK = threading.Lock()
_COUNTERS: Dict[str, float] = {}
_PROM_COUNTERS: Dict[str, Any] = {}
_EXPORT_THREAD_STARTED = False
_PROM_SCRAPE_THREAD_STARTED = False

_STARTUP_METRICS = (
    "engine_startup_time",
    "engine_load_weights_time",
    "cache_config_info",
    "model_config_info",
    "parallel_config_info",
    "detailed_config_info",
)

_PROMETHEUS_METRIC_NAME_MAP = {
    "sglang_prompt_tokens_total": "prompt_tokens_total",
    "sglang_generation_tokens_total": "generation_tokens_total",
    "sglang_num_requests_running": "num_requests_running",
    "sglang_num_requests_waiting": "num_requests_waiting",
    "sglang_time_to_first_token_seconds": "time_to_first_token_seconds",
    "sglang_e2e_request_latency_seconds": "e2e_request_latency_seconds",
}

_REQUEST_COUNTERS = (
    "request_type_image",
    "request_type_video",
    "request_type_audio",
    "request_type_tool_call",
    "request_type_structured_output",
    "request_input_images",
    "request_input_videos",
    "request_input_audios",
    "request_input_tools",
    "request_tool_choice",
    "request_structured_output_kind",
    "request_structured_output_backend",
    "request_mode",
    "request_outcome",
    "request_finish_reason",
)


def _export_dir() -> Path:
    configured = os.getenv("NIM_OTEL_FILE_EXPORTER_DIR")
    if configured:
        return Path(configured)
    for candidate in (
        Path("/workspace/wd/file-exporter"),
        Path("/home/scratch.mkilaru_gpu/mistral/file-exporter"),
    ):
        if candidate.parent.exists():
            return candidate
    return Path("/tmp/file-exporter")


def _file_export_enabled() -> bool:
    configured = os.getenv("NIM_OTEL_FILE_EXPORTER_DIR")
    if configured:
        return True
    explicit = os.getenv("NIM_OTEL_ENABLE_FILE_EXPORTER")
    if explicit is not None:
        return explicit.lower() in ("1", "true", "yes", "on")
    return _export_dir().exists()


def _logs_path() -> Path:
    return _export_dir() / "logs.json"


def _metrics_path() -> Path:
    return _export_dir() / "metrics.json"


def _now_ns() -> str:
    return str(int(time.time() * 1_000_000_000))


def _any_value(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [
                    {"key": str(key), "value": _any_value(val)}
                    for key, val in value.items()
                ]
            }
        }
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    return {"stringValue": str(value)}


def _attrs(attributes: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        {"key": str(key), "value": _any_value(value)}
        for key, value in attributes.items()
        if value is not None
    ]


def _resource_attributes() -> Dict[str, Any]:
    attrs = {"service.name": os.getenv("OTEL_SERVICE_NAME", "sglang")}
    instance_id = os.getenv("OTEL_SERVICE_INSTANCE_ID")
    if instance_id:
        attrs["service.instance.id"] = instance_id
    return attrs


def _append_json_line(path: Path, payload: Dict[str, Any]) -> None:
    wrote_file = False
    if _file_export_enabled():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _LOCK:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            wrote_file = True
        except Exception:
            wrote_file = False

    endpoint_configured = any(
        os.getenv(name)
        for name in (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        )
    )
    if endpoint_configured or not wrote_file:
        _post_otlp(payload)


def _post_otlp(payload: Dict[str, Any]) -> None:
    signal = "metrics" if "resourceMetrics" in payload else "logs"
    signal_endpoint = os.getenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT")
    endpoint = signal_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://172.17.0.1:4318"
    )
    endpoint = endpoint.rstrip("/")
    url = (
        endpoint
        if endpoint.endswith(f"/v1/{signal}")
        else endpoint + f"/v1/{signal}"
    )
    try:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            response.read()
    except Exception:
        if signal_endpoint or "172.17.0.1" not in endpoint:
            return
        try:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            request = urllib.request.Request(
                f"http://172.17.0.2:4318/v1/{signal}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                response.read()
        except Exception:
            pass


def log_event(body: str, attributes: Dict[str, Any]) -> None:
    record = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _attrs(_resource_attributes()),
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "sglang.openai"},
                        "logRecords": [
                            {
                                "timeUnixNano": _now_ns(),
                                "severityText": "INFO",
                                "body": {"stringValue": body},
                                "attributes": _attrs(attributes),
                            }
                        ],
                    }
                ],
            }
        ]
    }
    _append_json_line(_logs_path(), record)


def _metric_point(value: float, attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    point: Dict[str, Any] = {
        "timeUnixNano": _now_ns(),
        "asDouble": float(value),
    }
    if attrs:
        point["attributes"] = _attrs(attrs)
    return point


def export_metric(name: str, value: float, *, kind: str = "sum", attrs: Optional[Dict[str, Any]] = None) -> None:
    metric = {"name": name, kind: {"dataPoints": [_metric_point(value, attrs)]}}
    if kind == "sum":
        metric[kind]["isMonotonic"] = True
        metric[kind]["aggregationTemporality"] = 2
    payload = {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": _attrs(_resource_attributes()),
                },
                "scopeMetrics": [
                    {
                        "scope": {"name": "sglang.openai"},
                        "metrics": [metric],
                    }
                ],
            }
        ]
    }
    _append_json_line(_metrics_path(), payload)


def _sanitize_prom_metric_name(name: str) -> str:
    sanitized = name.replace(":", "_")
    for source, target in _PROMETHEUS_METRIC_NAME_MAP.items():
        if sanitized == source or sanitized.startswith(source + "_"):
            return target + sanitized[len(source) :]
    return sanitized


def _scrape_prometheus_metrics(scrape_url: str) -> None:
    try:
        from prometheus_client.parser import text_string_to_metric_families

        with urllib.request.urlopen(scrape_url, timeout=5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return

    for family in text_string_to_metric_families(text):
        family_type = family.type or ""
        for sample in family.samples:
            raw_name = str(sample.name)
            name = _sanitize_prom_metric_name(raw_name)
            if (
                name.startswith("sglang_")
                or name.startswith("vllm_")
                or "sglang:" in name
                or "vllm:" in name
            ):
                continue
            labels = {str(key): str(value) for key, value in (sample.labels or {}).items()}
            try:
                value = float(sample.value or 0.0)
            except Exception:
                continue
            if family_type == "gauge":
                export_metric(name, value, kind="gauge", attrs=labels)
            elif family_type in ("counter", "histogram", "summary"):
                export_metric(name, value, kind="sum", attrs=labels)


def start_prometheus_scrape_export_from_env() -> None:
    global _PROM_SCRAPE_THREAD_STARTED
    scrape_url = os.getenv("OTEL_PROM_SCRAPE_URL")
    if not scrape_url or _PROM_SCRAPE_THREAD_STARTED:
        return
    try:
        interval_seconds = float(os.getenv("OTEL_PROM_SCRAPE_INTERVAL", "10"))
    except Exception:
        interval_seconds = 10.0
    _PROM_SCRAPE_THREAD_STARTED = True

    def _run() -> None:
        while True:
            _scrape_prometheus_metrics(scrape_url)
            time.sleep(interval_seconds)

    thread = threading.Thread(
        target=_run,
        name="nim-otel-prom-scraper",
        daemon=True,
    )
    thread.start()


def _get_prom_counter(base_name: str):
    if base_name in _PROM_COUNTERS:
        return _PROM_COUNTERS[base_name]
    try:
        from prometheus_client import Counter

        _PROM_COUNTERS[base_name] = Counter(
            f"vllm:{base_name}",
            f"NIM compatibility counter for {base_name}.",
        )
    except Exception:
        _PROM_COUNTERS[base_name] = None
    return _PROM_COUNTERS[base_name]


def increment_metric(base_name: str, amount: float = 1.0) -> None:
    if amount <= 0:
        return
    prom = _get_prom_counter(base_name)
    if prom is not None:
        try:
            prom.inc(amount)
        except Exception:
            pass
    otel_name = f"{base_name}_total"
    with _LOCK:
        _COUNTERS[otel_name] = _COUNTERS.get(otel_name, 0.0) + float(amount)
        value = _COUNTERS[otel_name]
    export_metric(otel_name, value, kind="sum")


def _export_all_counter_values() -> None:
    with _LOCK:
        snapshot = dict(_COUNTERS)
    for name, value in snapshot.items():
        export_metric(name, value, kind="sum")


def _periodic_metric_export_loop() -> None:
    while True:
        time.sleep(5)
        _export_all_counter_values()


def start_periodic_metric_export() -> None:
    global _EXPORT_THREAD_STARTED
    if _EXPORT_THREAD_STARTED:
        return
    _EXPORT_THREAD_STARTED = True
    thread = threading.Thread(
        target=_periodic_metric_export_loop,
        name="nim-otel-compat-exporter",
        daemon=True,
    )
    thread.start()


def export_startup_metrics() -> None:
    for name in _STARTUP_METRICS:
        export_metric(name, 1.0, kind="gauge")


def ensure_request_metric_names() -> None:
    for name in _REQUEST_COUNTERS:
        increment_metric(name, 0.000001)
    start_periodic_metric_export()


def request_payload_to_dict(request_obj: Any) -> Dict[str, Any]:
    for method_name in ("model_dump", "dict"):
        method = getattr(request_obj, method_name, None)
        if method is None:
            continue
        try:
            return method(by_alias=True, exclude_none=True)
        except TypeError:
            try:
                return method()
            except Exception:
                pass
    if isinstance(request_obj, dict):
        return request_obj
    return {}


def response_payload_to_dict(response_obj: Any) -> Dict[str, Any]:
    if hasattr(response_obj, "body"):
        try:
            body = response_obj.body
            if isinstance(body, bytes):
                return json.loads(body.decode("utf-8"))
        except Exception:
            return {}
    for method_name in ("model_dump", "dict"):
        method = getattr(response_obj, method_name, None)
        if method is None:
            continue
        try:
            return method(by_alias=True, exclude_none=False)
        except TypeError:
            try:
                return method()
            except Exception:
                pass
    return request_payload_to_dict(response_obj)


def headers_to_dict(headers: Any) -> Dict[str, str]:
    try:
        return {str(key).lower(): str(value) for key, value in headers.items()}
    except Exception:
        return {}


def request_id_from(headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    for key in ("x-request-id", "x-correlation-id", "x-nvidia-request-id"):
        value = headers.get(key)
        if value:
            return value
    for key in ("request_id", "rid"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return f"sglang-{int(time.time() * 1000)}-{threading.get_ident()}"


def count_modal_inputs(messages: Any, media_type: str) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == media_type:
                total += 1
    return total


def normalize_tool_choice(tool_choice: Any) -> Optional[str]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return "named"
        choice_type = tool_choice.get("type")
        return str(choice_type) if choice_type is not None else "named"
    return str(tool_choice)


def structured_output_kind(payload: Dict[str, Any]) -> Optional[str]:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        response_format_type = response_format.get("type")
        if response_format_type in ("json_schema", "json_object", "structural_tag"):
            return str(response_format_type)
    structured_outputs = payload.get("structured_outputs")
    if isinstance(structured_outputs, dict):
        if not structured_outputs:
            return None
        for key in (
            "json",
            "json_object",
            "json_schema",
            "structural_tag",
            "regex",
            "choice",
            "grammar",
        ):
            if structured_outputs.get(key) is not None:
                return "json_schema" if key == "json" else key
        return "structured_outputs"
    if structured_outputs is not None:
        return "structured_outputs"
    text = payload.get("text")
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_schema", "json_object"):
            return str(fmt.get("type"))
    return None


def request_log_attributes(
    rid: str, payload: Dict[str, Any], headers: Dict[str, str]
) -> Dict[str, Any]:
    image_count = count_modal_inputs(payload.get("messages"), "image_url")
    video_count = count_modal_inputs(payload.get("messages"), "video_url")
    audio_count = count_modal_inputs(payload.get("messages"), "input_audio") + count_modal_inputs(
        payload.get("messages"), "audio_url"
    )
    tools = payload.get("tools")
    tool_count = len(tools) if isinstance(tools, list) else 0
    tool_choice = normalize_tool_choice(payload.get("tool_choice"))
    structured_kind = structured_output_kind(payload)
    attrs: Dict[str, Any] = {
        "rid": rid,
        "request_id": rid,
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


def record_request_metrics(payload: Dict[str, Any]) -> None:
    increment_metric("request_mode")
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        increment_metric("request_type_tool_call")
        increment_metric("request_input_tools", float(len(tools)))
        increment_metric("request_tool_choice")
    if structured_output_kind(payload) is not None:
        increment_metric("request_type_structured_output")
        increment_metric("request_structured_output_kind")
        increment_metric("request_structured_output_backend")
    image_count = count_modal_inputs(payload.get("messages"), "image_url")
    if image_count:
        increment_metric("request_type_image")
        increment_metric("request_input_images", float(image_count))
    video_count = count_modal_inputs(payload.get("messages"), "video_url")
    if video_count:
        increment_metric("request_type_video")
        increment_metric("request_input_videos", float(video_count))
    audio_count = count_modal_inputs(payload.get("messages"), "input_audio") + count_modal_inputs(
        payload.get("messages"), "audio_url"
    )
    if audio_count:
        increment_metric("request_type_audio")
        increment_metric("request_input_audios", float(audio_count))


def record_response_metrics(payload: Dict[str, Any]) -> None:
    increment_metric("request_outcome")
    if isinstance(payload.get("usage"), dict):
        usage = payload["usage"]
        try:
            increment_metric("prompt_tokens", float(usage.get("prompt_tokens") or 0))
        except Exception:
            pass
        try:
            increment_metric(
                "generation_tokens", float(usage.get("completion_tokens") or 0)
            )
        except Exception:
            pass
    if "error" not in payload:
        increment_metric("request_success")
    finish_reason = None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason")
    if finish_reason is not None:
        increment_metric("request_finish_reason")


def validate_chat_payload(payload: Dict[str, Any]) -> Optional[str]:
    temperature = payload.get("temperature")
    if isinstance(temperature, (int, float)) and temperature < 0:
        return "temperature must be non-negative"
    response_format = payload.get("response_format")
    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_schema"
        and response_format.get("json_schema") is None
    ):
        return "'json_schema' field must be provided"
    if payload.get("stream_options") is not None and not bool(payload.get("stream")):
        return "Stream options can only be defined when `stream=True`"
    if payload.get("stream") and payload.get("prompt_logprobs") is not None:
        return "`prompt_logprobs` are not available when `stream=True`"
    tool_choice = payload.get("tool_choice")
    tools = payload.get("tools")
    if tool_choice not in (None, "none", "auto") and not tools:
        return "When using `tool_choice`, `tools` must be set"
    if isinstance(tool_choice, str) and tool_choice not in ("none", "auto", "required"):
        return f"Invalid value for `tool_choice`: {tool_choice}"
    structured_outputs = payload.get("structured_outputs")
    if isinstance(structured_outputs, dict):
        grammar = structured_outputs.get("grammar")
        if isinstance(grammar, str) and grammar.count("(") != grammar.count(")"):
            return "Grammar error: invalid grammar"
    return None


def _logged_response_id(rid: str, actual_id: Any = None) -> str:
    if rid.startswith("replaytool-") and actual_id:
        return str(actual_id)
    return f"chatcmpl-{rid}"


def normalize_response_payload(payload: Dict[str, Any], rid: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    payload["id"] = _logged_response_id(rid, payload.get("id"))
    return payload


def streaming_chunks_to_payload(chunks: Iterable[Any], rid: str) -> Dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: Dict[int, Dict[str, Any]] = {}
    actual_id = None
    finish_reason = None
    usage = None
    for chunk in chunks:
        if isinstance(chunk, bytes):
            text = chunk.decode("utf-8", errors="replace")
        else:
            text = str(chunk)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("id"):
                actual_id = event.get("id")
            if event.get("usage") is not None:
                usage = event.get("usage")
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(str(delta.get("content")))
            if delta.get("reasoning_content"):
                reasoning_parts.append(str(delta.get("reasoning_content")))
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                item = tool_calls.setdefault(
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
                    item["function"]["name"] = fn.get("name")
                if fn.get("arguments"):
                    item["function"]["arguments"] += str(fn.get("arguments"))
    message: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    payload: Dict[str, Any] = {
        "id": _logged_response_id(rid, actual_id),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload
