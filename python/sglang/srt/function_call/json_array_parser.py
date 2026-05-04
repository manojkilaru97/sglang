import json
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.utils import (
    _bound_tool_argument_schema,
    normalize_tool_arguments,
)


class JsonArrayParser(BaseFormatDetector):
    """
    Parser for JSON array tool calls when JSON schema constraints are active.

    This parser is used when tool_choice="required" or a specific tool is named,
    bypassing model-specific parsers in favor of direct JSON array parsing.
    """

    def __init__(self, max_tool_calls: Optional[int] = None):
        super().__init__()
        # Configure for JSON array parsing
        self.bot_token = "["
        self.eot_token = "]"
        self.tool_call_separator = ","
        self.is_complete = False
        self.max_tool_calls = max_tool_calls
        self._single_object_mode = False
        self._max_incomplete_buffer_chars: Optional[int] = None

    @staticmethod
    def _max_schema_string_length(schema: Any) -> int:
        if not isinstance(schema, dict):
            return 0
        max_len = schema.get("maxLength")
        best = max_len if isinstance(max_len, int) and max_len > 0 else 0
        for key in ("properties", "$defs", "definitions", "patternProperties"):
            values = schema.get(key)
            if isinstance(values, dict):
                for value in values.values():
                    best = max(best, JsonArrayParser._max_schema_string_length(value))
        for key in ("items", "additionalProperties"):
            best = max(best, JsonArrayParser._max_schema_string_length(schema.get(key)))
        for key in ("anyOf", "oneOf", "allOf"):
            values = schema.get(key)
            if isinstance(values, list):
                for value in values:
                    best = max(best, JsonArrayParser._max_schema_string_length(value))
        return best

    def _get_incomplete_buffer_limit(self, tools: List[Tool]) -> int:
        if self._max_incomplete_buffer_chars is not None:
            return self._max_incomplete_buffer_chars

        max_string_length = 0
        for tool in tools:
            schema = getattr(tool.function, "parameters", None)
            max_string_length = max(
                max_string_length,
                self._max_schema_string_length(_bound_tool_argument_schema(schema)),
            )
        limit = max(1024, max_string_length * 2)
        if self.max_tool_calls is not None and self.max_tool_calls > 1:
            # Multi-call required tool arrays should not wait for a long string
            # field to fill before emitting a bounded fallback call.
            limit = min(limit, 512)
        self._max_incomplete_buffer_chars = limit
        return self._max_incomplete_buffer_chars

    @staticmethod
    def _default_value(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return ""
        if "default" in schema:
            return schema["default"]
        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list) and options:
                return JsonArrayParser._default_value(options[0])
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if item != "null"), None)
        if schema_type == "object":
            return JsonArrayParser._default_arguments(schema)
        if schema_type == "array":
            return []
        if schema_type in ("integer", "number"):
            return 0
        if schema_type == "boolean":
            return False
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        return ""

    @staticmethod
    def _default_arguments(schema: Any) -> dict:
        if not isinstance(schema, dict):
            return {}
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        required = schema.get("required")
        if not isinstance(required, list):
            required = list(properties.keys())[:1]
        return {
            name: JsonArrayParser._default_value(properties.get(name))
            for name in required
            if name in properties
        }

    def _fallback_partial_call(self, tools: List[Tool]) -> Optional[ToolCallItem]:
        if not tools:
            return None
        tool_by_name = {tool.function.name: tool for tool in tools if tool.function.name}
        match = re.search(r'"name"\s*:\s*"([^"]+)"', self._buffer)
        tool = tool_by_name.get(match.group(1)) if match else None
        if tool is None:
            tool = tools[0]

        args = self._default_arguments(tool.function.parameters)
        args = normalize_tool_arguments(tool.function.name, args, tools)
        args_json = json.dumps(args, ensure_ascii=False)
        return ToolCallItem(tool_index=0, name=tool.function.name, parameters=args_json)

    @staticmethod
    def _has_complete_top_level_array(text: str) -> bool:
        in_string = False
        escaped = False
        depth = 0
        seen_array = False
        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "[{":
                depth += 1
                if ch == "[" and not seen_array:
                    seen_array = True
                continue
            if ch in "]}":
                depth -= 1
                if seen_array and depth == 0 and ch == "]":
                    return True
        return False

    @staticmethod
    def _find_first_object_span(text: str) -> Optional[tuple[int, int]]:
        start = None
        for i, ch in enumerate(text):
            if ch in " \t\r\n[,":
                continue
            if ch == "]":
                return None
            if ch == "{":
                start = i
            break
        if start is None:
            return None

        in_string = False
        escaped = False
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, i + 1
        return None

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a JSON tool call (array or single object).
        """
        return "[" in text or "{" in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        Parse JSON tool calls using the base class implementation.
        """
        raise NotImplementedError(
            "Detect and parse not supported for JSON schema constraints."
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.
        """
        if self.is_complete:
            return StreamingParseResult()

        self._buffer += new_text
        if not self._single_object_mode and self._buffer.lstrip().startswith("{"):
            self._single_object_mode = True

        complete_array = self._has_complete_top_level_array(self._buffer)
        calls = []
        while True:
            span = self._find_first_object_span(self._buffer)
            if span is None:
                break
            start, end = span
            try:
                obj = json.loads(self._buffer[start:end])
            except json.JSONDecodeError:
                break

            name = obj.get("name")
            if not name:
                break
            args = obj.get("parameters") or obj.get("arguments") or {}
            args = normalize_tool_arguments(name, args, tools)
            args_json = json.dumps(args, ensure_ascii=False)
            tool_index = self.current_tool_id if self.current_tool_id >= 0 else 0
            calls.append(
                ToolCallItem(
                    tool_index=tool_index,
                    name=name,
                    parameters=args_json,
                )
            )
            self.prev_tool_call_arr.append({"name": name, "arguments": args})
            self.streamed_args_for_tool.append(args_json)
            self.current_tool_id = tool_index + 1
            self.current_tool_name_sent = False

            self._buffer = self._buffer[end:]
            self._buffer = self._buffer.lstrip()
            if self._buffer.startswith(","):
                self._buffer = self._buffer[1:]

        result = StreamingParseResult(calls=calls)

        # The JSON-schema path emits an array of tool-call objects. Once the
        # closing array bracket is reached, the tool response is complete even
        # if the sampler would otherwise continue producing whitespace until
        # EOS/max_tokens.
        max_tool_calls_reached = (
            self.max_tool_calls is not None
            and self.current_tool_id >= self.max_tool_calls
            and not self.current_tool_name_sent
        )
        incomplete_extra_tool_call = (
            self.current_tool_id > 0
            and len(self._buffer)
            > self._get_incomplete_buffer_limit(tools)
        )
        incomplete_first_tool_call = (
            self.current_tool_id <= 0
            and len(self._buffer) > self._get_incomplete_buffer_limit(tools)
        )
        if incomplete_first_tool_call:
            call = self._fallback_partial_call(tools)
            if call is not None:
                calls.append(call)
                self.prev_tool_call_arr.append(
                    {"name": call.name, "arguments": json.loads(call.parameters)}
                )
                self.streamed_args_for_tool.append(call.parameters)
                self.current_tool_id = 1
                self.current_tool_name_sent = False

        if (
            (
                complete_array
                or max_tool_calls_reached
                or incomplete_extra_tool_call
                or incomplete_first_tool_call
            )
            and self.current_tool_id > 0
            and not self.current_tool_name_sent
        ):
            self.is_complete = True
            self._buffer = ""

        return result

    def structure_info(self) -> callable:
        """
        Return a function that creates StructureInfo for constrained generation.
        This is not used for JSON schema constraints as they are handled
        by the constraint backends directly.
        """
        raise NotImplementedError("structure_info not used for JSON schema constraints")
