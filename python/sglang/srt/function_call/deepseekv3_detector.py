import json
import logging
import re
from typing import List

from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.utils import _is_complete_json
from sglang.srt.openai_api.protocol import Tool

logger = logging.getLogger(__name__)


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek models.
    Assumes function call format:
      '<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>get_current_weather\\n```json\\n{{"location": "Tokyo"}}\\n```<｜tool_call_end｜>
<｜tool_call_begin｜>function<｜tool_sep｜>get_current_weather\\n```json\\n{{"location": "Paris"}}\\n```<｜tool_call_end｜><｜tool_calls_end｜><｜end_of_sentence｜>'
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool_calls_begin｜>"
        self.eot_token = "<｜tool_calls_end｜>"
        
        # Regex to find one complete individual tool call string (used by findall)
        self.func_call_regex = r"<｜tool_call_begin｜>.*?<｜tool_call_end｜>"

        # Regex to extract details (name, args) from one individual tool call string
        # Correctly uses <｜tool_sep｜> and named groups for clarity and robustness
        # Also handles potential whitespace (including newlines) after arguments before the final ```
        self.func_detail_regex_str = r"<｜tool_call_begin｜>function<｜tool_sep｜>(?P<name>.*?)\\n```json\\n(?P<args>.*?)\\n```\s*<｜tool_call_end｜>"
        self.func_detail_regex_compiled = re.compile(self.func_detail_regex_str, re.DOTALL)

        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name using the new compiled regex
                func_detail_match = self.func_detail_regex_compiled.search(match_result)
                
                if func_detail_match:
                    func_name = func_detail_match.group("name").strip()
                    func_args_str = func_detail_match.group("args").strip()
                    try:
                        func_args = json.loads(func_args_str)
                        # construct match_result for parse_base_json
                        match_data = {"name": func_name, "parameters": func_args}
                        calls.extend(self.parse_base_json(match_data, tools))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSONDecodeError in detect_and_parse for {func_name}: {func_args_str} - {e}")
                else:
                    logger.error(f"Could not parse tool call details in detect_and_parse from: {match_result}")
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.
        Simplified version to avoid infinite loops.
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have tool call content - if so, just pass it through as normal text
        # The non-streaming detect_and_parse will handle the final parsing
        if self.bot_token in current_text or "<｜tool_call_begin｜>" in current_text:
            # For now, just pass through as normal text to avoid infinite loops
            # The final complete text will be parsed by detect_and_parse
            result_text = current_text
            self._buffer = ""
            return StreamingParseResult(normal_text=result_text, calls=[])
        
        # No tool call content, return as normal text
        result_text = current_text
        self._buffer = ""
        return StreamingParseResult(normal_text=result_text, calls=[])

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=">" + name + "\n```json\n",
            end="\n```<",
            trigger=">" + name + "\n```json\n",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            tool_call_separator="",
            call_rule_fmt='"<｜tool_call_begin｜>function<｜tool_sep｜>{name}\\n```json\\n" {arguments_rule} "\\n```<｜tool_call_end｜>"',
            function_format="json",
        )