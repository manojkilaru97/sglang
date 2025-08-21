import json
import logging
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer
from sglang.srt.function_call.utils import _is_complete_json

logger = logging.getLogger(__name__)


class DeepSeekV3Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3 model function call format.

    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{function_name}\n```json\n{json_arguments}\n```<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `function<｜tool▁sep｜>{function_name}`
    - Arguments: JSON code block between ````json` and ````
    - Supports multiple tool calls

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-0324?chat_template=default
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.func_call_regex = r"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        # Support the actual DeepSeek V3 format from the original template:
        # <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>FUNCTION_NAME<｜tool▁sep｜>JSON_ARGS<｜tool▁call▁end｜>
        self.func_detail_regex = (
            r"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>(?P<name>[^<｜]+)<｜tool▁sep｜>(?P<args>.*?)<｜tool▁call▁end｜>"
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False

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
        calls = []
        # Look for the pattern: <｜tool▁calls▁begin｜>FUNCTION_NAME<｜tool▁sep｜>\n```json\n{args}\n```\n<｜tool▁call▁end｜>
        m = re.search(self.func_detail_regex, text, re.DOTALL)
        try:
            if m:
                func_name = (m.group("name") or "").strip()
                args_text = (m.group("args") or "").strip()
                try:
                    func_args = json.loads(args_text)
                    # Create ToolCallItem directly since the model generates its own function names
                    calls.append(
                        ToolCallItem(
                            tool_index=-1,
                            name=func_name,
                            parameters=json.dumps(func_args, ensure_ascii=False),
                        )
                    )
                except Exception:
                    logger.warning("DeepSeekV3Detector: Failed to parse tool arguments as JSON")
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
        """
        self._buffer += new_text
        current_text = self._buffer

        # Check if we have a tool call (using the actual tokens from original template)
        has_tool_call = self.bot_token in current_text and "<｜tool▁call▁begin｜>" in current_text

        if not has_tool_call:
            # If we don't have a tool call pattern, return the text as normal content
            # but filter out any tool-related tokens that shouldn't be shown to user
            filtered_text = new_text
            for e_token in [self.eot_token, "```", "<｜tool▁call▁end｜>", self.bot_token, "<｜tool▁call▁begin｜>", "<｜tool▁sep｜>"]:
                if e_token in filtered_text:
                    filtered_text = filtered_text.replace(e_token, "")
            
            # If we have tool call tokens but no complete pattern yet, don't return them as content
            if any(token in new_text for token in [self.bot_token, "<｜tool▁call▁begin｜>", "<｜tool▁sep｜>"]):
                return StreamingParseResult(normal_text="")
            
            self._buffer = ""
            return StreamingParseResult(normal_text=filtered_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # Match the actual DeepSeek V3 format for streaming (from original template)
            # The model generates: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>FUNCTION_NAME<｜tool▁sep｜>JSON_ARGS
            partial_match = re.search(
                pattern=r"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>(?P<name>[^<｜]+?)<｜tool▁sep｜>(?P<args>.*?)(?=<｜tool▁call▁end｜>|$)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = (partial_match.group("name") or "").strip()
                func_args_raw = (partial_match.group("args") or "").strip()

                # Initialize state if this is the first tool call
                if not hasattr(self, 'current_tool_id') or self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    self.current_tool_name_sent = False

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

                    # Check if we have a complete tool call (either complete JSON or tool_call_end token)
                    has_tool_call_end = "<｜tool▁call▁end｜>" in current_text
                    
                    if _is_complete_json(func_args_raw) or has_tool_call_end:
                        # Update the stored arguments
                        try:
                            if func_args_raw:
                                parsed_args = json.loads(func_args_raw)
                                self.prev_tool_call_arr[self.current_tool_id][
                                    "arguments"
                                ] = parsed_args
                        except json.JSONDecodeError:
                            # If JSON parsing fails, store as string
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = func_args_raw

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = (
                            r"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>.*?(?:<｜tool▁call▁end｜>|$)"
                        )
                        match = re.search(
                            tool_call_end_pattern, current_text, re.DOTALL
                        )
                        if match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

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
            call_rule_fmt='"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n" {arguments_rule} "\\n```<｜tool▁call▁end｜>"',
            function_format="json",
        )
