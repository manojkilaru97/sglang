from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import orjson
import partial_json_parser
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.environ import envs


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _get_tool_schema_defs(tools: List[Tool]) -> dict:
    """
    Get consolidated $defs from all tools, validating for conflicts.

    Args:
        tools: List of tools to process

    Returns:
        Dictionary of consolidated $defs from all tools

    Raises:
        ValueError: If conflicting $defs are found
    """
    all_defs = {}
    for tool in tools:
        if tool.function.parameters is None:
            continue
        defs = tool.function.parameters.get("$defs", {})
        for def_name, def_schema in defs.items():
            bounded_def_schema = _bound_tool_argument_schema(def_schema)
            if def_name in all_defs and all_defs[def_name] != bounded_def_schema:
                raise ValueError(
                    f"Tool definition '{def_name}' has "
                    "multiple schemas, which is not "
                    "supported."
                )
            else:
                all_defs[def_name] = bounded_def_schema
    return all_defs


def _get_tool_schema(tool: Tool) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "enum": [tool.function.name]},
            "parameters": (
                _bound_tool_argument_schema(tool.function.parameters)
                if tool.function.parameters
                else {"type": "object", "properties": {}, "additionalProperties": False}
            ),
        },
        "required": ["name", "parameters"],
    }


_LONG_TOOL_STRING_FIELDS = {
    "body",
    "content",
    "message",
    "messages",
    "summary",
    "text",
}
_DEFAULT_TOOL_ARG_STRING_MAX_LENGTH = 512


def _tool_arg_string_max_length(field_name: Optional[str]) -> int:
    max_length = envs.SGLANG_TOOL_ARG_STRING_MAX_LENGTH.get()
    if max_length <= 0:
        return max_length
    if field_name in _LONG_TOOL_STRING_FIELDS:
        return max_length
    return min(max_length, _DEFAULT_TOOL_ARG_STRING_MAX_LENGTH)


def _bound_tool_argument_schema(schema: Any, field_name: Optional[str] = None) -> Any:
    """Prevent JSON-schema tool constraints from allowing unbounded containers."""
    max_length = _tool_arg_string_max_length(field_name)
    max_array_items = envs.SGLANG_TOOL_ARRAY_MAX_ITEMS.get()
    max_object_properties = envs.SGLANG_TOOL_OBJECT_MAX_PROPERTIES.get()
    if not isinstance(schema, dict):
        return schema

    capped = dict(schema)
    schema_type = capped.get("type")

    is_string = schema_type == "string" or (
        isinstance(schema_type, list) and "string" in schema_type
    )
    is_array = schema_type == "array" or (
        isinstance(schema_type, list) and "array" in schema_type
    )
    is_object = (
        schema_type == "object"
        or (isinstance(schema_type, list) and "object" in schema_type)
        or isinstance(capped.get("properties"), dict)
        or isinstance(capped.get("additionalProperties"), dict)
    )

    if is_string and max_length > 0:
        if "maxLength" not in capped:
            capped["maxLength"] = max_length
        # Some constrained backends are more reliable with an explicit finite
        # regex than maxLength alone.
        pattern = capped.get("pattern")
        if isinstance(pattern, str):
            if pattern.endswith("+$"):
                capped["pattern"] = f"{pattern[:-2]}{{1,{max_length}}}$"
            elif pattern.endswith("*$"):
                capped["pattern"] = f"{pattern[:-2]}{{0,{max_length}}}$"
        elif "pattern" not in capped:
            capped["pattern"] = rf"^[^\n]{{0,{max_length}}}$"
    if is_array and max_array_items > 0 and "maxItems" not in capped:
        capped["maxItems"] = max_array_items
    if is_object and max_object_properties > 0:
        object_max_properties = max_object_properties
        properties = capped.get("properties")
        if capped.get("additionalProperties") is False and isinstance(properties, dict):
            object_max_properties = min(object_max_properties, len(properties))
            capped.setdefault("propertyNames", {"enum": list(properties)})
            required = capped.get("required")
            if isinstance(required, list):
                capped["minProperties"] = max(
                    int(capped.get("minProperties", 0) or 0), len(required)
                )
        schema_max_properties = capped.get("maxProperties")
        if isinstance(schema_max_properties, int) and schema_max_properties > 0:
            object_max_properties = min(object_max_properties, schema_max_properties)
        capped["maxProperties"] = object_max_properties
    if (
        is_object
        and isinstance(capped.get("properties"), dict)
        and "additionalProperties" not in capped
    ):
        capped["additionalProperties"] = False

    if is_object and not isinstance(capped.get("properties"), dict):
        # Free-form `{type: object}` with no declared properties. xgrammar
        # otherwise lets the model emit arbitrarily nested JSON for additional
        # properties forever. Bound the values to scalars and the keys to a
        # short string so the grammar terminates.
        existing_addl = capped.get("additionalProperties")
        if existing_addl is None or existing_addl is True:
            string_max = _tool_arg_string_max_length(field_name)
            scalar_string_schema: Dict[str, Any] = {"type": "string"}
            if string_max > 0:
                scalar_string_schema["maxLength"] = string_max
                scalar_string_schema["pattern"] = rf"^[^\n]{{0,{string_max}}}$"
            capped["additionalProperties"] = {
                "anyOf": [
                    scalar_string_schema,
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            }
            property_name_max = min(128, string_max) if string_max > 0 else 128
            capped.setdefault(
                "propertyNames",
                {
                    "type": "string",
                    "maxLength": property_name_max,
                    "pattern": rf"^[^\n]{{1,{property_name_max}}}$",
                },
            )

    for key in ("properties", "$defs", "definitions"):
        values = capped.get(key)
        if isinstance(values, dict):
            capped[key] = {
                name: _bound_tool_argument_schema(value, name)
                for name, value in values.items()
            }

    if isinstance(capped.get("items"), dict):
        capped["items"] = _bound_tool_argument_schema(capped["items"], field_name)

    if isinstance(capped.get("additionalProperties"), dict):
        capped["additionalProperties"] = _bound_tool_argument_schema(
            capped["additionalProperties"], field_name
        )

    pattern_properties = capped.get("patternProperties")
    if isinstance(pattern_properties, dict):
        capped["patternProperties"] = {
            name: _bound_tool_argument_schema(value, name)
            for name, value in pattern_properties.items()
        }

    for key in ("anyOf", "oneOf", "allOf"):
        values = capped.get(key)
        if isinstance(values, list):
            capped[key] = [
                _bound_tool_argument_schema(value, field_name) for value in values
            ]

    return capped


def _coerce_string_to_object(value: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not isinstance(schema, dict):
        return None

    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    if not isinstance(properties, dict) or "city" not in properties:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None

    coerced = {"city": parts[0]}
    if "country" in properties and len(parts) > 1:
        coerced["country"] = ", ".join(parts[1:])

    if all(key in coerced for key in required):
        return coerced
    return None


def _normalize_string_to_schema_format(value: str, schema: Dict[str, Any]) -> str:
    fmt = schema.get("format")
    if not isinstance(fmt, str):
        return value
    normalized_format = fmt.strip().lower().replace("_", "-")
    if normalized_format not in {"iso 8601", "date-time", "datetime"}:
        return value

    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?",
        value.strip(),
    )
    if not match:
        return value
    date, hour, minute, second = match.groups()
    return f"{date}T{int(hour):02d}:{minute}:{second or '00'}"


def normalize_tool_arguments(
    tool_name: Optional[str], arguments: Any, tools: List[Tool]
) -> Any:
    """Normalize model output to the selected tool schema when safe.

    Some constrained decoders still choose the string branch of an anyOf even
    when the prompt asks for the object branch. When the schema has an object
    branch with familiar city/country fields and the model emitted
    "City, Country", convert it to the requested object shape.
    """
    if not isinstance(arguments, dict) or not tool_name:
        return arguments

    selected_tool = next(
        (tool for tool in tools if tool.function.name == tool_name), None
    )
    if selected_tool is None or not selected_tool.function.parameters:
        return arguments

    params = selected_tool.function.parameters
    properties = params.get("properties") or {}
    if not isinstance(properties, dict):
        return arguments

    normalized = dict(arguments)
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            continue

        if isinstance(value, str):
            normalized[key] = _normalize_string_to_schema_format(
                value, property_schema
            )

        variants = property_schema.get("anyOf") or property_schema.get("oneOf")
        if not isinstance(variants, list):
            continue

        object_variants = [
            variant
            for variant in variants
            if isinstance(variant, dict)
            and (
                variant.get("type") == "object"
                or isinstance(variant.get("properties"), dict)
            )
        ]
        if not object_variants or not isinstance(value, str):
            continue

        for object_schema in object_variants:
            coerced = _coerce_string_to_object(value, object_schema)
            if coerced is not None:
                normalized[key] = coerced
                break

    return normalized


def infer_type_from_json_schema(schema: Dict[str, Any]) -> Optional[str]:
    """
    Infer the primary type of a parameter from JSON Schema.

    Supports complex JSON Schema structures including:
    - Direct type field (including type arrays)
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Args:
        schema: JSON Schema definition

    Returns:
        Inferred type ('string', 'number', 'object', 'array', etc.) or None
    """
    if not isinstance(schema, dict):
        return None

    # Priority 1: Direct type field (including type arrays)
    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            # Handle type arrays: return first non-null type
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"  # If only null, default to string

    # Priority 2: Handle anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []

        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(sub_schema)
                if inferred_type:
                    types.append(inferred_type)

            if types:
                # If all types are the same, return unified type
                if len(set(types)) == 1:
                    return types[0]
                # When types differ, prioritize string (safest)
                if "string" in types:
                    return "string"
                # Otherwise return first type
                return types[0]

    # Priority 3: Handle enum (infer type from enum values)
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"

        # Infer type from enum values
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")

        # If type is uniform, return that type
        if len(enum_types) == 1:
            return enum_types.pop()
        # Mixed types, prioritize string
        return "string"

    # Priority 4: Handle allOf (must satisfy all types)
    if "allOf" in schema and isinstance(schema["allOf"], list):
        schemas = schema["allOf"]
        for sub_schema in schemas:
            inferred_type = infer_type_from_json_schema(sub_schema)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    # Priority 5: Infer object type
    if "properties" in schema:
        return "object"

    # Priority 6: Infer array type
    if "items" in schema:
        return "array"

    return None


def get_json_schema_constraint(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]],
    parallel_tool_calls: bool = True,
) -> Optional[dict]:
    """
    Get the JSON schema constraint for the specified tool choice.

    Args:
        tool_choice: The tool choice specification
        parallel_tool_calls: If False, constrain to exactly one tool call (maxItems=1)

    Returns:
        JSON schema dict, or None if no valid tools found
    """
    if isinstance(tool_choice, ToolChoice):
        # For specific function choice, return the user's parameters schema directly
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name == fn_name:
                schema = {
                    "type": "array",
                    "minItems": 1,
                    "items": _get_tool_schema(tool),
                }
                if not parallel_tool_calls:
                    schema["maxItems"] = 1
                else:
                    max_items = get_json_schema_max_items(
                        tools, tool_choice, parallel_tool_calls
                    )
                    if max_items is not None:
                        schema["maxItems"] = max_items
                return schema
        return None
    elif tool_choice == "required":
        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [_get_tool_schema(tool) for tool in tools],
            },
        }
        if not parallel_tool_calls:
            json_schema["maxItems"] = 1
        else:
            max_items = get_json_schema_max_items(
                tools, tool_choice, parallel_tool_calls
            )
            if max_items is not None:
                json_schema["maxItems"] = max_items
        json_schema_defs = _get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None


def get_json_schema_max_items(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]],
    parallel_tool_calls: bool = True,
) -> Optional[int]:
    if not parallel_tool_calls:
        return 1

    if isinstance(tool_choice, ToolChoice):
        max_named_parallel_calls = envs.SGLANG_NAMED_TOOL_MAX_PARALLEL_CALLS.get()
        return max_named_parallel_calls if max_named_parallel_calls > 0 else None

    if tool_choice == "required":
        max_parallel_calls = envs.SGLANG_TOOL_MAX_PARALLEL_CALLS.get()
        if max_parallel_calls <= 0:
            return None
        return min(max_parallel_calls, len(tools))

    return None
