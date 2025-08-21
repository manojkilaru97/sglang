#!/usr/bin/env python3
from jinja2 import Template

template_str = """
{%- if tools is defined and tools is not none %}
Tools provided: {{ tools | length }}
{%- for tool in tools %}
Tool {{ loop.index }}: {{ tool.function.name }} - {{ tool.function.description }}
{%- endfor %}
{%- else %}
No tools provided
{%- endif %}
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "Get weather of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"]
            }
        }
    }
]

template = Template(template_str)
result = template.render(tools=tools)
print("=== SIMPLE TEMPLATE TEST ===")
print(repr(result))
print("\n=== HUMAN READABLE ===")
print(result)
