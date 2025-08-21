#!/usr/bin/env python3
import json
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def load_template(template_path):
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template(template_name)

def render_with_tools():
    template = load_template("/sgl-workspace/sglang/examples/chat_template/chat_template_deepseek_perfect.jinja")
    
    messages = [{"role": "user", "content": "What is the weather like in Tokyo?"}]
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
    
    print("=== TOOLS DEBUG ===")
    print("Tools:", json.dumps(tools, indent=2))
    for tool in tools:
        print(f"Tool name: {tool['function']['name']}")
        print(f"Tool description: {tool['function']['description']}")
        print(f"Tool parameters: {tool['function']['parameters']}")
    
    context = {
        "messages": messages,
        "thinking": False,
        "add_generation_prompt": True,
        "bos_token": "<｜begin▁of▁sentence｜>",
        "tools": tools
    }
    
    result = template.render(**context)
    print("=== RENDERED TEMPLATE WITH TOOLS ===")
    print(repr(result))
    print("\n=== HUMAN READABLE ===")
    print(result)

if __name__ == "__main__":
    render_with_tools()
