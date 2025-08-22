#!/usr/bin/env python3
"""
Test script to verify thinking mode is working for non-tool scenarios first,
then test complex reasoning + tool calling.
"""

import json
import re
import requests
from typing import List, Dict, Any

SERVER_URL = "http://localhost:8003/v1/chat/completions"

def make_request(messages: List[Dict], tools: List[Dict] = None, thinking: bool = False, 
                stream: bool = False, model: str = "llama4") -> Dict[str, Any]:
    """Make a request to the chat completions endpoint."""
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    if tools:
        payload["tools"] = tools
        
    # DeepSeek server expects chat_template_kwargs at top-level (not under extra_body)
    if thinking is not None:
        payload["chat_template_kwargs"] = {"thinking": thinking}
    
    print(f"\n{'='*80}")
    print(f"REQUEST: thinking={thinking}, stream={stream}, tools={'Yes' if tools else 'No'}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        print("RESPONSE:")
        print(json.dumps(result, indent=2))
        
        if 'choices' in result and len(result['choices']) > 0:
            message = result['choices'][0]['message']
            content = message.get('content', '')
            reasoning_content = message.get('reasoning_content', None)
            tool_calls = message.get('tool_calls', [])
            
            print(f"\n🔍 ANALYSIS:")
            print(f"Content: {repr(content)}")
            print(f"Reasoning Content: {repr(reasoning_content)}")
            print(f"Tool Calls: {len(tool_calls) if tool_calls else 0} calls")
            
            return {
                "content": content, 
                "reasoning_content": reasoning_content,
                "tool_calls": tool_calls
            }
        
        return {"content": "", "reasoning_content": None, "tool_calls": []}
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from raw text containing DeepSeek tool-call markers.

    Expected pattern:
    <｜tool▁call▁begin｜>FUNCTION_NAME<｜tool▁sep｜>JSON_ARGUMENTS<｜tool▁call▁end｜>
    """
    if not text:
        return []
    pattern = (
        r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>"
    )
    matches = re.findall(pattern, text, flags=re.DOTALL)
    extracted: List[Dict[str, Any]] = []
    for func_name, args_str in matches:
        func_name = func_name.strip() or None
        # Try to parse JSON arguments; if fails, keep raw string
        try:
            args = json.loads(args_str)
        except Exception:
            args = args_str.strip()
        extracted.append({
            "type": "function",
            "function": {"name": func_name, "arguments": args},
        })
    return extracted

def test_thinking_without_tools():
    """Test if thinking mode works for complex problems without tools."""
    print(f"\n{'#'*80}")
    print("TESTING THINKING MODE - NO TOOLS")
    print(f"{'#'*80}")
    
    complex_questions = [
        {
            "name": "Math Problem",
            "message": "Solve this step by step: If a train leaves New York at 2:00 PM traveling at 80 mph toward Chicago (800 miles away), and another train leaves Chicago at 3:00 PM traveling at 70 mph toward New York, at what time will they meet and how far from New York?"
        },
        {
            "name": "Logic Puzzle", 
            "message": "Three people are in a room: Alice, Bob, and Charlie. Alice always tells the truth, Bob always lies, and Charlie sometimes tells the truth and sometimes lies. Alice says 'Bob is lying.' Bob says 'Charlie is telling the truth.' Charlie says 'Alice is telling the truth.' Who is telling the truth in this scenario? Think through this carefully."
        },
        {
            "name": "Complex Planning",
            "message": "I need to plan a dinner party for 8 people with a budget of $200. I need to consider appetizers, main course, dessert, and drinks. Some guests are vegetarian and one is gluten-free. Help me plan this step by step, considering costs and dietary restrictions."
        }
    ]
    
    for question in complex_questions:
        print(f"\n--- {question['name']} ---")
        
        # Test without thinking
        print("\n🚫 WITHOUT THINKING:")
        result_no_thinking = make_request(
            [{"role": "user", "content": question["message"]}],
            thinking=False
        )
        
        # Test with thinking
        print("\n🧠 WITH THINKING:")
        result_with_thinking = make_request(
            [{"role": "user", "content": question["message"]}],
            thinking=True
        )
        
        # Compare results
        print(f"\n📊 COMPARISON:")
        print(f"Without thinking - reasoning_content: {result_no_thinking.get('reasoning_content') is not None}")
        print(f"With thinking - reasoning_content: {result_with_thinking.get('reasoning_content') is not None}")
        
        if result_with_thinking.get('reasoning_content'):
            print("✅ Thinking mode is working!")
        else:
            print("❌ Thinking mode not activated")

def test_complex_reasoning_with_tools():
    """Test complex reasoning scenarios that should trigger both reasoning and tool calls."""
    print(f"\n{'#'*80}")
    print("TESTING COMPLEX REASONING + TOOL CALLING")
    print(f"{'#'*80}")
    
    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state/country"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    complex_scenarios = [
        {
            "name": "Trip Planning Decision",
            "message": "I'm planning a business trip and need to choose between New York, London, or Tokyo for next month. I need to consider weather conditions, flight costs (I'm flying from San Francisco), and the best time zones for meetings with European clients. Can you help me analyze these factors and make a recommendation? Please think through each factor systematically."
        },
        {
            "name": "Investment Analysis", 
            "message": "I have $10,000 to invest and I'm considering three options: 1) A tech stock that's up 15% this year, 2) A real estate investment trust with 8% dividend yield, or 3) A government bond at 4.5% guaranteed return. I need to research current market conditions for each option and calculate potential returns over 5 years. Please analyze this systematically and show your reasoning."
        },
        {
            "name": "Event Planning Optimization",
            "message": "I'm organizing a conference for 200 people in either Seattle, Austin, or Miami this summer. I need to consider weather patterns, venue costs, travel accessibility, and local attractions. Can you research and analyze these factors, then calculate estimated total costs and recommend the best location? Please think through this step by step."
        }
    ]
    
    for scenario in complex_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        result = make_request(
            [{"role": "user", "content": scenario["message"]}],
            tools=tools,
            thinking=True
        )
        
        print(f"\n📊 ANALYSIS:")
        has_reasoning = result.get('reasoning_content') is not None
        structured_tool_calls = result.get('tool_calls') or []
        extracted_reasoning_calls = extract_tool_calls_from_text(result.get('reasoning_content'))
        extracted_content_calls = extract_tool_calls_from_text(result.get('content'))
        any_tool_calls = bool(structured_tool_calls or extracted_reasoning_calls or extracted_content_calls)

        print(f"Reasoning content present: {has_reasoning}")
        print(f"Structured tool calls: {len(structured_tool_calls)}")
        print(f"Tool calls in reasoning: {len(extracted_reasoning_calls)}")
        print(f"Tool calls in content: {len(extracted_content_calls)}")

        if any_tool_calls and has_reasoning:
            print("✅ Complex reasoning + tool calling detected (structured or in reasoning)")
        elif any_tool_calls:
            print("⚠️ Tool calls detected without reasoning")
        elif has_reasoning:
            print("⚠️ Reasoning present but no tool calls detected")
        else:
            print("❌ Neither reasoning nor tool calls detected")

    # Also verify a simple thinking + tool call produces structured tool_calls
    print("\n--- Sanity Check: Simple Weather Tool with Thinking ---")
    simple = make_request(
        [{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        thinking=True
    )
    stc = simple.get('tool_calls') or []
    print(f"Structured tool calls (simple): {len(stc)}")
    if stc:
        print("✅ Structured tool calls present for simple thinking+tools request")
    else:
        print("⚠️ No structured tool calls for simple test; check parser/template")

def main():
    """Run all tests."""
    print("Testing Thinking Mode and Complex Reasoning + Tool Calling")
    print(f"Server URL: {SERVER_URL}")
    
    # First test thinking mode without tools
    test_thinking_without_tools()
    
    # Then test complex reasoning with tools
    test_complex_reasoning_with_tools()
    
    print(f"\n{'#'*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main()
