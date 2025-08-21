#!/usr/bin/env python3
"""
Test script for streaming tool calls with DeepSeek-V3.1
Displays raw responses neatly and combines tool call responses
"""

import json
import requests
import time
from typing import Dict, List, Any


def test_streaming_tool_call():
    """Test streaming tool calls and display responses neatly"""
    
    url = "http://127.0.0.1:8003/v1/chat/completions"
    
    payload = {
        "model": "llama4",
        "messages": [
            {"role": "user", "content": "What's the weather like in Tokyo and Paris? Also calculate 15 * 23."}
        ],
        "temperature": 0,
        "top_p": 0.9,
        "max_tokens": 3000,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                                "default": "celsius"
                            }
                        },
                        "required": ["city"]
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
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }
    
    print("🚀 Starting streaming tool call test...")
    print(f"📍 Endpoint: {url}")
    print(f"💬 Query: {payload['messages'][0]['content']}")
    print(f"🔧 Available tools: {[tool['function']['name'] for tool in payload['tools']]}")
    print("\n" + "="*80)
    
    # Storage for combining responses
    combined_response = {
        "role": "assistant",
        "content": "",
        "tool_calls": []
    }
    
    current_tool_calls = {}  # Track tool calls by index
    chunk_count = 0
    
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        print("📡 STREAMING RESPONSE:")
        print("-" * 50)
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                # Skip empty lines and "data: [DONE]"
                if not line_str.strip() or line_str.strip() == "data: [DONE]":
                    continue
                
                # Parse SSE format
                if line_str.startswith("data: "):
                    chunk_count += 1
                    json_str = line_str[6:]  # Remove "data: " prefix
                    
                    try:
                        chunk = json.loads(json_str)
                        
                        # Print chunk info
                        print(f"\n🔹 Chunk {chunk_count}:")
                        
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")
                        
                        # Handle role
                        if delta.get("role"):
                            print(f"   Role: {delta['role']}")
                            combined_response["role"] = delta["role"]
                        
                        # Handle content
                        if delta.get("content") is not None:
                            content = delta["content"]
                            print(f"   Content: {repr(content)}")
                            combined_response["content"] += content
                        
                        # Handle reasoning content
                        if delta.get("reasoning_content") is not None:
                            reasoning = delta["reasoning_content"]
                            print(f"   Reasoning: {repr(reasoning)}")
                        
                        # Handle tool calls
                        if delta.get("tool_calls"):
                            for tool_call_delta in delta["tool_calls"]:
                                tool_index = tool_call_delta.get("index", 0)
                                
                                # Initialize tool call if not exists
                                if tool_index not in current_tool_calls:
                                    current_tool_calls[tool_index] = {
                                        "id": None,
                                        "type": "function",
                                        "function": {
                                            "name": None,
                                            "arguments": ""
                                        }
                                    }
                                
                                # Update tool call
                                if tool_call_delta.get("id"):
                                    current_tool_calls[tool_index]["id"] = tool_call_delta["id"]
                                
                                if tool_call_delta.get("function"):
                                    func_delta = tool_call_delta["function"]
                                    if func_delta.get("name"):
                                        current_tool_calls[tool_index]["function"]["name"] = func_delta["name"]
                                    if func_delta.get("arguments") is not None:
                                        current_tool_calls[tool_index]["function"]["arguments"] += func_delta["arguments"]
                                
                                print(f"   Tool Call {tool_index}:")
                                print(f"     ID: {current_tool_calls[tool_index]['id']}")
                                print(f"     Name: {current_tool_calls[tool_index]['function']['name']}")
                                print(f"     Args: {repr(current_tool_calls[tool_index]['function']['arguments'])}")
                        
                        # Handle finish reason
                        if finish_reason:
                            print(f"   Finish Reason: {finish_reason}")
                            if finish_reason == "tool_calls":
                                print("   🔧 Tool calls completed!")
                        
                        # Show usage if available
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                            print(f"   Usage: {usage}")
                    
                    except json.JSONDecodeError as e:
                        print(f"❌ Failed to parse JSON: {e}")
                        print(f"   Raw line: {repr(line_str)}")
        
        print("\n" + "="*80)
        print("📋 COMBINED RESPONSE:")
        print("-" * 30)
        
        # Finalize combined response
        combined_response["tool_calls"] = list(current_tool_calls.values())
        
        # Display combined response nicely
        print(f"Role: {combined_response['role']}")
        print(f"Content: {repr(combined_response['content'])}")
        
        if combined_response["tool_calls"]:
            print(f"\nTool Calls ({len(combined_response['tool_calls'])}):")
            for i, tool_call in enumerate(combined_response["tool_calls"]):
                print(f"  {i+1}. {tool_call['function']['name']}({tool_call['function']['arguments']})")
                print(f"     ID: {tool_call['id']}")
                
                # Try to parse arguments as JSON for prettier display
                try:
                    args = json.loads(tool_call['function']['arguments'])
                    print(f"     Parsed Args: {json.dumps(args, indent=8)}")
                except:
                    print(f"     Raw Args: {tool_call['function']['arguments']}")
        
        print("\n" + "="*80)
        print("🎯 FINAL JSON RESPONSE:")
        print("-" * 25)
        print(json.dumps(combined_response, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_simple_tool_call():
    """Test a simple single tool call"""
    
    url = "http://127.0.0.1:8003/v1/chat/completions"
    
    payload = {
        "model": "llama4",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "temperature": 0,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "required"
    }
    
    print("\n" + "🔸" * 50)
    print("🧪 SIMPLE TOOL CALL TEST")
    print("🔸" * 50)
    print(f"💬 Query: {payload['messages'][0]['content']}")
    print(f"🔧 Tool: {payload['tools'][0]['function']['name']}")
    print()
    
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, stream=True)
        response.raise_for_status()
        
        tool_call = {"function": {"name": None, "arguments": ""}}
        
        for line in response.iter_lines():
            if line and line.decode('utf-8').startswith("data: "):
                json_str = line.decode('utf-8')[6:]
                if json_str.strip() == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(json_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    
                    if delta.get("tool_calls"):
                        tc = delta["tool_calls"][0]
                        if tc.get("function"):
                            if tc["function"].get("name"):
                                tool_call["function"]["name"] = tc["function"]["name"]
                            if tc["function"].get("arguments") is not None:
                                tool_call["function"]["arguments"] += tc["function"]["arguments"]
                
                except json.JSONDecodeError:
                    continue
        
        print(f"🎯 Final Tool Call: {tool_call['function']['name']}({tool_call['function']['arguments']})")
        
        # Parse arguments
        try:
            args = json.loads(tool_call['function']['arguments'])
            print(f"📋 Parsed Arguments: {json.dumps(args, indent=2)}")
        except:
            print(f"📋 Raw Arguments: {tool_call['function']['arguments']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🧪 DeepSeek-V3.1 Streaming Tool Call Test Suite")
    print("=" * 60)
    
    # Test 1: Simple tool call
    test_simple_tool_call()
    
    # Wait a bit between tests
    time.sleep(2)
    
    # Test 2: Complex multi-tool call
    test_streaming_tool_call()
    
    print("\n✅ All tests completed!")
