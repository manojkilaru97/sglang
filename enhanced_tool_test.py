#!/usr/bin/env python3
"""
Enhanced test script for streaming tool calls with DeepSeek-V3.1
Demonstrates multiple tool calls and tool call continuation scenarios
"""

import json
import requests
import time
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ToolCallState:
    """Track state of individual tool calls"""
    id: str = None
    name: str = None
    arguments: str = ""
    completed: bool = False


class StreamingToolCallTester:
    """Enhanced tester for streaming tool calls"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"
    
    def simulate_tool_responses(self, tool_calls: List[Dict]) -> List[Dict]:
        """Simulate tool responses for demonstration"""
        responses = []
        
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"])
                
                if func_name == "get_weather":
                    city = args.get("city", "Unknown")
                    response = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps({
                            "city": city,
                            "temperature": "22°C",
                            "condition": "Sunny",
                            "humidity": "65%"
                        })
                    }
                elif func_name == "calculate":
                    expression = args.get("expression", "0")
                    try:
                        # Simple eval for demo (don't use in production!)
                        result = eval(expression.replace("*", "*").replace("x", "*"))
                        response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({
                                "expression": expression,
                                "result": result
                            })
                        }
                    except:
                        response = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps({"error": "Invalid expression"})
                        }
                else:
                    response = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps({"error": "Unknown tool"})
                    }
                
                responses.append(response)
            except Exception as e:
                responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"error": str(e)})
                })
        
        return responses
    
    def test_streaming_with_conversation(self):
        """Test a full conversation with tool calls and responses"""
        
        print("🔄 FULL CONVERSATION TEST")
        print("=" * 60)
        
        # Initial request
        messages = [
            {"role": "user", "content": "Get weather for Tokyo and calculate 15 * 23"}
        ]
        
        tools = [
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
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
        
        conversation_round = 1
        
        while True:
            print(f"\n🔸 ROUND {conversation_round}")
            print("-" * 40)
            
            # Show current conversation
            print("💬 Current Conversation:")
            for i, msg in enumerate(messages[-3:], 1):  # Show last 3 messages
                role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}
                print(f"  {role_emoji.get(msg['role'], '❓')} {msg['role'].title()}: {msg.get('content', 'N/A')[:100]}...")
                if msg.get('tool_calls'):
                    for tc in msg['tool_calls']:
                        print(f"    🛠️  {tc['function']['name']}({tc['function']['arguments']})")
            
            # Make request
            payload = {
                "model": "llama4",
                "messages": messages,
                "temperature": 0,
                "stream": True,
                "tools": tools,
                "tool_choice": "auto"
            }
            
            print(f"\n📡 Streaming response for round {conversation_round}...")
            
            assistant_response = self.stream_and_parse(payload)
            messages.append(assistant_response)
            
            # If tool calls were made, simulate responses
            if assistant_response.get("tool_calls"):
                print(f"\n🔧 Simulating {len(assistant_response['tool_calls'])} tool response(s)...")
                
                tool_responses = self.simulate_tool_responses(assistant_response["tool_calls"])
                for response in tool_responses:
                    messages.append(response)
                    print(f"  ✅ Tool {response['tool_call_id']}: {response['content'][:100]}...")
                
                conversation_round += 1
                
                # Continue conversation to get final response
                if conversation_round <= 3:  # Limit rounds
                    continue
            
            break
        
        print(f"\n🎯 FINAL CONVERSATION ({len(messages)} messages):")
        print("=" * 50)
        for i, msg in enumerate(messages, 1):
            role_emoji = {"user": "👤", "assistant": "🤖", "tool": "🔧"}
            print(f"{i}. {role_emoji.get(msg['role'], '❓')} {msg['role'].title()}:")
            if msg.get('content'):
                print(f"   💬 {msg['content']}")
            if msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    print(f"   🛠️  {tc['function']['name']}({tc['function']['arguments']})")
            print()
    
    def stream_and_parse(self, payload: Dict) -> Dict:
        """Stream a request and parse the complete response"""
        
        combined_response = {
            "role": "assistant",
            "content": "",
            "tool_calls": []
        }
        
        current_tool_calls = {}
        chunk_count = 0
        
        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    if not line_str.strip() or line_str.strip() == "data: [DONE]":
                        continue
                    
                    if line_str.startswith("data: "):
                        chunk_count += 1
                        json_str = line_str[6:]
                        
                        try:
                            chunk = json.loads(json_str)
                            choice = chunk.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")
                            
                            # Handle role
                            if delta.get("role"):
                                combined_response["role"] = delta["role"]
                            
                            # Handle content
                            if delta.get("content") is not None:
                                combined_response["content"] += delta["content"]
                                if delta["content"]:  # Only print non-empty content
                                    print(f"   📝 {repr(delta['content'])}", end="", flush=True)
                            
                            # Handle tool calls
                            if delta.get("tool_calls"):
                                for tool_call_delta in delta["tool_calls"]:
                                    tool_index = tool_call_delta.get("index", 0)
                                    
                                    if tool_index not in current_tool_calls:
                                        current_tool_calls[tool_index] = {
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": None, "arguments": ""}
                                        }
                                        print(f"\n   🔧 Tool Call {tool_index} started...")
                                    
                                    # Update tool call
                                    if tool_call_delta.get("id"):
                                        current_tool_calls[tool_index]["id"] = tool_call_delta["id"]
                                    
                                    if tool_call_delta.get("function"):
                                        func_delta = tool_call_delta["function"]
                                        if func_delta.get("name"):
                                            current_tool_calls[tool_index]["function"]["name"] = func_delta["name"]
                                            print(f"      📛 Name: {func_delta['name']}")
                                        if func_delta.get("arguments") is not None:
                                            current_tool_calls[tool_index]["function"]["arguments"] += func_delta["arguments"]
                                            if func_delta["arguments"]:  # Only print non-empty arguments
                                                print(f"      🔢 Args: {repr(func_delta['arguments'])}", end="", flush=True)
                            
                            # Handle finish reason
                            if finish_reason:
                                print(f"\n   🏁 Finished: {finish_reason}")
                        
                        except json.JSONDecodeError:
                            continue
            
            # Finalize combined response
            combined_response["tool_calls"] = list(current_tool_calls.values())
            
            print(f"\n✅ Response completed: {len(combined_response['content'])} chars, {len(combined_response['tool_calls'])} tool calls")
            
            return combined_response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return combined_response
    
    def test_reasoning_with_tools(self):
        """Test thinking mode with tool calls"""
        
        print("\n🧠 REASONING + TOOLS TEST")
        print("=" * 50)
        
        payload = {
            "model": "llama4",
            "messages": [
                {"role": "user", "content": "I need to know the weather in Tokyo to plan my trip. Can you help?"}
            ],
            "temperature": 0,
            "stream": True,
            "chat_template_kwargs": {"thinking": True},
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
            "tool_choice": "auto"
        }
        
        print("🤔 Testing reasoning (thinking=True) with tools...")
        response = self.stream_and_parse(payload)
        
        print("\n📋 Final Response Summary:")
        print(f"   Content: {response['content'][:100]}...")
        print(f"   Tool Calls: {len(response.get('tool_calls', []))}")


def main():
    print("🚀 Enhanced DeepSeek-V3.1 Streaming Tool Call Test Suite")
    print("=" * 70)
    
    tester = StreamingToolCallTester()
    
    # Test 1: Reasoning with tools
    tester.test_reasoning_with_tools()
    
    time.sleep(2)
    
    # Test 2: Full conversation flow
    tester.test_streaming_with_conversation()
    
    print("\n🎉 All enhanced tests completed!")


if __name__ == "__main__":
    main()
