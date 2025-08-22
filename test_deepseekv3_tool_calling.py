#!/usr/bin/env python3
"""
Comprehensive test script for DeepSeek V3.1 tool calling functionality.

Tests:
1. Tool calling with and without thinking mode
2. Parallel tool calling
3. Streaming and non-streaming responses
4. Normal queries without tools (with/without thinking and streaming)
"""

import json
import time
import requests
from typing import List, Dict, Any

SERVER_URL = "http://localhost:8003/v1/chat/completions"

# Sample tools for testing
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
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
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
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
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def make_request(messages: List[Dict], tools: List[Dict] = None, thinking: bool = False, 
                stream: bool = False, model: str = "llama4") -> Dict[str, Any]:
    """Make a request to the chat completions endpoint."""
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": 512,
        "temperature": 0.1
    }
    
    if tools:
        payload["tools"] = tools
        
    if thinking is not None:
        payload["chat_template_kwargs"] = {"thinking": thinking}
    
    print(f"\n{'='*80}")
    print(f"REQUEST: thinking={thinking}, stream={stream}, tools={'Yes' if tools else 'No'}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print(f"{'='*80}")
    
    try:
        if stream:
            response = requests.post(SERVER_URL, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            print("STREAMING RESPONSE:")
            full_content = ""
            tool_calls = []
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0]['delta']
                                if 'content' in delta and delta['content']:
                                    print(delta['content'], end='', flush=True)
                                    full_content += delta['content']
                                if 'tool_calls' in delta and delta['tool_calls']:
                                    for tc in delta['tool_calls']:
                                        print(f"\nTOOL CALL: {tc}")
                                        tool_calls.append(tc)
                        except json.JSONDecodeError:
                            continue
            
            print(f"\n\nFULL CONTENT: {repr(full_content)}")
            if tool_calls:
                print(f"TOOL CALLS: {json.dumps(tool_calls, indent=2)}")
            
            return {"content": full_content, "tool_calls": tool_calls}
            
        else:
            response = requests.post(SERVER_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            print("NON-STREAMING RESPONSE:")
            print(json.dumps(result, indent=2))
            
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']
                content = message.get('content', '')
                tool_calls = message.get('tool_calls', [])
                
                return {"content": content, "tool_calls": tool_calls}
            
            return {"content": "", "tool_calls": []}
            
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}

def test_normal_queries():
    """Test normal queries without tools."""
    print(f"\n{'#'*80}")
    print("TESTING NORMAL QUERIES (NO TOOLS)")
    print(f"{'#'*80}")
    
    messages = [{"role": "user", "content": "What is the capital of France? Explain briefly."}]
    
    # Test all combinations
    for thinking in [False, True]:
        for stream in [False, True]:
            make_request(messages, tools=None, thinking=thinking, stream=stream)
            time.sleep(1)

def test_single_tool_calling():
    """Test single tool calling."""
    print(f"\n{'#'*80}")
    print("TESTING SINGLE TOOL CALLING")
    print(f"{'#'*80}")
    
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    
    # Test with and without thinking, with and without streaming
    for thinking in [False, True]:
        for stream in [False, True]:
            make_request(messages, tools=TOOLS, thinking=thinking, stream=stream)
            time.sleep(1)

def test_parallel_tool_calling():
    """Test parallel tool calling."""
    print(f"\n{'#'*80}")
    print("TESTING PARALLEL TOOL CALLING")
    print(f"{'#'*80}")
    
    messages = [{"role": "user", "content": "I need the weather in both New York and Los Angeles, and also calculate 15 * 23."}]
    
    # Test with and without thinking, with and without streaming
    for thinking in [False, True]:
        for stream in [False, True]:
            make_request(messages, tools=TOOLS, thinking=thinking, stream=stream)
            time.sleep(1)

def test_complex_tool_scenario():
    """Test a complex multi-turn tool calling scenario."""
    print(f"\n{'#'*80}")
    print("TESTING COMPLEX MULTI-TURN TOOL SCENARIO")
    print(f"{'#'*80}")
    
    # First request: user asks for weather
    messages = [{"role": "user", "content": "What's the weather in Tokyo? Use Celsius."}]
    
    result = make_request(messages, tools=TOOLS, thinking=True, stream=False)
    
    if result.get('tool_calls'):
        # Simulate tool response
        messages.append({
            "role": "assistant",
            "content": result.get('content', ''),
            "tool_calls": result['tool_calls']
        })
        
        # Add tool response
        for tool_call in result['tool_calls']:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get('id', 'call_123'),
                "content": json.dumps({
                    "location": "Tokyo, Japan",
                    "temperature": 22,
                    "unit": "celsius",
                    "description": "Sunny with light clouds"
                })
            })
        
        # Continue conversation
        messages.append({
            "role": "user", 
            "content": "Great! Now can you also search for recent news about Tokyo weather patterns?"
        })
        
        make_request(messages, tools=TOOLS, thinking=True, stream=True)

def test_edge_cases():
    """Test edge cases and error scenarios."""
    print(f"\n{'#'*80}")
    print("TESTING EDGE CASES")
    print(f"{'#'*80}")
    
    # Test with empty tools list
    messages = [{"role": "user", "content": "What's the weather?"}]
    make_request(messages, tools=[], thinking=False, stream=False)
    
    # Test with very long message
    long_message = "Please " + "really " * 100 + "help me with the weather."
    messages = [{"role": "user", "content": long_message}]
    make_request(messages, tools=TOOLS, thinking=True, stream=False)

def main():
    """Run all tests."""
    print("Starting DeepSeek V3.1 Tool Calling Tests")
    print(f"Server URL: {SERVER_URL}")
    
    try:
        # Test server connectivity
        response = requests.get("http://localhost:8003/health", timeout=5)
        print(f"Server health check: {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not reach health endpoint: {e}")
    
    # Run all test categories
    test_normal_queries()
    test_single_tool_calling() 
    test_parallel_tool_calling()
    test_complex_tool_scenario()
    test_edge_cases()
    
    print(f"\n{'#'*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'#'*80}")

if __name__ == "__main__":
    main()
