#!/usr/bin/env python3
"""
Test script to compare chat template behaviors between:
1. Original DeepSeek chat template
2. Combined DeepSeek chat template with tool support

Tests different scenarios:
- Non-thinking requests
- Thinking requests 
- With system prompts
- All with add_generation_prompt=True
"""

import json
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def load_template(template_path):
    """Load a Jinja2 template from file path."""
    template_dir = Path(template_path).parent
    template_name = Path(template_path).name
    
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template(template_name)

def render_template(template, messages, thinking=False, system_prompt=None, add_generation_prompt=True):
    """Render a template with given parameters."""
    # Add system message if provided
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    
    context = {
        "messages": messages,
        "thinking": thinking,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "<｜begin▁of▁sentence｜>",
        "tools": None  # No tools for these tests
    }
    
    return template.render(**context)

def test_scenario(name, original_template, combined_template, messages, thinking=False, system_prompt=None):
    """Test a specific scenario with both templates."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    
    print(f"Messages: {json.dumps(messages, indent=2)}")
    if system_prompt:
        print(f"System Prompt: {system_prompt}")
    print(f"Thinking: {thinking}")
    print(f"add_generation_prompt: True")
    
    print(f"\n{'-'*30} ORIGINAL TEMPLATE {'-'*30}")
    original_result = None
    try:
        original_result = render_template(original_template, messages, thinking, system_prompt)
        print(repr(original_result))
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    
    print(f"\n{'-'*30} PERFECT TEMPLATE {'-'*30}")
    combined_result = None
    try:
        combined_result = render_template(combined_template, messages, thinking, system_prompt)
        print(repr(combined_result))
    except Exception as e:
        print(f"ERROR: {e}")
        raise
    
    # Assert that both templates produce identical output for non-tool scenarios
    assert original_result == combined_result, f"Templates differ for scenario '{name}':\nOriginal: {repr(original_result)}\nPerfect:  {repr(combined_result)}"
    print(f"\n✅ ASSERTION PASSED: Templates produce identical output")
    
    print(f"\n{'-'*60}")

def main():
    # Load templates
    original_template = load_template("/home/scratch.mkilaru_coreai/DeepSeek-V3.1/assets/chat_template.jinja")
    combined_template = load_template("/sgl-workspace/sglang/examples/chat_template/chat_template_deepseek_perfect.jinja")
    
    # Test messages
    simple_messages = [
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    
    multi_turn_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with math?"}
    ]
    
    # Test scenarios
    test_scenario(
        "Simple User Query - Non-thinking",
        original_template, combined_template,
        simple_messages,
        thinking=False
    )
    
    test_scenario(
        "Simple User Query - Thinking",
        original_template, combined_template,
        simple_messages,
        thinking=True
    )
    
    test_scenario(
        "With System Prompt - Non-thinking", 
        original_template, combined_template,
        simple_messages,
        thinking=False,
        system_prompt="You are a helpful math tutor."
    )
    
    test_scenario(
        "With System Prompt - Thinking",
        original_template, combined_template, 
        simple_messages,
        thinking=True,
        system_prompt="You are a helpful math tutor."
    )
    
    test_scenario(
        "Multi-turn Conversation - Non-thinking",
        original_template, combined_template,
        multi_turn_messages,
        thinking=False
    )
    
    test_scenario(
        "Multi-turn Conversation - Thinking", 
        original_template, combined_template,
        multi_turn_messages,
        thinking=True
    )
    
    test_scenario(
        "Multi-turn with System - Thinking",
        original_template, combined_template,
        multi_turn_messages, 
        thinking=True,
        system_prompt="You are a helpful assistant that thinks step by step."
    )
    
    print(f"\n{'='*60}")
    print("🎉 ALL TESTS PASSED! Templates produce identical output for all non-tool scenarios.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
