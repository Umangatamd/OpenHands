#!/usr/bin/env python3
"""Quick test script for AMD LLM integration in OpenHands.

Usage:
    # Set your API key first
    export AMD_LLM_API_KEY="your-key-here"
    
    # Run the test
    python test_amd_llm.py
"""

import os
import sys

# Add the openhands directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openhands.core.config.llm_config import LLMConfig
from openhands.llm.amd_llm import AmdLLM, is_amd_model


def test_is_amd_model():
    """Test the is_amd_model function."""
    print("Testing is_amd_model()...")
    
    # Should be True
    assert is_amd_model("amd/gpt-5") == True
    assert is_amd_model("gpt-5") == True
    assert is_amd_model("gpt-5-codex") == True
    assert is_amd_model("claude-opus-4.5") == True
    assert is_amd_model("claude-sonnet-4.5") == True
    
    # Should be False
    assert is_amd_model("gpt-4") == False
    assert is_amd_model("claude-3-sonnet") == False
    assert is_amd_model("anthropic/claude-3") == False
    
    print("✓ is_amd_model() tests passed!")


def test_amd_llm_initialization():
    """Test AmdLLM initialization."""
    print("\nTesting AmdLLM initialization...")
    
    # Check if API key is set
    api_key = os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY")
    if not api_key:
        print("⚠ No API key found. Set AMD_LLM_API_KEY to test actual API calls.")
        print("  Skipping initialization test (requires API key).")
        return False
    
    # Test Claude model initialization
    config = LLMConfig(
        model="amd/claude-opus-4.5",
        temperature=0.0,
        max_output_tokens=1024,
    )
    
    try:
        llm = AmdLLM(config=config, service_id="test-claude")
        print(f"✓ AmdLLM initialized successfully: {llm}")
        return True
    except Exception as e:
        print(f"✗ AmdLLM initialization failed: {e}")
        return False


def test_amd_llm_completion():
    """Test AmdLLM completion (requires API key)."""
    print("\nTesting AmdLLM completion...")
    
    api_key = os.getenv("AMD_LLM_API_KEY") or os.getenv("LLM_GATEWAY_KEY")
    if not api_key:
        print("⚠ Skipping completion test (no API key)")
        return False
    
    # Test with Claude
    config = LLMConfig(
        model="amd/claude-sonnet-4.5",
        temperature=0.0,
        max_output_tokens=256,
    )
    
    try:
        llm = AmdLLM(config=config, service_id="test-completion")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ]
        
        print("  Sending test query...")
        response = llm.completion(messages=messages)
        
        content = response.choices[0].message.content
        print(f"  Response: {content}")
        print(f"  Usage: {response.usage}")
        print("✓ Completion test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Completion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("AMD LLM Integration Test for OpenHands")
    print("=" * 60)
    
    # Run tests
    test_is_amd_model()
    
    has_key = test_amd_llm_initialization()
    
    if has_key:
        test_amd_llm_completion()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()



