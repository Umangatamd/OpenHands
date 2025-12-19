"""LLM-based kernel result reflector.

This module provides functions to analyze kernel test results and suggest
targeted improvements.
"""

import json
import os
from typing import Any

# Try to import litellm for LLM calls
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

# Try to import anthropic for AMD gateway
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def is_amd_model(model: str) -> bool:
    """Check if the model is an AMD gateway model."""
    model_lower = model.lower()
    return (
        model.startswith("amd/") or
        "claude-sonnet-4.5" in model_lower or
        "claude-opus-4.5" in model_lower or
        "gpt-5" in model_lower
    )


def call_amd_gateway(messages: list, model: str, temperature: float = 0.2) -> str:
    """
    Call AMD's LLM gateway directly.
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not available")
    
    api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
    if not api_key:
        raise ValueError("AMD_LLM_API_KEY or LLM_GATEWAY_KEY not set")
    
    model_name = model.removeprefix("amd/")
    
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "unknown")
    
    client = anthropic.Anthropic(
        api_key="dummy",
        base_url="https://llm-api.amd.com/Anthropic",
        default_headers={
            "Ocp-Apim-Subscription-Key": api_key,
            "user": user,
            "anthropic-version": "2023-10-16",
        },
    )
    
    system_content = ""
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            filtered_messages.append(msg)
    
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        system=system_content if system_content else anthropic.NOT_GIVEN,
        messages=filtered_messages,
        temperature=temperature,
    )
    
    return response.content[0].text


# GEAK-agent Reflection Prompt (adapted for OpenHands)
REFLECTION_PROMPT = """
You are an expert Python programmer specializing in writing and optimizing Triton kernels for AMD GPUs using the ROCm environment.
You are tasked with analyzing a kernel optimization attempt and providing insights on how to improve it.

# Current Kernel:
```python
{kernel_code}
```

# Test Result:
- Correctness: {correctness_status}
- Speedup vs baseline: {speedup}x
- Test Output/Error: 
{test_output}

# Optimization History:
{history}

# Strategies Already Tried:
{tried_strategies}

## AMD GPU Optimization Guidelines:
1. **AMD Compatibility:** Code must be compatible with AMD GPUs and ROCm. **DO NOT use CUDA-specific features or functions (e.g., `tl.libdevice`).**
2. **Autotuning:** Maximize performance by exploring:
   - BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes [32, 64, 128, 256, 512, 1024, 2048]
   - num_stages: Pipeline depth (1 if no GEMM, 2 if single GEMM, 1 if two fused GEMMs)
   - num_warps: Range [1-16] ONLY. Values >16 are INVALID on AMD.
3. **Memory Access:** Optimize for coalesced access, minimize redundant reads/writes
4. **Algorithmic Improvements:** e.g., naive softmax vs online softmax vs fused softmax
5. **Common Pitfalls:**
   - Grid/Program ID mismatch (1D grid but using 2D program_id)
   - tl.arange arguments must be tl.constexpr
   - tl.dot inputs must be 2D and compatible types

## Your Task:
1. Analyze WHY this attempt achieved this result
2. Identify the specific bottleneck (memory, compute, correctness, launch_overhead)
3. Suggest a DIFFERENT approach than what's been tried
4. Provide specific code hints for improvement

**Output Format:**
Return a JSON object with this exact structure:
{{
    "analysis": "<Why this result occurred - be specific about technical reasons>",
    "is_improvement": <bool - compared to previous attempts>,
    "bottleneck": "<memory | compute | correctness | launch_overhead | other>",
    "root_cause": "<Specific technical reason for the bottleneck>",
    "next_strategy": {{
        "name": "<e.g., split-K, persistent kernel, vectorized loads, larger blocks>",
        "description": "<What specific changes to make to the kernel>",
        "expected_impact": "<Why this might help and estimated improvement (e.g., 1.5-2x)>",
        "code_hints": "<Specific code patterns, values, or decorators to try. E.g., BLOCK_M=256, num_warps=8, use tl.load with eviction_policy>"
    }},
    "avoid": ["<strategies that won't help based on this result>"],
    "exploration_status": {{
        "strategies_exhausted": <bool>,
        "confidence_in_next": "<high | medium | low>",
        "recommendation": "<continue | stop | try_radically_different>"
    }}
}}

Only output valid JSON, no additional text.
"""


def reflect_on_kernel_result(
    kernel_code: str,
    test_output: str,
    speedup: float = 0.0,
    correctness_status: str = "unknown",
    history: str = "",
    tried_strategies: str = "",
    model: str | None = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Analyze kernel test results and get targeted improvement suggestions.
    
    This function makes an LLM call to analyze why the kernel achieved its
    result and what specific changes should be tried next.
    
    Args:
        kernel_code: The Triton kernel code that was tested
        test_output: Output from running the test (stdout/stderr)
        speedup: Measured speedup vs baseline (e.g., 1.5 means 50% faster)
        correctness_status: "passed", "failed", or "unknown"
        history: Summary of previous optimization attempts
        tried_strategies: List of strategies already tried
        model: LLM model to use (default: from EVAL_LLM_MODEL env var)
        verbose: Whether to print the reflection result
    
    Returns:
        dict with analysis and next steps:
        {
            "analysis": str,
            "bottleneck": str,
            "next_strategy": {...},
            "exploration_status": {...}
        }
    
    Example:
        >>> result = reflect_on_kernel_result(
        ...     kernel_code=my_kernel,
        ...     test_output="Speedup: 1.3x",
        ...     speedup=1.3,
        ...     correctness_status="passed",
        ...     tried_strategies="tiled GEMM, added autotuning"
        ... )
        >>> print(result['next_strategy']['description'])
    """
    # Get model from env or use default
    if model is None:
        model = os.environ.get('EVAL_LLM_MODEL', 'amd/claude-sonnet-4.5')
    
    # Format history if not provided
    if not history:
        history = "No previous attempts"
    if not tried_strategies:
        tried_strategies = "None yet"
    
    try:
        prompt = REFLECTION_PROMPT.format(
            kernel_code=kernel_code,
            correctness_status=correctness_status,
            speedup=speedup,
            test_output=test_output[:2000],  # Limit output length
            history=history,
            tried_strategies=tried_strategies
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Use AMD gateway for AMD models, litellm otherwise
        if is_amd_model(model) and ANTHROPIC_AVAILABLE:
            result_text = call_amd_gateway(messages, model, temperature=0.2)
        elif LITELLM_AVAILABLE:
            api_key = os.environ.get('LLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
            response = completion(
                model=model,
                messages=messages,
                api_key=api_key,
                temperature=0.2,
            )
            result_text = response.choices[0].message.content
        else:
            return {
                "error": "No LLM backend available",
                "analysis": "Cannot analyze - no LLM backend",
                "bottleneck": "unknown",
                "next_strategy": {"name": "unknown", "description": "Configure LLM first"},
                "exploration_status": {"recommendation": "stop"}
            }
        
        # Parse JSON response
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0]
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0]
        
        result = json.loads(result_text.strip())
        
        # Ensure all expected fields exist
        if 'analysis' not in result:
            result['analysis'] = "Analysis not available"
        if 'bottleneck' not in result:
            result['bottleneck'] = "unknown"
        if 'next_strategy' not in result:
            result['next_strategy'] = {
                "name": "unknown",
                "description": "No suggestion available"
            }
        if 'exploration_status' not in result:
            result['exploration_status'] = {
                "recommendation": "continue"
            }
        
        if verbose:
            print(f"\n{'='*60}")
            print("KERNEL REFLECTION RESULT")
            print(f"{'='*60}")
            print(f"Bottleneck: {result.get('bottleneck', 'unknown')}")
            print(f"\nAnalysis: {result.get('analysis', '')}")
            print(f"\nRoot Cause: {result.get('root_cause', 'N/A')}")
            print(f"\n--- Next Strategy: {result['next_strategy'].get('name', 'N/A')} ---")
            print(f"Description: {result['next_strategy'].get('description', '')}")
            print(f"Expected Impact: {result['next_strategy'].get('expected_impact', '')}")
            if result['next_strategy'].get('code_hints'):
                print(f"Code Hints: {result['next_strategy']['code_hints']}")
            print(f"\nRecommendation: {result['exploration_status'].get('recommendation', 'continue')}")
            if result.get('avoid'):
                print(f"Avoid: {', '.join(result['avoid'])}")
            print(f"{'='*60}\n")
        
        return result
        
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Failed to parse LLM response as JSON: {e}",
            "raw_response": result_text if 'result_text' in locals() else None,
            "analysis": "Reflection failed - could not parse response",
            "bottleneck": "unknown",
            "next_strategy": {"name": "retry", "description": "Reflection failed, try again"},
            "exploration_status": {"recommendation": "continue"}
        }
        if verbose:
            print(f"Reflection Error: {error_result['error']}")
        return error_result
        
    except Exception as e:
        error_result = {
            "error": f"LLM call failed: {e}",
            "analysis": f"Reflection failed: {str(e)}",
            "bottleneck": "unknown",
            "next_strategy": {"name": "retry", "description": "Reflection failed"},
            "exploration_status": {"recommendation": "continue"}
        }
        if verbose:
            print(f"Reflection Error: {error_result['error']}")
        return error_result

