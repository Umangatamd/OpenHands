"""LLM-based kernel quality evaluator.

This module provides functions to evaluate Triton kernel quality using an LLM
before running expensive tests.
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


def call_amd_gateway(messages: list, model: str, temperature: float = 0.1) -> str:
    """
    Call AMD's LLM gateway directly.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (e.g., 'amd/claude-sonnet-4.5')
        temperature: Generation temperature
        
    Returns:
        Response content as string
    """
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not available. Install with: pip install anthropic")
    
    api_key = os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY")
    if not api_key:
        raise ValueError("AMD_LLM_API_KEY or LLM_GATEWAY_KEY environment variable not set")
    
    # Remove 'amd/' prefix
    model_name = model.removeprefix("amd/")
    
    # Get user for header
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER", "unknown")
    
    # Create Anthropic client with AMD gateway
    client = anthropic.Anthropic(
        api_key="dummy",  # Auth via header
        base_url="https://llm-api.amd.com/Anthropic",
        default_headers={
            "Ocp-Apim-Subscription-Key": api_key,
            "user": user,
            "anthropic-version": "2023-10-16",
        },
    )
    
    # Extract system message if present
    system_content = ""
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            filtered_messages.append(msg)
    
    # Call the API
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        system=system_content if system_content else anthropic.NOT_GIVEN,
        messages=filtered_messages,
        temperature=temperature,
    )
    
    return response.content[0].text


# GEAK-agent LLM Evaluator Prompt (adapted for fixed-config testing)
KERNEL_EVALUATION_PROMPT = """
Evaluate the following Triton kernel on a scale of 0.0 to 1.0 for each of the listed criteria.
For each criteria you must provide:
* A score (float between 0.0 (worst) and 1.0 (best))
* A brief justification (1–2 lines)
* Actionable suggestions, if any

**Target Hardware: Single AMD MI350X GPU (gfx950)**
- 256 Compute Units
- Wavefront size: 64 threads
- Max 32 waves per CU
- Max workgroup size: 1024 threads
- 288 GB HBM3 memory
- Cache line: 128 bytes
- Tests run on 1 GPU (HIP_VISIBLE_DEVICES=0)

**Evaluation Criteria:**
1. Fusion Intelligence: Does the kernel smartly fuse compatible operations to reduce memory I/O and kernel launches?
2. Block Size Appropriateness: Are the block sizes appropriate for the problem size and AMD hardware? (BLOCK should be multiples of 64, max 1024 for workgroup)
3. Memory Access Efficiency: Does it optimize memory layout, coalesced access (128-byte aligned), and reduce redundant reads/writes?
4. Algorithmic Complexity: Does it fuse multiple for-loops in one smartly? Are there redundant nested for-loops?
5. Warp/Wavefront Utilization: Does it use thread blocks that fully utilize AMD wavefronts (64 threads)? Is occupancy maximized?
6. Software Pipelining: Is num_stages set appropriately? (1 for no GEMM, 2 for single GEMM, 1 for fused double-GEMM). Valid range [1,16].
7. Numerical Stability: Is it numerically safe for large input ranges (e.g., uses max-subtraction in softmax, clamps, etc.)?
8. Correctness and Portability: Does the kernel handle edge cases (e.g., sizes not divisible by block size)? Does it avoid CUDA-only features like `tl.libdevice`?
9. Optimization Scope: Is the given kernel missing techniques from well-known optimization methods? e.g. softmax kernel vs online softmax vs fused softmax.

Evaluate the following Triton kernel using the criteria above:
```python
{kernel_code}
```

Provide scores and reasoning for each evaluation criteria in the JSON format as follows:
{{
    "fusion_intelligence": <float 0.0-1.0>,
    "block_size_appropriateness": <float 0.0-1.0>,
    "memory_access_efficiency": <float 0.0-1.0>,
    "algorithmic_complexity": <float 0.0-1.0>,
    "warp_wavefront_utilization": <float 0.0-1.0>,
    "software_pipelining": <float 0.0-1.0>,
    "numerical_stability": <float 0.0-1.0>,
    "correctness_and_portability": <float 0.0-1.0>,
    "optimization_scope": <float 0.0-1.0>,
    "reasoning": "<Your true and logical reasoning for scoring every single above criteria and any actionable suggestions to improve the performance on a given triton kernel. Line breaks are not allowed.>"
}}

Only output valid JSON, no additional text.
"""


def evaluate_kernel_quality(
    kernel_code: str,
    model: str | None = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Evaluate Triton kernel quality using LLM before running tests.
    
    This function makes an LLM call to analyze the kernel code and provide
    structured feedback on quality, potential issues, and improvements.
    
    Args:
        kernel_code: The Triton kernel code to evaluate
        model: LLM model to use (default: from EVAL_LLM_MODEL env var or gpt-4o-mini)
        verbose: Whether to print the evaluation result
    
    Returns:
        dict with scores, issues, and suggestions:
        {
            "scores": {...},
            "total_score": float,
            "top_issues": [...],
            "suggested_improvements": [...],
            "ready_to_test": bool,
            "brief_analysis": str
        }
    
    Example:
        >>> result = evaluate_kernel_quality(my_kernel_code)
        >>> if result['ready_to_test']:
        ...     # Run the actual test
        ... else:
        ...     print("Fix these issues first:", result['top_issues'])
    """
    if not LITELLM_AVAILABLE:
        return {
            "error": "litellm not available. Install with: pip install litellm",
            "scores": {},
            "total_score": 0.0,
            "top_issues": ["Cannot evaluate - litellm not installed"],
            "suggested_improvements": [],
            "ready_to_test": False,
            "brief_analysis": "Evaluation unavailable"
        }
    
    # Get model from env or use default
    if model is None:
        model = os.environ.get('EVAL_LLM_MODEL', 'amd/claude-sonnet-4.5')
    
    try:
        prompt = KERNEL_EVALUATION_PROMPT.format(kernel_code=kernel_code)
        messages = [{"role": "user", "content": prompt}]
        
        # Use AMD gateway for AMD models, litellm otherwise
        if is_amd_model(model) and ANTHROPIC_AVAILABLE:
            result_text = call_amd_gateway(messages, model, temperature=0.1)
        elif LITELLM_AVAILABLE:
            api_key = os.environ.get('LLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
            response = completion(
                model=model,
                messages=messages,
                api_key=api_key,
                temperature=0.1,
            )
            result_text = response.choices[0].message.content
        else:
            return {
                "error": "No LLM backend available. Install anthropic or litellm.",
                "scores": {},
                "total_score": 0.0,
                "top_issues": ["Cannot evaluate - no LLM backend"],
                "suggested_improvements": [],
                "ready_to_test": False,
                "brief_analysis": "Evaluation unavailable"
            }
        
        # Parse JSON response
        # Try to extract JSON if wrapped in markdown
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0]
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0]
        
        result = json.loads(result_text.strip())
        
        # GEAK-agent format: individual scores + reasoning
        # Score fields to look for (updated for fixed-config testing)
        score_fields = [
            'fusion_intelligence',
            'block_size_appropriateness', 
            'memory_access_efficiency',
            'algorithmic_complexity',
            'warp_wavefront_utilization',
            'software_pipelining',
            'numerical_stability',
            'correctness_and_portability',
            'optimization_scope'
        ]
        
        if verbose:
            print(f"\n{'='*60}")
            print("KERNEL EVALUATION (GEAK-agent format)")
            print(f"{'='*60}")
            
            total = 0.0
            for field in score_fields:
                score = result.get(field, 0.0)
                total += score
                # Format score with indicator
                indicator = "✓" if score >= 0.7 else ("○" if score >= 0.4 else "✗")
                print(f"  {indicator} {field}: {score:.2f}")
            
            print(f"\n  Total: {total:.1f}/9.0")
            print(f"\nReasoning:")
            reasoning = result.get('reasoning', 'No reasoning provided')
            # Wrap long reasoning
            if len(reasoning) > 200:
                print(f"  {reasoning[:200]}...")
            else:
                print(f"  {reasoning}")
            print(f"{'='*60}\n")
        
        return result
        
    except json.JSONDecodeError as e:
        error_result = {
            "error": f"Failed to parse LLM response as JSON: {e}",
            "raw_response": result_text if 'result_text' in locals() else None,
            "scores": {},
            "total_score": 0.0,
            "top_issues": ["Evaluation failed - could not parse response"],
            "suggested_improvements": [],
            "ready_to_test": False,
            "brief_analysis": "Evaluation failed"
        }
        if verbose:
            print(f"Evaluation Error: {error_result['error']}")
        return error_result
        
    except Exception as e:
        error_result = {
            "error": f"LLM call failed: {e}",
            "scores": {},
            "total_score": 0.0,
            "top_issues": [f"Evaluation failed: {str(e)}"],
            "suggested_improvements": [],
            "ready_to_test": False,
            "brief_analysis": "Evaluation failed"
        }
        if verbose:
            print(f"Evaluation Error: {error_result['error']}")
        return error_result

