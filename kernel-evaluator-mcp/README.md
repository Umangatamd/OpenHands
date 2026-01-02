# Kernel Evaluator MCP Server

An MCP (Model Context Protocol) server that provides LLM-based evaluation tools for Triton/AMD GPU kernel optimization.

## Features

- **evaluate_kernel_quality**: Analyze Triton kernels across 9 quality criteria specific to AMD GPUs
- **reflect_on_kernel_result**: Get targeted improvement suggestions based on test results
- **check_kernel_compatibility**: Quick AMD compatibility check for kernel code
- **get_amd_gpu_specs**: Reference specs for AMD MI350X GPU optimization

## Installation

```bash
pip install kernel-evaluator-mcp
```

Or install from source:

```bash
git clone https://github.com/amd/kernel-evaluator-mcp
cd kernel-evaluator-mcp
pip install -e .
```

## Usage

### With Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "kernel-evaluator": {
      "command": "kernel-evaluator-mcp",
      "env": {
        "AMD_LLM_API_KEY": "your-amd-gateway-key"
      }
    }
  }
}
```

### With Cursor

Add to Cursor settings (MCP Servers section):

```json
{
  "name": "kernel-evaluator",
  "command": "kernel-evaluator-mcp",
  "env": {
    "AMD_LLM_API_KEY": "your-amd-gateway-key"
  }
}
```

### With Any MCP Client (stdio)

```bash
# Set API key
export AMD_LLM_API_KEY="your-key"
# Or for OpenAI
export OPENAI_API_KEY="your-key"

# Run the server
kernel-evaluator-mcp
```

### Programmatic Usage

```python
from kernel_evaluator_mcp.server import evaluate_kernel_quality, reflect_on_kernel_result

# Evaluate a kernel
result = evaluate_kernel_quality("""
@triton.jit
def my_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, x * 2)
""")

print(f"Total Score: {result[total_score]}/9.0")
print(f"Ready to test: {result[ready_to_test]}")
```

## Tools

### evaluate_kernel_quality

Evaluates Triton kernel code across 9 criteria (0.0-1.0 each):

1. **fusion_intelligence** - Operation fusion efficiency
2. **block_size_appropriateness** - AMD-appropriate block sizes
3. **memory_access_efficiency** - Coalesced/aligned memory access
4. **algorithmic_complexity** - Loop efficiency
5. **warp_wavefront_utilization** - AMD wavefront (64 threads) usage
6. **software_pipelining** - num_stages configuration
7. **numerical_stability** - Safe numerical operations
8. **correctness_and_portability** - Edge cases and AMD compatibility
9. **optimization_scope** - Use of known optimization techniques

### reflect_on_kernel_result

Analyzes test results and provides:
- Root cause analysis
- Bottleneck identification (memory/compute/correctness)
- Next optimization strategy with code hints
- Confidence level and continue/stop recommendation

### check_kernel_compatibility

Quick scan for AMD compatibility:
- Detects CUDA-only features (tl.libdevice)
- Validates num_warps range (1-16)
- Checks block size limits (max 1024)
- Verifies wavefront alignment (multiples of 64)

### get_amd_gpu_specs

Returns AMD MI350X specifications:
- Hardware specs (CUs, wavefront size, memory)
- Optimization guidelines
- Memory hierarchy details

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AMD_LLM_API_KEY` | AMD LLM Gateway API key |
| `LLM_GATEWAY_KEY` | Alternative AMD gateway key |
| `OPENAI_API_KEY` | OpenAI API key (fallback) |
| `LLM_API_KEY` | Generic LLM API key |

## License

MIT License
