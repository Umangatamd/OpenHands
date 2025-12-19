---
name: kernel-optimization
type: knowledge
version: 2.0.0
agent: CodeActAgent
triggers:
- triton
- kernel
- gpu
- rocm
- amd
- optimize
- speedup
- gemm
- mla
- moe
- performance
- mi250
- mi250x
- mi300
- mi300x
- mi355x
- cdna
- hbm
- wavefront
- compute unit
- memory bandwidth
- tflops
- autotuning
- block size
---

# Triton Kernel Optimization for AMD GPUs (ROCm)

You have access to specialized kernel optimization tools. Use them via bash with Python.

## Available Tools

### 1. Evaluate Kernel Quality (BEFORE running tests)
```bash
/usr/bin/python3.12 << 'EOF'
import sys
sys.path.insert(0, "/home/upandey/OpenHands-dev/openhands/runtime/plugins/agent_skills")
from kernel_tools import evaluate_kernel_quality

kernel_code = """
# Your kernel code here
"""
result = evaluate_kernel_quality(kernel_code)
# Returns JSON with 9 scores (0.0-1.0 each) + reasoning:
# {
#   "fusion_intelligence": 0.0-1.0,
#   "block_size_appropriateness": 0.0-1.0,
#   "memory_access_efficiency": 0.0-1.0,
#   "algorithmic_complexity": 0.0-1.0,
#   "warp_wavefront_utilization": 0.0-1.0,
#   "software_pipelining": 0.0-1.0,
#   "numerical_stability": 0.0-1.0,
#   "correctness_and_portability": 0.0-1.0,
#   "optimization_scope": 0.0-1.0,
#   "reasoning": "Detailed reasoning and suggestions..."
# }
EOF
```

**Evaluation Criteria (0.0-1.0 each, max 9.0 total):**
1. **fusion_intelligence** - Fuses compatible operations to reduce memory I/O?
2. **block_size_appropriateness** - Block sizes appropriate for AMD hardware? (multiples of 64, max 1024)
3. **memory_access_efficiency** - Coalesced access (128-byte aligned), minimal redundant reads/writes?
4. **algorithmic_complexity** - Efficient loops, no redundant nesting, smart fusion?
5. **warp_wavefront_utilization** - Fills AMD wavefronts (64 threads per wavefront)?
6. **software_pipelining** - num_stages set correctly? (1 for no GEMM, 2 for single GEMM, 1 for fused)
7. **numerical_stability** - Safe for large inputs? (e.g., max-subtraction in softmax)
8. **correctness_and_portability** - Edge cases handled? No CUDA-only features like tl.libdevice?
9. **optimization_scope** - Uses known optimal algorithms? (e.g., online softmax vs naive)

### 2. Reflect on Test Results (AFTER running tests)
```bash
/usr/bin/python3.12 << 'EOF'
import sys
sys.path.insert(0, "/home/upandey/OpenHands-dev/openhands/runtime/plugins/agent_skills")
from kernel_tools import reflect_on_kernel_result, get_optimization_history

result = reflect_on_kernel_result(
    kernel_code=code,
    test_output="output from test",
    speedup=1.3,
    correctness_status="passed",  # or "failed"
    history=get_optimization_history(),
    tried_strategies="strategy1, strategy2"
)
# Returns: {analysis, bottleneck, root_cause, next_strategy, avoid, exploration_status}
EOF
```

### 3. Track Optimization Attempts
```bash
/usr/bin/python3.12 << 'EOF'
import sys
sys.path.insert(0, "/home/upandey/OpenHands-dev/openhands/runtime/plugins/agent_skills")
from kernel_tools import (
    get_optimization_memory,
    add_optimization_attempt,
    get_optimization_history,
    should_stop_optimizing,
    get_best_result
)

# Initialize at start
mem = get_optimization_memory(reset=True, target_speedup=1.5, max_attempts=10)

# After each test
add_optimization_attempt(strategy="Added split-K", speedup=1.4, correctness="passed")

# Check if should continue
should_stop, reason = should_stop_optimizing()

# Get final result
best = get_best_result()
EOF
```

## Optimization Workflow

1. **Start**: Initialize memory with `get_optimization_memory(reset=True)`
2. **Evaluate**: Before testing, run `evaluate_kernel_quality()` to catch obvious issues
3. **Test**: Run the actual test script (e.g., `python test_gemm.py`)
4. **Record**: Use `add_optimization_attempt()` to track the result
5. **Reflect**: Use `reflect_on_kernel_result()` to understand the result and get next steps
6. **Check**: Use `should_stop_optimizing()` to see if you should continue
7. **Iterate**: Apply the suggested strategy and repeat from step 2
8. **Report**: When done, use `get_best_result()` to report final outcome

---

## Current Hardware (from rocminfo/rocm-smi)

### System Configuration
```
CPU: AMD EPYC 9575F 64-Core Processor (128 compute units)
GPU: 1x AMD Instinct (gfx950 architecture) allocated for testing
     (System has 8 GPUs, but tests use HIP_VISIBLE_DEVICES=0)
```

### GPU Specifications (single GPU, from rocminfo)
| Specification | Value | Notes |
|---------------|-------|-------|
| Architecture | gfx950 | CDNA3+ |
| Compute Units (CUs) | **256** | Per GPU |
| Wavefront Size | **64 threads** | Fixed, cannot change |
| Max Waves per CU | **32** | = 2048 threads max per CU |
| Workgroup Max Size | **1024** | Max threads per block |
| Cache Line Size | **128 bytes** | Align memory accesses |
| Max GPU Clock | **2400 MHz** | Peak frequency |
| Memory Clock | **1900 MHz** | HBM speed |
| VRAM per GPU | **288 GB** | HBM3 memory |
| Power Cap | **1400W** | Per GPU |
| **GPUs Available** | **1** | Tests run on single GPU (HIP_VISIBLE_DEVICES=0) |

### Key Hardware Constraints

1. **Wavefront = 64 threads**: All thread blocks should be multiples of 64
2. **Max 1024 threads per block**: BLOCK_SIZE cannot exceed 1024
3. **256 CUs per GPU**: Massive parallelism available
4. **288 GB VRAM**: Large models fit in memory
5. **128-byte cache lines**: Align memory accesses to 128 bytes for best performance

### Memory Hierarchy

| Level | Size | Latency | Notes |
|-------|------|---------|-------|
| Registers | 256 VGPRs/wave | 1 cycle | Fastest, limited |
| LDS (Shared) | 64 KB/CU | ~20 cycles | For data reuse within block |
| L1 Cache | Per CU | ~50 cycles | Automatic caching |
| L2 Cache | Shared | ~100 cycles | Large Infinity Cache |
| HBM3 | 288 GB | ~300+ cycles | High bandwidth but high latency |

### Performance Implications

- **Memory-bound kernels**: With 256 CUs, most simple kernels are memory-bound
- **Thread occupancy**: Target high occupancy (many waves per CU) to hide memory latency
- **Vectorization**: Use vector loads/stores when possible (4x or 8x elements)
- **Coalescing**: Adjacent threads must access adjacent memory addresses

---

## AMD GPU Optimization Rules (CRITICAL)

### DO NOT USE (CUDA-only, will fail on ROCm):
- `tl.libdevice` functions
- NVIDIA-specific intrinsics

### Wavefront Size
- AMD wavefront = **64 threads** (not 32 like NVIDIA)
- Design thread blocks accordingly

### Fixed Kernel Parameters (No Autotuning)

**Note: Tests have a strict no-autotuning policy. Block sizes and other parameters are fixed.**


### Common Pitfalls

#### 1. Grid/Program ID Mismatch
```python
# BAD: 1D grid but using 2D program IDs
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)  # 1D
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)  # ERROR: axis 1 doesn't exist!

# GOOD: Either use 2D grid or compute from 1D
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))  # 2D
# OR
pid = tl.program_id(0)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

#### 2. tl.arange requires constexpr
```python
# BAD
offs = tl.arange(0, block_size)  # block_size is not constexpr

# GOOD
offs = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE: tl.constexpr
```

#### 3. tl.dot type requirements
- Inputs must be 2D blocks
- Compatible types: float16, bfloat16
- int32 NOT directly supported as input

#### 4. Config variables in kernel signature
```python
# BAD: BLOCK_SIZE in both Config AND kernel args
@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64})], ...)
@triton.jit
def kernel(x_ptr, BLOCK_SIZE):  # CONFLICT!
    ...

# GOOD: Remove from kernel args if in Config
@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': 64})], ...)
@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):  # Only constexpr
    ...
```

---

## Optimization Strategies

### Memory-Bound Kernels

- Use vectorized loads (`tl.load` with eviction policies)
- Fuse operations to reduce global memory roundtrips
- Consider persistent kernels for small inputs

### Compute-Bound Kernels
- Optimize tile sizes for register pressure
- Use split-K for better parallelism
- Explore num_warps sweet spot

### Algorithmic Improvements
- Naive softmax → Online softmax → Fused softmax
- Standard attention → Flash Attention
- Loop fusion to reduce intermediate storage

---

## Stopping Criteria

The tools will tell you to stop when:
- Target speedup is achieved
- Maximum attempts reached (default: 15)
- Plateau detected (5 attempts without improvement)

**DO NOT stop after first success** — keep trying to improve until the tools say to stop!
