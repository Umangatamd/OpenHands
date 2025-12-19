"""Kernel optimization tools for Triton/AMD GPU kernel development.

This module provides LLM-based evaluation and reflection tools for kernel optimization.

Can be used standalone or within OpenHands runtime.
"""

# Use relative imports to avoid circular dependencies with OpenHands core
from .evaluator import evaluate_kernel_quality
from .reflector import reflect_on_kernel_result
from .memory import (
    OptimizationMemory,
    get_optimization_memory,
    add_optimization_attempt,
    get_optimization_history,
    should_stop_optimizing,
    get_best_result,
)

__all__ = [
    'evaluate_kernel_quality',
    'reflect_on_kernel_result',
    'OptimizationMemory',
    'get_optimization_memory',
    'add_optimization_attempt',
    'get_optimization_history',
    'should_stop_optimizing',
    'get_best_result',
]

