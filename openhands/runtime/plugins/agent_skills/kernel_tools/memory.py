"""Optimization memory for tracking kernel optimization attempts.

This module provides a simple in-memory store for tracking optimization
attempts, results, and determining when to stop optimizing.
"""

from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class OptimizationAttempt:
    """Record of a single optimization attempt."""
    attempt_number: int
    strategy: str
    speedup: float
    correctness: str  # "passed", "failed", "error"
    eval_score: float = 0.0
    reflection_summary: str = ""
    code_snippet: str = ""  # First 500 chars of kernel


@dataclass
class OptimizationMemory:
    """
    Tracks all optimization attempts in a session.
    
    This class maintains state across the optimization process, tracking
    what has been tried, what worked, and when to stop.
    
    Attributes:
        attempts: List of all optimization attempts
        best_speedup: Best speedup achieved so far
        best_code: Code that achieved best speedup
        baseline_speedup: Starting speedup (usually 1.0)
        target_speedup: Target speedup to achieve (stop condition)
        max_attempts: Maximum number of attempts before stopping
        max_attempts_without_improvement: Stop if this many attempts without improvement
    """
    attempts: list[OptimizationAttempt] = field(default_factory=list)
    best_speedup: float = 0.0
    best_code: str = ""
    best_strategy: str = ""
    baseline_speedup: float = 1.0
    target_speedup: float = 2.0
    max_attempts: int = 15
    max_attempts_without_improvement: int = 5
    _attempts_since_improvement: int = 0
    
    def add_attempt(
        self,
        strategy: str,
        speedup: float,
        correctness: str = "unknown",
        eval_score: float = 0.0,
        reflection_summary: str = "",
        code: str = ""
    ) -> dict[str, Any]:
        """
        Record a new optimization attempt.
        
        Args:
            strategy: Description of the optimization strategy used
            speedup: Measured speedup (e.g., 1.5 = 50% faster)
            correctness: "passed", "failed", or "error"
            eval_score: Score from evaluate_kernel_quality
            reflection_summary: Summary from reflect_on_kernel_result
            code: The kernel code (will be truncated)
        
        Returns:
            dict with attempt status and whether it's a new best
        """
        attempt = OptimizationAttempt(
            attempt_number=len(self.attempts) + 1,
            strategy=strategy,
            speedup=speedup,
            correctness=correctness,
            eval_score=eval_score,
            reflection_summary=reflection_summary,
            code_snippet=code[:500] if code else ""
        )
        self.attempts.append(attempt)
        
        is_improvement = False
        if correctness == "passed" and speedup > self.best_speedup:
            self.best_speedup = speedup
            self.best_code = code
            self.best_strategy = strategy
            self._attempts_since_improvement = 0
            is_improvement = True
        else:
            self._attempts_since_improvement += 1
        
        return {
            "attempt_number": attempt.attempt_number,
            "is_improvement": is_improvement,
            "best_speedup": self.best_speedup,
            "attempts_since_improvement": self._attempts_since_improvement
        }
    
    def should_stop(self) -> tuple[bool, str]:
        """
        Check if optimization should stop.
        
        Returns:
            (should_stop, reason) tuple
        """
        if self.best_speedup >= self.target_speedup:
            return True, f"Target speedup achieved: {self.best_speedup:.2f}x >= {self.target_speedup}x"
        
        if len(self.attempts) >= self.max_attempts:
            return True, f"Maximum attempts reached: {len(self.attempts)} >= {self.max_attempts}"
        
        if self._attempts_since_improvement >= self.max_attempts_without_improvement:
            return True, f"Plateau reached: {self._attempts_since_improvement} attempts without improvement"
        
        return False, ""
    
    def get_history_summary(self, last_n: int = 5) -> str:
        """Get a summary of recent attempts for the reflector."""
        if not self.attempts:
            return "No previous attempts"
        
        recent = self.attempts[-last_n:]
        lines = []
        for a in recent:
            status = "✓" if a.correctness == "passed" else "✗"
            lines.append(
                f"Attempt {a.attempt_number} [{status}]: {a.strategy} → {a.speedup:.2f}x"
            )
        
        return "\n".join(lines)
    
    def get_tried_strategies(self) -> str:
        """Get list of strategies already tried."""
        if not self.attempts:
            return "None"
        
        strategies = [a.strategy for a in self.attempts]
        return ", ".join(strategies)
    
    def get_best_result(self) -> dict[str, Any]:
        """Get the best result achieved so far."""
        return {
            "speedup": self.best_speedup,
            "strategy": self.best_strategy,
            "code": self.best_code,
            "total_attempts": len(self.attempts),
            "improvement_vs_baseline": f"{((self.best_speedup / self.baseline_speedup) - 1) * 100:.1f}%"
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize memory to dict for persistence."""
        return {
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "strategy": a.strategy,
                    "speedup": a.speedup,
                    "correctness": a.correctness,
                    "eval_score": a.eval_score,
                    "reflection_summary": a.reflection_summary
                }
                for a in self.attempts
            ],
            "best_speedup": self.best_speedup,
            "best_strategy": self.best_strategy,
            "attempts_since_improvement": self._attempts_since_improvement
        }
    
    def __str__(self) -> str:
        should_stop, reason = self.should_stop()
        status = "STOPPED" if should_stop else "RUNNING"
        return (
            f"OptimizationMemory({status}): "
            f"{len(self.attempts)} attempts, "
            f"best={self.best_speedup:.2f}x, "
            f"since_improvement={self._attempts_since_improvement}"
        )


# Global memory instance (persists across IPython cells in same session)
_global_memory: OptimizationMemory | None = None


def get_optimization_memory(
    reset: bool = False,
    target_speedup: float = 2.0,
    max_attempts: int = 15,
    max_attempts_without_improvement: int = 5
) -> OptimizationMemory:
    """
    Get or create the global optimization memory.
    
    Args:
        reset: If True, create a fresh memory instance
        target_speedup: Target speedup to achieve
        max_attempts: Maximum attempts before stopping
        max_attempts_without_improvement: Stop if this many without improvement
    
    Returns:
        The global OptimizationMemory instance
    
    Example:
        >>> mem = get_optimization_memory(reset=True, target_speedup=1.5)
        >>> mem.add_attempt("tiled GEMM", speedup=1.2, correctness="passed")
    """
    global _global_memory
    
    if reset or _global_memory is None:
        _global_memory = OptimizationMemory(
            target_speedup=target_speedup,
            max_attempts=max_attempts,
            max_attempts_without_improvement=max_attempts_without_improvement
        )
    
    return _global_memory


def add_optimization_attempt(
    strategy: str,
    speedup: float,
    correctness: str = "unknown",
    eval_score: float = 0.0,
    reflection_summary: str = "",
    code: str = ""
) -> dict[str, Any]:
    """
    Add an optimization attempt to the global memory.
    
    Convenience function that uses the global memory instance.
    
    Args:
        strategy: Description of optimization strategy
        speedup: Measured speedup
        correctness: "passed", "failed", or "error"
        eval_score: Score from evaluator
        reflection_summary: Summary from reflector
        code: Kernel code
    
    Returns:
        dict with attempt status
    
    Example:
        >>> result = add_optimization_attempt(
        ...     strategy="Added split-K",
        ...     speedup=1.4,
        ...     correctness="passed"
        ... )
        >>> print(f"Attempt {result['attempt_number']}, best so far: {result['best_speedup']}x")
    """
    mem = get_optimization_memory()
    result = mem.add_attempt(
        strategy=strategy,
        speedup=speedup,
        correctness=correctness,
        eval_score=eval_score,
        reflection_summary=reflection_summary,
        code=code
    )
    
    # Print status
    print(f"\n📊 Attempt {result['attempt_number']}: {strategy}")
    print(f"   Speedup: {speedup:.2f}x | Correctness: {correctness}")
    if result['is_improvement']:
        print(f"   🎉 NEW BEST! Previous best: {result['best_speedup']:.2f}x")
    else:
        print(f"   Best so far: {result['best_speedup']:.2f}x ({result['attempts_since_improvement']} since improvement)")
    
    return result


def get_optimization_history() -> str:
    """
    Get summary of optimization history for the reflector.
    
    Returns:
        String summary of recent attempts
    
    Example:
        >>> history = get_optimization_history()
        >>> reflection = reflect_on_kernel_result(..., history=history)
    """
    mem = get_optimization_memory()
    return mem.get_history_summary()


def should_stop_optimizing() -> tuple[bool, str]:
    """
    Check if optimization should stop.
    
    Returns:
        (should_stop, reason) tuple
    
    Example:
        >>> should_stop, reason = should_stop_optimizing()
        >>> if should_stop:
        ...     print(f"Stopping: {reason}")
        ...     best = get_best_result()
    """
    mem = get_optimization_memory()
    should_stop, reason = mem.should_stop()
    
    if should_stop:
        print(f"\n🛑 Optimization Complete: {reason}")
    else:
        remaining = mem.max_attempts - len(mem.attempts)
        print(f"\n▶️ Continue optimizing ({remaining} attempts remaining)")
    
    return should_stop, reason


def get_best_result() -> dict[str, Any]:
    """
    Get the best optimization result achieved.
    
    Returns:
        dict with best speedup, strategy, and code
    
    Example:
        >>> best = get_best_result()
        >>> print(f"Best speedup: {best['speedup']}x using {best['strategy']}")
    """
    mem = get_optimization_memory()
    result = mem.get_best_result()
    
    print(f"\n🏆 Best Result:")
    print(f"   Speedup: {result['speedup']:.2f}x")
    print(f"   Strategy: {result['strategy']}")
    print(f"   Improvement: {result['improvement_vs_baseline']}")
    print(f"   Total attempts: {result['total_attempts']}")
    
    return result

