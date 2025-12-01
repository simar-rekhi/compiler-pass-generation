"""
Testing framework for correctness and performance validation.
"""
import torch
import numpy as np
import time
from typing import Tuple, Dict, Callable, Any, Optional
from baseline import benchmark_baseline
from triton_kernels import triton_matmul, triton_softmax


class TestFramework:
    """Framework for testing kernel correctness and performance."""
    
    def __init__(self, device: str = "cuda", tolerance: float = 1e-2):
        self.device = device
        self.tolerance = tolerance
    
    def generate_matmul_inputs(
        self, m: int, n: int, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random inputs for matrix multiplication."""
        a = torch.randn(m, k, device=self.device, dtype=torch.float16)
        b = torch.randn(k, n, device=self.device, dtype=torch.float16)
        return a, b
    
    def generate_softmax_inputs(
        self, shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Generate random inputs for softmax."""
        return torch.randn(*shape, device=self.device, dtype=torch.float16)
    
    def test_matmul_correctness(
        self, kernel_func: Callable, params: Dict[str, Any],
        m: int = 1024, n: int = 1024, k: int = 1024,
        num_tests: int = 5
    ) -> Tuple[bool, float]:
        """
        Test matmul kernel correctness against PyTorch baseline.
        Returns (is_correct, max_error).
        """
        all_errors = []
        
        for _ in range(num_tests):
            a, b = self.generate_matmul_inputs(m, n, k)
            
            # Reference result
            ref = torch.matmul(a, b)
            
            # Kernel result
            try:
                result = kernel_func(a, b, **params)
            except Exception as e:
                return False, float('inf'), str(e)
            
            # Compare
            error = torch.max(torch.abs(result - ref)).item()
            all_errors.append(error)
        
        max_error = max(all_errors)
        is_correct = max_error < self.tolerance
        return is_correct, max_error, None
    
    def test_softmax_correctness(
        self, kernel_func: Callable, params: Dict[str, Any],
        shape: Tuple[int, ...] = (1024, 1024),
        num_tests: int = 5
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Test softmax kernel correctness against PyTorch baseline.
        Returns (is_correct, max_error, error_message).
        """
        all_errors = []
        
        for _ in range(num_tests):
            x = self.generate_softmax_inputs(shape)
            
            # Reference result
            ref = torch.nn.functional.softmax(x, dim=-1)
            
            # Kernel result
            try:
                result = kernel_func(x, **params)
            except Exception as e:
                return False, float('inf'), str(e)
            
            # Compare
            error = torch.max(torch.abs(result - ref)).item()
            all_errors.append(error)
        
        max_error = max(all_errors)
        is_correct = max_error < self.tolerance
        return is_correct, max_error, None

    # test_framework.py

def benchmark_kernel(
    self, kernel_func: Callable, *args,
    warmup: int = 10, runs: int = 100, **kwargs
) -> Dict[str, float]:
    """
    Benchmark a kernel function.
    Returns timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        try:
            _ = kernel_func(*args, **kwargs)
        except Exception:
            return {"error": True, "median_ms": float('inf')}
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Actual timing
    times = []
    for _ in range(runs):
        try:
            start = time.perf_counter()
            _ = kernel_func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        except Exception:
            return {"error": True, "median_ms": float('inf')}
    if not times:
        return {"error": True, "median_ms": float('inf')}
    times.sort()
    median_time = times[len(times) // 2]
    mean_time = sum(times) / len(times)
    return {
        "error": False,
        "median_ms": median_time,
        "mean_ms": mean_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "runs": runs,
    }

    
    def full_test_matmul(
        self, params: Dict[str, Any],
        m: int = 1024, n: int = 1024, k: int = 1024
    ) -> Dict[str, Any]:
        """
        Run full test suite for matmul kernel.
        Returns results dictionary.
        """
        # Correctness test
        is_correct, max_error, error_msg = self.test_matmul_correctness(
            triton_matmul, params, m, n, k
        )
        
        if not is_correct:
            return {
                "correct": False,
                "error": error_msg or f"Max error: {max_error}",
                "max_error": max_error,
                "median_ms": float('inf'),
                "speedup": 0.0,
            }
        
        # Performance test
        a, b = self.generate_matmul_inputs(m, n, k)
        perf_stats = self.benchmark_kernel(triton_matmul, a, b, **params)
        
        # Baseline for comparison
        baseline_stats = benchmark_baseline(
            lambda x, y: torch.matmul(x, y), a, b
        )
        
        speedup = baseline_stats["median_ms"] / perf_stats["median_ms"] if perf_stats.get("median_ms", 0) > 0 else 0.0
        
        return {
            "correct": True,
            "error": None,
            "max_error": max_error,
            "median_ms": perf_stats.get("median_ms", float('inf')),
            "baseline_ms": baseline_stats["median_ms"],
            "speedup": speedup,
            "params": params,
        }
    
    def full_test_softmax(
        self, params: Dict[str, Any],
        shape: Tuple[int, ...] = (1024, 1024)
    ) -> Dict[str, Any]:
        """
        Run full test suite for softmax kernel.
        Returns results dictionary.
        """
        # Correctness test
        is_correct, max_error, error_msg = self.test_softmax_correctness(
            triton_softmax, params, shape
        )
        
        if not is_correct:
            return {
                "correct": False,
                "error": error_msg or f"Max error: {max_error}",
                "max_error": max_error,
                "median_ms": float('inf'),
                "speedup": 0.0,
            }
        
        # Performance test
        x = self.generate_softmax_inputs(shape)
        perf_stats = self.benchmark_kernel(triton_softmax, x, **params)
        
        # Baseline for comparison
        baseline_stats = benchmark_baseline(
            lambda x: torch.nn.functional.softmax(x, dim=-1), x
        )
        
        speedup = baseline_stats["median_ms"] / perf_stats["median_ms"] if perf_stats.get("median_ms", 0) > 0 else 0.0
        
        return {
            "correct": True,
            "error": None,
            "max_error": max_error,
            "median_ms": perf_stats.get("median_ms", float('inf')),
            "baseline_ms": baseline_stats["median_ms"],
            "speedup": speedup,
            "params": params,
        }

