"""
Baseline PyTorch implementations for benchmarking.
Includes matmul and softmax operations.
"""
import torch
import time
from typing import Tuple, Dict


def baseline_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Baseline PyTorch matrix multiplication."""
    return torch.matmul(a, b)


def baseline_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Baseline PyTorch softmax."""
    return torch.nn.functional.softmax(x, dim=dim)


def benchmark_baseline(
    func, *args, warmup: int = 10, runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark a function with warmup and multiple runs.
    Returns dictionary with timing statistics.
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Actual timing
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    times.sort()
    median_time = times[len(times) // 2]
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "median_ms": median_time,
        "mean_ms": mean_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "runs": runs,
    }


def generate_matmul_inputs(
    m: int, n: int, k: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random inputs for matrix multiplication."""
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    return a, b


def generate_softmax_inputs(
    shape: Tuple[int, ...], device: str = "cuda"
) -> torch.Tensor:
    """Generate random inputs for softmax."""
    return torch.randn(*shape, device=device, dtype=torch.float16)


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test matmul
    a, b = generate_matmul_inputs(1024, 1024, 1024, device)
    result = baseline_matmul(a, b)
    stats = benchmark_baseline(baseline_matmul, a, b)
    print(f"MatMul benchmark: {stats['median_ms']:.3f} ms (median)")
    
    # Test softmax
    x = generate_softmax_inputs((1024, 1024), device)
    result = baseline_softmax(x)
    stats = benchmark_baseline(baseline_softmax, x)
    print(f"Softmax benchmark: {stats['median_ms']:.3f} ms (median)")

