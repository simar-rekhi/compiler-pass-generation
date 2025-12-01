"""
Example script demonstrating how to use the optimization framework.
"""
import os
import torch
from optimizer import KernelOptimizer
from test_framework import TestFramework
from knowledge_archive import KnowledgeArchive
from triton_kernels import get_default_matmul_params, get_default_softmax_params


def example_simple_test():
    """Example: Simple kernel testing."""
    print("=" * 60)
    print("Example 1: Simple Kernel Testing")
    print("=" * 60)
    
    # Create test framework
    # example.py
    device = "cuda" if torch.cuda.is_available() and os.getenv("CUDA_AVAILABLE") != "false" else "cpu"
    framework = TestFramework(device=device, tolerance=1e-1)  # loosen tolerance if needed

    
    # Test matmul with default parameters
    print("\nTesting matmul with default parameters...")
    default_params = get_default_matmul_params()
    result = framework.full_test_matmul(default_params, m=512, n=512, k=512)
    
    print(f"Correct: {result['correct']}")
    if result['correct']:
        print(f"Speedup: {result['speedup']:.3f}x")
        print(f"Runtime: {result['median_ms']:.3f} ms")
        print(f"Max error: {result['max_error']:.6f}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")
    
    # Test softmax with default parameters
    print("\nTesting softmax with default parameters...")
    default_params = get_default_softmax_params()
    result = framework.full_test_softmax(default_params, shape=(512, 512))
    
    print(f"Correct: {result['correct']}")
    if result['correct']:
        print(f"Speedup: {result['speedup']:.3f}x")
        print(f"Runtime: {result['median_ms']:.3f} ms")
        print(f"Max error: {result['max_error']:.6f}")


def example_optimization():
    """Example: Run optimization."""
    print("\n" + "=" * 60)
    print("Example 2: Running Optimization")
    print("=" * 60)
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nNote: OPENAI_API_KEY not set. Using heuristic fallback.")
        print("Set OPENAI_API_KEY environment variable for LLM-based optimization.")
    
    # Create optimizer
    optimizer = KernelOptimizer(
        kernel_name="matmul",
        device="cuda" if os.getenv("CUDA_AVAILABLE") != "false" else "cpu",
        max_iterations=5,  # Small number for example
        llm_api_key=api_key,
    )
    
    # Run optimization
    print("\nStarting optimization...")
    results = optimizer.optimize()
    
    if results.get("success"):
        print(f"\nOptimization completed!")
        print(f"Best speedup: {results['best_speedup']:.3f}x")
        print(f"Best parameters: {results['best_params']}")
    else:
        print(f"\nOptimization failed: {results.get('error', 'Unknown error')}")


def example_archive_access():
    """Example: Accessing optimization archive."""
    print("\n" + "=" * 60)
    print("Example 3: Accessing Archive")
    print("=" * 60)
    
    archive = KnowledgeArchive()
    
    # Get statistics
    stats = archive.get_statistics("matmul")
    if stats:
        print("\nMatmul optimization statistics:")
        print(f"  Total attempts: {stats.get('total_attempts', 0)}")
        print(f"  Successful attempts: {stats.get('successful_attempts', 0)}")
        print(f"  Best speedup: {stats.get('best_speedup', 0.0):.3f}x")
        print(f"  Best parameters: {stats.get('best_params', {})}")
    else:
        print("\nNo optimization history for matmul yet.")
    
    # Get best kernel
    best = archive.get_best_kernel("matmul")
    if best:
        print("\nBest matmul kernel:")
        print(f"  Speedup: {best['speedup']:.3f}x")
        print(f"  Runtime: {best['runtime_ms']:.3f} ms")
        print(f"  Parameters: {best['parameters']}")
    
    # Get history
    history = archive.get_kernel_history("matmul", limit=5)
    if history:
        print(f"\nLast {len(history)} optimization attempts:")
        for i, entry in enumerate(history, 1):
            print(f"  {i}. Speedup: {entry.get('speedup', 0.0):.3f}x, "
                  f"Correct: {entry.get('correctness', False)}")


if __name__ == "__main__":
    print("Triton Kernel Optimization Framework - Examples")
    print("=" * 60)
    
    # Example 1: Simple testing
    try:
        example_simple_test()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Example 2: Optimization (uncomment to run)
    # try:
    #     example_optimization()
    # except Exception as e:
    #     print(f"Example 2 failed: {e}")
    
    # Example 3: Archive access
    try:
        example_archive_access()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)

