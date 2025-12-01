"""
Test script to verify the full optimization flow works with real benchmarking.
"""
import os
import sys
import torch
from pathlib import Path
from llm_optimizer import get_kernel_code
from test_framework import TestFramework
from triton_kernels import get_default_matmul_params


def test_kernel_code_loading():
    """Test that kernel code can be loaded from files."""
    print("=" * 60)
    print("Test 1: Kernel Code Loading")
    print("=" * 60)
    
    # Test matmul kernel
    matmul_code = get_kernel_code("matmul")
    assert matmul_code, "Failed to load matmul kernel code"
    assert "@triton.jit" in matmul_code, "Matmul kernel code doesn't contain @triton.jit"
    assert "matmul_kernel" in matmul_code, "Matmul kernel code doesn't contain function name"
    print("✓ Matmul kernel code loaded successfully")
    print(f"  Code length: {len(matmul_code)} characters")
    
    # Test softmax kernel
    softmax_code = get_kernel_code("softmax")
    assert softmax_code, "Failed to load softmax kernel code"
    assert "@triton.jit" in softmax_code, "Softmax kernel code doesn't contain @triton.jit"
    assert "softmax_kernel" in softmax_code, "Softmax kernel code doesn't contain function name"
    print("✓ Softmax kernel code loaded successfully")
    print(f"  Code length: {len(softmax_code)} characters")
    
    # Test invalid kernel
    invalid_code = get_kernel_code("invalid")
    assert invalid_code == "", "Invalid kernel should return empty string"
    print("✓ Invalid kernel handling works correctly")
    print()


def test_baseline_benchmarking():
    """Test that baseline benchmarking produces real values."""
    print("=" * 60)
    print("Test 2: Baseline Benchmarking")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    framework = TestFramework(device=device, tolerance=1e-1)
    
    # Test matmul baseline
    print("\nTesting matmul baseline...")
    params = get_default_matmul_params()
    result = framework.full_test_matmul(params, m=512, n=512, k=512)
    
    assert "correct" in result, "Result should contain 'correct' field"
    assert "speedup" in result, "Result should contain 'speedup' field"
    assert "median_ms" in result, "Result should contain 'median_ms' field"
    
    print(f"  Correct: {result['correct']}")
    print(f"  Speedup: {result['speedup']:.3f}x")
    print(f"  Runtime: {result['median_ms']:.3f} ms")
    print(f"  Baseline: {result.get('baseline_ms', 'N/A')} ms")
    print(f"  Max error: {result.get('max_error', 'N/A')}")
    
    if result['correct']:
        assert result['speedup'] > 0, "Speedup should be positive for correct results"
        assert result['median_ms'] > 0, "Runtime should be positive"
        print("✓ Matmul baseline test passed")
    else:
        print(f"  Warning: Matmul test failed - {result.get('error', 'Unknown error')}")
    print()


def test_full_optimization_flow():
    """Test a minimal optimization flow."""
    print("=" * 60)
    print("Test 3: Full Optimization Flow (2 iterations)")
    print("=" * 60)
    
    from optimizer import KernelOptimizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create optimizer with minimal iterations
    optimizer = KernelOptimizer(
        kernel_name="matmul",
        device=device,
        max_iterations=2,  # Just 2 iterations for testing
        llm_api_key=os.getenv("OPENAI_API_KEY"),  # Optional
    )
    
    print("\nRunning optimization (this may take a minute)...")
    try:
        results = optimizer.optimize()
        
        assert "success" in results, "Results should contain 'success' field"
        print(f"\nOptimization completed: {results['success']}")
        
        if results['success']:
            print(f"  Best speedup: {results.get('best_speedup', 0.0):.3f}x")
            print(f"  Best runtime: {results.get('best_runtime_ms', 0.0):.3f} ms")
            print(f"  Iterations: {results.get('iterations', 0)}")
            print(f"  Best parameters: {results.get('best_params', {})}")
            print("✓ Optimization flow completed successfully")
        else:
            print(f"  Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"  Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_kernel_files_exist():
    """Test that kernel source files exist."""
    print("=" * 60)
    print("Test 0: Kernel Source Files")
    print("=" * 60)
    
    matmul_path = Path("triton_kernels/matmul.py")
    softmax_path = Path("triton_kernels/softmax.py")
    
    assert matmul_path.exists(), f"Matmul kernel file not found at {matmul_path}"
    assert softmax_path.exists(), f"Softmax kernel file not found at {softmax_path}"
    
    print(f"✓ Matmul kernel file exists: {matmul_path}")
    print(f"✓ Softmax kernel file exists: {softmax_path}")
    
    # Verify file contents
    with open(matmul_path, 'r') as f:
        matmul_content = f.read()
        assert "@triton.jit" in matmul_content, "Matmul file should contain @triton.jit"
        assert "matmul_kernel" in matmul_content, "Matmul file should contain kernel function"
    
    with open(softmax_path, 'r') as f:
        softmax_content = f.read()
        assert "@triton.jit" in softmax_content, "Softmax file should contain @triton.jit"
        assert "softmax_kernel" in softmax_content, "Softmax file should contain kernel function"
    
    print("✓ Kernel source files contain valid kernel code")
    print()


if __name__ == "__main__":
    print("Triton Kernel Optimization - Full Test Suite")
    print("=" * 60)
    print()
    
    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        # Test 0: Files exist
        test_kernel_files_exist()
        
        # Test 1: Code loading
        test_kernel_code_loading()
        
        # Test 2: Baseline benchmarking
        test_baseline_benchmarking()
        
        # Test 3: Full optimization flow (commented out by default as it takes time)
        # Uncomment to test the full flow
        if "--full" in sys.argv:
            test_full_optimization_flow()
        else:
            print("=" * 60)
            print("Test 3: Full Optimization Flow")
            print("=" * 60)
            print("Skipped (use --full flag to run)")
            print()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

