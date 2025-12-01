# Colab Notebook Code for GitHub Repository

This is the updated Colab notebook code that works with the file-based kernel code reading system.

```python
# Always start from /content
%cd /content

# If the repo folder exists, update it; otherwise clone it
import os, subprocess, sys

if os.path.isdir("compiler-pass-generation"):
    %cd compiler-pass-generation
    !git pull
else:
    !git clone https://github.com/simar-rekhi/compiler-pass-generation.git
    %cd compiler-pass-generation

!ls

# Install dependencies
!pip install -q torch triton==3.5.0 numpy openai pydantic pyyaml tqdm

import os

# Set your OpenAI API key
# IMPORTANT: Replace "your-api-key-here" with your actual key or use environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # or use: os.getenv("OPENAI_API_KEY", "your-key")

# Verify kernel source files exist
from pathlib import Path

matmul_path = Path("triton_kernels/matmul.py")
softmax_path = Path("triton_kernels/softmax.py")

print("Checking kernel source files...")
print(f"Matmul kernel file exists: {matmul_path.exists()}")
print(f"Softmax kernel file exists: {softmax_path.exists()}")

if matmul_path.exists():
    code = matmul_path.read_text()
    print(f"Matmul kernel code length: {len(code)} characters")
    print(f"Contains @triton.jit: {'@triton.jit' in code}")

# Verify get_kernel_code works
from llm_optimizer import get_kernel_code

# Test kernel code loading
matmul_code = get_kernel_code("matmul")
softmax_code = get_kernel_code("softmax")

print(f"\nKernel code loaded successfully:")
print(f"Matmul code length: {len(matmul_code)} characters")
print(f"Softmax code length: {len(softmax_code)} characters")

# Make sure CUDA is visible
import torch

print("\nCUDA Information:")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

# Import optimizer (get_kernel_code should now work correctly from files)
from optimizer import KernelOptimizer

# Create optimizer instance
opt = KernelOptimizer(
    kernel_name="matmul",  # or "softmax"
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_iterations=10,
    llm_api_key=os.environ.get("OPENAI_API_KEY"),
)

print(f"\nStarting optimization for {opt.kernel_name} kernel...")
print(f"Device: {opt.device}")
print(f"Max iterations: {opt.max_iterations}")

# Run optimization
try:
    results = opt.optimize()

    if results.get("success"):
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best speedup: {results['best_speedup']:.3f}x")
        print(f"Best runtime: {results['best_runtime_ms']:.3f} ms")
        print(f"Best parameters: {results['best_params']}")
        print(f"Max error: {results.get('max_error', 'N/A')}")
        print(f"Iterations: {results.get('iterations', 'N/A')}")

        # Check archive
        from knowledge_archive import KnowledgeArchive
        archive = KnowledgeArchive()
        stats = archive.get_statistics("matmul")
        print(f"\nOptimization Statistics:")
        print(f"Total attempts: {stats.get('total_attempts', 0)}")
        print(f"Successful attempts: {stats.get('successful_attempts', 0)}")

    else:
        print("Optimization failed:", results.get("error"))

except Exception as e:
    print(f"Error during optimization: {e}")
    import traceback
    traceback.print_exc()
```

## Key Changes from Previous Version

1. **Removed the `get_kernel_code` override** - The function now correctly reads from files, so no override is needed
2. **Added file verification** - Checks that kernel source files exist and can be read
3. **Added kernel code loading test** - Verifies that `get_kernel_code()` works correctly
4. **Improved error handling** - Better error messages and traceback printing
5. **Added archive statistics** - Shows optimization statistics after completion

## Important Notes

1. **API Key**: Replace `"your-api-key-here"` with your actual OpenAI API key, or set it as an environment variable in Colab
2. **Dependencies**: All required packages are installed via pip
3. **File Structure**: The kernel source files (`triton_kernels/matmul.py`, `triton_kernels/softmax.py`) must exist in the repository
4. **CUDA**: The code will automatically use CUDA if available, otherwise falls back to CPU

## Expected Output

The optimization will:

- Load kernel source code from files
- Generate real benchmarking values
- Store results in `archive/` directory
- Generate reports in `reports/` directory
- Display optimization progress and final results
