"""
Colab Notebook Code - Ready to copy/paste into Google Colab
This code works with the file-based kernel code reading system.
"""

COLAB_CODE = """
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
# IMPORTANT: Replace with your actual key or use Colab's secrets
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Change this!
# Alternative: Use Colab secrets (recommended):
# from google.colab import userdata
# os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Verify kernel source files exist
from pathlib import Path

print("="*60)
print("Verifying Kernel Source Files")
print("="*60)

matmul_path = Path("triton_kernels/matmul.py")
softmax_path = Path("triton_kernels/softmax.py")

print(f"Matmul kernel file exists: {matmul_path.exists()}")
print(f"Softmax kernel file exists: {softmax_path.exists()}")

if matmul_path.exists():
    code = matmul_path.read_text()
    print(f"Matmul kernel code length: {len(code)} characters")
    print(f"Contains @triton.jit: {'@triton.jit' in code}")
else:
    print("WARNING: Matmul kernel file not found!")

# Verify get_kernel_code works
print("\n" + "="*60)
print("Testing Kernel Code Loading")
print("="*60)

from llm_optimizer import get_kernel_code

# Test kernel code loading
matmul_code = get_kernel_code("matmul")
softmax_code = get_kernel_code("softmax")

print(f"Matmul code loaded: {len(matmul_code) > 0} ({len(matmul_code)} chars)")
print(f"Softmax code loaded: {len(softmax_code) > 0} ({len(softmax_code)} chars)")

if not matmul_code:
    print("ERROR: Failed to load matmul kernel code!")
if not softmax_code:
    print("ERROR: Failed to load softmax kernel code!")

# Make sure CUDA is visible
import torch

print("\n" + "="*60)
print("CUDA Information")
print("="*60)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("WARNING: CUDA not available, will use CPU (slower)")

# Import optimizer (get_kernel_code should now work correctly from files)
print("\n" + "="*60)
print("Initializing Optimizer")
print("="*60)

from optimizer import KernelOptimizer

# Create optimizer instance
opt = KernelOptimizer(
    kernel_name="matmul",  # or "softmax"
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_iterations=10,
    llm_api_key=os.environ.get("OPENAI_API_KEY"),
)

print(f"Kernel: {opt.kernel_name}")
print(f"Device: {opt.device}")
print(f"Max iterations: {opt.max_iterations}")
print(f"API key set: {bool(os.environ.get('OPENAI_API_KEY'))}")

# Run optimization
print("\n" + "="*60)
print("Starting Optimization")
print("="*60)

try:
    results = opt.optimize()
    
    print("\n" + "="*60)
    if results.get("success"):
        print("OPTIMIZATION COMPLETE - SUCCESS")
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
        print(f"  Total attempts: {stats.get('total_attempts', 0)}")
        print(f"  Successful attempts: {stats.get('successful_attempts', 0)}")
        print(f"  Success rate: {stats.get('successful_attempts', 0) / max(stats.get('total_attempts', 1), 1) * 100:.1f}%")
        
        # Show report location
        report_path = Path("reports/matmul_optimization_report.txt")
        if report_path.exists():
            print(f"\nReport saved to: {report_path}")
    else:
        print("OPTIMIZATION COMPLETE - FAILED")
        print("="*60)
        print("Error:", results.get("error", "Unknown error"))
        
except Exception as e:
    print("="*60)
    print("OPTIMIZATION FAILED WITH EXCEPTION")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
"""

if __name__ == "__main__":
    print(COLAB_CODE)

