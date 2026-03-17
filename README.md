# Compiler Pass Generation - Triton Kernel Optimizer

An automated optimization framework that uses LLMs to suggest compiler pass parameters for Triton GPU kernels. The system implements a closed-loop refinement process that learns from previous attempts to optimize kernel performance.

## Features

### 1. Baseline Implementation

- PyTorch baseline implementations for matmul and softmax
- Automated benchmarking and performance measurement
- Correctness validation against reference implementations

### 2. Triton Kernels with Tunable Parameters

- Matmul kernel with configurable:
  - Block sizes (M, N, K dimensions)
  - Group size for program ID mapping
  - Pipeline stages
  - Number of warps
- Softmax kernel with configurable:
  - Block size

### 3. Testing Framework

- Automatic input generation
- Correctness testing with numerical validation
- Performance benchmarking with statistical analysis
- Stability testing across different input sizes

### 4. Knowledge Archive

- Stores all kernel versions and optimization attempts
- Tracks parameters, speedup, correctness, and metadata
- Provides history and statistics for analysis

### 5. LLM Integration

- Structured prompts including:
  - Kernel code
  - Hardware/device information
  - Performance goals and constraints
  - Optimization history
- Heuristic fallback when LLM unavailable

### 6. Closed-Loop Optimization

- Iterative refinement with feedback
- Parameter suggestion → testing → scoring → refinement
- Early stopping on good speedup
- Maximum iteration budget

### 7. Reporting System

- Comprehensive optimization reports
- Parameter impact analysis
- Stability analysis across input sizes
- Top-performing configurations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Optimize matmul kernel (default)
python optimizer.py --kernel matmul

# Optimize softmax kernel
python optimizer.py --kernel softmax

# Specify device and iterations
python optimizer.py --kernel matmul --device cuda --max-iterations 20

# With OpenAI API key
python optimizer.py --kernel matmul --api-key YOUR_API_KEY
# Or set environment variable: export OPENAI_API_KEY=your_key
```

### Programmatic Usage

```python
from optimizer import KernelOptimizer

# Create optimizer
optimizer = KernelOptimizer(
    kernel_name="matmul",
    device="cuda",
    max_iterations=20,
    llm_api_key="your-api-key"  # Optional
)

# Run optimization
results = optimizer.optimize()

print(f"Best speedup: {results['best_speedup']:.3f}x")
print(f"Best parameters: {results['best_params']}")
```

### Testing Individual Kernels

```python
from test_framework import TestFramework
from triton_kernels import triton_matmul

# Create test framework
framework = TestFramework(device="cuda")

# Test specific parameters
params = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_stages": 4,
    "num_warps": 8,
}

result = framework.full_test_matmul(params, m=1024, n=1024, k=1024)
print(f"Speedup: {result['speedup']:.3f}x")
print(f"Correct: {result['correct']}")
```

### Accessing Archive

```python
from knowledge_archive import KnowledgeArchive

archive = KnowledgeArchive()

# Get best kernel
best = archive.get_best_kernel("matmul")
print(f"Best speedup: {best['speedup']:.3f}x")

# Get optimization history
history = archive.get_kernel_history("matmul", limit=10)

# Get statistics
stats = archive.get_statistics("matmul")
print(f"Total attempts: {stats['total_attempts']}")
```

## Project Structure

```
compiler-pass-generation/
├── baseline.py              # PyTorch baseline implementations
├── triton_kernels.py        # Triton kernels with tunable parameters
├── test_framework.py        # Testing and benchmarking framework
├── knowledge_archive.py     # Storage for optimization results
├── llm_optimizer.py         # LLM integration for parameter suggestions
├── optimizer.py             # Main optimization loop
├── reporter.py              # Report generation system
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## How It Works

1. **Initialization**: Sets up baseline PyTorch functions and Triton kernels with default parameters
2. **Baseline Benchmarking**: Measures baseline performance for comparison
3. **Optimization Loop** (repeats up to max_iterations):
   - LLM suggests new parameter values based on:
     - Current kernel code
     - Hardware characteristics
     - Previous optimization attempts
     - Performance goals
   - Kernel is compiled and tested with new parameters
   - Results are scored (correctness + speedup)
   - Best results are stored in archive
   - Feedback is provided to LLM for next iteration
4. **Analysis**:
   - Parameter impact analysis
   - Stability testing across input sizes
   - Comprehensive reporting

## Tunable Parameters

### MatMul Kernel

- `BLOCK_SIZE_M`: Block size for M dimension (16, 32, 64, 128)
- `BLOCK_SIZE_N`: Block size for N dimension (16, 32, 64, 128)
- `BLOCK_SIZE_K`: Block size for K dimension (16, 32, 64)
- `GROUP_SIZE_M`: Group size for program ID mapping (1, 2, 4, 8)
- `num_stages`: Number of pipeline stages (1-5)
- `num_warps`: Number of warps per block (1, 2, 4, 8, 16)

### Softmax Kernel

- `BLOCK_SIZE`: Block size for processing (256, 512, 1024, 2048, 4096)

## Output

The optimizer generates:

- Console output with optimization progress
- Archive files in `archive/` directory:
  - `kernels.json`: All kernel versions and metadata
  - `metadata.json`: Optimization statistics
- Reports in `reports/` directory:
  - `{kernel}_optimization_report.txt`: Comprehensive optimization report

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Triton 2.0+
- CUDA-capable GPU (for optimal performance)
- OpenAI API key (optional, for LLM suggestions)

## Notes

- Without an OpenAI API key, the system falls back to heuristic-based parameter suggestions
- The framework is designed to work with CUDA, but CPU fallback is available
- Optimization results are stored persistently for later analysis
- The system learns from previous attempts to improve suggestions over time

------------------------------------------------------------------------------------------------------
# Quick Start Guide

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Optional: Set OpenAI API key** (for LLM-based optimization):

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Quick Examples

### 1. Test a Kernel

```bash
python example.py
```

This will:

- Test matmul and softmax kernels with default parameters
- Show correctness and performance results
- Display archive statistics

### 2. Run Optimization

```bash
# Optimize matmul kernel (5 iterations, quick test)
python optimizer.py --kernel matmul --max-iterations 5

# Optimize softmax kernel
python optimizer.py --kernel softmax --max-iterations 10

# Full optimization run
python optimizer.py --kernel matmul --max-iterations 20
```

### 3. Programmatic Usage

```python
from optimizer import KernelOptimizer

# Create optimizer
optimizer = KernelOptimizer(
    kernel_name="matmul",
    device="cuda",
    max_iterations=10,
)

# Run optimization
results = optimizer.optimize()
print(f"Best speedup: {results['best_speedup']:.3f}x")
```

## What Gets Generated

After running optimization, you'll find:

- **`archive/`** - Optimization history and metadata

  - `kernels.json` - All kernel versions and results
  - `metadata.json` - Optimization statistics

- **`reports/`** - Detailed optimization reports
  - `{kernel}_optimization_report.txt` - Comprehensive analysis

## Understanding Output

The optimizer prints:

- ✓ Correctness tests (pass/fail)
- Speedup over baseline (e.g., "2.3x")
- Runtime in milliseconds
- Parameter values being tested
- Best result found so far

Reports include:

- Summary of best results
- Parameter impact analysis
- Stability across input sizes
- Top-performing configurations

## Troubleshooting

### CUDA not available

The framework works on CPU but will be slower. Set device to "cpu":

```python
optimizer = KernelOptimizer(kernel_name="matmul", device="cpu")
```

### LLM not available

The system automatically falls back to heuristic-based parameter suggestions if no API key is provided.

### Import errors

Make sure all dependencies are installed:

```bash
pip install torch triton openai numpy pydantic pyyaml tqdm
```

### Out of memory

Reduce input sizes in test_framework.py or use smaller block sizes.

## Next Steps

1. Run optimization for your target kernel
2. Review the generated report
3. Check archive for optimization history
4. Experiment with different input sizes
5. Analyze parameter impact on performance

For more details, see the main README.md file.
