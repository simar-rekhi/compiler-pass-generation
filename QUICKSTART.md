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
