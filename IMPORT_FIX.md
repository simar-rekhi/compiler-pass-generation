# Fix for ImportError: cannot import name 'triton_matmul'

## Problem

There's a naming conflict:

- `triton_kernels.py` (file) - contains functions like `triton_matmul`
- `triton_kernels/` (directory) - contains kernel source files

When Python sees a directory with `__init__.py`, it treats it as a package and ignores the `.py` file with the same name.

## Solution

Updated `triton_kernels/__init__.py` to:

1. Import the parent `triton_kernels.py` file using `importlib`
2. Re-export all public functions from that file
3. This allows imports like `from triton_kernels import triton_matmul` to work

## What Changed

- `triton_kernels/__init__.py` now re-exports:
  - `triton_matmul`
  - `triton_softmax`
  - `get_default_matmul_params`
  - `get_default_softmax_params`
  - `get_tunable_matmul_params`
  - `get_tunable_softmax_params`

## Testing

After pulling this fix, imports should work:

```python
from triton_kernels import triton_matmul  # Should work now!
```

The error should be resolved! 🎉
