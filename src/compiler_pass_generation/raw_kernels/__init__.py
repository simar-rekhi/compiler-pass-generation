"""
Triton kernel source code directory.
This directory contains kernel source files for LLM optimization prompts.

This __init__.py re-exports functions from the parent triton_kernels.py module
to allow imports from both the package and the file to work.
"""

# Re-export all functions from parent triton_kernels.py module
# This allows imports like "from triton_kernels import triton_matmul" to work
# even though triton_kernels is now a directory

import sys
import os
import importlib.util

# Get parent directory path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_parent_file = os.path.join(_parent_dir, "triton_kernels.py")

# Import the parent triton_kernels.py file as a module
if os.path.exists(_parent_file):
    spec = importlib.util.spec_from_file_location("_triton_kernels_module", _parent_file)
    if spec and spec.loader:
        _triton_kernels_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_triton_kernels_module)
        
        # Re-export all public functions
        triton_matmul = _triton_kernels_module.triton_matmul
        triton_softmax = _triton_kernels_module.triton_softmax
        get_default_matmul_params = _triton_kernels_module.get_default_matmul_params
        get_default_softmax_params = _triton_kernels_module.get_default_softmax_params
        get_tunable_matmul_params = _triton_kernels_module.get_tunable_matmul_params
        get_tunable_softmax_params = _triton_kernels_module.get_tunable_softmax_params
        
        __all__ = [
            'triton_matmul',
            'triton_softmax',
            'get_default_matmul_params',
            'get_default_softmax_params',
            'get_tunable_matmul_params',
            'get_tunable_softmax_params',
        ]
    else:
        raise ImportError("Could not load triton_kernels.py module")
else:
    raise ImportError(f"triton_kernels.py not found at {_parent_file}")

