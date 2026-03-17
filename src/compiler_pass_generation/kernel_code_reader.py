"""
Isolated kernel code reader - no imports from triton_kernels.
This module only reads files and never imports anything that could trigger JITFunction inspection.
"""


def get_kernel_code(kernel_name: str) -> str:
    """
    Get kernel source code from file.
    Reads from triton_kernels/{kernel_name}.py
    Uses basic file I/O to avoid any inspection issues.
    """
    file_path = f"triton_kernels/{kernel_name}.py"
    
    try:
        # Use basic file operations to avoid any pathlib/inspect issues
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Kernel source file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading kernel source file {file_path}: {e}")
        return ""

