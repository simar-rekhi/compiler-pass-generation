import os

def get_kernel_code(kernel_name: str) -> str:
    """
    Get kernel source code from file.
    Reads from raw_kernels/{kernel_name}.py
    """
    # Dynamically find the path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "raw_kernels", f"{kernel_name}.py")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Kernel source file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading kernel source file {file_path}: {e}")
        return ""