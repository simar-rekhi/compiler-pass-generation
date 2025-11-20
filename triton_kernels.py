"""
Triton kernel implementations with tunable parameters.
These kernels can be optimized through compiler pass parameters.
"""
import torch
import triton
import triton.language as tl
from typing import Dict, Any, Optional


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Triton matrix multiplication kernel with tunable parameters.
    
    Parameters:
    - BLOCK_SIZE_M: Block size for M dimension
    - BLOCK_SIZE_N: Block size for N dimension
    - BLOCK_SIZE_K: Block size for K dimension
    - GROUP_SIZE_M: Group size for program ID mapping
    - num_stages: Number of pipeline stages
    - num_warps: Number of warps per block
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask)


def triton_matmul(
    a: torch.Tensor, b: torch.Tensor,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 64,
    BLOCK_SIZE_K: int = 32,
    GROUP_SIZE_M: int = 8,
    num_stages: int = 3,
    num_warps: int = 4,
) -> torch.Tensor:
    """
    Triton matrix multiplication wrapper with tunable parameters.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    
    return c


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, n_elements, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton softmax kernel with tunable parameters.
    
    Parameters:
    - BLOCK_SIZE: Block size for processing elements (should be >= n_cols for efficiency)
    """
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * n_cols
    
    # Load row (assumes BLOCK_SIZE >= n_cols for optimal performance)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    input_ptrs = row_start_ptr + col_offsets
    
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Compute softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Store result
    output_row_start_ptr = output_ptr + row_idx * n_cols
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(
    x: torch.Tensor,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    """
    Triton softmax wrapper with tunable parameters.
    """
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    
    n_elements = n_cols
    grid = (n_rows,)
    
    softmax_kernel[grid](
        y, x, n_elements, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y


def get_default_matmul_params() -> Dict[str, Any]:
    """Get default parameters for matmul kernel."""
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_stages": 3,
        "num_warps": 4,
    }


def get_default_softmax_params() -> Dict[str, Any]:
    """Get default parameters for softmax kernel."""
    return {
        "BLOCK_SIZE": 1024,
    }


def get_tunable_matmul_params() -> Dict[str, list]:
    """Get valid ranges for tunable matmul parameters."""
    return {
        "BLOCK_SIZE_M": [16, 32, 64, 128],
        "BLOCK_SIZE_N": [16, 32, 64, 128],
        "BLOCK_SIZE_K": [16, 32, 64],
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "num_stages": [1, 2, 3, 4, 5],
        "num_warps": [1, 2, 4, 8, 16],
    }


def get_tunable_softmax_params() -> Dict[str, list]:
    """Get valid ranges for tunable softmax parameters."""
    return {
        "BLOCK_SIZE": [256, 512, 1024, 2048, 4096],
    }

