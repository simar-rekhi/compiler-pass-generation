"""
Triton softmax kernel source code.
This file contains the raw kernel source for LLM optimization prompts.
"""
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

