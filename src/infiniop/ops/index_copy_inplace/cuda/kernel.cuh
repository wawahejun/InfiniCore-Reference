#ifndef __INDEX_COPY_INPLACE_CUDA_H__
#define __INDEX_COPY_INPLACE_CUDA_H__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::index_copy_inplace::cuda {

// Note: This GPU kernel implements index_copy_ with undefined behavior for duplicate indices
// This matches PyTorch's GPU behavior where duplicate indices result in nondeterministic outcomes
// The parallel execution order determines which value "wins" for duplicate indices
// This is consistent with PyTorch documentation stating behavior is undefined for duplicate indices
template<typename T, typename IndexT>
__global__ void index_copy_inplace_kernel(
    T *target_data,
    const T *source_data,
    const IndexT *index_data,
    const int *target_shape,
    const int *source_shape,
    const int *index_shape,
    const int *target_strides,
    const int *source_strides,
    const int *index_strides,
    int dim,
    int ndim,
    size_t total_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    // Calculate source coordinates from linear index
    int in_coords[8]; // Support up to 8D tensors
    int temp = idx;
    for (int d = ndim - 1; d >= 0; --d) {
        in_coords[d] = temp % source_shape[d];
        temp /= source_shape[d];
    }
    
    // Get the index value for the current position in the specified dimension
    int src_dim_idx = in_coords[dim];
    if (src_dim_idx >= index_shape[0]) {
        return; // Skip if source dim index is out of bounds for index tensor
    }
    
    // Get the target index from the index tensor
    IndexT target_idx = index_data[src_dim_idx];
    
    // Check bounds for target index
    if (target_idx < 0 || target_idx >= target_shape[dim]) {
        return; // Skip out of bounds target indices
    }
    
    // Calculate target coordinates (copy source coords and modify dim)
    int out_coords[8];
    for (int d = 0; d < ndim; ++d) {
        out_coords[d] = in_coords[d];
    }
    out_coords[dim] = static_cast<int>(target_idx);
    
    // Calculate source offset
    size_t in_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        in_offset += in_coords[d] * source_strides[d];
    }
    
    // Calculate target offset
    size_t out_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        out_offset += out_coords[d] * target_strides[d];
    }
    
    // Copy the value from source to target at the indexed position
    target_data[out_offset] = source_data[in_offset];
}

} // namespace op::index_copy_inplace::cuda

#endif // __INDEX_COPY_INPLACE_CUDA_H__