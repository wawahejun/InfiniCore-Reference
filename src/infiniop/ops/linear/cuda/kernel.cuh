#ifndef __LINEAR_CUDA_H__
#define __LINEAR_CUDA_H__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace op::linear::cuda {

// Helper function to compute offset from multi-dimensional indices and strides
__device__ int compute_offset(const int *indices, const int *strides, int ndim) {
    int offset = 0;
    for (int i = 0; i < ndim; i++) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

template<typename T>
__global__ void linear_kernel(
    T *y,
    const T *x,
    const T *w,
    const T *b,
    const int *x_shape,
    const int *w_shape,
    const int *x_strides,
    const int *w_strides,
    const int *y_strides,
    int batch_size,
    int in_features,
    int out_features,
    int x_ndim,
    size_t total_output_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output_elements) return;
    
    // Calculate batch and output feature indices from linear index
    int batch_idx = idx / out_features;
    int out_idx = idx % out_features;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // x_ndim is now passed as a parameter
    
    // Perform dot product: y[batch, out] = sum(x[batch, in] * w[out, in]) + b[out]
    // Use float accumulation for better precision with half precision types
    float sum = 0.0f;
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
        // Convert linear batch index to multi-dimensional indices for x
        int x_indices[10]; // Support up to 10 dimensions
        int temp_batch_idx = batch_idx;
        for (int i = x_ndim - 2; i >= 0; i--) {
            x_indices[i] = temp_batch_idx % x_shape[i];
            temp_batch_idx /= x_shape[i];
        }
        x_indices[x_ndim - 1] = in_idx;
        
        // Compute x offset with strides
        int x_offset = compute_offset(x_indices, x_strides, x_ndim);
        
        // Compute w offset with strides
        int w_indices[2] = {out_idx, in_idx};
        int w_offset = compute_offset(w_indices, w_strides, 2);
        
        sum += static_cast<float>(x[x_offset]) * static_cast<float>(w[w_offset]);
    }
    if (b != nullptr) {
        sum += static_cast<float>(b[out_idx]);
    }
    
    // Convert linear batch index to multi-dimensional indices for y
    int y_indices[10]; // Support up to 10 dimensions
    int temp_batch_idx = batch_idx;
    for (int i = x_ndim - 2; i >= 0; i--) {
        y_indices[i] = temp_batch_idx % x_shape[i];
        temp_batch_idx /= x_shape[i];
    }
    y_indices[x_ndim - 1] = out_idx;
    
    // Compute y offset with strides
    int y_offset = compute_offset(y_indices, y_strides, x_ndim);
    y[y_offset] = static_cast<T>(sum);
}

} // namespace op::linear::cuda

#endif // __LINEAR_CUDA_H__