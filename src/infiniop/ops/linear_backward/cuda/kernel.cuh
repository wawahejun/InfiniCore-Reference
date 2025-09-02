#ifndef __LINEAR_BACKWARD_KERNEL_CUH__
#define __LINEAR_BACKWARD_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../../utils/custom_types.h"
#include "../../../tensor.h"
#include <cuda_bf16.h>

namespace op::linear_backward::cuda {

// Linear backward CUDA kernel
// Computes gradients for linear layer:
// grad_x = grad_y * w^T
// grad_w = grad_y^T * x  
// grad_b = sum(grad_y, dim=0)
//
// Parameters:
// - grad_x: gradient w.r.t. input [batch_size, in_features]
// - grad_w: gradient w.r.t. weight [out_features, in_features]
// - grad_b: gradient w.r.t. bias [out_features]
// - grad_y: gradient w.r.t. output [batch_size, out_features]
// - x: input tensor [batch_size, in_features]
// - w: weight tensor [out_features, in_features]
// - x_shape: shape of x tensor
// - w_shape: shape of w tensor
// - x_strides: strides of x tensor
// - w_strides: strides of w tensor
// - grad_y_strides: strides of grad_y tensor
// - grad_x_strides: strides of grad_x tensor (if not null)
// - grad_w_strides: strides of grad_w tensor (if not null)
// - grad_b_strides: strides of grad_b tensor (if not null)
// Helper function to compute offset from multi-dimensional indices and strides
__device__ int compute_offset(const int *indices, const int *strides, int ndim) {
    int offset = 0;
    for (int i = 0; i < ndim; i++) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

__global__ void linear_backward_kernel(
    void* grad_x,
    void* grad_w, 
    void* grad_b,
    const void* grad_y,
    const void* x,
    const void* w,
    const int* grad_y_shape,
    const int* x_shape,
    const int* w_shape,
    const int* grad_y_strides,
    const int* x_strides,
    const int* w_strides,
    const int* grad_x_strides,
    const int* grad_w_strides,
    const int* grad_b_strides,
    int batch_size,
    int in_features,
    int out_features,
    int x_ndim,
    infiniDtype_t dtype) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (dtype == INFINI_DTYPE_F16) {
        half* grad_x_ptr = (half*)grad_x;
        half* grad_w_ptr = (half*)grad_w;
        half* grad_b_ptr = (half*)grad_b;
        const half* grad_y_ptr = (const half*)grad_y;
        const half* x_ptr = (const half*)x;
        const half* w_ptr = (const half*)w;
        
        // Compute grad_x = grad_y @ w^T (grad_y * w)
        if (grad_x_ptr && idx < batch_size * in_features) {
            // Convert linear index to multi-dimensional indices for grad_x
            int temp_idx = idx;
            int x_indices[4]; // Support up to 4D tensors
            for (int i = x_ndim - 1; i >= 0; i--) {
                x_indices[i] = temp_idx % x_shape[i];
                temp_idx /= x_shape[i];
            }
            
            // Get in_feature index
            int in = x_indices[x_ndim - 1];
            
            float sum = 0.0f;
            for (int out = 0; out < out_features; out++) {
                // Compute grad_y offset
                int grad_y_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    grad_y_indices[i] = x_indices[i];
                }
                grad_y_indices[x_ndim - 1] = out;
                int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                
                // Compute w offset: w[out][in]
                int w_offset = out * w_strides[0] + in * w_strides[1];
                
                sum += __half2float(grad_y_ptr[grad_y_offset]) * 
                       __half2float(w_ptr[w_offset]);
            }
            
            // Compute grad_x offset
            int grad_x_offset = compute_offset(x_indices, grad_x_strides, x_ndim);
            grad_x_ptr[grad_x_offset] = __float2half(sum);
        }
        
        // Compute grad_w = grad_y^T @ x
        if (grad_w_ptr && idx < out_features * in_features) {
            int out = idx / in_features;
            int in = idx % in_features;
            
            float sum = 0.0f;
            // Iterate over all batch elements
            int total_batch_elements = 1;
            for (int i = 0; i < x_ndim - 1; i++) {
                total_batch_elements *= x_shape[i];
            }
            
            for (int batch_idx = 0; batch_idx < total_batch_elements; batch_idx++) {
                // Convert batch_idx to multi-dimensional indices
                int temp_idx = batch_idx;
                int batch_indices[4];
                for (int i = x_ndim - 2; i >= 0; i--) {
                    batch_indices[i] = temp_idx % x_shape[i];
                    temp_idx /= x_shape[i];
                }
                
                // Compute grad_y offset
                int grad_y_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    grad_y_indices[i] = batch_indices[i];
                }
                grad_y_indices[x_ndim - 1] = out;
                int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                
                // Compute x offset
                int x_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    x_indices[i] = batch_indices[i];
                }
                x_indices[x_ndim - 1] = in;
                int x_offset = compute_offset(x_indices, x_strides, x_ndim);
                
                sum += __half2float(grad_y_ptr[grad_y_offset]) * 
                       __half2float(x_ptr[x_offset]);
            }
            
            // Compute grad_w offset
            int grad_w_offset = out * grad_w_strides[0] + in * grad_w_strides[1];
            grad_w_ptr[grad_w_offset] = __float2half(sum);
        }
        
        // Compute grad_b = sum(grad_y, dim=0)
        if (grad_b_ptr && idx < out_features) {
            float sum = 0.0f;
            // Iterate over all batch elements
            int total_batch_elements = 1;
            for (int i = 0; i < x_ndim - 1; i++) {
                total_batch_elements *= x_shape[i];
            }
            
            for (int batch_idx = 0; batch_idx < total_batch_elements; batch_idx++) {
                // Convert batch_idx to multi-dimensional indices
                int temp_idx = batch_idx;
                int batch_indices[4];
                for (int i = x_ndim - 2; i >= 0; i--) {
                    batch_indices[i] = temp_idx % x_shape[i];
                    temp_idx /= x_shape[i];
                }
                
                // Compute grad_y offset
                int grad_y_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    grad_y_indices[i] = batch_indices[i];
                }
                grad_y_indices[x_ndim - 1] = idx;
                int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                
                sum += __half2float(grad_y_ptr[grad_y_offset]);
            }
            
            // Compute grad_b offset
            int grad_b_offset = idx * grad_b_strides[0];
            grad_b_ptr[grad_b_offset] = __float2half(sum);
        }
    }
    else if (dtype == INFINI_DTYPE_F32) {
        float* grad_x_ptr = (float*)grad_x;
        float* grad_w_ptr = (float*)grad_w;
        float* grad_b_ptr = (float*)grad_b;
        const float* grad_y_ptr = (const float*)grad_y;
        const float* x_ptr = (const float*)x;
        const float* w_ptr = (const float*)w;
        
        // Compute grad_x = grad_y @ w^T (grad_y * w)
        if (grad_x_ptr && idx < batch_size * in_features) {
            // Convert linear index to multi-dimensional indices for grad_x
            int temp_idx = idx;
            int x_indices[4]; // Support up to 4D tensors
            for (int i = x_ndim - 1; i >= 0; i--) {
                x_indices[i] = temp_idx % x_shape[i];
                temp_idx /= x_shape[i];
            }
            
            // Get in_feature index
            int in = x_indices[x_ndim - 1];
            
            float sum = 0.0f;
            for (int out = 0; out < out_features; out++) {
                // Compute grad_y offset
                int grad_y_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    grad_y_indices[i] = x_indices[i];
                }
                grad_y_indices[x_ndim - 1] = out;
                int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                
                // Compute w offset: w[out][in]
                int w_offset = out * w_strides[0] + in * w_strides[1];
                
                sum += grad_y_ptr[grad_y_offset] * w_ptr[w_offset];
            }
            
            // Compute grad_x offset
            int grad_x_offset = compute_offset(x_indices, grad_x_strides, x_ndim);
            grad_x_ptr[grad_x_offset] = sum;
        }
        
        // Compute grad_w = grad_y^T @ x
        if (grad_w_ptr && idx < out_features * in_features) {
            int out = idx / in_features;
            int in = idx % in_features;
            
            float sum = 0.0f;
            // Iterate over all batch elements
            int total_batch_elements = 1;
            for (int i = 0; i < x_ndim - 1; i++) {
                total_batch_elements *= x_shape[i];
            }
            
            for (int batch_idx = 0; batch_idx < total_batch_elements; batch_idx++) {
                // Convert batch_idx to multi-dimensional indices
                int temp_idx = batch_idx;
                int batch_indices[4];
                for (int i = x_ndim - 2; i >= 0; i--) {
                    batch_indices[i] = temp_idx % x_shape[i];
                    temp_idx /= x_shape[i];
                }
                
                // Compute grad_y offset
                int grad_y_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    grad_y_indices[i] = batch_indices[i];
                }
                grad_y_indices[x_ndim - 1] = out;
                int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                
                // Compute x offset
                int x_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    x_indices[i] = batch_indices[i];
                }
                x_indices[x_ndim - 1] = in;
                int x_offset = compute_offset(x_indices, x_strides, x_ndim);
                
                sum += grad_y_ptr[grad_y_offset] * x_ptr[x_offset];
            }
            
            // Compute grad_w offset
            int grad_w_offset = out * grad_w_strides[0] + in * grad_w_strides[1];
            grad_w_ptr[grad_w_offset] = sum;
        }
        
        // Compute grad_b = sum(grad_y, dim=0)
        if (grad_b_ptr && idx < out_features) {
            float sum = 0.0f;
            // Iterate over all batch elements
            int total_batch_elements = 1;
            for (int i = 0; i < x_ndim - 1; i++) {
                total_batch_elements *= x_shape[i];
            }
            
            for (int batch_idx = 0; batch_idx < total_batch_elements; batch_idx++) {
                // Convert batch_idx to multi-dimensional indices
                int temp_idx = batch_idx;
                int batch_indices[4];
                for (int i = x_ndim - 2; i >= 0; i--) {
                    batch_indices[i] = temp_idx % x_shape[i];
                    temp_idx /= x_shape[i];
                }
                
                // Compute grad_y offset
                int grad_y_indices[4];
                for (int i = 0; i < x_ndim - 1; i++) {
                    grad_y_indices[i] = batch_indices[i];
                }
                grad_y_indices[x_ndim - 1] = idx;
                int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                
                sum += grad_y_ptr[grad_y_offset];
            }
            
            // Compute grad_b offset
            int grad_b_offset = idx * grad_b_strides[0];
            grad_b_ptr[grad_b_offset] = sum;
         }
     }
     else if (dtype == INFINI_DTYPE_BF16) {
         __nv_bfloat16* grad_x_ptr = (__nv_bfloat16*)grad_x;
         __nv_bfloat16* grad_w_ptr = (__nv_bfloat16*)grad_w;
         __nv_bfloat16* grad_b_ptr = (__nv_bfloat16*)grad_b;
         const __nv_bfloat16* grad_y_ptr = (const __nv_bfloat16*)grad_y;
         const __nv_bfloat16* x_ptr = (const __nv_bfloat16*)x;
         const __nv_bfloat16* w_ptr = (const __nv_bfloat16*)w;
         
         // Compute grad_x = grad_y @ w^T (grad_y * w)
         if (grad_x_ptr && idx < batch_size * in_features) {
             // Convert linear index to multi-dimensional indices for grad_x
             int temp_idx = idx;
             int x_indices[4]; // Support up to 4D tensors
             for (int i = x_ndim - 1; i >= 0; i--) {
                 x_indices[i] = temp_idx % x_shape[i];
                 temp_idx /= x_shape[i];
             }
             
             // Get in_feature index
             int in = x_indices[x_ndim - 1];
             
             float sum = 0.0f;
             for (int out = 0; out < out_features; out++) {
                 // Compute grad_y offset
                 int grad_y_indices[4];
                 for (int i = 0; i < x_ndim - 1; i++) {
                     grad_y_indices[i] = x_indices[i];
                 }
                 grad_y_indices[x_ndim - 1] = out;
                 int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                 
                 // Compute w offset: w[out][in]
                 int w_offset = out * w_strides[0] + in * w_strides[1];
                 
                 sum += __bfloat162float(grad_y_ptr[grad_y_offset]) * 
                        __bfloat162float(w_ptr[w_offset]);
             }
             
             // Compute grad_x offset
             int grad_x_offset = compute_offset(x_indices, grad_x_strides, x_ndim);
             grad_x_ptr[grad_x_offset] = __float2bfloat16(sum);
         }
         
         // Compute grad_w = grad_y^T @ x
         if (grad_w_ptr && idx < out_features * in_features) {
             int out = idx / in_features;
             int in = idx % in_features;
             
             float sum = 0.0f;
             // Iterate over all batch elements
             int total_batch_elements = 1;
             for (int i = 0; i < x_ndim - 1; i++) {
                 total_batch_elements *= x_shape[i];
             }
             
             for (int batch_idx = 0; batch_idx < total_batch_elements; batch_idx++) {
                 // Convert batch_idx to multi-dimensional indices
                 int temp_idx = batch_idx;
                 int batch_indices[4];
                 for (int i = x_ndim - 2; i >= 0; i--) {
                     batch_indices[i] = temp_idx % x_shape[i];
                     temp_idx /= x_shape[i];
                 }
                 
                 // Compute grad_y offset
                 int grad_y_indices[4];
                 for (int i = 0; i < x_ndim - 1; i++) {
                     grad_y_indices[i] = batch_indices[i];
                 }
                 grad_y_indices[x_ndim - 1] = out;
                 int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                 
                 // Compute x offset
                 int x_indices[4];
                 for (int i = 0; i < x_ndim - 1; i++) {
                     x_indices[i] = batch_indices[i];
                 }
                 x_indices[x_ndim - 1] = in;
                 int x_offset = compute_offset(x_indices, x_strides, x_ndim);
                 
                 sum += __bfloat162float(grad_y_ptr[grad_y_offset]) * 
                        __bfloat162float(x_ptr[x_offset]);
             }
             
             // Compute grad_w offset
             int grad_w_offset = out * grad_w_strides[0] + in * grad_w_strides[1];
             grad_w_ptr[grad_w_offset] = __float2bfloat16(sum);
         }
         
         // Compute grad_b = sum(grad_y, dim=0)
         if (grad_b_ptr && idx < out_features) {
             float sum = 0.0f;
             // Iterate over all batch elements
             int total_batch_elements = 1;
             for (int i = 0; i < x_ndim - 1; i++) {
                 total_batch_elements *= x_shape[i];
             }
             
             for (int batch_idx = 0; batch_idx < total_batch_elements; batch_idx++) {
                 // Convert batch_idx to multi-dimensional indices
                 int temp_idx = batch_idx;
                 int batch_indices[4];
                 for (int i = x_ndim - 2; i >= 0; i--) {
                     batch_indices[i] = temp_idx % x_shape[i];
                     temp_idx /= x_shape[i];
                 }
                 
                 // Compute grad_y offset
                 int grad_y_indices[4];
                 for (int i = 0; i < x_ndim - 1; i++) {
                     grad_y_indices[i] = batch_indices[i];
                 }
                 grad_y_indices[x_ndim - 1] = idx;
                 int grad_y_offset = compute_offset(grad_y_indices, grad_y_strides, x_ndim);
                 
                 sum += __bfloat162float(grad_y_ptr[grad_y_offset]);
             }
             
             // Compute grad_b offset
             int grad_b_offset = idx * grad_b_strides[0];
             grad_b_ptr[grad_b_offset] = __float2bfloat16(sum);
         }
     }
}

} // namespace op::linear_backward::cuda

#endif // __LINEAR_BACKWARD_KERNEL_CUH__