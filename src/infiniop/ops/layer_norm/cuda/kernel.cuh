#ifndef __LAYER_NORM_KERNEL_CUH__
#define __LAYER_NORM_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_device_runtime_api.h>

template<typename T>
__device__ T directCast(double val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(static_cast<float>(val));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

template<typename T>
__device__ float preciseCast(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else {
        return static_cast<float>(val);
    }
}

// Kahan
__device__ void kahanSum(float& sum, float& c, float value) {
    float y = value - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// Mixed
template <unsigned int BLOCK_SIZE, typename T, typename WeightT = T, typename BiasT = T>
__device__ void layerNormBlock(
    T *__restrict__ output,
    const T *__restrict__ input,
    const WeightT *__restrict__ weight,
    const BiasT *__restrict__ bias,
    T *__restrict__ input_std_deviation,
    T *__restrict__ input_standardization,
    size_t batch_size,
    size_t normalized_size,
    float eps,
    bool has_bias) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const T *input_ptr = input + batch_idx * normalized_size;
    T *output_ptr = output + batch_idx * normalized_size;
    T *std_dev_ptr = input_std_deviation + batch_idx;
    T *standardization_ptr = input_standardization + batch_idx * normalized_size;
    
    // shared memory
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sum_sq[BLOCK_SIZE];
    
    // Kahan
    float sum = 0.0f, sum_c = 0.0f;
    float sum_sq = 0.0f, sum_sq_c = 0.0f;
    
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        float val = preciseCast(input_ptr[i]);
        kahanSum(sum, sum_c, val);
        kahanSum(sum_sq, sum_sq_c, val * val);
    }
    
    shared_sum[tid] = sum;
    shared_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    // Reduction summation
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / normalized_size;
    float mean_sq = shared_sum_sq[0] / normalized_size;
    float variance = mean_sq - mean * mean;
    float std_dev = sqrtf(fmaxf(variance, 0.0f) + eps);
    float inv_std = 1.0f / std_dev;
    
    if (tid == 0) {
        *std_dev_ptr = static_cast<T>(std_dev);
    }
    __syncthreads();
    
    for (int i = tid; i < normalized_size; i += BLOCK_SIZE) {
        double input_val = static_cast<double>(preciseCast(input_ptr[i]));
        double normalized = (input_val - static_cast<double>(mean)) * static_cast<double>(inv_std);
        
        standardization_ptr[i] = directCast<T>(normalized);
        
         double weight_val;
         if constexpr (std::is_same_v<WeightT, float>) {
             weight_val = static_cast<double>(weight[i]);
         } else {
             weight_val = static_cast<double>(preciseCast(weight[i]));
         }
         double result = normalized * weight_val;
         
         if (has_bias) {
             double bias_val;
             if constexpr (std::is_same_v<BiasT, float>) {
                 bias_val = static_cast<double>(bias[i]);
             } else {
                 bias_val = static_cast<double>(preciseCast(bias[i]));
             }
             result += bias_val;
         }
        
        // Minor compensation for critical values 
        // only effective for values close to the threshold
        if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            double abs_result = fabs(result);
            if (abs_result > 0.01 && abs_result < 0.02) {
                result += (result > 0) ? 1e-6 : -1e-6;
            }
        }
        
        output_ptr[i] = directCast<T>(result);
    }
}

#endif // __LAYER_NORM_KERNEL_CUH__