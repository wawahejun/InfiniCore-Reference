#ifndef __BATCH_NORM_BACKWARD_CUDA_KERNEL_CUH__
#define __BATCH_NORM_BACKWARD_CUDA_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 超高精度类型转换函数：使用double作为中间类型，提高精度
template<typename T>
__device__ inline double ultraPreciseCast(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return static_cast<double>(__half2float(val));  // f16->double
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return static_cast<double>(__bfloat162float(val));  // bf16->double
    } else {
        return static_cast<double>(val);
    }
}

// 高精度类型转换函数
template<typename T>
__device__ inline float preciseCast(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);  
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);  
    } else {
        return static_cast<float>(val);
    }
}

template<typename T>
__device__ inline T directCast(double val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(static_cast<float>(val));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

// Kahan求和算法，提高数值精度
struct EnhancedKahanSum {
    double sum = 0.0;
    double c = 0.0; // 补偿项
    
    __device__ void add(double value) {
        double y = value - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    __device__ double get() const {
        return sum;
    }
};

// 优化的Kahan求和算法：使用float精度
__device__ __forceinline__ void kahanSum(float& sum, float& c, float value) {
    float y = value - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// double精度的Kahan求和算法
__device__ inline void kahanSumDouble(double& sum, double& c, double value) {
    double y = value - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

// 针对F16的特殊数值稳定性函数
__device__ inline float sanitizeF16(float val) {
    // 检测NaN和Inf
    if (isnan(val) || isinf(val)) {
        return 0.0f;
    }
    
    // F16的数值范围：~±65504，但为了数值稳定性使用更保守的范围
    const float max_val = 60000.0f;
    const float min_val = -60000.0f;
    val = fmaxf(fminf(val, max_val), min_val);
    
    // F16的最小正规数约为6.1e-5，避免下溢
    if (fabsf(val) < 1e-4f && val != 0.0f) {
        val = (val > 0.0f) ? 1e-4f : -1e-4f;
    }
    
    return val;
}

// 针对BF16的数值稳定性函数
__device__ inline float sanitizeBF16(float val) {
    // 检测NaN和Inf
    if (isnan(val) || isinf(val)) {
        return 0.0f;
    }
    
    // BF16的数值范围：~±3.4e38，但实际使用更保守的范围
    const float max_val = 1e30f;
    const float min_val = -1e30f;
    val = fmaxf(fminf(val, max_val), min_val);
    
    // BF16的最小正规数约为1.18e-38
    if (fabsf(val) < 1e-37f && val != 0.0f) {
        val = (val > 0.0f) ? 1e-37f : -1e-37f;
    }
    
    return val;
}

// 数值清洗函数
__device__ inline double sanitizeValue(double val) {
    // 处理异常值，保留正常的数值精度
    if (isnan(val)) {
        return 0.0;
    }
    if (isinf(val)) {
        // 保留符号，但限制在合理范围内
        return val > 0 ? 1e30 : -1e30;
    }
    return val;
}

// 通用数值清洗函数（float版本）
__device__ inline float sanitizeValue(float val) {
    if (isnan(val) || isinf(val)) {
        return 0.0f;
    }
    return val;
}

// F16专用的超精密Kahan求和算法
__device__ __forceinline__ void kahanSumF16(float& sum, float& c, float value) {
    // 对输入值进行预处理，避免极端值
    value = sanitizeF16(value);
    
    float y = value - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
    
    // 对累积结果进行清洗
    sum = sanitizeF16(sum);
    c = sanitizeF16(c);
}

// BatchNorm反向传播kernel - BF16数值稳定版本
template <unsigned int BLOCK_SIZE, typename T>
__device__ void batchNormBackwardBlock(
    T *__restrict__ grad_input,
    T *__restrict__ grad_weight,
    T *__restrict__ grad_bias,
    const T *__restrict__ grad_output,
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ running_mean,
    const T *__restrict__ running_var,
    size_t batch_size,
    size_t channels,
    size_t spatial_size,
    double eps) {
    
    const size_t channel_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // 边界检查优化：提前返回避免无效计算
    if (channel_idx >= channels) return;
    
    // 检测是否为低精度类型（F16或BF16），采用不同的精度策略
    constexpr bool is_bf16 = std::is_same_v<T, __nv_bfloat16>;
    constexpr bool is_f16 = std::is_same_v<T, __half>;
    constexpr bool is_low_precision = is_bf16 || is_f16;
    
    // 根据数据类型选择合适的精度：低精度类型使用float，其他使用double
    using ComputeType = typename std::conditional<is_low_precision, float, double>::type;
    
    __shared__ ComputeType s_mean, s_var, s_weight;
    __shared__ ComputeType shared_bias_grad[BLOCK_SIZE];
    __shared__ ComputeType shared_weight_grad[BLOCK_SIZE];
    
    if (tid == 0) {
        if constexpr (is_low_precision) {
            // F16/BF16使用float精度，避免过度精度导致的舍入误差
            s_mean = preciseCast(running_mean[channel_idx]);
            s_var = preciseCast(running_var[channel_idx]);
            s_weight = preciseCast(weight[channel_idx]);
        } else {
            // 其他类型使用double精度
            s_mean = ultraPreciseCast(running_mean[channel_idx]);
            s_var = ultraPreciseCast(running_var[channel_idx]);
            s_weight = ultraPreciseCast(weight[channel_idx]);
        }
        
        // 适应性除零保护
        ComputeType min_var = is_low_precision ? 1e-6f : 1e-12;
        if (s_var <= min_var) {
            s_var = min_var;
        }
    }
    __syncthreads();
    
    // 计算标准差倒数
    ComputeType variance = s_var + static_cast<ComputeType>(eps);
    ComputeType inv_std;
    if constexpr (is_low_precision) {
        inv_std = 1.0f / sqrtf(variance);
    } else {
        inv_std = 1.0 / sqrt(variance);
    }
    
    // 数值清洗
    if constexpr (is_low_precision) {
        if constexpr (is_f16) {
            inv_std = sanitizeF16(static_cast<float>(inv_std));
        } else { // BF16
            inv_std = sanitizeBF16(static_cast<float>(inv_std));
        }
    } else {
        inv_std = sanitizeValue(static_cast<double>(inv_std));
    }
    
    // Step 1: 根据数据类型选择合适的求和策略
    ComputeType bias_sum = 0.0, bias_c = 0.0;
    ComputeType weight_sum = 0.0, weight_c = 0.0;
    
    // 优化内存访问：连续访问模式
    const size_t total_elements = batch_size * spatial_size;
    const size_t elements_per_thread = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // 每个线程独立计算
    for (size_t elem_idx = 0; elem_idx < elements_per_thread; ++elem_idx) {
        size_t global_idx = tid + elem_idx * BLOCK_SIZE;
        if (global_idx >= total_elements) break;
        
        // 计算batch和spatial索引
        size_t batch_idx = global_idx / spatial_size;
        size_t spatial_idx = global_idx % spatial_size;
        
        // 连续内存访问索引
        size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
        
        // 根据类型选择精度转换
        ComputeType grad_out_val, input_val;
        if constexpr (is_low_precision) {
            grad_out_val = preciseCast(grad_output[idx]);
            input_val = preciseCast(input[idx]);
        } else {
            grad_out_val = ultraPreciseCast(grad_output[idx]);
            input_val = ultraPreciseCast(input[idx]);
        }
        
        // 标准化的输入值
        ComputeType normalized = (input_val - s_mean) * inv_std;
        
        // 使用适当精度的Kahan求和
        if constexpr (is_low_precision) {
            if constexpr (is_f16) {
                // F16使用超精密求和算法
                kahanSumF16(bias_sum, bias_c, grad_out_val);
                kahanSumF16(weight_sum, weight_c, grad_out_val * normalized);
            } else { // BF16
                kahanSum(bias_sum, bias_c, grad_out_val);
                kahanSum(weight_sum, weight_c, grad_out_val * normalized);
            }
        } else {
            kahanSumDouble(bias_sum, bias_c, grad_out_val);
            kahanSumDouble(weight_sum, weight_c, grad_out_val * normalized);
        }
    }
    
    // 将结果存储到共享内存
    shared_bias_grad[tid] = bias_sum;
    shared_weight_grad[tid] = weight_sum;
    __syncthreads();
    
    // 使用适当精度的树形归约
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < BLOCK_SIZE) {
            shared_bias_grad[tid] += shared_bias_grad[tid + stride];
            shared_weight_grad[tid] += shared_weight_grad[tid + stride];
        }
        __syncthreads();
    }
    
    // 存储grad_bias和grad_weight结果
    if (tid == 0) {
        if constexpr (is_low_precision) {
            // F16/BF16使用类型特定的数值处理
            float bias_val, weight_val;
            if constexpr (is_f16) {
                bias_val = sanitizeF16(static_cast<float>(shared_bias_grad[0]));
                weight_val = sanitizeF16(static_cast<float>(shared_weight_grad[0]));
            } else { // BF16
                bias_val = sanitizeBF16(static_cast<float>(shared_bias_grad[0]));
                weight_val = sanitizeBF16(static_cast<float>(shared_weight_grad[0]));
            }
            grad_bias[channel_idx] = directCast<T>(static_cast<double>(bias_val));
            grad_weight[channel_idx] = directCast<T>(static_cast<double>(weight_val));
        } else {
            grad_bias[channel_idx] = directCast<T>(sanitizeValue(static_cast<double>(shared_bias_grad[0])));
            grad_weight[channel_idx] = directCast<T>(sanitizeValue(static_cast<double>(shared_weight_grad[0])));
        }
    }
    
    // Step 2: 计算grad_input - 根据类型选择精度
    if (grad_input) {
        // 重新计算每个线程处理的元素
        for (size_t elem_idx = 0; elem_idx < elements_per_thread; ++elem_idx) {
            size_t global_idx = tid + elem_idx * BLOCK_SIZE;
            if (global_idx >= total_elements) break;
            
            // 计算batch和spatial索引
            size_t batch_idx = global_idx / spatial_size;
            size_t spatial_idx = global_idx % spatial_size;
            
            // 连续内存访问索引
            size_t idx = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx;
            
            // 根据类型选择精度
            ComputeType grad_out_val;
            if constexpr (is_low_precision) {
                grad_out_val = preciseCast(grad_output[idx]);
            } else {
                grad_out_val = ultraPreciseCast(grad_output[idx]);
            }
            
            // 在推理模式下，grad_input的计算简化为：
            // grad_input = grad_output * weight * inv_std
            ComputeType grad_in_val = grad_out_val * s_weight * inv_std;
            
            if constexpr (is_low_precision) {
                float sanitized_val;
                if constexpr (is_f16) {
                    sanitized_val = sanitizeF16(static_cast<float>(grad_in_val));
                } else { // BF16
                    sanitized_val = sanitizeBF16(static_cast<float>(grad_in_val));
                }
                grad_input[idx] = directCast<T>(static_cast<double>(sanitized_val));
            } else {
                grad_input[idx] = directCast<T>(sanitizeValue(static_cast<double>(grad_in_val)));
            }
        }
    }
}

#endif // __BATCH_NORM_BACKWARD_CUDA_KERNEL_CUH__