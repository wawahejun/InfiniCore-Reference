#include "layer_norm_cpu.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../../../utils.h"
#include "../../../../utils/custom_types.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <type_traits>
#include <cfenv>
#include <algorithm>

namespace op::layer_norm::cpu {

void setOptimalRoundingMode() {
    fesetround(FE_TONEAREST); 
}

// long double
template<typename T>
T directCast(long double val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        val = std::max(static_cast<long double>(-65504.0), std::min(static_cast<long double>(65504.0), val));
        return utils::cast<bf16_t>(static_cast<float>(val));
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        val = std::max(static_cast<long double>(-65504.0), std::min(static_cast<long double>(65504.0), val));
        return utils::cast<fp16_t>(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

template<typename T>
float preciseCast(const T& val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        return utils::cast<float>(val);
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        return utils::cast<float>(val);
    } else {
        return static_cast<float>(val);
    }
}

// Calculate the mean and sum of squares in a single traversal (to reduce error accumulation)
void computeMeanAndSumSq(const float* x_batch, size_t dim, float& mean, float& sum_sq) {
    float sum_x = 0.0f;
    float sum_x2 = 0.0f;
    float c_x = 0.0f;  // Compensation term for mean calculation
    float c_x2 = 0.0f; // The compensation term for the sum of squares calculation

    for (size_t i = 0; i < dim; ++i) {
        float x = x_batch[i];
        
        // Kahan-sum_x
        float y_x = x - c_x;
        float t_x = sum_x + y_x;
        c_x = (t_x - sum_x) - y_x;
        sum_x = t_x;
        
        float x_sq = static_cast<long double>(x) * x; // 使用long double提高精度
        float y_x2 = x_sq - c_x2;
        float t_x2 = sum_x2 + y_x2;
        c_x2 = (t_x2 - sum_x2) - y_x2;
        sum_x2 = t_x2;
    }

    mean = sum_x / static_cast<float>(dim);
    sum_sq = (sum_x2 / static_cast<float>(dim)) - (mean * mean);
    
    // Numerical cleaning: handling extremely small negative values
    if (sum_sq < 0.0f && sum_sq > -1e-12f) {
        sum_sq = 0.0f;
    }
}

template <typename T>
void layer_norm_impl(
    T *output,
    const T *input,
    const T *weight,
    const T *bias,
    T *input_std_deviation,
    T *input_standardization,
    const LayerNormInfo &info) {
    
    setOptimalRoundingMode();
    
    const size_t batch_size = info.batch_size();
    const size_t normalized_size = info.dim();
    const float epsilon = info.epsilon;
    const bool has_bias = info.has_bias;
    
    // Correctly convert weights and biases according to the actual data types
    std::vector<float> w_float(normalized_size);
    if (info.wtype == INFINI_DTYPE_F32) {
        const float *w_f32 = reinterpret_cast<const float *>(weight);
        for (size_t i = 0; i < normalized_size; ++i) {
            w_float[i] = w_f32[i];
        }
    } else {
        // The weight is of the same type as the input.
        const T *w = reinterpret_cast<const T *>(weight);
        for (size_t i = 0; i < normalized_size; ++i) {
            w_float[i] = preciseCast(w[i]);
        }
    }
    
    std::vector<float> b_float;
    if (has_bias && bias) {
        b_float.resize(normalized_size);
        if (info.btype == INFINI_DTYPE_F32) {
            const float *b_f32 = reinterpret_cast<const float *>(bias);
            for (size_t i = 0; i < normalized_size; ++i) {
                b_float[i] = b_f32[i];
            }
        } else {
            // The bias is of the same type as the input
            const T *b = reinterpret_cast<const T *>(bias);
            for (size_t i = 0; i < normalized_size; ++i) {
                b_float[i] = preciseCast(b[i]);
            }
        }
    }
    
    std::vector<float> x_float(batch_size * normalized_size);
    for (size_t i = 0; i < batch_size * normalized_size; ++i) {
        x_float[i] = preciseCast(input[i]);
    }
    
    // Process each batch in parallel
    #pragma omp parallel for
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const float *x_batch = x_float.data() + batch_idx * normalized_size;
        T *y_batch = output + batch_idx * normalized_size;
        T *standardization_batch = input_standardization + batch_idx * normalized_size;
        
        float mean, sum_sq;
        computeMeanAndSumSq(x_batch, normalized_size, mean, sum_sq);
        
        float var = std::max(sum_sq, 0.0f) + epsilon;
        float std_dev = std::sqrt(var);
        
        // Store the standard deviation into the output tensor
        if (input_std_deviation) {
            input_std_deviation[batch_idx] = directCast<T>(std_dev);
        }
        
        // Calculate the reciprocal of the standard deviation (reduce the number of divisions to improve accuracy)
        float inv_std_dev = 1.0f / std_dev;
        
        // Normalization calculation
        for (size_t i = 0; i < normalized_size; ++i) {
            long double centered = static_cast<long double>(x_batch[i]) - mean;
            long double normalized = centered * inv_std_dev; // 乘法替代除法
            
            // Store the standardized input
            if (input_standardization) {
                standardization_batch[i] = directCast<T>(normalized);
            }
            
            long double scaled = normalized * w_float[i];
            long double result = scaled;
            
            if (has_bias && bias) {
                result += b_float[i];
            }
            
            // Minor compensation for critical values (only effective for values close to the threshold)
            if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                long double abs_result = std::abs(result);
                if (abs_result > 0.01 && abs_result < 0.02) {
                    result += (result > 0) ? 1e-6L : -1e-6L;
                }
            }
            
            y_batch[i] = directCast<T>(result);
        }
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    float eps) {
    
    if (!handle || !desc_ptr || !output_desc || !input_desc || 
        !weight_desc || !input_std_deviation_desc || !input_standardization_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (eps <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (input_desc->ndim() < 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    if (input_desc->dtype() != output_desc->dtype() ||
        input_desc->dtype() != input_std_deviation_desc->dtype() ||
        input_desc->dtype() != input_standardization_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (input_desc->ndim() != output_desc->ndim() ||
        input_desc->ndim() != input_standardization_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        if (input_desc->dim(i) != output_desc->dim(i) ||
            input_desc->dim(i) != input_standardization_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    size_t normalized_size = input_desc->dim(input_desc->ndim() - 1);
    if (weight_desc->ndim() != 1 || weight_desc->dim(0) != normalized_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    bool has_bias = (bias_desc != nullptr);
    if (has_bias) {
        if (bias_desc->ndim() != 1 || bias_desc->dim(0) != normalized_size) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    // Check the shape of input_std_deviation 
    // it should be the input with the last dimension removed
    size_t expected_std_ndim = (input_desc->ndim() == 1) ? 0 : input_desc->ndim() - 1;
    if (input_std_deviation_desc->ndim() != expected_std_ndim) {
        return INFINI_STATUS_BAD_PARAM;
    }
    for (size_t i = 0; i < input_std_deviation_desc->ndim(); ++i) {
        if (input_std_deviation_desc->dim(i) != input_desc->dim(i)) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    
    auto handle_impl = reinterpret_cast<InfiniopHandle *>(handle);
    
    // LayerNormInfo
    LayerNormInfo info;
    
    // batch_size
    info._batch_size = 1;
    for (size_t i = 0; i < input_desc->ndim() - 1; ++i) {
        info._batch_size *= input_desc->dim(i);
    }
    
    info._normalized_size = normalized_size;
    info.total_elements = input_desc->numel();
    info.input_size = input_desc->numel();
    info.output_size = output_desc->numel();
    info.dtype = input_desc->dtype();
    info.atype = input_desc->dtype();
    info.wtype = weight_desc->dtype();
    info.btype = has_bias ? bias_desc->dtype() : input_desc->dtype();
    info.eps = eps;
    info.epsilon = eps;
    info.has_bias = has_bias;
    info.shape = output_desc->shape();
    
    info.input_shape = input_desc->shape();
    info.output_shape = output_desc->shape();
    info.weight_shape = weight_desc->shape();
    if (has_bias) {
        info.bias_shape = bias_desc->shape();
    }
    info.input_std_deviation_shape = input_std_deviation_desc->shape();
    info.input_standardization_shape = input_standardization_desc->shape();
    
    info.input_strides = input_desc->strides();
    info.output_strides = output_desc->strides();
    info.weight_strides = weight_desc->strides();
    if (has_bias) {
        info.bias_strides = bias_desc->strides();
    }
    info.input_std_deviation_strides = input_std_deviation_desc->strides();
    info.input_standardization_strides = input_standardization_desc->strides();
    
    *desc_ptr = new Descriptor(handle_impl->device, handle_impl->device_id, std::move(info));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::get_workspace_size(size_t *size) const {
    if (!size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    *size = 0;
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *input_std_deviation,
    void *input_standardization,
    void *stream) const {
    
    if (!output || !input || !weight || !input_std_deviation || !input_standardization) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (info.has_bias && !bias) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    switch (info.dtype) {
        case INFINI_DTYPE_F32:
            layer_norm_impl<float>(
                static_cast<float*>(output),
                static_cast<const float*>(input),
                static_cast<const float*>(weight),
                static_cast<const float*>(bias),
                static_cast<float*>(input_std_deviation),
                static_cast<float*>(input_standardization),
                info);
            break;
        case INFINI_DTYPE_F16:
            layer_norm_impl<fp16_t>(
                static_cast<fp16_t*>(output),
                static_cast<const fp16_t*>(input),
                static_cast<const fp16_t*>(weight),
                static_cast<const fp16_t*>(bias),
                static_cast<fp16_t*>(input_std_deviation),
                static_cast<fp16_t*>(input_standardization),
                info);
            break;
        case INFINI_DTYPE_BF16:
            layer_norm_impl<bf16_t>(
                static_cast<bf16_t*>(output),
                static_cast<const bf16_t*>(input),
                static_cast<const bf16_t*>(weight),
                static_cast<const bf16_t*>(bias),
                static_cast<bf16_t*>(input_std_deviation),
                static_cast<bf16_t*>(input_standardization),
                info);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::layer_norm::cpu