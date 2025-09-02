#include "linear_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../tensor.h"
#include "../../../../utils/custom_types.h"

namespace op::linear_backward::cpu {

// Linear backward kernel implementation
template<typename T>
static infiniStatus_t linear_backward_kernel(
    T *grad_x,
    T *grad_w,
    T *grad_b,
    const T *grad_y,
    const T *x,
    const T *w,
    const std::vector<int> &grad_y_dims,
    const std::vector<int> &x_dims,
    const std::vector<int> &w_dims,
    const std::vector<ptrdiff_t> &grad_y_strides,
    const std::vector<ptrdiff_t> &x_strides,
    const std::vector<ptrdiff_t> &w_strides,
    const std::vector<ptrdiff_t> &grad_x_strides,
    const std::vector<ptrdiff_t> &grad_w_strides,
    const std::vector<ptrdiff_t> &grad_b_strides) {
    
    int grad_y_ndim = grad_y_dims.size();
    int batch_size = 1;
    for (int i = 0; i < grad_y_ndim - 1; i++) {
        batch_size *= grad_y_dims[i];
    }
    
    int in_features = x_dims[x_dims.size() - 1];
    int out_features = w_dims[0];
    
    // Helper function to compute multi-dimensional index to linear offset
    auto compute_offset = [](const std::vector<int>& indices, const std::vector<ptrdiff_t>& strides) -> ptrdiff_t {
        ptrdiff_t offset = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            offset += indices[i] * strides[i];
        }
        return offset;
    };
    
    // Compute grad_x = grad_y * w
      if (grad_x) {
          std::vector<int> batch_indices(grad_y_dims.size() - 1, 0);
          for (int batch = 0; batch < batch_size; batch++) {
              // Convert linear batch index to multi-dimensional indices
              int temp_batch = batch;
              for (int i = grad_y_dims.size() - 2; i >= 0; i--) {
                  batch_indices[i] = temp_batch % grad_y_dims[i];
                  temp_batch /= grad_y_dims[i];
              }
              
              for (int in = 0; in < in_features; in++) {
                  if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                      float sum = 0.0f;
                      for (int out = 0; out < out_features; out++) {
                          // Compute grad_y index with strides
                          std::vector<int> grad_y_indices = batch_indices;
                          grad_y_indices.push_back(out);
                          ptrdiff_t grad_y_offset = compute_offset(grad_y_indices, grad_y_strides);
                          
                          // Compute w index with strides
                          std::vector<int> w_indices = {out, in};
                          ptrdiff_t w_offset = compute_offset(w_indices, w_strides);
                          
                          sum += utils::cast<float>(grad_y[grad_y_offset]) * utils::cast<float>(w[w_offset]);
                      }
                      
                      // Compute grad_x index with strides
                      std::vector<int> grad_x_indices = batch_indices;
                      grad_x_indices.push_back(in);
                      ptrdiff_t grad_x_offset = compute_offset(grad_x_indices, grad_x_strides);
                      grad_x[grad_x_offset] = utils::cast<T>(sum);
                  } else {
                      T sum{};
                      for (int out = 0; out < out_features; out++) {
                          // Compute grad_y index with strides
                          std::vector<int> grad_y_indices = batch_indices;
                          grad_y_indices.push_back(out);
                          ptrdiff_t grad_y_offset = compute_offset(grad_y_indices, grad_y_strides);
                          
                          // Compute w index with strides
                          std::vector<int> w_indices = {out, in};
                          ptrdiff_t w_offset = compute_offset(w_indices, w_strides);
                          
                          sum += grad_y[grad_y_offset] * w[w_offset];
                      }
                      
                      // Compute grad_x index with strides
                      std::vector<int> grad_x_indices = batch_indices;
                      grad_x_indices.push_back(in);
                      ptrdiff_t grad_x_offset = compute_offset(grad_x_indices, grad_x_strides);
                      grad_x[grad_x_offset] = sum;
                  }
              }
          }
      }
     
     // Compute grad_w = grad_y^T * x
     if (grad_w) {
         // Initialize grad_w to zero
         for (int out = 0; out < out_features; out++) {
             for (int in = 0; in < in_features; in++) {
                 std::vector<int> grad_w_indices = {out, in};
                 ptrdiff_t grad_w_offset = compute_offset(grad_w_indices, grad_w_strides);
                 grad_w[grad_w_offset] = T{};
             }
         }
         
         for (int out = 0; out < out_features; out++) {
             for (int in = 0; in < in_features; in++) {
                 if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                     float sum = 0.0f;
                     std::vector<int> batch_indices(grad_y_dims.size() - 1, 0);
                     for (int batch = 0; batch < batch_size; batch++) {
                         // Convert linear batch index to multi-dimensional indices
                         int temp_batch = batch;
                         for (int i = grad_y_dims.size() - 2; i >= 0; i--) {
                             batch_indices[i] = temp_batch % grad_y_dims[i];
                             temp_batch /= grad_y_dims[i];
                         }
                         
                         // Compute grad_y index with strides
                         std::vector<int> grad_y_indices = batch_indices;
                         grad_y_indices.push_back(out);
                         ptrdiff_t grad_y_offset = compute_offset(grad_y_indices, grad_y_strides);
                         
                         // Compute x index with strides
                         std::vector<int> x_indices = batch_indices;
                         x_indices.push_back(in);
                         ptrdiff_t x_offset = compute_offset(x_indices, x_strides);
                         
                         sum += utils::cast<float>(grad_y[grad_y_offset]) * utils::cast<float>(x[x_offset]);
                     }
                     
                     std::vector<int> grad_w_indices = {out, in};
                     ptrdiff_t grad_w_offset = compute_offset(grad_w_indices, grad_w_strides);
                     grad_w[grad_w_offset] = utils::cast<T>(sum);
                 } else {
                     T sum{};
                     std::vector<int> batch_indices(grad_y_dims.size() - 1, 0);
                     for (int batch = 0; batch < batch_size; batch++) {
                         // Convert linear batch index to multi-dimensional indices
                         int temp_batch = batch;
                         for (int i = grad_y_dims.size() - 2; i >= 0; i--) {
                             batch_indices[i] = temp_batch % grad_y_dims[i];
                             temp_batch /= grad_y_dims[i];
                         }
                         
                         // Compute grad_y index with strides
                         std::vector<int> grad_y_indices = batch_indices;
                         grad_y_indices.push_back(out);
                         ptrdiff_t grad_y_offset = compute_offset(grad_y_indices, grad_y_strides);
                         
                         // Compute x index with strides
                         std::vector<int> x_indices = batch_indices;
                         x_indices.push_back(in);
                         ptrdiff_t x_offset = compute_offset(x_indices, x_strides);
                         
                         sum += grad_y[grad_y_offset] * x[x_offset];
                     }
                     
                     std::vector<int> grad_w_indices = {out, in};
                     ptrdiff_t grad_w_offset = compute_offset(grad_w_indices, grad_w_strides);
                     grad_w[grad_w_offset] = sum;
                 }
             }
         }
     }
     
     // Compute grad_b = sum(grad_y, dim=0)
     if (grad_b) {
         for (int out = 0; out < out_features; out++) {
             if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                 float sum = 0.0f;
                 std::vector<int> batch_indices(grad_y_dims.size() - 1, 0);
                 for (int batch = 0; batch < batch_size; batch++) {
                     // Convert linear batch index to multi-dimensional indices
                     int temp_batch = batch;
                     for (int i = grad_y_dims.size() - 2; i >= 0; i--) {
                         batch_indices[i] = temp_batch % grad_y_dims[i];
                         temp_batch /= grad_y_dims[i];
                     }
                     
                     // Compute grad_y index with strides
                     std::vector<int> grad_y_indices = batch_indices;
                     grad_y_indices.push_back(out);
                     ptrdiff_t grad_y_offset = compute_offset(grad_y_indices, grad_y_strides);
                     
                     sum += utils::cast<float>(grad_y[grad_y_offset]);
                 }
                 
                 // Compute grad_b index with strides
                 std::vector<int> grad_b_indices = {out};
                 ptrdiff_t grad_b_offset = compute_offset(grad_b_indices, grad_b_strides);
                 grad_b[grad_b_offset] = utils::cast<T>(sum);
             } else {
                 T sum{};
                 std::vector<int> batch_indices(grad_y_dims.size() - 1, 0);
                 for (int batch = 0; batch < batch_size; batch++) {
                     // Convert linear batch index to multi-dimensional indices
                     int temp_batch = batch;
                     for (int i = grad_y_dims.size() - 2; i >= 0; i--) {
                         batch_indices[i] = temp_batch % grad_y_dims[i];
                         temp_batch /= grad_y_dims[i];
                     }
                     
                     // Compute grad_y index with strides
                     std::vector<int> grad_y_indices = batch_indices;
                     grad_y_indices.push_back(out);
                     ptrdiff_t grad_y_offset = compute_offset(grad_y_indices, grad_y_strides);
                     
                     sum += grad_y[grad_y_offset];
                 }
                 
                 // Compute grad_b index with strides
                 std::vector<int> grad_b_indices = {out};
                 ptrdiff_t grad_b_offset = compute_offset(grad_b_indices, grad_b_strides);
                 grad_b[grad_b_offset] = sum;
             }
         }
     }
    
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t grad_x_desc,
    infiniopTensorDescriptor_t grad_w_desc,
    infiniopTensorDescriptor_t grad_b_desc) {

    // Check device type
    if (handle->device != INFINI_DEVICE_CPU) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    // Check data types
    auto grad_y_dtype = grad_y_desc->dtype();
    auto x_dtype = x_desc->dtype();
    auto w_dtype = w_desc->dtype();

    if (grad_y_dtype != x_dtype || grad_y_dtype != w_dtype) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check gradient tensor data types if provided
    if (grad_x_desc) {
        auto grad_x_dtype = grad_x_desc->dtype();
        if (grad_x_dtype != grad_y_dtype) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    if (grad_w_desc) {
        auto grad_w_dtype = grad_w_desc->dtype();
        if (grad_w_dtype != grad_y_dtype) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    if (grad_b_desc) {
        auto grad_b_dtype = grad_b_desc->dtype();
        if (grad_b_dtype != grad_y_dtype) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    // Check dimensions
    auto grad_y_shape = grad_y_desc->shape();
    auto x_shape = x_desc->shape();
    auto w_shape = w_desc->shape();
    
    int grad_y_ndim = grad_y_shape.size();
    int x_ndim = x_shape.size();
    int w_ndim = w_shape.size();

    if (w_ndim != 2 || x_ndim < 1 || grad_y_ndim < 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Get dimensions
    std::vector<int> grad_y_dims(grad_y_shape.begin(), grad_y_shape.end());
    std::vector<int> x_dims(x_shape.begin(), x_shape.end());
    std::vector<int> w_dims(w_shape.begin(), w_shape.end());

    // Check dimension compatibility
    int in_features = x_dims[x_ndim - 1];
    int out_features = w_dims[0];
    int weight_in_features = w_dims[1];

    if (in_features != weight_in_features) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (grad_y_dims[grad_y_ndim - 1] != out_features) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check gradient tensor dimensions if provided
    if (grad_x_desc) {
        auto grad_x_shape = grad_x_desc->shape();
        int grad_x_ndim = grad_x_shape.size();
        std::vector<int> grad_x_dims(grad_x_shape.begin(), grad_x_shape.end());

        if (grad_x_ndim != x_ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }
        for (int i = 0; i < x_ndim; i++) {
            if (grad_x_dims[i] != x_dims[i]) {
                return INFINI_STATUS_BAD_PARAM;
            }
        }
    }

    if (grad_w_desc) {
        auto grad_w_shape = grad_w_desc->shape();
        int grad_w_ndim = grad_w_shape.size();
        std::vector<int> grad_w_dims(grad_w_shape.begin(), grad_w_shape.end());

        if (grad_w_ndim != 2 || grad_w_dims[0] != out_features || grad_w_dims[1] != in_features) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    if (grad_b_desc) {
        auto grad_b_shape = grad_b_desc->shape();
        int grad_b_ndim = grad_b_shape.size();
        std::vector<int> grad_b_dims(grad_b_shape.begin(), grad_b_shape.end());

        if (grad_b_ndim != 1 || grad_b_dims[0] != out_features) {
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    // Create descriptor
    auto desc = new Descriptor();
    desc->device_type = INFINI_DEVICE_CPU;
    desc->_handle = reinterpret_cast<device::cpu::Handle *>(handle);

    // Clone tensor descriptors
    desc->_grad_y_desc = grad_y_desc;
    desc->_x_desc = x_desc;
    desc->_w_desc = w_desc;
    
    if (grad_x_desc) {
        desc->_grad_x_desc = grad_x_desc;
    } else {
        desc->_grad_x_desc = nullptr;
    }
    
    if (grad_w_desc) {
        desc->_grad_w_desc = grad_w_desc;
    } else {
        desc->_grad_w_desc = nullptr;
    }
    
    if (grad_b_desc) {
        desc->_grad_b_desc = grad_b_desc;
    } else {
        desc->_grad_b_desc = nullptr;
    }
    
    // Save tensor shapes and data type to avoid accessing descriptors later
    desc->_grad_y_dims = std::vector<int>(grad_y_dims.begin(), grad_y_dims.end());
    desc->_x_dims = std::vector<int>(x_dims.begin(), x_dims.end());
    desc->_w_dims = std::vector<int>(w_dims.begin(), w_dims.end());
    desc->_dtype = grad_y_dtype;
    
    // Save tensor strides
    desc->_grad_y_strides = grad_y_desc->strides();
    desc->_x_strides = x_desc->strides();
    desc->_w_strides = w_desc->strides();
    
    if (grad_x_desc) {
        desc->_grad_x_strides = grad_x_desc->strides();
    }
    
    if (grad_w_desc) {
        desc->_grad_w_strides = grad_w_desc->strides();
    }
    
    if (grad_b_desc) {
        desc->_grad_b_strides = grad_b_desc->strides();
    }

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_x,
    void *grad_w,
    void *grad_b,
    const void *grad_y,
    const void *x,
    const void *w,
    void *stream) const {

    // Use saved data type and shapes
    auto grad_y_dtype = _dtype;
    const auto& grad_y_dims = _grad_y_dims;
    const auto& x_dims = _x_dims;
    const auto& w_dims = _w_dims;

    // Call kernel based on data type
    if (grad_y_dtype == INFINI_DTYPE_F32) {
        return linear_backward_kernel<float>(
            static_cast<float *>(grad_x),
            static_cast<float *>(grad_w),
            static_cast<float *>(grad_b),
            static_cast<const float *>(grad_y),
            static_cast<const float *>(x),
            static_cast<const float *>(w),
            grad_y_dims, x_dims, w_dims,
            _grad_y_strides, _x_strides, _w_strides,
            _grad_x_strides, _grad_w_strides, _grad_b_strides);
    } else if (grad_y_dtype == INFINI_DTYPE_F16) {
        return linear_backward_kernel<fp16_t>(
            static_cast<fp16_t *>(grad_x),
            static_cast<fp16_t *>(grad_w),
            static_cast<fp16_t *>(grad_b),
            static_cast<const fp16_t *>(grad_y),
            static_cast<const fp16_t *>(x),
            static_cast<const fp16_t *>(w),
            grad_y_dims, x_dims, w_dims,
            _grad_y_strides, _x_strides, _w_strides,
            _grad_x_strides, _grad_w_strides, _grad_b_strides);
    } else if (grad_y_dtype == INFINI_DTYPE_BF16) {
        return linear_backward_kernel<bf16_t>(
            static_cast<bf16_t *>(grad_x),
            static_cast<bf16_t *>(grad_w),
            static_cast<bf16_t *>(grad_b),
            static_cast<const bf16_t *>(grad_y),
            static_cast<const bf16_t *>(x),
            static_cast<const bf16_t *>(w),
            grad_y_dims, x_dims, w_dims,
            _grad_y_strides, _x_strides, _w_strides,
            _grad_x_strides, _grad_w_strides, _grad_b_strides);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::linear_backward::cpu