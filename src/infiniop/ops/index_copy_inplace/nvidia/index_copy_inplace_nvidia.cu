#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "index_copy_inplace_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"

namespace op::index_copy_inplace::nvidia {

Descriptor::~Descriptor() = default;



infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t source_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = target_desc->dtype();
    
    // Check data types - 支持所有合法类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_BOOL);
    
    // Check that target and source have same dtype
    if (target_desc->dtype() != source_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check that index is integer type
    auto index_dtype = index_desc->dtype();
    if (index_dtype != INFINI_DTYPE_I32 && index_dtype != INFINI_DTYPE_I64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check dimension bounds
    auto target_shape = target_desc->shape();
    auto source_shape = source_desc->shape();
    if (dim < 0 || dim >= static_cast<int>(target_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (dim < 0 || dim >= static_cast<int>(source_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Check that target and source have same shape except possibly at dim
    if (target_shape.size() != source_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    auto desc = new Descriptor();
    desc->device_type = handle->device;  // 设置设备类型
    desc->device_id = handle->device_id;  // 设置设备ID
    desc->_target_desc = target_desc;
    desc->_source_desc = source_desc;
    desc->_index_desc = index_desc;
    desc->_dim = dim;
    desc->_handle = handle;
    
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *target,
    const void *source,
    const void *index,
    void *stream) const {
    
    auto target_shape = _target_desc->shape();
    auto source_shape = _source_desc->shape();
    auto index_shape = _index_desc->shape();
    auto target_strides = _target_desc->strides();
    auto source_strides = _source_desc->strides();
    auto index_strides = _index_desc->strides();
    auto dtype = _target_desc->dtype();
    auto index_dtype = _index_desc->dtype();
    
    // Calculate total elements based on source shape (we iterate over source elements)
    size_t total_elements = 1;
    for (size_t s : source_shape) {
        total_elements *= s;
    }
    
    // Copy shape and stride data to device
    int *d_target_shape, *d_source_shape, *d_index_shape;
    int *d_target_strides, *d_source_strides, *d_index_strides;
    
    int ndim = target_shape.size();
     
     // Declare variables before any goto statements
     int block_size = 256;
     int grid_size = (total_elements + block_size - 1) / block_size;
     cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
     
     cudaError_t err;
     err = cudaMalloc(&d_target_shape, ndim * sizeof(int));
    if (err != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
     err = cudaMalloc(&d_source_shape, ndim * sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_target_shape);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_index_shape, sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_target_shape);
         cudaFree(d_source_shape);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_target_strides, ndim * sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_target_shape);
         cudaFree(d_source_shape);
         cudaFree(d_index_shape);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_source_strides, ndim * sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_target_shape);
         cudaFree(d_source_shape);
         cudaFree(d_index_shape);
         cudaFree(d_target_strides);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_index_strides, sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_target_shape);
         cudaFree(d_source_shape);
         cudaFree(d_index_shape);
         cudaFree(d_target_strides);
         cudaFree(d_source_strides);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
    
    std::vector<int> h_target_shape(target_shape.begin(), target_shape.end());
    std::vector<int> h_source_shape(source_shape.begin(), source_shape.end());
    std::vector<int> h_target_strides(target_strides.begin(), target_strides.end());
    std::vector<int> h_source_strides(source_strides.begin(), source_strides.end());
    int h_index_shape = index_shape[0];
    int h_index_stride = index_strides[0];
    
    err = cudaMemcpy(d_target_shape, h_target_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_source_shape, h_source_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_index_shape, &h_index_shape, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_target_strides, h_target_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_source_strides, h_source_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_index_strides, &h_index_stride, sizeof(int), cudaMemcpyHostToDevice);
     if (err != cudaSuccess) goto cleanup;
    
    // Dispatch based on data type and index type
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                cuda::index_copy_inplace_kernel<__half, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<__half*>(target),
                    static_cast<const __half*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F32:
                cuda::index_copy_inplace_kernel<float, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<float*>(target),
                    static_cast<const float*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F64:
                cuda::index_copy_inplace_kernel<double, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<double*>(target),
                    static_cast<const double*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BF16:
                cuda::index_copy_inplace_kernel<__nv_bfloat16, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<__nv_bfloat16*>(target),
                    static_cast<const __nv_bfloat16*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I8:
                cuda::index_copy_inplace_kernel<int8_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int8_t*>(target),
                    static_cast<const int8_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I16:
                cuda::index_copy_inplace_kernel<int16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int16_t*>(target),
                    static_cast<const int16_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I32:
                cuda::index_copy_inplace_kernel<int32_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int32_t*>(target),
                    static_cast<const int32_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I64:
                cuda::index_copy_inplace_kernel<int64_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int64_t*>(target),
                    static_cast<const int64_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U8:
                cuda::index_copy_inplace_kernel<uint8_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint8_t*>(target),
                    static_cast<const uint8_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U16:
                cuda::index_copy_inplace_kernel<uint16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint16_t*>(target),
                    static_cast<const uint16_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U32:
                cuda::index_copy_inplace_kernel<uint32_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint32_t*>(target),
                    static_cast<const uint32_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U64:
                cuda::index_copy_inplace_kernel<uint64_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint64_t*>(target),
                    static_cast<const uint64_t*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BOOL:
                cuda::index_copy_inplace_kernel<bool, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<bool*>(target),
                    static_cast<const bool*>(source),
                    static_cast<const int32_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        // Similar dispatch for int64_t index type
        switch (dtype) {
            case INFINI_DTYPE_F16:
                cuda::index_copy_inplace_kernel<__half, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<__half*>(target),
                    static_cast<const __half*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F32:
                cuda::index_copy_inplace_kernel<float, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<float*>(target),
                    static_cast<const float*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F64:
                cuda::index_copy_inplace_kernel<double, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<double*>(target),
                    static_cast<const double*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BF16:
                cuda::index_copy_inplace_kernel<__nv_bfloat16, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<__nv_bfloat16*>(target),
                    static_cast<const __nv_bfloat16*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I8:
                cuda::index_copy_inplace_kernel<int8_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int8_t*>(target),
                    static_cast<const int8_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I16:
                cuda::index_copy_inplace_kernel<int16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int16_t*>(target),
                    static_cast<const int16_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I32:
                cuda::index_copy_inplace_kernel<int32_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int32_t*>(target),
                    static_cast<const int32_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I64:
                cuda::index_copy_inplace_kernel<int64_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<int64_t*>(target),
                    static_cast<const int64_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U8:
                cuda::index_copy_inplace_kernel<uint8_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint8_t*>(target),
                    static_cast<const uint8_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U16:
                cuda::index_copy_inplace_kernel<uint16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint16_t*>(target),
                    static_cast<const uint16_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U32:
                cuda::index_copy_inplace_kernel<uint32_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint32_t*>(target),
                    static_cast<const uint32_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U64:
                cuda::index_copy_inplace_kernel<uint64_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<uint64_t*>(target),
                    static_cast<const uint64_t*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BOOL:
                cuda::index_copy_inplace_kernel<bool, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<bool*>(target),
                    static_cast<const bool*>(source),
                    static_cast<const int64_t*>(index),
                    d_target_shape, d_source_shape, d_index_shape,
                    d_target_strides, d_source_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }
    
    
cleanup:
    // Cleanup device memory
    cudaFree(d_target_shape);
    cudaFree(d_source_shape);
    cudaFree(d_index_shape);
    cudaFree(d_target_strides);
    cudaFree(d_source_strides);
    cudaFree(d_index_strides);
    
    return (err == cudaSuccess) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::index_copy_inplace::nvidia