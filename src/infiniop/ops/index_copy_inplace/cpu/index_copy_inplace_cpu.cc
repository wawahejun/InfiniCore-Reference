#include "index_copy_inplace_cpu.h"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <cstring>
#include <cstdio>

namespace op::index_copy_inplace::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t source_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = target_desc->dtype();

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
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (i != static_cast<size_t>(dim) && target_shape[i] != source_shape[i]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    auto desc = new Descriptor();
    desc->_target_desc = target_desc;
    desc->_source_desc = source_desc;
    desc->_index_desc = index_desc;
    desc->_dim = dim;
    desc->_handle = handle;
    desc->device_type = INFINI_DEVICE_CPU;
    desc->device_id = handle->device_id;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

template<typename T, typename IndexT>
static void index_copy_inplace_kernel(
    const T *source_data,
    T *target_data,
    const IndexT *index_data,
    const std::vector<size_t> &source_shape,
    const std::vector<size_t> &target_shape,
    const std::vector<size_t> &index_shape,
    const std::vector<ptrdiff_t> &source_strides,
    const std::vector<ptrdiff_t> &target_strides,
    const std::vector<ptrdiff_t> &index_strides,
    int dim) {

    // Process each element in the source tensor along the specified dimension
    // For each source element at position i, copy it to target[index[i]]
    // Note: When duplicate indices exist, this implementation follows "last value wins" strategy
    // This matches PyTorch's CPU behavior for index_copy_ with duplicate indices
    for (size_t source_idx = 0; source_idx < source_shape[dim]; ++source_idx) {
        // Get the target index for this source element
        IndexT target_idx = index_data[source_idx];
        
        // Check bounds for target index
        if (target_idx < 0 || target_idx >= static_cast<IndexT>(target_shape[dim])) {
            continue;
        }
        
        // Calculate the number of elements in each slice (excluding the dim dimension)
        size_t slice_size = 1;
        for (size_t d = 0; d < source_shape.size(); ++d) {
            if (d != static_cast<size_t>(dim)) {
                slice_size *= source_shape[d];
            }
        }
        
        // Copy all elements in this slice
        for (size_t elem_in_slice = 0; elem_in_slice < slice_size; ++elem_in_slice) {
            // Calculate coordinates for this element within the slice
            std::vector<size_t> source_coords(source_shape.size());
            source_coords[dim] = source_idx;  // Set the dim coordinate for source
            
            size_t temp = elem_in_slice;
            
            // Fill coordinates for all dimensions except dim
            for (int d = source_shape.size() - 1; d >= 0; --d) {
                if (d != dim) {
                    size_t dim_size = source_shape[d];
                    source_coords[d] = temp % dim_size;
                    temp /= dim_size;
                }
            }
            
            // Calculate target coordinates
            std::vector<size_t> target_coords = source_coords;
            target_coords[dim] = static_cast<size_t>(target_idx);
            
            // Calculate source offset - strides are in elements
            ptrdiff_t source_offset = 0;
            for (size_t d = 0; d < source_coords.size(); ++d) {
                source_offset += source_coords[d] * source_strides[d];
            }
            
            // Calculate target offset - strides are in elements
            ptrdiff_t target_offset = 0;
            for (size_t d = 0; d < target_coords.size(); ++d) {
                target_offset += target_coords[d] * target_strides[d];
            }
            
            // Copy the value: target[index[i]] = source[i]
            target_data[target_offset] = source_data[source_offset];
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *target,
    const void *source,
    const void *index,
    void *stream) const {


    
    // Check if descriptors are valid pointers
    if (!_target_desc || !_source_desc || !_index_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (!target || !source || !index) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto target_shape = _target_desc->shape();
    auto source_shape = _source_desc->shape();
    auto index_shape = _index_desc->shape();
    auto target_strides = _target_desc->strides();
    auto source_strides = _source_desc->strides();
    auto index_strides = _index_desc->strides();
    auto dtype = _target_desc->dtype();
    auto index_dtype = _index_desc->dtype();

    // Dispatch based on data type and index type
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                index_copy_inplace_kernel<uint16_t, int32_t>(
                    static_cast<const uint16_t*>(source),
                    static_cast<uint16_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F32:
                index_copy_inplace_kernel<float, int32_t>(
                    static_cast<const float*>(source),
                    static_cast<float*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F64:
                index_copy_inplace_kernel<double, int32_t>(
                    static_cast<const double*>(source),
                    static_cast<double*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BF16:
                index_copy_inplace_kernel<uint16_t, int32_t>(
                    static_cast<const uint16_t*>(source),
                    static_cast<uint16_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I8:
                index_copy_inplace_kernel<int8_t, int32_t>(
                    static_cast<const int8_t*>(source),
                    static_cast<int8_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I16:
                index_copy_inplace_kernel<int16_t, int32_t>(
                    static_cast<const int16_t*>(source),
                    static_cast<int16_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I32:
                index_copy_inplace_kernel<int32_t, int32_t>(
                    static_cast<const int32_t*>(source),
                    static_cast<int32_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I64:
                index_copy_inplace_kernel<int64_t, int32_t>(
                    static_cast<const int64_t*>(source),
                    static_cast<int64_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U8:
                index_copy_inplace_kernel<uint8_t, int32_t>(
                    static_cast<const uint8_t*>(source),
                    static_cast<uint8_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U16:
                index_copy_inplace_kernel<uint16_t, int32_t>(
                    static_cast<const uint16_t*>(source),
                    static_cast<uint16_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U32:
                index_copy_inplace_kernel<uint32_t, int32_t>(
                    static_cast<const uint32_t*>(source),
                    static_cast<uint32_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U64:
                index_copy_inplace_kernel<uint64_t, int32_t>(
                    static_cast<const uint64_t*>(source),
                    static_cast<uint64_t*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BOOL:
                index_copy_inplace_kernel<bool, int32_t>(
                    static_cast<const bool*>(source),
                    static_cast<bool*>(target),
                    static_cast<const int32_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                index_copy_inplace_kernel<uint16_t, int64_t>(
                    static_cast<const uint16_t*>(source),
                    static_cast<uint16_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F32:
                index_copy_inplace_kernel<float, int64_t>(
                    static_cast<const float*>(source),
                    static_cast<float*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F64:
                index_copy_inplace_kernel<double, int64_t>(
                    static_cast<const double*>(source),
                    static_cast<double*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BF16:
                index_copy_inplace_kernel<uint16_t, int64_t>(
                    static_cast<const uint16_t*>(source),
                    static_cast<uint16_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I8:
                index_copy_inplace_kernel<int8_t, int64_t>(
                    static_cast<const int8_t*>(source),
                    static_cast<int8_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I16:
                index_copy_inplace_kernel<int16_t, int64_t>(
                    static_cast<const int16_t*>(source),
                    static_cast<int16_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I32:
                index_copy_inplace_kernel<int32_t, int64_t>(
                    static_cast<const int32_t*>(source),
                    static_cast<int32_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I64:
                index_copy_inplace_kernel<int64_t, int64_t>(
                    static_cast<const int64_t*>(source),
                    static_cast<int64_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U8:
                index_copy_inplace_kernel<uint8_t, int64_t>(
                    static_cast<const uint8_t*>(source),
                    static_cast<uint8_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U16:
                index_copy_inplace_kernel<uint16_t, int64_t>(
                    static_cast<const uint16_t*>(source),
                    static_cast<uint16_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U32:
                index_copy_inplace_kernel<uint32_t, int64_t>(
                    static_cast<const uint32_t*>(source),
                    static_cast<uint32_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U64:
                index_copy_inplace_kernel<uint64_t, int64_t>(
                    static_cast<const uint64_t*>(source),
                    static_cast<uint64_t*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BOOL:
                index_copy_inplace_kernel<bool, int64_t>(
                    static_cast<const bool*>(source),
                    static_cast<bool*>(target),
                    static_cast<const int64_t*>(index),
                    source_shape, target_shape, index_shape,
                    source_strides, target_strides, index_strides,
                    _dim);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::index_copy_inplace::cpu