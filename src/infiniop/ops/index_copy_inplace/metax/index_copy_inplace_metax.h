#ifndef __INDEX_COPY_INPLACE_METAX_H__
#define __INDEX_COPY_INPLACE_METAX_H__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"

namespace op::index_copy_inplace::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    infiniDtype_t _index_dtype;
    std::vector<size_t> _source_shape;
    std::vector<size_t> _target_shape;
    std::vector<size_t> _index_shape;
    std::vector<ptrdiff_t> _source_strides;
    std::vector<ptrdiff_t> _target_strides;
    std::vector<ptrdiff_t> _index_strides;
    int _dim;

public:
    Descriptor() = default;
    Descriptor(infiniDtype_t dtype,
               infiniDtype_t index_dtype,
               const std::vector<size_t> &source_shape,
               const std::vector<size_t> &target_shape,
               const std::vector<size_t> &index_shape,
               const std::vector<ptrdiff_t> &source_strides,
               const std::vector<ptrdiff_t> &target_strides,
               const std::vector<ptrdiff_t> &index_strides,
               int dim,
               infiniDevice_t device,
               int device_id)
        : InfiniopDescriptor{device, device_id},
          _dtype(dtype), _index_dtype(index_dtype),
          _source_shape(source_shape), _target_shape(target_shape), _index_shape(index_shape),
          _source_strides(source_strides), _target_strides(target_strides), _index_strides(index_strides),
          _dim(dim) {}
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t target,
        infiniopTensorDescriptor_t source,
        int dim,
        infiniopTensorDescriptor_t index);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *target,
        const void *source,
        const void *index,
        void *stream) const;

private:
    template <typename T, typename IndexT>
    infiniStatus_t indexCopyInplaceMetax(
        const void *source_data,
        void *target_data,
        const void *index_data,
        void *stream) const;
};

} // namespace op::index_copy_inplace::metax

#endif // __INDEX_COPY_INPLACE_METAX_H__