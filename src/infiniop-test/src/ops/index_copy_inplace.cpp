#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::index_copy_inplace {
struct Test::Attributes {
    std::shared_ptr<Tensor> target;
    std::shared_ptr<Tensor> source;
    std::shared_ptr<Tensor> index;
    std::shared_ptr<Tensor> ans;
    std::vector<uint8_t> dim;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("target") == tensors.end()
        || tensors.find("source") == tensors.end()
        || tensors.find("index") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    if (attributes.find("dim") == attributes.end()) {
        throw std::runtime_error("Missing dim attribute");
    }

    test->_attributes->target = tensors["target"];
    test->_attributes->source = tensors["source"];
    test->_attributes->index = tensors["index"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->dim = attributes["dim"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopIndexCopyInplaceDescriptor_t op_desc;
    auto target = _attributes->target->to(device, device_id);
    auto source = _attributes->source->to(device, device_id);
    auto index = _attributes->index->to(device, device_id);
    

    
    // Extract dim value from attributes
    int32_t dim_value = 0;
    if (_attributes->dim.size() >= sizeof(int32_t)) {
        dim_value = *reinterpret_cast<const int32_t*>(_attributes->dim.data());
    }
    
    CHECK_OR(infiniopCreateIndexCopyInplaceDescriptor(handle, &op_desc,
                                                     target->desc(),
                                                     source->desc(),
                                                     dim_value,
                                                     index->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create index_copy_inplace descriptor."));
    
    size_t workspace_size;
    CHECK_OR(infiniopGetIndexCopyInplaceWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }
    
    CHECK_OR(infiniopIndexCopyInplace(op_desc, workspace, workspace_size,
                                     target->data(),
                                     source->data(),
                                     index->data(),
                                     nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(target, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopIndexCopyInplace(
                op_desc, workspace, workspace_size,
                target->data(),
                source->data(),
                index->data(),
                nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyIndexCopyInplaceDescriptor(op_desc);
    if (workspace) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"dim"};
}

std::vector<std::string> Test::tensor_names() {
    return {"target", "source", "index", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- target: " << _attributes->target->info() << std::endl;
    oss << "- source: " << _attributes->source->info() << std::endl;
    oss << "- index: " << _attributes->index->info() << std::endl;
    if (_attributes->dim.size() >= sizeof(int32_t)) {
        int32_t dim_value = *reinterpret_cast<const int32_t*>(_attributes->dim.data());
        oss << "- dim: " << dim_value << std::endl;
    }
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::index_copy_inplace