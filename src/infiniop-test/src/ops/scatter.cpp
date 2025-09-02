#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::scatter {
struct Test::Attributes {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> index;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> ans;
    std::vector<uint8_t> dim;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("input") == tensors.end()
        || tensors.find("index") == tensors.end()
        || tensors.find("src") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    if (attributes.find("dim") == attributes.end()) {
        throw std::runtime_error("Missing dim attribute");
    }

    test->_attributes->input = tensors["input"];
    test->_attributes->index = tensors["index"];
    test->_attributes->src = tensors["src"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->dim = attributes["dim"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopScatterDescriptor_t op_desc;
    auto input = _attributes->input->to(device, device_id);
    auto index = _attributes->index->to(device, device_id);
    auto src = _attributes->src->to(device, device_id);
    
    // Extract dim value from attributes
    int32_t dim_value = 0;
    if (_attributes->dim.size() >= sizeof(int32_t)) {
        dim_value = *reinterpret_cast<const int32_t*>(_attributes->dim.data());
    }
    
    CHECK_OR(infiniopCreateScatterDescriptor(handle, &op_desc,
                                            input->desc(),
                                            input->desc(), // output same as input for in-place operation
                                            index->desc(),
                                            src->desc(),
                                            dim_value),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create scatter descriptor."));
    
    size_t workspace_size;
    CHECK_OR(infiniopGetScatterWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }
    
    CHECK_OR(infiniopScatter(op_desc, workspace, workspace_size,
                            input->data(), // output
                            input->data(), // input
                            index->data(),
                            src->data(),
                            nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(input, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopScatter(
                op_desc, workspace, workspace_size,
                input->data(), // output
                input->data(), // input
                index->data(),
                src->data(),
                nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyScatterDescriptor(op_desc);
    if (workspace) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"dim"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "index", "src", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- index: " << _attributes->index->info() << std::endl;
    oss << "- src: " << _attributes->src->info() << std::endl;
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

} // namespace infiniop_test::scatter