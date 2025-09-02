#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::linear {
struct Test::Attributes {
    bool has_bias;
    
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    
    // Check required tensors
    if (tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("output") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test: missing required tensors");
    }
    
    // Check has_bias attribute
    if (attributes.find("has_bias") == attributes.end()) {
        throw std::runtime_error("Invalid Test: missing has_bias attribute");
    }
    
    test->_attributes->has_bias = *reinterpret_cast<const bool*>(attributes["has_bias"].data());
    
    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->output = tensors["output"];
    test->_attributes->ans = tensors["ans"];
    
    // Bias is optional based on has_bias flag
    if (test->_attributes->has_bias) {
        if (tensors.find("bias") == tensors.end()) {
            throw std::runtime_error("Invalid Test: has_bias is true but bias tensor is missing");
        }
        test->_attributes->bias = tensors["bias"];
    } else {
        test->_attributes->bias = nullptr;
    }
    
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    try {
        infiniopLinearDescriptor_t op_desc;
        auto input = _attributes->input->to(device, device_id);
        auto weight = _attributes->weight->to(device, device_id);
        auto output = _attributes->output->to(device, device_id);
        
        std::shared_ptr<Tensor> bias_device = nullptr;
        if (_attributes->has_bias && _attributes->bias) {
            bias_device = _attributes->bias->to(device, device_id);
        }
        
        // Create linear descriptor
        auto status = infiniopCreateLinearDescriptor(handle, &op_desc,
                                               input->desc(),
                                               weight->desc(),
                                               bias_device ? bias_device->desc() : nullptr,
                                               output->desc());
        if (status != INFINI_STATUS_SUCCESS) {
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to create linear descriptor.");
        }
    
        // Get workspace size
        size_t workspace_size;
        status = infiniopGetLinearWorkspaceSize(op_desc, &workspace_size);
        if (status != INFINI_STATUS_SUCCESS) {
            infiniopDestroyLinearDescriptor(op_desc);
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size.");
        }
        
        // Allocate workspace
        void *workspace = nullptr;
        if (workspace_size > 0) {
            auto status_malloc = infinirtMalloc(&workspace, workspace_size);
            if (status_malloc != INFINI_STATUS_SUCCESS) {
                infiniopDestroyLinearDescriptor(op_desc);
                return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace.");
            }
        }
        
        // Execute linear operation
        status = infiniopLinear(op_desc, workspace, workspace_size,
                               output->data(),
                               input->data(),
                               weight->data(),
                               bias_device ? bias_device->data() : nullptr,
                               nullptr);
        if (status != INFINI_STATUS_SUCCESS) {
            if (workspace) infinirtFree(workspace);
            infiniopDestroyLinearDescriptor(op_desc);
            return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution.");
        }
    
        // Check correctness
        try {
            allClose(output, _attributes->ans, _rtol, _atol);
        } catch (const std::exception &e) {
            if (workspace) infinirtFree(workspace);
            infiniopDestroyLinearDescriptor(op_desc);
            return TEST_FAILED(RESULT_INCORRECT, e.what());
        }
        
        // Benchmark
        double elapsed_time = benchmark(
            [op_desc, workspace, workspace_size, output, input, weight, bias_device]() {
                infiniopLinear(
                    op_desc, workspace, workspace_size,
                    output->data(),
                    input->data(),
                    weight->data(),
                    bias_device ? bias_device->data() : nullptr,
                    nullptr);
            },
            warm_ups, iterations);
        
        // Cleanup
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyLinearDescriptor(op_desc);
        
        return TEST_PASSED(elapsed_time);
    } catch (const std::exception &e) {
        return TEST_FAILED(OP_EXECUTION_FAILED, std::string("Exception during execution: ") + e.what());
    }
}

std::vector<std::string> Test::attribute_names() {
    return {"has_bias"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "weight", "bias", "output", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"output"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "linear(";
    oss << "input=" << _attributes->input->info();
    oss << ", weight=" << _attributes->weight->info();
    if (_attributes->has_bias && _attributes->bias) {
        oss << ", bias=" << _attributes->bias->info();
    }
    oss << ", output=" << _attributes->output->info();
    oss << ", has_bias=" << (_attributes->has_bias ? "true" : "false");
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::linear