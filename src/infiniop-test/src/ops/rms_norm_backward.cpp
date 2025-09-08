#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::rms_norm_backward {
struct Test::Attributes {
    float epsilon;
    
    std::shared_ptr<Tensor> grad_output;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> grad_input;
    std::shared_ptr<Tensor> grad_weight;
    std::shared_ptr<Tensor> ans_grad_input;
    std::shared_ptr<Tensor> ans_grad_weight;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    
    // Check required tensors
    if (tensors.find("grad_output") == tensors.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("grad_input") == tensors.end()
        || tensors.find("grad_weight") == tensors.end()
        || tensors.find("ans_grad_input") == tensors.end()
        || tensors.find("ans_grad_weight") == tensors.end()) {
        throw std::runtime_error("Invalid Test: missing required tensors");
    }
    
    // Check required attributes
    if (attributes.find("epsilon") == attributes.end()) {
        throw std::runtime_error("Invalid Test: missing epsilon attribute");
    }
    
    test->_attributes->epsilon = *reinterpret_cast<const float*>(attributes["epsilon"].data());
    
    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->grad_input = tensors["grad_input"];
    test->_attributes->grad_weight = tensors["grad_weight"];
    test->_attributes->ans_grad_input = tensors["ans_grad_input"];
    test->_attributes->ans_grad_weight = tensors["ans_grad_weight"];
    
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, 
    size_t warm_ups, size_t iterations) {
    
    // Move tensors to device
    auto grad_output = _attributes->grad_output->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto grad_input = _attributes->grad_input->to(device, device_id);
    auto grad_weight = _attributes->grad_weight->to(device, device_id);
    
    // Create RMS Norm backward descriptor
    infiniopRMSNormBackwardDescriptor_t op_desc;
    auto status = infiniopCreateRMSNormBackwardDescriptor(handle, &op_desc,
                                                        grad_input->desc(),
                                                        grad_weight->desc(),
                                                        grad_output->desc(),
                                                        input->desc(),
                                                        weight->desc(),
                                                        _attributes->epsilon);
    if (status != INFINI_STATUS_SUCCESS) {
        return TEST_FAILED(OP_CREATION_FAILED, "Failed to create RMS Norm backward descriptor.");
    }
    
    // Get workspace size
    size_t workspace_size;
    status = infiniopGetRMSNormBackwardWorkspaceSize(op_desc, &workspace_size);
    if (status != INFINI_STATUS_SUCCESS) {
        infiniopDestroyRMSNormBackwardDescriptor(op_desc);
        return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size.");
    }
    
    // Allocate workspace
    void *workspace = nullptr;
    if (workspace_size > 0) {
        auto status_malloc = infinirtMalloc(&workspace, workspace_size);
        if (status_malloc != INFINI_STATUS_SUCCESS) {
            infiniopDestroyRMSNormBackwardDescriptor(op_desc);
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace.");
        }
    }
    
    // Execute RMS Norm backward operation
    status = infiniopRMSNormBackward(op_desc, workspace, workspace_size,
                                   grad_input->data(),
                                   grad_weight->data(),
                                   grad_output->data(),
                                   input->data(),
                                   weight->data(),
                                   nullptr);
    
    if (status != INFINI_STATUS_SUCCESS) {
        if (workspace) infinirtFree(workspace);
        infiniopDestroyRMSNormBackwardDescriptor(op_desc);
        return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during RMS Norm backward execution.");
    }
    
    // Check correctness for grad_input
    try {
        allClose(grad_input, _attributes->ans_grad_input, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) infinirtFree(workspace);
        infiniopDestroyRMSNormBackwardDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, std::string("grad_input mismatch: ") + e.what());
    }
    
    // Check correctness for grad_weight
    try {
        allClose(grad_weight, _attributes->ans_grad_weight, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) infinirtFree(workspace);
        infiniopDestroyRMSNormBackwardDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, std::string("grad_weight mismatch: ") + e.what());
    }
    
    // Benchmark
    double elapsed_time = benchmark(
        [op_desc, workspace, workspace_size, grad_input, grad_weight, grad_output, input, weight]() {
            infiniopRMSNormBackward(
                op_desc, workspace, workspace_size,
                grad_input->data(),
                grad_weight->data(),
                grad_output->data(),
                input->data(),
                weight->data(),
                nullptr);
        },
        warm_ups, iterations);
    
    // Cleanup
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyRMSNormBackwardDescriptor(op_desc);
    
    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"epsilon"};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_output", "input", "weight", "grad_input", "grad_weight", "ans_grad_input", "ans_grad_weight"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_input", "grad_weight"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "rms_norm_backward(";
    oss << "grad_output=" << _attributes->grad_output->info();
    oss << ", input=" << _attributes->input->info();
    oss << ", weight=" << _attributes->weight->info();
    oss << ", grad_input=" << _attributes->grad_input->info();
    oss << ", grad_weight=" << _attributes->grad_weight->info();
    oss << ", epsilon=" << _attributes->epsilon;
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::rms_norm_backward