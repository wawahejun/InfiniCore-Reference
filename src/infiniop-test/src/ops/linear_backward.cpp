#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::linear_backward {
struct Test::Attributes {
    bool has_bias;
    
    std::shared_ptr<Tensor> grad_y;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> grad_x;
    std::shared_ptr<Tensor> grad_w;
    std::shared_ptr<Tensor> grad_b;
    std::shared_ptr<Tensor> ans_grad_x;
    std::shared_ptr<Tensor> ans_grad_w;
    std::shared_ptr<Tensor> ans_grad_b;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    
    // Check required tensors
    if (tensors.find("grad_y") == tensors.end()
        || tensors.find("x") == tensors.end()
        || tensors.find("w") == tensors.end()
        || tensors.find("grad_x") == tensors.end()
        || tensors.find("grad_w") == tensors.end()
        || tensors.find("ans_grad_x") == tensors.end()
        || tensors.find("ans_grad_w") == tensors.end()) {
        throw std::runtime_error("Invalid Test: missing required tensors");
    }
    
    // Check has_bias attribute
    if (attributes.find("has_bias") == attributes.end()) {
        throw std::runtime_error("Invalid Test: missing has_bias attribute");
    }
    
    test->_attributes->has_bias = *reinterpret_cast<const bool*>(attributes["has_bias"].data());
    
    test->_attributes->grad_y = tensors["grad_y"];
    test->_attributes->x = tensors["x"];
    test->_attributes->w = tensors["w"];
    test->_attributes->grad_x = tensors["grad_x"];
    test->_attributes->grad_w = tensors["grad_w"];
    test->_attributes->ans_grad_x = tensors["ans_grad_x"];
    test->_attributes->ans_grad_w = tensors["ans_grad_w"];
    
    // Bias gradients are optional based on has_bias flag
    if (test->_attributes->has_bias) {
        if (tensors.find("grad_b") == tensors.end() || tensors.find("ans_grad_b") == tensors.end()) {
            throw std::runtime_error("Invalid Test: has_bias is true but grad_b or ans_grad_b tensor is missing");
        }
        test->_attributes->grad_b = tensors["grad_b"];
        test->_attributes->ans_grad_b = tensors["ans_grad_b"];
    } else {
        test->_attributes->grad_b = nullptr;
        test->_attributes->ans_grad_b = nullptr;
    }
    
    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    
    auto grad_y = _attributes->grad_y->to(device, device_id);
    auto x = _attributes->x->to(device, device_id);
    auto w = _attributes->w->to(device, device_id);
    auto grad_x = _attributes->grad_x->to(device, device_id);
    auto grad_w = _attributes->grad_w->to(device, device_id);
    
    std::shared_ptr<Tensor> grad_b_device = nullptr;
    if (_attributes->has_bias && _attributes->grad_b) {
        grad_b_device = _attributes->grad_b->to(device, device_id);
    }
        
    // Create linear backward descriptor
    infiniopLinearBackwardDescriptor_t op_desc;
    auto status = infiniopCreateLinearBackwardDescriptor(handle, &op_desc,
                                                       grad_y->desc(),
                                                       x->desc(),
                                                       w->desc(),
                                                       grad_x->desc(),
                                                       grad_w->desc(),
                                                       grad_b_device ? grad_b_device->desc() : nullptr);
    if (status != INFINI_STATUS_SUCCESS) {
        return TEST_FAILED(OP_CREATION_FAILED, "Failed to create linear backward descriptor.");
    }

    // Get workspace size
    size_t workspace_size;
    status = infiniopGetLinearBackwardWorkspaceSize(op_desc, &workspace_size);
    if (status != INFINI_STATUS_SUCCESS) {
        infiniopDestroyLinearBackwardDescriptor(op_desc);
        return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size.");
    }
    
    // Allocate workspace
    void *workspace = nullptr;
    if (workspace_size > 0) {
        auto status_malloc = infinirtMalloc(&workspace, workspace_size);
        if (status_malloc != INFINI_STATUS_SUCCESS) {
            infiniopDestroyLinearBackwardDescriptor(op_desc);
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace.");
        }
    }
    
    // Execute linear backward operation
    status = infiniopLinearBackward(op_desc, workspace, workspace_size,
                                   grad_x->data(),
                                   grad_w->data(),
                                   grad_b_device ? grad_b_device->data() : nullptr,
                                   grad_y->data(),
                                   x->data(),
                                   w->data(),
                                   nullptr);
    
    if (status != INFINI_STATUS_SUCCESS) {
        if (workspace) infinirtFree(workspace);
        infiniopDestroyLinearBackwardDescriptor(op_desc);
        return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution.");
    }

    // Check correctness for grad_x
    try {
        allClose(grad_x, _attributes->ans_grad_x, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) infinirtFree(workspace);
        infiniopDestroyLinearBackwardDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, std::string("grad_x mismatch: ") + e.what());
    }
    
    // Check correctness for grad_w
    try {
        allClose(grad_w, _attributes->ans_grad_w, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) infinirtFree(workspace);
        infiniopDestroyLinearBackwardDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, std::string("grad_w mismatch: ") + e.what());
    }
    
    // Check correctness for grad_b if has_bias
    if (_attributes->has_bias && grad_b_device && _attributes->ans_grad_b) {
        try {
            allClose(grad_b_device, _attributes->ans_grad_b, _rtol, _atol);
        } catch (const std::exception &e) {
            if (workspace) infinirtFree(workspace);
            infiniopDestroyLinearBackwardDescriptor(op_desc);
            return TEST_FAILED(RESULT_INCORRECT, std::string("grad_b mismatch: ") + e.what());
        }
    }

    // Benchmark
     double elapsed_time = benchmark(
         [op_desc, workspace, workspace_size, grad_x, grad_w, grad_b_device, grad_y, x, w]() {
             infiniopLinearBackward(
                 op_desc, workspace, workspace_size,
                 grad_x->data(),
                 grad_w->data(),
                 grad_b_device ? grad_b_device->data() : nullptr,
                 grad_y->data(),
                 x->data(),
                 w->data(),
                 nullptr);
         },
         warm_ups, iterations);
    
    // Cleanup
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyLinearBackwardDescriptor(op_desc);
    
    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"has_bias"};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_y", "x", "w", "grad_x", "grad_w", "grad_b", "ans_grad_x", "ans_grad_w", "ans_grad_b"};
}

std::vector<std::string> Test::output_names() {
    return {"grad_x", "grad_w", "grad_b"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "linear_backward(";
    oss << "grad_y=" << _attributes->grad_y->info();
    oss << ", x=" << _attributes->x->info();
    oss << ", w=" << _attributes->w->info();
    oss << ", grad_x=" << _attributes->grad_x->info();
    oss << ", grad_w=" << _attributes->grad_w->info();
    if (_attributes->has_bias && _attributes->grad_b) {
        oss << ", grad_b=" << _attributes->grad_b->info();
    }
    oss << ", has_bias=" << (_attributes->has_bias ? "true" : "false");
    oss << ", rtol=" << _rtol << ", atol=" << _atol;
    oss << ")";
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::linear_backward