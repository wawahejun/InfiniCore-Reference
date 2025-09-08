#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>
#include <chrono>

namespace infiniop_test::layer_norm {
struct Test::Attributes {
    float eps;
    bool has_bias;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> input_std_deviation;
    std::shared_ptr<Tensor> input_standardization;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("eps") == attributes.end()
        || attributes.find("has_bias") == attributes.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("output") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->eps = *reinterpret_cast<float *>(attributes["eps"].data());
    test->_attributes->has_bias = *reinterpret_cast<bool *>(attributes["has_bias"].data());

    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->output = tensors["output"];
    
    // 只有当has_bias为true时才检查和设置bias张量
    if (test->_attributes->has_bias) {
        if (tensors.find("bias") == tensors.end()) {
            throw std::runtime_error("Invalid Test: Missing bias tensor when has_bias is true");
        }
        test->_attributes->bias = tensors["bias"];
    } else {
        test->_attributes->bias = nullptr;
    }
    
    // 创建临时的std_deviation和standardization张量
    auto input_shape = test->_attributes->input->shape();
    auto input_dtype = test->_attributes->input->ggml_type();
    
    // input_std_deviation: 形状为batch维度，即去掉最后的normalized维度
    std::vector<size_t> std_dev_shape;
    for (size_t i = 0; i < input_shape.size() - test->_attributes->weight->shape().size(); ++i) {
        std_dev_shape.push_back(input_shape[i]);
    }
    if (std_dev_shape.empty()) std_dev_shape.push_back(1);
    
    // 创建连续的步长
    std::vector<ptrdiff_t> std_dev_strides(std_dev_shape.size());
    if (!std_dev_shape.empty()) {
        std_dev_strides[std_dev_shape.size() - 1] = 1;
        for (int i = std_dev_shape.size() - 2; i >= 0; --i) {
            std_dev_strides[i] = std_dev_strides[i + 1] * std_dev_shape[i + 1];
        }
    }
    
    // 计算内存大小
    size_t std_dev_size = 1;
    for (auto dim : std_dev_shape) std_dev_size *= dim;
    
    size_t input_size = 1;
    for (auto dim : input_shape) input_size *= dim;
    
    // 创建内存
    auto std_dev_memory = std::make_shared<Memory>(std_dev_size * ggmlTypeSize(input_dtype), INFINI_DEVICE_CPU, 0);
    auto standardization_memory = std::make_shared<Memory>(input_size * ggmlTypeSize(input_dtype), INFINI_DEVICE_CPU, 0);
    
    test->_attributes->input_std_deviation = std::make_shared<Tensor>(
        std_dev_memory, 0, std_dev_shape, std_dev_strides, input_dtype);
    test->_attributes->input_standardization = std::make_shared<Tensor>(
        standardization_memory, 0, input_shape, test->_attributes->input->strides(), input_dtype);

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopLayerNormDescriptor_t op_desc;
    infiniopTensorDescriptor_t bias_desc = _attributes->has_bias ? _attributes->bias->desc() : nullptr;
    CHECK_OR(infiniopCreateLayerNormDescriptor(handle, &op_desc,
                                               _attributes->output->desc(),
                                               _attributes->input->desc(),
                                               _attributes->weight->desc(),
                                               bias_desc,
                                               _attributes->input_std_deviation->desc(),
                                               _attributes->input_standardization->desc(),
                                               _attributes->eps),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create LayerNorm descriptor"));

    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto bias = _attributes->has_bias ? _attributes->bias->to(device, device_id) : nullptr;
    auto output = _attributes->output->to(device, device_id);
    auto input_std_deviation = _attributes->input_std_deviation->to(device, device_id);
    auto input_standardization = _attributes->input_standardization->to(device, device_id);

    size_t workspace_size;
    CHECK_OR(infiniopGetLayerNormWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    void* bias_data = _attributes->has_bias ? bias->data() : nullptr;
    CHECK_OR(infiniopLayerNorm(op_desc,
                               workspace, workspace_size,
                               output->data(),
                               input->data(),
                               weight->data(),
                               bias_data,
                               input_std_deviation->data(),
                               input_standardization->data(),
                               nullptr), // stream
             return TEST_FAILED(OP_EXECUTION_FAILED, "LayerNorm execution failed"));

    try {
        allClose(output, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopLayerNorm(
                op_desc, workspace, workspace_size,
                output->data(),
                input->data(),
                weight->data(),
                bias_data,
                input_std_deviation->data(),
                input_standardization->data(),
                nullptr);
        },
        warm_ups, iterations);

    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyLayerNormDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"eps", "has_bias", "normalized_shape"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "weight", "bias", "ans", "output"};
}

std::vector<std::string> Test::output_names() {
    return {"output"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "LayerNorm Test:\n";
    oss << "  eps: " << _attributes->eps << "\n";
    oss << "  has_bias: " << (_attributes->has_bias ? "true" : "false") << "\n";
    oss << "  normalized_shape: [";
    // Use weight shape as normalized_shape
    auto weight_shape = _attributes->weight->shape();
    for (size_t i = 0; i < weight_shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << weight_shape[i];
    }
    oss << "]\n";
    // Helper lambda to format shape
    auto format_shape = [](const std::vector<size_t>& shape) {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << shape[i];
        }
        ss << "]";
        return ss.str();
    };
    
    oss << "  input shape: " << format_shape(_attributes->input->shape()) << "\n";
    oss << "  weight shape: " << format_shape(_attributes->weight->shape()) << "\n";
    if (_attributes->has_bias && _attributes->bias) {
        oss << "  bias shape: " << format_shape(_attributes->bias->shape()) << "\n";
    } else {
        oss << "  bias: none\n";
    }
    oss << "  output shape: " << format_shape(_attributes->output->shape());
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::layer_norm