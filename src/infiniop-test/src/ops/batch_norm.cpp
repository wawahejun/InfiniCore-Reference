#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>
#include <chrono>

namespace infiniop_test::batch_norm {
struct Test::Attributes {
    float eps;
    bool training;
    float momentum;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> running_mean;
    std::shared_ptr<Tensor> running_var;
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> output;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("eps") == attributes.end()
        || attributes.find("training") == attributes.end()
        || attributes.find("momentum") == attributes.end()
        || tensors.find("input") == tensors.end()
        || tensors.find("weight") == tensors.end()
        || tensors.find("bias") == tensors.end()
        || tensors.find("running_mean") == tensors.end()
        || tensors.find("running_var") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("output") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->eps = *reinterpret_cast<float *>(attributes["eps"].data());
    test->_attributes->training = *reinterpret_cast<bool *>(attributes["training"].data());
    test->_attributes->momentum = *reinterpret_cast<float *>(attributes["momentum"].data());

    test->_attributes->input = tensors["input"];
    test->_attributes->weight = tensors["weight"];
    test->_attributes->bias = tensors["bias"];
    test->_attributes->running_mean = tensors["running_mean"];
    test->_attributes->running_var = tensors["running_var"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->output = tensors["output"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopBatchNormDescriptor_t op_desc;
    CHECK_OR(infiniopCreateBatchNormDescriptor(handle, &op_desc,
                                               _attributes->output->desc(),
                                               _attributes->input->desc(),
                                               _attributes->weight->desc(),
                                               _attributes->bias->desc(),
                                               _attributes->running_mean->desc(),
                                               _attributes->running_var->desc(),
                                               _attributes->momentum,
                                               _attributes->eps),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create BatchNorm descriptor"));

    auto input = _attributes->input->to(device, device_id);
    auto weight = _attributes->weight->to(device, device_id);
    auto bias = _attributes->bias->to(device, device_id);
    auto running_mean = _attributes->running_mean->to(device, device_id);
    auto running_var = _attributes->running_var->to(device, device_id);
    auto output = _attributes->output->to(device, device_id);

    size_t workspace_size;
    CHECK_OR(infiniopGetBatchNormWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopBatchNorm(op_desc,
                               workspace, workspace_size,
                               output->data(),
                               input->data(),
                               weight->data(),
                               bias->data(),
                               running_mean->data(),
                               running_var->data(),
                               nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "BatchNorm execution failed"));

    try {
        allClose(output, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopBatchNorm(
                op_desc, workspace, workspace_size,
                output->data(),
                input->data(),
                weight->data(),
                bias->data(),
                running_mean->data(),
                running_var->data(),
                nullptr);
        },
        warm_ups, iterations);

    if (workspace) {
        infinirtFree(workspace);
    }
    // Note: BatchNorm descriptor cleanup is handled automatically

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"eps", "training", "momentum"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "weight", "bias", "running_mean", "running_var", "ans", "output"};
}

std::vector<std::string> Test::output_names() {
    return {"output"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "BatchNorm Test:\n";
    oss << "  eps: " << _attributes->eps << "\n";
    oss << "  training: " << (_attributes->training ? "true" : "false") << "\n";
    oss << "  momentum: " << _attributes->momentum << "\n";
    oss << "- input: " << _attributes->input->info() << "\n";
    oss << "- weight: " << _attributes->weight->info() << "\n";
    oss << "- bias: " << _attributes->bias->info() << "\n";
    oss << "- running_mean: " << _attributes->running_mean->info() << "\n";
    oss << "- running_var: " << _attributes->running_var->info() << "\n";
    oss << "- output: " << _attributes->output->info();
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::batch_norm