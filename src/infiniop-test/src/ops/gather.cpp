#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include "infiniop/ops/gather.h"
#include <iomanip>
#include <iostream>

namespace infiniop_test::gather {
struct Test::Attributes {
    std::shared_ptr<Tensor> input, index, output, ans;
    std::vector<uint8_t> dim;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }
    
    test->_attributes->input = tensors["input"];
    test->_attributes->index = tensors["index"];
    test->_attributes->output = tensors["output"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->dim = attributes["dim"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    
    try {
        auto input = _attributes->input->to(device, device_id);
        auto index = _attributes->index->to(device, device_id);
        auto output = _attributes->output->to(device, device_id);
        auto ans = _attributes->ans->to(device, device_id);
        
        // 创建gather描述符
        infiniopGatherDescriptor_t gather_desc;
        int dim = static_cast<int>(_attributes->dim[0]);
        auto status = infiniopCreateGatherDescriptor(handle, &gather_desc, input->desc(), output->desc(), dim, index->desc());
        if (status != INFINI_STATUS_SUCCESS) {
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to create gather descriptor.");
        }
        
        // 获取工作空间大小
        size_t workspace_size;
        status = infiniopGetGatherWorkspaceSize(gather_desc, &workspace_size);
        if (status != INFINI_STATUS_SUCCESS) {
            infiniopDestroyGatherDescriptor(gather_desc);
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size.");
        }
        
        // 分配工作空间
        void* workspace = nullptr;
        if (workspace_size > 0) {
            auto status_malloc = infinirtMalloc(&workspace, workspace_size);
            if (status_malloc != INFINI_STATUS_SUCCESS) {
                infiniopDestroyGatherDescriptor(gather_desc);
                return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace.");
            }
        }
        
        // 执行gather操作
         status = infiniopGather(gather_desc, workspace, workspace_size, 
                               output->data(), input->data(), index->data(), nullptr);
         
         if (status != INFINI_STATUS_SUCCESS) {
             if (workspace) infinirtFree(workspace);
             infiniopDestroyGatherDescriptor(gather_desc);
             return TEST_FAILED(OP_EXECUTION_FAILED, "Gather operation failed.");
         }
         
         // 检查结果
         try {
             allClose(output, ans, _rtol, _atol);
         } catch (const std::exception &e) {
             if (workspace) infinirtFree(workspace);
             infiniopDestroyGatherDescriptor(gather_desc);
             return TEST_FAILED(RESULT_INCORRECT, e.what());
         }
         
         // 性能测试
         double elapsed_time = benchmark(
             [=]() {
                 infiniopGather(gather_desc, workspace, workspace_size, 
                               output->data(), input->data(), index->data(), nullptr);
             },
             warm_ups, iterations);
         
         // 清理资源
         if (workspace) infinirtFree(workspace);
         infiniopDestroyGatherDescriptor(gather_desc);
         
         return TEST_PASSED(elapsed_time);
        
    } catch (const std::exception& e) {
        return TEST_FAILED(OP_EXECUTION_FAILED, std::string("Exception: ") + e.what());
    }
}

std::vector<std::string> Test::attribute_names() {
    return {"dim"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "index", "output", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"output"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << "Gather Test:" << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- index: " << _attributes->index->info() << std::endl;
    oss << "- output: " << _attributes->output->info() << std::endl;
    oss << "  Dim: " << *reinterpret_cast<const uint32_t*>(_attributes->dim.data()) << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::gather