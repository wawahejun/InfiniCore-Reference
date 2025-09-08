import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def batch_norm(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
    training: bool = False,
    momentum: float = 0.1,
) -> np.ndarray:
    """
    实现BatchNorm1d功能，支持3维张量(Batch, Channel, Dim)
    Args:
        input: 输入张量，形状为 (N, C, L) 其中 N=batch_size, C=num_features, L=length
        weight: 缩放参数，形状为 (C,)
        bias: 偏移参数，形状为 (C,)
        running_mean: 运行时均值，形状为 (C,)
        running_var: 运行时方差，形状为 (C,)
        eps: 避免除零的小常数
        training: 是否为训练模式
        momentum: 动量参数
    Returns:
        输出张量，形状与input相同
    """
    # 保存原始数据类型
    original_dtype = input.dtype
    
    # 转换为计算数据类型（避免bfloat16的精度问题）
    compute_dtype = np.float32
    input_compute = input.astype(compute_dtype)
    weight_compute = weight.astype(compute_dtype)
    bias_compute = bias.astype(compute_dtype)
    running_mean_compute = running_mean.astype(compute_dtype)
    running_var_compute = running_var.astype(compute_dtype)
    
    # 使用PyTorch的BatchNorm1d模块进行计算
    input_tensor = torch.from_numpy(input_compute)
    
    # 创建BatchNorm1d模块，使用float32数据类型
    num_features = input_tensor.shape[1]  # Channel dimension
    bn = torch.nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True, dtype=torch.float32)
    
    # 设置模块参数
    with torch.no_grad():
        bn.weight.copy_(torch.from_numpy(weight_compute))
        bn.bias.copy_(torch.from_numpy(bias_compute))
        bn.running_mean.copy_(torch.from_numpy(running_mean_compute))
        bn.running_var.copy_(torch.from_numpy(running_var_compute))
    
    bn.train()
    
    # 计算输出
    output = bn(input_tensor)
    
    # 确保输出保持原始数据类型
    return output.detach().numpy().astype(original_dtype)


class BatchNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        output: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        eps: float = 1e-5,
        training: bool = False,
        momentum: float = 0.1,
    ):  
        super().__init__("batch_norm")
        # 保存原始数据类型
        original_dtype = input.dtype
        
        self.input = input
        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        # 确保running_var保持原始数据类型
        self.running_var = running_var.astype(original_dtype)
        self.output = output
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.eps = eps
        self.training = training
        self.momentum = momentum

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # 添加参数
        test_writer.add_float32(test_writer.gguf_key("eps"), self.eps)
        test_writer.add_bool(test_writer.gguf_key("training"), self.training)
        test_writer.add_float32(test_writer.gguf_key("momentum"), self.momentum)
        
        # 添加形状信息
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)

        # 添加步长信息
        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )

        # 添加张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=np_dtype_to_ggml(self.weight.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("bias"), self.bias, raw_dtype=np_dtype_to_ggml(self.bias.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_mean"), self.running_mean, raw_dtype=np_dtype_to_ggml(self.running_mean.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_var"), self.running_var, raw_dtype=np_dtype_to_ggml(self.running_var.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )

        # 计算高精度答案
        ans = batch_norm(
            self.input.astype(np.float64),
            self.weight.astype(np.float64),
            self.bias.astype(np.float64),
            self.running_mean.astype(np.float64),
            self.running_var.astype(np.float64),
            self.eps,
            self.training,
            self.momentum,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


def gen_gguf(filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # 测试用例配置: (batch_size, num_features, length), stride_input, stride_output, training
    _TEST_CASES_ = [
        # 基本测试用例
        ((2, 4, 8), None, None, False),
        ((4, 8, 16), None, None, False),
        ((1, 16, 32), None, None, False),
        ((8, 32, 64), None, None, False),
        ((2, 64, 128), None, None, False),
        
        # 训练模式测试
        ((2, 4, 8), None, None, True),
        ((4, 8, 16), None, None, True),
        
        # 非连续张量测试
        ((2, 4, 8), (64, 8, 1), None, False),
        ((4, 8, 16), (256, 16, 1), None, False),
        ((2, 4, 8), None, (64, 8, 1), False),
        ((4, 8, 16), None, (256, 16, 1), False),
        ((2, 4, 8), (64, 8, 1), (32, 8, 1), False),
    ]

    _TENSOR_DTYPES_ = [
        np.float32,
        np.float16,
        bfloat16,
    ]

    for dtype in _TENSOR_DTYPES_:
        for shape_input, stride_input, stride_output, training in _TEST_CASES_:
            batch_size, num_features, length = shape_input
            
            # 生成随机输入数据
            input = np.random.randn(*shape_input).astype(dtype)
            
            # 生成权重和偏置
            weight = np.random.randn(num_features).astype(dtype)
            bias = np.random.randn(num_features).astype(dtype)
            
            # 生成运行时统计量
            running_mean = np.random.randn(num_features).astype(dtype)
            running_var = np.abs(np.random.randn(num_features)).astype(dtype) + 1.0  # 确保方差足够大，避免数值不稳定
            
            # 计算正确的输出
            output = batch_norm(
                input=input,
                weight=weight,
                bias=bias,
                running_mean=running_mean,
                running_var=running_var,
                eps=1e-5,
                training=training,
                momentum=0.1,
            )
            
            # 处理非连续张量
            input = process_zero_stride_tensor(input, stride_input)
            output = process_zero_stride_tensor(output, stride_output)
            
            test_case = BatchNormTestCase(
                input=input,
                weight=weight,
                bias=bias,
                running_mean=running_mean,
                running_var=running_var,
                output=output,
                shape_input=shape_input,
                stride_input=stride_input,
                shape_output=shape_input,  # 输出形状与输入相同
                stride_output=stride_output,
                eps=1e-5,
                training=training,
                momentum=0.1,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    gen_gguf("batch_norm.gguf")