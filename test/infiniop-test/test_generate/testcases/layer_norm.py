import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def layer_norm(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray = None,
    eps: float = 1e-5,
    normalized_shape: List[int] = None,
) -> np.ndarray:
    """
    实现LayerNorm功能
    Args:
        input: 输入张量
        weight: 缩放参数
        bias: 偏移参数，可以为None
        eps: 避免除零的小常数
        normalized_shape: 需要归一化的维度形状
    Returns:
        输出张量，形状与input相同
    """
    # 保存原始数据类型
    original_dtype = input.dtype
    
    # 转换为计算数据类型（避免bfloat16的精度问题）
    compute_dtype = np.float32
    input_compute = input.astype(compute_dtype)
    weight_compute = weight.astype(compute_dtype)
    bias_compute = bias.astype(compute_dtype) if bias is not None else None
    
    # 使用PyTorch的LayerNorm模块进行计算
    input_tensor = torch.from_numpy(input_compute)
    
    # 创建LayerNorm模块，使用float32数据类型
    if normalized_shape is None:
        normalized_shape = input_tensor.shape[-1:]
    
    # 根据是否有bias决定elementwise_affine参数
    has_bias = bias is not None
    ln = torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True, dtype=torch.float32)
    
    # 设置权重和偏置
    with torch.no_grad():
        ln.weight.copy_(torch.from_numpy(weight_compute))
        if has_bias:
            ln.bias.copy_(torch.from_numpy(bias_compute))
        else:
            ln.bias.zero_()
    
    # 执行LayerNorm
    with torch.no_grad():
        output_tensor = ln(input_tensor)
    
    # 确保输出保持原始数据类型
    return output_tensor.detach().numpy().astype(original_dtype)


class LayerNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray = None,
        output: np.ndarray = None,
        shape_input: List[int] | None = None,
        stride_input: List[int] | None = None,
        shape_output: List[int] | None = None,
        stride_output: List[int] | None = None,
        eps: float = 1e-5,
        has_bias: bool = True,
    ):  
        super().__init__("layer_norm")
        
        self.input = input
        self.weight = weight
        self.bias = bias
        self.output = output
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.eps = eps
        self.has_bias = has_bias

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # 添加参数
        test_writer.add_float32(test_writer.gguf_key("eps"), self.eps)
        
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

        # 添加是否有bias的标志
        test_writer.add_bool(test_writer.gguf_key("has_bias"), self.has_bias)
        
        # 添加张量数据
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=np_dtype_to_ggml(self.weight.dtype)
        )
        if self.has_bias and self.bias is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("bias"), self.bias, raw_dtype=np_dtype_to_ggml(self.bias.dtype)
            )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )

        # 计算高精度答案
        bias_f64 = self.bias.astype(np.float64) if self.has_bias and self.bias is not None else None
        ans = layer_norm(
            self.input.astype(np.float64),
            self.weight.astype(np.float64),
            bias_f64,
            eps=self.eps,
            normalized_shape=list(self.input.shape[-len(self.weight.shape):]),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


def gen_gguf(filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    
    # 测试用例配置 - LayerNorm只支持对最后一维进行归一化，且输入至少3维
    _TEST_CASES_ = [
        # (input_shape, normalized_shape, stride_input, stride_output, has_bias)
        # 基础3D测试用例
        ((1, 4, 8), [8], None, None, True),
        ((1, 4, 8), [8], None, None, False),  # 无bias
        ((2, 3, 4), [4], None, None, True),
        ((2, 3, 4), [4], None, None, False),
        ((1, 5, 10), [10], None, None, True),
        ((3, 6, 8), [8], None, None, True),
        
        # 4D测试用例
        ((2, 4, 6, 8), [8], None, None, True),
        ((2, 4, 6, 8), [8], None, None, False),
        ((1, 2, 3, 16), [16], None, None, True),
        ((4, 8, 12, 16), [16], None, None, True),
        
        # 5D测试用例
        ((1, 2, 3, 4, 5), [5], None, None, True),
        ((1, 2, 3, 4, 5), [5], None, None, False),
        ((2, 4, 8, 16, 32), [32], None, None, True),
        
        # 大尺寸测试用例
        ((1, 16, 32), [32], None, None, True),
        ((8, 12, 64), [64], None, None, True),
        ((4, 8, 128), [128], None, None, False),
        
        # 边界情况测试
        ((1, 1, 1), [1], None, None, True),
        ((1, 1, 1), [1], None, None, False),
        ((10, 20, 1), [1], None, None, True),
        ((1, 1, 512), [512], None, None, True),
        
        # 非连续张量测试（最后一维必须连续）
        ((2, 4, 8), [8], [32, 8, 1], None, True),  # 输入非连续但最后一维连续
        ((2, 4, 8), [8], [32, 8, 1], None, False),
        ((1, 3, 6), [6], [18, 6, 1], [18, 6, 1], True),  # 输入输出都非连续
    ]
    
    _TENSOR_DTYPES_ = [np.float32, np.float16, bfloat16]
    _EPS_VALUES_ = [1e-5, 1e-4, 1e-6]  # 不同的epsilon值
    
    # 为每种数据类型生成测试用例
    for dtype in _TENSOR_DTYPES_:
        for i, (input_shape, normalized_shape, stride_input, stride_output, has_bias) in enumerate(_TEST_CASES_):
            # 选择不同的epsilon值
            eps = _EPS_VALUES_[i % len(_EPS_VALUES_)]
            
            # 生成随机输入数据
            np.random.seed(42 + i)  # 固定种子确保可重现
            input_data = np.random.randn(*input_shape).astype(dtype)
            
            # 生成权重
            weight = np.random.randn(*normalized_shape).astype(dtype)
            
            # 根据has_bias决定是否生成偏置
            bias = np.random.randn(*normalized_shape).astype(dtype) if has_bias else None
            
            # 计算正确的输出
            output = layer_norm(
                input=input_data,
                weight=weight,
                bias=bias,
                eps=eps,
                normalized_shape=normalized_shape,
            )
            
            # 处理非连续张量
            input_data = process_zero_stride_tensor(input_data, stride_input)
            output = process_zero_stride_tensor(output, stride_output)
            
            test_case = LayerNormTestCase(
                input=input_data,
                weight=weight,
                bias=bias,
                output=output,
                shape_input=input_shape,
                stride_input=stride_input,
                shape_output=input_shape,  # 输出形状与输入相同
                stride_output=stride_output,
                eps=eps,
                has_bias=has_bias,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    gen_gguf("layer_norm.gguf")