import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # input_shape, weight_shape, bias_shape, output_shape
    # 基本测试案例
    ((1, 512), (256, 512), (256,), (1, 256)),  # 基本线性变换
    ((2, 1024), (512, 1024), (512,), (2, 512)),  # 批量处理
    ((4, 2048), (1024, 2048), (1024,), (4, 1024)),  # 更大的维度
    ((1, 768), (768, 768), (768,), (1, 768)),  # 方形权重矩阵
    
    # 无bias测试案例
    ((8, 256), (128, 256), None, (8, 128)),  # 无bias情况
    ((3, 512), (1024, 512), None, (3, 1024)),  # 无bias情况，不同维度
    ((1, 1), (1, 1), None, (1, 1)),  # 最小情况，无bias
    ((16, 4096), (2048, 4096), None, (16, 2048)),  # 大批量，无bias
    
    # 边界情况测试
    ((1, 1), (1, 1), (1,), (1, 1)),  # 最小情况，有bias
    ((1, 2), (3, 2), (3,), (1, 3)),  # 小维度测试
    ((5, 10), (20, 10), (20,), (5, 20)),  # 中等维度测试
    
    # 多维批次测试
    ((2, 4, 8), (16, 8), (16,), (2, 4, 16)),  # 3D输入，2D权重，1D bias
    ((3, 5, 7), (11, 7), None, (3, 5, 11)),  # 3D输入，2D权重，无bias
    ((2, 3, 4, 6), (8, 6), (8,), (2, 3, 4, 8)),  # 4D输入，2D权重，1D bias
    ((1, 2, 3, 4), (5, 4), None, (1, 2, 3, 5)),  # 4D输入，2D权重，无bias
    
    # 大维度测试
    ((32, 512), (1024, 512), (1024,), (32, 1024)),  # 较大批次
    ((64, 256), (512, 256), None, (64, 512)),  # 较大批次，无bias
    ((128, 128), (256, 128), (256,), (128, 256)),  # 大批次，中等维度
    
    # 特殊比例测试
    ((7, 13), (17, 13), (17,), (7, 17)),  # 质数维度
    ((6, 24), (12, 24), None, (6, 12)),  # 偶数维度，无bias
    ((9, 27), (81, 27), (81,), (9, 81)),  # 3的倍数维度
    
    # 极端情况测试
    ((1, 4096), (1, 4096), (1,), (1, 1)),  # 输出维度为1
    ((1000, 1), (10, 1), (10,), (1000, 10)),  # 输入维度为1
    ((100, 100), (100, 100), None, (100, 100)),  # 方形矩阵，无bias
    
    # 不规则维度测试
    ((13, 37), (73, 37), (73,), (13, 73)),  # 不规则维度组合
    ((11, 23), (47, 23), None, (11, 47)),  # 不规则维度组合，无bias
    
    # 非连续步长测试案例
    ((4, 8, 16), (32, 16), (32,), (4, 8, 32)),  # 3D输入非连续步长
    ((2, 6, 12), (24, 12), None, (2, 6, 24)),  # 3D输入非连续步长，无bias
    ((3, 4, 5, 8), (16, 8), (16,), (3, 4, 5, 16)),  # 4D输入非连续步长
    ((2, 3, 4, 6), (12, 6), None, (2, 3, 4, 12)),  # 4D输入非连续步长，无bias
    ((8, 16), (32, 16), (32,), (8, 32)),  # 2D非连续步长基本测试
    ((12, 24), (48, 24), None, (12, 48)),  # 2D非连续步长，无bias
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch implementation for linear transformation
def linear_torch(input_tensor, weight, bias=None):
    """PyTorch参考实现"""
    return torch.nn.functional.linear(input_tensor, weight, bias)


def test(
    handle,
    device,
    input_shape,
    weight_shape,
    bias_shape,
    output_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    """测试linear算子"""
    # 检测是否为非连续步长测试案例（基于输入维度数量）
    is_non_contiguous = len(input_shape) > 2
    
    test_type = "(non-contiguous)" if is_non_contiguous else ""
    print(
        f"Testing Linear on {InfiniDeviceNames[device]} with input_shape:{input_shape} weight_shape:{weight_shape} "
        f"bias_shape:{bias_shape} output_shape:{output_shape} dtype:{InfiniDtypeNames[dtype]} {test_type}"
    )
    
    # 创建输入张量
    input_tensor = TestTensor(input_shape, None, dtype, device)
    weight_tensor = TestTensor(weight_shape, None, dtype, device)
    
    # 对于非连续步长测试，创建非连续的输入张量
    if is_non_contiguous:
        # 创建一个更大的张量然后取切片来模拟非连续步长
        torch_input = input_tensor.torch_tensor()
        # 转置操作会创建非连续的张量
        if len(input_shape) == 3:
            torch_input = torch_input.transpose(0, 1).contiguous().transpose(0, 1)
        elif len(input_shape) == 4:
            torch_input = torch_input.transpose(1, 2).contiguous().transpose(1, 2)
        # 更新输入张量的数据
        input_tensor._torch_tensor = torch_input
    
    # 创建bias张量（如果需要）
    bias_tensor = None
    if bias_shape is not None:
        bias_tensor = TestTensor(bias_shape, None, dtype, device)
    
    # 创建输出张量
    output_tensor = TestTensor(output_shape, None, dtype, device, mode="zeros")

    
    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    status = LIBINFINIOP.infiniopCreateLinearDescriptor(
        handle,
        ctypes.byref(descriptor),
        input_tensor.descriptor,
        weight_tensor.descriptor,
        (bias_tensor.descriptor if bias_tensor is not None else None),
        output_tensor.descriptor,
    )
    check_error(status)
    
    # 获取工作空间大小
    workspace_size = ctypes.c_size_t()
    device_type = ctypes.c_int()
    LIBINFINIOP.infiniopGetDescriptorDeviceType(descriptor, ctypes.byref(device_type))
    status = LIBINFINIOP.infiniopGetLinearWorkspaceSize(
        descriptor, ctypes.byref(workspace_size)
    )
    check_error(status)

    # 创建工作空间
    workspace = TestWorkspace(workspace_size.value, device)

    # 执行算子
    check_error(
        LIBINFINIOP.infiniopLinear(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output_tensor.data(),
            input_tensor.data(),
            weight_tensor.data(),
            bias_tensor.data() if bias_tensor is not None else None,
            None,
        )
    )

    # 获取结果
    result_tensor = output_tensor.actual_tensor()
    if result_tensor.dtype == torch.bfloat16:
        result = result_tensor.float().cpu().numpy()
    else:
        result = result_tensor.cpu().numpy()

    # PyTorch参考实现
    torch_input = input_tensor.torch_tensor()
    torch_weight = weight_tensor.torch_tensor()
    torch_bias = bias_tensor.torch_tensor() if bias_tensor is not None else None
    
    expected_tensor = linear_torch(torch_input, torch_weight, torch_bias)
    if expected_tensor.dtype == torch.bfloat16:
        expected = expected_tensor.float().cpu().numpy()
    else:
        expected = expected_tensor.cpu().numpy()

    # 比较结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(torch.from_numpy(result), torch.from_numpy(expected), atol=atol, rtol=rtol)

    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
    
    result_for_compare = result_tensor.float() if result_tensor.dtype == torch.bfloat16 else result_tensor
    expected_for_compare = expected_tensor.float() if expected_tensor.dtype == torch.bfloat16 else expected_tensor
    assert torch.allclose(
        result_for_compare, expected_for_compare, atol=atol, rtol=rtol
    ), f"Linear test failed for dtype {InfiniDtypeNames[dtype]}"

    # 性能测试
    if PROFILE:
        profile_operation(
            lambda: LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.ptr,
                workspace_size.value,
                output_tensor.ptr,
                input_tensor.ptr,
                weight_tensor.ptr,
                bias_tensor.ptr if bias_tensor is not None else None,
                sync,
            ),
            NUM_PRERUN,
            NUM_ITERATIONS,
            f"Linear {InfiniDtypeNames[dtype]} {input_shape}x{weight_shape}",
        )

    # 清理资源
    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # 运行测试
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")