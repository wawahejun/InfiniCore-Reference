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

# 测试用例：(grad_y_shape, x_shape, w_shape, grad_x_shape, grad_w_shape, grad_b_shape, has_bias)
# grad_y: 输出梯度, x: 前向输入, w: 权重, grad_x: 输入梯度, grad_w: 权重梯度, grad_b: 偏置梯度
_TEST_CASES = [
    # 有bias的情况
    ((1, 256), (1, 512), (256, 512), (1, 512), (256, 512), (256,), True),  # 基本线性变换反向
    ((2, 512), (2, 1024), (512, 1024), (2, 1024), (512, 1024), (512,), True),  # 批量处理
    ((4, 1024), (4, 2048), (1024, 2048), (4, 2048), (1024, 2048), (1024,), True),  # 更大的维度
    ((1, 768), (1, 768), (768, 768), (1, 768), (768, 768), (768,), True),  # 方形权重矩阵
    
    # 无bias的情况
    ((8, 128), (8, 256), (128, 256), (8, 256), (128, 256), None, False),  # 无bias情况
    ((3, 1024), (3, 512), (1024, 512), (3, 512), (1024, 512), None, False),  # 无bias情况，不同维度
    ((1, 1), (1, 1), (1, 1), (1, 1), (1, 1), None, False),  # 最小情况
    ((16, 2048), (16, 4096), (2048, 4096), (16, 4096), (2048, 4096), None, False),  # 大批量
    
    # 非连续步长测试案例
    # 3D输入非连续步长测试
    ((2, 4, 16), (2, 4, 8), (16, 8), (2, 4, 8), (16, 8), (16,), True),  # 3D输入非连续步长，有bias
    ((3, 5, 11), (3, 5, 7), (11, 7), (3, 5, 7), (11, 7), None, False),  # 3D输入非连续步长，无bias
    ((4, 6, 24), (4, 6, 12), (24, 12), (4, 6, 12), (24, 12), (24,), True),  # 3D输入非连续步长，较大维度
    
    # 4D输入非连续步长测试
    ((2, 3, 4, 8), (2, 3, 4, 6), (8, 6), (2, 3, 4, 6), (8, 6), (8,), True),  # 4D输入非连续步长，有bias
    ((1, 2, 3, 5), (1, 2, 3, 4), (5, 4), (1, 2, 3, 4), (5, 4), None, False),  # 4D输入非连续步长，无bias
    ((2, 4, 6, 12), (2, 4, 6, 8), (12, 8), (2, 4, 6, 8), (12, 8), (12,), True),  # 4D输入非连续步长，较大维度
]

# 支持的数据类型
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# 容差映射
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def linear_backward_torch(grad_y, x, w, has_bias=True):
    """PyTorch参考实现"""
    # 保存原始形状
    original_grad_y_shape = grad_y.shape
    original_x_shape = x.shape
    
    # 将多维张量重塑为2D进行计算
    if grad_y.dim() > 2:
        # 将前面的维度合并为batch维度
        batch_size = 1
        for dim in grad_y.shape[:-1]:
            batch_size *= dim
        # 使用reshape而不是view来处理非连续张量
        grad_y_2d = grad_y.reshape(batch_size, grad_y.shape[-1])
        x_2d = x.reshape(batch_size, x.shape[-1])
    else:
        grad_y_2d = grad_y
        x_2d = x
    
    # 计算输入梯度 grad_x = grad_y @ w
    grad_x_2d = torch.mm(grad_y_2d, w)
    
    # 计算权重梯度 grad_w = grad_y.T @ x
    grad_w = torch.mm(grad_y_2d.t(), x_2d)
    
    # 计算偏置梯度 grad_b = sum(grad_y, dim=0)
    if has_bias:
        if grad_y.dim() > 2:
            # 对于多维张量，需要对除最后一个维度外的所有维度求和
            grad_b = grad_y_2d.sum(dim=0)
        else:
            grad_b = grad_y.sum(dim=0)
    else:
        grad_b = None
    
    # 将结果重塑回原始形状
    if grad_y.dim() > 2:
        grad_x = grad_x_2d.reshape(original_x_shape)
    else:
        grad_x = grad_x_2d
    
    return grad_x, grad_w, grad_b


def test(
    handle,
    device,
    grad_y_shape,
    x_shape,
    w_shape,
    grad_x_shape,
    grad_w_shape,
    grad_b_shape,
    has_bias,
    dtype=InfiniDtype.F16,
    sync=None,
):
    """测试linear_backward算子"""
    has_bias = grad_b_shape is not None
    
    # 检测是否为非连续步长测试案例（基于输入维度数量）
    is_non_contiguous = len(x_shape) > 2
    
    test_type = "(non-contiguous)" if is_non_contiguous else ""
    print(
        f"Testing LinearBackward on {InfiniDeviceNames[device]} with grad_y_shape:{grad_y_shape} x_shape:{x_shape} w_shape:{w_shape} "
        f"grad_x_shape:{grad_x_shape} grad_w_shape:{grad_w_shape} grad_b_shape:{grad_b_shape} dtype:{InfiniDtypeNames[dtype]} {test_type}"
    )
    
    # 创建张量
    grad_y_tensor = TestTensor(grad_y_shape, None, dtype, device)
    x_tensor = TestTensor(x_shape, None, dtype, device)
    
    # 对于非连续步长测试，创建非连续的输入张量
    if is_non_contiguous:
        # 创建非连续的grad_y和x张量
        torch_grad_y = grad_y_tensor.torch_tensor()
        torch_x = x_tensor.torch_tensor()
        
        # 转置操作会创建非连续的张量
        if len(grad_y_shape) == 3:
            torch_grad_y = torch_grad_y.transpose(0, 1).contiguous().transpose(0, 1)
            torch_x = torch_x.transpose(0, 1).contiguous().transpose(0, 1)
        elif len(grad_y_shape) == 4:
            torch_grad_y = torch_grad_y.transpose(1, 2).contiguous().transpose(1, 2)
            torch_x = torch_x.transpose(1, 2).contiguous().transpose(1, 2)
        
        # 更新张量的数据
        grad_y_tensor._torch_tensor = torch_grad_y
        x_tensor._torch_tensor = torch_x
    w_tensor = TestTensor(w_shape, None, dtype, device)
    grad_x_tensor = TestTensor(grad_x_shape, None, dtype, device)
    grad_w_tensor = TestTensor(grad_w_shape, None, dtype, device)
    grad_b_tensor = TestTensor(grad_b_shape, None, dtype, device) if has_bias else None
    
    # 创建描述符
    desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLinearBackwardDescriptor(
            handle, ctypes.byref(desc), 
            grad_y_tensor.descriptor, x_tensor.descriptor, w_tensor.descriptor,
            grad_x_tensor.descriptor, grad_w_tensor.descriptor, 
            grad_b_tensor.descriptor if has_bias else None
        )
    )
    
    # 获取工作空间大小
    workspace_size = c_uint64()
    check_error(
        LIBINFINIOP.infiniopGetLinearBackwardWorkspaceSize(
            desc, ctypes.byref(workspace_size)
        )
    )
    
    # 创建工作空间
    workspace = TestWorkspace(workspace_size.value, device)
    
    # 执行算子
    check_error(
        LIBINFINIOP.infiniopLinearBackward(
            desc,
            workspace.data(),
            workspace_size.value,
            grad_x_tensor.data(),
            grad_w_tensor.data(),
            grad_b_tensor.data() if has_bias else None,
            grad_y_tensor.data(),
            x_tensor.data(),
            w_tensor.data(),
            None,
        )
    )
    
    # 获取结果
    grad_x_result_tensor = grad_x_tensor.actual_tensor()
    grad_w_result_tensor = grad_w_tensor.actual_tensor()
    grad_b_result_tensor = grad_b_tensor.actual_tensor() if has_bias else None
    
    if grad_x_result_tensor.dtype == torch.bfloat16:
        grad_x_result = grad_x_result_tensor.float().cpu().numpy()
    else:
        grad_x_result = grad_x_result_tensor.cpu().numpy()
        
    if grad_w_result_tensor.dtype == torch.bfloat16:
        grad_w_result = grad_w_result_tensor.float().cpu().numpy()
    else:
        grad_w_result = grad_w_result_tensor.cpu().numpy()
        
    if has_bias:
        if grad_b_result_tensor.dtype == torch.bfloat16:
            grad_b_result = grad_b_result_tensor.float().cpu().numpy()
        else:
            grad_b_result = grad_b_result_tensor.cpu().numpy()
    else:
        grad_b_result = None
    
    # PyTorch参考实现
    torch_grad_y = grad_y_tensor.torch_tensor()
    torch_x = x_tensor.torch_tensor()
    torch_w = w_tensor.torch_tensor()
    
    expected_grad_x, expected_grad_w, expected_grad_b = linear_backward_torch(
        torch_grad_y, torch_x, torch_w, has_bias
    )
    
    if expected_grad_x.dtype == torch.bfloat16:
        expected_grad_x_np = expected_grad_x.float().cpu().numpy()
    else:
        expected_grad_x_np = expected_grad_x.cpu().numpy()
        
    if expected_grad_w.dtype == torch.bfloat16:
        expected_grad_w_np = expected_grad_w.float().cpu().numpy()
    else:
        expected_grad_w_np = expected_grad_w.cpu().numpy()
        
    if has_bias:
        if expected_grad_b.dtype == torch.bfloat16:
            expected_grad_b_np = expected_grad_b.float().cpu().numpy()
        else:
            expected_grad_b_np = expected_grad_b.cpu().numpy()
    else:
        expected_grad_b_np = None
    
    # 获取容差
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    if DEBUG:
        print(f"grad_x shape: {grad_x_result.shape}, Expected shape: {expected_grad_x_np.shape}")
        print(f"grad_w shape: {grad_w_result.shape}, Expected shape: {expected_grad_w_np.shape}")
        if has_bias:
            print(f"grad_b shape: {grad_b_result.shape}, Expected shape: {expected_grad_b_np.shape}")
        print(f"Max diff grad_x: {abs(grad_x_result - expected_grad_x_np).max()}")
        print(f"Max diff grad_w: {abs(grad_w_result - expected_grad_w_np).max()}")
        if has_bias:
            print(f"Max diff grad_b: {abs(grad_b_result - expected_grad_b_np).max()}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        debug(torch.from_numpy(grad_x_result), torch.from_numpy(expected_grad_x_np), atol=atol, rtol=rtol)
        debug(torch.from_numpy(grad_w_result), torch.from_numpy(expected_grad_w_np), atol=atol, rtol=rtol)
        if has_bias:
            debug(torch.from_numpy(grad_b_result), torch.from_numpy(expected_grad_b_np), atol=atol, rtol=rtol)
    
    # 比较结果
    assert grad_x_result.shape == expected_grad_x_np.shape, f"grad_x shape mismatch: {grad_x_result.shape} vs {expected_grad_x_np.shape}"
    assert grad_w_result.shape == expected_grad_w_np.shape, f"grad_w shape mismatch: {grad_w_result.shape} vs {expected_grad_w_np.shape}"
    if has_bias:
        assert grad_b_result.shape == expected_grad_b_np.shape, f"grad_b shape mismatch: {grad_b_result.shape} vs {expected_grad_b_np.shape}"
    
    grad_x_for_compare = grad_x_result_tensor.float() if grad_x_result_tensor.dtype == torch.bfloat16 else grad_x_result_tensor
    expected_grad_x_for_compare = expected_grad_x.float() if expected_grad_x.dtype == torch.bfloat16 else expected_grad_x
    assert torch.allclose(
        grad_x_for_compare, expected_grad_x_for_compare, atol=atol, rtol=rtol
    ), f"Linear backward grad_x test failed for dtype {InfiniDtypeNames[dtype]}"
    
    grad_w_for_compare = grad_w_result_tensor.float() if grad_w_result_tensor.dtype == torch.bfloat16 else grad_w_result_tensor
    expected_grad_w_for_compare = expected_grad_w.float() if expected_grad_w.dtype == torch.bfloat16 else expected_grad_w
    assert torch.allclose(
        grad_w_for_compare, expected_grad_w_for_compare, atol=atol, rtol=rtol
    ), f"Linear backward grad_w test failed for dtype {InfiniDtypeNames[dtype]}"
    
    if has_bias:
        grad_b_for_compare = grad_b_result_tensor.float() if grad_b_result_tensor.dtype == torch.bfloat16 else grad_b_result_tensor
        expected_grad_b_for_compare = expected_grad_b.float() if expected_grad_b.dtype == torch.bfloat16 else expected_grad_b
        assert torch.allclose(
            grad_b_for_compare, expected_grad_b_for_compare, atol=atol, rtol=rtol
        ), f"Linear backward grad_b test failed for dtype {InfiniDtypeNames[dtype]}"
    
    # 性能测试
    if PROFILE:
        profile_operation(
            lambda: LIBINFINIOP.infiniopLinearBackward(
                desc,
                workspace.data(),
                workspace_size,
                grad_y_tensor.data(),
                x_tensor.data(),
                w_tensor.data(),
                grad_x_tensor.data(),
                grad_w_tensor.data(),
                grad_b_tensor.data() if has_bias else None,
                sync,
            ),
            NUM_PRERUN,
            NUM_ITERATIONS,
            f"LinearBackward {InfiniDtypeNames[dtype]} {grad_y_shape}x{x_shape}x{w_shape}",
        )
    
    # 清理资源
    check_error(LIBINFINIOP.infiniopDestroyLinearBackwardDescriptor(desc))


if __name__ == "__main__":
    args = get_args()
    
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    
    print("\033[92mTest passed!\033[0m")