import torch
import ctypes
from ctypes import c_uint64, c_float
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
from enum import Enum, auto
import torch.nn.functional as F

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_INPUT = auto()

# Test cases: (shape) - 只支持3维全连续张量 (Batch, Channel, Dim)
_TEST_CASES_ = [
    # 小尺寸测试
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 1),
    (2, 1, 1),
    (2, 2, 4),
    (13, 2, 4),
    (3, 5, 7),
    (8, 3, 16),
    
    # 不同批次大小测试
    (1, 4, 32),
    (4, 4, 32),
    (8, 4, 32),
    (16, 4, 32),
    (32, 4, 32),
    
    # 不同通道数测试
    (4, 1, 64),
    (4, 3, 64),
    (4, 8, 64),
    (4, 16, 64),
    (4, 32, 64),
    (4, 64, 64),
    (4, 128, 64),
    
    # 不同空间维度测试
    (4, 8, 1),
    (4, 8, 8),
    (4, 8, 32),
    (4, 8, 64),
    (4, 8, 128),
    (4, 8, 256),
    (4, 8, 512),
    (4, 8, 1024),
    
    # 大尺寸测试用例 - 验证数值稳定性优化
    (16, 4, 256),
    (16, 4, 1024),
    (16, 4, 2816),
    (4, 8, 5632),
    (8, 16, 2048),
    (32, 8, 1024),
    
    # 特殊比例测试用例
    (100, 1, 10),
    (10, 100, 10),
    (10, 10, 100),
    (1, 256, 256),
    (256, 1, 256),
    (256, 256, 1),
    
    # 边界和压力测试用例
    (1, 1, 10000),
    (1, 1000, 10),
    (1000, 1, 10),
    (64, 64, 64),
    (128, 32, 128),
]

# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_INPUT,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_EXPANDED_TEST_CASES_ = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16] 

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},  
    InfiniDtype.F32: {"atol": 2e-5, "rtol": 2e-5}, 
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},  
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def batch_norm_backward(handle, grad_input, grad_weight, grad_bias, grad_output, input_tensor, weight, running_mean, running_var):
    """Call the InfiniOp BatchNormBackward implementation"""
    # Create descriptor
    desc = infiniopOperatorDescriptor_t()
    
    # Create BatchNormBackward descriptor
    check_error(
        LIBINFINIOP.infiniopCreateBatchNormBackwardDescriptor(
            handle,
            ctypes.byref(desc),
            grad_input.descriptor,
            grad_weight.descriptor,
            grad_bias.descriptor if grad_bias is not None else None,
            grad_output.descriptor,
            input_tensor.descriptor,
            weight.descriptor,
            running_mean.descriptor,
            running_var.descriptor,
            c_float(1e-5), # eps - 用于数值稳定性
        )
    )
    
    # Get workspace size
    workspace_size = c_uint64()
    check_error(
        LIBINFINIOP.infiniopGetBatchNormBackwardWorkspaceSize(
            desc, ctypes.byref(workspace_size)
        )
    )
    
    # Create workspace
    workspace = TestWorkspace(workspace_size.value, input_tensor.device)
    
    # Execute BatchNormBackward
    check_error(
        LIBINFINIOP.infiniopBatchNormBackward(
            desc,
            workspace.data(),
            workspace.size(),
            grad_input.data(),
            grad_weight.data(),
            grad_bias.data() if grad_bias is not None else None,
            grad_output.data(),
            input_tensor.data(),
            weight.data(),
            running_mean.data(),
            running_var.data(),
            None,  # stream
        )
    )
    
    # Destroy descriptor
    check_error(LIBINFINIOP.infiniopDestroyBatchNormBackwardDescriptor(desc))


def test(
    handle,
    device,
    batch_size,
    channels,
    dim_size,
    inplace,
    tensor_dtype,
    sync,
):
    """Test function for BatchNormBackward operator"""
    input_shape = (batch_size, channels, dim_size)
    inplace_str = "inplace" if inplace == Inplace.INPLACE_INPUT else "out-of-place"
    print(
        f"Testing BatchNormBackward on {InfiniDeviceNames[device]} with shape:{input_shape} "
        f"dtype:{InfiniDtypeNames[tensor_dtype]} mode:{inplace_str}"
    )

    # Create input tensors for forward pass (全连续张量)
    # For F16 and BF16, use smaller scale to reduce numerical errors
    if tensor_dtype == InfiniDtype.F16:
        input_tensor = TestTensor(
            input_shape, None, tensor_dtype, device, mode="random", scale=0.5, bias=-0.1
        )
    elif tensor_dtype == InfiniDtype.BF16:
        input_tensor = TestTensor(
            input_shape, None, tensor_dtype, device, mode="random", scale=0.2, bias=-0.1
        )
    else:
        input_tensor = TestTensor(
            input_shape, None, tensor_dtype, device, mode="random", scale=2.0, bias=-1.0
        )
    
    # BatchNorm parameters (1D tensors with length = channels)
    param_shape = (channels,)
    
    weight = TestTensor(
        param_shape, None, tensor_dtype, device, mode="ones"
    )
    bias = TestTensor(
        param_shape, None, tensor_dtype, device, mode="zeros"
    )
    running_mean = TestTensor(
        param_shape, None, tensor_dtype, device, mode="zeros"
    )
    running_var = TestTensor(
        param_shape, None, tensor_dtype, device, mode="ones"
    )
    
    # Create grad_output tensor (same shape as input, 全连续张量)
    # For F16 and BF16, use smaller scale to reduce numerical errors
    if tensor_dtype == InfiniDtype.F16:
        grad_output = TestTensor(
            input_shape, None, tensor_dtype, device, mode="random", scale=0.1, bias=0.0
        )
    elif tensor_dtype == InfiniDtype.BF16:
        grad_output = TestTensor(
            input_shape, None, tensor_dtype, device, mode="random", scale=0.05, bias=0.0
        )
    else:
        grad_output = TestTensor(
            input_shape, None, tensor_dtype, device, mode="random", scale=1.0, bias=0.0
        )
    
    # Create gradient tensors (全连续张量)
    if inplace == Inplace.INPLACE_INPUT:
        grad_input = grad_output  # Reuse grad_output for inplace
    else:
        grad_input = TestTensor(
            input_shape, None, tensor_dtype, device, mode="zeros"
        )
    
    grad_weight = TestTensor(
        param_shape, None, tensor_dtype, device, mode="zeros"
    )
    grad_bias = TestTensor(
        param_shape, None, tensor_dtype, device, mode="zeros"
    )
    
    # After forward pass, get the updated running statistics from our C++ implementation
    updated_running_mean = running_mean.torch_tensor().clone()
    updated_running_var = running_var.torch_tensor().clone()
    
    # Convert to PyTorch tensors for reference computation using SAME statistics
    input_torch = input_tensor.torch_tensor().clone().requires_grad_(True)
    weight_torch = weight.torch_tensor().clone().requires_grad_(True)
    bias_torch = bias.torch_tensor().clone().requires_grad_(True)
    # Save grad_output before it might be modified by inplace operation
    grad_output_torch = grad_output.torch_tensor().clone()
    
    # Ensure all tensors use the same dtype to avoid mixed precision issues
    target_dtype = input_torch.dtype
    weight_torch = weight_torch.to(dtype=target_dtype)
    bias_torch = bias_torch.to(dtype=target_dtype)
    grad_output_torch = grad_output_torch.to(dtype=target_dtype)
    updated_running_mean = updated_running_mean.to(dtype=target_dtype)
    updated_running_var = updated_running_var.to(dtype=target_dtype)
    
    # PyTorch BatchNorm1d for 3D tensors (Batch, Channel, Dim) - 符合competition.md要求
    if len(input_shape) == 3:
        torch_bn = torch.nn.BatchNorm1d(channels, affine=True, track_running_stats=True)
    else:
        raise ValueError(f"Only 3D tensors are supported according to competition.md, got {len(input_shape)}D")
    
    # Convert BatchNorm to the same dtype as input to avoid mixed precision issues
    torch_bn = torch_bn.to(dtype=input_torch.dtype)
    
    # Use the SAME statistics that our C++ forward pass produced
    torch_bn.weight.data = weight_torch.clone()
    torch_bn.bias.data = bias_torch.clone()
    torch_bn.running_mean.data = updated_running_mean.clone()  # Use updated values from C++ forward
    torch_bn.running_var.data = updated_running_var.clone()    # Use updated values from C++ forward
    torch_bn.eval()  
    
    # PyTorch forward pass to get reference gradients using SAME statistics
    input_torch.retain_grad()  # Ensure gradient is retained for non-leaf tensor
    torch_bn.weight.requires_grad_(True)
    torch_bn.bias.requires_grad_(True)
    
    output_torch = torch_bn(input_torch)
    
    # Backward pass
    output_torch.backward(grad_output_torch)
    
    # Get reference gradients
    grad_input_expected = input_torch.grad.clone()
    grad_weight_expected = torch_bn.weight.grad.clone()
    grad_bias_expected = torch_bn.bias.grad.clone()
    
    # Now both implementations use exactly the same forward pass statistics
    
    # Execute InfiniOp BatchNormBackward
    if PROFILE:
        profile_operation(
            lambda: batch_norm_backward(
                handle, grad_input, grad_weight, grad_bias, grad_output, 
                input_tensor, weight, running_mean, running_var
            ),
            f"InfiniOp BatchNormBackward {InfiniDtypeNames[tensor_dtype]} {input_shape}",
            NUM_PRERUN,
            NUM_ITERATIONS,
            sync,
        )
    else:
        batch_norm_backward(
            handle, grad_input, grad_weight, grad_bias, grad_output, 
            input_tensor, weight, running_mean, running_var
        )
    
    # Compare results
    # For inplace mode, grad_input and grad_output are the same tensor
    # We need to compare with the modified grad_output tensor
    if inplace == Inplace.INPLACE_INPUT:
        actual_grad_input = grad_output.actual_tensor()  # grad_input is grad_output in inplace mode
    else:
        actual_grad_input = grad_input.actual_tensor()
    actual_grad_weight = grad_weight.actual_tensor()
    actual_grad_bias = grad_bias.actual_tensor()
    
    # Check correctness
    tolerance = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    atol, rtol = tolerance
    
    # Always print debug info for the first test to understand the issue
    print(f"\nDebug info for shape {input_shape}:")
    print(f"actual_grad_input shape: {actual_grad_input.shape}, dtype: {actual_grad_input.dtype}")
    print(f"expected_grad_input shape: {grad_input_expected.shape}, dtype: {grad_input_expected.dtype}")
    print(f"actual_grad_input stats: min={actual_grad_input.min():.6f}, max={actual_grad_input.max():.6f}, mean={actual_grad_input.mean():.6f}")
    print(f"expected_grad_input stats: min={grad_input_expected.min():.6f}, max={grad_input_expected.max():.6f}, mean={grad_input_expected.mean():.6f}")
    print(f"actual_grad_weight shape: {actual_grad_weight.shape}, dtype: {actual_grad_weight.dtype}")
    print(f"expected_grad_weight shape: {grad_weight_expected.shape}, dtype: {grad_weight_expected.dtype}")
    print(f"actual_grad_weight stats: min={actual_grad_weight.min():.6f}, max={actual_grad_weight.max():.6f}, mean={actual_grad_weight.mean():.6f}")
    print(f"expected_grad_weight stats: min={grad_weight_expected.min():.6f}, max={grad_weight_expected.max():.6f}, mean={grad_weight_expected.mean():.6f}")
    print(f"Tolerance: atol={atol}, rtol={rtol}")
    
    if DEBUG:
        debug(actual_grad_input, grad_input_expected.detach(), atol=atol, rtol=rtol)
        debug(actual_grad_weight, grad_weight_expected.detach(), atol=atol, rtol=rtol)
        debug(actual_grad_bias, grad_bias_expected.detach(), atol=atol, rtol=rtol)
    
    torch.testing.assert_close(
        actual_grad_input,
        grad_input_expected,
        atol=atol,
        rtol=rtol,
        msg=f"BatchNormBackward grad_input test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
    )
    
    torch.testing.assert_close(
        actual_grad_weight,
        grad_weight_expected,
        atol=atol,
        rtol=rtol,
        msg=f"BatchNormBackward grad_weight test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
    )
    
    torch.testing.assert_close(
        actual_grad_bias,
        grad_bias_expected,
        atol=atol,
        rtol=rtol,
        msg=f"BatchNormBackward grad_bias test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
    )
    
    # Clean up
    input_tensor.destroy_desc()
    weight.destroy_desc()
    bias.destroy_desc()
    running_mean.destroy_desc()
    running_var.destroy_desc()
    grad_output.destroy_desc()
    grad_weight.destroy_desc()
    grad_bias.destroy_desc()
    
    # For inplace, grad_input is the same as grad_output, so don't destroy twice
    if inplace != Inplace.INPLACE_INPUT:
        grad_input.destroy_desc()


if __name__ == "__main__":
    args = get_args()
    
    # Update global settings
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(device, test, _EXPANDED_TEST_CASES_, _TENSOR_DTYPES)
    
    print("\033[92mBatchNormBackward test passed!\033[0m")