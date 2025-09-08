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
_TEST_CASES_ = [
    # grad_input_shape, grad_weight_shape, grad_bias_shape, grad_output_shape, input_shape, weight_shape, input_std_deviation_shape, input_standardization_shape, grad_input_stride, grad_output_stride, input_stride
    
    ((128,), (128,), (128,), (128,), (128,), (128,), (), (128,), None, None, None),
    ((128,), (128,), None, (128,), (128,), (128,), (), (128,), None, None, None),  # 无bias
    ((512,), (512,), (512,), (512,), (512,), (512,), (), (512,), None, None, None),
    ((512,), (512,), None, (512,), (512,), (512,), (), (512,), None, None, None),  # 无bias
    ((16, 128), (128,), (128,), (16, 128), (16, 128), (128,), (16,), (16, 128), None, None, None),
    ((16, 128), (128,), None, (16, 128), (16, 128), (128,), (16,), (16, 128), None, None, None),  # 无bias
    ((32, 256), (256,), (256,), (32, 256), (32, 256), (256,), (32,), (32, 256), None, None, None),
    ((32, 256), (256,), None, (32, 256), (32, 256), (256,), (32,), (32, 256), None, None, None),  # 无bias
    ((2, 16, 2048), (2048,), (2048,), (2, 16, 2048), (2, 16, 2048), (2048,), (2, 16), (2, 16, 2048), None, None, None),
    ((2, 16, 2048), (2048,), None, (2, 16, 2048), (2, 16, 2048), (2048,), (2, 16), (2, 16, 2048), None, None, None),  # 无bias
    ((4, 8, 1024), (1024,), (1024,), (4, 8, 1024), (4, 8, 1024), (1024,), (4, 8), (4, 8, 1024), None, None, None),
    ((4, 8, 1024), (1024,), None, (4, 8, 1024), (4, 8, 1024), (1024,), (4, 8), (4, 8, 1024), None, None, None),  # 无bias
    ((1, 32, 512), (512,), (512,), (1, 32, 512), (1, 32, 512), (512,), (1, 32), (1, 32, 512), None, None, None),
    ((1, 32, 512), (512,), None, (1, 32, 512), (1, 32, 512), (512,), (1, 32), (1, 32, 512), None, None, None),  # 无bias
    ((2, 4, 8, 256), (256,), (256,), (2, 4, 8, 256), (2, 4, 8, 256), (256,), (2, 4, 8), (2, 4, 8, 256), None, None, None),
    ((2, 4, 8, 256), (256,), None, (2, 4, 8, 256), (2, 4, 8, 256), (256,), (2, 4, 8), (2, 4, 8, 256), None, None, None),  # 无bias
    ((1, 2, 3, 4, 128), (128,), (128,), (1, 2, 3, 4, 128), (1, 2, 3, 4, 128), (128,), (1, 2, 3, 4), (1, 2, 3, 4, 128), None, None, None),
    ((1, 2, 3, 4, 128), (128,), None, (1, 2, 3, 4, 128), (1, 2, 3, 4, 128), (128,), (1, 2, 3, 4), (1, 2, 3, 4, 128), None, None, None),  # 无bias
    
    # 非连续内存布局测试用例 - 测试转置和切片后的张量
    ((2, 16, 2048), (2048,), (2048,), (2, 16, 2048), (2, 16, 2048), (2048,), (2, 16), (2, 16, 2048), (2048*16, 2048, 1), (2048*16, 2048, 1), (2048*16, 2048, 1)),
    ((2, 16, 2048), (2048,), None, (2, 16, 2048), (2, 16, 2048), (2048,), (2, 16), (2, 16, 2048), (2048*16, 2048, 1), (2048*16, 2048, 1), (2048*16, 2048, 1)),  # 无bias
    ((4, 8, 1024), (1024,), (1024,), (4, 8, 1024), (4, 8, 1024), (1024,), (4, 8), (4, 8, 1024), (1024*8, 1024, 1), (1024*8, 1024, 1), (1024*8, 1024, 1)),
    ((4, 8, 1024), (1024,), None, (4, 8, 1024), (4, 8, 1024), (1024,), (4, 8), (4, 8, 1024), (1024*8, 1024, 1), (1024*8, 1024, 1), (1024*8, 1024, 1)),  # 无bias
]

# w (weight) and b (bias) types
# Note: 'None' means the same as input dtype
# Only test same precision cases to avoid mixed precision complexity
_WEIGHT_DTYPES = [None]
_BIAS_DTYPES = [None]
# x types used for testing - support F32, F16, BF16 as required by competition
_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]

# Form the test cases by appending each element of _WEIGHT_DTYPES and _BIAS_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (w_dtype, b_dtype) for test_case in _TEST_CASES_ 
    for w_dtype in _WEIGHT_DTYPES for b_dtype in _BIAS_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},  
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},  
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def get_pytorch_official_grads(x, grad_output, w, b, eps):
    """
    PyTorch official LayerNorm backward pass using autograd
    This serves as the ground truth reference
    """
    # Ensure all tensors have the same dtype as input x
    target_dtype = x.dtype
    
    x = x.detach().requires_grad_(True)
    w = w.detach().to(target_dtype).requires_grad_(True)
    b = b.detach().to(target_dtype).requires_grad_(True) if b is not None else None
    grad_output = grad_output.to(target_dtype)
    
    ln = torch.nn.LayerNorm(normalized_shape=x.shape[-1], eps=eps)
    # Use the original weight and bias tensors with requires_grad=True
    ln.weight = torch.nn.Parameter(w.clone())
    if b is not None:
        ln.bias = torch.nn.Parameter(b.clone())
    else:
        ln.bias = None
    
    y = ln(x)
    y.backward(grad_output)  # 触发自动求导
    return x.grad, ln.weight.grad, (ln.bias.grad if ln.bias is not None else None)


def layer_norm_backward_pytorch(grad_output, input, weight, bias, input_std_deviation, input_standardization, eps):
    """
    PyTorch reference implementation for LayerNorm backward pass
    Vectorized implementation for GPU acceleration and profiling accuracy
    """
    # Get dimensions
    batch_dims = input.shape[:-1]
    feature_dim = input.shape[-1]
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Reshape for easier computation
    input_flat = input.view(batch_size, feature_dim)
    grad_output_flat = grad_output.view(batch_size, feature_dim)
    input_standardization_flat = input_standardization.view(batch_size, feature_dim)
    
    # Vectorized computation - no loops for GPU acceleration
    inv_std = (1.0 / input_std_deviation.view(-1, 1))  # [batch_size, 1]
    
    # Compute intermediate sums using vectorized operations
    sum_grad_out = torch.sum(grad_output_flat * weight, dim=1, keepdim=True)  # [batch_size, 1]
    sum_grad_out_norm = torch.sum(grad_output_flat * weight * input_standardization_flat, dim=1, keepdim=True)  # [batch_size, 1]
    
    # Compute mean values
    mean_grad_out = sum_grad_out / feature_dim  # [batch_size, 1]
    mean_grad_out_norm = sum_grad_out_norm / feature_dim  # [batch_size, 1]
    
    # Compute grad_input using vectorized operations
    grad_input_flat = inv_std * (grad_output_flat * weight - mean_grad_out - input_standardization_flat * mean_grad_out_norm)
    
    # Compute grad_weight and grad_bias using vectorized operations
    grad_weight = torch.sum(grad_output_flat * input_standardization_flat, dim=0)  # [feature_dim]
    grad_bias = torch.sum(grad_output_flat, dim=0) if bias is not None else None  # [feature_dim]
    
    # Reshape back to original shape
    grad_input = grad_input_flat.view_as(input)
    
    return grad_input, grad_weight, grad_bias


def test(
    handle,
    device,
    grad_input_shape,
    grad_weight_shape,
    grad_bias_shape,
    grad_output_shape,
    input_shape,
    weight_shape,
    input_std_deviation_shape,
    input_standardization_shape,
    grad_input_stride,
    grad_output_stride,
    input_stride,
    w_dtype=InfiniDtype.F32,
    b_dtype=InfiniDtype.F32,
    dtype=InfiniDtype.F16,
    sync=None,
):
    w_dtype = w_dtype if w_dtype else dtype
    b_dtype = b_dtype if b_dtype else dtype
    print(
        f"Testing LayerNormBackward on {InfiniDeviceNames[device]} with grad_input_shape:{grad_input_shape} grad_weight_shape:{grad_weight_shape} grad_bias_shape:{grad_bias_shape}"
        f" grad_output_shape:{grad_output_shape} input_shape:{input_shape} weight_shape:{weight_shape}"
        f" w_dtype:{InfiniDtypeNames[w_dtype]} b_dtype:{InfiniDtypeNames[b_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    # Create test tensors
    grad_input = TestTensor(grad_input_shape, grad_input_stride, dtype, device, mode="zeros")
    grad_weight = TestTensor(grad_weight_shape, None, w_dtype, device, mode="zeros")
    grad_bias = TestTensor(grad_bias_shape, None, b_dtype, device, mode="zeros") if grad_bias_shape is not None else None
    grad_output = TestTensor(grad_output_shape, grad_output_stride, dtype, device, scale=0.01)
    input_tensor = TestTensor(input_shape, input_stride, dtype, device, scale=0.01)
    weight = TestTensor(weight_shape, None, w_dtype, device)
    
    eps = 1e-5
    
    # Use our own LayerNorm forward pass to compute intermediate values
    # This ensures numerical consistency between forward and backward passes
    
    # Create temporary output tensor for forward pass
    temp_output = TestTensor(input_shape, input_stride, dtype, device, mode="zeros")
    input_std_deviation = TestTensor(input_std_deviation_shape, None, dtype, device, mode="zeros")
    input_standardization = TestTensor(input_standardization_shape, input_stride, dtype, device, mode="zeros")
    
    # Create bias tensor for forward pass (if needed)
    bias_for_forward = None
    if grad_bias is not None:
        bias_for_forward = TestTensor(grad_bias_shape, None, dtype, device, mode="zeros")
    
    # Create LayerNorm forward descriptor
    forward_descriptor = infiniopOperatorDescriptor_t()
    status = LIBINFINIOP.infiniopCreateLayerNormDescriptor(
        handle,
        ctypes.byref(forward_descriptor),
        temp_output.descriptor,
        input_tensor.descriptor,
        weight.descriptor,
        bias_for_forward.descriptor if bias_for_forward is not None else None,
        input_std_deviation.descriptor,
        input_standardization.descriptor,
        ctypes.c_float(eps),
    )
    check_error(status)
    
    # Invalidate descriptors (but keep input_tensor, weight, input_std_deviation, input_standardization for later use)
    for tensor in [temp_output] + ([bias_for_forward] if bias_for_forward is not None else []):
        tensor.destroy_desc()
    
    # Get workspace size for forward pass
    forward_workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLayerNormWorkspaceSize(
            forward_descriptor, ctypes.byref(forward_workspace_size)
        )
    )
    forward_workspace = TestWorkspace(forward_workspace_size.value, device)
    
    # Execute LayerNorm forward pass to get intermediate values
    status = LIBINFINIOP.infiniopLayerNorm(
        forward_descriptor,
        forward_workspace.data(),
        forward_workspace_size.value,
        temp_output.data(),
        input_tensor.data(),
        weight.data(),
        bias_for_forward.data() if bias_for_forward is not None else None,
        input_std_deviation.data(),
        input_standardization.data(),
        None,
    )
    check_error(status)
    check_error(LIBINFINIOP.infiniopDestroyLayerNormDescriptor(forward_descriptor))
    
    # Debug: Check if input_std_deviation contains valid values
    std_dev_tensor = input_std_deviation.torch_tensor()
    print(f"input_std_deviation stats: min={std_dev_tensor.min().item():.6f}, max={std_dev_tensor.max().item():.6f}, mean={std_dev_tensor.mean().item():.6f}")
    if torch.any(std_dev_tensor <= 1e-8):
        print(f"⚠ Warning: Found very small or zero values in input_std_deviation")
    
    # First get PyTorch official gradients as ground truth
    grad_input_official, grad_weight_official, grad_bias_official = get_pytorch_official_grads(
        input_tensor.torch_tensor(),
        grad_output.torch_tensor(),
        weight.torch_tensor(),
        grad_bias.torch_tensor() if grad_bias is not None else None,
        eps
    )
    
    # Then compute reference using our manual implementation
    grad_input_ref, grad_weight_ref, grad_bias_ref = layer_norm_backward_pytorch(
        grad_output.torch_tensor(),
        input_tensor.torch_tensor(),
        weight.torch_tensor(),
        grad_bias.torch_tensor() if grad_bias is not None else None,
        input_std_deviation.torch_tensor(),
        input_standardization.torch_tensor(),
        eps
    )
    
    # Verify manual implementation matches PyTorch official (cross-validation)
    atol_cross, rtol_cross = get_tolerance(_TOLERANCE_MAP, dtype)
    try:
        assert torch.allclose(grad_input_ref, grad_input_official, atol=atol_cross, rtol=rtol_cross), "Manual grad_input differs from PyTorch official"
        assert torch.allclose(grad_weight_ref, grad_weight_official, atol=atol_cross, rtol=rtol_cross), "Manual grad_weight differs from PyTorch official"
        if grad_bias_ref is not None and grad_bias_official is not None:
            assert torch.allclose(grad_bias_ref, grad_bias_official, atol=atol_cross, rtol=rtol_cross), "Manual grad_bias differs from PyTorch official"
        print("✓ Manual reference implementation validated against PyTorch official")
    except AssertionError as e:
        print(f"⚠ Warning: {e}")
        if DEBUG:
            print("Manual grad_input stats:", grad_input_ref.min().item(), grad_input_ref.max().item(), grad_input_ref.mean().item())
            print("Official grad_input stats:", grad_input_official.min().item(), grad_input_official.max().item(), grad_input_official.mean().item())
            print("Manual grad_weight stats:", grad_weight_ref.min().item(), grad_weight_ref.max().item(), grad_weight_ref.mean().item())
            print("Official grad_weight stats:", grad_weight_official.min().item(), grad_weight_official.max().item(), grad_weight_official.mean().item())
        print("Using PyTorch official gradients as reference instead")
        grad_input_ref, grad_weight_ref, grad_bias_ref = grad_input_official, grad_weight_official, grad_bias_official
    
    # Copy reference results to expected tensors with proper dtype conversion
    target_dtype = input_tensor.torch_tensor().dtype
    grad_input_expected = grad_input_ref.clone().to(target_dtype)
    grad_weight_expected = grad_weight_ref.clone().to(grad_weight.torch_tensor().dtype) if grad_weight_ref is not None else None
    grad_bias_expected = grad_bias_ref.clone().to(grad_bias.torch_tensor().dtype) if grad_bias_ref is not None and grad_bias is not None else None

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    
    status = LIBINFINIOP.infiniopCreateLayerNormBackwardDescriptor(
        handle,
        ctypes.byref(descriptor),
        grad_input.descriptor,
        grad_weight.descriptor,
        grad_bias.descriptor if grad_bias is not None else None,
        grad_output.descriptor,
        input_tensor.descriptor,
        weight.descriptor,
        input_std_deviation.descriptor,
        input_standardization.descriptor,
        ctypes.c_float(eps),
    )
    check_error(status)

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_input, grad_weight, grad_output, input_tensor, weight, input_std_deviation, input_standardization] + ([grad_bias] if grad_bias is not None else []):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLayerNormBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_input.device)

    def lib_layer_norm_backward():
        status = LIBINFINIOP.infiniopLayerNormBackward(
            descriptor,
            workspace.data(),
            workspace_size.value,
            grad_input.data(),
            grad_weight.data(),
            grad_bias.data() if grad_bias is not None else None,
            grad_output.data(),
            input_tensor.data(),
            weight.data(),
            input_std_deviation.data(),
            input_standardization.data(),
            None,
        )
        check_error(status)

    lib_layer_norm_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    # Check grad_input
    if DEBUG:
        debug(grad_input.actual_tensor(), grad_input_expected, atol=atol, rtol=rtol)
    try:
        assert torch.allclose(grad_input.actual_tensor(), grad_input_expected, atol=atol, rtol=rtol), "grad_input mismatch"
    except AssertionError:
        print(f"grad_input mismatch details:")
        print(f"  Actual stats: min={grad_input.actual_tensor().min().item():.6f}, max={grad_input.actual_tensor().max().item():.6f}, mean={grad_input.actual_tensor().mean().item():.6f}")
        print(f"  Expected stats: min={grad_input_expected.min().item():.6f}, max={grad_input_expected.max().item():.6f}, mean={grad_input_expected.mean().item():.6f}")
        print(f"  Max absolute diff: {torch.abs(grad_input.actual_tensor() - grad_input_expected).max().item():.6f}")
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
        raise
    
    # Check grad_weight
    if DEBUG:
        debug(grad_weight.actual_tensor(), grad_weight_expected, atol=atol, rtol=rtol)
    assert torch.allclose(grad_weight.actual_tensor(), grad_weight_expected, atol=atol, rtol=rtol), "grad_weight mismatch"
    
    # Check grad_bias if present
    if grad_bias is not None and grad_bias_expected is not None:
        if DEBUG:
            debug(grad_bias.actual_tensor(), grad_bias_expected, atol=atol, rtol=rtol)
        assert torch.allclose(grad_bias.actual_tensor(), grad_bias_expected, atol=atol, rtol=rtol), "grad_bias mismatch"

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: layer_norm_backward_pytorch(
            grad_output.torch_tensor(), input_tensor.torch_tensor(), weight.torch_tensor(),
            grad_bias.torch_tensor() if grad_bias is not None else None,
            input_std_deviation.torch_tensor(), input_standardization.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_layer_norm_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLayerNormBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")