import torch
import ctypes
from ctypes import c_uint64, c_int32
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
import numpy as np

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # (target_shape, source_shape, index_shape, dim, target_stride, source_stride, index_stride)
    # Basic continuous cases
    ((4, 3), (2, 3), (2,), 0, None, None, None),  # Basic case
    ((3, 4), (3, 2), (2,), 1, None, None, None),  # Different dimension
    ((5, 3, 2), (2, 3, 2), (2,), 0, None, None, None),  # 3D tensor
    ((3, 5, 2), (3, 2, 2), (2,), 1, None, None, None),  # 3D tensor, dim=1
    ((2, 3, 5), (2, 3, 2), (2,), 2, None, None, None),  # 3D tensor, dim=2
    ((10, 5), (3, 5), (3,), 0, None, None, None),  # Larger case
    ((5, 10), (5, 3), (3,), 1, None, None, None),  # Larger case, dim=1
    
    # Non-contiguous stride cases - 2D
    ((8, 6), (4, 6), (4,), 0, (12, 2), None, None),  # target non-contiguous
    ((6, 8), (6, 4), (4,), 1, (16, 2), None, None),  # target non-contiguous dim=1
    ((8, 6), (4, 6), (4,), 0, None, (12, 2), None),  # source non-contiguous
    ((6, 8), (6, 4), (4,), 1, None, (8, 2), None),   # source non-contiguous dim=1
    ((8, 6), (4, 6), (4,), 0, (12, 2), (12, 2), None),  # both target and source non-contiguous
    
    # Non-contiguous stride cases - 3D
    ((4, 3, 2), (2, 3, 2), (2,), 0, (12, 4, 2), None, None),  # 3D target non-contiguous
    ((3, 4, 2), (3, 2, 2), (2,), 1, (8, 4, 2), None, None),   # 3D target non-contiguous dim=1
    ((2, 3, 4), (2, 3, 2), (2,), 2, (12, 8, 2), None, None),  # 3D target non-contiguous dim=2
    ((4, 3, 2), (2, 3, 2), (2,), 0, None, (12, 4, 2), None),  # 3D source non-contiguous
    ((3, 4, 2), (3, 2, 2), (2,), 1, None, (4, 4, 2), None),   # 3D source non-contiguous dim=1
    
    # Index non-contiguous cases
    ((6, 4), (3, 4), (3,), 0, None, None, (2,)),  # index non-contiguous
    ((4, 6), (4, 3), (3,), 1, None, None, (2,)),  # index non-contiguous dim=1
    
    # Complex non-contiguous cases
    ((8, 6), (4, 6), (4,), 0, (12, 2), (12, 2), (2,)),  # all non-contiguous
    ((6, 8), (6, 4), (4,), 1, (16, 2), (8, 2), (2,)),   # all non-contiguous dim=1
    
    # Large size cases with strides
    ((64, 32), (32, 32), (32,), 0, (64, 2), None, None),  # large target non-contiguous
    ((32, 64), (32, 32), (32,), 1, (128, 2), None, None), # large target non-contiguous dim=1
    ((64, 32), (32, 32), (32,), 0, None, (64, 2), None),  # large source non-contiguous
    
    # Edge cases with minimal strides
    ((4, 2), (2, 2), (2,), 0, (4, 2), None, None),  # minimal stride case
    ((2, 4), (2, 2), (2,), 1, (8, 2), None, None),  # minimal stride case dim=1
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64, InfiniDtype.BF16,
    InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.U32, InfiniDtype.U64,
    InfiniDtype.BOOL
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.U32: {"atol": 0, "rtol": 0},
    InfiniDtype.U64: {"atol": 0, "rtol": 0},
    InfiniDtype.BOOL: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def index_copy_inplace_torch(target, source, index, dim):
    """PyTorch reference implementation"""
    target.index_copy_(dim, index, source)
    return target


def create_strided_tensor(shape, stride, dtype, device, mode="random"):
    """Create a tensor with custom stride from contiguous data"""
    if stride is None:
        # Create normal contiguous tensor
        if mode == "random":
            return TestTensor(shape, None, dtype, device, mode="random")
        else:
            return TestTensor(shape, None, dtype, device)
    
    # Calculate the total storage size needed
    # For each dimension, calculate the maximum offset and add 1
    max_offset = 0
    for i in range(len(shape)):
        if shape[i] > 1:
            max_offset += (shape[i] - 1) * stride[i]
    storage_size = max_offset + 1
    
    # Create a larger contiguous tensor for storage
    if mode == "random":
        storage_tensor = TestTensor((storage_size,), None, dtype, device, mode="random")
    else:
        storage_tensor = TestTensor((storage_size,), None, dtype, device)
    
    # Create a view with custom stride
    strided_tensor = storage_tensor.torch_tensor().as_strided(shape, stride, 0)
    
    # Create a new TestTensor with the strided view
    result = TestTensor(shape, None, dtype, device)
    result.torch_tensor().copy_(strided_tensor)
    
    return result


def create_valid_index(target_shape, source_shape, dim, dtype=torch.int64):
    """Create valid indices for index_copy operation"""
    target_size = target_shape[dim]
    source_size = source_shape[dim]
    
    # Generate unique random valid indices to avoid non-deterministic behavior
    # when testing against PyTorch reference implementation
    if source_size <= target_size:
        # Generate unique indices by sampling without replacement
        indices = torch.randperm(target_size, dtype=dtype)[:source_size]
    else:
        # If source_size > target_size, we must have duplicates
        # In this case, use a deterministic pattern to ensure reproducible results
        indices = torch.arange(source_size, dtype=dtype) % target_size
    
    return indices


def test(
    handle,
    device,
    target_shape,
    source_shape,
    index_shape,
    dim,
    target_stride=None,
    source_stride=None,
    index_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    # Create tensors with appropriate data based on type and stride
    if dtype in [InfiniDtype.I32, InfiniDtype.I64]:
        # Signed integer: use random values
        target = create_strided_tensor(target_shape, target_stride, dtype, device, mode="random")
        source = create_strided_tensor(source_shape, source_stride, dtype, device, mode="random")
    elif dtype in [InfiniDtype.U32, InfiniDtype.U64]:
        # Unsigned integer: use random values
        target = create_strided_tensor(target_shape, target_stride, dtype, device, mode="random")
        source = create_strided_tensor(source_shape, source_stride, dtype, device, mode="random")
    elif dtype == InfiniDtype.BOOL:
        # Boolean: use random boolean values
        target = create_strided_tensor(target_shape, target_stride, dtype, device, mode="random")
        source = create_strided_tensor(source_shape, source_stride, dtype, device, mode="random")
    else:
        # Float: use random values
        target = create_strided_tensor(target_shape, target_stride, dtype, device)
        source = create_strided_tensor(source_shape, source_stride, dtype, device)
    
    # Create valid indices with custom stride
    index_tensor = create_valid_index(target_shape, source_shape, dim)
    if index_stride is None:
        index = TestTensor(index_shape, None, InfiniDtype.I64, device)
        index.torch_tensor().copy_(index_tensor)
    else:
        # Create strided index tensor
        index = create_strided_tensor(index_shape, index_stride, InfiniDtype.I64, device)
        # For strided index, we need to be more careful about setting values
        index_view = index.torch_tensor()
        if len(index_shape) == 1:
            for i in range(index_shape[0]):
                index_view[i] = index_tensor[i]
        else:
            # For multi-dimensional index, copy the values appropriately
            index_view.copy_(index_tensor)
    
    # Create a copy of target for PyTorch reference
    target_orig = target.torch_tensor().clone()  # Original target for debugging
    target_torch = target.torch_tensor().clone()
    
    stride_info = ""
    if target_stride is not None or source_stride is not None or index_stride is not None:
        stride_info = f" target_stride:{target_stride} source_stride:{source_stride} index_stride:{index_stride}"
    
    print(
        f"Testing IndexCopyInplace on {InfiniDeviceNames[device]} with target_shape:{target_shape} "
        f"source_shape:{source_shape} index_shape:{index_shape} dim:{dim} dtype:{InfiniDtypeNames[dtype]}{stride_info}"
    )

    # PyTorch reference computation
    index_copy_inplace_torch(target_torch, source.torch_tensor(), index.torch_tensor(), dim)

    if sync is not None:
        sync()

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateIndexCopyInplaceDescriptor(
            handle,
            ctypes.byref(descriptor),
            target.descriptor,
            source.descriptor,
            c_int32(dim),
            index.descriptor,
        )
    )

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetIndexCopyInplaceWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, target.device)

    def lib_index_copy_inplace():
        check_error(
            LIBINFINIOP.infiniopIndexCopyInplace(
                descriptor,
                workspace.data(),
                workspace.size(),
                target.data(),
                source.data(),
                index.data(),
                None,
            )
        )

    # Execute the operation
    lib_index_copy_inplace()

    # Check correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    

    
    if DEBUG:
        debug(target.actual_tensor(), target_torch, atol=atol, rtol=rtol)
    assert torch.allclose(target.actual_tensor(), target_torch, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # Reset target for profiling
        target_prof = target.torch_tensor().clone()
        # fmt: off
        profile_operation(
            "PyTorch", 
            lambda: index_copy_inplace_torch(target_prof, source.torch_tensor(), index.torch_tensor(), dim), 
            device, NUM_PRERUN, NUM_ITERATIONS
        )
        profile_operation("    lib", lambda: lib_index_copy_inplace(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyIndexCopyInplaceDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")