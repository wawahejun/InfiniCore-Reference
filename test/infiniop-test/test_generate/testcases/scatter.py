import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_scatter(input_tensor: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch scatter_"""
    result = input_tensor.clone()
    # Convert index to long type as required by PyTorch
    index_long = index.long()
    result.scatter_(dim, index_long, src)
    return result

def row_major_strides(shape):
    """生成张量的行优先stride
    
    Args:
        shape: 张量形状
    
    Returns:
        行优先strides列表
    """
    stride = 1
    strides = [1]
    for dim in reversed(shape[1:]):
        stride *= dim
        strides.insert(0, stride)
    return strides

def column_major_strides(shape):
    """生成张量的列优先stride
    
    Args:
        shape: 张量形状
    
    Returns:
        列优先strides列表
    """
    stride = 1
    strides = [stride]
    for dim in shape[:-1]:
        stride *= dim
        strides.append(stride)
    return strides

def create_strided_tensor(shape: List[int], stride: List[int], dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor with custom strides"""
    # Calculate the minimum storage size needed
    if len(shape) == 0 or len(stride) == 0:
        return torch.empty(shape, dtype=dtype)
    
    # Calculate the maximum offset that will be accessed
    max_offset = 0
    for i in range(len(shape)):
        if shape[i] > 1:  # Only consider dimensions with size > 1
            max_offset += (shape[i] - 1) * stride[i]
    
    # Total storage size needed is max_offset + 1
    total_size = max_offset + 1
    
    # Create storage and fill with random data
    if dtype in [torch.int32, torch.int64]:
        storage_data = torch.randint(-50, 50, (total_size,), dtype=dtype)
    else:
        storage_data = torch.randn(total_size, dtype=torch.float32) * 2.0
        storage_data = storage_data.to(dtype)
    
    # Create the strided view
    return torch.as_strided(storage_data, shape, stride)

class ScatterTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        index: torch.Tensor,
        src: torch.Tensor,
        dim: int,
        input_shape: List[int],
        index_shape: List[int],
        src_shape: List[int],
        input_stride: List[int] | None,
        index_stride: List[int] | None,
        src_stride: List[int] | None,
    ):
        super().__init__("scatter")
        self.input_tensor = input_tensor
        self.index = index
        self.src = src
        self.dim = dim
        self.input_shape = input_shape
        self.index_shape = index_shape
        self.src_shape = src_shape
        self.input_stride = input_stride
        self.index_stride = index_stride
        self.src_stride = src_stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add dimension parameter
        test_writer.add_array(test_writer.gguf_key("dim"), [self.dim])
        
        # Add shapes
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.index_shape)
        test_writer.add_array(test_writer.gguf_key("src.shape"), self.src_shape)
        
        # Add strides
        input_strides = self.input_stride if self.input_stride is not None else contiguous_gguf_strides(self.input_shape)
        index_strides = self.index_stride if self.index_stride is not None else contiguous_gguf_strides(self.index_shape)
        src_strides = self.src_stride if self.src_stride is not None else contiguous_gguf_strides(self.src_shape)
        
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*input_strides))
        test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*index_strides))
        test_writer.add_array(test_writer.gguf_key("src.strides"), gguf_strides(*src_strides))
        
        # Convert tensors to numpy
        if self.input_tensor.dtype == torch.bfloat16:
            input_numpy = self.input_tensor.detach().cpu().float().numpy().astype(bfloat16)
        else:
            input_numpy = self.input_tensor.detach().cpu().numpy()
            
        if self.src.dtype == torch.bfloat16:
            src_numpy = self.src.detach().cpu().float().numpy().astype(bfloat16)
        else:
            src_numpy = self.src.detach().cpu().numpy()
            
        index_numpy = self.index.detach().cpu().numpy()
        
        # Add tensors
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=np_dtype_to_ggml(input_numpy.dtype),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("index"),
            index_numpy,
            raw_dtype=np_dtype_to_ggml(index_numpy.dtype),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("src"),
            src_numpy,
            raw_dtype=np_dtype_to_ggml(src_numpy.dtype),
        )
        
        # Generate expected result
        expected_output = reference_scatter(self.input_tensor, self.dim, self.index, self.src)
        if expected_output.dtype == torch.bfloat16:
            expected_numpy = expected_output.detach().cpu().float().numpy().astype(bfloat16)
        else:
            expected_numpy = expected_output.detach().cpu().numpy()
            
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            expected_numpy,
            raw_dtype=np_dtype_to_ggml(expected_numpy.dtype),
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("scatter.gguf")
    test_cases: List[ScatterTestCase] = []

    # Test cases: (input_shape, index_shape, src_shape, dim, input_stride, index_stride, src_stride)
    _TEST_CASES_ = [
        # Basic contiguous tests - 1D cases
        ((10,), (3,), (3,), 0, None, None, None),
        ((8,), (4,), (4,), 0, None, None, None),
        
        # 2D cases
        ((5, 8), (3, 8), (3, 8), 0, None, None, None),
        ((8, 5), (8, 3), (8, 3), 1, None, None, None),
        ((4, 6), (2, 3), (2, 3), 0, None, None, None),
        ((6, 4), (3, 2), (3, 2), 1, None, None, None),
        
        # 3D cases
        ((4, 6, 8), (2, 6, 8), (2, 6, 8), 0, None, None, None),
        ((6, 4, 8), (6, 2, 8), (6, 2, 8), 1, None, None, None),
        ((6, 8, 4), (6, 8, 2), (6, 8, 2), 2, None, None, None),
        
        # Non-contiguous stride tests - input tensor with non-contiguous strides
        ((4, 4), (2, 4), (2, 4), 0, (4, 1), None, None),  # input: column-major
        ((4, 4), (4, 2), (4, 2), 1, (4, 1), None, None),  # input: column-major
        ((6, 8), (3, 8), (3, 8), 0, (8, 1), None, None),  # input: row-major
        ((8, 6), (8, 3), (8, 3), 1, (1, 8), None, None),  # input: column-major
        
        # Non-contiguous stride tests - src tensor with non-contiguous strides
        ((4, 4), (2, 4), (2, 4), 0, None, None, (4, 1)),  # src: column-major
        ((4, 4), (4, 2), (4, 2), 1, None, None, (4, 1)),  # src: column-major
        ((6, 8), (3, 8), (3, 8), 0, None, None, (8, 1)),  # src: row-major
        ((8, 6), (8, 3), (8, 3), 1, None, None, (1, 8)),  # src: column-major
        
        # 3D non-contiguous tests
        ((4, 6, 8), (2, 6, 8), (2, 6, 8), 0, (48, 8, 1), None, None),  # input: row-major
        ((4, 6, 8), (2, 6, 8), (2, 6, 8), 0, (1, 4, 24), None, None),  # input: column-major
        ((4, 6, 8), (2, 6, 8), (2, 6, 8), 0, None, None, (48, 8, 1)),  # src: row-major
        ((4, 6, 8), (2, 6, 8), (2, 6, 8), 0, None, None, (1, 4, 24)),  # src: column-major
        
        # Mixed non-contiguous tests - both input and src
        ((4, 4), (2, 4), (2, 4), 0, (4, 1), None, (4, 1)),  # both column-major
        ((6, 8), (3, 8), (3, 8), 0, (8, 1), None, (1, 6)),  # input: row-major, src: column-major
        ((8, 6), (8, 3), (8, 3), 1, (1, 8), None, (6, 1)),  # input: column-major, src: row-major
        
        # Edge cases
        ((1,), (1,), (1,), 0, None, None, None),
        ((2, 3, 4), (1, 3, 4), (1, 3, 4), 0, None, None, None),
        
        # Large tensor with non-contiguous strides
        ((100, 100), (20, 100), (20, 100), 0, (100, 1), None, None),  # input: row-major
        ((100, 100), (100, 20), (100, 20), 1, (1, 100), None, None),  # input: column-major
    ]

    # Data types to test
    _TENSOR_DTYPES_ = [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
    ]
    
    # Index tensor data types
    _INDEX_DTYPES_ = [
        torch.int32,
        torch.int64,
    ]

    for tensor_dtype in _TENSOR_DTYPES_:
        for index_dtype in _INDEX_DTYPES_:
            for input_shape, index_shape, src_shape, dim, input_stride, index_stride, src_stride in _TEST_CASES_:
                # Generate test data with custom strides
                if input_stride is not None:
                    input_data = create_strided_tensor(input_shape, input_stride, tensor_dtype)
                else:
                    if tensor_dtype in [torch.int32, torch.int64]:
                        input_data = torch.randint(-50, 50, input_shape, dtype=tensor_dtype)
                    else:
                        input_data = torch.randn(input_shape, dtype=torch.float32) * 2.0
                        input_data = input_data.to(tensor_dtype)
                
                if src_stride is not None:
                    src_data = create_strided_tensor(src_shape, src_stride, tensor_dtype)
                else:
                    if tensor_dtype in [torch.int32, torch.int64]:
                        src_data = torch.randint(-50, 50, src_shape, dtype=tensor_dtype)
                    else:
                        src_data = torch.randn(src_shape, dtype=torch.float32) * 2.0
                        src_data = src_data.to(tensor_dtype)
                
                # Generate valid indices for the given dimension
                max_index = input_shape[dim] - 1
                # Skip test cases where we need more indices than available
                if index_shape[dim] > max_index + 1:
                    continue  # Skip this test case
                
                # Generate unique indices for the scatter dimension
                def generate_unique_indices(shape, dim, max_val, dtype):
                    """Generate indices with unique values along the scatter dimension"""
                    indices = torch.zeros(shape, dtype=dtype)
                    
                    # For each slice along the scatter dimension, generate unique indices
                    if len(shape) == 1:
                        # 1D case: simply use unique random indices
                        unique_indices = torch.randperm(max_val + 1, dtype=dtype)[:shape[0]]
                        indices[:] = unique_indices
                    else:
                        # Multi-dimensional case: ensure uniqueness along scatter dimension
                        for idx in np.ndindex(shape):
                            if idx[dim] == 0:  # Only process the first slice along scatter dim
                                # Create a slice tuple for all positions except scatter dim
                                slice_idx = list(idx)
                                # Generate unique indices for this slice
                                num_indices = shape[dim]
                                if num_indices <= max_val + 1:
                                    unique_vals = torch.randperm(max_val + 1, dtype=dtype)[:num_indices]
                                    # Assign these unique values to all positions in this slice
                                    for i in range(num_indices):
                                        slice_idx[dim] = i
                                        indices[tuple(slice_idx)] = unique_vals[i]
                    return indices
                
                # Generate indices with custom strides if specified
                if index_stride is not None:
                    # For non-contiguous index tensors, we need to be more careful
                    # Create contiguous unique indices first
                    indices_base = generate_unique_indices(index_shape, dim, max_index, index_dtype)
                    # Create storage for strided index tensor
                    max_offset = 0
                    for i in range(len(index_shape)):
                        if index_shape[i] > 1:
                            max_offset += (index_shape[i] - 1) * index_stride[i]
                    storage_size = max_offset + 1
                    # Initialize with valid default values
                    index_storage = torch.zeros(storage_size, dtype=index_dtype)
                    # Fill the storage with unique indices at strided positions
                    for idx in np.ndindex(index_shape):
                        offset = sum(idx[i] * index_stride[i] for i in range(len(idx)))
                        index_storage[offset] = indices_base[idx]
                    indices = torch.as_strided(index_storage, index_shape, index_stride)
                else:
                    indices = generate_unique_indices(index_shape, dim, max_index, index_dtype)
                
                test_case = ScatterTestCase(
                    input_tensor=input_data,
                    index=indices,
                    src=src_data,
                    dim=dim,
                    input_shape=list(input_shape),
                    index_shape=list(index_shape),
                    src_shape=list(src_shape),
                    input_stride=list(input_stride) if input_stride is not None else None,
                    index_stride=list(index_stride) if index_stride is not None else None,
                    src_stride=list(src_stride) if src_stride is not None else None,
                )
                test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
    print(f"Generated {len(test_cases)} test cases for Scatter operator")