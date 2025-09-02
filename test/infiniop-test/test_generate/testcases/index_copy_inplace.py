import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_index_copy_inplace(target: torch.Tensor, source: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """Reference implementation using PyTorch index_copy_"""
    result = target.clone()
    # Convert index to long type as required by PyTorch
    index_long = index.long()
    result.index_copy_(dim, index_long, source)
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

class IndexCopyInplaceTestCase(InfiniopTestCase):
    def __init__(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        index: torch.Tensor,
        dim: int,
        target_shape: List[int],
        source_shape: List[int],
        index_shape: List[int],
        target_stride: List[int] | None,
        source_stride: List[int] | None,
        index_stride: List[int] | None,
    ):
        super().__init__("index_copy_inplace")
        self.target = target
        self.source = source
        self.index = index
        self.dim = dim
        self.target_shape = target_shape
        self.source_shape = source_shape
        self.index_shape = index_shape
        self.target_stride = target_stride
        self.source_stride = source_stride
        self.index_stride = index_stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add dimension parameter
        test_writer.add_array(test_writer.gguf_key("dim"), [self.dim])
        
        # Add shapes
        test_writer.add_array(test_writer.gguf_key("target.shape"), self.target_shape)
        test_writer.add_array(test_writer.gguf_key("source.shape"), self.source_shape)
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.index_shape)
        
        # Add strides
        target_strides = self.target_stride if self.target_stride is not None else contiguous_gguf_strides(self.target_shape)
        source_strides = self.source_stride if self.source_stride is not None else contiguous_gguf_strides(self.source_shape)
        index_strides = self.index_stride if self.index_stride is not None else contiguous_gguf_strides(self.index_shape)
        

        
        test_writer.add_array(test_writer.gguf_key("target.strides"), gguf_strides(*target_strides))
        test_writer.add_array(test_writer.gguf_key("source.strides"), gguf_strides(*source_strides))
        test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*index_strides))
        
        # Convert tensors to numpy
        if self.target.dtype == torch.bfloat16:
            target_numpy = self.target.detach().cpu().float().numpy().astype(bfloat16)
        else:
            target_numpy = self.target.detach().cpu().numpy()
            
        if self.source.dtype == torch.bfloat16:
            source_numpy = self.source.detach().cpu().float().numpy().astype(bfloat16)
        else:
            source_numpy = self.source.detach().cpu().numpy()
            
        index_numpy = self.index.detach().cpu().numpy()
        
        # Add tensors
        test_writer.add_tensor(
            test_writer.gguf_key("target"),
            target_numpy,
            raw_dtype=np_dtype_to_ggml(target_numpy.dtype),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("source"),
            source_numpy,
            raw_dtype=np_dtype_to_ggml(source_numpy.dtype),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("index"),
            index_numpy,
            raw_dtype=np_dtype_to_ggml(index_numpy.dtype),
        )
        
        # Generate expected result
        expected_output = reference_index_copy_inplace(self.target, self.source, self.index, self.dim)
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
    test_writer = InfiniopTestWriter("index_copy_inplace.gguf")
    test_cases: List[IndexCopyInplaceTestCase] = []

    # Test cases: (target_shape, source_shape, index_shape, dim, target_stride, source_stride, index_stride)
    _TEST_CASES_ = [
        # Basic contiguous tests - 保留少量基本测试
        ((10,), (3,), (3,), 0, None, None, None),
        ((5, 8), (3, 8), (3,), 0, None, None, None),
        ((8, 5), (8, 3), (3,), 1, None, None, None),
        ((4, 6, 8), (2, 6, 8), (2,), 0, None, None, None),
        ((6, 4, 8), (6, 2, 8), (2,), 1, None, None, None),
        ((6, 8, 4), (6, 8, 2), (2,), 2, None, None, None),
        
        # Non-contiguous stride tests - target tensor with non-contiguous strides
        ((4, 4), (2, 4), (2,), 0, (4, 1), None, None),  # target: column-major
        ((4, 4), (4, 2), (2,), 1, (4, 1), None, None),  # target: column-major
        ((6, 8), (3, 8), (3,), 0, (8, 1), None, None),  # target: row-major
        ((8, 6), (8, 3), (3,), 1, (1, 8), None, None),  # target: column-major
        
        # Non-contiguous stride tests - source tensor with non-contiguous strides
        ((4, 4), (2, 4), (2,), 0, None, (4, 1), None),  # source: column-major
        ((4, 4), (4, 2), (2,), 1, None, (4, 1), None),  # source: column-major
        ((6, 8), (3, 8), (3,), 0, None, (8, 1), None),  # source: row-major
        ((8, 6), (8, 3), (3,), 1, None, (1, 8), None),  # source: column-major
        
        # 3D non-contiguous tests
        ((4, 6, 8), (2, 6, 8), (2,), 0, (48, 8, 1), None, None),  # target: row-major
        ((4, 6, 8), (2, 6, 8), (2,), 0, (1, 4, 24), None, None),  # target: column-major
        ((4, 6, 8), (2, 6, 8), (2,), 0, None, (48, 8, 1), None),  # source: row-major
        ((4, 6, 8), (2, 6, 8), (2,), 0, None, (1, 4, 24), None),  # source: column-major
        
        # Mixed non-contiguous tests - both target and source
        ((4, 4), (2, 4), (2,), 0, (4, 1), (4, 1), None),  # both column-major
        ((6, 8), (3, 8), (3,), 0, (8, 1), (1, 6), None),  # target: row-major, source: column-major
        ((8, 6), (8, 3), (3,), 1, (1, 8), (6, 1), None),  # target: column-major, source: row-major
        
        # Edge cases
        ((1,), (1,), (1,), 0, None, None, None),
        ((2, 3, 4), (1, 3, 4), (1,), 0, None, None, None),
        
        # Large tensor with non-contiguous strides
        ((100, 100), (20, 100), (20,), 0, (100, 1), None, None),  # target: row-major
        ((100, 100), (100, 20), (20,), 1, (1, 100), None, None),  # target: column-major
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
            for target_shape, source_shape, index_shape, dim, target_stride, source_stride, index_stride in _TEST_CASES_:
                # Generate test data with custom strides
                if target_stride is not None:
                    target_data = create_strided_tensor(target_shape, target_stride, tensor_dtype)
                else:
                    if tensor_dtype in [torch.int32, torch.int64]:
                        target_data = torch.randint(-50, 50, target_shape, dtype=tensor_dtype)
                    else:
                        target_data = torch.randn(target_shape, dtype=torch.float32) * 2.0
                        target_data = target_data.to(tensor_dtype)
                
                if source_stride is not None:
                    source_data = create_strided_tensor(source_shape, source_stride, tensor_dtype)
                else:
                    if tensor_dtype in [torch.int32, torch.int64]:
                        source_data = torch.randint(-50, 50, source_shape, dtype=tensor_dtype)
                    else:
                        source_data = torch.randn(source_shape, dtype=torch.float32) * 2.0
                        source_data = source_data.to(tensor_dtype)
                
                # Generate valid indices for the given dimension
                max_index = target_shape[dim] - 1
                # Skip test cases where we need more indices than available
                if index_shape[0] > max_index + 1:
                    continue  # Skip this test case
                
                # Generate indices with custom strides if specified
                if index_stride is not None:
                    # For non-contiguous index tensors, we need to be more careful
                    # Create contiguous indices first
                    indices_base = torch.randperm(max_index + 1, dtype=index_dtype)[:index_shape[0]]
                    # Create storage for strided index tensor, fill with valid default values
                    max_offset = (index_shape[0] - 1) * index_stride[0] if index_shape[0] > 1 else 0
                    storage_size = max_offset + 1
                    # Initialize with the first valid index to avoid using 0 as default
                    index_storage = torch.full((storage_size,), indices_base[0].item(), dtype=index_dtype)
                    # Fill the storage with valid indices at strided positions
                    for i in range(index_shape[0]):
                        index_storage[i * index_stride[0]] = indices_base[i]
                    indices = torch.as_strided(index_storage, index_shape, index_stride)
                else:
                    indices = torch.randperm(max_index + 1, dtype=index_dtype)[:index_shape[0]]
                
                test_case = IndexCopyInplaceTestCase(
                    target=target_data,
                    source=source_data,
                    index=indices,
                    dim=dim,
                    target_shape=list(target_shape),
                    source_shape=list(source_shape),
                    index_shape=list(index_shape),
                    target_stride=list(target_stride) if target_stride is not None else None,
                    source_stride=list(source_stride) if source_stride is not None else None,
                    index_stride=list(index_stride) if index_stride is not None else None,
                )
                test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
    print(f"Generated {len(test_cases)} test cases for IndexCopyInplace operator")