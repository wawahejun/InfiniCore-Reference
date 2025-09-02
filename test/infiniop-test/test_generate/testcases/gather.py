import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_gather(input_tensor: torch.Tensor, index_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """参考实现的gather操作"""
    return torch.gather(input_tensor, dim, index_tensor)

class GatherTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        index_tensor: torch.Tensor,
        dim: int,
    ):
        super().__init__("gather")
        self.input = input_tensor
        self.index = index_tensor
        self.dim = dim
        
        # 计算预期输出
        self.output = reference_gather(input_tensor, index_tensor, dim)
        
    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # 写入dim属性
        test_writer.add_array(test_writer.gguf_key("dim"), [self.dim])
        
        # 写入形状信息
        test_writer.add_array(test_writer.gguf_key("input.shape"), list(self.input.shape))
        test_writer.add_array(test_writer.gguf_key("index.shape"), list(self.index.shape))
        test_writer.add_array(test_writer.gguf_key("output.shape"), list(self.output.shape))
        
        # 写入步长信息
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*contiguous_gguf_strides(self.input.shape)))
        test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*contiguous_gguf_strides(self.index.shape)))
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*contiguous_gguf_strides(self.output.shape)))
        
        # 处理输入张量
        if self.input.dtype == torch.bfloat16:
            input_numpy = self.input.detach().cpu().float().numpy().astype(bfloat16)
        else:
            input_numpy = self.input.detach().cpu().numpy()
        
        # 处理索引张量（始终为整数类型）
        index_numpy = self.index.detach().cpu().numpy()
        
        # 处理输出张量
        if self.output.dtype == torch.bfloat16:
            output_numpy = self.output.detach().cpu().float().numpy().astype(bfloat16)
        else:
            output_numpy = self.output.detach().cpu().numpy()
        
        # 写入张量数据
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
            test_writer.gguf_key("output"),
            output_numpy,
            raw_dtype=np_dtype_to_ggml(output_numpy.dtype),
        )
        
        # 计算并写入答案（使用双精度以提高精度）
        ans_numpy = self.output.detach().cpu().double().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(ans_numpy.dtype),
        )

def test(input_shape, index_shape, dim, dtype=torch.float32):
    """Generate a single test case"""
    # Create input tensor based on data type
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # Integer types: use randint with appropriate range
        if dtype == torch.int8:
            input_tensor = torch.randint(-50, 50, input_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            input_tensor = torch.randint(-1000, 1000, input_shape, dtype=torch.int32).to(dtype)
        else:
            input_tensor = torch.randint(-50, 50, input_shape, dtype=dtype)

    elif dtype == torch.bool:
        # Boolean type
        input_tensor = torch.randint(0, 2, input_shape, dtype=torch.int32).bool()
    elif dtype == torch.bfloat16:
        # bfloat16: generate as float32 then convert
        input_tensor = torch.randn(input_shape, dtype=torch.float32).to(dtype)
    else:
        # Float types: use randn
        input_tensor = torch.randn(input_shape, dtype=dtype) * 2.0
    
    # Create index tensor - indices must be valid for the gather dimension
    max_index = input_shape[dim] - 1
    index_tensor = torch.randint(0, max_index + 1, index_shape, dtype=torch.int64)
    
    return GatherTestCase(
        input_tensor=input_tensor,
        index_tensor=index_tensor,
        dim=dim,
    )

if __name__ == "__main__":
    import gguf
    test_writer = InfiniopTestWriter("gather.gguf")
    test_cases = []
    
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    _TEST_CASES_ = [
        # 1D cases
        {"input_shape": [10], "index_shape": [5], "dim": 0},
        {"input_shape": [8], "index_shape": [3], "dim": 0},
        
        # Basic 2D cases
        {"input_shape": [4, 6], "index_shape": [4, 3], "dim": 1},
        {"input_shape": [5, 4], "index_shape": [3, 4], "dim": 0},
        {"input_shape": [6, 8], "index_shape": [6, 4], "dim": 1},
        {"input_shape": [7, 3], "index_shape": [4, 3], "dim": 0},
        
        # 3D cases
        {"input_shape": [2, 3, 4], "index_shape": [2, 2, 4], "dim": 1},
        {"input_shape": [3, 4, 5], "index_shape": [3, 4, 3], "dim": 2},
        {"input_shape": [4, 3, 6], "index_shape": [2, 3, 6], "dim": 0},
        {"input_shape": [2, 5, 3], "index_shape": [2, 3, 3], "dim": 1},
        
        # 4D cases
        {"input_shape": [2, 3, 4, 5], "index_shape": [2, 3, 4, 2], "dim": 3},
        {"input_shape": [3, 2, 4, 3], "index_shape": [2, 2, 4, 3], "dim": 0},
        {"input_shape": [2, 4, 3, 5], "index_shape": [2, 2, 3, 5], "dim": 1},
        
        # Edge cases - small dimensions
        {"input_shape": [1, 5], "index_shape": [1, 3], "dim": 1},  # single row
        {"input_shape": [5, 1], "index_shape": [3, 1], "dim": 0},  # single column
        {"input_shape": [1, 1], "index_shape": [1, 1], "dim": 0},  # minimal case
        {"input_shape": [1, 1], "index_shape": [1, 1], "dim": 1},  # minimal case dim 1
        
        # Edge cases - large dimensions
        {"input_shape": [100, 50], "index_shape": [100, 20], "dim": 1},
        {"input_shape": [50, 100], "index_shape": [30, 100], "dim": 0},
        
        # Edge cases - single element gather
        {"input_shape": [10, 8], "index_shape": [10, 1], "dim": 1},
        {"input_shape": [8, 10], "index_shape": [1, 10], "dim": 0},
        
        # Complex 3D edge cases
        {"input_shape": [1, 1, 5], "index_shape": [1, 1, 3], "dim": 2},
        {"input_shape": [5, 1, 1], "index_shape": [3, 1, 1], "dim": 0},
        {"input_shape": [1, 5, 1], "index_shape": [1, 3, 1], "dim": 1},
    ]
    
    _TENSOR_DTYPES_ = [
        # Float types
        torch.float64,
        torch.float32, 
        torch.float16,
        torch.bfloat16,
        # Integer types
        torch.int32,
        torch.int64,
        torch.int8,
        torch.int16,
        # Bool type
        torch.bool,
    ]
    
    for dtype in _TENSOR_DTYPES_:
        for test_config in _TEST_CASES_:
            test_case = test(
                input_shape=test_config["input_shape"],
                index_shape=test_config["index_shape"],
                dim=test_config["dim"],
                dtype=dtype
            )
            test_cases.append(test_case)
    
    print(f"Generated {len(test_cases)} test cases for Gather operator")
    test_writer.add_tests(test_cases)
    test_writer.save()
    print("Gather test cases saved to gather.gguf")