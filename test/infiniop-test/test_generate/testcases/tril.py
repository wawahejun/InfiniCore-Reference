import torch
import gguf
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_tril(input: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    return torch.tril(input, diagonal=diagonal)

class TrilTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: torch.Tensor,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        diagonal: int,
    ):
        super().__init__("tril")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.diagonal = diagonal

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)

        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )

        # Add diagonal parameter
        test_writer.add_int32(test_writer.gguf_key("diagonal"), self.diagonal)

        if self.input.dtype == torch.bfloat16:
            bits = self.input.view(torch.uint16).cpu().numpy()
            arr_in = bits.view(bfloat16)
            raw_dtype = np_dtype_to_ggml(bfloat16)
        else:
            arr_in = self.input.cpu().numpy()
            raw_dtype = np_dtype_to_ggml(arr_in.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            arr_in,
            raw_dtype=raw_dtype,
        )

        if self.output.dtype == torch.bfloat16:
            bits = self.output.view(torch.uint16).cpu().numpy()
            arr_out = bits.view(bfloat16)
            raw_dtype = np_dtype_to_ggml(bfloat16)
        else:
            arr_out = self.output.cpu().numpy()
            raw_dtype = np_dtype_to_ggml(arr_out.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            arr_out,
            raw_dtype=raw_dtype,
        )
        
        # Add answer tensor (using double precision for higher accuracy)
        ans_numpy = self.output.detach().cpu().double().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(ans_numpy.dtype),
        )

def test(shape, diagonal, dtype=torch.float32):
    """Generate a single test case for tril operation"""
    # Generate input tensor based on dtype
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # Integer types: use randint
        if dtype == torch.int8:
            input_tensor = torch.randint(-50, 50, shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            input_tensor = torch.randint(-1000, 1000, shape, dtype=torch.int32).to(dtype)
        else:
            input_tensor = torch.randint(-50, 50, shape, dtype=dtype)
    elif dtype == torch.bool:
        # Boolean type
        input_tensor = torch.randint(0, 2, shape, dtype=torch.int32).bool()
    elif dtype == torch.bfloat16:
        # bfloat16: generate as float32 then convert
        input_tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
    else:
        # Float types: use randn
        input_tensor = torch.randn(shape, dtype=dtype) * 2.0
    
    # Compute expected output using PyTorch reference
    output_tensor = reference_tril(input_tensor, diagonal)
    
    return TrilTestCase(
        input=input_tensor,
        shape_input=list(shape),
        stride_input=None,  # Use contiguous strides
        output=output_tensor,
        shape_output=list(shape),
        stride_output=None,  # Use contiguous strides
        diagonal=diagonal,
    )

if __name__ == "__main__":
    import gguf
    test_writer = InfiniopTestWriter("tril.gguf")
    test_cases = []
    
    # Test cases for 2D tensors only (as required)
    _TEST_CASES_ = [
        # (shape, diagonal)
        {"shape": (3, 3), "diagonal": 0},
        {"shape": (3, 3), "diagonal": 1},
        {"shape": (3, 3), "diagonal": -1},
        {"shape": (4, 4), "diagonal": 0},
        {"shape": (4, 4), "diagonal": 2},
        {"shape": (4, 4), "diagonal": -2},
        {"shape": (5, 3), "diagonal": 0},
        {"shape": (3, 5), "diagonal": 0},
        {"shape": (8, 8), "diagonal": 0},
        {"shape": (16, 16), "diagonal": 0},
        {"shape": (32, 32), "diagonal": 0},
        {"shape": (1, 1), "diagonal": 0},
        {"shape": (2, 1), "diagonal": 0},
        {"shape": (1, 2), "diagonal": 0},
    ]

    _TENSOR_DTYPES_ = [
        torch.float64,
        torch.float32, 
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.int16,
        torch.bool,
    ]

    for dtype in _TENSOR_DTYPES_:
        for test_config in _TEST_CASES_:
            test_case = test(
                shape=test_config["shape"],
                diagonal=test_config["diagonal"],
                dtype=dtype
            )
            test_cases.append(test_case)

    print(f"Generated {len(test_cases)} test cases for Tril operator")
    test_writer.add_tests(test_cases)
    test_writer.save()
    print("Tril test cases saved to tril.gguf")