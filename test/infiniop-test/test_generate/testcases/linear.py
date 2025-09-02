import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Reference implementation of linear operation: y = x * w + b"""
    return torch.nn.functional.linear(input, weight, bias)

class LinearTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        shape_weight: List[int] | None,
        stride_weight: List[int] | None,
        shape_bias: List[int] | None,
        stride_bias: List[int] | None,
        output: torch.Tensor,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        has_bias: bool,
    ):
        super().__init__("linear")
        self.input = input
        self.weight = weight
        self.bias = bias
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.shape_weight = shape_weight
        self.stride_weight = stride_weight
        self.shape_bias = shape_bias
        self.stride_bias = stride_bias
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.has_bias = has_bias

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add shapes
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.shape"), self.shape_weight)
        if self.shape_bias is not None and self.has_bias:
            test_writer.add_array(test_writer.gguf_key("bias.shape"), self.shape_bias)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)

        # Add strides
        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        
        strides_weight = self.stride_weight if self.stride_weight is not None else contiguous_gguf_strides(self.shape_weight)
        test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*strides_weight))
        
        if self.has_bias:
            strides_bias = self.stride_bias if self.stride_bias is not None else contiguous_gguf_strides(self.shape_bias)
            test_writer.add_array(test_writer.gguf_key("bias.strides"), gguf_strides(*strides_bias))
        
        strides_output = self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output)
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*strides_output))

        # Add has_bias parameter
        test_writer.add_bool(test_writer.gguf_key("has_bias"), self.has_bias)

        # Helper function to add tensor
        def add_tensor_helper(tensor, key):
            if tensor.dtype == torch.bfloat16:
                arr = tensor.detach().cpu().float().numpy().astype(bfloat16)
                raw_dtype = np_dtype_to_ggml(bfloat16)
            else:
                arr = tensor.detach().cpu().numpy()
                raw_dtype = np_dtype_to_ggml(arr.dtype)
            test_writer.add_tensor(
                test_writer.gguf_key(key),
                arr,
                raw_dtype=raw_dtype,
            )

        # Add input tensor
        add_tensor_helper(self.input, "input")

        # Add weight tensor
        add_tensor_helper(self.weight, "weight")

        # Add bias tensor (if exists)
        if self.has_bias and self.bias is not None:
            add_tensor_helper(self.bias, "bias")

        # Add output tensor
        add_tensor_helper(self.output, "output")
        
        # Add answer tensor (using double precision for higher accuracy)
        ans_numpy = self.output.detach().cpu().double().numpy()
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_numpy,
            raw_dtype=np_dtype_to_ggml(ans_numpy.dtype),
        )

def test(input_shape, out_features, has_bias=True, dtype=torch.float32, non_contiguous=False, non_contiguous_type="transpose"):
    """Generate a single test case for linear operation"""
    in_features = input_shape[-1]
    
    # Generate input tensor based on dtype
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # Integer types: use randint
        if dtype == torch.int8:
            input_tensor = torch.randint(-50, 50, input_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            input_tensor = torch.randint(-1000, 1000, input_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int32:
            input_tensor = torch.randint(-10000, 10000, input_shape, dtype=dtype)
        else:  # int64
            input_tensor = torch.randint(-10000, 10000, input_shape, dtype=dtype)
    elif dtype == torch.bool:
        input_tensor = torch.randint(0, 2, input_shape, dtype=dtype)
    else:
        # Float types: use randn
        input_tensor = torch.randn(input_shape, dtype=dtype) * 0.1
    
    # Generate weight tensor
    weight_shape = (out_features, in_features)
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        if dtype == torch.int8:
            weight_tensor = torch.randint(-50, 50, weight_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            weight_tensor = torch.randint(-1000, 1000, weight_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int32:
            weight_tensor = torch.randint(-10000, 10000, weight_shape, dtype=dtype)
        else:  # int64
            weight_tensor = torch.randint(-10000, 10000, weight_shape, dtype=dtype)
    elif dtype == torch.bool:
        weight_tensor = torch.randint(0, 2, weight_shape, dtype=dtype)
    else:
        weight_tensor = torch.randn(weight_shape, dtype=dtype) * 0.1
    
    # Generate bias tensor (if needed)
    bias_tensor = None
    if has_bias:
        bias_shape = (out_features,)
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            if dtype == torch.int8:
                bias_tensor = torch.randint(-50, 50, bias_shape, dtype=torch.int32).to(dtype)
            elif dtype == torch.int16:
                bias_tensor = torch.randint(-1000, 1000, bias_shape, dtype=torch.int32).to(dtype)
            elif dtype == torch.int32:
                bias_tensor = torch.randint(-10000, 10000, bias_shape, dtype=dtype)
            else:  # int64
                bias_tensor = torch.randint(-10000, 10000, bias_shape, dtype=dtype)
        elif dtype == torch.bool:
            bias_tensor = torch.randint(0, 2, bias_shape, dtype=dtype)
        else:
            bias_tensor = torch.randn(bias_shape, dtype=dtype) * 0.1
    
    # Create non-contiguous tensors if requested
    if non_contiguous:
        if non_contiguous_type == "transpose":
            # Create non-contiguous input tensor by transposing dimensions
            if len(input_shape) == 2:
                # For 2D: transpose and then transpose back to get non-contiguous memory layout
                input_tensor = input_tensor.transpose(0, 1).transpose(0, 1)
            elif len(input_shape) == 3:
                # For 3D: transpose first two dimensions and back
                input_tensor = input_tensor.transpose(0, 1).transpose(0, 1)
            elif len(input_shape) >= 4:
                # For 4D+: transpose first two dimensions and back
                input_tensor = input_tensor.transpose(0, 1).transpose(0, 1)
        elif non_contiguous_type == "slice":
            # Create non-contiguous input tensor by slicing (stride increase)
            if len(input_shape) == 2:
                # For 2D: slice every 2nd element in the first dimension to avoid changing feature dimension
                input_tensor = input_tensor[::2, :]
            elif len(input_shape) == 3:
                # For 3D: slice every 2nd element in the first dimension
                input_tensor = input_tensor[::2, :, :]
            elif len(input_shape) >= 4:
                # For 4D+: slice every 2nd element in the first dimension
                input_tensor = input_tensor[::2, ...]
    
    # Compute reference output
    output_tensor = reference_linear(input_tensor, weight_tensor, bias_tensor)
    
    return LinearTestCase(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        shape_input=list(input_tensor.shape),
        stride_input=list(input_tensor.stride()) if non_contiguous else None,
        shape_weight=list(weight_tensor.shape),
        stride_weight=None,
        shape_bias=list(bias_tensor.shape) if bias_tensor is not None else None,
        stride_bias=None,
        output=output_tensor,
        shape_output=list(output_tensor.shape),
        stride_output=None,
        has_bias=has_bias,
    )

if __name__ == "__main__":
    import gguf
    test_writer = InfiniopTestWriter("linear.gguf")
    test_cases = []
    
    # Test cases with different input shapes and features
    _TEST_CASES_ = [
        # Basic 2D cases
        {"input_shape": (4, 8), "out_features": 16, "has_bias": True},
        {"input_shape": (4, 8), "out_features": 16, "has_bias": False},
        {"input_shape": (1, 512), "out_features": 256, "has_bias": True},
        {"input_shape": (1, 512), "out_features": 256, "has_bias": False},
        {"input_shape": (32, 128), "out_features": 64, "has_bias": True},
        {"input_shape": (32, 128), "out_features": 64, "has_bias": False},
        
        # 3D cases (batch processing)
        {"input_shape": (2, 4, 8), "out_features": 16, "has_bias": True},
        {"input_shape": (2, 4, 8), "out_features": 16, "has_bias": False},
        {"input_shape": (8, 16, 32), "out_features": 64, "has_bias": True},
        {"input_shape": (8, 16, 32), "out_features": 64, "has_bias": False},
        
        # Edge cases
        {"input_shape": (1, 1), "out_features": 1, "has_bias": True},
        {"input_shape": (1, 1), "out_features": 1, "has_bias": False},
        {"input_shape": (1, 2048), "out_features": 4096, "has_bias": True},
        {"input_shape": (1, 2048), "out_features": 4096, "has_bias": False},
        
        # Non-contiguous stride test cases (transpose)
        {"input_shape": (4, 8), "out_features": 16, "has_bias": True, "non_contiguous": True, "non_contiguous_type": "transpose"},
        {"input_shape": (4, 8), "out_features": 16, "has_bias": False, "non_contiguous": True, "non_contiguous_type": "transpose"},
        {"input_shape": (2, 4, 8), "out_features": 16, "has_bias": True, "non_contiguous": True, "non_contiguous_type": "transpose"},
        {"input_shape": (2, 4, 8), "out_features": 16, "has_bias": False, "non_contiguous": True, "non_contiguous_type": "transpose"},
        {"input_shape": (3, 5, 7), "out_features": 12, "has_bias": True, "non_contiguous": True, "non_contiguous_type": "transpose"},
        {"input_shape": (3, 5, 7), "out_features": 12, "has_bias": False, "non_contiguous": True, "non_contiguous_type": "transpose"},
        
        # Non-contiguous stride test cases (slice)
        {"input_shape": (4, 8), "out_features": 16, "has_bias": True, "non_contiguous": True, "non_contiguous_type": "slice"},
        {"input_shape": (4, 8), "out_features": 16, "has_bias": False, "non_contiguous": True, "non_contiguous_type": "slice"},
        {"input_shape": (2, 4, 8), "out_features": 16, "has_bias": True, "non_contiguous": True, "non_contiguous_type": "slice"},
        {"input_shape": (2, 4, 8), "out_features": 16, "has_bias": False, "non_contiguous": True, "non_contiguous_type": "slice"},
        {"input_shape": (3, 6, 8), "out_features": 12, "has_bias": True, "non_contiguous": True, "non_contiguous_type": "slice"},
        {"input_shape": (3, 6, 8), "out_features": 12, "has_bias": False, "non_contiguous": True, "non_contiguous_type": "slice"},
    ]

    _TENSOR_DTYPES_ = [
        torch.float32, 
        torch.float16,
        torch.bfloat16,
    ]

    for dtype in _TENSOR_DTYPES_:
        for test_config in _TEST_CASES_:
            test_case = test(
                input_shape=test_config["input_shape"],
                out_features=test_config["out_features"],
                has_bias=test_config["has_bias"],
                dtype=dtype,
                non_contiguous=test_config.get("non_contiguous", False),
                non_contiguous_type=test_config.get("non_contiguous_type", "transpose")
            )
            test_cases.append(test_case)

    print(f"Generated {len(test_cases)} test cases for Linear operator")
    test_writer.add_tests(test_cases)
    test_writer.save()
    print("Linear test cases saved to linear.gguf")