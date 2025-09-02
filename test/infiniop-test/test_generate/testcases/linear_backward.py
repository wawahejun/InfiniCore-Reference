import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_linear_backward(grad_y: torch.Tensor, x: torch.Tensor, w: torch.Tensor, has_bias: bool = True):
    """Reference implementation of linear backward operation"""
    # grad_x = grad_y @ w
    grad_x = torch.matmul(grad_y, w)
    
    # grad_w = grad_y.T @ x (for 2D) or batch-wise for higher dimensions
    if grad_y.dim() == 2:
        grad_w = torch.matmul(grad_y.transpose(-2, -1), x)
    else:
        # For batch dimensions, we need to sum over batch dimensions
        batch_dims = grad_y.shape[:-1]
        grad_y_reshaped = grad_y.reshape(-1, grad_y.shape[-1])
        x_reshaped = x.reshape(-1, x.shape[-1])
        grad_w = torch.matmul(grad_y_reshaped.transpose(-2, -1), x_reshaped)
    
    # grad_b = sum(grad_y, dim=batch_dims) if has_bias
    grad_b = None
    if has_bias:
        if grad_y.dim() == 2:
            grad_b = torch.sum(grad_y, dim=0)
        else:
            # Sum over all dimensions except the last one
            dims_to_sum = list(range(grad_y.dim() - 1))
            grad_b = torch.sum(grad_y, dim=dims_to_sum)
    
    return grad_x, grad_w, grad_b

class LinearBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_y: torch.Tensor,
        x: torch.Tensor,
        w: torch.Tensor,
        grad_x: torch.Tensor,
        grad_w: torch.Tensor,
        grad_b: torch.Tensor | None,
        shape_grad_y: List[int] | None,
        stride_grad_y: List[int] | None,
        shape_x: List[int] | None,
        stride_x: List[int] | None,
        shape_w: List[int] | None,
        stride_w: List[int] | None,
        shape_grad_x: List[int] | None,
        stride_grad_x: List[int] | None,
        shape_grad_w: List[int] | None,
        stride_grad_w: List[int] | None,
        shape_grad_b: List[int] | None,
        stride_grad_b: List[int] | None,
        has_bias: bool,
    ):
        super().__init__("linear_backward")
        self.grad_y = grad_y
        self.x = x
        self.w = w
        self.grad_x = grad_x
        self.grad_w = grad_w
        self.grad_b = grad_b
        self.shape_grad_y = shape_grad_y
        self.stride_grad_y = stride_grad_y
        self.shape_x = shape_x
        self.stride_x = stride_x
        self.shape_w = shape_w
        self.stride_w = stride_w
        self.shape_grad_x = shape_grad_x
        self.stride_grad_x = stride_grad_x
        self.shape_grad_w = shape_grad_w
        self.stride_grad_w = stride_grad_w
        self.shape_grad_b = shape_grad_b
        self.stride_grad_b = stride_grad_b
        self.has_bias = has_bias

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add shapes
        if self.shape_grad_y is not None:
            test_writer.add_array(test_writer.gguf_key("grad_y.shape"), self.shape_grad_y)
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        if self.shape_w is not None:
            test_writer.add_array(test_writer.gguf_key("w.shape"), self.shape_w)
        if self.shape_grad_x is not None:
            test_writer.add_array(test_writer.gguf_key("grad_x.shape"), self.shape_grad_x)
        if self.shape_grad_w is not None:
            test_writer.add_array(test_writer.gguf_key("grad_w.shape"), self.shape_grad_w)
        if self.shape_grad_b is not None and self.has_bias:
            test_writer.add_array(test_writer.gguf_key("grad_b.shape"), self.shape_grad_b)

        # Add strides
        strides_grad_y = self.stride_grad_y if self.stride_grad_y is not None else contiguous_gguf_strides(self.shape_grad_y)
        test_writer.add_array(test_writer.gguf_key("grad_y.strides"), gguf_strides(*strides_grad_y))
        
        strides_x = self.stride_x if self.stride_x is not None else contiguous_gguf_strides(self.shape_x)
        test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*strides_x))
        
        strides_w = self.stride_w if self.stride_w is not None else contiguous_gguf_strides(self.shape_w)
        test_writer.add_array(test_writer.gguf_key("w.strides"), gguf_strides(*strides_w))
        
        strides_grad_x = self.stride_grad_x if self.stride_grad_x is not None else contiguous_gguf_strides(self.shape_grad_x)
        test_writer.add_array(test_writer.gguf_key("grad_x.strides"), gguf_strides(*strides_grad_x))
        
        strides_grad_w = self.stride_grad_w if self.stride_grad_w is not None else contiguous_gguf_strides(self.shape_grad_w)
        test_writer.add_array(test_writer.gguf_key("grad_w.strides"), gguf_strides(*strides_grad_w))
        
        if self.has_bias:
            strides_grad_b = self.stride_grad_b if self.stride_grad_b is not None else contiguous_gguf_strides(self.shape_grad_b)
            test_writer.add_array(test_writer.gguf_key("grad_b.strides"), gguf_strides(*strides_grad_b))

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

        # Add input tensors
        add_tensor_helper(self.grad_y, "grad_y")
        add_tensor_helper(self.x, "x")
        add_tensor_helper(self.w, "w")
        
        # Add output tensors
        add_tensor_helper(self.grad_x, "grad_x")
        add_tensor_helper(self.grad_w, "grad_w")
        if self.has_bias and self.grad_b is not None:
            add_tensor_helper(self.grad_b, "grad_b")
        
        # Compute reference answers using double precision
        grad_y_double = self.grad_y.detach().cpu().double()
        x_double = self.x.detach().cpu().double()
        w_double = self.w.detach().cpu().double()
        
        ans_grad_x, ans_grad_w, ans_grad_b = reference_linear_backward(
            grad_y_double, x_double, w_double, self.has_bias
        )
        
        # Add answer tensors
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_x"),
            ans_grad_x.numpy(),
            raw_dtype=np_dtype_to_ggml(ans_grad_x.numpy().dtype),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_w"),
            ans_grad_w.numpy(),
            raw_dtype=np_dtype_to_ggml(ans_grad_w.numpy().dtype),
        )
        
        if self.has_bias and ans_grad_b is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("ans_grad_b"),
                ans_grad_b.numpy(),
                raw_dtype=np_dtype_to_ggml(ans_grad_b.numpy().dtype),
            )

def test(input_shape, out_features, has_bias=True, dtype=torch.float32, non_contiguous=False, non_contiguous_type="transpose"):
    """Generate a single test case for linear backward operation"""
    in_features = input_shape[-1]
    
    # Generate grad_y tensor (gradient of output)
    grad_y_shape = input_shape[:-1] + (out_features,)
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        # Integer types: use randint
        if dtype == torch.int8:
            grad_y = torch.randint(-50, 50, grad_y_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            grad_y = torch.randint(-1000, 1000, grad_y_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int32:
            grad_y = torch.randint(-10000, 10000, grad_y_shape, dtype=dtype)
        else:  # int64
            grad_y = torch.randint(-10000, 10000, grad_y_shape, dtype=dtype)
    elif dtype == torch.bool:
        grad_y = torch.randint(0, 2, grad_y_shape, dtype=dtype)
    else:
        # Float types: use randn
        grad_y = torch.randn(grad_y_shape, dtype=dtype) * 0.1
    
    # Generate x tensor (input)
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        if dtype == torch.int8:
            x = torch.randint(-50, 50, input_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            x = torch.randint(-1000, 1000, input_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int32:
            x = torch.randint(-10000, 10000, input_shape, dtype=dtype)
        else:  # int64
            x = torch.randint(-10000, 10000, input_shape, dtype=dtype)
    elif dtype == torch.bool:
        x = torch.randint(0, 2, input_shape, dtype=dtype)
    else:
        x = torch.randn(input_shape, dtype=dtype) * 0.1
    
    # Generate w tensor (weight)
    w_shape = (out_features, in_features)
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        if dtype == torch.int8:
            w = torch.randint(-50, 50, w_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int16:
            w = torch.randint(-1000, 1000, w_shape, dtype=torch.int32).to(dtype)
        elif dtype == torch.int32:
            w = torch.randint(-10000, 10000, w_shape, dtype=dtype)
        else:  # int64
            w = torch.randint(-10000, 10000, w_shape, dtype=dtype)
    elif dtype == torch.bool:
        w = torch.randint(0, 2, w_shape, dtype=dtype)
    else:
        w = torch.randn(w_shape, dtype=dtype) * 0.1
    
    # Create non-contiguous tensors if requested
    if non_contiguous:
        if non_contiguous_type == "transpose":
            # Create non-contiguous tensors by transposing dimensions
            if len(input_shape) == 2:
                # For 2D: transpose and then transpose back to get non-contiguous memory layout
                grad_y = grad_y.transpose(0, 1).transpose(0, 1)
                x = x.transpose(0, 1).transpose(0, 1)
            elif len(input_shape) == 3:
                # For 3D: transpose first two dimensions and back
                grad_y = grad_y.transpose(0, 1).transpose(0, 1)
                x = x.transpose(0, 1).transpose(0, 1)
            elif len(input_shape) >= 4:
                # For 4D+: transpose first two dimensions and back
                grad_y = grad_y.transpose(0, 1).transpose(0, 1)
                x = x.transpose(0, 1).transpose(0, 1)
        elif non_contiguous_type == "slice":
            # Create non-contiguous tensors by slicing (stride increase)
            if len(input_shape) == 2:
                # For 2D: slice every 2nd element in the first dimension to avoid changing feature dimension
                grad_y = grad_y[::2, :]
                x = x[::2, :]
            elif len(input_shape) == 3:
                # For 3D: slice every 2nd element in the first dimension
                grad_y = grad_y[::2, :, :]
                x = x[::2, :, :]
            elif len(input_shape) >= 4:
                # For 4D+: slice every 2nd element in the first dimension
                grad_y = grad_y[::2, ...]
                x = x[::2, ...]
    
    # Compute reference gradients
    grad_x, grad_w, grad_b = reference_linear_backward(grad_y, x, w, has_bias)
    
    return LinearBackwardTestCase(
        grad_y=grad_y,
        x=x,
        w=w,
        grad_x=grad_x,
        grad_w=grad_w,
        grad_b=grad_b,
        shape_grad_y=list(grad_y.shape),
        stride_grad_y=list(grad_y.stride()) if non_contiguous else None,
        shape_x=list(x.shape),
        stride_x=list(x.stride()) if non_contiguous else None,
        shape_w=list(w.shape),
        stride_w=None,
        shape_grad_x=list(grad_x.shape),
        stride_grad_x=None,
        shape_grad_w=list(grad_w.shape),
        stride_grad_w=None,
        shape_grad_b=list(grad_b.shape) if grad_b is not None else None,
        stride_grad_b=None,
        has_bias=has_bias,
    )

if __name__ == "__main__":
    import gguf
    test_writer = InfiniopTestWriter("linear_backward.gguf")
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

    print(f"Generated {len(test_cases)} test cases for Linear Backward operator")
    test_writer.add_tests(test_cases)
    test_writer.save()
    print("Linear Backward test cases saved to linear_backward.gguf")