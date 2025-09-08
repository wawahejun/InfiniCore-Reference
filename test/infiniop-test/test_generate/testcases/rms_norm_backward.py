import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides



def reference_rms_norm_backward(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5):
    """
    Reference implementation of RMS Norm backward operation using PyTorch RMSNorm
    
    Args:
        grad_output: Gradient of the output, shape [..., D]
        input: Input tensor from forward pass, shape [..., D]
        weight: Weight tensor, shape [D]
        epsilon: Small constant for numerical stability
        
    Returns:
        grad_input: Gradient w.r.t. input, shape [..., D]
        grad_weight: Gradient w.r.t. weight, shape [D]
    """
    # Ensure tensors require gradients
    input = input.clone().detach().requires_grad_(True)
    weight = weight.clone().detach().requires_grad_(True)
    
    # Create RMSNorm layer and set weight
    rms_norm = torch.nn.RMSNorm(input.shape[-1], eps=epsilon, device=input.device, dtype=input.dtype)
    rms_norm.weight.data = weight.clone()
    rms_norm.weight.requires_grad_(True)
    
    # Forward pass
    output = rms_norm(input)
    
    # Backward pass
    output.backward(grad_output)
    
    grad_input = input.grad.clone()
    grad_weight = rms_norm.weight.grad.clone()
    
    return grad_input, grad_weight

def numpy_rms_norm_backward(grad_output: np.ndarray, input: np.ndarray, weight: np.ndarray, epsilon: float = 1e-5):
    """
    NumPy implementation of RMS Norm backward for reference
    """
    # Convert numpy dtype to torch dtype
    if grad_output.dtype == np.float32:
        torch_dtype = torch.float32
    elif grad_output.dtype == np.float16:
        torch_dtype = torch.float16
    elif grad_output.dtype == bfloat16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # Convert to torch tensors with proper dtype
    grad_output_torch = torch.from_numpy(grad_output.astype(np.float32)).to(torch_dtype)
    input_torch = torch.from_numpy(input.astype(np.float32)).to(torch_dtype)
    weight_torch = torch.from_numpy(weight.astype(np.float32)).to(torch_dtype)
    
    # Compute gradients using PyTorch
    grad_input_torch, grad_weight_torch = reference_rms_norm_backward(
        grad_output_torch, input_torch, weight_torch, epsilon
    )
    
    # Convert back to numpy with original dtype
    if grad_output.dtype == bfloat16:
        grad_input = grad_input_torch.detach().to(torch.float32).numpy().astype(bfloat16)
        grad_weight = grad_weight_torch.detach().to(torch.float32).numpy().astype(bfloat16)
    else:
        grad_input = grad_input_torch.detach().numpy().astype(grad_output.dtype)
        grad_weight = grad_weight_torch.detach().numpy().astype(grad_output.dtype)
    
    return grad_input, grad_weight

class RMSNormBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_output: np.ndarray,
        input: np.ndarray,
        weight: np.ndarray,
        grad_input: np.ndarray,
        grad_weight: np.ndarray,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        shape_weight: List[int] | None,
        stride_weight: List[int] | None,
        shape_grad_input: List[int] | None,
        stride_grad_input: List[int] | None,
        shape_grad_weight: List[int] | None,
        stride_grad_weight: List[int] | None,
        epsilon: float = 1e-5,
    ):
        super().__init__("rms_norm_backward")
        self.grad_output = grad_output
        self.input = input
        self.weight = weight
        self.grad_input = grad_input
        self.grad_weight = grad_weight
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.shape_weight = shape_weight
        self.stride_weight = stride_weight
        self.shape_grad_input = shape_grad_input
        self.stride_grad_input = stride_grad_input
        self.shape_grad_weight = shape_grad_weight
        self.stride_grad_weight = stride_grad_weight
        self.epsilon = epsilon

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add epsilon parameter
        test_writer.add_float32(test_writer.gguf_key("epsilon"), self.epsilon)
        
        # Add shapes
        if self.shape_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape_grad_output)
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.shape"), self.shape_weight)
        if self.shape_grad_input is not None:
            test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.shape_grad_input)
        if self.shape_grad_weight is not None:
            test_writer.add_array(test_writer.gguf_key("grad_weight.shape"), self.shape_grad_weight)

        # Add strides
        strides_grad_output = self.stride_grad_output if self.stride_grad_output is not None else contiguous_gguf_strides(self.shape_grad_output)
        test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*strides_grad_output))
        
        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        
        strides_weight = self.stride_weight if self.stride_weight is not None else contiguous_gguf_strides(self.shape_weight)
        test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*strides_weight))
        
        strides_grad_input = self.stride_grad_input if self.stride_grad_input is not None else contiguous_gguf_strides(self.shape_grad_input)
        test_writer.add_array(test_writer.gguf_key("grad_input.strides"), gguf_strides(*strides_grad_input))
        
        strides_grad_weight = self.stride_grad_weight if self.stride_grad_weight is not None else contiguous_gguf_strides(self.shape_grad_weight)
        test_writer.add_array(test_writer.gguf_key("grad_weight.strides"), gguf_strides(*strides_grad_weight))

        # Add tensors
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"), self.grad_output, raw_dtype=np_dtype_to_ggml(self.grad_output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=np_dtype_to_ggml(self.weight.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"), self.grad_input, raw_dtype=np_dtype_to_ggml(self.grad_input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_weight"), self.grad_weight, raw_dtype=np_dtype_to_ggml(self.grad_weight.dtype)
        )
        
        # Calculate reference answers using numpy implementation
        ans_grad_input, ans_grad_weight = numpy_rms_norm_backward(
            self.grad_output, self.input, self.weight, self.epsilon
        )
        
        # Add answer tensors for verification
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_input"), ans_grad_input, raw_dtype=np_dtype_to_ggml(ans_grad_input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_weight"), ans_grad_weight, raw_dtype=np_dtype_to_ggml(ans_grad_weight.dtype)
        )

def create_test_case(
    input_shape: tuple,
    dtype: np.dtype = np.float32,
    epsilon: float = 1e-5,
    non_contiguous: bool = False,
    non_contiguous_type: str = "transpose"
) -> RMSNormBackwardTestCase:
    """
    Create a single test case for RMS Norm backward
    """
    # Generate random numpy arrays
    # Note: Don't set seed here to ensure different test cases have different data
    input_array = (np.random.randn(*input_shape) * 0.1).astype(dtype)
    grad_output_array = (np.random.randn(*input_shape) * 0.1).astype(dtype)
    
    # Make arrays non-contiguous if requested
    if non_contiguous:
        if non_contiguous_type == "transpose" and len(input_shape) >= 2:
            # Transpose the last two dimensions
            axes = list(range(len(input_shape)-2)) + [-1, -2]
            input_array = np.transpose(input_array, axes)
            grad_output_array = np.transpose(grad_output_array, axes)
        elif non_contiguous_type == "slice":
            # Create a larger array and slice it, but keep last dimension contiguous
            larger_shape = tuple(dim * 2 for dim in input_shape)
            larger_input = (np.random.randn(*larger_shape) * 0.1).astype(dtype)
            larger_grad_output = (np.random.randn(*larger_shape) * 0.1).astype(dtype)
            
            # Only slice non-last dimensions to keep last dimension contiguous
            slices = tuple(slice(0, dim, 2) if i < len(input_shape) - 1 else slice(None) 
                          for i, dim in enumerate(input_shape))
            input_array = larger_input[slices]
            grad_output_array = larger_grad_output[slices]
    
    # Weight dimension should match the last dimension of the actual input array
    weight_array = (np.random.randn(input_array.shape[-1]) * 0.5 + 1.0).astype(dtype)
    
    # Initialize grad_input and grad_weight as zero arrays (these will be filled by the operation)
    grad_input_init = np.zeros_like(input_array)
    grad_weight_init = np.zeros_like(weight_array)
    
    # Determine shapes and strides
    shape_grad_output = list(grad_output_array.shape)
    shape_input = list(input_array.shape)
    shape_weight = list(weight_array.shape)
    shape_grad_input = list(grad_input_init.shape)
    shape_grad_weight = list(grad_weight_init.shape)
    
    # For numpy arrays, calculate strides in bytes and convert to element strides
    stride_grad_output = [s // grad_output_array.itemsize for s in grad_output_array.strides] if non_contiguous else None
    stride_input = [s // input_array.itemsize for s in input_array.strides] if non_contiguous else None
    stride_weight = None  # Weight is always contiguous
    stride_grad_input = [s // grad_input_init.itemsize for s in grad_input_init.strides] if non_contiguous else None
    stride_grad_weight = None  # Weight gradient is always contiguous
    
    return RMSNormBackwardTestCase(
        grad_output=grad_output_array,
        input=input_array,
        weight=weight_array,
        grad_input=grad_input_init,
        grad_weight=grad_weight_init,
        shape_grad_output=shape_grad_output,
        stride_grad_output=stride_grad_output,
        shape_input=shape_input,
        stride_input=stride_input,
        shape_weight=shape_weight,
        stride_weight=stride_weight,
        shape_grad_input=shape_grad_input,
        stride_grad_input=stride_grad_input,
        shape_grad_weight=shape_grad_weight,
        stride_grad_weight=stride_grad_weight,
        epsilon=epsilon,
    )

def gen_gguf(filename: str):
    """Generate GGUF test file for RMS Norm backward"""
    # Set seed once at the beginning for reproducibility
    np.random.seed(42)
    
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    
    # Test configurations
    _TEST_CASES_ = [
        # Basic 2D cases
        {"input_shape": (4, 8), "epsilon": 1e-5},
        {"input_shape": (1, 512), "epsilon": 1e-5},
        {"input_shape": (32, 128), "epsilon": 1e-5},
        {"input_shape": (16, 256), "epsilon": 1e-6},
        
        # 3D cases (batch processing)
        {"input_shape": (2, 4, 8), "epsilon": 1e-5},
        {"input_shape": (8, 16, 32), "epsilon": 1e-5},
        {"input_shape": (4, 8, 64), "epsilon": 1e-6},

        # Edge cases
        {"input_shape": (1, 1), "epsilon": 1e-5},
        {"input_shape": (1, 2048), "epsilon": 1e-5},
        {"input_shape": (512, 1), "epsilon": 1e-5},
        
        # Different epsilon values
        {"input_shape": (4, 8), "epsilon": 1e-4},
        {"input_shape": (4, 8), "epsilon": 1e-7},
        
        # Non-contiguous cases (only slice type to keep last dimension contiguous)
        {"input_shape": (4, 8), "epsilon": 1e-5, "non_contiguous": True, "non_contiguous_type": "slice"},
        {"input_shape": (2, 4, 8), "epsilon": 1e-5, "non_contiguous": True, "non_contiguous_type": "slice"},
    ]
    
    # Data types to test
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float16,
        bfloat16,
    ]
    
    # Generate test cases
    for dtype in _TENSOR_DTYPES_:
        for test_config in _TEST_CASES_:
            test_case = create_test_case(
                input_shape=test_config["input_shape"],
                dtype=dtype,
                epsilon=test_config["epsilon"],
                non_contiguous=test_config.get("non_contiguous", False),
                non_contiguous_type=test_config.get("non_contiguous_type", "transpose")
            )
            test_cases.append(test_case)
    
    print(f"Generated {len(test_cases)} test cases for RMS Norm Backward operator")
    test_writer.add_tests(test_cases)
    test_writer.save()
    print(f"RMS Norm Backward test cases saved to {filename}")

if __name__ == "__main__":
    gen_gguf("rms_norm_backward.gguf")