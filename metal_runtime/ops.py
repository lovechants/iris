from pathlib import Path
from typing import Tuple, Optional
from metal_runtime.api import launch, empty_like 
from metal_runtime.runtime import MetalBuffer
from metal_runtime.dtype import DType

OPS_DIR = Path(__file__).parent / "ops"


def _load_kernel_source(filename: str) -> str:
    path = OPS_DIR / filename
    with open(path, "r") as f:
        return f.read()


ELEMENTWISE_SOURCE = _load_kernel_source("elementwise.metal")


DTYPE_TO_METAL_NAME = {
    DType.FLOAT32: "float",
    DType.FLOAT16: "half",
    DType.INT32: "int",
    DType.UINT32: "uint",
    DType.INT16: "short",
    DType.UINT16: "ushort",
    DType.INT8: "char",
    DType.UINT8: "uchar",
}


def _elementwise_binary_op(
    a: MetalBuffer,
    b: MetalBuffer,
    out: Optional[MetalBuffer],
    kernel_name: str,
    grid: Optional[Tuple[int, ...]] = None,
    block: Optional[Tuple[int, ...]] = None,
) -> MetalBuffer:
    if a.dtype != b.dtype:
        raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")
    
    n = a.numel
    if b.numel != n:
        raise ValueError("Input buffers must have same number of elements")
    
    if out is None:
        out = empty_like(a)
    elif out.numel != n:
        raise ValueError("Output buffer must have same number of elements as inputs")
    elif out.dtype != a.dtype:
        raise ValueError(f"Output dtype {out.dtype} must match input dtype {a.dtype}")
    
    if grid is None:
        grid = (n,)
    if block is None:
        block = (256,)
    
    metal_type_name = DTYPE_TO_METAL_NAME[a.dtype]
    templated_kernel = f"{kernel_name}_{metal_type_name}"
    
    launch(ELEMENTWISE_SOURCE, templated_kernel, grid, block, [a, b, out, n])
    return out


def _unary_op(
    a: MetalBuffer,
    out: Optional[MetalBuffer],
    kernel_name: str,
    grid: Optional[Tuple[int, ...]] = None,
    block: Optional[Tuple[int, ...]] = None,
) -> MetalBuffer:
    n = a.numel
    
    if out is None:
        out = empty_like(a)
    elif out.numel != n:
        raise ValueError("Output buffer must have same number of elements as input")
    elif out.dtype != a.dtype:
        raise ValueError(f"Output dtype {out.dtype} must match input dtype {a.dtype}")
    
    if grid is None:
        grid = (n,)
    if block is None:
        block = (256,)
    
    metal_type_name = DTYPE_TO_METAL_NAME[a.dtype]
    templated_kernel = f"{kernel_name}_{metal_type_name}"
    
    launch(ELEMENTWISE_SOURCE, templated_kernel, grid, block, [a, out, n])
    return out


def add(a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _elementwise_binary_op(a, b, out, "add")


def sub(a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _elementwise_binary_op(a, b, out, "sub")


def mul(a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _elementwise_binary_op(a, b, out, "mul")


def div(a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _elementwise_binary_op(a, b, out, "div")


def relu(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _unary_op(a, out, "relu")


def sigmoid(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"sigmoid only supports float types, got {a.dtype}")
    return _unary_op(a, out, "sigmoid")


def tanh(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"tanh only supports float types, got {a.dtype}")
    return _unary_op(a, out, "tanh_op")


def exp(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"exp only supports float types, got {a.dtype}")
    return _unary_op(a, out, "exp_op")


def log(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"log only supports float types, got {a.dtype}")
    return _unary_op(a, out, "log_op")


def sqrt(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"sqrt only supports float types, got {a.dtype}")
    return _unary_op(a, out, "sqrt_op")


def neg(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _unary_op(a, out, "neg")


def abs(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    return _unary_op(a, out, "abs_op")
