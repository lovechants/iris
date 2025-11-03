from pathlib import Path
from typing import Tuple, Optional, Union
from metal_runtime.api import launch, empty_like
from metal_runtime.runtime import MetalBuffer
from metal_runtime.dtype import DType
from metal_runtime import api
import numpy as np
import ctypes
from metal_runtime.ir_capture import current, is_capturing

OPS_DIR = Path(__file__).parent / "ops"

def _load_kernel_source(filename: str) -> str:
    path = OPS_DIR / filename
    with open(path, "r") as f:
        return f.read()

def _register_operation(op_name: str, inputs: list, result: MetalBuffer, attrs: dict = None):
    if is_capturing():
        builder = current()
        if builder is not None:
            node = builder.make_node(op_name, inputs, attrs)
            builder.register(result, node)

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

def _scalar_op(
    a: MetalBuffer,
    scalar: Union[float, int],
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

    if a.dtype == DType.FLOAT16:
        scalar_val = float(scalar)
    elif a.dtype == DType.FLOAT32:
        scalar_val = float(scalar)
    else:
        scalar_val = int(scalar)

    if grid is None:
        grid = (n,)
    if block is None:
        block = (256,)

    metal_type_name = DTYPE_TO_METAL_NAME[a.dtype]
    templated_kernel = f"{kernel_name}_{metal_type_name}"

    launch(ELEMENTWISE_SOURCE, templated_kernel, grid, block, [a, out, scalar_val, n])
    return out

def _scalar_right_op(
    scalar: Union[float, int],
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

    if a.dtype == DType.FLOAT16:
        scalar_val = float(scalar)
    elif a.dtype == DType.FLOAT32:
        scalar_val = float(scalar)
    else:
        scalar_val = int(scalar)

    if grid is None:
        grid = (n,)
    if block is None:
        block = (256,)

    metal_type_name = DTYPE_TO_METAL_NAME[a.dtype]
    templated_kernel = f"{kernel_name}_{metal_type_name}"

    launch(ELEMENTWISE_SOURCE, templated_kernel, grid, block, [scalar_val, a, out, n])
    return out

def _cast_op(
    a: MetalBuffer,
    out: Optional[MetalBuffer],
    in_dtype: DType,
    out_dtype: DType,
    grid: Optional[Tuple[int, ...]] = None,
    block: Optional[Tuple[int, ...]] = None,
) -> MetalBuffer:
    n = a.numel

    if out is None:
        out = api.empty(a.shape, dtype=out_dtype)
    elif out.numel != n:
        raise ValueError("Output buffer must have same number of elements as input")
    elif out.dtype != out_dtype:
        raise ValueError(f"Output buffer dtype must be {out_dtype}")

    if grid is None:
        grid = (n,)
    if block is None:
        block = (256,)

    in_metal_name = DTYPE_TO_METAL_NAME[in_dtype]
    out_metal_name = DTYPE_TO_METAL_NAME[out_dtype]
    templated_kernel = f"cast_{in_metal_name}_to_{out_metal_name}"

    launch(ELEMENTWISE_SOURCE, templated_kernel, grid, block, [a, out, n])
    return out

def _get_result_dtype(dtype1: DType, dtype2: DType) -> DType:
    if DType.FLOAT32 in (dtype1, dtype2):
        return DType.FLOAT32
    if DType.FLOAT16 in (dtype1, dtype2):
        return DType.FLOAT16
    if DType.INT32 in (dtype1, dtype2):
        return DType.INT32
    if DType.UINT32 in (dtype1, dtype2):
        return DType.UINT32
    if DType.INT16 in (dtype1, dtype2):
        return DType.INT16
    if DType.UINT16 in (dtype1, dtype2):
        return DType.UINT16
    return DType.INT8

def cast(
    a: MetalBuffer, dtype: DType, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    """Casts a MetalBuffer to a new dtype using a Metal kernel."""
    if a.dtype == dtype:
        return a

    result = _cast_op(a, out, a.dtype, dtype)
    _register_operation("cast", [a], result, {"dtype": dtype})
    return result


def add(
    a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    result = _elementwise_binary_op(a, b, out, "add")
    _register_operation("add", [a, b], result)
    return result


def sub(
    a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    result = _elementwise_binary_op(a, b, out, "sub")
    _register_operation("sub", [a, b], result)
    return result


def mul(
    a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    result = _elementwise_binary_op(a, b, out, "mul")
    _register_operation("mul", [a, b], result)
    return result


def div(
    a: MetalBuffer, b: MetalBuffer, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    result_dtype = (
        DType.FLOAT32
        if a.dtype.itemsize >= 4 or b.dtype.itemsize >= 4
        else DType.FLOAT16
    )

    if a.dtype != result_dtype:
        a = cast(a, result_dtype)
    if b.dtype != result_dtype:
        b = cast(b, result_dtype)

    if out is None:
        out = api.empty(a.shape, dtype=result_dtype)
    elif out.dtype != result_dtype:
        raise ValueError(
            f"Output buffer dtype {out.dtype} does not match expected result dtype {result_dtype}"
        )

    result = _elementwise_binary_op(a, b, out, "div")
    _register_operation("div", [a, b], result)
    return result


def relu(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    result = _unary_op(a, out, "relu")
    _register_operation("relu", [a], result)
    return result


def sigmoid(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"sigmoid only supports float types, got {a.dtype}")
    result = _unary_op(a, out, "sigmoid")
    _register_operation("sigmoid", [a], result)
    return result


def tanh(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"tanh only supports float types, got {a.dtype}")
    result = _unary_op(a, out, "tanh_op")
    _register_operation("tanh", [a], result)
    return result


def exp(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"exp only supports float types, got {a.dtype}")
    result = _unary_op(a, out, "exp_op")
    _register_operation("exp", [a], result)
    return result


def log(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"log only supports float types, got {a.dtype}")
    result = _unary_op(a, out, "log_op")
    _register_operation("log", [a], result)
    return result


def sqrt(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    if a.dtype not in [DType.FLOAT32, DType.FLOAT16]:
        raise ValueError(f"sqrt only supports float types, got {a.dtype}")
    result = _unary_op(a, out, "sqrt_op")
    _register_operation("sqrt", [a], result)
    return result


def neg(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    result = _unary_op(a, out, "neg")
    _register_operation("neg", [a], result)
    return result


def abs(a: MetalBuffer, out: Optional[MetalBuffer] = None) -> MetalBuffer:
    result = _unary_op(a, out, "abs_op")
    _register_operation("abs", [a], result)
    return result


def mul_scalar(
    a: MetalBuffer, scalar: Union[float, int], out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    if a.dtype == DType.FLOAT16 and isinstance(scalar, float):
        scalar = float(scalar)

    result = _scalar_op(a, scalar, out, "mul_scalar")
    _register_operation("mul_scalar", [a], result, {"scalar": scalar})
    return result


def add_scalar(
    a: MetalBuffer, scalar: Union[float, int], out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    if a.dtype == DType.FLOAT16 and isinstance(scalar, float):
        scalar = float(scalar)

    result = _scalar_op(a, scalar, out, "add_scalar")
    _register_operation("add_scalar", [a], result, {"scalar": scalar})
    return result


def sub_scalar(
    a: MetalBuffer, scalar: Union[float, int], out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    if a.dtype == DType.FLOAT16 and isinstance(scalar, float):
        scalar = float(scalar)

    result = _scalar_op(a, scalar, out, "sub_scalar")
    _register_operation("sub_scalar", [a], result, {"scalar": scalar})
    return result


def rsub_scalar(
    scalar: Union[float, int], a: MetalBuffer, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    if a.dtype == DType.FLOAT16 and isinstance(scalar, float):
        scalar = float(scalar)

    result = _scalar_right_op(scalar, a, out, "rsub_scalar")
    _register_operation("rsub_scalar", [a], result, {"scalar": scalar})
    return result


def div_scalar(
    a: MetalBuffer, scalar: Union[float, int], out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    result_dtype = DType.FLOAT32 if a.dtype.itemsize >= 4 else DType.FLOAT16

    if a.dtype != result_dtype:
        a = cast(a, result_dtype)

    if out is None:
        out = api.empty(a.shape, dtype=result_dtype)
    elif out.dtype != result_dtype:
        raise ValueError(
            f"Output buffer dtype {out.dtype} does not match expected result dtype {result_dtype}"
        )

    if result_dtype == DType.FLOAT16:
        scalar_val = float(scalar)
    elif result_dtype == DType.FLOAT32:
        scalar_val = float(scalar)
    else:
        scalar_val = int(scalar)

    grid = (a.numel,)
    block = (256,)

    metal_type_name = DTYPE_TO_METAL_NAME[result_dtype]
    templated_kernel = f"div_scalar_{metal_type_name}"

    launch(
        ELEMENTWISE_SOURCE, templated_kernel, grid, block, [a, out, scalar_val, a.numel]
    )
    _register_operation("div_scalar", [a], out, {"scalar": scalar})
    return out


def rdiv_scalar(
    scalar: Union[float, int], a: MetalBuffer, out: Optional[MetalBuffer] = None
) -> MetalBuffer:
    result_dtype = DType.FLOAT32 if a.dtype.itemsize >= 4 else DType.FLOAT16

    if a.dtype != result_dtype:
        a = cast(a, result_dtype)

    if out is None:
        out = api.empty(a.shape, dtype=result_dtype)
    elif out.dtype != result_dtype:
        raise ValueError(
            f"Output buffer dtype {out.dtype} does not match expected result dtype {result_dtype}"
        )

    if result_dtype == DType.FLOAT16:
        scalar_val = float(scalar)
    elif result_dtype == DType.FLOAT32:
        scalar_val = float(scalar)
    else:
        scalar_val = int(scalar)

    grid = (a.numel,)
    block = (256,)

    metal_type_name = DTYPE_TO_METAL_NAME[result_dtype]
    templated_kernel = f"rdiv_scalar_{metal_type_name}"

    launch(
        ELEMENTWISE_SOURCE, templated_kernel, grid, block, [scalar_val, a, out, a.numel]
    )
    _register_operation("rdiv_scalar", [a], out, {"scalar": scalar})
    return out
