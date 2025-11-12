from metal_runtime import api
from metal_runtime.runtime import MetalBuffer
from metal_runtime.ops import _register_operation
from typing import Optional
from pathlib import Path

OPS_DIR = Path(__file__).parent / "ops"

def _load_kernel_source(filename: str) -> str:
    path = OPS_DIR / filename
    with open(path, "r") as f:
        return f.read()

api.register_kernel("gather", _load_kernel_source("gather.metal"), "gather")

def gather(input_buf: MetalBuffer, indices: MetalBuffer, axis: int = 0) -> MetalBuffer:
    if axis >= len(input_buf.shape):
        raise ValueError(f"Axis {axis} out of bounds for shape {input_buf.shape}")
    
    if indices.dtype not in [api.DType.INT32, api.DType.UINT32]:
        raise ValueError("Indices must be integer type")
    
    output_shape = list(input_buf.shape)
    output_shape[axis] = indices.shape[0]
    
    output = api.empty(output_shape, dtype=input_buf.dtype)
    
    api.launch_kernel(
        "gather", 
        (output.numel,), 
        (256,), 
        [input_buf, indices, output, axis, input_buf.shape[axis], output.shape[axis]]
    )
    
    _register_operation("gather", [input_buf, indices], output, {"axis": axis})
    return output
