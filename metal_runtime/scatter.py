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

api.register_kernel("copy", _load_kernel_source("copy.metal"), "copy")
api.register_kernel("scatter", _load_kernel_source("scatter.metal"), "scatter")

def scatter(input_buf: MetalBuffer, indices: MetalBuffer, values: MetalBuffer, axis: int = 0) -> MetalBuffer:
    if axis >= len(input_buf.shape):
        raise ValueError(f"Axis {axis} out of bounds for shape {input_buf.shape}")
    
    if indices.dtype not in [api.DType.INT32, api.DType.UINT32]:
        raise ValueError("Indices must be integer type")
    
    output = api.empty_like(input_buf)
    api.launch_kernel("copy", (output.numel,), (256,), [input_buf, output, output.numel])
    
    api.launch_kernel(
        "scatter", 
        (values.numel,), 
        (256,), 
        [output, indices, values, axis, input_buf.shape[axis], values.numel]  
    )
    
    _register_operation("scatter", [input_buf, indices, values], output, {"axis": axis})
    return output
