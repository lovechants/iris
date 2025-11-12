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

api.register_kernel("reduce_sum", _load_kernel_source("reduce_sum.metal"), "reduce_sum")
api.register_kernel("reduce_max", _load_kernel_source("reduce_max.metal"), "reduce_max")
api.register_kernel("reduce_min", _load_kernel_source("reduce_min.metal"), "reduce_min")
api.register_kernel("reduce_sum_axis", _load_kernel_source("reduce_sum.metal"), "reduce_sum_axis")
api.register_kernel("reduce_max_axis", _load_kernel_source("reduce_max.metal"), "reduce_max_axis")
api.register_kernel("reduce_min_axis", _load_kernel_source("reduce_min.metal"), "reduce_min_axis")

def reduce_sum(buf: MetalBuffer, axis: Optional[int] = None) -> MetalBuffer:
    if axis is None:
        return _reduce_global(buf, "sum")
    else:
        return _reduce_axis(buf, axis, "sum")

def reduce_max(buf: MetalBuffer, axis: Optional[int] = None) -> MetalBuffer:
    if axis is None:
        return _reduce_global(buf, "max")
    else:
        return _reduce_axis(buf, axis, "max")

def reduce_min(buf: MetalBuffer, axis: Optional[int] = None) -> MetalBuffer:
    if axis is None:
        return _reduce_global(buf, "min")
    else:
        return _reduce_axis(buf, axis, "min")

def _reduce_global(buf: MetalBuffer, op: str) -> MetalBuffer:
    output = api.empty((1,), dtype=buf.dtype)
    
    api.launch_kernel(f"reduce_{op}", (256,), (256,), [buf, output, buf.numel])
    _register_operation(f"reduce_{op}", [buf], output)
    return output

def _reduce_axis(buf: MetalBuffer, axis: int, op: str) -> MetalBuffer:
    if axis >= len(buf.shape):
        raise ValueError(f"Axis {axis} out of bounds for shape {buf.shape}")
    
    output_shape = list(buf.shape)
    output_shape[axis] = 1
    
    output = api.empty(output_shape, dtype=buf.dtype)
    
    api.launch_kernel(
        f"reduce_{op}_axis", 
        (buf.shape[0], buf.shape[1], 1), 
        (1, 1, 1), 
        [buf, output, buf.shape[0], buf.shape[1], axis]
    )
    
    _register_operation(f"reduce_{op}_axis", [buf], output, {"axis": axis})
    return output
