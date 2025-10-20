import numpy as np 
from typing import Tuple, Any, List, Callable
from metal_runtime.runtime import MetalRuntime, get_runtime, MetalBuffer
from metal_runtime.launcher import KernelLauncher
from metal_runtime.dtype import DType

_runtime = None 
_launcher = None 

def _get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = get_runtime()
    return _runtime

def _get_launcher():
    global _launcher
    if _launcher is None:
        _launcher = KernelLauncher(_get_runtime())
    return _launcher

def asarray(array: np.ndarray) -> MetalBuffer:
    return _get_runtime().upload(array)

def empty(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> MetalBuffer:
    return _get_runtime().allocate(shape, dtype)

def empty_like(buffer: MetalBuffer) -> MetalBuffer:
    return _get_runtime().allocate(buffer.shape, buffer.dtype)

def zeros(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> MetalBuffer:
    np_array = np.zeros(shape, dtype=dtype.to_numpy())
    return _get_runtime().upload(np_array)

def ones(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> MetalBuffer:
    np_array = np.ones(shape, dtype=dtype.to_numpy())
    return _get_runtime().upload(np_array)

def to_numpy(buffer: MetalBuffer) -> MetalBuffer:
    return _get_runtime().download(buffer)

def synchronize():
    _get_runtime().synchronize()

def launch(source: str, function_name: str, grid: Tuple[int, ...], block: Tuple[int, ...], args: List[Any]):
    grid_3d = grid + (1, ) * (3 - len(grid))
    block_3d = block + (1, ) * (3 - len(block))
    _get_launcher().launch(source, function_name, grid_3d, block_3d, args)

class KernelRegistry:
    def __init__(self):
        self.kernels = {}

    def register(self, name: str, source: str, function_name: str):
        self.kernels[name] = (source, function_name)

    def get(self, name: str) -> Tuple[str, str]:
        if name not in self.kernels:
            raise ValueError(f"Kernel '{name}' not registered")
        return self.kernels[name]

_registry = KernelRegistry()


def register_kernel(name: str, source: str, function_name: str):
    _registry.register(name, source, function_name)

def launch_kernel(name: str, grid: Tuple[int, ...], block: Tuple[int, ...], args: List[Any]):
    source, function_name = _registry.get(name)
    launch(source, function_name, grid, block, args)
