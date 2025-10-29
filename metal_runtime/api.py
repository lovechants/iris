from contextlib import contextmanager
import numpy as np
from typing import Optional, Tuple, Any, List, Callable
from metal_runtime.runtime import MetalRuntime, get_runtime, MetalBuffer
from metal_runtime.launcher import KernelLauncher
from metal_runtime.dtype import DType
from pathlib import Path
import json, time

LOG_PATH = Path.home() / ".iris_cache" / "iris_log.jsonl"

def _reset_log():
    LOG_PATH.parent.mkdir(exist_ok=True)
    LOG_PATH.write_text("") 

_runtime = None
_launcher = None
_persistent_default = False

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

def enable_persistence_default(value: bool = True):
    """
    Enables or disables persistent GPU buffer residency by default.
    When True, all new allocations use persistent=True unless explicitly overridden.
    """
    global _persistent_default
    _persistent_default = bool(value)


def asarray(array: np.ndarray, *, persistent: Optional[bool] = None) -> MetalBuffer:
    if persistent is None:
        persistent = _persistent_default

    buf = _get_runtime().upload(array)
    if persistent:
        setattr(buf, "_persistent", True)
    return buf


def empty(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32, *, persistent: Optional[bool] = None) -> MetalBuffer:
    if persistent is None:
        persistent = _persistent_default
    return _get_runtime().allocate(shape, dtype, persistent=persistent)


def empty_like(buffer: MetalBuffer, *, persistent: Optional[bool] = None) -> MetalBuffer:
    if persistent is None:
        persistent = _persistent_default
    return _get_runtime().allocate(buffer.shape, buffer.dtype, persistent=persistent)


def zeros(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> MetalBuffer:
    np_array = np.zeros(shape, dtype=dtype.to_numpy())
    return _get_runtime().upload(np_array)


def ones(shape: Tuple[int, ...], dtype: DType = DType.FLOAT32) -> MetalBuffer:
    np_array = np.ones(shape, dtype=dtype.to_numpy())
    return _get_runtime().upload(np_array)


def to_numpy(buffer: MetalBuffer) -> np.ndarray:
    return _get_runtime().download(buffer)


def synchronize():
    _get_runtime().synchronize()


def launch(
    source: str,
    function_name: str,
    grid: Tuple[int, ...],
    block: Tuple[int, ...],
    args: List[Any],
):
    grid_3d = grid + (1,) * (3 - len(grid))
    block_3d = block + (1,) * (3 - len(block))
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


def launch_kernel(
    name: str, grid: Tuple[int, ...], block: Tuple[int, ...], args: List[Any]
):
    source, function_name = _registry.get(name)
    launch(source, function_name, grid, block, args)


import json, time
from pathlib import Path

def log_event(name: str, duration_ms: float, phase: str = "run"):
    log_dir = Path.home() / ".iris_cache"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "iris_log.jsonl"
    entry = {
        "timestamp": time.time(),
        "kernel": name,
        "time_ms": duration_ms,
        "phase": phase,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


@contextmanager 
def persistent_buffers():
    """Context that avoids freeing pooled buffers until exit"""
    rt = _get_runtime()
    before = dict(rt._buffer_pool)
    try:
        yield
    finally:
        rt._buffer_pool.clear()
        rt._buffer_pool.update(before)

_reset_log()
