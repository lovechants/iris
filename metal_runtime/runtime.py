import numpy as np 
from typing import Optional, Tuple 
import Metal
from .dtype import DType, get_alignment 

class MetalBuffer: 
    def __init__(self, buffer: Metal.MTLBuffer, shape: Tuple[int, ...], dtype: DType):
        self.buffer = buffer
        self.shape = shape
        self.dtype = dtype
        self.size = int(np.prod(shape)) * dtype.size

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

class MetalRuntime:
    _instance: Optional["MetalRuntime"] = None
