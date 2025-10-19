import numpy as np
import objc  
from ctypes import c_void_p, cast, memmove  
from typing import Optional, Tuple, List
import Metal
from Foundation import NSData  # <-- Added for the robust upload method
from .dtype import DType, get_alignment

class MetalBuffer:
    def __init__(self, buffer, shape: Tuple[int, ...], dtype: DType):
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
    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found.")
        self.queue = self.device.newCommandQueue()

    @classmethod
    def get_instance(cls) -> "MetalRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def allocate(self, shape: Tuple[int, ...], dtype: DType) -> MetalBuffer:
        size = int(np.prod(shape)) * dtype.size
        alignment = get_alignment(dtype)
        aligned_size = ((size + alignment - 1) // alignment) * alignment

        buffer = self.device.newBufferWithLength_options_(
            aligned_size, Metal.MTLResourceStorageModeShared
        )
        if buffer is None:
            raise RuntimeError(f"Failed to allocate buffer of size {aligned_size}")
        
        return MetalBuffer(buffer, shape, dtype)

    def upload(self, array: np.ndarray) -> MetalBuffer:
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        
        dtype = DType.from_numpy(array.dtype)
        
        data_bytes = array.tobytes()
        ns_data = NSData.dataWithBytes_length_(data_bytes, len(data_bytes))
        
        buffer = self.device.newBufferWithBytes_length_options_(
            ns_data,
            len(data_bytes),
            Metal.MTLResourceStorageModeShared
        )
        
        return MetalBuffer(buffer, array.shape, dtype)

    def download(self, metal_buffer: MetalBuffer) -> np.ndarray:
        """
        Data from buffer->python
        Too many complexities with output can't read the type etc, making the
        buffer is fine but reading back from it is WIP 
        Don't really think its a momumental need right now just a nice utility so TODO for now
        We don't need to read back to the CPU as long as we get the kernel output anyway
        """
        raise NotImplementedError("Downloading back to CPU not implemented yet")

    def synchronize(self):
        cmd_buffer = self.queue.commandBuffer()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

def get_runtime() -> MetalRuntime:
    return MetalRuntime.get_instance()
