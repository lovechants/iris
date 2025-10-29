import numpy as np
import objc
from ctypes import c_void_p, cast, memmove, POINTER, c_char
from typing import Optional, Tuple, List
import Metal
from Foundation import NSData  # <-- Added for the robust upload method
from .dtype import DType, get_alignment
import threading

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

    # These are just tensors all over again
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            from metal_runtime import ops

            return ops.mul_scalar(self, other)
        elif isinstance(other, MetalBuffer):
            from metal_runtime import ops

            return ops.mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, MetalBuffer):
            from metal_runtime import ops

            return ops.add(self, other)
        elif isinstance(other, (int, float)):
            from metal_runtime import ops

            return ops.add_scalar(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, MetalBuffer):
            from metal_runtime import ops

            return ops.sub(self, other)
        elif isinstance(other, (int, float)):
            from metal_runtime import ops

            return ops.sub_scalar(self, other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            from metal_runtime import ops

            return ops.rsub_scalar(other, self)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            from metal_runtime import ops

            return ops.div_scalar(self, other)
        elif isinstance(other, MetalBuffer):
            from metal_runtime import ops

            return ops.div(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            from metal_runtime import ops

            return ops.rdiv_scalar(other, self)
        return NotImplemented

    def __neg__(self):
        from metal_runtime import ops

        return ops.neg(self)

    def __abs__(self):
        from metal_runtime import ops

        return ops.abs(self)

    def exp(self):
        from metal_runtime import ops

        return ops.exp(self)

    def log(self):
        from metal_runtime import ops

        return ops.log(self)

    def sqrt(self):
        from metal_runtime import ops

        return ops.sqrt(self)

    def relu(self):
        from metal_runtime import ops

        return ops.relu(self)

    def sigmoid(self):
        from metal_runtime import ops

        return ops.sigmoid(self)

    def tanh(self):
        from metal_runtime import ops

        return ops.tanh(self)


class MetalRuntime:
    _instance: Optional["MetalRuntime"] = None

    def __init__(self):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device found.")
        self.queue = self.device.newCommandQueue()
        self._buffer_pool = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MetalRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def allocate(self, shape: Tuple[int, ...], dtype: DType, persistent: bool = False) -> MetalBuffer:
        size = int(np.prod(shape)) * dtype.size
        alignment = get_alignment(dtype)
        aligned_size = ((size + alignment - 1) // alignment) * alignment

        # Metal doesn't allow zero-sized buffers, allocate minimum size
        """
        FAILED tests/test_ops.py::TestEdgeCases::test_empty_buffer - RuntimeError: Failed to allocate buffer of size 0
        1 failed, 25 passed, 3 warnings in 0.53s
        """
        if aligned_size == 0:
            aligned_size = alignment

        buf = self._allocate_pooled(aligned_size)
        metal_buf = MetalBuffer(buf, shape, dtype)
        if not persistent:
            import weakref
            weakref.finalize(metal_buf, self._release_pooled, buf)
        return metal_buf

    def _allocate_pooled(self, size: int) -> MetalBuffer:
        with self._lock:
            aligned = ((size + 255) // 256) * 256 #rounding to nearest 256 bytes
            pool = self._buffer_pool.setdefault(aligned, [])
            if pool:
                return pool.pop()
        buf = self.device.newBufferWithLength_options_(aligned, Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise RuntimeError(f"Failed to allocated pooled buffer of size {aligned}")
        return buf

    def _release_pooled(self, buffer):
        with self._lock:
            self._buffer_pool.setdefault(buffer.length(), []).append(buffer)


    def upload(self, array: np.ndarray) -> MetalBuffer:
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)

        dtype = DType.from_numpy(array.dtype)

        data_bytes = array.tobytes()
        ns_data = NSData.dataWithBytes_length_(data_bytes, len(data_bytes))

        buffer = self.device.newBufferWithBytes_length_options_(
            ns_data, len(data_bytes), Metal.MTLResourceStorageModeShared
        )

        return MetalBuffer(buffer, array.shape, dtype)

    def download(self, metal_buffer: MetalBuffer) -> np.ndarray:
        contents = metal_buffer.buffer.contents()
        total_bytes = metal_buffer.size

        tuple_of_bytes = contents[0:total_bytes]

        all_bytes = b"".join(tuple_of_bytes)

        array_flat = np.frombuffer(all_bytes, dtype=metal_buffer.dtype.to_numpy())

        return array_flat.reshape(metal_buffer.shape)

    """
    This and download have been plaguing me all day, trying ctype, PyObjC stuff and everything 
    turns out can just use numpy numpy is so goated thank you numpy 
    some comments to make sense of it peek -> download since its similar ideas
    """

    def peek(self, metal_buffer: MetalBuffer, dtype: DType, index: int = 0):
        offset_bytes = index * dtype.size
        contents = metal_buffer.buffer.contents()

        # Slicing returns a tuple of single-byte BYTES objects.
        # e.g., (b'\x00', b'\xe4', b'\xc0', b'\x46')
        tuple_of_bytes = contents[offset_bytes : offset_bytes + dtype.size]

        # Join the tuple of bytes into a single bytes object.
        value_bytes = b"".join(tuple_of_bytes)

        return np.frombuffer(value_bytes, dtype=dtype.to_numpy())[0]

    def synchronize(self):
        cmd_buffer = self.queue.commandBuffer()
        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()


def get_runtime() -> MetalRuntime:
    return MetalRuntime.get_instance()
