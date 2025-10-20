import numpy as np 
from enum import Enum
from typing import Optional

class DType(Enum):
    FLOAT32 = ("float", 4, np.float32)
    FLOAT16 = ("half", 2, np.float16)
    INT32 = ("int", 4, np.int32)
    UINT32 = ("uint", 4, np.uint32)
    INT16 = ("short", 2, np.int16)
    UINT16 = ("ushort", 2, np.uint16)
    INT8 = ("char", 1, np.int8)
    UINT8 = ("uchar", 1, np.uint8)

    def __init__(self, metal_name: str, size: int, numpy_dtype: type):
        self.metal_name = metal_name
        self.size = size
        self.numpy_dtype = numpy_dtype

    """
    TODO: Add more types: https://numpy.org/doc/stable/user/basics.types.html
    - bool 
    - int4 (unpack two 4-bit values from a single uint8)
    - vector types 
    - complex float 
    """
    @staticmethod
    def from_numpy(dtype: np.dtype) -> "DType":
        type_map = {
            np.float32: DType.FLOAT32,
            np.float16: DType.FLOAT16,
            np.int32: DType.INT32,
            np.uint32: DType.UINT32,
            np.int16: DType.INT16,
            np.uint16: DType.UINT16,
            np.int8: DType.INT8,
            np.uint8: DType.UINT8,               
        }
        np_type = dtype.type 
        if np_type not in type_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return type_map[np_type]

    def to_numpy(self) -> np.dtype:
        return np.dtype(self.numpy_dtype)

def get_alignment(dtype: DType) -> int:
    return max(dtype.size, 4)
