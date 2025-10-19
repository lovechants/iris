import pytest
import numpy as np
from metal_runtime.dtype import DType, get_alignment

class TestDType:
    def test_float32_properties(self):
        dt = DType.FLOAT32
        assert dt.metal_name == "float"
        assert dt.size == 4
        assert dt.numpy_dtype == np.float32

    def test_float16_properties(self):
        dt = DType.FLOAT16
        assert dt.metal_name == "half"
        assert dt.size == 2
        assert dt.numpy_dtype == np.float16

    def test_int32_properties(self):
        dt = DType.INT32
        assert dt.metal_name == "int"
        assert dt.size == 4
        assert dt.numpy_dtype == np.int32

    def test_uint8_properties(self):
        dt = DType.UINT8
        assert dt.metal_name == "uchar"
        assert dt.size == 1
        assert dt.numpy_dtype == np.uint8

    def test_from_numpy_float32(self):
        dt = DType.from_numpy(np.dtype(np.float32))
        assert dt == DType.FLOAT32

    def test_from_numpy_int32(self):
        dt = DType.from_numpy(np.dtype(np.int32))
        assert dt == DType.INT32

    def test_from_numpy_float16(self):
        dt = DType.from_numpy(np.dtype(np.float16))
        assert dt == DType.FLOAT16

    def test_from_numpy_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            DType.from_numpy(np.dtype(np.float64))

    def test_to_numpy_float32(self):
        dt = DType.FLOAT32
        np_dt = dt.to_numpy()
        assert np_dt == np.dtype(np.float32)

    def test_to_numpy_int16(self):
        dt = DType.INT16
        np_dt = dt.to_numpy()
        assert np_dt == np.dtype(np.int16)

    def test_alignment_float32(self):
        assert get_alignment(DType.FLOAT32) == 4

    def test_alignment_float16(self):
        assert get_alignment(DType.FLOAT16) == 4

    def test_alignment_uint8(self):
        assert get_alignment(DType.UINT8) == 4

    def test_alignment_int32(self):
        assert get_alignment(DType.INT32) == 4
