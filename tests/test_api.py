from numpy.testing import assert_
import pytest
import numpy as np 
from metal_runtime import api
from metal_runtime.dtype import DType

ADD_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& n [[buffer(3)]],  
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}
"""

class TestAPI:
    def test_asarray(self):
        arr = np.random.randn(100).astype(np.float32)
        buf = api.asarray(arr)
        assert buf.shape == (100, )
        assert buf.dtype == DType.FLOAT32

    def test_empty(self):
        buf = api.empty((256, ), DType.FLOAT32)
        assert buf.shape == (256, )
        assert buf.dtype == DType.FLOAT32

    def test_empty_like(self):
        original = api.empty((128, ),DType.FLOAT32)
        new_buf = api.empty_like(original)
        assert new_buf.shape == original.shape
        assert new_buf.dtype == original.dtype

    def test_zeros(self):
        buf = api.zeros((64,), DType.FLOAT32)
        result = api.to_numpy(buf)
        np.testing.assert_array_equal(result, np.zeros(64, dtype=np.float32))

    def test_ones(self):
        buf = api.ones((32,), DType.FLOAT32)
        result = api.to_numpy(buf)
        np.testing.assert_array_equal(result, np.ones(32, dtype=np.float32))

    def test_to_numpy(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf = api.asarray(original)
        result = api.to_numpy(buf)
        np.testing.assert_array_equal(result, original)

    def test_synchronize(self):
        api.synchronize()

    def test_launch_1d(self):
        n = 512
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a + b

        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = api.empty((n,), DType.FLOAT32)

        api.launch(ADD_KERNEL, "add", grid=(n,), block=(128,), args=[a_buf, b_buf, c_buf, n])

        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_launch_2d(self):
        n = 256
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a + b

        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = api.empty((n,), DType.FLOAT32)

        api.launch(
            ADD_KERNEL, "add", grid=(n, 1), block=(64, 1), args=[a_buf, b_buf, c_buf, n]
        )

        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_register_and_launch_kernel(self):
        n = 128
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a + b

        api.register_kernel("test_add", ADD_KERNEL, "add")

        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = api.empty((n,), DType.FLOAT32)

        api.launch_kernel("test_add", grid=(n,), block=(64,), args=[a_buf, b_buf, c_buf, n])

        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    pytest.mark.xfail("Not registered kernel")
    def test_launch_kernel_not_registered(self):
        with pytest.raises(ValueError, match="not registered"):
            api.launch_kernel("nonexistent", grid=(1,), block=(1,), args=[])
