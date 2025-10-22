import pytest
import numpy as np
from metal_runtime.decorator import kernel, metal, KernelFunction
from metal_runtime import api


class TestKernelDecorator:
    def test_basic_decorator(self):
        @kernel(
            param_types={
                "a": "device const float*",
                "b": "device const float*",
                "c": "device float*",
                "n": "uint",
            }
        )
        def add(a, b, c, n):
            i = metal.thread_id()
            if i < n:
                c[i] = a[i] + b[i]

        assert isinstance(add, KernelFunction)
        assert add.function_name == "add"

    def test_decorator_without_params(self):
        @kernel
        def simple():
            pass

        assert isinstance(simple, KernelFunction)
        assert simple.function_name == "simple"

    def test_custom_name(self):
        @kernel(
            name="custom_kernel",
            param_types={"x": "device float*"}
        )
        def my_func(x):
            pass

        assert my_func.function_name == "custom_kernel"

    def test_msl_source_generation(self):
        @kernel(
            param_types={
                "a": "device const float*",
                "out": "device float*",
            }
        )
        def double(a, out):
            tid = metal.thread_id()
            out[tid] = a[tid] * 2

        msl = double.msl_source
        assert "kernel void double(" in msl
        assert "device const float* a" in msl
        assert "device float* out" in msl

    def test_kernel_cannot_be_called_directly(self):
        @kernel(param_types={"x": "device float*"})
        def func(x):
            pass

        with pytest.raises(RuntimeError, match="cannot be called directly"):
            func()

    def test_end_to_end_execution(self):
        @kernel(
            param_types={
                "a": "device const float*",
                "b": "device const float*",
                "c": "device float*",
                "n": "uint",
            }
        )
        def vector_add(a, b, c, n):
            i = metal.thread_id()
            if i < n:
                c[i] = a[i] + b[i]

        n = 512
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)
        expected = a_np + b_np

        a_buf = api.asarray(a_np)
        b_buf = api.asarray(b_np)
        c_buf = api.empty((n,), api.DType.FLOAT32)

        api.launch_kernel("vector_add", grid=(n,), block=(128,), args=[a_buf, b_buf, c_buf, n])

        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_multiple_kernels(self):
        @kernel(
            param_types={
                "x": "device const float*",
                "y": "device float*",
            }
        )
        def negate(x, y):
            tid = metal.thread_id()
            y[tid] = -x[tid]

        @kernel(
            param_types={
                "x": "device const float*",
                "y": "device float*",
            }
        )
        def square(x, y):
            tid = metal.thread_id()
            y[tid] = x[tid] * x[tid]

        assert negate.function_name == "negate"
        assert square.function_name == "square"
        assert "negate" in negate.msl_source
        assert "square" in square.msl_source
