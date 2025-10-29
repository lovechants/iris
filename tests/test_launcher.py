import pytest
import numpy as np
from metal_runtime.runtime import get_runtime
from metal_runtime.launcher import KernelLauncher
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

MULTIPLY_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void multiply(
    device const float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int& n [[buffer(3)]], 
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = a[tid] * scale;
    }
}
"""

SENTINEL_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void write_sentinel(
    device float* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid == 0) {
        out[0] = 12345.0f;
    }
}
"""


class TestKernelLauncher:
    @pytest.fixture
    def runtime(self):
        return get_runtime()

    @pytest.fixture
    def launcher(self, runtime):
        return KernelLauncher(runtime)

    # @pytest.mark.xfail(reason="CPU download is not implemented.")
    def test_simple_add(self, runtime, launcher):
        n = 1024
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a + b

        a_buf = runtime.upload(a)
        b_buf = runtime.upload(b)
        c_buf = runtime.allocate((n,), DType.FLOAT32)

        launcher.launch(
            ADD_KERNEL,
            "add",
            grid=(n, 1, 1),
            block=(256, 1, 1),
            args=[a_buf, b_buf, c_buf, n],
        )

        result = runtime.download(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    # @pytest.mark.xfail(reason="CPU download is not implemented.")
    def test_multiply_scalar(self, runtime, launcher):
        n = 512
        a = np.random.randn(n).astype(np.float32)
        scale = 2.5
        expected = a * scale

        a_buf = runtime.upload(a)
        b_buf = runtime.allocate((n,), DType.FLOAT32)

        launcher.launch(
            MULTIPLY_KERNEL,
            "multiply",
            grid=(n, 1, 1),
            block=(128, 1, 1),
            args=[a_buf, b_buf, scale, n],
        )

        result = runtime.download(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_pipeline_caching(self, launcher):
        pipeline1 = launcher._get_pipeline(ADD_KERNEL, "add")
        pipeline2 = launcher._get_pipeline(ADD_KERNEL, "add")
        assert pipeline1 is pipeline2

    def test_invalid_function_name(self, launcher):
        with pytest.raises(ValueError, match="not found"):
            launcher._get_pipeline(ADD_KERNEL, "nonexistent")

    def test_block_size_exceeds_maximum(self, runtime, launcher):
        a_buf = runtime.allocate((100,), DType.FLOAT32)
        b_buf = runtime.allocate((100,), DType.FLOAT32)
        c_buf = runtime.allocate((100,), DType.FLOAT32)

        with pytest.raises(ValueError, match="exceeds maximum"):
            launcher.launch(
                ADD_KERNEL,
                "add",
                grid=(100, 1, 1),
                block=(2048, 1, 1),
                args=[a_buf, b_buf, c_buf, 100],
            )

    # @pytest.mark.xfail(reason="CPU download is not implemented.")
    def test_different_grid_sizes(self, runtime, launcher):
        sizes = [64, 256, 1024]
        for n in sizes:
            a = np.random.randn(n).astype(np.float32)
            b = np.random.randn(n).astype(np.float32)
            expected = a + b

            a_buf = runtime.upload(a)
            b_buf = runtime.upload(b)
            c_buf = runtime.allocate((n,), DType.FLOAT32)

            launcher.launch(
                ADD_KERNEL,
                "add",
                grid=(n, 1, 1),
                block=(64, 1, 1),
                args=[a_buf, b_buf, c_buf, n],
            )

            result = runtime.download(c_buf)
            np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_kernel_execution_with_sentinel(self, runtime, launcher):
        """
        Verifies that a kernel actually runs and modifies memory on the GPU
        without needing to download the entire buffer.
        """
        initial_value = 0.0
        output_buf = runtime.upload(np.full((10,), initial_value, dtype=np.float32))

        launcher.launch(
            SENTINEL_KERNEL,
            "write_sentinel",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            args=[output_buf],
        )

        result = runtime.peek(output_buf, DType.FLOAT32, index=0)
        assert result == 12345.0, (
            f"Kernel did not execute correctly. Expected 12345.0, got {result}"
        )
