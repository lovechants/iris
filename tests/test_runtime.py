import pytest
import numpy as np
from metal_runtime.runtime import MetalRuntime, get_runtime
from metal_runtime.dtype import DType, get_alignment

# Note: Any test with download just expected to fail until can figure out a way to access the object data properly


class TestMetalRuntime:
    @pytest.fixture
    def runtime(self):
        return get_runtime()

    def test_singleton(self, runtime):
        runtime2 = get_runtime()
        assert runtime is runtime2

    def test_device_initialization(self, runtime):
        assert runtime.device is not None
        assert runtime.queue is not None

    def test_buffer_allocation_float32(self, runtime):
        shape = (1024,)
        buf = runtime.allocate(shape, DType.FLOAT32)
        assert buf.shape == shape
        assert buf.dtype == DType.FLOAT32
        assert buf.size == 1024 * 4
        assert buf.numel == 1024

    def test_buffer_allocation_multidim(self, runtime):
        shape = (32, 32)
        buf = runtime.allocate(shape, DType.FLOAT32)
        assert buf.shape == shape
        assert buf.ndim == 2
        assert buf.numel == 1024

    #@pytest.mark.xfail(reason="Download functionality is not yet implemented.")
    def test_upload_download_float32(self, runtime):
        original = np.random.randn(256).astype(np.float32)
        metal_buf = runtime.upload(original)
        retrieved = runtime.download(metal_buf)
        np.testing.assert_array_equal(original, retrieved)

    #@pytest.mark.xfail(reason="Download functionality is not yet implemented.")
    def test_upload_download_int32(self, runtime):
        original = np.arange(128, dtype=np.int32)
        metal_buf = runtime.upload(original)
        retrieved = runtime.download(metal_buf)
        np.testing.assert_array_equal(original, retrieved)

    #@pytest.mark.xfail(reason="Download functionality is not yet implemented.")
    def test_upload_download_multidim(self, runtime):
        original = np.random.randn(16, 16).astype(np.float32)
        metal_buf = runtime.upload(original)
        retrieved = runtime.download(metal_buf)
        np.testing.assert_array_equal(original, retrieved)
        assert retrieved.shape == original.shape

    #@pytest.mark.xfail(reason="Download functionality is not yet implemented.")
    def test_non_contiguous_array(self, runtime):
        base = np.random.randn(10, 10).astype(np.float32)
        non_contiguous = base[::2, ::2]
        assert not non_contiguous.flags.c_contiguous
        metal_buf = runtime.upload(non_contiguous)
        retrieved = runtime.download(metal_buf)
        np.testing.assert_array_equal(non_contiguous, retrieved)

    def test_synchronize(self, runtime):
        runtime.synchronize()

    def test_multiple_buffers(self, runtime):
        buf1 = runtime.allocate((100,), DType.FLOAT32)
        buf2 = runtime.allocate((200,), DType.INT32)
        buf3 = runtime.allocate((50,), DType.FLOAT16)
        assert buf1.size == 100 * 4
        assert buf2.size == 200 * 4
        assert buf3.size == 50 * 2
