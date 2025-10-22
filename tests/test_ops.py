import pytest
import numpy as np
from metal_runtime import api, ops
from metal_runtime.dtype import DType


class TestBinaryOps:
    @pytest.mark.parametrize("dtype,np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
        (DType.INT32, np.int32),
        (DType.UINT32, np.uint32),
    ])
    def test_add(self, dtype, np_dtype):
        n = 1024
        a = np.random.randn(n).astype(np_dtype)
        b = np.random.randn(n).astype(np_dtype)
        expected = a + b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.add(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    @pytest.mark.parametrize("dtype,np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.INT32, np.int32),
    ])
    def test_sub(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        b = np.random.randn(n).astype(np_dtype)
        expected = a - b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.sub(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)
    
    def test_mul(self):
        n = 768
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a * b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.mul(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_div(self):
        n = 256
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32) + 1.0
        expected = a / b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.div(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_mismatched_dtypes(self):
        a_buf = api.empty((100,), DType.FLOAT32)
        b_buf = api.empty((100,), DType.INT32)
        
        with pytest.raises(ValueError, match="Dtype mismatch"):
            ops.add(a_buf, b_buf)
    
    def test_mismatched_sizes(self):
        a_buf = api.empty((100,), DType.FLOAT32)
        b_buf = api.empty((200,), DType.FLOAT32)
        
        with pytest.raises(ValueError, match="same number of elements"):
            ops.add(a_buf, b_buf)
    
    def test_with_preallocated_output(self):
        n = 256
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a + b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        out_buf = api.empty((n,), DType.FLOAT32)
        
        result_buf = ops.add(a_buf, b_buf, out_buf)
        assert result_buf is out_buf
        
        result = api.to_numpy(result_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestUnaryOps:
    @pytest.mark.parametrize("dtype,np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.INT32, np.int32),
        (DType.UINT32, np.uint32),
    ])
    def test_relu(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        expected = np.maximum(a, 0)
        
        a_buf = api.asarray(a)
        b_buf = ops.relu(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_sigmoid(self):
        n = 256
        a = np.random.randn(n).astype(np.float32)
        expected = 1 / (1 + np.exp(-a))
        
        a_buf = api.asarray(a)
        b_buf = ops.sigmoid(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=2e-5)
    
    def test_sigmoid_wrong_dtype(self):
        a_buf = api.empty((100,), DType.INT32)
        with pytest.raises(ValueError, match="only supports float"):
            ops.sigmoid(a_buf)
    
    def test_tanh(self):
        n = 384
        a = np.random.randn(n).astype(np.float32)
        expected = np.tanh(a)
        
        a_buf = api.asarray(a)
        b_buf = ops.tanh(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=2e-5)
    
    def test_exp(self):
        n = 512
        a = np.random.randn(n).astype(np.float32) * 0.5
        expected = np.exp(a)
        
        a_buf = api.asarray(a)
        b_buf = ops.exp(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_log(self):
        n = 256
        a = np.abs(np.random.randn(n).astype(np.float32)) + 0.1
        expected = np.log(a)
        
        a_buf = api.asarray(a)
        b_buf = ops.log(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_sqrt(self):
        n = 512
        a = np.abs(np.random.randn(n).astype(np.float32))
        expected = np.sqrt(a)
        
        a_buf = api.asarray(a)
        b_buf = ops.sqrt(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_neg(self):
        n = 384
        a = np.random.randn(n).astype(np.float32)
        expected = -a
        
        a_buf = api.asarray(a)
        b_buf = ops.neg(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_abs(self):
        n = 768
        a = np.random.randn(n).astype(np.float32)
        expected = np.abs(a)
        
        a_buf = api.asarray(a)
        b_buf = ops.abs(a_buf)
        
        result = api.to_numpy(b_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_with_preallocated_output(self):
        n = 256
        a = np.random.randn(n).astype(np.float32)
        expected = -a
        
        a_buf = api.asarray(a)
        out_buf = api.empty((n,), DType.FLOAT32)
        
        result_buf = ops.neg(a_buf, out_buf)
        assert result_buf is out_buf
        
        result = api.to_numpy(result_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEdgeCases:
    def test_empty_buffer(self):
        a_buf = api.empty((0,), DType.FLOAT32)
        b_buf = api.empty((0,), DType.FLOAT32)
        c_buf = ops.add(a_buf, b_buf)
        assert c_buf.numel == 0
    
    def test_single_element(self):
        a = np.array([5.0], dtype=np.float32)
        b = np.array([3.0], dtype=np.float32)
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.add(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, np.array([8.0]))
    
    def test_large_buffer(self):
        n = 1048576
        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        expected = a + b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.add(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
