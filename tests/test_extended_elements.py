import pytest
import numpy as np
from metal_runtime import api, ops
from metal_runtime.dtype import DType

class TestExtendedBinaryOps:
    """Tests for binary operations, extending to more types."""
    
    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT16, np.float16),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_add_extended_types(self, dtype, np_dtype):
        n = 1024
        a = np.random.randn(n).astype(np_dtype)
        b = np.random.randn(n).astype(np_dtype)
        expected = a + b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.add(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        # Use a higher tolerance for float16
        rtol = 2e-3 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT16, np.float16),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_sub_extended_types(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        b = np.random.randn(n).astype(np_dtype)
        expected = a - b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.sub(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-2
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT16, np.float16),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_mul_extended_types(self, dtype, np_dtype):
        n = 768
        a = np.random.randn(n).astype(np_dtype)
        b = np.random.randn(n).astype(np_dtype)
        expected = a * b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.mul(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-2
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT16, np.float16),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_div_extended_types(self, dtype, np_dtype):
        n = 256
        a = np.random.randn(n).astype(np_dtype)
        
        # Create the constant with the target dtype to avoid float64 promotion
        one = np.array(1.0, dtype=np_dtype)
        b = np.random.randn(n).astype(np_dtype) + one
        
        expected = a / b
        
        a_buf = api.asarray(a)
        b_buf = api.asarray(b)
        c_buf = ops.div(a_buf, b_buf)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)


class TestScalarOps:
    """Tests for all scalar operations."""
    
    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
        (DType.INT32, np.int32),
        (DType.UINT32, np.uint32),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_mul_scalar(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        scalar = 3.14 if np_dtype in (np.float32, np.float16) else 7
        expected = a * scalar
        
        a_buf = api.asarray(a)
        c_buf = ops.mul_scalar(a_buf, scalar)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
        (DType.INT32, np.int32),
        (DType.UINT32, np.uint32),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_add_scalar(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        scalar = 1.5 if np_dtype in (np.float32, np.float16) else 5
        expected = a + scalar
        
        a_buf = api.asarray(a)
        c_buf = ops.add_scalar(a_buf, scalar)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
        (DType.INT32, np.int32),
        (DType.UINT32, np.uint32),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_sub_scalar(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        scalar = 2.5 if np_dtype in (np.float32, np.float16) else 3
        expected = a - scalar
        
        a_buf = api.asarray(a)
        c_buf = ops.sub_scalar(a_buf, scalar)
        
        result = api.to_numpy(c_buf)
        rtol = 1e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
        (DType.UINT32, np.uint32),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_div_scalar(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        scalar = 2.0 if np_dtype in (np.float32, np.float16) else 4
        expected = a / scalar
        
        a_buf = api.asarray(a)
        c_buf = ops.div_scalar(a_buf, scalar)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
        (DType.INT32, np.int32),
        (DType.UINT32, np.uint32),
        (DType.INT16, np.int16),
        (DType.UINT16, np.uint16),
        (DType.INT8, np.int8),
        (DType.UINT8, np.uint8),
    ])
    def test_rsub_scalar(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        scalar = 10.0 if np_dtype in (np.float32, np.float16) else 20
        expected = scalar - a
        
        a_buf = api.asarray(a)
        c_buf = ops.rsub_scalar(scalar, a_buf)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)

    @pytest.mark.parametrize("dtype, np_dtype", [
        (DType.FLOAT32, np.float32),
        (DType.FLOAT16, np.float16),
    ])
    def test_rdiv_scalar(self, dtype, np_dtype):
        n = 512
        a = np.random.randn(n).astype(np_dtype)
        
        # FIX: Create the 0.1 constant with the target dtype to avoid float64 promotion
        offset = np.array(0.1, dtype=np_dtype)
        a = np.abs(a) + offset # Avoid division by zero
        
        scalar = 100.0 if np_dtype in (np.float32, np.float16) else 200
        expected = scalar / a
        
        a_buf = api.asarray(a)
        c_buf = ops.rdiv_scalar(scalar, a_buf)
        
        result = api.to_numpy(c_buf)
        rtol = 5e-2 if np_dtype == np.float16 else 1e-3
        np.testing.assert_allclose(result, expected, rtol=rtol)


class TestMetalBufferOperators:
    """Tests for MetalBuffer operator overloads."""
    
    def test_binary_operators(self):
        n = 256
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)
        
        a_buf = api.asarray(a_np)
        b_buf = api.asarray(b_np)
        
        # Test +
        result = api.to_numpy(a_buf + b_buf)
        expected = a_np + b_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test -
        result = api.to_numpy(a_buf - b_buf)
        expected = a_np - b_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test *
        result = api.to_numpy(a_buf * b_buf)
        expected = a_np * b_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test /
        result = api.to_numpy(a_buf / b_buf)
        expected = a_np / b_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_scalar_operators(self):
        n = 256
        a_np = np.random.randn(n).astype(np.float32)
        scalar = 3.14
        
        a_buf = api.asarray(a_np)
        
        # Test buffer * scalar
        result = api.to_numpy(a_buf * scalar)
        expected = a_np * scalar
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test scalar * buffer
        result = api.to_numpy(scalar * a_buf)
        expected = scalar * a_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_unary_operators(self):
        n = 256
        a_np = np.random.randn(n).astype(np.float32)
        
        a_buf = api.asarray(a_np)
        
        # Test -
        result = api.to_numpy(-a_buf)
        expected = -a_np
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test abs()
        result = api.to_numpy(abs(a_buf))
        expected = np.abs(a_np)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_method_based_operations(self):
        n = 256
        a_np = np.random.randn(n).astype(np.float32)
        
        a_buf = api.asarray(a_np)
        
        # Test exp()
        result = api.to_numpy(a_buf.exp())
        expected = np.exp(a_np)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test relu()
        result = api.to_numpy(a_buf.relu())
        expected = np.maximum(0, a_np)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test sigmoid()
        result = api.to_numpy(a_buf.sigmoid())
        expected = 1 / (1 + np.exp(-a_np))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test tanh()
        result = api.to_numpy(a_buf.tanh())
        expected = np.tanh(a_np)
        np.testing.assert_allclose(result, expected, rtol=2e-5)
        
        # Test log()
        a_np_pos = np.abs(a_np) + 0.1  # Use positive values for log
        a_buf_pos = api.asarray(a_np_pos)
        result = api.to_numpy(a_buf_pos.log())
        expected = np.log(a_np_pos)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        
        # Test sqrt()
        result = api.to_numpy(a_buf_pos.sqrt())
        expected = np.sqrt(a_np_pos)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
