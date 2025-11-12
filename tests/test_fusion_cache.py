import pytest
import numpy as np
import time
from metal_runtime import api, ops, gather, scatter, reduction
from metal_runtime.fusion_cache import get_fused_operation

@pytest.fixture
def sample_data():
    a = api.asarray(np.random.randn(1000).astype(np.float32))
    b = api.asarray(np.random.randn(1000).astype(np.float32))
    return a, b

def test_add_mul_relu_pattern(sample_data):
    a, b = sample_data
    fused_op = get_fused_operation("add_mul_relu")
    result = fused_op(a, b)
    expected = ops.relu(ops.mul_scalar(ops.add(a, b), 2.0))
    np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)

def test_add_sigmoid_pattern(sample_data):
    a, b = sample_data
    fused_op = get_fused_operation("add_sigmoid")
    result = fused_op(a, b)
    expected = ops.sigmoid(ops.add(a, b))
    np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)

def test_mul_tanh_pattern(sample_data):
    a, b = sample_data
    fused_op = get_fused_operation("mul_tanh")
    result = fused_op(a, b)
    expected = ops.tanh(ops.mul(a, b))
    np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)

def test_add_scalar_mul_scalar_pattern(sample_data):
    a, _ = sample_data
    fused_op = get_fused_operation("add_scalar_mul_scalar")
    result = fused_op(a, 2.0, 3.0)
    expected = ops.mul_scalar(ops.add_scalar(a, 2.0), 3.0)
    np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)

def test_reduce_sum_reduce_max_pattern(sample_data):
    a, _ = sample_data
    fused_op = get_fused_operation("reduce_sum_reduce_max")
    result = fused_op(a)
    expected = reduction.reduce_max(reduction.reduce_sum(a))
    np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)

def test_gather_add_pattern(sample_data):
    a, _ = sample_data
    idx = api.asarray(np.random.randint(0, 1000, (100,)).astype(np.int32))
    fused_op = get_fused_operation("gather_add")
    result = fused_op(a, idx)
    expected = ops.add(gather.gather(a, idx), 1.0)
    np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)

def test_fused_context_caching(sample_data):
    a, b = sample_data
    
    with api.fused():
        result1 = ops.add(ops.mul_scalar(ops.add(a, b), 2.0), 3.0)
    
    with api.fused():
        result2 = ops.add(ops.mul_scalar(ops.add(a, b), 2.0), 3.0)
    
    np.testing.assert_allclose(api.to_numpy(result1), api.to_numpy(result2), rtol=1e-5)

def test_pattern_performance(sample_data):
    a, b = sample_data
    
    start = time.perf_counter()
    for _ in range(100):
        result_unfused = ops.relu(ops.mul_scalar(ops.add(a, b), 2.0))
    api.synchronize()
    unfused_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(100):
        result_fused = get_fused_operation("add_mul_relu")(a, b)
    api.synchronize()
    fused_time = time.perf_counter() - start
    
    assert fused_time < unfused_time / 2
    np.testing.assert_allclose(api.to_numpy(result_unfused), api.to_numpy(result_fused), rtol=1e-5)

def test_cache_hit_miss():
    a = api.asarray(np.random.randn(100).astype(np.float32))
    b = api.asarray(np.random.randn(100).astype(np.float32))
    
    op1 = get_fused_operation("add_mul_relu")
    op2 = get_fused_operation("add_mul_relu")
    
    assert op1 is op2
    
    result1 = op1(a, b)
    result2 = op2(a, b)
    np.testing.assert_allclose(api.to_numpy(result1), api.to_numpy(result2), rtol=1e-5)

def test_invalid_pattern():
    with pytest.raises(ValueError):
        get_fused_operation("invalid_pattern")

def test_different_input_sizes():
    for size in [10, 100, 1000, 10000]:
        a = api.asarray(np.random.randn(size).astype(np.float32))
        b = api.asarray(np.random.randn(size).astype(np.float32))
        
        fused_op = get_fused_operation("add_mul_relu")
        result = fused_op(a, b)
        expected = ops.relu(ops.mul_scalar(ops.add(a, b), 2.0))
        np.testing.assert_allclose(api.to_numpy(result), api.to_numpy(expected), rtol=1e-5)
