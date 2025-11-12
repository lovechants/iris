import pytest
import numpy as np
from metal_runtime import api
from metal_runtime.reduction import reduce_sum, reduce_max, reduce_min
from metal_runtime.gather import gather
from metal_runtime.scatter import scatter

def test_reduce_sum():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    buf = api.asarray(data)
    
    result = reduce_sum(buf)
    result_np = api.to_numpy(result)
    
    assert np.allclose(result_np, np.sum(data))
    assert result_np.shape == (1,)

def test_reduce_max():
    data = np.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=np.float32)
    buf = api.asarray(data)
    
    result = reduce_max(buf)
    result_np = api.to_numpy(result)
    
    assert np.allclose(result_np, np.max(data))
    assert result_np.shape == (1,)

def test_reduce_min():
    data = np.array([5.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float32)
    buf = api.asarray(data)
    
    result = reduce_min(buf)
    result_np = api.to_numpy(result)
    
    assert np.allclose(result_np, np.min(data))
    assert result_np.shape == (1,)

def test_gather():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.uint32)
    
    buf = api.asarray(data)
    idx_buf = api.asarray(indices)
    
    result = gather(buf, idx_buf)
    result_np = api.to_numpy(result)
    
    expected = np.array([10.0, 30.0, 50.0], dtype=np.float32)
    assert np.allclose(result_np, expected)

def test_scatter():
    data = np.zeros(5, dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.uint32)
    values = np.array([10.0, 30.0, 50.0], dtype=np.float32)
    
    buf = api.asarray(data)
    idx_buf = api.asarray(indices)
    val_buf = api.asarray(values)
    
    result = scatter(buf, idx_buf, val_buf)
    result_np = api.to_numpy(result)
    
    expected = np.array([10.0, 0.0, 30.0, 0.0, 50.0], dtype=np.float32)
    assert np.allclose(result_np, expected)

@pytest.mark.skip(reason="Waiting for async compilation implementation")
def test_fused_with_reductions():
    from metal_runtime.ops import add  
    
    a_np = np.random.randn(1000).astype(np.float32)
    b_np = np.random.randn(1000).astype(np.float32)
    
    a = api.asarray(a_np)
    b = api.asarray(b_np)
    
    with api.fused():
        c = add(a, b)
        d = reduce_sum(c)
    
    result = api.to_numpy(d)
    expected = np.sum(a_np + b_np)
    
    assert np.allclose(result, expected)
