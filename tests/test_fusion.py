import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.ir_capture import capture
from metal_runtime.fusion import fuse
from metal_runtime.executor import execute

def test_simple_fusion():
    a_np = np.random.randn(10000).astype(np.float32)
    b_np = np.random.randn(10000).astype(np.float32)
    
    a = api.asarray(a_np)
    b = api.asarray(b_np)
    
    with capture() as builder:
        c = ops.add(a, b)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        builder.set_output(e)
    
    original_count = len(builder.graph.nodes)
    
    fused_graph = fuse(builder.graph)
    fused_count = len(fused_graph.nodes)
    
    assert fused_count < original_count
    
    inputs = {
        builder.buf_to_node[id(a)]: a,
        builder.buf_to_node[id(b)]: b,
    }
    
    outputs = execute(fused_graph, inputs)
    result = api.to_numpy(outputs[fused_graph.outputs[0]])
    
    expected = np.maximum((a_np + b_np) * 2.0, 0)
    
    assert np.allclose(result, expected, rtol=1e-5)

def test_fusion_correctness():
    n = 100000
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    
    a_unfused = api.asarray(a_np)
    b_unfused = api.asarray(b_np)
    
    c = ops.add(a_unfused, b_unfused)
    d = ops.mul_scalar(c, 2.0)
    e = ops.relu(d)
    unfused_result = api.to_numpy(e)
    
    a_fused = api.asarray(a_np)
    b_fused = api.asarray(b_np)
    
    with capture() as builder:
        c = ops.add(a_fused, b_fused)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        builder.set_output(e)
    
    fused_graph = fuse(builder.graph)
    
    inputs = {
        builder.buf_to_node[id(a_fused)]: a_fused,
        builder.buf_to_node[id(b_fused)]: b_fused,
    }
    
    outputs = execute(fused_graph, inputs)
    fused_result = api.to_numpy(outputs[fused_graph.outputs[0]])
    
    assert np.allclose(unfused_result, fused_result, rtol=1e-5)

def test_fusion_speedup():
    n = 1_000_000
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    
    unfused_times = []
    for _ in range(5):
        a = api.asarray(a_np)
        b = api.asarray(b_np)
        
        start = time.perf_counter()
        c = ops.add(a, b)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        api.synchronize()
        unfused_times.append(time.perf_counter() - start)
    
    fused_times = []
    for _ in range(5):
        a = api.asarray(a_np)
        b = api.asarray(b_np)
        
        with capture() as builder:
            c = ops.add(a, b)
            d = ops.mul_scalar(c, 2.0)
            e = ops.relu(d)
            builder.set_output(e)
        
        fused_graph = fuse(builder.graph)
        
        inputs = {
            builder.buf_to_node[id(a)]: a,
            builder.buf_to_node[id(b)]: b,
        }
        
        start = time.perf_counter()
        outputs = execute(fused_graph, inputs)
        api.synchronize()
        fused_times.append(time.perf_counter() - start)
    
    unfused_avg = np.median(unfused_times)
    fused_avg = np.median(fused_times)
    speedup = unfused_avg / fused_avg
    
    print(f"\nUnfused: {unfused_avg*1000:.2f}ms")
    print(f"Fused: {fused_avg*1000:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    assert speedup > 1.3

def test_no_fusion_single_op():
    a = api.asarray(np.random.randn(1000).astype(np.float32))
    
    with capture() as builder:
        b = ops.relu(a)
        builder.set_output(b)
    
    original_count = len(builder.graph.nodes)
    fused_graph = fuse(builder.graph)
    
    assert len(fused_graph.nodes) == original_count

def test_fusion_multiple_chains():
    a = api.asarray(np.random.randn(1000).astype(np.float32))
    b = api.asarray(np.random.randn(1000).astype(np.float32))
    
    with capture() as builder:
        c = ops.add(a, b)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        
        f = ops.neg(a)
        g = ops.abs(f)
        
        builder.set_output(e)
        builder.set_output(g)
    
    original_count = len(builder.graph.nodes)
    fused_graph = fuse(builder.graph)
    
    assert len(fused_graph.nodes) < original_count
