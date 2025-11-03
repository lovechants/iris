import numpy as np
from metal_runtime import api, ops
from metal_runtime.ir_capture import capture

def test_basic_capture():
    a = api.asarray(np.array([1, 2, 3], dtype=np.float32))
    b = api.asarray(np.array([4, 5, 6], dtype=np.float32))
    
    with capture() as builder:
        c = ops.add(a, b)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        builder.set_output(e)
    
    assert len(builder.graph.nodes) == 3
    
    nodes = builder.graph.topo_sort()
    assert nodes[0].op == "add"
    assert nodes[1].op == "mul_scalar"
    assert nodes[2].op == "relu"

def test_graph_structure():
    a = api.asarray(np.random.randn(100).astype(np.float32))
    b = api.asarray(np.random.randn(100).astype(np.float32))
    
    with capture() as builder:
        c = ops.add(a, b)
        d = ops.mul(c, b)
        builder.set_output(d)
    
    nodes = builder.graph.topo_sort()
    assert nodes[1].inputs[0] == nodes[0]

def test_multiple_outputs():
    a = api.asarray(np.random.randn(100).astype(np.float32))
    
    with capture() as builder:
        b = ops.relu(a)
        c = ops.neg(a)
        builder.set_output(b)
        builder.set_output(c)
    
    assert len(builder.graph.outputs) == 2

def test_no_capture_normal_execution():
    a = api.asarray(np.array([1, 2, 3], dtype=np.float32))
    b = api.asarray(np.array([4, 5, 6], dtype=np.float32))
    
    c = ops.add(a, b)
    result = api.to_numpy(c)
    
    expected = np.array([5, 7, 9], dtype=np.float32)
    assert np.allclose(result, expected)
