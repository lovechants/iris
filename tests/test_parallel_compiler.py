import pytest
import time
from metal_runtime.parallel_compiler import ParallelCompiler
from metal_runtime.ir import IRGraph, IRNode
from metal_runtime.dtype import DType

def test_graph_hash():
    compiler = ParallelCompiler()
    graph = IRGraph()
    node1 = IRNode("input", [], {})
    node1.dtype = DType.FLOAT32
    node2 = IRNode("add", [node1], {"scalar": 1.0})
    node2.dtype = DType.FLOAT32
    graph.add_node(node1)
    graph.add_node(node2)
    graph.outputs = [node2]
    hash1 = compiler._graph_hash(graph)
    graph2 = IRGraph()
    node3 = IRNode("input", [], {})
    node3.dtype = DType.FLOAT32
    node4 = IRNode("add", [node3], {"scalar": 1.0})
    node4.dtype = DType.FLOAT32
    graph2.add_node(node3)
    graph2.add_node(node4)
    graph2.outputs = [node4]
    hash2 = compiler._graph_hash(graph2)
    assert hash1 == hash2
    graph3 = IRGraph()
    node5 = IRNode("input", [], {})
    node5.dtype = DType.FLOAT32
    node6 = IRNode("mul", [node5], {"scalar": 2.0})
    node6.dtype = DType.FLOAT32
    graph3.add_node(node5)
    graph3.add_node(node6)
    graph3.outputs = [node6]
    hash3 = compiler._graph_hash(graph3)
    assert hash1 != hash3

def test_parallel_compilation():
    compiler = ParallelCompiler(num_workers=2)
    graph = IRGraph()
    node1 = IRNode("input", [], {})
    node1.dtype = DType.FLOAT32
    node2 = IRNode("fused", [node1], {"ops": ["add"], "attrs": [{"scalar": 1.0}]})
    node2.dtype = DType.FLOAT32
    graph.add_node(node1)
    graph.add_node(node2)
    graph.outputs = [node2]
    graph_hash = compiler.compile_async(graph)
    start = time.time()
    timeout = start + 0.5
    while not compiler.is_compiling(graph_hash) and compiler.get_compiled(graph_hash) is None and time.time() < timeout:
        time.sleep(0.001)
    assert compiler.is_compiling(graph_hash) or compiler.get_compiled(graph_hash) is not None
    timeout = time.time() + 5.0
    while compiler.is_compiling(graph_hash) and time.time() < timeout:
        time.sleep(0.01)
    compiled = compiler.get_compiled(graph_hash)
    assert compiled is not None
    assert compiled.name.startswith("fused")
    assert compiled.hash == graph_hash

def test_multiple_compilation():
    compiler = ParallelCompiler(num_workers=2)
    graphs = []
    hashes = []
    for i in range(3):
        graph = IRGraph()
        node1 = IRNode("input", [], {})
        node1.dtype = DType.FLOAT32
        node2 = IRNode("fused", [node1], {"ops": ["add"], "attrs": [{"scalar": float(i)}]})
        node2.dtype = DType.FLOAT32
        graph.add_node(node1)
        graph.add_node(node2)
        graph.outputs = [node2]
        graphs.append(graph)
        hashes.append(compiler.compile_async(graph))
    start = time.time()
    timeout = start + 0.5
    while all(not compiler.is_compiling(h) for h in hashes) and all(compiler.get_compiled(h) is None for h in hashes) and time.time() < timeout:
        time.sleep(0.001)
    for h in hashes:
        assert compiler.is_compiling(h) or compiler.get_compiled(h) is not None
    timeout = time.time() + 5.0
    while any(compiler.is_compiling(h) for h in hashes) and time.time() < timeout:
        time.sleep(0.01)
    for h in hashes:
        compiled = compiler.get_compiled(h)
        assert compiled is not None
