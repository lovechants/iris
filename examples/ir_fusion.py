import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.ir_capture import capture
from metal_runtime.fusion import fuse
from metal_runtime.executor import execute

def benchmark_with_cache():
    n = 1000000
    print(f"Array size: {n:,} elements\n")
    
    a = api.asarray(np.random.randn(n).astype(np.float32))
    b = api.asarray(np.random.randn(n).astype(np.float32))
    
    print("Unfused execution:")
    unfused_times = []
    for _ in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        start = time.perf_counter()
        c = ops.add(a_tmp, b_tmp)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        api.synchronize()
        unfused_times.append(time.perf_counter() - start)
    
    unfused_median = np.median(unfused_times[2:])
    print(f"  Time: {unfused_median*1000:.2f} ms\n")
    
    print("Building fused graph")
    start = time.perf_counter()
    
    with capture() as builder:
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        c = ops.add(a_tmp, b_tmp)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
    
    fused_graph = fuse(builder.graph)
    graph_time = time.perf_counter() - start
    print(f"Graph built in {graph_time*1000:.2f} ms\n")
    
    input_nodes = []
    for node in fused_graph.nodes:
        if node.op == "input":
            input_nodes.append(node)
    
    if not input_nodes:
        for node in fused_graph.nodes:
            if not node.inputs:
                input_nodes.append(node)
    
    print("Fused execution:")
    fused_times = []
    for _ in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        inputs = {}
        if len(input_nodes) >= 2:
            inputs[input_nodes[0]] = a_tmp
            inputs[input_nodes[1]] = b_tmp
        
        start = time.perf_counter()
        execute(fused_graph, inputs)
        api.synchronize()
        fused_times.append(time.perf_counter() - start)
    
    fused_median = np.median(fused_times[2:])
    speedup = unfused_median / fused_median
    
    print(f"  Time: {fused_median*1000:.2f} ms\n")
    
    print(f"Speedup: {speedup:.2f}x")
    print(f"Memory traffic reduced by {(1 - 3/7)*100:.0f}%")

if __name__ == "__main__":
    benchmark_with_cache()
