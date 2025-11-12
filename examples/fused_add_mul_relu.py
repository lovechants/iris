import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.ir_capture import capture
from metal_runtime.fusion import fuse
from metal_runtime.executor import execute

class FusedAddMulRelu:
    """Pre-built fused operation that can be reused."""
    
    def __init__(self):
        self.fused_graph = None
        self.input_nodes = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the fused graph once."""
        with capture() as builder:
            a = api.asarray(np.random.randn(1000000).astype(np.float32))
            b = api.asarray(np.random.randn(1000000).astype(np.float32))
            c = ops.add(a, b)
            d = ops.mul_scalar(c, 2.0)
            e = ops.relu(d)
        
        self.fused_graph = fuse(builder.graph)
        
        self.input_nodes = []
        for node in self.fused_graph.nodes:
            if node.op == "input":
                self.input_nodes.append(node)
        
        if not self.input_nodes:
            for node in self.fused_graph.nodes:
                if not node.inputs:
                    self.input_nodes.append(node)
    
    def __call__(self, a, b):
        inputs = {}
        if len(self.input_nodes) >= 2:
            inputs[self.input_nodes[0]] = a
            inputs[self.input_nodes[1]] = b
        else:
            if self.fused_graph.nodes:
                node = self.fused_graph.nodes[0]
                inputs[node] = a
                if len(node.inputs) < 2:
                    node.inputs = [a, b]
        
        return execute(self.fused_graph, inputs)

def benchmark():
    n = 1000000
    print(f"Array size: {n:,} elements\n")
    
    a = api.asarray(np.random.randn(n).astype(np.float32))
    b = api.asarray(np.random.randn(n).astype(np.float32))
    
    print("Unfused execution")
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
    
    print("Creating fused operation")
    start = time.perf_counter()
    fused_op = FusedAddMulRelu()
    create_time = time.perf_counter() - start
    print(f"  Created in {create_time*1000:.2f} ms\n")
    
    print("Fused execution:")
    fused_times = []
    for _ in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        start = time.perf_counter()
        result = fused_op(a_tmp, b_tmp)
        api.synchronize()
        fused_times.append(time.perf_counter() - start)
    
    fused_median = np.median(fused_times[2:])
    speedup = unfused_median / fused_median
    
    print(f"  Time: {fused_median*1000:.2f} ms\n")
    
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
