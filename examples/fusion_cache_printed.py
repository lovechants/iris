import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.ir_capture import capture
from metal_runtime.fusion import fuse
from metal_runtime.executor import execute

"""
Print statements for the benchmarking for expected behavior 
"""


class FusedKernelCache:
    _cache = {}
    
    @classmethod
    def get(cls, op_pattern):
        if op_pattern not in cls._cache:
            print(f"Cache miss - creating new kernel for {op_pattern}")
            cls._cache[op_pattern] = cls._create(op_pattern)
        else:
            print(f"Cache hit - reusing kernel for {op_pattern}")
        return cls._cache[op_pattern]
    
    @staticmethod
    def _create(op_pattern):
        if op_pattern == "add_mul_relu":
            return AddMulReluKernel()
        raise ValueError(f"Unknown pattern: {op_pattern}")

class AddMulReluKernel:
    def __init__(self):
        self._graph = None
        self._input_nodes = None
        self._initialized = False
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        
        print("Initializing fused kernel graph")
        start = time.perf_counter()
        
        with capture() as builder:
            a = api.asarray(np.zeros(1000000, dtype=np.float32))
            b = api.asarray(np.zeros(1000000, dtype=np.float32))
            c = ops.add(a, b)
            d = ops.mul_scalar(c, 2.0)
            e = ops.relu(d)
        
        self._graph = fuse(builder.graph)
        self._input_nodes = [n for n in self._graph.nodes if n.op == "input"]
        if not self._input_nodes:
            self._input_nodes = [n for n in self._graph.nodes if not n.inputs]
        
        self._initialized = True
        init_time = time.perf_counter() - start
        print(f"Graph initialization took {init_time*1000:.2f} ms")
    
    def __call__(self, a, b):
        self._ensure_initialized()
        
        if len(self._input_nodes) >= 2:
            inputs = {self._input_nodes[0]: a, self._input_nodes[1]: b}
        else:
            if self._graph.nodes:
                node = self._graph.nodes[0]
                inputs = {node: a}
                if len(node.inputs) < 2:
                    node.inputs = [a, b]
            else:
                inputs = {}
        
        return execute(self._graph, inputs)

def benchmark():
    n = 1000000
    print(f"Array size: {n:,} elements\n")
    
    a = api.asarray(np.random.randn(n).astype(np.float32))
    b = api.asarray(np.random.randn(n).astype(np.float32))
    
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
    
    unfused_time = np.median(unfused_times[2:])
    print(f"Unfused: {unfused_time*1000:.2f} ms\n")
    
    print("First execution (should initialize):")
    fused_kernel = FusedKernelCache.get("add_mul_relu")
    
    fused_times = []
    for i in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        start = time.perf_counter()
        result = fused_kernel(a_tmp, b_tmp)
        api.synchronize()
        fused_times.append(time.perf_counter() - start)
    
    fused_time = np.median(fused_times[2:])
    print(f"Fused: {fused_time*1000:.2f} ms")
    print(f"Speedup: {unfused_time/fused_time:.2f}x")
    
    print("Second execution (should use cache):")
    fused_kernel2 = FusedKernelCache.get("add_mul_relu")
    
    fused_times2 = []
    for i in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        start = time.perf_counter()
        result = fused_kernel2(a_tmp, b_tmp)
        api.synchronize()
        fused_times2.append(time.perf_counter() - start)
    
    fused_time2 = np.median(fused_times2[2:])
    print(f"Fused (cached): {fused_time2*1000:.2f} ms")
    print(f"Speedup: {unfused_time/fused_time2:.2f}x")

if __name__ == "__main__":
    benchmark()
