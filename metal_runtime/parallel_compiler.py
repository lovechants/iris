from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import hashlib
import json
import traceback
from typing import Dict, Optional, Tuple
from metal_runtime.ir import IRGraph, IRNode
from metal_runtime.runtime import MetalRuntime, get_runtime
from metal_runtime.fusion_codegen import generate_fused
from metal_runtime.jit import get_jit_cache
from metal_runtime.logging import get_logger, LogLevel, OperationType
import numpy as np  

class CompiledKernel:
    def __init__(self, source: str, name: str, hash: str, library=None):
        self.source = source
        self.name = name
        self.hash = hash
        self.library = library

def json_serializer(obj):
    """Converts NumPy numeric types to native Python types for JSON serialization."""
    if isinstance(obj, np.number):
        return obj.item()  
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class ParallelCompiler:
    def __init__(self, num_workers: int = 2):
        self.workers = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="compiler")
        self.compiled_cache: Dict[str, CompiledKernel] = {}
        self.in_progress: Dict[str, bool] = {}
        self._cache_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self.runtime = get_runtime()
        self.logger = get_logger()

    def compile_async(self, graph: IRGraph) -> str:
        graph_hash = self._graph_hash(graph)
        with self._cache_lock:
            if graph_hash in self.compiled_cache:
                return graph_hash
        with self._progress_lock:
            if graph_hash in self.in_progress:
                return graph_hash
            self.in_progress[graph_hash] = True
        self.workers.submit(self._compile_worker, graph, graph_hash)
        return graph_hash

    def _compile_worker(self, graph: IRGraph, graph_hash: str):
        t0 = time.perf_counter()
        try:
            kernel_source, kernel_name = self._compile_graph(graph)
            compiled = CompiledKernel(source=kernel_source, name=kernel_name, hash=graph_hash)
            with self._cache_lock:
                self.compiled_cache[graph_hash] = compiled
            with self._progress_lock:
                self.in_progress.pop(graph_hash, None)
            t1 = time.perf_counter()
            self.logger.profile(f"compile:{kernel_name}", phase="compile", time_ms=(t1 - t0) * 1000)
        except Exception as e:
            self.logger.error(f"Compilation error for {graph_hash}: {e}")
            with self._progress_lock:
                self.in_progress.pop(graph_hash, None)

    def get_compiled(self, graph_hash: str) -> Optional[CompiledKernel]:
        with self._cache_lock:
            return self.compiled_cache.get(graph_hash)

    def is_compiling(self, graph_hash: str) -> bool:
        with self._progress_lock:
            return graph_hash in self.in_progress

    def _compile_graph(self, graph: IRGraph) -> Tuple[str, str]:
        for node in graph.topo_sort():
            if node.op == "fused":
                return generate_fused(node)
        raise NotImplementedError("Only fused graphs are currently supported")

    def _graph_hash(self, graph: IRGraph) -> str:
        hasher = hashlib.md5()
        sorted_nodes = graph.topo_sort()
        node_positions = {node: i for i, node in enumerate(sorted_nodes)}
        for node in sorted_nodes:
            hasher.update(node.op.encode())
            for key, value in sorted(node.attrs.items()):
                try:
                    value_str = json.dumps(value, sort_keys=True, default=json_serializer)
                    hasher.update(f"{key}:{value_str}".encode())
                except TypeError:
                    hasher.update(f"{key}:{str(value)}".encode())

            for input_node in node.inputs:
                if input_node in node_positions:
                    hasher.update(str(node_positions[input_node]).encode())
        return hasher.hexdigest()

_parallel_compiler = None

def get_parallel_compiler() -> ParallelCompiler:
    global _parallel_compiler
    if _parallel_compiler is None:
        _parallel_compiler = ParallelCompiler()
    return _parallel_compiler
