import time
import numpy as np
from typing import Dict, Any, Callable
from metal_runtime.ir_capture import capture
from metal_runtime.fusion import fuse
from metal_runtime.executor import execute
from metal_runtime import api, ops, scatter, gather, reduction


class FusionCache:
    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, pattern_key: str):
        if pattern_key not in cls._cache:
            cls._cache[pattern_key] = cls._create(pattern_key)
        return cls._cache[pattern_key]

    @staticmethod
    def _create(pattern_key: str):
        if pattern_key == "add_mul_relu":
            return AddMulRelu()
        elif pattern_key == "add_sigmoid":
            return AddSigmoid()
        elif pattern_key == "mul_tanh":
            return MulTanh()
        elif pattern_key == "add_scalar_mul_scalar":
            return AddScalarMulScalar()
        elif pattern_key == "add_scalar_sub_scalar":
            return AddScalarSubScalar()
        elif pattern_key == "mul_scalar_div_scalar":
            return MulScalarDivScalar()
        elif pattern_key == "reduce_sum_reduce_max":
            return ReduceSumReduceMax()
        elif pattern_key == "gather_add":
            return GatherAdd()
        elif pattern_key == "scatter_mul":
            return ScatterMul()
        else:
            raise ValueError(f"Unknown pattern: {pattern_key}")


def _set_graph_outputs(builder):
    if not builder.graph.outputs:
        for node in builder.graph.nodes:
            if not node.users and node.op != "input":
                builder.graph.outputs.append(node)


class AddMulRelu:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            b = api.asarray(np.zeros(1, dtype=np.float32))
            c = ops.add(a, b)
            d = ops.mul_scalar(c, 2.0)
            e = ops.relu(d)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, b):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a, self._input_nodes[1]: b}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class AddSigmoid:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            b = api.asarray(np.zeros(1, dtype=np.float32))
            c = ops.add(a, b)
            d = ops.sigmoid(c)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, b):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a, self._input_nodes[1]: b}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class MulTanh:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            b = api.asarray(np.zeros(1, dtype=np.float32))
            c = ops.mul(a, b)
            d = ops.tanh(c)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, b):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a, self._input_nodes[1]: b}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class AddScalarMulScalar:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            c = ops.add_scalar(a, 2.0)
            d = ops.mul_scalar(c, 3.0)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, scalar1, scalar2):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class AddScalarSubScalar:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            c = ops.add_scalar(a, 2.0)
            d = ops.sub_scalar(c, 3.0)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, scalar1, scalar2):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class MulScalarDivScalar:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            c = ops.mul_scalar(a, 2.0)
            d = ops.div_scalar(c, 3.0)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, scalar1, scalar2):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class ReduceSumReduceMax:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            c = reduction.reduce_sum(a)
            d = reduction.reduce_max(c)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class GatherAdd:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            idx = api.asarray(np.zeros(1, dtype=np.int32))
            c = gather.gather(a, idx)
            d = ops.add(c, 1.0)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, idx):
        self._ensure_initialized()
        inputs = {self._input_nodes[0]: a, self._input_nodes[1]: idx}
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


class ScatterMul:
    def __init__(self):
        self._graph = None
        self._input_nodes = None

    def _ensure_initialized(self):
        if self._graph is not None:
            return

        with capture() as builder:
            a = api.asarray(np.zeros(1, dtype=np.float32))
            idx = api.asarray(np.zeros(1, dtype=np.int32))
            val = api.asarray(np.zeros(1, dtype=np.float32))
            c = scatter.scatter(a, idx, val)
            d = ops.mul(c, 2.0)

        _set_graph_outputs(builder)
        self._graph = fuse(builder.graph)
        self._input_nodes = sorted(
            [n for n in self._graph.nodes if n.op == "input"], key=lambda n: n.id
        )

    def __call__(self, a, idx, val):
        self._ensure_initialized()
        inputs = {
            self._input_nodes[0]: a,
            self._input_nodes[1]: idx,
            self._input_nodes[2]: val,
        }
        results = execute(self._graph, inputs)
        return results[self._graph.outputs[0]]


_cache = FusionCache()


def get_fused_operation(pattern_key: str):
    return _cache.get(pattern_key)
