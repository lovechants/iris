from contextlib import contextmanager
from typing import Optional, Dict, List
from threading import local
from metal_runtime.ir import IRGraph, IRNode
from metal_runtime.runtime import MetalBuffer

_context = local()


class MinimalIRBuilder:
    __slots__ = ("graph", "buf_to_node", "node_to_buf")

    def __init__(self):
        self.graph = IRGraph()
        self.buf_to_node = {}
        self.node_to_buf = {}

    def make_node(self, op: str, inputs: List[MetalBuffer], attrs: Dict = None):
        input_nodes = tuple(
            self.buf_to_node.get(id(buf)) or self._create_input(buf)
            for buf in inputs
        )

        node = IRNode(op, list(input_nodes), attrs)

        if inputs:
            node.shape = inputs[0].shape
            node.dtype = inputs[0].dtype

        self.graph.add_node(node)
        return node

    def _create_input(self, buf):
        node = IRNode("input", [])
        node.shape = buf.shape
        node.dtype = buf.dtype
        self.graph.add_node(node)  
        self.buf_to_node[id(buf)] = node
        self.node_to_buf[node] = buf
        return node

    def register(self, buf: MetalBuffer, node: IRNode):
        self.buf_to_node[id(buf)] = node
        self.node_to_buf[node] = buf

    def set_output(self, buf: MetalBuffer):
        node = self.buf_to_node.get(id(buf))
        if node and node not in self.graph.outputs:
            self.graph.outputs.append(node)


@contextmanager
def capture():
    builder = MinimalIRBuilder()
    _context.builder = builder
    try:
        yield builder
    finally:
        _context.builder = None


def current() -> Optional[MinimalIRBuilder]:
    return getattr(_context, "builder", None)


def is_capturing() -> bool:
    return current() is not None
