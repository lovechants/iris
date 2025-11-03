from contextlib import contextmanager
from typing import Optional, Dict, List
from threading import local
from metal_runtime.ir import IRGraph, IRNode
from metal_runtime.runtime import MetalBuffer

_context = local()

class IRBuilder:
    def __init__(self):
        self.graph = IRGraph()
        self.buf_to_node: Dict[int, IRNode] = {}
        self.node_to_buf: Dict[IRNode, MetalBuffer] = {}  
    
    def make_node(self, op: str, inputs: List[MetalBuffer], attrs: Dict = None) -> IRNode:
        input_nodes = []
        for buf in inputs:
            if id(buf) in self.buf_to_node:
                input_nodes.append(self.buf_to_node[id(buf)])
            else:
                placeholder_node = IRNode("input", [])
                placeholder_node.shape = buf.shape
                placeholder_node.dtype = buf.dtype
                self.buf_to_node[id(buf)] = placeholder_node
                self.node_to_buf[placeholder_node] = buf
                input_nodes.append(placeholder_node)
        
        node = IRNode(op, input_nodes, attrs)
        node.shape = inputs[0].shape if inputs else None
        node.dtype = inputs[0].dtype if inputs else None
        self.graph.add_node(node)
        return node
    
    def register(self, buf: MetalBuffer, node: IRNode):
        self.buf_to_node[id(buf)] = node
        self.node_to_buf[node] = buf
    
    def set_output(self, buf: MetalBuffer):
        node = self.buf_to_node[id(buf)]
        if node not in self.graph.outputs:
            self.graph.outputs.append(node)

@contextmanager
def capture():
    builder = IRBuilder()
    _context.builder = builder
    try:
        yield builder
    finally:
        _context.builder = None

def current() -> Optional[IRBuilder]:
    return getattr(_context, 'builder', None)

def is_capturing() -> bool:
    return current() is not None
