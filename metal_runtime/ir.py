from typing import List, Dict, Set, Optional, Any
from metal_runtime.runtime import MetalBuffer
from metal_runtime.dtype import DType

class IRNode:
    _id_counter = 0
    
    def __init__(self, op: str, inputs: List['IRNode'], attrs: Dict[str, Any] = None):
        self.id = IRNode._id_counter
        IRNode._id_counter += 1
        
        self.op = op
        self.inputs = inputs
        self.attrs = attrs or {}
        self.users: List[IRNode] = []
        
        for inp in inputs:
            inp.users.append(self)
        
        self.shape: Optional[tuple] = None
        self.dtype: Optional[DType] = None
    
    def __repr__(self):
        return f"Node({self.id}: {self.op})"


class IRGraph:
    def __init__(self):
        self.nodes: List[IRNode] = []
        self.outputs: List[IRNode] = []
    
    def add_node(self, node: IRNode) -> IRNode:
        self.nodes.append(node)
        return node
    
    def topo_sort(self) -> List[IRNode]:
        all_nodes = set(self.nodes)
        for output in self.outputs:
            def collect_inputs(node):
                for inp in node.inputs:
                    all_nodes.add(inp)
                    collect_inputs(inp)
            collect_inputs(output)
        
        visited: Set[int] = set()
        order: List[IRNode] = []
        
        def visit(node: IRNode):
            if node.id in visited:
                return
            visited.add(node.id)
            for inp in node.inputs:
                visit(inp)
            if node.op != "input":
                order.append(node)
        
        for out in self.outputs:
            visit(out)
        
        return order
