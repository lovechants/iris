from metal_runtime.ir import IRGraph, IRNode
from typing import List, Tuple, Dict

ELEMENTWISE = {"add", "mul", "sub", "div", "relu", "sigmoid", "tanh", 
               "exp", "log", "sqrt", "neg", "abs",
               "add_scalar", "mul_scalar", "sub_scalar", "div_scalar"}

def can_fuse(node: IRNode) -> bool:
    return node.op in ELEMENTWISE

def find_chains(graph: IRGraph) -> List[List[IRNode]]:
    visited = set()
    chains = []
    for node in graph.topo_sort():
        if node.id in visited or not can_fuse(node):
            continue
        chain = [node]
        visited.add(node.id)
        current = node
        while len(current.users) == 1:
            next_node = current.users[0]
            if not can_fuse(next_node):
                break
            if len(next_node.inputs) > 1 and next_node.inputs[0] != current:
                break
            chain.append(next_node)
            visited.add(next_node.id)
            current = next_node
        if len(chain) > 1:
            chains.append(chain)
    return chains

def fuse(graph: IRGraph) -> IRGraph:
    chains = find_chains(graph)
    if not chains:
        return graph
    
    new_graph = IRGraph()
    node_map = {}  
    fused_set = {n.id for chain in chains for n in chain}
    
    for node in graph.topo_sort():
        if node.id in fused_set:
            continue
        
        new_inputs = [node_map.get(inp.id, inp) for inp in node.inputs]
        new_node = IRNode(node.op, new_inputs, node.attrs)
        new_node.shape = node.shape
        new_node.dtype = node.dtype
        node_map[node.id] = new_node
        new_graph.add_node(new_node)
    
    for chain in chains:
        first = chain[0]
        last = chain[-1]
        
        new_inputs = [node_map.get(inp.id, inp) for inp in first.inputs]
        fused_node = IRNode(
            "fused",
            new_inputs,
            {"ops": [n.op for n in chain], "attrs": [n.attrs for n in chain]}
        )
        fused_node.shape = last.shape
        fused_node.dtype = last.dtype
        
        node_map[last.id] = fused_node
        new_graph.add_node(fused_node)
    
    new_graph.outputs = [node_map[n.id] for n in graph.outputs]
    
    new_graph._node_map = node_map
    
    return new_graph
