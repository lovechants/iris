from metal_runtime.ir import IRGraph, IRNode
from typing import List, Dict

ELEMENTWISE = {
    "add",
    "mul",
    "sub",
    "div",
    "relu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "neg",
    "abs",
    "add_scalar",
    "mul_scalar",
    "sub_scalar",
    "div_scalar",
}

REDUCTIONS = {
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_sum_axis",
    "reduce_max_axis",
    "reduce_min_axis",
}
INDEX_OPERATIONS = {"gather", "scatter"}

def can_fuse(node: IRNode) -> bool:
    return node.op in ELEMENTWISE

def find_chains(graph: IRGraph) -> List[List[IRNode]]:
    all_nodes = {n.id: n for n in graph.nodes}
    visited = set()
    chains = []
    for node_id in sorted(all_nodes.keys()):
        node = all_nodes[node_id]
        if node.id in visited or not can_fuse(node):
            continue
        chain = [node]
        visited.add(node.id)
        current = node
        while len(current.users) == 1:
            nxt = current.users[0]
            is_simple_successor = all(inp == current or inp.op == "input" for inp in nxt.inputs)
            if not can_fuse(nxt) or not is_simple_successor:
                 break
            chain.append(nxt)
            visited.add(nxt.id)
            current = nxt
        if len(chain) > 1:
            chains.append(chain)
    return chains

def fuse(graph: IRGraph) -> IRGraph:
    chains = find_chains(graph)
    if not chains:
        return graph

    new_graph = IRGraph()
    node_map: Dict[int, IRNode] = {}
    fused_set = {n.id for chain in chains for n in chain}

    for node in sorted(graph.nodes, key=lambda n: n.id):
        if node.id in fused_set:
            continue
        
        new_inputs = [node_map[inp.id] for inp in node.inputs]
        new_node = IRNode(node.op, new_inputs, node.attrs)
        new_node.shape = node.shape
        new_node.dtype = node.dtype
        node_map[node.id] = new_node
        new_graph.add_node(new_node)

    for chain in chains:
        first = chain[0]
        last = chain[-1]
        
        new_inputs = [node_map[inp.id] for inp in first.inputs]
        fused_node = IRNode(
            "fused",
            new_inputs,
            {"ops": [n.op for n in chain], "attrs": [n.attrs for n in chain]},
        )
        fused_node.shape = last.shape
        fused_node.dtype = last.dtype
        new_graph.add_node(fused_node)
        
        for node_in_chain in chain:
            node_map[node_in_chain.id] = fused_node
            
    for node in new_graph.nodes:
        if node.op != "fused":
             for i, inp in enumerate(node.inputs):
                 if inp.id in node_map:
                     node.inputs[i] = node_map[inp.id]
    
    new_graph.outputs = [node_map[n.id] for n in graph.outputs]
    new_graph._node_map = node_map
    return new_graph
