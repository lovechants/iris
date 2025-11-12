from metal_runtime.ir import IRGraph, IRNode
from metal_runtime.runtime import MetalBuffer
from metal_runtime import api, ops
from metal_runtime.fusion_codegen import generate_fused
from typing import Dict
from metal_runtime.parallel_compiler import get_parallel_compiler

def execute(graph: IRGraph, inputs: Dict[IRNode, MetalBuffer]) -> Dict[IRNode, MetalBuffer]:
    cache: Dict[IRNode, MetalBuffer] = {}

    if hasattr(graph, "_input_map"):
        for new_input, old_input in graph._input_map.items():
            if old_input in inputs:
                cache[new_input] = inputs[old_input]
    else:
        cache.update(inputs)

    compiler = get_parallel_compiler()
    graph_hash = compiler._graph_hash(graph)
    compiled = compiler.get_compiled(graph_hash)

    for node in graph.topo_sort():
        if node in cache:
            continue

        if node.op == "fused":
            input_bufs = [cache[inp] for inp in node.inputs]
            out = api.empty_like(input_bufs[0])
            if compiled:
                src, kernel_name = compiled.source, compiled.name
            else:
                src, kernel_name = generate_fused(node)
            args = list(input_bufs)
            for i, attrs in enumerate(node.attrs["attrs"]):
                if "scalar" in attrs:
                    args.append(attrs["scalar"])
            args.extend([out, input_bufs[0].numel])
            api.launch(src, kernel_name, (input_bufs[0].numel,), (256,), args)
            cache[node] = out
            continue

        if node.op in ["reduce_sum", "reduce_max", "reduce_min"]:
            from metal_runtime.reduction import reduce_sum, reduce_max, reduce_min
            input_bufs = [cache[inp] for inp in node.inputs]
            axis = node.attrs.get("axis") if node.attrs else None
            if node.op == "reduce_sum":
                cache[node] = reduce_sum(input_bufs[0], axis)
            elif node.op == "reduce_max":
                cache[node] = reduce_max(input_bufs[0], axis)
            else:
                cache[node] = reduce_min(input_bufs[0], axis)
            continue

        if node.op in ["gather", "scatter"]:
            from metal_runtime.gather import gather
            from metal_runtime.scatter import scatter
            if node.op == "gather":
                input_buf = cache[node.inputs[0]]
                idx = cache[node.inputs[1]]
                axis = node.attrs.get("axis", 0)
                cache[node] = gather(input_buf, idx, axis)
            else:
                input_buf = cache[node.inputs[0]]
                idx = cache[node.inputs[1]]
                val = cache[node.inputs[2]]
                axis = node.attrs.get("axis", 0)
                cache[node] = scatter(input_buf, idx, val, axis)
            continue

        if node.op == "input":
            continue

        input_bufs = [cache[inp] for inp in node.inputs]
        op_func = getattr(ops, node.op)
        if node.attrs and "scalar" in node.attrs:
            cache[node] = op_func(input_bufs[0], node.attrs["scalar"])
        elif len(input_bufs) == 1:
            cache[node] = op_func(input_bufs[0])
        else:
            cache[node] = op_func(*input_bufs)

    return {n: cache[n] for n in graph.outputs}
