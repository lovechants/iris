from metal_runtime.ir import IRGraph, IRNode
from metal_runtime.runtime import MetalBuffer
from metal_runtime import api, ops
from metal_runtime.fusion_codegen import generate_fused
from typing import Dict
import hashlib

def execute(graph: IRGraph, inputs: Dict[IRNode, MetalBuffer]) -> Dict[IRNode, MetalBuffer]:
    cache = dict(inputs)
    for node in graph.topo_sort():
        if node in cache:
            continue
        if node.op == "fused":
            input_bufs = [cache[inp] for inp in node.inputs]
            out = api.empty_like(input_bufs[0])
            src, kernel_name = generate_fused(node)
            args = list(input_bufs)
            for i, attrs in enumerate(node.attrs["attrs"]):
                if "scalar" in attrs:
                    args.append(attrs["scalar"])
            args.extend([out, input_bufs[0].numel])
            api.launch(src, kernel_name, (input_bufs[0].numel,), (256,), args)
            cache[node] = out
        else:
            input_bufs = [cache[inp] for inp in node.inputs]
            op_func = getattr(ops, node.op)
            if node.attrs and "scalar" in node.attrs:
                result = op_func(input_bufs[0], node.attrs["scalar"])
            elif len(input_bufs) == 1:
                result = op_func(input_bufs[0])
            else:
                result = op_func(*input_bufs)
            cache[node] = result
    return {node: cache[node] for node in graph.outputs}
