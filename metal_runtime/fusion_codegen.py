from metal_runtime.ir import IRNode
from metal_runtime.dtype import DType
import hashlib

def generate_fused(node: IRNode) -> tuple[str, str]:
    ops = node.attrs["ops"]
    attrs_list = node.attrs["attrs"]
    dtype_map = {DType.FLOAT32: "float", DType.FLOAT16: "half"}
    metal_type = dtype_map[node.dtype]
    
    ops_str = "_".join(ops)
    kernel_hash = hashlib.sha256(ops_str.encode()).hexdigest()[:8]
    kernel_name = f"fused_{kernel_hash}"
    
    kernel = f"#include <metal_stdlib>\nusing namespace metal;\n\n"
    kernel += f"kernel void {kernel_name}(\n"  
    
    buf_idx = 0
    for i in range(len(node.inputs)):
        kernel += f"    device const {metal_type}* in{i} [[buffer({buf_idx})]],\n"
        buf_idx += 1
    
    scalar_map = {}
    for i, (op, attrs) in enumerate(zip(ops, attrs_list)):
        if "scalar" in op and "scalar" in attrs:
            scalar_map[i] = buf_idx
            kernel += f"    constant {metal_type}& s{i} [[buffer({buf_idx})]],\n"
            buf_idx += 1
    
    kernel += f"    device {metal_type}* out [[buffer({buf_idx})]],\n"
    kernel += f"    constant uint& n [[buffer({buf_idx+1})]],\n"
    kernel += f"    uint tid [[thread_position_in_grid]]\n"
    kernel += ") {\n"
    kernel += "    if (tid < n) {\n"
    
    if len(node.inputs) > 0:
        kernel += f"        {metal_type} val = in0[tid];\n"
    else:
        kernel += f"        {metal_type} val = 0.0f;\n"
    
    for i, (op, attrs) in enumerate(zip(ops, attrs_list)):
        if op == "add":
            if i == 0 and len(node.inputs) > 1:
                kernel += f"        val = val + in1[tid];\n"
            else:
                kernel += f"        val = val + in0[tid];\n"
        elif op == "mul":
            if i == 0 and len(node.inputs) > 1:
                kernel += f"        val = val * in1[tid];\n"
            else:
                kernel += f"        val = val * in0[tid];\n"
        elif op == "add_scalar":
            kernel += f"        val = val + s{i};\n"
        elif op == "mul_scalar":
            kernel += f"        val = val * s{i};\n"
        elif op == "relu":
            kernel += "        val = max(val, 0.0f);\n"
        elif op == "sigmoid":
            kernel += "        val = 1.0f / (1.0f + exp(-val));\n"
        elif op == "tanh":
            kernel += "        val = tanh(val);\n"
        elif op == "exp":
            kernel += "        val = exp(val);\n"
        elif op == "log":
            kernel += "        val = log(val);\n"
        elif op == "sqrt":
            kernel += "        val = sqrt(val);\n"
        elif op == "neg":
            kernel += "        val = -val;\n"
        elif op == "abs":
            kernel += "        val = abs(val);\n"
    
    kernel += "        out[tid] = val;\n"
    kernel += "    }\n}\n"
    
    return kernel, kernel_name
