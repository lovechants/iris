import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.ir_capture import capture
from metal_runtime.fusion import fuse
from metal_runtime.executor import execute
from metal_runtime.logging import LogLevel, OperationType, configure_logging, get_logger

def e2e():
    configure_logging(level=LogLevel.PROFILE)
    logger = get_logger()

    n = 8_388_608  
    logger.info(f"Starting e2e benchmark with {n:,} elements")

    with logger.timed_operation(OperationType.MEMORY, "Data Allocation & Upload"):
        x_np = np.random.randn(n).astype(np.float32)
        y_np = np.random.randn(n).astype(np.float32)
        x = api.asarray(x_np, persistent=True)
        y = api.asarray(y_np, persistent=True)

    with logger.timed_operation(OperationType.FUSION, "Graph Capture & Fusion"):
        with capture() as builder:
            a = ops.add(x, y)
            b = ops.mul_scalar(a, 1.25)
            c = ops.relu(b)
            d = ops.exp(c)
            builder.set_output(d)
        graph = builder.graph
        fused_graph = fuse(graph)
        logger.info(f"Original graph: {len(graph.nodes)} nodes. Fused graph: {len(fused_graph.nodes)} nodes.")

    with logger.timed_operation(OperationType.COMPUTE, "Correctness Verification"):
        fused_inputs = {}
        node_map = fused_graph._node_map
        for old_node, buf in builder.node_to_buf.items():
            if old_node.id in node_map:
                fused_inputs[node_map[old_node.id]] = buf

        fused_outputs = execute(fused_graph, fused_inputs)
        fused_result = api.to_numpy(fused_outputs[fused_graph.outputs[0]])
        expected_result = np.exp(np.maximum((x_np + y_np) * 1.25, 0))
        assert np.allclose(fused_result, expected_result, rtol=1e-5, atol=1e-5)
        logger.info("Correctness check PASSED.")

    logger.info("Starting performance benchmark")
    execute(fused_graph, fused_inputs)
    api.synchronize()

    with logger.timed_operation(OperationType.COMPUTE, "Benchmark Loop (10 runs)"):
        for _ in range(10):
            execute(fused_graph, fused_inputs)
        api.synchronize() 

    total_time_ms = logger.operation_stack[-1].time_ms
    avg_time_ms = total_time_ms / 10
    logger.profile(f"Average execution time per run: {avg_time_ms:.4f} ms")

if __name__ == "__main__":
    e2e()
