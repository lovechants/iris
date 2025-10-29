import Metal
import struct
import numpy as np
from typing import Any, List, Tuple
from metal_runtime.runtime import MetalRuntime, MetalBuffer
from metal_runtime.compiler import get_compiler
from metal_runtime.jit import get_jit_cache

class KernelLauncher:
    def __init__(self, runtime: MetalRuntime):
        self.runtime = runtime
        self.compiler = get_compiler()
        self._pipeline_cache = {}

    def _get_pipeline(
        self, source: str, function_name: str
    ) -> "Metal.MTLComputePipelineState":
        cache_key = (hash(source), function_name)
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        library = get_jit_cache().compile(source, function_name, self.runtime.device)
        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise ValueError(
                f"Function '{function_name}' not found in compiled library"
            )

        pipeline, error = (
            self.runtime.device.newComputePipelineStateWithFunction_error_(
                function, None
            )
        )

        if pipeline is None:
            error_msg = error.localizedDescription() if error else "Unknown error"
            raise RuntimeError(f"Failed to create pipeline state: {error_msg}")

        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def _encode_argument(
        self, arg: Any, index: int, encoder: "Metal.MTLComputeCommandEncoder"
    ):
        if isinstance(arg, MetalBuffer):
            encoder.setBuffer_offset_atIndex_(arg.buffer, 0, index)
        elif isinstance(arg, int):
            data = struct.pack("i", arg)
            encoder.setBytes_length_atIndex_(data, len(data), index)
        elif isinstance(arg, float) or isinstance(arg, np.floating):
            if isinstance(arg, np.floating):
                arg = float(arg)
            data = struct.pack("f", arg)
            encoder.setBytes_length_atIndex_(data, len(data), index)
        else:
            raise TypeError(f"Unsupported argument type: {type(arg)}")

    def launch(
        self,
        source: str,
        function_name: str,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: List[Any],
    ):
        pipeline = self._get_pipeline(source, function_name)

        max_threads = pipeline.maxTotalThreadsPerThreadgroup()
        block_size = block[0] * block[1] * block[2]
        if block_size > max_threads:
            raise ValueError(f"Block size {block_size} exceeds maximum {max_threads}")

        cmd_buffer = self.runtime.queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        for i, arg in enumerate(args):
            self._encode_argument(arg, i, encoder)

        grid_size = Metal.MTLSize(grid[0], grid[1], grid[2])
        threadgroup_size = Metal.MTLSize(block[0], block[1], block[2])

        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()

        cmd_buffer.commit()
        cmd_buffer.waitUntilCompleted()

        if cmd_buffer.status() == Metal.MTLCommandBufferStatusError:
            error = cmd_buffer.error()
            error_msg = error.localizedDescription() if error else "Unknown error"
            raise RuntimeError(f"Kernel execution failed: {error_msg}")
