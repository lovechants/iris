import pytest
import numpy as np
from metal_runtime.compiler import get_compiler, CompilationError
from metal_runtime.runtime import get_runtime

VECTOR_ADD_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1024) { // Add a boundary check for robustness
        return;
    }
    result[id] = a[id] + b[id];
}
"""


class TestKernelExecution:
    """
    This test class validates that a compiled kernel is in a valid state
    for execution on the GPU. It does not run the kernel, but it creates
    the MTLComputePipelineState, which is the final object before execution.
    """

    @pytest.fixture
    def device(self):
        return get_runtime().device

    def test_create_compute_pipeline_state(self, device):
        compiler = get_compiler()

        library = compiler.compile(VECTOR_ADD_KERNEL, device)
        assert library is not None, "Library should not be None after compilation"

        function = library.newFunctionWithName_("vector_add")
        assert function is not None, (
            "Function 'vector_add' should be found in the library"
        )

        pipeline_state, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )

        assert pipeline_state is not None, "MTLComputePipelineState should not be None"
        assert error is None, (
            f"Error should be None, but got: {error.localizedDescription() if error else 'Unknown Error'}"
        )

        assert pipeline_state.threadExecutionWidth() > 0, (
            "Thread execution width should be positive"
        )
        assert pipeline_state.maxTotalThreadsPerThreadgroup() > 0, (
            "Max threads per threadgroup should be positive"
        )

    def test_create_pipeline_state_with_invalid_kernel(self, device):
        compiler = get_compiler()

        invalid_source = "kernel void broken() { invalid_syntax; }"

        with pytest.raises(CompilationError):
            compiler.compile(invalid_source, device)

    def test_create_pipeline_state_with_nonexistent_function(self, device):
        compiler = get_compiler()

        library = compiler.compile(VECTOR_ADD_KERNEL, device)

        nonexistent_function = library.newFunctionWithName_("nonexistent_kernel")
        assert nonexistent_function is None, (
            "Should return None for a function that does not exist"
        )
