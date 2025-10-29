import pytest
from metal_runtime.codegen import generate_msl, MSLCodegenError
import Metal


class TestCodegen:
    def test_simple_add_kernel(self):
        def add(a, b, c, n):
            i = metal.thread_id()
            if i < n:
                c[i] = a[i] + b[i]

        param_types = {
            "a": "device const float*",
            "b": "device const float*",
            "c": "device float*",
            "n": "uint",
        }

        msl = generate_msl(add, "add", param_types)

        assert "#include <metal_stdlib>" in msl
        assert "using namespace metal;" in msl
        assert "kernel void add(" in msl
        assert "device const float* a [[buffer(0)]]" in msl
        assert "device const float* b [[buffer(1)]]" in msl
        assert "device float* c [[buffer(2)]]" in msl
        assert "constant uint& n [[buffer(3)]]" in msl
        assert "uint tid [[thread_position_in_grid]]" in msl
        assert "uint i = tid;" in msl
        assert "if (i < n)" in msl
        assert "c[i] = (a[i] + b[i]);" in msl

    def test_multiply_kernel(self):
        def mul(x, y, scale):
            tid = metal.thread_id()
            y[tid] = x[tid] * scale

        param_types = {
            "x": "device const float*",
            "y": "device float*",
            "scale": "float",
        }

        msl = generate_msl(mul, "mul", param_types)

        assert "kernel void mul(" in msl
        assert "device const float* x" in msl
        assert "device float* y" in msl
        assert "constant float& scale" in msl
        assert "y[tid] = (x[tid] * scale);" in msl

    def test_for_loop(self):
        def loop_kernel(out, n):
            for i in range(n):
                out[i] = i

        param_types = {
            "out": "device float*",
            "n": "uint",
        }

        msl = generate_msl(loop_kernel, "loop_kernel", param_types)

        assert "for (uint i = 0; i < n; i += 1)" in msl
        assert "out[i] = i;" in msl

    def test_for_loop_with_start_end(self):
        def loop_kernel(out, start, end):
            for i in range(start, end):
                out[i] = i

        param_types = {
            "out": "device float*",
            "start": "uint",
            "end": "uint",
        }

        msl = generate_msl(loop_kernel, "loop_kernel", param_types)

        assert "for (uint i = start; i < end; i += 1)" in msl

    def test_if_else(self):
        def conditional(x, y, threshold):
            tid = metal.thread_id()
            if x[tid] > threshold:
                y[tid] = x[tid]
            else:
                y[tid] = 0

        param_types = {
            "x": "device const float*",
            "y": "device float*",
            "threshold": "float",
        }

        msl = generate_msl(conditional, "conditional", param_types)

        assert "if (x[tid] > threshold)" in msl
        assert "} else {" in msl
        assert "y[tid] = 0;" in msl

    def test_comparison_operators(self):
        def compare(a, b, out):
            tid = metal.thread_id()
            if a[tid] < b[tid]:
                out[tid] = 1
            elif a[tid] > b[tid]:
                out[tid] = 2
            elif a[tid] == b[tid]:
                out[tid] = 3

        param_types = {
            "a": "device const float*",
            "b": "device const float*",
            "out": "device float*",
        }

        msl = generate_msl(compare, "compare", param_types)

        assert "if (a[tid] < b[tid])" in msl
        assert "} else if (a[tid] > b[tid])" in msl or "if (a[tid] > b[tid])" in msl
        assert "} else if (a[tid] == b[tid])" in msl or "if (a[tid] == b[tid])" in msl

    def test_binary_operations(self):
        def ops(a, b, out):
            tid = metal.thread_id()
            out[tid] = (a[tid] + b[tid]) * (a[tid] - b[tid]) / 2

        param_types = {
            "a": "device const float*",
            "b": "device const float*",
            "out": "device float*",
        }

        msl = generate_msl(ops, "ops", param_types)

        assert "+" in msl
        assert "-" in msl
        assert "*" in msl
        assert "/" in msl

    def test_unary_operations(self):
        def unary(x, out):
            tid = metal.thread_id()
            out[tid] = -x[tid]

        param_types = {
            "x": "device const float*",
            "out": "device float*",
        }

        msl = generate_msl(unary, "unary", param_types)

        assert "out[tid] = -x[tid];" in msl

    def test_missing_param_type(self):
        def kernel_func(a, b):
            pass

        param_types = {"a": "device float*"}

        with pytest.raises(MSLCodegenError, match="Unknown type"):
            generate_msl(kernel_func, "kernel_func", param_types)
