import time
import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api, ops
from metal_runtime.dtype import DType

# Ensure persistent buffer residency is enabled globally
api.enable_persistence_default(True)


def benchmark_unfused_operations(n):
    """Benchmarks a chain of operations executed in separate kernels."""
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    scalar_val = 2.0

    a_buf = api.asarray(a_np)
    b_buf = api.asarray(b_np)

    start = time.perf_counter()

    c_buf = ops.add(a_buf, b_buf)                 # Kernel 1: add
    d_buf = ops.mul_scalar(c_buf, scalar_val)     # Kernel 2: mul_scalar
    e_buf = ops.relu(d_buf)                       # Kernel 3: relu

    api.synchronize()
    result = api.to_numpy(e_buf)

    end = time.perf_counter()
    return end - start

@kernel(
    param_types={
        "a": "device const float*",
        "b": "device const float*",
        "scalar": "device const float*",
        "out": "device float*",
        "n": "uint",
    }
)
def add_mul_relu_fused(a, b, scalar, out, n):
    i = metal.thread_id()
    if i < n:
        val = a[i] + b[i]           # Add
        val = val * scalar[0]       # Multiply
        out[i] = val if val > 0 else 0  # ReLU


def benchmark_fused_operation(n):
    """Benchmarks the same logic using a single fused kernel."""
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    scalar_np = np.array([2.0], dtype=np.float32)

    # Upload once
    a_buf = api.asarray(a_np)
    b_buf = api.asarray(b_np)
    scalar_buf = api.asarray(scalar_np)
    out_buf = api.empty_like(a_buf)

    start = time.perf_counter()

    api.launch_kernel(
        "add_mul_relu_fused",
        grid=(n,),
        block=(256,),
        args=[a_buf, b_buf, scalar_buf, out_buf, n],
    )

    api.synchronize()
    result = api.to_numpy(out_buf)

    end = time.perf_counter()
    return end - start



def run_benchmark(n):
    """Runs both kernels and compares against NumPy baseline."""

    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    start = time.perf_counter()
    c_np = np.maximum((a_np + b_np) * 2.0, 0)
    numpy_time = time.perf_counter() - start

    unfused_time = benchmark_unfused_operations(n)
    fused_time = benchmark_fused_operation(n)

    print(f"  NumPy time:                 {numpy_time * 1000:.4f} ms")
    print(f"  Unfused Metal time:        {unfused_time * 1000:.4f} ms"
          f"  ({numpy_time / unfused_time:.2f}x speedup vs NumPy)")
    print(f"  FUSED Metal time:          {fused_time * 1000:.4f} ms"
          f"    ({numpy_time / fused_time:.2f}x speedup vs NumPy)")
    print(f"  Fusion Speedup (Metal):      {unfused_time / fused_time:.2f}x")


def main():
    print("Demonstrating Persistent Buffers + Kernel Fusion.\n")
    print("(Persistent residency keeps data on GPU between launches.)\n")

    for n in [10_000, 100_000, 1_000_000]:
        run_benchmark(n)


if __name__ == "__main__":
    main()
