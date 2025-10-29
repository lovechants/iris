import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api
from management.partition_util import partition_1d, explain_partition, print_device_info

print_device_info()
print()


@kernel(
    param_types={
        "a": "device const float*",
        "b": "device const float*",
        "c": "device float*",
        "n": "uint",
    }
)
def vector_add(a, b, c, n):
    i = metal.thread_id()
    if i < n:
        c[i] = a[i] + b[i]


n = 512
grid_size, block_size = partition_1d(n)

print(f"Recommended partitioning for {n} elements:")
print(f"  Grid size: {grid_size}")
print(f"  Block size: {block_size}")
print()
print(explain_partition(n, grid_size, block_size))

a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)

a_buf = api.asarray(a_np)
b_buf = api.asarray(b_np)
c_buf = api.empty((n,), api.DType.FLOAT32)

api.launch_kernel(
    "vector_add", grid=grid_size, block=block_size, args=[a_buf, b_buf, c_buf, n]
)
