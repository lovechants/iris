import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api

@kernel(
    param_types={
        "a": "device const float*",
        "b": "device const float*",
        "c": "device float*",
        "n": "uint",
    }
)
def add(a, b, c, n):
    i = metal.thread_id()
    if i < n:
        c[i] = a[i] + b[i]


def main():
    n = 1024
    
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    
    a_buf = api.asarray(a)
    b_buf = api.asarray(b)
    c_buf = api.empty((n,), api.DType.FLOAT32)
    
    api.launch_kernel("add", grid=(n,), block=(256,), args=[a_buf, b_buf, c_buf, n])
    
    result = api.to_numpy(c_buf)
    expected = a + b
    
    print(f"Max error: {np.max(np.abs(result - expected))}")
    print(f"Results match: {np.allclose(result, expected, rtol=1e-5)}")


if __name__ == "__main__":
    main()
