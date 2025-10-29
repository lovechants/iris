import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api


@kernel(
    param_types={
        "x": "device const float*",
        "y": "device float*",
        "n": "uint",
    }
)
def scale_by_two(x, y, n):
    i = metal.thread_id()
    if i < n:
        y[i] = x[i] * 2.0


def main():
    n = 10

    # Simple test data
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)

    print("Input:")
    print(x)
    print("Expected output:")
    print(x * 2.0)

    # Metal computation
    x_buf = api.asarray(x)
    y_buf = api.empty((n,), api.DType.FLOAT32)

    api.launch_kernel("scale_by_two", grid=(n,), block=(1,), args=[x_buf, y_buf, n])

    # Get result
    result = api.to_numpy(y_buf)

    print("Metal result:")
    print(result)
    print(f"Max error: {np.max(np.abs(result - x * 2.0))}")
    print(f"Results match: {np.allclose(result, x * 2.0)}")


if __name__ == "__main__":
    main()
