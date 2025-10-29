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
def scale_and_add(x, y, n):
    i = metal.thread_id()
    if i < n:
        # Two operations - should still work
        y[i] = x[i] * 2.0 + 1.0


@kernel(
    param_types={
        "x": "device const float*",
        "y": "device float*",
        "n": "uint",
    }
)
def simple_accumulate(x, y, n):
    i = metal.thread_id()
    if i < n:
        # This might trigger the variable scoping issue
        acc = 0.0
        for j in range(3):
            acc = acc + x[i * 3 + j]
        y[i] = acc


def main():
    n = 5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    x_buf = api.asarray(x)
    y_buf = api.empty((n,), api.DType.FLOAT32)

    api.launch_kernel("scale_and_add", grid=(n,), block=(1,), args=[x_buf, y_buf, n])
    result = api.to_numpy(y_buf)

    print("Input:", x)
    print("Expected:", x * 2.0 + 1.0)
    print("Metal:", result)
    print("Match:", np.allclose(result, x * 2.0 + 1.0))

    print("\nTesting simple_accumulate")
    # Test with 3 values per thread
    x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

    x2_buf = api.asarray(x2)
    y2_buf = api.empty((3,), api.DType.FLOAT32)  # 3 threads, each sums 3 values

    api.launch_kernel(
        "simple_accumulate", grid=(3,), block=(1,), args=[x2_buf, y2_buf, 3]
    )
    result2 = api.to_numpy(y2_buf)

    print("Input:", x2)
    print(
        "Expected sums:",
        [x2[0] + x2[1] + x2[2], x2[3] + x2[4] + x2[5], x2[6] + x2[7] + x2[8]],
    )
    print("Metal:", result2)


if __name__ == "__main__":
    main()
