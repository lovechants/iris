import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api
import time


@kernel(
    param_types={
        "A": "device const float*",
        "B": "device const float*",
        "C": "device float*",
        "M": "uint",
        "N": "uint",
        "K": "uint",
    }
)
def matmul_simple(A, B, C, M, N, K):
    i = metal.thread_id()
    if i < M:
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc = acc + A[i * K + k] * B[k * N + j]
            C[i * N + j] = acc


def main():
    # Small test first
    M, K, N = 4, 4, 4

    # Simple test matrices
    A = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float32,
    )

    B = np.array(
        [
            [16.0, 15.0, 14.0, 13.0],
            [12.0, 11.0, 10.0, 9.0],
            [8.0, 7.0, 6.0, 5.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    print("A:")
    print(A)
    print("B:")
    print(B)
    print("Expected A @ B:")
    print(A @ B)

    # Metal computation
    A_buf = api.asarray(A.flatten())
    B_buf = api.asarray(B.flatten())
    C_buf = api.empty((M * N,), api.DType.FLOAT32)

    print("\nRunning Metal matmul")
    api.launch_kernel(
        "matmul_simple", grid=(M,), block=(1,), args=[A_buf, B_buf, C_buf, M, N, K]
    )

    # Get result
    C_metal = api.to_numpy(C_buf).reshape(M, N)

    print("Metal result:")
    print(C_metal)
    print(f"Max error: {np.max(np.abs(C_metal - (A @ B)))}")
    print(f"Results match: {np.allclose(C_metal, A @ B, rtol=1e-4)}")


if __name__ == "__main__":
    main()
