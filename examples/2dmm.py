import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api


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
def matmul_2d(A, B, C, M, N, K):
    col = metal.thread_id_x()
    row = metal.thread_id_y()

    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc = acc + A[row * K + k] * B[k * N + col]
        C[row * N + col] = acc


def main():
    M, K, N = 4, 4, 4

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

    A_buf = api.asarray(A.flatten())
    B_buf = api.asarray(B.flatten())
    C_buf = api.empty((M * N,), api.DType.FLOAT32)

    print("\nRunning 2D Metal matmul")
    api.launch_kernel(
        "matmul_2d", grid=(N, M), block=(1, 1), args=[A_buf, B_buf, C_buf, M, N, K]
    )

    C_metal = api.to_numpy(C_buf).reshape(M, N)

    print("Metal result:")
    print(C_metal)
    print(f"Max error: {np.max(np.abs(C_metal - (A @ B)))}")
    print(f"Results match: {np.allclose(C_metal, A @ B, rtol=1e-4)}")


if __name__ == "__main__":
    main()
