import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.decorator import kernel, metal


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
            acc += A[row * K + k] * B[k * N + col]
        C[row * N + col] = acc


def main():
    M = N = K = 4096
    TILE = 16

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    A_buf = api.asarray(A.flatten(), persistent=True)
    B_buf = api.asarray(B.flatten(), persistent=True)
    C_buf = api.empty((M * N,), api.DType.FLOAT32, persistent=True)

    api.launch_kernel("matmul_2d", grid=(N, M), block=(TILE, TILE),
                      args=[A_buf, B_buf, C_buf, M, N, K])
    api.synchronize()

    t0 = time.perf_counter()
    for i in range(5):
        print(f"Iteration {i}")
        api.launch_kernel("matmul_2d", grid=(N, M), block=(TILE, TILE),
                          args=[A_buf, B_buf, C_buf, M, N, K])
        ops.add(C_buf, A_buf, C_buf)
        ops.relu(C_buf, C_buf)
        ops.mul_scalar(C_buf, 1.0001)
        api.synchronize()
        time.sleep(0.25)
    t1 = time.perf_counter()

    checksum = float(np.sum(api.to_numpy(C_buf)))
    print(f"Checksum: {checksum:.6f}")
    print(f"\nTotal elapsed {(t1-t0):.3f}s")

if __name__ == "__main__":
    main()
