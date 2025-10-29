import numpy as np
from metal_runtime.decorator import kernel, metal
from metal_runtime import api


@kernel(
    param_types={
        "A": "device const float*",
        "B": "device const float*",
        "C": "device float*",
        "D": "uint",  # depth (z‑dimension)
        "H": "uint",  # height (y)
        "W": "uint",  # width  (x)
    }
)
def add_3d(A, B, C, D, H, W):
    """
    perform elementwise addition across a 3D volume (also testing to see if the docstring will pop)
    """
    x = metal.thread_id_x()
    y = metal.thread_id_y()
    z = metal.thread_id_z()

    if x < W and y < H and z < D:
        idx = (z * H * W) + (y * W) + x
        C[idx] = A[idx] + B[idx]


def main():
    D, H, W = 4, 3, 5
    shape = (D, H, W)

    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)

    expected = A + B

    print("A:")
    print(A)
    print("\nB:")
    print(B)
    print("\nExpected result (NumPy):")
    print(expected)

    A_buf = api.asarray(A.flatten())
    B_buf = api.asarray(B.flatten())
    C_buf = api.empty_like(A_buf)

    print("\nRunning 3D Metal add kernel")
    api.launch_kernel(
        "add_3d",
        grid=(W, H, D),     # (x, y, z)
        block=(1, 1, 1),
        args=[A_buf, B_buf, C_buf, D, H, W],
    )

    C_metal = api.to_numpy(C_buf).reshape(shape)

    print("\nMetal result:")
    print(C_metal)

    max_err = np.max(np.abs(C_metal - expected))
    print(f"\nMax error: {max_err}")
    print(f"Results match: {np.allclose(C_metal, expected, rtol=1e-4)}")


if __name__ == "__main__":
    main()
