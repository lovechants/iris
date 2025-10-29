import numpy as np
from metal_runtime import api, ops
from metal_runtime.dtype import DType
from metal_runtime.api import persistent_buffers
from metal_runtime.api import _get_runtime
N = 10**6
a_np = np.random.randn(N).astype(np.float32)
b_np = np.random.randn(N).astype(np.float32)

with persistent_buffers():

    a_buf = api.asarray(a_np, persistent=True)
    b_buf = api.asarray(b_np, persistent=True)
    out_buf = api.empty_like(a_buf, persistent=True)

    for _ in range(100):
        ops.add(a_buf, b_buf, out_buf)

    result = api.to_numpy(out_buf)

    
    print("Buffer pool usage:")
    for k, v in _get_runtime()._buffer_pool.items():
        print(f"  size={k}B -> {len(v)} free buffers")

print(result[:5])

