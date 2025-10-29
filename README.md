# iris 

A minimal Metal GPU runtime and compiler experiment.

---

## Overview

Iris is a Metal runtime designed for exploring the
intersection of compiler code generation, GPU kernel scheduling, and dynamic
execution. The current implementation provides a fully functional Python runtime
that transparently builds Metal Shading Language (MSL) kernels at runtime, 
launches them through a persistent buffer manager, and supports automatic AST‑to‑MSL 
code generation for elementwise operators.

This is **not** a framework clone it's just for fun and potentially hooking into julia (framework).

---

## Capabilities

- **Dynamic Kernel Codegen**
  - Translates Python AST → valid Metal Shading Language kernels
  - Generates and registers kernels on the fly (similar to Triton / XLA lower passes)

- **End‑to‑End Metal Runtime**
  - Runtime API with automatic upload, execution, and synchronization
  - Unified buffer abstraction (`MetalBuffer`) mapping directly to Metal `MTLBuffer`
  - Built‑in operator suite (`add`, `mul`, `relu`, `tanh`, `log`, etc.)

- **Persistent GPU Buffer Residency**
  - Optional memory pool and stream‑based reuse, minimizing host/device transfers
  - Reduces launch and transfer latency by an order of magnitude on typical workloads

- **Compiler Layer**
  - Caching, incremental library compilation, and Metal IR validation
  - Modular code generator allowing AST fusion and operator composition


---

## Development Status

Core runtime - Done 
Codegen - Done 
Kernel Lib - Elementwise done (2D + 3D ops supported)
Presistent Mem Pool - Done
Profiler - Planned 
TUI - Planned 

---

## Roadmap


1. **Visual Runtime Inspector**
   - Lightweight text‑UI (planned Rust tool) for introspection and visualization

2. **Advanced Features**
   - Graph‑level fusion IR
   - Vectorized types (`float2`, `float4`)
   - Shared memory primitives

---

## Example

```python
import numpy as np
from metal_runtime import api, ops

N = 1_000_000
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)

a_buf = api.asarray(a, persistent=True)
b_buf = api.asarray(b, persistent=True)
out_buf = api.empty_like(a_buf, persistent=True)

ops.add(a_buf, b_buf, out_buf)
result = api.to_numpy(out_buf)
print(result[:5])
```

or

```python 
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
    x = metal.thread_id_x()
    y = metal.thread_id_y()
    z = metal.thread_id_z()

    if x < W and y < H and z < D:
        idx = (z * H * W) + (y * W) + x
        C[idx] = A[idx] + B[idx]

```
