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

TUI - Skeleton  

---

## Roadmap

Right now we just keep a JIT cache instead of other cache methods.
The plan is to create an IR for tiling + fusion 
- for example add -> mul -> relu 
Then having the compiler know when to tile or allowing for the decorator to look like 

```python 
@kernel(tile=N) # 16 etc 
    kernel impl 
```

Right now it's missing 
1. Reduction ops 
2. broadcasting mismatches
3. tensor ops (reshape, transpose, view)
4. no native matmul optimization (shared mem, explict tiling)
5. no autodiff or grad comp 
6. no multi-gpu support (not planned)
7. disk based cache

---

## Considerations (performance)

- discussed already, but matmul is naive and not properly implemented
- no autotuner for auto block selection 

#### Codegen limitations

- type inference just default forces floats (ambigous cases)
- threadgroup support not impl 
- no dead code elemimation of optimization passes
- compare ops are a brittle right now 

---

## Examples

There's a few selection of examples 

> Now there are more than a few examples some are mostly focused on proper development but here is the standout examples 
```
/examples
├── 2dmm.py                   # 2D matmul example
├── 3dadd.py                  # 3D indexing demo
├── elementwise.py            # Pre-built ops
├── fusion_demo.py            # The simple fusion `with api.fused()`
├── layer_norm.py             # Proper end to end example  
├── mat_mul.py                # Snall verifiable matmul
├── tui_demo.py               # Shows off the Rust TUI capabilities (early demo)
├── vector_add.py             # The "Hello, World!" of @kernel
└── persistent_buffers.py     # See persistent buffers working / mem management 
```

```python 
python -m examples.<example_name>

```

> Don't include .py when running with the -m flag


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

## TUI 

You can also run a small tui visualizer -- for now in a separate terminal / pane 

#### Using the TUI 

Keyboard Controls

q - quit 

r - reset the log file 

s - cycle through sort modes 

f - cycle through phase filters (none -> run -> compile -> hit -> none)

View Modes 

1 - Timeline view 

2 - Stats view 

3 - Kernel details view

4 - Memory view 

Navigation 
Up/Down (Arrows) - Move selection of kernels 

```bash
cd iris/tui 
```
& 

```Rust
cargo build 
cargo run
```

