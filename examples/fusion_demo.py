import numpy as np
import time
from metal_runtime import api, ops

def main():
    n = 1000000
    print(f"Array size: {n:,} elements\n")
    
    a = api.asarray(np.random.randn(n).astype(np.float32))
    b = api.asarray(np.random.randn(n).astype(np.float32))
    
    print("Unfused execution:")
    unfused_times = []
    for _ in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        start = time.perf_counter()
        c = ops.add(a_tmp, b_tmp)
        d = ops.mul_scalar(c, 2.0)
        e = ops.relu(d)
        api.synchronize()
        unfused_times.append(time.perf_counter() - start)
    
    unfused_median = np.median(unfused_times[2:])
    print(f"  Time: {unfused_median*1000:.2f} ms")
    print(f"  Memory: 4 loads + 3 stores\n")
    
    print("Fused execution:")
    fused_times = []
    for _ in range(10):
        a_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        b_tmp = api.asarray(np.random.randn(n).astype(np.float32))
        
        start = time.perf_counter()
        with api.fused():
            c = ops.add(a_tmp, b_tmp)
            d = ops.mul_scalar(c, 2.0)
            e = ops.relu(d)
        api.synchronize()
        fused_times.append(time.perf_counter() - start)
    
    fused_median = np.median(fused_times[2:])
    speedup = unfused_median / fused_median
    
    print(f"  Time: {fused_median*1000:.2f} ms")
    print(f"  Memory: 2 loads + 1 store\n")
    
    print(f"Speedup: {speedup:.2f}x")
    print(f"Memory traffic reduced by {(1 - 3/7)*100:.0f}%")

if __name__ == "__main__":
    main()
