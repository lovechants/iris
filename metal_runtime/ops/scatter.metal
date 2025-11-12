#include <metal_stdlib>
using namespace metal;

kernel void scatter(
    device float* output [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    constant uint& axis [[buffer(3)]],
    constant uint& input_dim [[buffer(4)]],
    constant uint& values_count [[buffer(5)]],  
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= values_count) return;  
    
    uint idx = indices[tid];
    if (idx >= input_dim) return;
    
    if (axis == 0) {
        output[idx] = values[tid];
    } else if (axis == 1) {
        uint row = tid / input_dim;
        uint col = tid % input_dim;
        output[row * input_dim + idx] = values[tid];
    }
}
