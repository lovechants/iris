#include <metal_stdlib>
using namespace metal;

kernel void gather(
    device const float* input [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& axis [[buffer(3)]],
    constant uint& input_dim [[buffer(4)]],
    constant uint& output_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= output_dim) return;
    
    uint idx = indices[tid];
    if (idx >= input_dim) return;
    
    if (axis == 0) {
        output[tid] = input[idx];
    } else if (axis == 1) {
        uint row = tid / output_dim;
        uint col = tid % output_dim;
        output[row * output_dim + col] = input[row * input_dim + idx];
    }
}
