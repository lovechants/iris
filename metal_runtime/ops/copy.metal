#include <metal_stdlib>
using namespace metal;

kernel void copy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;
    output[tid] = input[tid];
}
