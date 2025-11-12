#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    threadgroup float partial[256];
    
    partial[tid] = (tid < n) ? input[tid] : 0.0f;
    
    for (uint stride = 256/2; stride > 0; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
    }
    
    if (tid == 0) {
        output[0] = partial[0];
    }
}

kernel void reduce_sum_axis(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& dim_x [[buffer(2)]],
    constant uint& dim_y [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint x = tid.x;
    uint y = tid.y;
    
    if (x >= dim_x || y >= dim_y) return;
    
    if (axis == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < dim_x; i++) {
            sum += input[i * dim_y + y];
        }
        output[y] = sum;
    } else if (axis == 1) {
        float sum = 0.0f;
        for (uint j = 0; j < dim_y; j++) {
            sum += input[x * dim_y + j];
        }
        output[x] = sum;
    }
}
