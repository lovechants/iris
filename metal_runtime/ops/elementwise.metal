#include <metal_stdlib>
using namespace metal;

// Just binary ops 

template <typename T>
kernel void add(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* c [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}


template <typename T>
kernel void sub(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* c [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid < n) {
    c[tid] = a[tid] - b[tid];
  }
}


template <typename T>
kernel void mul(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* c [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid < n) {
    c[tid] = a[tid] * b[tid];
  }
}

template <typename T>
kernel void div(
  device const T* a [[buffer(0)]],
  device const T* b [[buffer(1)]],
  device T* c [[buffer(2)]],
  constant uint& n [[buffer(3)]],
  uint tid [[thread_position_in_grid]]
) {
  if (tid < n) {
    c[tid] = a[tid] / b[tid];
  }
}


#define INSTANTIATE_BINARY_OP(op_name) \
  template [[kernel]] void op_name<float>(device const float*, device const float*, device float*, constant uint&, uint); \
  template [[kernel]] void op_name<half>(device const half*, device const half*, device half*, constant uint&, uint); \
  // finish this 
  template [[kernel]] void op_name<float>(device const float*, device const float*, device float*, constant uint&, uint); \
  template [[kernel]] void op_name<float>(device const float*, device const float*, device float*, constant uint&, uint); \
  template [[kernel]] void op_name<float>(device const float*, device const float*, device float*, constant uint&, uint); \
  template [[kernel]] void op_name<float>(device const float*, device const float*, device float*, constant uint&, uint); \





