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

template <typename T>
kernel void relu(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = max(a[tid], static_cast<T>(0));
    }
}

template <typename T>
kernel void sigmoid(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-a[tid]));
    }
}

template <typename T>
kernel void tanh(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = metal::tanh(a[tid]);
    }
}

template <typename T>
kernel void exp(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = metal::exp(a[tid]);
    }
}

template <typename T>
kernel void log(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = metal::log(a[tid]);
    }
}

template <typename T>
kernel void sqrt(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = metal::sqrt(a[tid]);
    }
}

template <typename T>
kernel void neg(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = -a[tid];
    }
}

template <typename T>
kernel void abs(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = metal::abs(a[tid]);
    }
}

#define INSTANTIATE_BINARY_OP(op_name) \
  template [[kernel]] void op_name<float>(device const float*, device const float*, device float*, constant uint&, uint); \
  template [[kernel]] void op_name<half>(device const half*, device const half*, device half*, constant uint&, uint); \
  template [[kernel]] void op_name<int>(device const int*, device const int*, device int*, constant uint&, uint); \
  template [[kernel]] void op_name<uint>(device const uint*, device const uint*, device uint*, constant uint&, uint); \
  template [[kernel]] void op_name<short>(device const short*, device const short*, device short*, constant uint&, uint); \
  template [[kernel]] void op_name<ushort>(device const ushort*, device const ushort*, device ushort*, constant uint&, uint); \
  template [[kernel]] void op_name<char>(device const char*, device const char*, device char*, constant uint&, uint); \
  template [[kernel]] void op_name<uchar>(device const uchar*, device const uchar*, device uchar*, constant uint&, uint); \


#define INSTANTIATE_UNARY_FLOAT_OP(op_name) \
    template [[kernel]] void op_name<float>(device const float*, device float*, constant uint&, uint); \
    template [[kernel]] void op_name<half>(device const half*, device half*, constant uint&, uint);

#define INSTANTIATE_UNARY_ALL_OP(op_name) \
    template [[kernel]] void op_name<float>(device const float*, device float*, constant uint&, uint); \
    template [[kernel]] void op_name<half>(device const half*, device half*, constant uint&, uint); \
    template [[kernel]] void op_name<int>(device const int*, device int*, constant uint&, uint); \
    template [[kernel]] void op_name<uint>(device const uint*, device uint*, constant uint&, uint); \
    template [[kernel]] void op_name<short>(device const short*, device short*, constant uint&, uint); \
    template [[kernel]] void op_name<ushort>(device const ushort*, device ushort*, constant uint&, uint); \
    template [[kernel]] void op_name<char>(device const char*, device char*, constant uint&, uint); \
    template [[kernel]] void op_name<uchar>(device const uchar*, device uchar*, constant uint&, uint);


INSTANTIATE_BINARY_OP(add);
INSTANTIATE_BINARY_OP(sub);
INSTANTIATE_BINARY_OP(mul);
INSTANTIATE_BINARY_OP(div);

INSTANTIATE_UNARY_ALL_OP(relu);
INSTANTIATE_UNARY_ALL_OP(neg);
INSTANTIATE_UNARY_ALL_OP(abs);

INSTANTIATE_UNARY_FLOAT_OP(sigmoid);
INSTANTIATE_UNARY_FLOAT_OP(tanh);
INSTANTIATE_UNARY_FLOAT_OP(exp);
INSTANTIATE_UNARY_FLOAT_OP(log);
INSTANTIATE_UNARY_FLOAT_OP(sqrt);
