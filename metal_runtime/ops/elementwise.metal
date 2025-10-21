#include <metal_stdlib>
using namespace metal;

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
kernel void tanh_op(
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
kernel void exp_op(
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
kernel void log_op(
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
kernel void sqrt_op(
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
kernel void abs_op(
    device const T* a [[buffer(0)]],
    device T* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < n) {
        b[tid] = metal::abs(a[tid]);
    }
}

#define INSTANTIATE_BINARY_OP(op_name, type_name, type) \
  template [[host_name(#op_name "_" #type_name)]] [[kernel]] void op_name<type>( \
    device const type*, device const type*, device type*, constant uint&, uint);

#define INSTANTIATE_UNARY_OP(op_name, type_name, type) \
  template [[host_name(#op_name "_" #type_name)]] [[kernel]] void op_name<type>( \
    device const type*, device type*, constant uint&, uint);

INSTANTIATE_BINARY_OP(add, float, float)
INSTANTIATE_BINARY_OP(add, half, half)
INSTANTIATE_BINARY_OP(add, int, int)
INSTANTIATE_BINARY_OP(add, uint, uint)
INSTANTIATE_BINARY_OP(add, short, short)
INSTANTIATE_BINARY_OP(add, ushort, ushort)
INSTANTIATE_BINARY_OP(add, char, char)
INSTANTIATE_BINARY_OP(add, uchar, uchar)

INSTANTIATE_BINARY_OP(sub, float, float)
INSTANTIATE_BINARY_OP(sub, half, half)
INSTANTIATE_BINARY_OP(sub, int, int)
INSTANTIATE_BINARY_OP(sub, uint, uint)
INSTANTIATE_BINARY_OP(sub, short, short)
INSTANTIATE_BINARY_OP(sub, ushort, ushort)
INSTANTIATE_BINARY_OP(sub, char, char)
INSTANTIATE_BINARY_OP(sub, uchar, uchar)

INSTANTIATE_BINARY_OP(mul, float, float)
INSTANTIATE_BINARY_OP(mul, half, half)
INSTANTIATE_BINARY_OP(mul, int, int)
INSTANTIATE_BINARY_OP(mul, uint, uint)
INSTANTIATE_BINARY_OP(mul, short, short)
INSTANTIATE_BINARY_OP(mul, ushort, ushort)
INSTANTIATE_BINARY_OP(mul, char, char)
INSTANTIATE_BINARY_OP(mul, uchar, uchar)

INSTANTIATE_BINARY_OP(div, float, float)
INSTANTIATE_BINARY_OP(div, half, half)
INSTANTIATE_BINARY_OP(div, int, int)
INSTANTIATE_BINARY_OP(div, uint, uint)
INSTANTIATE_BINARY_OP(div, short, short)
INSTANTIATE_BINARY_OP(div, ushort, ushort)
INSTANTIATE_BINARY_OP(div, char, char)
INSTANTIATE_BINARY_OP(div, uchar, uchar)

INSTANTIATE_UNARY_OP(relu, float, float)
INSTANTIATE_UNARY_OP(relu, half, half)
INSTANTIATE_UNARY_OP(relu, int, int)
INSTANTIATE_UNARY_OP(relu, uint, uint)
INSTANTIATE_UNARY_OP(relu, short, short)
INSTANTIATE_UNARY_OP(relu, ushort, ushort)
INSTANTIATE_UNARY_OP(relu, char, char)
INSTANTIATE_UNARY_OP(relu, uchar, uchar)

INSTANTIATE_UNARY_OP(sigmoid, float, float)
INSTANTIATE_UNARY_OP(sigmoid, half, half)

INSTANTIATE_UNARY_OP(tanh_op, float, float)
INSTANTIATE_UNARY_OP(tanh_op, half, half)

INSTANTIATE_UNARY_OP(exp_op, float, float)
INSTANTIATE_UNARY_OP(exp_op, half, half)

INSTANTIATE_UNARY_OP(log_op, float, float)
INSTANTIATE_UNARY_OP(log_op, half, half)

INSTANTIATE_UNARY_OP(sqrt_op, float, float)
INSTANTIATE_UNARY_OP(sqrt_op, half, half)

INSTANTIATE_UNARY_OP(neg, float, float)
INSTANTIATE_UNARY_OP(neg, half, half)
INSTANTIATE_UNARY_OP(neg, int, int)
INSTANTIATE_UNARY_OP(neg, uint, uint)
INSTANTIATE_UNARY_OP(neg, short, short)
INSTANTIATE_UNARY_OP(neg, ushort, ushort)
INSTANTIATE_UNARY_OP(neg, char, char)
INSTANTIATE_UNARY_OP(neg, uchar, uchar)

INSTANTIATE_UNARY_OP(abs_op, float, float)
INSTANTIATE_UNARY_OP(abs_op, half, half)
INSTANTIATE_UNARY_OP(abs_op, int, int)
INSTANTIATE_UNARY_OP(abs_op, uint, uint)
INSTANTIATE_UNARY_OP(abs_op, short, short)
INSTANTIATE_UNARY_OP(abs_op, ushort, ushort)
INSTANTIATE_UNARY_OP(abs_op, char, char)
INSTANTIATE_UNARY_OP(abs_op, uchar, uchar)
