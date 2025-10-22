import numpy as np
from metal_runtime import api, ops


def main():
    n = 512
    
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    
    a_buf = api.asarray(a)
    b_buf = api.asarray(b)
    
    
    add_result = api.to_numpy(ops.add(a_buf, b_buf))
    add_expected = a + b
    print(f"Add - Max error: {np.max(np.abs(add_result - add_expected)):.6e}")
    
    mul_result = api.to_numpy(ops.mul(a_buf, b_buf))
    mul_expected = a * b
    print(f"Mul - Max error: {np.max(np.abs(mul_result - mul_expected)):.6e}")
    
    relu_result = api.to_numpy(ops.relu(a_buf))
    relu_expected = np.maximum(a, 0)
    print(f"ReLU - Max error: {np.max(np.abs(relu_result - relu_expected)):.6e}")
    
    sigmoid_result = api.to_numpy(ops.sigmoid(a_buf))
    sigmoid_expected = 1 / (1 + np.exp(-a))
    print(f"Sigmoid - Max error: {np.max(np.abs(sigmoid_result - sigmoid_expected)):.6e}")
    
    exp_result = api.to_numpy(ops.exp(a_buf * 0.5))
    exp_expected = np.exp(a * 0.5)
    print(f"Exp - Max error: {np.max(np.abs(exp_result - exp_expected)):.6e}")
    

if __name__ == "__main__":
    main()
