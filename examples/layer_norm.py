import numpy as np
import time
from metal_runtime import api, ops
from metal_runtime.logging import get_logger, configure_logging, LogLevel

EPS = 1e-5

def layer_norm_numpy(x, gamma, beta):
    """The ground truth implementation in NumPy."""
    mean = np.mean(x)
    var = np.var(x)
    x_normalized = (x - mean) / np.sqrt(var + EPS)
    return gamma * x_normalized + beta

def run_layernorm_chain(x_buf, gamma_buf, beta_buf, mean_val, std_dev_val):
    """
    This function contains the chain of element-wise ops that we want to fuse.
    It takes pre-computed scalars.
    """
    x_minus_mean = ops.sub_scalar(x_buf, mean_val)
    
    x_normalized = ops.div_scalar(x_minus_mean, std_dev_val)
    
    # gamma * x_normalized + beta
    scaled = ops.mul(gamma_buf, x_normalized)
    shifted = ops.add(beta_buf, scaled)
    return shifted

def main():
    configure_logging(level=LogLevel.INFO)
    logger = get_logger()

    # Realistic dimensions for a language model
    batch_size = 32
    seq_len = 128
    embedding_dim = 768
    n_elements = batch_size * seq_len * embedding_dim

    logger.info(f"Running LayerNorm Demo on {n_elements:,} elements")

    x_np = np.random.randn(n_elements).astype(np.float32)
    gamma_np = np.ones(n_elements).astype(np.float32)
    beta_np = np.zeros(n_elements).astype(np.float32)

    mean_val = np.mean(x_np)
    var_val = np.var(x_np)
    std_dev_val = np.sqrt(var_val + EPS)
    
    expected_result = layer_norm_numpy(x_np, gamma_np, beta_np)
    
    x_buf = api.asarray(x_np)
    gamma_buf = api.asarray(gamma_np)
    beta_buf = api.asarray(beta_np)

    logger.info("Benchmarking naive, multi-kernel execution")
    # Warm-up run
    run_layernorm_chain(x_buf, gamma_buf, beta_buf, mean_val, std_dev_val)
    api.synchronize()

    t0 = time.perf_counter()
    result_unfused_buf = run_layernorm_chain(x_buf, gamma_buf, beta_buf, mean_val, std_dev_val)
    api.synchronize()
    t1 = time.perf_counter()
    unfused_time_ms = (t1 - t0) * 1000
    
    logger.info(f"Unfused execution took: {unfused_time_ms:.2f} ms")
    assert np.allclose(api.to_numpy(result_unfused_buf), expected_result, atol=1e-4)
    logger.info("Unfused result is CORRECT.")

    logger.info("Benchmarking single-kernel fused execution...")
    # Warm-up run
    with api.fused():
        run_layernorm_chain(x_buf, gamma_buf, beta_buf, mean_val, std_dev_val)
    api.synchronize()
    
    t0 = time.perf_counter()
    with api.fused():
        result_fused_buf = run_layernorm_chain(x_buf, gamma_buf, beta_buf, mean_val, std_dev_val)
    api.synchronize()
    t1 = time.perf_counter()
    fused_time_ms = (t1 - t0) * 1000

    logger.info(f"Fused execution took:   {fused_time_ms:.2f} ms")
    assert np.allclose(api.to_numpy(result_fused_buf), expected_result, atol=1e-4)
    logger.info("Fused result is CORRECT.")

    speedup = unfused_time_ms / fused_time_ms
    logger.info(f"Achieved a {speedup:.2f}x speedup with fusion")
    logger.info("(Note: Speedup is for the element-wise portion only)")

if __name__ == "__main__":
    main()
