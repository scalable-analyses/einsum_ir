import etops

top_config = etops.TensorOperationConfig(
    backend    =   "tpp",
    data_type  =   etops.float32,
    prim_first =   etops.prim.zero,
    prim_main  =   etops.prim.gemm,
    prim_last  =   etops.prim.none,
    dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
    exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
    dim_sizes  =   (512,             512,             512            ),
    strides    = (((1,               0,               512            ),   # in0
                   (0,               512,             1              ),   # in1
                   (1,               512,             0              )),) # out
)

# Create the TensorOperation instance
top = etops.TensorOperation(top_config)

# Create the Model for performance prediction (independent of TensorOperation)
model = etops.Model(micro_arch=etops.arch.m4)

# Create input and output arrays
import numpy as np
A = np.random.randn(512,512).astype(np.float32)
B = np.random.randn(512,512).astype(np.float32)
C = np.random.randn(512,512).astype(np.float32)

# Execute the operation
top.execute(A, B, C)

C_np = np.einsum("km,nk->nm", A, B)

# Compute absolute and relative errors
error_abs = np.max( np.abs(C - C_np) )
error_rel = np.max( np.abs(C - C_np) / (np.abs(C_np) + 1e-8) )
print("Column-major GEMM operation:")
print(f"  Max absolute error: {error_abs:.6e}")
print(f"  Max relative error: {error_rel:.6e}")

# benchmarking 
import time

num_iters = 10000

print("running benchmark...")
start_time = time.time()
for _ in range(num_iters):
    top.execute(A, B, C)
end_time = time.time()
print("benchmark completed.")

total_time = end_time - start_time

avg_time_per_iter = total_time / num_iters
estimated_time = model.predict(top_config)

flops = 512 * 512 * (512 + 511)
gflops = flops / (avg_time_per_iter * 1e9)
es_gflops = model.predict_gflops(top_config)

print(f"Total time for {num_iters} iterations: {total_time:.8f} s")
print("****************************")
print(f"Execution time: {avg_time_per_iter:.8f} s")
print(f"Estimated: {estimated_time:.8f} s")
print("****************************")
print(f"Achieved GFLOPS: {gflops:.2f} GFLOPS")
print(f"Estimated GFLOPS: {es_gflops:.2f} GFLOPS")