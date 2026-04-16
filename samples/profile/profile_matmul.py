import etops

top_config = etops.TensorOperationConfig(
    backend    =   "tpp",
    data_type  =   etops.float32,
    prim_first =   etops.prim.zero,
    prim_main  =   etops.prim.gemm,
    prim_last  =   etops.prim.none,
    dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
    exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
    dim_sizes  =   (1024,            1024,            1024           ),
    strides    = (((1,               0,               1024           ),   # in0
                   (0,               1024,            1              ),   # in1
                   (1,               1024,            0              )),) # out
)

# Create the TensorOperation instance
top = etops.TensorOperation(top_config)

top.configure_papi("default")

# Create input and output arrays
import numpy as np
A = np.random.randn(1024,1024).astype(np.float32)
B = np.random.randn(1024,1024).astype(np.float32)
C = np.random.randn(1024, 1024).astype(np.float32)

# Execute the operation
top.execute(A, B, C)

C_np = np.einsum("km,nk->nm", A, B)

# Compute absolute and relative errors
error_abs = np.max( np.abs(C - C_np) )
error_rel = np.max( np.abs(C - C_np) / (np.abs(C_np) + 1e-8) )
print("Column-major GEMM operation:")
print(f"  Max absolute error: {error_abs:.6e}")
print(f"  Max relative error: {error_rel:.6e}")