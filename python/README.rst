etops
=====

The `etops` package provides a Python interface for einsum tree operations. It enables users to define, configure, optimize, and execute complex tensor contractions and elementwise operations. The package is built on top of the einsum_ir C++ backend and exposes advanced features such as dimension fusion, dimension splitting, and backend-specific optimizations.

Main Features
-------------
- Abstractions for tensor operations and configuration
- Support for multiple data types (float32, float64)
- Primitive operations: zero, copy, relu, gemm, brgemm, etc.
- Dimension execution strategies: primitive, sequential, shared, space-filling curve (SFC)
- Dimension and stride configuration for advanced memory layouts
- Interface for built-in contraction optimizer
- Pythonic API with dataclass-based configuration

Installation
------------
Install the package using pip:

.. code-block:: bash

    pip install etops

Quick Example
-------------
Below is a minimal example showing how to configure and execute tensor operations:

.. code-block:: python

    import etops

    # -----------------------------------------
    # First example:
    #   Column-major GEMM operation
    #   Compares the result with NumPy's einsum
    # -----------------------------------------
    # Define a column-major GEMM configuration
    top_config = etops.TensorOperationConfig(
        data_type  = etops.float32,
        prim_first = etops.prim.zero,
        prim_main  = etops.prim.gemm,
        prim_last  = etops.prim.none,
        dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  = (64,              32,              128            ),
        strides_in0= (1,               0,               64             ),
        strides_in1= (0,               128,             1              ),
        strides_out= (1,               64,              0              )
    )

    # Create the TensorOperation instance
    top = etops.TensorOperation(top_config)

    # Create input and output arrays
    import numpy as np
    A = np.random.randn(128,64).astype(np.float32)
    B = np.random.randn(32,128).astype(np.float32)
    C = np.random.randn(32, 64).astype(np.float32)

    # Execute the operation
    top.execute(A, B, C)

    C_np = np.einsum("km,nk->nm", A, B)

    # Compute absolute and relative errors
    error_abs = np.max( np.abs(C - C_np) )
    error_rel = np.max( np.abs(C - C_np) / (np.abs(C_np) + 1e-8) )
    print("Column-major GEMM operation:")
    print(f"  Max absolute error: {error_abs:.6e}")
    print(f"  Max relative error: {error_rel:.6e}")

    # -----------------------------------------
    # Second example:
    #   Batched GEMM operation
    #   Compares the result with torch's einsum
    # -----------------------------------------
    # Define a batched GEMM configuration
    batched_config = etops.TensorOperationConfig(
        data_type  = etops.float32,
        prim_first = etops.prim.zero,
        prim_main  = etops.prim.gemm,
        prim_last  = etops.prim.none,
        dim_types  = (etops.dim.c,       etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types = (etops.exec.shared, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  = (48,                64,              32,              128            ),
        strides_in0= (128*64,            1,               0,               64             ),
        strides_in1= (32*128,            0,               128,             1              ),
        strides_out= (32*64,             1,               64,              0              )
    )
    # Create the batched TensorOperation instance
    top = etops.TensorOperation(batched_config)

    import torch
    # Create input and output arrays
    A = torch.randn(48, 128, 64, dtype=torch.float32)
    B = torch.randn(48, 32, 128, dtype=torch.float32)
    C = torch.randn(48, 32, 64,  dtype=torch.float32)

    # Execute the operation
    top.execute(A, B, C)

    C_torch = torch.einsum("bkm,bnk->bnm", A, B)

    # Compute absolute and relative errors
    error_abs = torch.max(torch.abs(C - C_torch))
    error_rel = torch.max(torch.abs(C - C_torch) / (torch.abs(C_torch) + 1e-8))

    print("Batched GEMM operation:")
    print(f"  Max absolute error: {error_abs:.6e}")
    print(f"  Max relative error: {error_rel:.6e}")

    # -----------------------------------------------
    # Third example:
    #   Batch-reduce GEMM operation with optimization
    #   Compares the result with torch's einsum
    # -----------------------------------------------
    # Define a batch-reduce GEMM configuration
    batched_config = etops.TensorOperationConfig(
        data_type  = etops.float32,
        prim_first = etops.prim.zero,
        prim_main  = etops.prim.gemm,
        prim_last  = etops.prim.none,
        dim_types  = (etops.dim.k,    etops.dim.m,    etops.dim.n,    etops.dim.k   ),
        exec_types = (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq),
        dim_sizes  = (48,             64,             32,             128           ),
        strides_in0= (128*64,         1,              0,              64            ),
        strides_in1= (32*128,         0,              128,            1             ),
        strides_out= (0,              1,              64,             0             )
    )

    # Optimize the configuration
    optimized_config = etops.optimize( batched_config,
                                       target_m=16,
                                       target_n=12,
                                       target_k=64,
                                       num_threads=1,
                                       generate_sfc=True,
                                       br_gemm_support=True,
                                       packed_gemm_support=True )

    # Create the optimized TensorOperation instance
    top = etops.TensorOperation(optimized_config)

    import torch
    # Create input and output arrays
    A = torch.randn(48, 128, 64, dtype=torch.float32)
    B = torch.randn(48, 32, 128, dtype=torch.float32)
    C = torch.randn(    32, 64,  dtype=torch.float32)

    # Execute the operation
    top.execute(A, B, C)

    C_torch = torch.einsum("bkm,bnk->nm", A, B)

    # Compute absolute and relative errors
    error_abs = torch.max(torch.abs(C - C_torch))
    error_rel = torch.max(torch.abs(C - C_torch) / (torch.abs(C_torch) + 1e-8))
    print("Batch-reduce GEMM operation:")
    print(f"  Max absolute error: {error_abs:.6e}")
    print(f"  Max relative error: {error_rel:.6e}")

See the source code and inline documentation for more advanced usage.
