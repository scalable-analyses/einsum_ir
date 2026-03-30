etops
=====

The `etops` package provides a Python interface for the Tiled Execution IR (TEIR). It enables users to define, configure, optimize, and execute complex tensor contractions and elementwise operations. The package supports multiple backends: TPP (CPU via libxsmm) and tileir (GPU via direct TileIR construction, optional).

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

For GPU support with the tileir backend (optional):

.. code-block:: bash

    pip install etops[tileir-cuda13]

Unary Examples
--------------
Below are some examples showing how to configure and execute unary tensor operations:

.. code-block:: python

    import etops

    # ---------------------------------------
    # First example:
    #   Matrix transpose using copy primitive
    #   Compares the result with NumPy
    # ---------------------------------------
    # Define a transpose configuration
    top_config = etops.TensorOperationConfig(
        backend    =   "tpp",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,     etops.dim.c    ),
        exec_types =   (etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (3,               4              ),
        strides    = (((4,               1               ),   # in
                       (1,               3               )),) # out
    )

    # Compile the tensor operation
    top = etops.compile(top_config)

    # Create input and output arrays
    import numpy as np
    A = np.random.randn(3,4).astype(np.float32)
    B = np.random.randn(4,3).astype(np.float32)

    top.execute(A, None, B)

    B_np = np.einsum("ij->ji", A)

    # Check correctness
    error_abs = np.max( np.abs(B - B_np) )
    print("Matrix Transpose using copy primitive:")
    print(f"  Max absolute error: {error_abs:.6e}")

    # -------------------------------------------------
    # Second example:
    #   Permutation of a 4D tensor using copy primitive
    #   Compares the result with NumPy
    # -------------------------------------------------
    # Define a permutation configuration
    perm_config = etops.TensorOperationConfig(
        backend     =   "tpp",
        data_type   =   etops.float32,
        prim_first  =   etops.prim.none,
        prim_main   =   etops.prim.copy,
        prim_last   =   etops.prim.none,
        dim_types   =   (etops.dim.c,    etops.dim.c,     etops.dim.c,     etops.dim.c    ),
        exec_types  =   (etops.exec.seq, etops.exec.seq,  etops.exec.prim, etops.exec.prim),
        dim_sizes   =   (2,              4,               3,               5              ),
        strides     = (((3*4*5,          5,               4*5,             1              ),   # in
                        (3,              2*3,             1,               4*2*3          )),) # out
    )

    # Compile the tensor operation
    perm_op = etops.compile(perm_config)

    # Create input and output arrays
    A = np.random.randn(2,3,4,5).astype(np.float32)
    B = np.random.randn(5,4,2,3).astype(np.float32)

    perm_op.execute(A, None, B)

    # Check correctness
    B_np = np.einsum("abcd->dcab", A)

    error_abs = np.max( np.abs(B - B_np) )
    print("4D Tensor Permutation using copy primitive:")
    print(f"  Max absolute error: {error_abs:.6e}")

    # -------------------------------------------------
    # Third example:
    #   Permutation of a 4D tensor using copy primitive
    #   Uses the built-in optimization routine
    #   Compares the result with NumPy
    # -------------------------------------------------
    perm_config = etops.TensorOperationConfig(
        backend     =   "tpp",
        data_type   =   etops.float32,
        prim_first  =   etops.prim.none,
        prim_main   =   etops.prim.copy,
        prim_last   =   etops.prim.none,
        dim_types   =   (etops.dim.c,    etops.dim.c,     etops.dim.c,    etops.dim.c   ),
        exec_types  =   (etops.exec.seq, etops.exec.seq,  etops.exec.seq, etops.exec.seq),
        dim_sizes   =   (2,              4,               3,              5             ),
        strides     = (((3*4*5,          5,               4*5,            1             ),   # in
                        (3,              2*3,             1,              4*2*3         )),) # out
    )

    # Use default optimization config
    optimized_config = etops.optimize(perm_config)

    # Compile the tensor operation
    perm_op = etops.compile(optimized_config)

    # Create input and output arrays
    A = np.random.randn(2,3,4,5).astype(np.float32)
    B = np.random.randn(5,4,2,3).astype(np.float32)

    # Execute the operation
    perm_op.execute(A, None, B)

    B_np = np.einsum("abcd->dcab", A)

    # Check correctness
    error_abs = np.max( np.abs(B - B_np) )
    print("4D Tensor Permutation using optimized copy primitive:")
    print(f"  Max absolute error: {error_abs:.6e}")

Binary Examples
---------------
Below are some examples showing how to configure and execute binary tensor operations:

.. code-block:: python

    import etops

    # -----------------------------------------
    # First example:
    #   Column-major GEMM operation
    #   Compares the result with NumPy's einsum
    # -----------------------------------------
    # Define a column-major GEMM configuration
    top_config = etops.TensorOperationConfig(
        backend    =   "tpp",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (64,              32,              128            ),
        strides    = (((1,               0,               64             ),   # in0
                       (0,               128,             1              ),   # in1
                       (1,               64,              0              )),) # out
    )

    # Compile the tensor operation
    top = etops.compile(top_config)

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
    batched_config =    etops.TensorOperationConfig(
        backend    =    "tpp",
        data_type  =    etops.float32,
        prim_first =    etops.prim.zero,
        prim_main  =    etops.prim.gemm,
        prim_last  =    etops.prim.none,
        dim_types  =   (etops.dim.c,       etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.shared, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (48,                64,              32,              128            ),
        strides    = (((128*64,            1,               0,               64             ),   # in0
                       (32*128,            0,               128,             1              ),   # in1
                       (32*64,             1,               64,              0              )),) # out
    )

    # Compile the tensor operation
    top = etops.compile(batched_config)

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

    #--------------------------------------------
    # Third example:
    #   GEMM operation with row-major first input
    #   packed to column-major
    #   Compares the result with NumPy's einsum
    # -------------------------------------------

    # Define a row-major GEMM configuration with packing
    top_config = etops.TensorOperationConfig(
        backend    =   "tpp",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (64,              32,              128            ),
        strides    = (((1,               0,               64             ),   # in 0
                       (0,               128,             1              ),   # in 1
                       (1,               64,              0              )),  # out
                      ((128,             0,               1              ),   # packing in 0
                       (0,               0,               0              ),   # packing in 1
                       (0,               0,               0              )),) # packing out
    )

    # Compile the tensor operation
    top = etops.compile(top_config)

    # Create input and output arrays
    import numpy as np
    A = np.random.randn(64,128).astype(np.float32)
    B = np.random.randn(32,128).astype(np.float32)
    C = np.random.randn(32, 64).astype(np.float32)

    # Execute the operation
    top.execute(A, B, C)

    A_T = np.transpose(A)
    C_np = np.einsum("km,nk->nm", A_T, B)

    # Compute absolute and relative errors
    error_abs = np.max( np.abs(C - C_np) )
    error_rel = np.max( np.abs(C - C_np) / (np.abs(C_np) + 1e-8) )
    print("GEMM operation with packing:")
    print(f"  Max absolute error: {error_abs:.6e}")
    print(f"  Max relative error: {error_rel:.6e}")

    # -----------------------------------------------
    # Fourth example:
    #   Batch-reduce GEMM operation with optimization
    #   Compares the result with torch's einsum
    # -----------------------------------------------
    # Define a batch-reduce GEMM configuration
    batched_config =   etops.TensorOperationConfig(
        backend    =   "tpp",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.k,    etops.dim.m,    etops.dim.n,    etops.dim.k   ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (48,             64,             32,             128           ),
        strides    = (((128*64,         1,              0,              64            ),   # in0
                       (32*128,         0,              128,            1             ),   # in1
                       (0,              1,              64,             0             )),) # out
    )

    # Optimize the configuration
    optimized_config = etops.optimize(
        batched_config,
        {
            "target_m":            16,
            "target_n":            12,
            "target_k":            64,
            "num_threads":         4,
            "br_gemm_support":     True,
            "packed_gemm_support": True
        }
    )

    # # Compile the tensor operation
    top = etops.compile(optimized_config)

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

TileIR GPU Example
------------------

.. code-block:: python

    import etops
    import cupy as cp

    #
    # Tile GEMM:
    # abcd,efab->efcd
    # a:32,b:64,c:8,d:256,e:32,f:64
    #

    # Define config
    config = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float16,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.m,       etops.dim.n,       etops.dim.k,    etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.shared, etops.exec.shared, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (8,                 32,                32,             256,             64,              64             ),
        strides    = (((256,               0,                 131072,         1,               0,               2048           ),   # in0
                    (0,                 131072,            64,             0,               2048,            1              ),   # in1
                    (256,               131072,            0,              1,               2048,            0              )),) # out
    )

    # Compile the tensor operation
    top = etops.compile(config)

    # Create input and output matrices
    # A: (k_total × m_total) = (a*b × c*d) = (2048 × 2048)  → acts as in0 (abcd flat)
    # B: (n_total × k_total) = (e*f × a*b) = (2048 × 2048)  → acts as in1 (efab flat)
    # C: (n_total × m_total) = (e*f × c*d) = (2048 × 2048)  → acts as out (efcd flat)
    A = cp.random.randn(2048, 2048).astype(cp.float16)
    B = cp.random.randn(2048, 2048).astype(cp.float16)
    C = cp.zeros((2048, 2048), dtype=cp.float16)

    # Execute the operation
    top.execute(A, B, C)

    # Verify correctness against CuPy matmul reference
    C_ref = cp.matmul(B.astype(cp.float32), A.astype(cp.float32))

    error_abs = float(cp.max(cp.abs(C.astype(cp.float32) - C_ref.astype(cp.float32))))
    error_rel = float(cp.max(cp.abs(C.astype(cp.float32) - C_ref.astype(cp.float32))
                                / (cp.abs(C_ref) + 1e-4)))
    print("Tiled GEMM using TileIR backend (abcd,efab->efcd):")
    print(f"  Max absolute error: {error_abs:.6e}")
    print(f"  Max relative error: {error_rel:.6e}")

    # -------------------------------------------------------------------
    # Benchmark — report FP16 TOPs
    # FLOPs per call = 2 * m_total * n_total * k_total
    #               = 2 * 2048 * 2048 * 2048 ≈ 17.18 GFLOPs
    # -------------------------------------------------------------------
    N_WARMUP = 10
    N_BENCH  = 100

    for _ in range(N_WARMUP):
        top.execute(A, B, C)
    cp.cuda.Device().synchronize()

    t_start = cp.cuda.Event()
    t_end   = cp.cuda.Event()
    t_start.record()
    for _ in range(N_BENCH):
        top.execute(A, B, C)
    t_end.record()
    t_end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(t_start, t_end)
    elapsed_s  = elapsed_ms / 1000.0

    flops = 2 * 2048 * 2048 * 2048
    tops  = (flops * N_BENCH) / elapsed_s / 1e12
    print(f"  Throughput: {tops:.2f} FP16 TOPs  "
          f"({elapsed_ms / N_BENCH:.3f} ms / call)")

TileIR Environment Variables
----------------------------

``ETOPS_DUMP_IR``
    Controls dumping of generated TileIR intermediate representation and/or
    SASS disassembly to stderr. Useful for debugging kernel generation.

    The value is a **comma-separated list** of tokens (case-insensitive,
    whitespace-tolerant).  Multiple tokens may be combined freely.

    +--------------------+--------------------------------------------------------------+
    | Token              | Effect                                                       |
    +====================+==============================================================+
    | *(unset / empty)*  | No dump (default)                                            |
    +--------------------+--------------------------------------------------------------+
    | ``tileir_before``  | Dump TileIR text before optimization passes                  |
    +--------------------+--------------------------------------------------------------+
    | ``tileir_after``   | Dump TileIR text after all optimization passes               |
    +--------------------+--------------------------------------------------------------+
    | ``tileir_all``     | Dump TileIR text before passes, after each individual pass   |
    |                    | (with pass name), and after all passes                       |
    +--------------------+--------------------------------------------------------------+
    | ``sass``           | Dump SASS disassembly of the compiled cubin (requires        |
    |                    | ``cuobjdump`` on ``PATH``)                                   |
    +--------------------+--------------------------------------------------------------+

    **Examples:**

    .. code-block:: bash

        # TileIR after optimization passes only
        ETOPS_DUMP_IR=tileir_after python my_script.py

        # SASS disassembly only
        ETOPS_DUMP_IR=sass python my_script.py

        # TileIR before + after passes, plus SASS
        ETOPS_DUMP_IR=tileir_before,tileir_after,sass python my_script.py

        # Full TileIR trace (every pass) plus SASS
        ETOPS_DUMP_IR=tileir_all,sass python my_script.py

    .. note::

        The ``sass`` token requires ``cuobjdump`` (part of the CUDA Toolkit)
        to be available on ``PATH``.  A ``RuntimeError`` is raised if
        ``cuobjdump`` cannot be found when SASS dumping is requested.


JSON Serialization
------------------

Configurations can be serialized to and from JSON:

.. code-block:: python

    import etops

    config = etops.TensorOperationConfig(
        backend    =   "tpp",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (64,              32,              128            ),
        strides    = (((1,               0,               64             ),   # in0
                       (0,               128,             1              ),   # in1
                       (1,               64,              0              )),) # out
    )

    # Serialize to JSON
    json_str = config.to_json(indent=2)
    print(json_str)

    # Save to file
    config.save("config.json")

    # Load from file
    loaded_config = etops.TensorOperationConfig.load("config.json")

    # Deserialize from JSON
    config2 = etops.TensorOperationConfig.from_json(json_str)