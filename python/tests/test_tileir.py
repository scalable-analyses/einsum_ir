"""
Tests for the tileir backend.

End-to-end correctness tests covering all tensor layouts and data types
with backend="tileir" and @pytest.mark.tileir.
"""

import pytest

import etops


def _etops_available():
    """Check if etops and its tileir optimizer internals are importable."""
    try:
        import etops.backends._tileir.optimizer  # noqa: F401
        import etops.backends._tileir.transforms  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_etops = pytest.mark.skipif(
    not _etops_available(),
    reason="etops tileir optimizer internals not available",
)


class TestTileIRBinary:
    """End-to-end tests for the tileir backend (binary contractions)."""

    def _config_with_dtype(self, base_config, data_type):
        """Create a new TensorOperationConfig from base_config with a different data_type.

        Args:
            base_config: Base TensorOperationConfig to copy from.
            data_type: New data_type to use.

        Returns:
            New TensorOperationConfig with the specified data_type.
        """
        return etops.TensorOperationConfig(
            backend=base_config.backend,
            data_type=data_type,
            prim_first=base_config.prim_first,
            prim_main=base_config.prim_main,
            prim_last=base_config.prim_last,
            dim_types=base_config.dim_types,
            exec_types=base_config.exec_types,
            dim_sizes=base_config.dim_sizes,
            strides=base_config.strides,
        )

    # =========================================================================
    # Configurations -- GEMM layout (column-major: abcd,efab->efcd)
    # =========================================================================

    # fmt: off
    GEMM_SEQ = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.k,    etops.dim.k,    etops.dim.m,    etops.dim.m,    etops.dim.n,    etops.dim.n   ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (48,             16,             4,              32,             2,              8             ),
        strides    = (((2048,           128,            32,             1,              0,              0             ),   # in0
                       (16,             1,              0,              0,              6144,           768           ),   # in1
                       (0,              0,              32,             1,              1024,           128           )),) # out
    )
    # fmt: on

    # fmt: off
    GEMM_SEQ_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.n,    etops.dim.m,    etops.dim.k,    etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (2,              4,              48,             32,              8,               16             ),
        strides    = (((0,              32,             2048,           1,               0,               128            ),   # in0: abcd -> (a*b, c*d) = (768, 128)
                       (6144,           0,              16,             0,               768,             1              ),   # in1: efab -> (e*f, a*b) = (16, 768)
                       (1024,           32,             0,              1,               128,             0              )),) # out: efcd -> (e*f, c*d) = (16, 128)
    )
    # fmt: on

    # fmt: off
    GEMM_SEQ_MULTI_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.n,    etops.dim.k,    etops.dim.m,     etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (2,              48,             4,               32,              8,               16             ),
        strides    = (((0,              2048,           32,              1,               0,               128            ),   # in0
                       (6144,           16,             0,               0,               768,             1              ),   # in1
                       (1024,           0,              32,              1,               128,             0              )),) # out
    )
    # fmt: on

    # fmt: off
    GEMM_SHARED_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.n,       etops.dim.m,       etops.dim.k,    etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.shared, etops.exec.shared, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (2,                 4,                 48,             32,              8,               16             ),
        strides    = (((0,                 32,                2048,           1,               0,               128            ),   # in0
                       (6144,              0,                 16,             0,               768,             1              ),   # in1
                       (1024,              32,                0,              1,               128,             0              )),) # out
    )
    # fmt: on

    # =========================================================================
    # Configurations -- tensor contraction layout (acbd,eafb->ecfd)
    # =========================================================================

    # fmt: off
    TC_SEQ = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.k,    etops.dim.k,    etops.dim.m,    etops.dim.m,    etops.dim.n,    etops.dim.n   ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (48,             16,             4,              32,             2,              8             ),
        strides    = (((2048,           32,             512,            1,              0,              0             ),   # in0
                       (128,            1,              0,              0,              6144,           16            ),   # in1
                       (0,              0,              256,            1,              1024,           32            )),) # out
    )
    # fmt: on

    # fmt: off
    TC_SEQ_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k,    etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (4,              2,              48,             32,              8,               16             ),
        strides    = (((512,            0,              2048,           1,               0,               32             ),   # in0
                       (0,              6144,           128,            0,               16,              1              ),   # in1
                       (256,            1024,           0,              1,               32,              0              )),) # out
    )
    # fmt: on

    # fmt: off
    TC_SEQ_MULTI_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.n,    etops.dim.m,     etops.dim.m,     etops.dim.n,     etops.dim.k,     etops.dim.k    ),
        exec_types =   (etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (2,              4,               32,              8,               48,              16             ),
        strides    = (((0,              512,             1,               0,               2048,            32             ),   # in0
                       (6144,           0,               0,               16,              128,             1              ),   # in1
                       (1024,           256,             1,               32,              0,               0              )),) # out
    )
    # fmt: on

    # fmt: off
    TC_SHARED = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.zero,
        prim_main  =   etops.prim.gemm,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.n,       etops.dim.m,       etops.dim.k,    etops.dim.m,     etops.dim.n,     etops.dim.k    ),
        exec_types =   (etops.exec.shared, etops.exec.shared, etops.exec.seq, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (2,                 4,                 48,             32,              8,               16             ),
        strides    = (((0,                 512,               2048,           1,               0,               32             ),   # in0
                       (6144,              0,                 128,            0,               16,              1              ),   # in1
                       (1024,              256,               0,              1,               32,              0              )),) # out
    )
    # fmt: on

    # =========================================================================
    # Float32 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            GEMM_SEQ_PRIM,
            GEMM_SEQ_MULTI_PRIM,
            GEMM_SHARED_PRIM,
        ],
    )
    def test_execute_gemm_fp32(self, config):
        """Test GEMM execution: einsum('abcd,efab->efcd', A, B).

        The dimension sizes are: |a|=48, |b|=16, |c|=4, |d|=32, |e|=2, |f|=8.

        This is a GEMM where tensors are viewed as matrices:
        - in0 is shaped as (a*b, c*d) = (768, 128) matrix KM
        - in1 is shaped as (e*f, a*b) = (16, 768) matrix NK
        - out is shaped as (e*f, c*d) = (16, 128) matrix NM
        """
        import cupy as cp

        config = self._config_with_dtype(config, etops.float32)
        op = etops.compile(config)

        in0 = cp.random.randn(768, 128).astype(cp.float32)
        in1 = cp.random.randn(16, 768).astype(cp.float32)
        out = cp.random.randn(16, 128).astype(cp.float32)

        op.execute(in0, in1, out)

        expected = cp.matmul(in1, in0)
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            TC_SEQ_PRIM,
            TC_SEQ_MULTI_PRIM,
            TC_SHARED,
        ],
    )
    def test_execute_tc_fp32(self, config):
        """Test tensor contraction: einsum('acbd,eafb->ecfd', A, B).

        The dimension sizes are: |a|=48, |b|=16, |c|=4, |d|=32, |e|=2, |f|=8.
        """
        import cupy as cp

        config = self._config_with_dtype(config, etops.float32)
        op = etops.compile(config)

        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float32)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float32)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float32)

        op.execute(in0, in1, out)

        expected = cp.einsum("acbd,eafb->ecfd", in0, in1)
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Float64 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            GEMM_SEQ_PRIM,
            GEMM_SEQ_MULTI_PRIM,
            GEMM_SHARED_PRIM,
        ],
    )
    def test_execute_gemm_fp64(self, config):
        """Test GEMM execution with float64."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float64)
        op = etops.compile(config)

        in0 = cp.random.randn(768, 128).astype(cp.float64)
        in1 = cp.random.randn(16, 768).astype(cp.float64)
        out = cp.random.randn(16, 128).astype(cp.float64)

        op.execute(in0, in1, out)

        expected = cp.matmul(in1, in0)
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            TC_SEQ_PRIM,
            TC_SEQ_MULTI_PRIM,
            TC_SHARED,
        ],
    )
    def test_execute_tc_fp64(self, config):
        """Test tensor contraction with float64."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float64)
        op = etops.compile(config)

        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float64)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float64)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float64)

        op.execute(in0, in1, out)

        expected = cp.einsum("acbd,eafb->ecfd", in0, in1)
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Float16 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            GEMM_SEQ_PRIM,
            GEMM_SEQ_MULTI_PRIM,
            GEMM_SHARED_PRIM,
        ],
    )
    def test_execute_gemm_fp16(self, config):
        """Test GEMM execution with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        op = etops.compile(config)

        in0 = cp.random.randn(768, 128).astype(cp.float16)
        in1 = cp.random.randn(16, 768).astype(cp.float16)
        out = cp.random.randn(16, 128).astype(cp.float16)

        op.execute(in0, in1, out)

        # Compare against float32 reference
        expected = cp.matmul(in1.astype(cp.float32), in0.astype(cp.float32))
        cp.testing.assert_allclose(
            out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            GEMM_SEQ,
        ],
    )
    def test_optimize_and_execute_gemm_fp16(self, config):
        """Test GEMM optimization and execution with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        opt_config = etops.get_default_optimization_config("tileir")
        config = etops.optimize(config, opt_config)

        op = etops.compile(config)

        in0 = cp.random.randn(768, 128).astype(cp.float16)
        in1 = cp.random.randn(16, 768).astype(cp.float16)
        out = cp.random.randn(16, 128).astype(cp.float16)

        op.execute(in0, in1, out)

        # Compare against float32 reference
        expected = cp.matmul(in1.astype(cp.float32), in0.astype(cp.float32))
        cp.testing.assert_allclose(
            out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            TC_SEQ_PRIM,
            TC_SEQ_MULTI_PRIM,
            TC_SHARED,
        ],
    )
    def test_execute_tc_fp16(self, config):
        """Test tensor contraction with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        op = etops.compile(config)

        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float16)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float16)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float16)

        op.execute(in0, in1, out)

        # Compare against float32 reference
        expected = cp.einsum(
            "acbd,eafb->ecfd",
            in0.astype(cp.float32),
            in1.astype(cp.float32),
        )
        cp.testing.assert_allclose(
            out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2
        )

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            TC_SEQ,
        ],
    )
    def test_optimize_and_execute_tc_fp16(self, config):
        """Test tensor contraction optimization and execution with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        opt_config = etops.get_default_optimization_config("tileir")
        config = etops.optimize(config, opt_config)
        op = etops.compile(config)

        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float16)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float16)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float16)

        op.execute(in0, in1, out)

        # Compare against float32 reference
        expected = cp.einsum(
            "acbd,eafb->ecfd",
            in0.astype(cp.float32),
            in1.astype(cp.float32),
        )
        cp.testing.assert_allclose(
            out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2
        )

    # =========================================================================
    # Bfloat16 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            GEMM_SEQ_PRIM,
            GEMM_SEQ_MULTI_PRIM,
            GEMM_SHARED_PRIM,
        ],
    )
    def test_execute_gemm_bf16(self, config):
        """Test GEMM execution with bfloat16."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available for bfloat16 support")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bfloat16 test")

        config = self._config_with_dtype(config, etops.bfloat16)
        op = etops.compile(config)

        in0 = torch.randn(768, 128, dtype=torch.bfloat16, device="cuda")
        in1 = torch.randn(16, 768, dtype=torch.bfloat16, device="cuda")
        out = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")

        op.execute(in0, in1, out)

        # Compare against float32 reference
        expected = torch.matmul(in1.float(), in0.float())
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            TC_SEQ_PRIM,
            TC_SEQ_MULTI_PRIM,
            TC_SHARED,
        ],
    )
    def test_execute_tc_bf16(self, config):
        """Test tensor contraction with bfloat16."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available for bfloat16 support")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bfloat16 test")

        config = self._config_with_dtype(config, etops.bfloat16)
        op = etops.compile(config)

        in0 = torch.randn(48, 4, 16, 32, dtype=torch.bfloat16, device="cuda")
        in1 = torch.randn(2, 48, 8, 16, dtype=torch.bfloat16, device="cuda")
        out = torch.randn(2, 4, 8, 32, dtype=torch.bfloat16, device="cuda")

        op.execute(in0, in1, out)

        # Compare against float32 reference using einsum
        expected = torch.einsum("acbd,eafb->ecfd", in0.float(), in1.float())
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    # =========================================================================
    # TensorFloat-32 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            GEMM_SEQ_PRIM,
            GEMM_SEQ_MULTI_PRIM,
            GEMM_SHARED_PRIM,
        ],
    )
    def test_execute_gemm_tf32(self, config):
        """Test GEMM execution with tensorfloat32."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.tfloat32)
        op = etops.compile(config)

        # TF32 uses float32 storage
        in0 = cp.random.randn(768, 128).astype(cp.float32)
        in1 = cp.random.randn(16, 768).astype(cp.float32)
        out = cp.random.randn(16, 128).astype(cp.float32)

        op.execute(in0, in1, out)

        expected = cp.matmul(in1, in0)
        cp.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            TC_SEQ_PRIM,
            TC_SEQ_MULTI_PRIM,
            TC_SHARED,
        ],
    )
    def test_execute_tc_tf32(self, config):
        """Test tensor contraction with tensorfloat32."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.tfloat32)
        op = etops.compile(config)

        # TF32 uses float32 storage
        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float32)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float32)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float32)

        op.execute(in0, in1, out)

        expected = cp.einsum("acbd,eafb->ecfd", in0, in1)
        cp.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)


class TestTileIRUnary:
    """End-to-end tests for unary operations in the tileir backend."""

    def _config_with_dtype(self, base_config, data_type):
        """Create a new TensorOperationConfig from base_config with a different data_type.

        Args:
            base_config: Base TensorOperationConfig to copy from.
            data_type: New data_type to use.

        Returns:
            New TensorOperationConfig with the specified data_type.
        """
        return etops.TensorOperationConfig(
            backend=base_config.backend,
            data_type=data_type,
            prim_first=base_config.prim_first,
            prim_main=base_config.prim_main,
            prim_last=base_config.prim_last,
            dim_types=base_config.dim_types,
            exec_types=base_config.exec_types,
            dim_sizes=base_config.dim_sizes,
            strides=base_config.strides,
        )

    # =========================================================================
    # Configurations -- Copy (contiguous 1D: 1024 elements)
    # =========================================================================

    # All-sequential copy: 2 dims, all seq.
    # Layout: contiguous 1D vector of 32*32 = 1024 elements.
    # fmt: off
    COPY_SEQ = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,    etops.dim.c   ),
        exec_types =   (etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (32,             32            ),
        strides    = (((32,             1              ),   # in
                       (32,             1              )),) # out
    )
    # fmt: on

    # Copy with prim dims: row-major contiguous.
    # fmt: off
    COPY_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,       etops.dim.c,     etops.dim.c    ),
        exec_types =   (etops.exec.shared, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (4,                 16,              16             ),
        strides    = (((256,               16,              1              ),   # in
                       (256,               16,              1              )),) # out
    )
    # fmt: on

    # Transpose copy: in is row-major, out is transposed within prim tile.
    # Shape: (4, 4, 16) = 256 elements.
    # fmt: off
    COPY_TRANSPOSE_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,       etops.dim.c,     etops.dim.c    ),
        exec_types =   (etops.exec.shared, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (4,                 4,               16             ),
        strides    = (((64,                16,              1              ),   # in:  row-major
                       (64,                1,               4              )),) # out: transposed
    )
    # fmt: on

    # Copy with multi-prim: 3 prim dims.
    # fmt: off
    COPY_MULTI_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,       etops.dim.c,     etops.dim.c,     etops.dim.c    ),
        exec_types =   (etops.exec.shared, etops.exec.prim, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (2,                 4,               8,               16             ),
        strides    = (((512,               128,             16,              1              ),   # in
                       (512,               128,             16,              1              )),) # out
    )
    # fmt: on

    # =========================================================================
    # Configurations -- ReLU (element-wise max(in, 0) -> out)
    # =========================================================================

    # fmt: off
    RELU_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.relu,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,       etops.dim.c,     etops.dim.c    ),
        exec_types =   (etops.exec.shared, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (4,                 16,              16             ),
        strides    = (((256,               16,              1              ),   # in
                       (256,               16,              1              )),) # out
    )
    # fmt: on

    # ReLU with transpose: in row-major, out column-major within tile.
    # fmt: off
    RELU_TRANSPOSE_PRIM = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.relu,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,       etops.dim.c,     etops.dim.c    ),
        exec_types =   (etops.exec.shared, etops.exec.prim, etops.exec.prim),
        dim_sizes  =   (4,                 4,               16             ),
        strides    = (((64,                16,              1              ),   # in:  row-major
                       (64,                1,               4              )),) # out: transposed
    )
    # fmt: on

    # ReLU with all-seq (optimizer test input).
    # fmt: off
    RELU_SEQ = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.relu,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,    etops.dim.c   ),
        exec_types =   (etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (32,             32            ),
        strides    = (((32,             1              ),   # in
                       (32,             1              )),) # out
    )
    # fmt: on

    # =========================================================================
    # Copy -- Float32 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            COPY_PRIM,
            COPY_MULTI_PRIM,
        ],
    )
    def test_copy_fp32(self, config):
        """Test contiguous copy execution with float32."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float32)
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.random.randn(total).astype(cp.float32)
        out = cp.random.randn(total).astype(cp.float32)

        op.execute(in0, None, out)

        cp.testing.assert_allclose(out, in0, rtol=0, atol=0)

    @pytest.mark.tileir
    def test_copy_transpose_fp32(self):
        """Test transpose copy: verify output is transposed version of input."""
        import cupy as cp

        config = self.COPY_TRANSPOSE_PRIM
        op = etops.compile(config)

        # Shape: (4, 4, 16) = 256 elements.
        # in: row-major strides (64, 16, 1)
        # out: transposed strides (64, 1, 4)
        # The config maps (a, b, c) -> in[a*64 + b*16 + c], out[a*64 + c*4 + b]
        # So for a fixed a, the 2D subtensor (b, c) is transposed.
        in0 = cp.arange(256, dtype=cp.float32)
        out = cp.zeros(256, dtype=cp.float32)

        op.execute(in0, None, out)

        # Manually verify: reshape, transpose inner dims, flatten.
        in_3d = in0.reshape(4, 4, 16)
        expected_3d = cp.transpose(in_3d, (0, 2, 1))  # (4, 16, 4)
        expected = expected_3d.reshape(-1)

        cp.testing.assert_allclose(out, expected, rtol=0, atol=0)

    @pytest.mark.tileir
    def test_optimize_and_copy_fp32(self):
        """Test copy: optimize all-seq config then compile+execute."""
        import cupy as cp

        config = self.COPY_SEQ
        opt_config = etops.get_default_optimization_config("tileir")
        config = etops.optimize(config, opt_config)

        # Verify optimizer assigned exec_types
        has_prim = any(et == etops.exec.prim for et in config.exec_types)
        assert has_prim, "Optimizer should assign at least one prim dim."

        op = etops.compile(config)

        total = 1024  # 32*32
        in0 = cp.random.randn(total).astype(cp.float32)
        out = cp.random.randn(total).astype(cp.float32)

        op.execute(in0, None, out)

        cp.testing.assert_allclose(out, in0, rtol=0, atol=0)

    # =========================================================================
    # Copy -- Float64 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            COPY_PRIM,
        ],
    )
    def test_copy_fp64(self, config):
        """Test copy execution with float64."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float64)
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.random.randn(total).astype(cp.float64)
        out = cp.random.randn(total).astype(cp.float64)

        op.execute(in0, None, out)

        cp.testing.assert_allclose(out, in0, rtol=0, atol=0)

    # =========================================================================
    # Copy -- Float16 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            COPY_PRIM,
        ],
    )
    def test_copy_fp16(self, config):
        """Test copy execution with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.random.randn(total).astype(cp.float16)
        out = cp.random.randn(total).astype(cp.float16)

        op.execute(in0, None, out)

        cp.testing.assert_allclose(out, in0, rtol=0, atol=0)

    # =========================================================================
    # ReLU -- Float32 tests
    # =========================================================================

    @pytest.mark.tileir
    @pytest.mark.parametrize(
        "config",
        [
            RELU_PRIM,
        ],
    )
    def test_relu_fp32(self, config):
        """Test ReLU execution with float32."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float32)
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        # Use values spanning negative and positive.
        in0 = cp.linspace(-5.0, 5.0, total, dtype=cp.float32)
        out = cp.zeros(total, dtype=cp.float32)

        op.execute(in0, None, out)

        expected = cp.maximum(in0, 0.0)
        cp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.tileir
    def test_relu_transpose_fp32(self):
        """Test ReLU with transpose: max(in, 0) with permuted output layout."""
        import cupy as cp

        config = self.RELU_TRANSPOSE_PRIM
        op = etops.compile(config)

        # Shape: (4, 4, 16) = 256 elements.
        # in strides: (64, 16, 1) — row-major (a, b, c)
        # out strides: (64, 1, 4) — transposed: for fixed a, (b, c) -> (c, b)
        in0 = cp.linspace(-5.0, 5.0, 256, dtype=cp.float32)
        out = cp.zeros(256, dtype=cp.float32)

        op.execute(in0, None, out)

        # Reference: relu then transpose.
        relu_in = cp.maximum(in0, 0.0)
        in_3d = relu_in.reshape(4, 4, 16)
        expected_3d = cp.transpose(in_3d, (0, 2, 1))  # (4, 16, 4)
        expected = expected_3d.reshape(-1)

        cp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.tileir
    def test_relu_all_negative_fp32(self):
        """Test ReLU with all-negative input: output should be all zeros."""
        import cupy as cp

        config = self.RELU_PRIM
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.full(total, -3.14, dtype=cp.float32)
        out = cp.ones(total, dtype=cp.float32) * 99.0

        op.execute(in0, None, out)

        expected = cp.zeros(total, dtype=cp.float32)
        cp.testing.assert_allclose(out, expected, rtol=0, atol=0)

    @pytest.mark.tileir
    def test_relu_all_positive_fp32(self):
        """Test ReLU with all-positive input: output should equal input."""
        import cupy as cp

        config = self.RELU_PRIM
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.arange(1, total + 1, dtype=cp.float32)
        out = cp.zeros(total, dtype=cp.float32)

        op.execute(in0, None, out)

        cp.testing.assert_allclose(out, in0, rtol=0, atol=0)

    @pytest.mark.tileir
    def test_optimize_and_relu_fp32(self):
        """Test ReLU: optimize all-seq config then compile+execute."""
        import cupy as cp

        config = self.RELU_SEQ
        opt_config = etops.get_default_optimization_config("tileir")
        config = etops.optimize(config, opt_config)

        has_prim = any(et == etops.exec.prim for et in config.exec_types)
        assert has_prim, "Optimizer should assign at least one prim dim."

        op = etops.compile(config)

        total = 1024  # 32*32
        in0 = cp.linspace(-10.0, 10.0, total, dtype=cp.float32)
        out = cp.zeros(total, dtype=cp.float32)

        op.execute(in0, None, out)

        expected = cp.maximum(in0, 0.0)
        cp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    # =========================================================================
    # ReLU -- Float64 tests
    # =========================================================================

    @pytest.mark.tileir
    def test_relu_fp64(self):
        """Test ReLU execution with float64."""
        import cupy as cp

        config = self._config_with_dtype(self.RELU_PRIM, etops.float64)
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.linspace(-5.0, 5.0, total, dtype=cp.float64)
        out = cp.zeros(total, dtype=cp.float64)

        op.execute(in0, None, out)

        expected = cp.maximum(in0, 0.0)
        cp.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

    # =========================================================================
    # ReLU -- Float16 tests
    # =========================================================================

    @pytest.mark.tileir
    def test_relu_fp16(self):
        """Test ReLU execution with float16."""
        import cupy as cp

        config = self._config_with_dtype(self.RELU_PRIM, etops.float16)
        op = etops.compile(config)

        total = 1
        for s in config.dim_sizes:
            total *= s

        in0 = cp.linspace(-5.0, 5.0, total, dtype=cp.float16)
        out = cp.zeros(total, dtype=cp.float16)

        op.execute(in0, None, out)

        expected = cp.maximum(in0, cp.float16(0.0))
        cp.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)

    # =========================================================================
    # Permutation regression tests
    # =========================================================================

    # Permutation [b,e,h,i] -> [e,h,i,b], b=60, e=h=i=8.
    # fmt: off
    PERM_BEHI_TO_EHIB_SEQ = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,    etops.dim.c,    etops.dim.c,    etops.dim.c   ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (8,              8,              8,              60             ),
        strides    = (((64,             8,              1,              512            ),   # in0: [b,e,h,i]
                       (3840,           480,            60,             1              )),) # out: [e,h,i,b]
    )
    # fmt: on

    # Permutation [a,c,d,b] -> [a,b,c,d], a=60, b=60, c=20, d=20.
    # fmt: off
    PERM_ACDB_TO_ABCD_SEQ = etops.TensorOperationConfig(
        backend    =   "tileir",
        data_type  =   etops.float32,
        prim_first =   etops.prim.none,
        prim_main  =   etops.prim.copy,
        prim_last  =   etops.prim.none,
        dim_types  =   (etops.dim.c,    etops.dim.c,    etops.dim.c,    etops.dim.c   ),
        exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq, etops.exec.seq),
        dim_sizes  =   (60,             20,             60,             20             ),
        strides    = (((24000,          1200,           1,              60             ),   # in0: [a,c,d,b]
                       (24000,          20,             400,            1              )),) # out: [a,b,c,d]
    )
    # fmt: on

    @pytest.mark.tileir
    def test_optimize_permutation_volume_cap(self):
        """Optimizer caps padded prim tile volume for large permutations."""
        from etops.backends._tileir.config_analysis import _next_pow2

        config = self.PERM_ACDB_TO_ABCD_SEQ
        opt = etops.optimize(config)

        prim_volume = 1
        for et, sz in zip(opt.exec_types, opt.dim_sizes):
            if et == etops.exec.prim:
                prim_volume *= _next_pow2(sz)

        assert prim_volume <= 32768, (
            f"Padded prim tile volume {prim_volume} exceeds 32768"
        )

    @pytest.mark.tileir
    def test_optimize_permutation_tileiras_workaround(self):
        """Optimizer avoids tileiras padded-64 crash for asymmetric strides."""
        from etops.backends._tileir.config_analysis import _next_pow2

        config = self.PERM_BEHI_TO_EHIB_SEQ
        opt = etops.optimize(config)

        # The problematic dim (original dim 3, size=60) must not be the
        # innermost prim dim after optimization.
        prim_dims = [
            (sz, s_in, s_out)
            for et, sz, s_in, s_out in zip(
                opt.exec_types,
                opt.dim_sizes,
                opt.strides[0][0],
                opt.strides[0][1],
            )
            if et == etops.exec.prim
        ]
        assert len(prim_dims) > 0, "Optimizer must assign at least one prim dim"

        # Innermost prim dim is the last in the config ordering.
        innermost_sz = prim_dims[-1][0]
        padded = _next_pow2(innermost_sz)
        innermost_s_in = prim_dims[-1][1]
        innermost_s_out = prim_dims[-1][2]

        # The tileiras bug triggers when padded=64, size%4==0,
        # and unit stride in exactly one of in0/out.
        is_buggy = (
            padded == 64
            and innermost_sz % 4 == 0
            and (innermost_s_in == 1) != (innermost_s_out == 1)
        )
        assert not is_buggy, (
            f"Innermost prim dim (size={innermost_sz}, padded={padded}) "
            f"would trigger the tileiras padded-64 crash"
        )

    @pytest.mark.tileir
    def test_compile_and_execute_permutation_behi_to_ehib(self):
        """End-to-end: optimize, compile, execute [b,e,h,i]->[e,h,i,b]."""
        import cupy as cp
        import numpy as np

        config = self.PERM_BEHI_TO_EHIB_SEQ
        opt = etops.optimize(config)
        op = etops.compile(opt)

        # Original config dims: (b=8, e=8, h=8, i=60)
        # strides_in0=(64,8,1,512), strides_out=(3840,480,60,1)
        dim_sizes = tuple(config.dim_sizes)
        strides_in = tuple(config.strides[0][0])
        strides_out = tuple(config.strides[0][1])
        total = 1
        for s in dim_sizes:
            total *= s

        in_np = np.arange(total, dtype=np.float32)
        in_gpu = cp.asarray(in_np)
        out_gpu = cp.zeros(total, dtype=cp.float32)

        op.execute(in_gpu, None, out_gpu)
        out_np = cp.asnumpy(out_gpu)

        # The copy operation does:
        #   for each multi-index (d0,d1,d2,d3):
        #     out_buf[sum(di*strides_out[i])] = in_buf[sum(di*strides_in[i])]
        # Verify element-by-element using the original strides.
        byte_size = np.float32().itemsize
        in_view = np.lib.stride_tricks.as_strided(
            in_np,
            shape=dim_sizes,
            strides=tuple(s * byte_size for s in strides_in),
        )
        out_view = np.lib.stride_tricks.as_strided(
            out_np,
            shape=dim_sizes,
            strides=tuple(s * byte_size for s in strides_out),
        )
        np.testing.assert_allclose(out_view, in_view, rtol=0, atol=0)

    @pytest.mark.tileir
    def test_compile_and_execute_permutation_acdb_to_abcd(self):
        """End-to-end: optimize, compile, execute [a,c,d,b]->[a,b,c,d]."""
        import cupy as cp
        import numpy as np

        config = self.PERM_ACDB_TO_ABCD_SEQ
        opt = etops.optimize(config)
        op = etops.compile(opt)

        # Original config dims: (a=60, c=20, d=60, b=20)
        # strides_in0=(24000,1200,1,60), strides_out=(24000,20,400,1)
        dim_sizes = tuple(config.dim_sizes)
        strides_in = tuple(config.strides[0][0])
        strides_out = tuple(config.strides[0][1])
        total = 1
        for s in dim_sizes:
            total *= s

        in_np = np.arange(total, dtype=np.float32)
        in_gpu = cp.asarray(in_np)
        out_gpu = cp.zeros(total, dtype=cp.float32)

        op.execute(in_gpu, None, out_gpu)
        out_np = cp.asnumpy(out_gpu)

        byte_size = np.float32().itemsize
        in_view = np.lib.stride_tricks.as_strided(
            in_np,
            shape=dim_sizes,
            strides=tuple(s * byte_size for s in strides_in),
        )
        out_view = np.lib.stride_tricks.as_strided(
            out_np,
            shape=dim_sizes,
            strides=tuple(s * byte_size for s in strides_out),
        )
        np.testing.assert_allclose(out_view, in_view, rtol=0, atol=0)


class TestTileIRTransforms:
    """Tests for config transforms and the optimizer dim-splitting pipeline."""

    @_skip_no_etops
    def test_largest_factor_leq_basic(self):
        """Verify _largest_factor_leq for representative inputs."""
        from etops.backends._tileir.optimizer import _largest_factor_leq

        # Exact match: n <= cap returns n.
        assert _largest_factor_leq(64, 64) == 64
        assert _largest_factor_leq(32, 64) == 32

        # Composite numbers split nicely.
        assert _largest_factor_leq(305, 64) == 61  # 305 = 5 * 61
        assert _largest_factor_leq(100, 64) == 50  # 100 = 2 * 50
        assert _largest_factor_leq(72, 64) == 36  # 72 = 2 * 36

        # Prime > cap falls back to 1.
        assert _largest_factor_leq(71, 64) == 1

        # Edge cases.
        assert _largest_factor_leq(1, 1) == 1
        assert _largest_factor_leq(128, 64) == 64  # 128 = 2 * 64

    @_skip_no_etops
    def test_largest_factor_leq_invalid_inputs(self):
        """_largest_factor_leq raises ValueError for n < 1 or cap < 1."""
        from etops.backends._tileir.optimizer import _largest_factor_leq

        with pytest.raises(ValueError, match="n must be >= 1"):
            _largest_factor_leq(0, 64)
        with pytest.raises(ValueError, match="cap must be >= 1"):
            _largest_factor_leq(64, 0)
        with pytest.raises(ValueError, match="n must be >= 1"):
            _largest_factor_leq(-1, 64)

    @_skip_no_etops
    def test_split_dim_binary_preserves_semantics(self):
        """Splitting a binary config dim preserves stride relationships."""
        from etops.backends._tileir.transforms import split_dim

        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (305,            100,            71            ),
            strides    = (((1,              0,              305           ),   # in0: M*K row-major
                           (0,              71,             1             ),   # in1: N*K row-major
                           (1,              305,            0             )),) # out: M*N col-major
        )
        # fmt: on

        # Split M=305 -> outer=5, inner=61.
        split = split_dim(config, 0, 61)
        assert tuple(split.dim_sizes) == (5, 61, 100, 71)
        assert split.dim_types[0] == etops.dim.m
        assert split.dim_types[1] == etops.dim.m
        # Stride: outer = 61*1 = 61, inner = 1 for in0.
        assert split.strides[0][0][0] == 61
        assert split.strides[0][0][1] == 1
        # in1 stride for M is 0, so both halves are 0.
        assert split.strides[0][1][0] == 0
        assert split.strides[0][1][1] == 0
        # out stride for M was 1 -> outer = 61*1 = 61, inner = 1.
        assert split.strides[0][2][0] == 61
        assert split.strides[0][2][1] == 1

    @_skip_no_etops
    def test_fuse_dims_roundtrip(self):
        """Split then fuse recovers the original config."""
        from etops.backends._tileir.transforms import fuse_dims, split_dim

        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.none,
            prim_main  =   etops.prim.copy,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.c,       etops.dim.c    ),
            exec_types =   (etops.exec.shared, etops.exec.prim),
            dim_sizes  =   (120,               16             ),
            strides    = (((16,                1              ),   # in
                           (16,                1              )),) # out
        )
        # fmt: on

        split = split_dim(config, 0, 60)
        assert tuple(split.dim_sizes) == (2, 60, 16)

        fused = fuse_dims(split, 0, 1)
        assert tuple(fused.dim_sizes) == tuple(config.dim_sizes)
        assert tuple(fused.strides[0][0]) == tuple(config.strides[0][0])
        assert tuple(fused.strides[0][1]) == tuple(config.strides[0][1])

    @_skip_no_etops
    def test_reorder_dims_identity(self):
        """Identity permutation is a no-op."""
        from etops.backends._tileir.transforms import reorder_dims

        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  =   (32,              16,              8              ),
            strides    = (((1,               0,               32            ),
                           (0,               8,               1             ),
                           (1,               32,              0             )),)
        )
        # fmt: on

        reordered = reorder_dims(config, (0, 1, 2))
        assert tuple(reordered.dim_sizes) == tuple(config.dim_sizes)
        assert tuple(reordered.dim_types) == tuple(config.dim_types)

    @_skip_no_etops
    def test_reorder_dims_reversal(self):
        """Reversing dimension order swaps all arrays."""
        from etops.backends._tileir.transforms import reorder_dims

        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types =   (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  =   (32,              16,              8              ),
            strides    = (((1,               0,               32            ),
                           (0,               8,               1             ),
                           (1,               32,              0             )),)
        )
        # fmt: on

        reordered = reorder_dims(config, (2, 1, 0))
        assert tuple(reordered.dim_sizes) == (8, 16, 32)
        assert tuple(reordered.dim_types) == (etops.dim.k, etops.dim.n, etops.dim.m)
        assert reordered.strides[0][0] == (32, 0, 1)
        assert reordered.strides[0][1] == (1, 8, 0)
        assert reordered.strides[0][2] == (0, 32, 1)

    @_skip_no_etops
    def test_binary_optimizer_splits_oversized_dims(self):
        """Optimizer splits dims > max_prim_dim_size for binary configs."""
        from etops.backends._tileir.config_analysis import _next_pow2

        # The reported failing config: M=305, N=100, K=71.
        # Without splitting, padded volume = 512*128*128 = 8.4M -> timeout.
        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (305,            100,            71            ),
            strides    = (((1,              0,              305           ),   # in0
                           (0,              71,             1             ),   # in1
                           (1,              305,            0             )),) # out
        )
        # fmt: on

        opt = etops.optimize(config)

        # Composite dims (305, 100) must be split; prime K=71 stays as-is
        # because it has no factor <= 64 other than 1.
        # Verify all non-prime prim dims are <= 64.
        for dt, et, sz in zip(opt.dim_types, opt.exec_types, opt.dim_sizes):
            if et == etops.exec.prim and sz > 64:
                # Only acceptable if the dim is prime (cannot split).
                from etops.backends._tileir.optimizer import _largest_factor_leq

                assert _largest_factor_leq(sz, 64) == 1, (
                    f"Prim dim size {sz} exceeds max_prim_dim_size=64 and is not prime"
                )

    @_skip_no_etops
    def test_binary_optimizer_small_dims_unchanged(self):
        """Dims already <= max_prim_dim_size are not split."""
        # All dims are small: M=32, N=16, K=8.
        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (32,             16,             8             ),
            strides    = (((1,              0,              32            ),
                           (0,              8,              1             ),
                           (1,              32,             0             )),)
        )
        # fmt: on

        opt = etops.optimize(config)

        # Should have exactly 3 prim dims (M, N, K) since all fit.
        prim_count = sum(1 for et in opt.exec_types if et == etops.exec.prim)
        assert prim_count == 3, f"Expected 3 prim dims, got {prim_count}"

    @_skip_no_etops
    def test_binary_optimizer_prime_dim_unsplit(self):
        """Prime dim > cap remains unsplit (cannot factor)."""
        # K=71 is prime and > 64. Since it has no factor <= 64 (other than 1),
        # the optimizer cannot split it. It should remain as a single prim dim.
        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (32,             16,             71            ),
            strides    = (((1,              0,              32            ),
                           (0,              16,             1             ),
                           (1,              32,             0             )),)
        )
        # fmt: on

        opt = etops.optimize(config)

        # K=71 is prime and cannot be split. It must appear as a single
        # prim K dim with its original size of 71.
        prim_k_sizes = [
            sz
            for dt, et, sz in zip(opt.dim_types, opt.exec_types, opt.dim_sizes)
            if dt == etops.dim.k and et == etops.exec.prim
        ]
        assert 71 in prim_k_sizes, (
            f"Expected prime K=71 to remain unsplit as a prim dim, "
            f"got prim K sizes: {prim_k_sizes}"
        )
        # There should be exactly one K prim dim (the unsplit 71).
        assert len(prim_k_sizes) == 1, (
            f"Expected exactly 1 prim K dim, got {len(prim_k_sizes)}: {prim_k_sizes}"
        )

    @_skip_no_etops
    def test_binary_optimizer_k_outer_is_seq(self):
        """K outer dims must be seq (reduction cannot be shared)."""
        # K=128 splits to (2, 64). The outer K=2 must be seq.
        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (32,             16,             128           ),
            strides    = (((1,              0,              32            ),
                           (0,              128,            1             ),
                           (1,              32,             0             )),)
        )
        # fmt: on

        opt = etops.optimize(config)

        # Find K dims and their exec types.
        k_info = [
            (et, sz)
            for dt, et, sz in zip(opt.dim_types, opt.exec_types, opt.dim_sizes)
            if dt == etops.dim.k
        ]
        # At least one K dim should be prim, and any non-prim K must be seq.
        for et, sz in k_info:
            if et != etops.exec.prim:
                assert et == etops.exec.seq, (
                    f"Non-prim K dim (size={sz}) should be seq, got {et}"
                )

    @_skip_no_etops
    def test_unary_optimizer_splits_oversized_dims(self):
        """Optimizer splits dims > max_prim_dim_size for unary configs.

        Uses a large-enough config so that not all dims fit within the
        prim tile volume cap. The split allows the optimizer to push
        the outer factor to shared while keeping the inner as prim.
        """
        # fmt: off
        # Logical shape: (120, 32, 16).
        # Padded volume without split: 128*32*16 = 65536 > 32768 cap.
        # After split 120 -> (2, 60): padded 60*32*16 = 64*32*16 = 32768 fits.
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.none,
            prim_main  =   etops.prim.copy,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.c,    etops.dim.c,    etops.dim.c   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (120,            32,             16            ),
            strides    = (((512,            16,             1             ),   # in
                           (16,             512,            1             )),) # out
        )
        # fmt: on

        opt = etops.optimize(config)

        # The padded prim tile volume must not exceed the default cap (32768).
        from etops.backends._tileir.config_analysis import _next_pow2

        prim_volume = 1
        for et, sz in zip(opt.exec_types, opt.dim_sizes):
            if et == etops.exec.prim:
                prim_volume *= _next_pow2(sz)
        assert prim_volume <= 32768, (
            f"Padded prim volume {prim_volume} exceeds cap 32768"
        )

        # The original 120-dim should have been split. Verify it's gone.
        assert 120 not in opt.dim_sizes, (
            f"Dim size 120 should have been split, got sizes: {opt.dim_sizes}"
        )

        # The outer factor (2) should be shared, inner (60) should be prim.
        shared_sizes = [
            sz
            for et, sz in zip(opt.exec_types, opt.dim_sizes)
            if et == etops.exec.shared
        ]
        assert 2 in shared_sizes, (
            f"Expected outer factor 2 as shared dim, got shared: {shared_sizes}"
        )

    @_skip_no_etops
    def test_default_optimization_config_has_max_prim_dim_size(self):
        """Default optimization config includes max_prim_dim_size."""
        defaults = etops.get_default_optimization_config("tileir")
        assert "max_prim_dim_size" in defaults
        assert defaults["max_prim_dim_size"] == 64

    @pytest.mark.tileir
    def test_binary_optimizer_compile_oversized_gemm(self):
        """End-to-end: optimize+compile GEMM with oversized dims (M=305, N=100, K=72)."""
        import cupy as cp

        # Use K=72 (not prime) so the split is clean: 72 -> 36*2 or similar.
        # This is close to the reported failing config.
        # fmt: off
        config = etops.TensorOperationConfig(
            backend    =   "tileir",
            data_type  =   etops.float32,
            prim_first =   etops.prim.zero,
            prim_main  =   etops.prim.gemm,
            prim_last  =   etops.prim.none,
            dim_types  =   (etops.dim.m,    etops.dim.n,    etops.dim.k   ),
            exec_types =   (etops.exec.seq, etops.exec.seq, etops.exec.seq),
            dim_sizes  =   (305,            100,            72            ),
            strides    = (((1,              0,              305           ),   # in0: M*K col-major
                           (0,              72,             1             ),   # in1: N*K row-major
                           (1,              305,            0             )),) # out: M*N col-major
        )
        # fmt: on

        opt = etops.optimize(config)
        op = etops.compile(opt)

        # Allocate: in0 is M x K = 305 x 72, in1 is N x K = 100 x 72,
        # out is M x N = 305 x 100.
        in0 = cp.random.randn(305 * 72).astype(cp.float32)
        in1 = cp.random.randn(100 * 72).astype(cp.float32)
        out = cp.zeros(305 * 100, dtype=cp.float32)

        op.execute(in0, in1, out)

        # Reference: out[m,n] = sum_k in0[m,k] * in1[n,k].
        #   in0 strides: M=1, K=305 -> col-major (M, K) = (305, 72)
        #   in1 strides: N=72, K=1  -> row-major (N, K) = (100, 72)
        #   out strides: M=1, N=305 -> col-major (M, N) = (305, 100)
        in0_mat = in0.reshape(72, 305).T  # (305, 72) = (M, K)
        in1_mat = in1.reshape(100, 72)  # (100, 72) = (N, K)
        # Out(M,N) = In0(M,K) @ In1(N,K)^T = (305,72) @ (72,100)
        expected = in0_mat @ in1_mat.T  # (305, 100) = (M, N)
        # out is col-major (M, N) -> flatten in Fortran order.
        expected_flat = expected.ravel(order="F")

        cp.testing.assert_allclose(out, expected_flat, rtol=1e-3, atol=1e-3)
