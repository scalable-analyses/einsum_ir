"""
Tests for the tileir backend.

End-to-end correctness tests covering all tensor layouts and data types
with backend="tileir" and @pytest.mark.tileir.
"""

import pytest

import etops
from etops.types import DataType


class TestTileIR:
    """End-to-end tests for the tileir backend."""

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

    # =========================================================================
    # Configurations -- tensor contraction layout (acbd,eafb->ecfd)
    # =========================================================================

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
