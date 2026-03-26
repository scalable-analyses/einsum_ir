"""
Tests for etops public API.
"""

import pytest
import numpy as np

import etops
from etops.types import DataType, PrimType, ExecType, DimType


class TestPublicAPI:
    """Tests for public API exports."""

    def test_compile_exists(self):
        """Test that compile function is exported."""
        assert hasattr(etops, "compile")
        assert callable(etops.compile)

    def test_optimize_exists(self):
        """Test that optimize function is exported."""
        assert hasattr(etops, "optimize")
        assert callable(etops.optimize)

    def test_list_backends_exists(self):
        """Test that list_backends function is exported."""
        assert hasattr(etops, "list_backends")
        assert callable(etops.list_backends)

    def test_tensor_operation_config_exists(self):
        """Test that TensorOperationConfig is exported."""
        assert hasattr(etops, "TensorOperationConfig")

    def test_types_exported(self):
        """Test that type enums are exported."""
        assert hasattr(etops, "DataType")
        assert hasattr(etops, "PrimType")
        assert hasattr(etops, "ExecType")
        assert hasattr(etops, "DimType")
        assert hasattr(etops, "ErrorType")

    def test_namespaces_exported(self):
        """Test that convenience namespaces are exported."""
        assert hasattr(etops, "prim")
        assert hasattr(etops, "dim")
        assert hasattr(etops, "exec")

    def test_aliases_exported(self):
        """Test that convenience aliases are exported."""
        assert hasattr(etops, "dtype")
        assert hasattr(etops, "float32")
        assert hasattr(etops, "float64")


class TestListBackends:
    """Tests for list_backends function."""

    def test_returns_list(self):
        """Test that list_backends returns a list."""
        result = etops.list_backends()
        assert isinstance(result, list)

    def test_includes_tpp(self):
        """Test that TPP backend is always available."""
        backends = etops.list_backends()
        assert "tpp" in backends


class TestCompile:
    """Tests for compile function."""

    def test_compile_with_tpp_backend(self):
        """Test compiling with TPP backend."""
        # Column-major GEMM: einsum("km,nk->nm", A, B)
        M, N, K = 64, 32, 128
        config = etops.TensorOperationConfig(
            backend="tpp",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main  = etops.prim.gemm,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  = (M,               N,               K              ),
            strides=(
                        ((1,               0,               M              ),  #in0: km
                         (0,               K,               1              ),  #in1: nk
                         (1,               M,               0              )), #out: nm
            ),
        )
        op = etops.compile(config)
        assert hasattr(op, "execute")
        assert callable(op.execute)

    def test_compile_with_unknown_backend(self):
        """Test that unknown backend raises ValueError."""
        # We need to create a config with invalid backend
        # This is caught by TensorOperationConfig validation
        with pytest.raises(ValueError, match="Unsupported backend"):
            M, N, K = 64, 32, 128
            etops.TensorOperationConfig(
                backend="unknown",
                data_type=etops.float32,
                prim_first=etops.prim.zero,
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),
                strides=(
                            ((1,               0,               M              ),  #in0: km
                             (0,               K,               1              ),  #in1: nk
                             (1,               M,               0              )), #out: nm
                ),
            )


class TestOptimize:
    """Tests for optimize function."""

    def test_optimize_tpp(self):
        """Test optimizing TPP configuration."""
        # Column-major GEMM: einsum("km,nk->nm", A, B)
        M, N, K = 64, 32, 128
        config = etops.TensorOperationConfig(
            backend="tpp",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main  = etops.prim.gemm,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  = (M,               N,               K              ),
            strides=(
                        ((1,               0,               M              ),  #in0: km
                         (0,               K,               1              ),  #in1: nk
                         (1,               M,               0              )), #out: nm
            ),
        )
        optimized = etops.optimize(config)
        assert optimized.backend == "tpp"
        assert optimized.data_type == etops.float32


class TestEndToEnd:
    """End-to-end tests for tensor operations."""

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

    @pytest.mark.tpp
    def test_gemm_execution(self):
        """Test GEMM execution through the full pipeline."""
        # Column-major GEMM: einsum("km,nk->nm", A, B)
        M, N, K = 64, 32, 128
        
        config = etops.TensorOperationConfig(
            backend="tpp",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main  = etops.prim.gemm,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  = (M,               N,               K              ),
            strides=(
                        ((1,               0,               M              ),  #in0: km
                         (0,               K,               1              ),  #in1: nk
                         (1,               M,               0              )), #out: nm
            ),
        )
        
        # Compile
        op = etops.compile(config)
        
        # Create tensors
        in0 = np.random.randn(128, 64).astype(np.float32)
        in1 = np.random.randn(32, 128).astype(np.float32)
        out = np.random.randn(32,  64).astype(np.float32)
        
        # Execute
        op.execute(in0, in1, out)
        
        # Verify result
        expected = np.einsum("km,nk->nm", in0, in1)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.tpp
    def test_unary_copy_execution(self):
        """Test unary copy execution through the full pipeline."""
        # Matrix transpose: einsum("ij->ji", A)
        config = etops.TensorOperationConfig(
            backend="tpp",
            data_type  = etops.float32,
            prim_first = etops.prim.none,
            prim_main  = etops.prim.copy,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.c,     etops.dim.c    ),
            exec_types = (etops.exec.prim, etops.exec.prim),
            dim_sizes  = (3,               4              ),
            strides=(
                        ((4,               1              ),  # in0: ij
                         (1,               3              )), # out: ji
            ),
        )
        
        # Compile
        op = etops.compile(config)
        
        # Create tensors
        in0 = np.random.randn(3, 4).astype(np.float32)
        out = np.random.randn(4, 3).astype(np.float32)
        
        # Execute
        op.execute(in0, None, out)
        
        # Verify result (transpose)
        expected = np.einsum("ij->ji", in0)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    CUTILE_GEMM_SEQ = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_GEMM_SEQ_PRIM = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_GEMM_SEQ_MULTI_PRIM = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_GEMM_SHARED_PRIM = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_TC_SEQ = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_TC_SEQ_PRIM = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_TC_SEQ_MULTI_PRIM = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    CUTILE_TC_SHARED = etops.TensorOperationConfig(
        backend    =   "cutile",
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

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_GEMM_SEQ_PRIM,
        CUTILE_GEMM_SEQ_MULTI_PRIM,
        CUTILE_GEMM_SHARED_PRIM
    ])
    def test_cutile_execute_gemm_fp32(self, config):
        """Test GEMM execution in cutile: einsum('abcd,efab->efcd', A, B).

        The dimension sizes are: |a|=48, |b|=16, |c|=4, |d|=32, |e|=2, |f|=8.
        
        This is a GEMM where tensors are viewed as matrices:
        - in0 is shaped as (a*b, c*d) = (768, 128) matrix KM
        - in1 is shaped as (e*f, a*b) = (16, 768) matrix NK
        - out is shaped as (e*f, c*d) = (16, 128) matrix NM
        """
        import cupy as cp
        
        # Compile
        config = self._config_with_dtype(config, etops.float32)
        op = etops.compile(config)
        
        # Create tensors on GPU
        in0 = cp.random.randn(768, 128).astype(cp.float32)
        in1 = cp.random.randn(16, 768).astype(cp.float32)
        out = cp.random.randn(16, 128).astype(cp.float32)
        
        # Execute
        op.execute(in0, in1, out)
        
        # Verify result using cupy
        expected = cp.matmul(in1, in0)
        
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_TC_SEQ_PRIM,
        CUTILE_TC_SEQ_MULTI_PRIM,
        CUTILE_TC_SHARED
    ])
    def test_cutile_execute_tc_fp32(self, config):
        """Test tensor contraction in cutile: einsum('acbd,eafb->ecfd', A, B).

        The dimension sizes are: |a|=48, |b|=16, |c|=4, |d|=32, |e|=2, |f|=8.
        """
        import cupy as cp

        # Compile
        config = self._config_with_dtype(config, etops.float32)
        op = etops.compile(config)
        
        # Create tensors on GPU
        in0 = cp.random.randn(48,4,16,32).astype(cp.float32)
        in1 = cp.random.randn(2,48,8,16).astype(cp.float32)
        out = cp.random.randn(2,4,8,32).astype(cp.float32)
        
        # Execute
        op.execute(in0, in1, out)
        
        # Verify result using cupy
        expected = cp.einsum( 'acbd,eafb->ecfd', in0, in1 )
        
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Float64 tests
    # =========================================================================

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_GEMM_SEQ_PRIM,
        CUTILE_GEMM_SEQ_MULTI_PRIM,
        CUTILE_GEMM_SHARED_PRIM
    ])
    def test_cutile_execute_gemm_fp64(self, config):
        """Test GEMM execution in cutile with float64."""
        import cupy as cp
        
        config = self._config_with_dtype(config, etops.float64)
        op = etops.compile(config)
        
        in0 = cp.random.randn(768, 128).astype(cp.float64)
        in1 = cp.random.randn(16, 768).astype(cp.float64)
        out = cp.random.randn(16, 128).astype(cp.float64)
        
        op.execute(in0, in1, out)
        
        expected = cp.matmul(in1, in0)
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_TC_SEQ_PRIM,
        CUTILE_TC_SEQ_MULTI_PRIM,
        CUTILE_TC_SHARED
    ])
    def test_cutile_execute_tc_fp64(self, config):
        """Test tensor contraction in cutile with float64."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float64)
        op = etops.compile(config)
        
        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float64)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float64)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float64)
        
        op.execute(in0, in1, out)
        
        expected = cp.einsum('acbd,eafb->ecfd', in0, in1)
        cp.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    # =========================================================================
    # Float16 tests
    # =========================================================================

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_GEMM_SEQ_PRIM,
        CUTILE_GEMM_SEQ_MULTI_PRIM,
        CUTILE_GEMM_SHARED_PRIM
    ])
    def test_cutile_execute_gemm_fp16(self, config):
        """Test GEMM execution in cutile with float16."""
        import cupy as cp
        
        config = self._config_with_dtype(config, etops.float16)
        op = etops.compile(config)
        
        in0 = cp.random.randn(768, 128).astype(cp.float16)
        in1 = cp.random.randn(16, 768).astype(cp.float16)
        out = cp.random.randn(16, 128).astype(cp.float16)
        
        op.execute(in0, in1, out)
        
        # Compare against float32 reference
        expected = cp.matmul(in1.astype(cp.float32), in0.astype(cp.float32))
        cp.testing.assert_allclose(out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_GEMM_SEQ
    ])
    def test_cutile_optimize_and_execute_gemm_fp16(self, config):
        """Test GEMM optimization and execution in cutile with float16."""
        import cupy as cp
        
        # Compile
        config = self._config_with_dtype(config, etops.float16)
        opt_config = etops.get_default_optimization_config("cutile")
        config = etops.optimize( config, opt_config )

        op = etops.compile(config)
        
        # Create tensors on GPU
        in0 = cp.random.randn(768, 128).astype(cp.float16)
        in1 = cp.random.randn(16, 768).astype(cp.float16)
        out = cp.random.randn(16, 128).astype(cp.float16)
        
        # Execute
        op.execute(in0, in1, out)
        
        # Verify result using cupy
        expected = cp.matmul(in1, in0)
        
        cp.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_TC_SEQ_PRIM,
        CUTILE_TC_SEQ_MULTI_PRIM,
        CUTILE_TC_SHARED
    ])
    def test_cutile_execute_tc_fp16(self, config):
        """Test tensor contraction in cutile with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        op = etops.compile(config)
        
        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float16)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float16)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float16)
        
        op.execute(in0, in1, out)
        
        # Compare against float32 reference
        expected = cp.einsum('acbd,eafb->ecfd',
                             in0.astype(cp.float32),
                             in1.astype(cp.float32))
        cp.testing.assert_allclose(out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_TC_SEQ
    ])
    def test_cutile_execute_tc_fp16(self, config):
        """Test tensor contraction in cutile with float16."""
        import cupy as cp

        config = self._config_with_dtype(config, etops.float16)
        opt_config = etops.get_default_optimization_config("cutile")
        config = etops.optimize(config, opt_config)
        op = etops.compile(config)
        
        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float16)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float16)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float16)
        
        op.execute(in0, in1, out)
        
        # Compare against float32 reference
        expected = cp.einsum('acbd,eafb->ecfd',
                             in0.astype(cp.float32),
                             in1.astype(cp.float32))
        cp.testing.assert_allclose(out.astype(cp.float32), expected, rtol=1e-2, atol=1e-2)

    # =========================================================================
    # Bfloat16 tests
    # =========================================================================

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_GEMM_SEQ_PRIM,
        CUTILE_GEMM_SEQ_MULTI_PRIM,
        CUTILE_GEMM_SHARED_PRIM
    ])
    def test_cutile_execute_gemm_bf16(self, config):
        """Test GEMM execution in cutile with bfloat16.
        
        Uses PyTorch for bfloat16 support since cuda.tile has better support
        for torch.bfloat16 tensors than CuPy arrays with ml_dtypes.bfloat16.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available for bfloat16 support")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bfloat16 test")
        
        config = self._config_with_dtype(config, etops.bfloat16)
        op = etops.compile(config)
        
        in0 = torch.randn(768, 128, dtype=torch.bfloat16, device='cuda')
        in1 = torch.randn(16, 768, dtype=torch.bfloat16, device='cuda')
        out = torch.randn(16, 128, dtype=torch.float32, device='cuda')
        
        op.execute(in0, in1, out)
        
        # Compare against float32 reference
        expected = torch.matmul(in1.float(), in0.float())
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_TC_SEQ_PRIM,
        CUTILE_TC_SEQ_MULTI_PRIM,
        CUTILE_TC_SHARED
    ])
    def test_cutile_execute_tc_bf16(self, config):
        """Test tensor contraction in cutile with bfloat16.
        
        Uses PyTorch for bfloat16 support since cuda.tile has better support
        for torch.bfloat16 tensors than CuPy arrays with ml_dtypes.bfloat16.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available for bfloat16 support")
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for bfloat16 test")

        config = self._config_with_dtype(config, etops.bfloat16)
        op = etops.compile(config)
        
        in0 = torch.randn(48, 4, 16, 32, dtype=torch.bfloat16, device='cuda')
        in1 = torch.randn(2, 48, 8, 16, dtype=torch.bfloat16, device='cuda')
        out = torch.randn(2, 4, 8, 32, dtype=torch.float32, device='cuda')
        
        op.execute(in0, in1, out)
        
        # Compare against float32 reference using einsum
        expected = torch.einsum('acbd,eafb->ecfd', in0.float(), in1.float())
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    # =========================================================================
    # TensorFloat-32 tests
    # =========================================================================

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_GEMM_SEQ_PRIM,
        CUTILE_GEMM_SEQ_MULTI_PRIM,
        CUTILE_GEMM_SHARED_PRIM
    ])
    def test_cutile_execute_gemm_tf32(self, config):
        """Test GEMM execution in cutile with tensorfloat32."""
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

    @pytest.mark.cutile
    @pytest.mark.parametrize("config", [
        CUTILE_TC_SEQ_PRIM,
        CUTILE_TC_SEQ_MULTI_PRIM,
        CUTILE_TC_SHARED
    ])
    def test_cutile_execute_tc_tf32(self, config):
        """Test tensor contraction in cutile with tensorfloat32. """
        import cupy as cp

        config = self._config_with_dtype(config, etops.tfloat32)
        op = etops.compile(config)
        
        # TF32 uses float32 storage
        in0 = cp.random.randn(48, 4, 16, 32).astype(cp.float32)
        in1 = cp.random.randn(2, 48, 8, 16).astype(cp.float32)
        out = cp.random.randn(2, 4, 8, 32).astype(cp.float32)
        
        op.execute(in0, in1, out)
        
        expected = cp.einsum('acbd,eafb->ecfd', in0, in1)
        cp.testing.assert_allclose(out, expected, rtol=1e-2, atol=1e-2)