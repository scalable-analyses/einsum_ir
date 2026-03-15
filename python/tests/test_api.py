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
