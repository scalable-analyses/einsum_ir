"""
Tests for etops types module.
"""

import pytest


def _bindings_available():
    """Check if C++ bindings are compiled and available."""
    try:
        import etops._etops_core
        return True
    except ImportError:
        return False


# Skip all enum consistency tests if bindings are not available
pytestmark = pytest.mark.skipif(
    not _bindings_available(),
    reason="C++ bindings not compiled"
)


class TestEnumConsistency:
    """Verify Python enum values match C++ bindings."""

    def test_datatype_matches_bindings(self):
        """DataType values must match C++ dtype_t enum."""
        from etops._etops_core import DataType as CppDataType
        from etops.types import DataType

        assert DataType.float32.value == int(CppDataType.float32)
        assert DataType.float64.value == int(CppDataType.float64)

    def test_primtype_matches_bindings(self):
        """PrimType values must match C++ prim_t enum."""
        from etops._etops_core import PrimType as CppPrimType
        from etops.types import PrimType

        for name in ["none", "zero", "copy", "relu", "gemm", "brgemm"]:
            assert getattr(PrimType, name).value == int(getattr(CppPrimType, name))

    def test_exectype_matches_bindings(self):
        """ExecType values must match C++ exec_t enum."""
        from etops._etops_core import ExecType as CppExecType
        from etops.types import ExecType

        for name in ["seq", "prim", "shared", "sfc"]:
            assert getattr(ExecType, name).value == int(getattr(CppExecType, name))

    def test_dimtype_matches_bindings(self):
        """DimType values must match C++ dim_t enum."""
        from etops._etops_core import DimType as CppDimType
        from etops.types import DimType

        for name in ["c", "m", "n", "k"]:
            assert getattr(DimType, name).value == int(getattr(CppDimType, name))

    def test_errortype_matches_bindings(self):
        """ErrorType values must match C++ error_t enum."""
        from etops._etops_core import ErrorType as CppErrorType
        from etops.types import ErrorType

        for name in ["success", "compilation_failed", "invalid_stride_shape",
                     "invalid_optimization_config"]:
            assert getattr(ErrorType, name).value == int(getattr(CppErrorType, name))


class TestNamespaceAliases:
    """Tests for namespace convenience classes."""

    def test_prim_namespace(self):
        """prim namespace provides access to PrimType values."""
        from etops.types import prim, PrimType

        assert prim.none == PrimType.none
        assert prim.zero == PrimType.zero
        assert prim.copy == PrimType.copy
        assert prim.relu == PrimType.relu
        assert prim.gemm == PrimType.gemm
        assert prim.brgemm == PrimType.brgemm

    def test_dim_namespace(self):
        """dim namespace provides access to DimType values."""
        from etops.types import dim, DimType

        assert dim.c == DimType.c
        assert dim.m == DimType.m
        assert dim.n == DimType.n
        assert dim.k == DimType.k

    def test_exec_namespace(self):
        """exec namespace provides access to ExecType values."""
        from etops.types import exec, ExecType

        assert exec.seq == ExecType.seq
        assert exec.prim == ExecType.prim
        assert exec.shared == ExecType.shared
        assert exec.sfc == ExecType.sfc


class TestConvenienceAliases:
    """Tests for module-level convenience aliases."""

    def test_dtype_alias(self):
        """dtype is an alias for DataType."""
        from etops.types import dtype, DataType
        assert dtype is DataType

    def test_float32_alias(self):
        """float32 alias matches DataType.float32."""
        from etops.types import float32, DataType
        assert float32 == DataType.float32

    def test_float64_alias(self):
        """float64 alias matches DataType.float64."""
        from etops.types import float64, DataType
        assert float64 == DataType.float64
