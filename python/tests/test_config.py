"""
Tests for etops config module.
"""

import pytest
import json
import tempfile
import os

import etops
from etops.config import TensorOperationConfig
from etops.types import DataType, PrimType, ExecType, DimType


class TestTensorOperationConfig:
    """Tests for TensorOperationConfig dataclass."""

    def test_binary_config_valid(self):
        """Test valid binary contraction configuration."""
        # Column-major GEMM: einsum("km,nk->nm", A, B)
        M, N, K = 64, 32, 128
        config = TensorOperationConfig(
            backend="tpp",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main  = etops.prim.gemm,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  = (M,               N,               K              ),
            strides=(
                        ((1,               0,               M              ),  # in0: km
                         (0,               K,               1              ),  # in1: nk
                         (1,               M,               0              )), # out: nm
            ),
        )
        assert config.backend == "tpp"
        assert config.data_type == DataType.float32
        assert config.prim_main == PrimType.gemm

    def test_unary_config_valid(self):
        """Test valid unary operation configuration."""
        # Matrix transpose: einsum("ij->ji", A)
        config = TensorOperationConfig(
            backend="tpp",
            data_type  = etops.float32,
            prim_first = etops.prim.none,
            prim_main  = etops.prim.copy,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.c,     etops.dim.c    ),
            exec_types = (etops.exec.prim, etops.exec.prim),
            dim_sizes  = (128,             128            ),
            strides=(
                        ((128,             1              ),  # in0: ij
                         (1,               128            )), # out: ji
            ),
        )
        assert config.prim_main == PrimType.copy

    def test_backend_required(self):
        """Test that backend is required."""
        with pytest.raises(TypeError):
            # Missing backend argument
            M, N, K = 64, 32, 128
            TensorOperationConfig(
                data_type=etops.float32,
                prim_first=etops.prim.zero,
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),
                strides=(
                            ((1,               0,               M              ),  # in0: km
                             (0,               K,               1              ),  # in1: nk
                             (1,               M,               0              )), # out: nm
                ),
            )

    def test_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            M, N, K = 64, 32, 128
            TensorOperationConfig(
                backend="invalid",
                data_type=etops.float32,
                prim_first=etops.prim.zero,
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),
                strides=(
                            ((1,               0,               M              ),  # in0: km
                             (0,               K,               1              ),  # in1: nk
                             (1,               M,               0              )), # out: nm
                ),
            )

    def test_invalid_prim_main(self):
        """Test that invalid prim_main raises ValueError."""
        with pytest.raises(ValueError, match="Invalid prim_main"):
            M, N, K = 64, 32, 128
            TensorOperationConfig(
                backend="tpp",
                data_type=etops.float32,
                prim_first=etops.prim.none,
                prim_main  = etops.prim.none,  # Invalid for main
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),
                strides=(
                            ((1,               0,               M              ),  # in0: km
                             (0,               K,               1              ),  # in1: nk
                             (1,               M,               0              )), # out: nm
                ),
            )

    def test_dim_types_length_mismatch(self):
        """Test that mismatched dim_types length raises ValueError."""
        with pytest.raises(ValueError, match="exec_types length"):
            M, N, K = 64, 32, 128
            TensorOperationConfig(
                backend="tpp",
                data_type=etops.float32,
                prim_first=etops.prim.zero,
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim),  # Wrong length
                dim_sizes  = (M,               N,               K              ),
                strides=(
                            ((1,               0,               M              ),  # in0: km
                             (0,               K,               1              ),  # in1: nk
                             (1,               M,               0              )), # out: nm
                ),
            )

    def test_dim_sizes_not_positive(self):
        """Test that non-positive dim_sizes raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            M, N, K = 64, 0, 128
            TensorOperationConfig(
                backend="tpp",
                data_type=etops.float32,
                prim_first=etops.prim.zero,
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),  # N=0 is invalid
                strides=(
                            ((1,               0,               M              ),  # in0: km
                             (0,               K,               1              ),  # in1: nk
                             (1,               M,               0              )), # out: nm
                ),
            )

    def test_strides_empty(self):
        """Test that empty strides raises ValueError."""
        with pytest.raises(ValueError, match="at least one level"):
            M, N, K = 64, 32, 128
            TensorOperationConfig(
                backend="tpp",
                data_type=etops.float32,
                prim_first=etops.prim.zero,
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),
                strides=(),  # Empty
            )

    def test_unary_invalid_dim_type(self):
        """Test that unary operation with non-c dim_type raises ValueError."""
        with pytest.raises(ValueError, match="all dim_types must be etops.dim.c"):
            TensorOperationConfig(
                backend="tpp",
                data_type  = etops.float32,
                prim_first = etops.prim.none,
                prim_main  = etops.prim.copy,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.c    ),  # m is invalid for unary
                exec_types = (etops.exec.prim, etops.exec.prim),
                dim_sizes  = (128,             128            ),
                strides=(
                            ((128,             1              ),  # in0: ij
                             (1,               128            )), # out: ji
                ),
            )

    def test_unary_invalid_prim_first(self):
        """Test that unary operation with non-none prim_first raises ValueError."""
        with pytest.raises(ValueError, match="prim_first must be etops.prim.none"):
            TensorOperationConfig(
                backend="tpp",
                data_type  = etops.float32,
                prim_first = etops.prim.zero,  # Invalid for unary
                prim_main  = etops.prim.copy,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.c,     etops.dim.c    ),
                exec_types = (etops.exec.prim, etops.exec.prim),
                dim_sizes  = (128,             128            ),
                strides=(
                            ((128,             1              ),  # in0: ij
                             (1,               128            )), # out: ji
                ),
            )

    def test_binary_invalid_prim_first(self):
        """Test that binary operation with invalid prim_first raises ValueError."""
        with pytest.raises(ValueError, match="prim_first must be etops.prim.none or"):
            M, N, K = 64, 32, 128
            TensorOperationConfig(
                backend="tpp",
                data_type=etops.float32,
                prim_first=etops.prim.relu,  # Invalid for binary
                prim_main  = etops.prim.gemm,
                prim_last  = etops.prim.none,
                dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
                exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
                dim_sizes  = (M,               N,               K              ),
                strides=(
                            ((1,               0,               M              ),  # in0: km
                             (0,               K,               1              ),  # in1: nk
                             (1,               M,               0              )), # out: nm
                ),
            )


class TestTensorOperationConfigSerialization:
    """Tests for TensorOperationConfig JSON serialization."""

    def test_to_json(self):
        """Test serialization to JSON."""
        M, N, K = 64, 32, 128
        config = TensorOperationConfig(
            backend="tpp",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main  = etops.prim.gemm,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  = (M,               N,               K              ),
            strides=(
                        ((1,               0,               M              ),  # in0: km
                         (0,               K,               1              ),  # in1: nk
                         (1,               M,               0              )), # out: nm
            ),
        )
        json_str = config.to_json()
        data = json.loads(json_str)
        
        assert data["backend"] == "tpp"
        assert data["data_type"] == "float32"
        assert data["prim_first"] == "zero"
        assert data["prim_main"] == "gemm"
        assert data["prim_last"] == "none"
        assert data["dim_types"] == ["m", "n", "k"]
        assert data["dim_sizes"] == [M, N, K]

    def test_from_json(self):
        """Test deserialization from JSON."""
        M, N, K = 64, 32, 128
        json_str = json.dumps({
            "backend": "tpp",
            "data_type": "float32",
            "prim_first": "zero",
            "prim_main": "gemm",
            "prim_last": "none",
            "dim_types": ["m", "n", "k"],
            "exec_types": ["prim", "prim", "prim"],
            "dim_sizes": [M, N, K],
            "strides": [[[1, 0, M], [0, K, 1], [1, M, 0]]],
        })
        config = TensorOperationConfig.from_json(json_str)
        
        assert config.backend == "tpp"
        assert config.data_type == DataType.float32
        assert config.prim_main == PrimType.gemm
        assert config.dim_types == (DimType.m, DimType.n, DimType.k)

    def test_from_json_missing_field(self):
        """Test deserialization with missing field raises ValueError."""
        json_str = json.dumps({
            "backend": "tpp",
            "data_type": "float32",
            # Missing other fields
        })
        with pytest.raises(ValueError, match="Missing required fields"):
            TensorOperationConfig.from_json(json_str)

    def test_save_and_load(self):
        """Test save and load round-trip."""
        M, N, K = 64, 32, 128
        config = TensorOperationConfig(
            backend="tpp",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main  = etops.prim.gemm,
            prim_last  = etops.prim.none,
            dim_types  = (etops.dim.m,     etops.dim.n,     etops.dim.k    ),
            exec_types = (etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes  = (M,               N,               K              ),
            strides=(
                        ((1,               0,               M              ),  # in0: km
                         (0,               K,               1              ),  # in1: nk
                         (1,               M,               0              )), # out: nm
            ),
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded = TensorOperationConfig.load(f.name)
            os.unlink(f.name)
        
        assert loaded.backend == config.backend
        assert loaded.data_type == config.data_type
        assert loaded.prim_main == config.prim_main
        assert loaded.dim_sizes == config.dim_sizes
