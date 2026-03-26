"""
Tests for cutile JIT compiler make_tensor_view integration.
"""

import pytest
import etops
from etops.backends._cutile.config_parser import ConfigParser

@pytest.mark.cutile
class TestJitCompilerHelpers:
    """Tests for JitCompiler helper methods with cuda.tile."""

    def _make_compiler(self, strides_in0, strides_in1, strides_out,
                       dim_sizes=(64, 32, 128)):
        """Create a JitCompiler instance for testing."""
        from etops.backends._cutile.jit_compiler import JitCompiler
        
        config = etops.TensorOperationConfig(
            backend="cutile",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main=etops.prim.gemm,
            prim_last=etops.prim.none,
            dim_types=(etops.dim.m, etops.dim.n, etops.dim.k),
            exec_types=(etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes=dim_sizes,
            strides=(
                (strides_in0, strides_in1, strides_out),
            ),
        )
        cv = ConfigParser(config, verify_input=False)
        return JitCompiler(cv)

    def test_get_tensor_view_args_filters_stride_zero(self):
        """Test that dimensions with stride 0 are filtered out and sorted by stride descending."""
        compiler = self._make_compiler(
            strides_in0=(1, 0, 64),
            strides_in1=(0, 128, 1),
            strides_out=(1, 64, 0),
        )
        
        # in0 tensor: M and K dims (N stride=0 means not present)
        # Stride-sorted: K (stride 64) > M (stride 1), so shape=(128, 64), strides=(64, 1)
        shape, strides = compiler._get_tensor_view_args(
            compiler.cv.stride_sorted_indices_in0, compiler.cv.strides_in0, "in0"
        )
        assert shape == (128, 64)
        assert strides == (64, 1)
        
        # in1 tensor: N and K dims (M stride=0 means not present)
        # Stride-sorted: N (stride 128) > K (stride 1), so shape=(32, 128), strides=(128, 1)
        shape, strides = compiler._get_tensor_view_args(
            compiler.cv.stride_sorted_indices_in1, compiler.cv.strides_in1, "in1"
        )
        assert shape == (32, 128)
        assert strides == (128, 1)
        
        # out tensor: M and N dims (K stride=0 means not present)
        # Stride-sorted: N (stride 64) > M (stride 1), so shape=(32, 64), strides=(64, 1)
        shape, strides = compiler._get_tensor_view_args(
            compiler.cv.stride_sorted_indices_out, compiler.cv.strides_out, "out"
        )
        assert shape == (32, 64)
        assert strides == (64, 1)

    def test_validate_binary_operation_gemv(self):
        """Test that gemm main primitive passes validation."""
        compiler = self._make_compiler(
            strides_in0=(1, 0, 64),
            strides_in1=(0, 128, 1),
            strides_out=(1, 64, 0),
        )
        
        # Should not raise for gemm
        compiler._validate_binary_operation()


@pytest.mark.cutile
class TestConfigParserSortedIndices:
    """Tests for ConfigParser stride-sorted index derivation."""

    def _make_parser(self, strides_in0, strides_in1, strides_out,
                     dim_sizes=(64, 32, 128)):
        """Create a ConfigParser instance for testing."""
        config = etops.TensorOperationConfig(
            backend="cutile",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main=etops.prim.gemm,
            prim_last=etops.prim.none,
            dim_types=(etops.dim.m, etops.dim.n, etops.dim.k),
            exec_types=(etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes=dim_sizes,
            strides=(
                (strides_in0, strides_in1, strides_out),
            ),
        )
        return ConfigParser(config, verify_input=False)

    def test_stride_sorted_indices_basic(self):
        """Test that stride-sorted indices are derived correctly."""
        cv = self._make_parser(
            strides_in0=(1, 0, 64),    # M=1, N=0, K=64 -> sorted: K, M
            strides_in1=(0, 128, 1),   # M=0, N=128, K=1 -> sorted: N, K
            strides_out=(1, 64, 0),    # M=1, N=64, K=0 -> sorted: N, M
        )
        
        # in0: config indices [0, 2] (stride 0 filtered), stride-sorted [2, 0]
        assert cv.config_indices_in_in0 == [0, 2]
        assert cv.stride_sorted_indices_in0 == [2, 0]  # K (stride 64) > M (stride 1)
        
        # in1: config indices [1, 2] (stride 0 filtered), stride-sorted [1, 2]
        assert cv.config_indices_in_in1 == [1, 2]
        assert cv.stride_sorted_indices_in1 == [1, 2]  # N (stride 128) > K (stride 1)
        
        # out: config indices [0, 1] (stride 0 filtered), stride-sorted [1, 0]
        assert cv.config_indices_in_out == [0, 1]
        assert cv.stride_sorted_indices_out == [1, 0]  # N (stride 64) > M (stride 1)

    def test_stride_sorted_indices_equal_strides_stable(self):
        """Test that equal strides preserve config order (stable sort)."""
        cv = self._make_parser(
            strides_in0=(64, 0, 64),   # M=64, N=0, K=64 -> sorted: M, K (stable, M first in config)
            strides_in1=(0, 128, 128), # M=0, N=128, K=128 -> sorted: N, K (stable, N first in config)
            strides_out=(32, 32, 0),   # M=32, N=32, K=0 -> sorted: M, N (stable, M first in config)
        )
        
        # in0: config indices [0, 2], both stride 64, stable sort keeps config order
        assert cv.stride_sorted_indices_in0 == [0, 2]
        
        # in1: config indices [1, 2], both stride 128, stable sort keeps config order
        assert cv.stride_sorted_indices_in1 == [1, 2]
        
        # out: config indices [0, 1], both stride 32, stable sort keeps config order
        assert cv.stride_sorted_indices_out == [0, 1]

    def test_prim_stride_sorted_indices(self):
        """Test that prim stride-sorted indices are derived correctly."""
        cv = self._make_parser(
            strides_in0=(1, 0, 64),
            strides_in1=(0, 128, 1),
            strides_out=(1, 64, 0),
        )
        
        # All dimensions are prim, so prim lists should match stride-sorted lists
        assert cv.prim_stride_sorted_indices_in0 == cv.stride_sorted_indices_in0
        assert cv.prim_stride_sorted_indices_in1 == cv.stride_sorted_indices_in1
        assert cv.prim_stride_sorted_indices_out == cv.stride_sorted_indices_out


@pytest.mark.cutile
class TestGeneratedKernelHeader:
    """Tests for the generated kernel header with make_tensor_view calls."""

    def _make_compiler(self, strides_in0, strides_in1, strides_out,
                       dim_sizes=(64, 32, 128)):
        """Create a JitCompiler instance for testing."""
        from etops.backends._cutile.jit_compiler import JitCompiler
        
        config = etops.TensorOperationConfig(
            backend="cutile",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main=etops.prim.gemm,
            prim_last=etops.prim.none,
            dim_types=(etops.dim.m, etops.dim.n, etops.dim.k),
            exec_types=(etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes=dim_sizes,
            strides=(
                (strides_in0, strides_in1, strides_out),
            ),
        )
        cv = ConfigParser(config, verify_input=False)
        return JitCompiler(cv)

    def test_header_contains_make_tensor_view_for_in0(self):
        """Test that header contains make_tensor_view call for in0 tensor."""
        compiler = self._make_compiler(
            strides_in0=(1, 0, 64),
            strides_in1=(0, 128, 1),
            strides_out=(1, 64, 0),
        )
        
        header = compiler.generate_header_string()
        
        # Should have make_tensor_view for in0 with stride-sorted dims
        # in0: K (stride 64) > M (stride 1), so shape=(128, 64), strides=(64, 1)
        assert "ct.make_tensor_view(in0, (128, 64), (64, 1))" in header

    def test_header_contains_make_tensor_view_for_in1(self):
        """Test that header contains make_tensor_view call for in1 tensor."""
        compiler = self._make_compiler(
            strides_in0=(1, 0, 64),
            strides_in1=(0, 128, 1),
            strides_out=(1, 64, 0),
        )
        
        header = compiler.generate_header_string()
        
        # Should have make_tensor_view for in1 with stride-sorted dims
        # in1: N (stride 128) > K (stride 1), so shape=(32, 128), strides=(128, 1)
        assert "ct.make_tensor_view(in1, (32, 128), (128, 1))" in header

    def test_header_contains_make_tensor_view_for_out(self):
        """Test that header contains make_tensor_view call for out tensor."""
        compiler = self._make_compiler(
            strides_in0=(1, 0, 64),
            strides_in1=(0, 128, 1),
            strides_out=(1, 64, 0),
        )
        
        header = compiler.generate_header_string()
        
        # Should have make_tensor_view for out with stride-sorted dims
        # out: N (stride 64) > M (stride 1), so shape=(32, 64), strides=(64, 1)
        assert "ct.make_tensor_view(out, (32, 64), (64, 1))" in header

    def test_header_structure(self):
        """Test the overall structure of the generated header."""
        compiler = self._make_compiler(
            strides_in0=(1, 0, 64),     # M, K (N stride=0)
            strides_in1=(0, 128, 1),    # N, K (M stride=0)
            strides_out=(1, 64, 0),     # M, N (K stride=0)
        )
        
        header = compiler.generate_header_string()
        
        # Check structure
        assert "import cuda.tile as ct" in header
        assert "@ct.kernel()" in header
        assert "def contraction_kernel(in0, in1, out):" in header
        assert "pid = ct.bid(0)" in header
        
        # Check make_tensor_view calls appear after pid
        lines = header.split("\n")
        pid_line_idx = next(i for i, l in enumerate(lines) if "pid = ct.bid(0)" in l)
        in0_view_idx = next(i for i, l in enumerate(lines) if "ct.make_tensor_view(in0" in l)
        
        # make_tensor_view should come after pid
        assert in0_view_idx > pid_line_idx


@pytest.mark.cutile
class TestEndToEndKernelGeneration:
    """End-to-end tests for kernel generation with make_tensor_view."""

    def test_jit_kernel_generates_valid_code(self):
        """Test that jit_kernel() generates valid Python code."""
        from etops.backends._cutile.jit_compiler import JitCompiler
        
        config = etops.TensorOperationConfig(
            backend="cutile",
            data_type=etops.float32,
            prim_first=etops.prim.zero,
            prim_main=etops.prim.gemm,
            prim_last=etops.prim.none,
            dim_types=(etops.dim.m, etops.dim.n, etops.dim.k),
            exec_types=(etops.exec.prim, etops.exec.prim, etops.exec.prim),
            dim_sizes=(64, 32, 128),
            strides=(
                ((1, 0, 64), (0, 128, 1), (1, 64, 0)),
            ),
        )
        cv = ConfigParser(config, verify_input=False)
        compiler = JitCompiler(cv)
        
        # Generate the kernel
        compiler.jit_kernel()
        
        # Check that the kernel string is non-empty and contains expected elements
        kernel = compiler.string_kernel
        assert len(kernel) > 0
        assert "import cuda.tile as ct" in kernel
        assert "ct.make_tensor_view" in kernel
