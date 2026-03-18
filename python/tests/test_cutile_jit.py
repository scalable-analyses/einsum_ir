"""
Tests for cutile JIT compiler make_tensor_view integration.
"""

import pytest
import etops
from etops.backends._cutile.config_parser import ConfigParser

@pytest.mark.cutile
class TestJitCompilerHelpers:
    """Tests for JitCompiler helper methods with cuda.tile."""

    def _make_compiler(self, strides_left, strides_right, strides_output,
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
                (strides_left, strides_right, strides_output),
            ),
        )
        cv = ConfigParser(config, verify_input=False)
        return JitCompiler(cv)

    def test_get_tensor_view_args_filters_stride_zero(self):
        """Test that dimensions with stride 0 are filtered out."""
        compiler = self._make_compiler(
            strides_left=(1, 0, 64),
            strides_right=(0, 128, 1),
            strides_output=(1, 64, 0),
        )
        
        # Left tensor: M and K dims (N stride=0 means not present)
        shape, strides = compiler._get_tensor_view_args(
            compiler.cv.strides_left, "left"
        )
        assert shape == (64, 128)
        assert strides == (1, 64)
        
        # Right tensor: N and K dims (M stride=0 means not present)
        shape, strides = compiler._get_tensor_view_args(
            compiler.cv.strides_right, "right"
        )
        assert shape == (32, 128)
        assert strides == (128, 1)
        
        # Output tensor: M and N dims (K stride=0 means not present)
        shape, strides = compiler._get_tensor_view_args(
            compiler.cv.strides_output, "output"
        )
        assert shape == (64, 32)
        assert strides == (1, 64)

    def test_validate_binary_operation_gemv(self):
        """Test that gemm main primitive passes validation."""
        compiler = self._make_compiler(
            strides_left=(1, 0, 64),
            strides_right=(0, 128, 1),
            strides_output=(1, 64, 0),
        )
        
        # Should not raise for gemm
        compiler._validate_binary_operation()



@pytest.mark.cutile
class TestGeneratedKernelHeader:
    """Tests for the generated kernel header with make_tensor_view calls."""

    def _make_compiler(self, strides_left, strides_right, strides_output,
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
                (strides_left, strides_right, strides_output),
            ),
        )
        cv = ConfigParser(config, verify_input=False)
        return JitCompiler(cv)

    def test_header_contains_make_tensor_view_for_left(self):
        """Test that header contains make_tensor_view call for left tensor."""
        compiler = self._make_compiler(
            strides_left=(1, 0, 64),
            strides_right=(0, 128, 1),
            strides_output=(1, 64, 0),
        )
        
        header = compiler.generate_header_string()
        
        # Should have make_tensor_view for left with filtered dims
        assert "ct.make_tensor_view(left, (64, 128), (1, 64))" in header

    def test_header_contains_make_tensor_view_for_right(self):
        """Test that header contains make_tensor_view call for right tensor."""
        compiler = self._make_compiler(
            strides_left=(1, 0, 64),
            strides_right=(0, 128, 1),
            strides_output=(1, 64, 0),
        )
        
        header = compiler.generate_header_string()
        
        # Should have make_tensor_view for right with filtered dims
        assert "ct.make_tensor_view(right, (32, 128), (128, 1))" in header

    def test_header_contains_make_tensor_view_for_output(self):
        """Test that header contains make_tensor_view call for output tensor."""
        compiler = self._make_compiler(
            strides_left=(1, 0, 64),
            strides_right=(0, 128, 1),
            strides_output=(1, 64, 0),
        )
        
        header = compiler.generate_header_string()
        
        # Should have make_tensor_view for output with filtered dims
        assert "ct.make_tensor_view(output, (64, 32), (1, 64))" in header

    def test_header_structure(self):
        """Test the overall structure of the generated header."""
        compiler = self._make_compiler(
            strides_left=(1, 0, 64),      # M, K (N stride=0)
            strides_right=(0, 128, 1),    # N, K (M stride=0)
            strides_output=(1, 64, 0),    # M, N (K stride=0)
        )
        
        header = compiler.generate_header_string()
        
        # Check structure
        assert "import cuda.tile as ct" in header
        assert "@ct.kernel()" in header
        assert "def contraction_kernel(left, right, output):" in header
        assert "pid = ct.bid(0)" in header
        
        # Check make_tensor_view calls appear after pid
        lines = header.split("\n")
        pid_line_idx = next(i for i, l in enumerate(lines) if "pid = ct.bid(0)" in l)
        left_view_idx = next(i for i, l in enumerate(lines) if "ct.make_tensor_view(left" in l)
        
        # make_tensor_view should come after pid
        assert left_view_idx > pid_line_idx


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
