"""
Cutile backend implementation using cuda.tile.

This backend provides GPU-accelerated tensor operations using NVIDIA's
cuda.tile framework. It is optional and requires cuda.tile to be installed.

Kernel JIT compilation and execution are implemented; the optimizer is not
yet implemented (etops.optimize() will raise NotImplementedError for this backend).
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from etops.config import TensorOperationConfig
    from etops.backends.base import CompiledOperation

from etops.backends._cutile.config_parser import ConfigParser
from etops.backends._cutile.jit_compiler import JitCompiler
from etops.backends._cutile.cache import InMemoryCache


# Module-level cache: persists across etops.compile() calls within the same process.
# Kernels with identical configurations are compiled only once.
_global_cache = InMemoryCache()


class CutileOperation:
    """
    Compiled cutile operation.

    This class wraps a compiled cuda.tile kernel and provides the execute()
    method required by the CompiledOperation protocol.
    """

    def __init__(self, kernel_module, grid_size: int):
        """
        Initialize cutile operation.

        Args:
            kernel_module: Compiled kernel module from cuda.tile
            grid_size: Grid size for kernel launch
        """
        self._kernel = kernel_module
        self._grid_size = grid_size

    def execute(
        self,
        in0,
        in1: Optional[object],
        out,
    ) -> None:
        """
        Execute the tensor operation on GPU.

        Args:
            in0: First input tensor (must be on CUDA device)
            in1: Second input tensor (None for unary operations)
            out: Output tensor (must be pre-allocated on CUDA device)
        """
        import cupy as cp
        import cuda.tile as ct

        ct.launch(
            cp.cuda.get_current_stream(),
            (self._grid_size,),
            self._kernel.contraction_kernel,
            (in0, in1, out),
        )


def create_operation(config: "TensorOperationConfig") -> CutileOperation:
    """
    Create and compile a cutile operation from configuration.

    Args:
        config: Tensor operation configuration

    Returns:
        Compiled CutileOperation instance

    Raises:
        ValueError: If config is not executable (e.g. missing prim exec_types)
        ImportError: If cuda.tile or cupy is not installed
    """
    config_parser = ConfigParser(config)
    config_parser.verify_executable_config()

    jit = JitCompiler(config_parser)
    kernel_module = _global_cache.get_or_compile(config_parser, jit)

    return CutileOperation(kernel_module, config_parser.grid_size)


def get_default_optimization_config() -> dict:
    """
    Get default optimization configuration for cutile backend.

    Returns:
        Dictionary with optimization parameters

    Raises:
        NotImplementedError: cuTile optimizer is not yet implemented
    """
    raise NotImplementedError(
        "cuTile backend optimization is not yet implemented. "
        "Assign exec_types manually or use backend='tpp' for now."
    )


def optimize_config(
    config: "TensorOperationConfig",
    optimization_config: Optional[dict] = None
) -> "TensorOperationConfig":
    """
    Optimize a tensor operation configuration for cutile backend.

    Args:
        config: Tensor operation configuration to optimize
        optimization_config: Cutile-specific optimization parameters

    Returns:
        Optimized TensorOperationConfig

    Raises:
        NotImplementedError: cuTile optimizer is not yet implemented
    """
    raise NotImplementedError(
        "cuTile backend optimization is not yet implemented. "
        "Assign exec_types manually or use backend='tpp' for now."
    )


__all__ = [
    "create_operation",
    "CutileOperation",
    "get_default_optimization_config",
    "optimize_config",
]
