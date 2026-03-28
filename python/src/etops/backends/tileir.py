"""
TileIR backend implementation for etops.

This backend generates GPU tensor kernels by directly constructing TileIR
object graphs (``cuda.tile._ir.ops.*`` classes), bypassing the
``@ct.kernel`` Python frontend.  Compiled cubins are launched via CuPy
``RawModule``.

Supports:
  * Binary contractions: GEMM and BRGEMM (3-pointer ABI: in0, in1, out).
  * Unary operations: copy, zero, relu (2-pointer ABI: in0, out).
  * Data types: FP32, FP64, FP16, BF16, TF32.
  * Exec types: shared (grid-mapped), sequential (for-loops), prim (tile).
  * ``prim_first``: ``zero`` (beta=0) or ``none`` (beta=1) [binary only].
  * ``prim_last``: ``none`` or ``relu`` [binary only].
"""

from __future__ import annotations

__all__ = [
    "create_operation",
    "TileIROperation",
    "get_default_optimization_config",
    "optimize_config",
]

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from etops.config import TensorOperationConfig

_logger = logging.getLogger(__name__)


# =============================================================================
# Compiled operation wrapper
# =============================================================================


class TileIROperation:
    """Compiled tileir operation.

    Wraps a :class:`~etops.backends._tileir.compiler.TileIRKernel` and
    provides the ``execute()`` method required by the
    :class:`~etops.backends.base.CompiledOperation` protocol.
    """

    def __init__(self, kernel: object, grid_size: int) -> None:
        """Initialize.

        Args:
            kernel: A :class:`TileIRKernel` instance.
            grid_size: Grid size for kernel launch.
        """
        self._kernel = kernel
        self._grid_size = grid_size

    def execute(
        self,
        in0: object,
        in1: Optional[object],
        out: object,
    ) -> None:
        """Execute the tensor operation on GPU.

        Args:
            in0: First input tensor (CuPy ndarray on CUDA device).
            in1: Second input tensor (CuPy ndarray on CUDA device).
                Ignored for unary operations.
            out: Output tensor (pre-allocated, on CUDA device).
        """
        self._kernel(in0, in1, out)


# =============================================================================
# Factory
# =============================================================================


def create_operation(config: "TensorOperationConfig") -> TileIROperation:
    """Create and compile a tileir operation from configuration.

    Dispatches to the binary or unary pipeline based on ``prim_main``.

    Args:
        config: Tensor operation configuration with ``backend="tileir"``.

    Returns:
        Compiled :class:`TileIROperation` instance.

    Raises:
        ValueError: If config is not executable.
        ImportError: If ``cuda.tile`` or ``cupy`` is not installed.
    """
    from etops.types import PrimType

    is_unary = config.prim_main in (PrimType.copy, PrimType.relu)

    if is_unary:
        from etops.backends._tileir.config_analysis import analyze_unary_config
        from etops.backends._tileir.compiler import compile_unary_analysis

        unary_analysis = analyze_unary_config(config)
        unary_kernel = compile_unary_analysis(unary_analysis)
        return TileIROperation(unary_kernel, unary_analysis.grid_size)

    from etops.backends._tileir.config_analysis import analyze_binary_config
    from etops.backends._tileir.compiler import compile_binary_analysis

    binary_analysis = analyze_binary_config(config)
    binary_kernel = compile_binary_analysis(binary_analysis)
    return TileIROperation(binary_kernel, binary_analysis.grid_size)


# =============================================================================
# Optimizer
# =============================================================================


def get_default_optimization_config() -> dict:
    """Get default optimization configuration for the tileir backend.

    Returns:
        Dictionary with default optimization parameters:
        - ``max_grid``: Maximum CUDA grid size (2^24 - 1).
        - ``max_prim_dims``: Maximum number of prim dimensions for unary
          operations (default: 5).
        - ``max_prim_tile_volume``: Maximum padded tile volume (product of
          ``next_pow2(size)`` for all prim dims) for unary operations.
          Tiles exceeding this limit cause the tileiras compiler to time
          out or produce slow code.  Default: 32768 (2^15).
    """
    return {
        "max_grid": 16777215,  # 2^24 - 1
        "max_prim_dims": 5,
        "max_prim_tile_volume": 32768,  # 2^15
    }


def optimize_config(
    config: "TensorOperationConfig",
    optimization_config: Optional[dict] = None,
) -> "TensorOperationConfig":
    """Optimize a tensor operation configuration for the tileir backend.

    Dispatches to the binary or unary optimizer based on ``prim_main``.

    **Binary algorithm:**

    1. Add synthetic prim dimensions for any missing M/N/K types.
    2. Select prim representatives (preferring unit stride).
    3. Sort shared dims by stride for memory locality.
    4. Interleave shared M and N dimensions.
    5. Demote outermost shared dims to seq if grid would exceed *max_grid*.
    6. Produce ordering: ``[shared] -> [seq] -> [prims]``.

    **Unary algorithm:**

    1. Select prim dims: unit-stride dims from in0 and out, then fill
       to *max_prim_dims* by smallest min-stride, subject to
       *max_prim_tile_volume* (padded tile volume cap).
    2. Safety check: avoid a known tileiras assembler crash when the
       innermost prim dim pads to 64 with asymmetric unit strides.
    3. Remaining dims become shared (up to *max_grid*), then seq.
    4. Produce ordering: ``[shared] -> [seq] -> [prims]``.

    Args:
        config: Tensor operation configuration to optimize.
        optimization_config: Optional dict with keys:
            - ``max_grid``: Maximum allowed grid size (default: 2^24 - 1).
            - ``max_prim_dims``: Max prim dims for unary (default: 5).
            - ``max_prim_tile_volume``: Max padded prim tile volume for
              unary (default: 32768).

    Returns:
        Optimized :class:`TensorOperationConfig` with exec_types and
        dimension order set for tileir execution.
    """
    from etops.backends._tileir.optimizer import optimize

    return optimize(config, optimization_config)
