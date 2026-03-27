"""
TileIR backend implementation for etops.

This backend generates GPU tensor contraction kernels by directly
constructing TileIR object graphs (``cuda.tile._ir.ops.*`` classes),
bypassing the ``@ct.kernel`` Python frontend.  Compiled cubins are
launched via CuPy ``RawModule`` with a pointer-only ABI (in0, in1, out).

Supports:
  * Binary contractions: GEMM and BRGEMM.
  * Data types: FP32, FP64, FP16, BF16, TF32.
  * Exec types: shared (grid-mapped), sequential (for-loops), prim (MMA tile).
  * ``prim_first``: ``zero`` (beta=0) or ``none`` (beta=1).
  * ``prim_last``: ``none`` or ``relu``.
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
    from etops.backends.base import CompiledOperation

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
            out: Output tensor (pre-allocated, on CUDA device).
        """
        self._kernel(in0, in1, out)


# =============================================================================
# Factory
# =============================================================================


def create_operation(config: "TensorOperationConfig") -> TileIROperation:
    """Create and compile a tileir operation from configuration.

    Args:
        config: Tensor operation configuration with ``backend="tileir"``.

    Returns:
        Compiled :class:`TileIROperation` instance.

    Raises:
        ValueError: If config is not executable.
        ImportError: If ``cuda.tile`` or ``cupy`` is not installed.
    """
    from etops.backends._tileir.config_analysis import analyze_config
    from etops.backends._tileir.compiler import compile_analysis

    analysis = analyze_config(config)
    kernel = compile_analysis(analysis)

    return TileIROperation(kernel, analysis.grid_size)


# =============================================================================
# Optimizer
# =============================================================================


def get_default_optimization_config() -> dict:
    """Get default optimization configuration for the tileir backend.

    Returns:
        Dictionary with default optimization parameters:
        - ``max_grid``: Maximum CUDA grid size (2^24 - 1).
    """
    return {
        "max_grid": 16777215,  # 2^24 - 1
    }


def optimize_config(
    config: "TensorOperationConfig",
    optimization_config: Optional[dict] = None,
) -> "TensorOperationConfig":
    """Optimize a tensor operation configuration for the tileir backend.

    The algorithm:

    1. Add synthetic prim dimensions for any missing M/N/K types.
    2. Select prim representatives (preferring unit stride).
    3. Sort shared dims by stride for memory locality.
    4. Interleave shared M and N dimensions.
    5. Demote outermost shared dims to seq if grid would exceed *max_grid*.
    6. Produce ordering: ``[shared] → [seq] → [prims]``.

    Args:
        config: Tensor operation configuration to optimize.
        optimization_config: Optional dict with keys:
            - ``max_grid``: Maximum allowed grid size (default: 2^24 - 1).

    Returns:
        Optimized :class:`TensorOperationConfig` with exec_types and
        dimension order set for tileir execution.
    """
    from etops.backends._tileir.optimizer import optimize

    return optimize(config, optimization_config)
