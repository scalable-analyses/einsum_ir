"""
TileIR backend implementation for etops.

This backend generates GPU tensor kernels by directly constructing TileIR
object graphs (``cuda.tile._ir.ops.*`` classes), bypassing the
``@ct.kernel`` Python frontend.  Compiled cubins are launched via CuPy
``RawModule``.

Supports:
  * Binary contractions: GEMM and BRGEMM (3-pointer ABI: in0, in1, out).
  * Unary operations: copy, relu (2-pointer ABI: in0, out).
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

    def __init__(self, kernel: object) -> None:
        """Initialize.

        Args:
            kernel: A :class:`TileIRKernel` instance.  The kernel
                manages its own grid size internally.
        """
        self._kernel = kernel

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
                Required for binary contractions.  Pass ``None`` for
                unary operations.
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
        return TileIROperation(unary_kernel)

    from etops.backends._tileir.config_analysis import analyze_binary_config
    from etops.backends._tileir.compiler import compile_binary_analysis

    binary_analysis = analyze_binary_config(config)
    binary_kernel = compile_binary_analysis(binary_analysis)
    return TileIROperation(binary_kernel)


# =============================================================================
# Optimizer
# =============================================================================


def get_default_optimization_config() -> dict:
    """Get default optimization configuration for the tileir backend.

    Returns:
        Dictionary with default optimization parameters:

        - ``max_grid``: Maximum CUDA grid size (2^24 - 1).
        - ``max_prim_dim_size``: Maximum raw size for any single
          dimension (binary and unary).  Dimensions exceeding this
          limit are split into an outer/inner pair via factorization
          before prim selection. Default: 64.
        - ``max_prim_dims``: Maximum number of prim dimensions for
          unary operations (default: 5).
        - ``max_prim_tile_volume``: Maximum padded tile volume (product
          of ``next_pow2(size)`` for all prim dims).  Tiles exceeding
          this limit cause the cuda.tile compiler to time out or
          produce slow code.  Default: 32768 (2^15).
    """
    return {
        "max_grid": 16777215,  # 2^24 - 1
        "max_prim_dim_size": 64,
        "max_prim_dims": 5,
        "max_prim_tile_volume": 32768,  # 2^15
    }


def optimize_config(
    config: "TensorOperationConfig",
    optimization_config: Optional[dict] = None,
) -> "TensorOperationConfig":
    """Optimize a tensor operation configuration for the tileir backend.

    Dispatches to the binary or unary optimizer based on ``prim_main``.
    Both pipelines use a transformation-based approach: split oversized
    dims, assign exec types, reorder, and fuse compatible neighbors.

    TODO: Update binary pipeline description
    **Binary pipeline:**

    1. Add synthetic size-1 prim dims for any missing M/N/K types.
    2. Split dims exceeding *max_prim_dim_size* via factorization.
    3. Select prim representatives (preferring unit stride).
    4. Assign exec types: K outers -> seq, M/N/C outers -> shared.
    5. Reorder: shared -> seq -> prim (N, K, M innermost).
    6. Fuse adjacent compatible dims.

    **Unary pipeline:**

    1. Split dims exceeding *max_prim_dim_size* via factorization.
    2. Select prim dims: unit-stride dims from in0 and out, then fill
       to *max_prim_dims* by smallest min-stride, subject to
       *max_prim_tile_volume* (padded tile volume cap).
    3. Safety check: avoid a known cuda.tile assembler crash when the
       innermost prim dim pads to 64 with asymmetric unit strides.
    4. Remaining dims become shared (up to *max_grid*), then seq.
    5. Reorder: shared -> seq -> prim.
    6. Fuse adjacent compatible dims.

    Args:
        config: Tensor operation configuration to optimize.
        optimization_config: Optional dict with keys documented in
            :func:`get_default_optimization_config`.

    Returns:
        Optimized :class:`TensorOperationConfig` with exec_types and
        dimension order set for tileir execution.
    """

    # TODO: Add optimization_config support

    from etops.backends._tileir.optimizer import Optimizer

    optimizer = Optimizer(config)
    optimizer.optimize()
    return optimizer.get_optimized_config()
