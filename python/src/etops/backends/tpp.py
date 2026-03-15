"""
TPP backend implementation using C++ libxsmm bindings.

This backend uses the einsum_ir C++ backend for high-performance tensor
operations on CPU using Tensor Processing Primitives (TPP).
"""

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from etops.config import TensorOperationConfig

from etops._etops_core import TensorOperation as CppTensorOperation
from etops.types import ErrorType


class TppOperation:
    """
    Compiled TPP operation wrapping C++ implementation.

    This class wraps the C++ TensorOperation and provides the execute()
    method required by the CompiledOperation protocol.
    """

    def __init__(self, cpp_op: CppTensorOperation):
        """
        Initialize TPP operation.
        
        Args:
            cpp_op: Compiled C++ TensorOperation instance
        """
        self._op = cpp_op

    def execute(
        self,
        in0: np.ndarray,
        in1: np.ndarray | None,
        out: np.ndarray
    ) -> None:
        """
        Execute the tensor operation.
        
        Args:
            in0: First input tensor
            in1: Second input tensor (None for unary operations)
            out: Output tensor (must be pre-allocated)
        """
        self._op.execute(in0, in1, out)


def create_operation(config: "TensorOperationConfig") -> TppOperation:
    """
    Create and compile a TPP operation from configuration.

    Args:
        config: Tensor operation configuration

    Returns:
        Compiled TppOperation instance

    Raises:
        RuntimeError: If compilation fails
    """
    cpp_op = CppTensorOperation()

    err = cpp_op.setup(
        config.backend,
        int(config.data_type),
        int(config.prim_first),
        int(config.prim_main),
        int(config.prim_last),
        [int(dt) for dt in config.dim_types],
        [int(et) for et in config.exec_types],
        list(config.dim_sizes),
        [[[int(s) for s in tensor] for tensor in level] for level in config.strides]
    )

    if int(err) != int(ErrorType.success):
        raise RuntimeError(f"TPP compilation failed with error: {err}")

    return TppOperation(cpp_op)


def get_default_optimization_config() -> dict:
    """
    Get default optimization configuration for TPP backend.
    
    Returns:
        Dictionary with optimization parameters:
        - target_m (int): Target M block size
        - target_n (int): Target N block size
        - target_k (int): Target K block size
        - num_threads (int): Number of threads
        - packed_gemm_support (bool): Packed GEMM support
        - br_gemm_support (bool): Batch-reduce GEMM support
        - packing_support (bool): Packing support
        - sfc_support (bool): SFC support
        - l2_cache_size (int): L2 cache size in bytes
    """
    return CppTensorOperation.get_default_optimization_config("tpp")


def optimize_config(
    config: "TensorOperationConfig",
    optimization_config: dict | None = None
) -> "TensorOperationConfig":
    """
    Optimize a tensor operation configuration for TPP backend.

    Args:
        config: Tensor operation configuration to optimize
        optimization_config: TPP-specific optimization parameters.
            Valid keys:
            - target_m (int): Target M block size
            - target_n (int): Target N block size
            - target_k (int): Target K block size
            - num_threads (int): Number of threads
            - packed_gemm_support (bool): Packed GEMM support
            - br_gemm_support (bool): Batch-reduce GEMM support
            - packing_support (bool): Packing support
            - sfc_support (bool): SFC support
            - l2_cache_size (int): L2 cache size in bytes

    Returns:
        Optimized TensorOperationConfig

    Raises:
        RuntimeError: If optimization fails
    """
    from etops.types import DataType, PrimType, ExecType, DimType, ErrorType

    if optimization_config is None:
        optimization_config = get_default_optimization_config()

    result = CppTensorOperation.optimize(
        "tpp",
        int(config.data_type),
        int(config.prim_first),
        int(config.prim_main),
        int(config.prim_last),
        [int(dt) for dt in config.dim_types],
        [int(et) for et in config.exec_types],
        list(config.dim_sizes),
        [[[int(s) for s in tensor] for tensor in level] for level in config.strides],
        optimization_config
    )

    err = result[0]
    if int(err) != int(ErrorType.success):
        raise RuntimeError(f"TPP optimization failed with error: {err}")

    from etops.config import TensorOperationConfig
    return TensorOperationConfig(
        backend="tpp",
        data_type=DataType(int(result[1])),
        prim_first=PrimType(int(result[2])),
        prim_main=PrimType(int(result[3])),
        prim_last=PrimType(int(result[4])),
        dim_types=tuple(DimType(int(dt)) for dt in result[5]),
        exec_types=tuple(ExecType(int(et)) for et in result[6]),
        dim_sizes=tuple(result[7]),
        strides=tuple(
            tuple(tuple(tensor) for tensor in level)
            for level in result[8]
        ),
    )


__all__ = ["create_operation", "get_default_optimization_config", "optimize_config", "TppOperation"]
