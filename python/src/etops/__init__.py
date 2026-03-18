"""
Einsum tree operations with multi-backend support.

This module provides a unified interface for tensor operations across
different backends (TPP for CPU, cutile for GPU).

Example:
    >>> import etops
    >>> 
    >>> # List available backends
    >>> etops.list_backends()
    ['tpp']
    >>> 
    >>> # Create and compile a configuration
    >>> config = etops.TensorOperationConfig(
    ...     backend="tpp",
    ...     data_type=etops.float32,
    ...     prim_main=etops.prim.gemm,
    ...     ...
    ... )
    >>> op = etops.compile(config)
    >>> 
    >>> # Execute
    >>> op.execute(in0, in1, out)
"""

from typing import Optional

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Import types (Python-defined, single source of truth)
from etops.types import (
    DataType,
    PrimType,
    ExecType,
    DimType,
    ErrorType,
    prim,
    dim,
    exec,
    dtype,
    float32,
    float64,
    float16,
    bfloat16,
    tfloat32
)

# Import config
from etops.config import TensorOperationConfig

# Import backend registry
from etops.backends import get_backend, list_backends, get_optimizer, get_default_optimization_config


def compile(config: TensorOperationConfig):
    """
    Compile a tensor operation from configuration.

    Args:
        config: Tensor operation configuration with explicit backend

    Returns:
        Compiled operation with execute() method

    Raises:
        ValueError: If backend is not specified or unknown
        ImportError: If backend dependencies are not installed
        RuntimeError: If compilation fails
    """
    factory = get_backend(config.backend)
    return factory(config)


def optimize(
    config: TensorOperationConfig,
    optimization_config: Optional[dict] = None
) -> TensorOperationConfig:
    """
    Optimize a tensor operation configuration.

    Dispatches to the backend-specific optimizer registered for the config's backend.

    Args:
        config: The tensor operation configuration to optimize
        optimization_config: Backend-specific optimization parameters.
            If None, uses the backend's default optimization config.
            For TPP backend, valid keys are:
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
        ValueError: If backend is unknown
        ImportError: If backend dependencies are not installed
        NotImplementedError: If backend does not support optimization
        RuntimeError: If optimization fails
    """
    optimizer = get_optimizer(config.backend)
    return optimizer(config, optimization_config)


__all__ = [
    # Main API
    "compile",
    "optimize",
    "list_backends",
    "TensorOperationConfig",
    # Types
    "DataType",
    "PrimType",
    "ExecType",
    "DimType",
    "ErrorType",
    # Convenience namespaces
    "prim",
    "dim",
    "exec",
    # Convenience aliases
    "dtype",
    "float32",
    "float64",
]
