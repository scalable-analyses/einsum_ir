"""
Einsum tree operations.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._etops_core import (
    TensorOperation as _CppOp,
    DataType        as _DataType,
    PrimType        as _PrimType,
    ExecType        as _ExecType,
    DimType         as _DimType,
    ErrorType       as _ErrorType
)

# Make _ErrorType the *single* public alias
ErrorType = _ErrorType

# Public tokens
DataType = _DataType
PrimType = _PrimType
ExecType = _ExecType
DimType  = _DimType

#: Alias for DataType
dtype = DataType

#: Alias for DataType.float32
float32: _DataType = _DataType.float32
#: Alias for DataType.float64
float64: _DataType = _DataType.float64

class prim:
    """Namespace for primitive types used in tensor operations."""
    #: Alias for PrimType.none
    none   = PrimType.none
    #: Alias for PrimType.zero
    zero   = PrimType.zero
    #: Alias for PrimType.relu
    relu   = PrimType.relu
    #: Alias for PrimType.copy
    copy   = PrimType.copy
    #: Alias for PrimType.gemm
    gemm   = PrimType.gemm
    #: Alias for PrimType.brgemm
    brgemm = PrimType.brgemm

    __all__ = [
        "none",
        "zero",
        "relu",
        "copy",
        "gemm",
        "brgemm"
    ]

    @classmethod
    def __dir__(cls):
        return cls.__all__


class etype:
    """Namespace for execution types used in tensor operations."""
    #: Alias for ExecType.prim
    prim   = ExecType.prim
    #: Alias for ExecType.seq
    seq    = ExecType.seq
    #: Alias for ExecType.shared
    shared = ExecType.shared
    #: Alias for ExecType.sfc
    sfc    = ExecType.sfc

    __all__ = [
        "prim",
        "seq",
        "shared",
        "sfc"
    ]

    @classmethod
    def __dir__(cls):
        return cls.__all__

_exec = etype
exec = _exec  # noqa: A003

class dim:
    """Namespace for dimension types used in tensor operations."""
    #: Alias for DimType.c
    c = DimType.c
    #: Alias for DimType.m
    m = DimType.m
    #: Alias for DimType.n
    n = DimType.n
    #: Alias for DimType.k
    k = DimType.k

    __all__ = [
        "c",
        "m",
        "n",
        "k"
    ]

    @classmethod
    def __dir__(cls):
        return cls.__all__

# Helpers
from dataclasses import dataclass
from typing import Sequence, Union

@dataclass(frozen=True)
class TensorOperationConfig:
    """
    Configuration for tensor operations.

    Supports both binary contractions and unary operations.
    The operation type is automatically determined from prim_main.

    Binary Contractions:
      - prim_main: etops.prim.gemm or etops.prim.brgemm
      - dim_types: combination of etops.dim.m, .n, .k, .c
      - prim_first: etops.prim.zero or .none (optional first touch)
      - prim_last: etops.prim.relu or .none (optional last touch)
      - strides_in0, strides_in1, strides_out: all required

    Unary Operations:
      - prim_main: etops.prim.copy, or .zero
      - dim_types: must be etops.dim.c for all dimensions
      - prim_first: must be etops.prim.none
      - prim_last: must be etops.prim.none
      - strides_in0, strides_out: required
      - strides_in1: ignored (can be empty tuple or dummy values)
    """
    data_type: _DataType
    prim_first: _PrimType
    prim_main: _PrimType
    prim_last: _PrimType
    dim_types: Sequence[_DimType]
    exec_types: Sequence[_ExecType]
    dim_sizes: Sequence[int]
    strides_in0: Sequence[int]
    strides_in1: Sequence[int]
    strides_out: Sequence[int]

    def apply(self, op: _CppOp, num_threads: int = 0) -> None:
        """
        Apply this configuration to a TensorOperation instance.
        Args:
            op: The TensorOperation instance to configure
            num_threads: Number of threads to use for execution (automatically determined if <1)
        Raises:
            RuntimeError: If the setup fails.
        """
        err = op.setup(
            self.data_type,
            self.prim_first,
            self.prim_main,
            self.prim_last,
            tuple(self.dim_types),
            tuple(self.exec_types),
            tuple(self.dim_sizes),
            tuple(self.strides_in0),
            tuple(self.strides_in1),
            tuple(self.strides_out),
            num_threads
        )
        if err != ErrorType.success:
            raise RuntimeError(f"einsum_ir TensorOperation setup failed: {err}")

class TensorOperation(_CppOp):
    def __init__(self, config: Union[TensorOperationConfig, None] = None, num_threads: int = 0):
        """
        Create a new tensor operation instance.
        Args:
            config: Optional configuration to apply to the operation
            num_threads: Number of threads to use for execution (automatically determined if <1)
        Raises:
            RuntimeError: If the setup fails
        """
        super().__init__()
        if config is not None:
            config.apply(self, num_threads=num_threads)

def optimize(
    config: TensorOperationConfig,
    target_m: int = 0,
    target_n: int = 0,
    target_k: int = 0,
    num_threads: int = 0,
    br_gemm_support: bool = True,
    packed_gemm_support: bool = True,
    l2_cache_size: int = 0
) -> TensorOperationConfig:
    """
    Optimize a tensor operation configuration.

    The operation type is automatically determined from config.prim_main.

    Binary contractions:
      Uses ContractionOptimizer with provided target sizes and GEMM support flags.

    Unary operations:
      Uses UnaryOptimizer. The target_m, target_n, target_k, br_gemm_support,
      and packed_gemm_support parameters are ignored for unary operations.

    Args:
        config: The original tensor operation configuration
        target_m: Target M block size for optimization (ignored for unary operations)
        target_n: Target N block size for optimization (ignored for unary operations)
        target_k: Target K block size for optimization ((ignored for unary operations)
        num_threads: Number of threads for parallel execution (auto-determined if <1)
        br_gemm_support: Whether backend supports batch-reduce GEMM (ignored for unary operations)
        packed_gemm_support: Whether backend supports packed GEMM (ignored for unary operations)
        l2_cache_size: Size of the L2 cache in bytes (default: 1MiB if <1)

    Returns:
        Optimized configuration.

    Raises:
        RuntimeError: If optimization fails
    """
    # Use default L2 cache size if not provided
    if l2_cache_size <= 0:
        l2_cache_size = 1024 * 1024  # Default: 1MiB L2 cache

    # Call the static C++ method
    result = _CppOp.optimize(
        config.data_type,
        config.prim_first,
        config.prim_main,
        config.prim_last,
        config.dim_types,
        config.exec_types,
        config.dim_sizes,
        config.strides_in0,
        config.strides_in1,
        config.strides_out,
        target_m,
        target_n,
        target_k,
        num_threads,
        br_gemm_support,
        packed_gemm_support,
        l2_cache_size
    )
    
    # Unpack the result tuple
    (err,
     opt_data_type,
     opt_prim_first,
     opt_prim_main,
     opt_prim_last,
     opt_dim_types,
     opt_exec_types,
     opt_dim_sizes,
     opt_strides_in0,
     opt_strides_in1,
     opt_strides_out) = result
    
    # Check for errors
    if err != ErrorType.success:
        raise RuntimeError(f"einsum_ir TensorOperation optimization failed: {err}")
    
    # Return new optimized Config
    return TensorOperationConfig(
        data_type=opt_data_type,
        prim_first=opt_prim_first,
        prim_main=opt_prim_main,
        prim_last=opt_prim_last,
        dim_types=opt_dim_types,
        exec_types=opt_exec_types,
        dim_sizes=opt_dim_sizes,
        strides_in0=opt_strides_in0,
        strides_in1=opt_strides_in1,
        strides_out=opt_strides_out
    )

__all__ = [
    "TensorOperation",
    "TensorOperationConfig",
    "DataType",
    "PrimType",
    "ExecType",
    "DimType",
    "dtype",
    "float32",
    "float64",
    "prim",
    "etype",
    "exec",
    "dim",
    "optimize",
    "ErrorType"
]
