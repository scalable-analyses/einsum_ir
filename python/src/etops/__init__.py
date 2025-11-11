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
    packing_strides_in0: Sequence[int] = ()
    packing_strides_in1: Sequence[int] = ()
    num_threads_omp: int = 0
    num_threads_sfc_m: int = 0
    num_threads_sfc_n: int = 0

    def apply(self, op: _CppOp) -> None:
        """
        Apply this configuration to a TensorOperation instance.
        Args:
            op: The TensorOperation instance to configure
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
            tuple(self.packing_strides_in0),
            tuple(self.packing_strides_in1),
            self.num_threads_omp,
            self.num_threads_sfc_m,
            self.num_threads_sfc_n
        )
        if err != ErrorType.success:
            raise RuntimeError(f"einsum_ir TensorOperation setup failed: {err}")

class TensorOperation(_CppOp):
    def __init__(self, config: Union[TensorOperationConfig, None] = None):
        """
        Create a new tensor operation instance.
        Args:
            config: Optional configuration to apply to the operation
        Raises:
            RuntimeError: If the setup fails
        """
        super().__init__()
        if config is not None:
            config.apply(self)

def optimize(
    config: TensorOperationConfig,
    target_m: int,
    target_n: int,
    target_k: int, 
    num_threads: int = 0,
    generate_sfc: bool = False,
    br_gemm_support: bool = True,
    packing_support: bool = False,
    packed_gemm_support: bool = True,
    l2_cache_size: int = 0
) -> TensorOperationConfig:
    """
    Optimize a tensor operation configuration using ContractionOptimizer.
    
    Args:
        config: The original tensor operation configuration
        target_m: Target M block size for optimization
        target_n: Target N block size for optimization  
        target_k: Target K block size for optimization
        num_threads: Number of threads for parallel execution automatically determined if <1
        generate_sfc: Whether to generate a SFC iteration
        br_gemm_support: Whether backend supports batch-reduce GEMM
        packing_support: Whether backend supports packing
        packed_gemm_support: Whether backend supports packed GEMM
        l2_cache_size: Size of the L2 cache in bytes (default: 1MiB if <1)
        
    Returns:
        Optimized configuration with potentially modified dimensions and execution types.
        
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
        config.packing_strides_in0,
        config.packing_strides_in1,
        target_m,
        target_n,
        target_k,
        num_threads,
        generate_sfc,
        br_gemm_support,
        packing_support,
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
     opt_strides_out,
     opt_packing_strides_in0,
     opt_packing_strides_in1,
     opt_num_threads_omp,
     opt_num_threads_sfc_m,
     opt_num_threads_sfc_n) = result
    
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
        strides_out=opt_strides_out,
        packing_strides_in0=opt_packing_strides_in0,
        packing_strides_in1=opt_packing_strides_in1,
        num_threads_omp=opt_num_threads_omp,
        num_threads_sfc_m=opt_num_threads_sfc_m,
        num_threads_sfc_n=opt_num_threads_sfc_n
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
