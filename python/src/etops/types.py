"""
Type definitions for etops tensor operations.

This module defines all enum types used throughout etops. Python is the single
source of truth for these definitions - the C++ bindings accept int values
and cast internally.
"""

from enum import IntEnum


class DataType(IntEnum):
    """
    Data types for tensor elements.
    
    Attributes:
        float32: 32-bit floating point (single precision)
        float64: 64-bit floating point (double precision)
    """
    float32 = 0
    float64 = 1


class PrimType(IntEnum):
    """
    Primitive operation types.
    
    Attributes:
        none: No operation
        zero: Zero initialization
        copy: Copy operation
        relu: ReLU activation
        gemm: General matrix multiply
        brgemm: Batch-reduce GEMM
    """
    none = 0
    zero = 1
    copy = 2
    relu = 3
    gemm = 4
    brgemm = 5


class ExecType(IntEnum):
    """
    Execution types for dimensions.
    
    Attributes:
        seq: Sequential execution
        prim: Primitive execution (innermost loop)
        shared: Shared memory execution
        sfc: Space-filling curve execution
    """
    seq = 0
    prim = 1
    shared = 2
    sfc = 3


class DimType(IntEnum):
    """
    Dimension types for tensor contractions.
    
    Attributes:
        c: Batch dimension (all tesnors)
        m: M dimension (first input and output)
        n: N dimension (second input and output)
        k: K dimension (both inputs, contraction axis)
    """
    c = 0
    m = 1
    n = 2
    k = 3


class ErrorType(IntEnum):
    """
    Error codes from tensor operations.
    
    Attributes:
        success: Operation completed successfully
        compilation_failed: Compilation failed
        invalid_stride_shape: Invalid stride shape
        invalid_optimization_config: Invalid optimization configuration
    """
    success = 0
    compilation_failed = 1
    invalid_stride_shape = 2
    invalid_optimization_config = 3


# =============================================================================
# Convenience namespace classes for ergonomic access
# =============================================================================

class prim:
    """
    Namespace for primitive types used in tensor operations.
    
    Provides convenient access to PrimType enum values.
    
    Example:
        >>> etops.prim.gemm
        <PrimType.gemm: 4>
    """
    none = PrimType.none
    zero = PrimType.zero
    relu = PrimType.relu
    copy = PrimType.copy
    gemm = PrimType.gemm
    brgemm = PrimType.brgemm

    __all__ = ["none", "zero", "relu", "copy", "gemm", "brgemm"]

    @classmethod
    def __dir__(cls):
        return cls.__all__


class dim:
    """
    Namespace for dimension types used in tensor operations.
    
    Provides convenient access to DimType enum values.
    
    Example:
        >>> etops.dim.m
        <DimType.m: 1>
    """
    c = DimType.c
    m = DimType.m
    n = DimType.n
    k = DimType.k

    __all__ = ["c", "m", "n", "k"]

    @classmethod
    def __dir__(cls):
        return cls.__all__


class exec:
    """
    Namespace for execution types used in tensor operations.
    
    Provides convenient access to ExecType enum values.
    
    Example:
        >>> etops.exec.prim
        <ExecType.prim: 1>
    """
    seq = ExecType.seq
    prim = ExecType.prim
    shared = ExecType.shared
    sfc = ExecType.sfc

    __all__ = ["seq", "prim", "shared", "sfc"]

    @classmethod
    def __dir__(cls):
        return cls.__all__


# =============================================================================
# Module-level convenience aliases
# =============================================================================

#: Alias for DataType
dtype = DataType

#: Alias for DataType.float32
float32: DataType = DataType.float32

#: Alias for DataType.float64
float64: DataType = DataType.float64


__all__ = [
    # Enum types
    "DataType",
    "PrimType",
    "ExecType",
    "DimType",
    "ErrorType",
    # Namespace classes
    "prim",
    "dim",
    "exec",
    "etype",
    # Convenience aliases
    "dtype",
    "float32",
    "float64",
]
