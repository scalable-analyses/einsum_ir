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
from typing import Sequence, Union, Optional, Dict

@dataclass(frozen=True)
class TensorOperationConfig:
    """
    Configuration for tensor operations.

    Supports both binary contractions and unary operations.
    The operation type is automatically determined from prim_main.

    The strides parameter is a 3D tensor with shape [LEVEL][TENSOR][DIMENSION]:

    LEVEL dimension (axis 0):
      - [0] = Primary memory layout strides
      - [1] = Secondary level (e.g., packing strides, L1 cache)
      - [2+] = Additional levels (e.g., L2, L3 cache)

      For user input, typically only level 0 is provided.
      The optimizer may add additional levels (e.g., packing).

    TENSOR dimension (axis 1):
      - Binary operations: [0]=in0, [1]=in1, [2]=out
      - Unary operations: [0]=in, [1]=out

    DIMENSION dimension (axis 2):
      - Corresponds to dimension indices

    Binary Contractions:
      - backend: "tpp" or None
      - prim_main: etops.prim.gemm or etops.prim.brgemm
      - dim_types: combination of etops.dim.m, .n, .k, .c
      - prim_first: etops.prim.zero or .none (optional first touch)
      - prim_last: etops.prim.relu or .none (optional last touch)
      - strides: shape [1 or more][3][num_dims]

    Unary Operations:
      - backend: "tpp" or None
      - prim_main: etops.prim.copy or .zero
      - dim_types: must be etops.dim.c for all dimensions
      - prim_first: must be etops.prim.none
      - prim_last: must be etops.prim.none
      - strides: shape [1][2][num_dims]
    """
    data_type:  _DataType
    prim_first: _PrimType
    prim_main:  _PrimType
    prim_last:  _PrimType
    dim_types:  Sequence[_DimType]
    exec_types: Sequence[_ExecType]
    dim_sizes:  Sequence[int]
    strides:    Sequence[Sequence[Sequence[int]]]  # [LEVEL][TENSOR][DIMENSION]
    backend:    Optional[str] = None

    def __post_init__(self):
        """Validate configuration at creation time."""
        # Validate backend
        if self.backend is not None and self.backend not in ["tpp"]:
            raise ValueError(f"Unsupported backend: '{self.backend}'. Currently only 'tpp' is supported.")

        # Determine operation type from prim_main
        is_binary = self.prim_main in [PrimType.gemm, PrimType.brgemm]
        is_unary  = self.prim_main in [PrimType.copy, PrimType.zero]

        if not is_binary and not is_unary:
            raise ValueError(
                f"Invalid prim_main: {self.prim_main}. "
                f"Must be gemm/brgemm (binary) or copy/zero (unary)."
            )

        # Validate dimension consistency
        num_dims = len(self.dim_types)
        if len(self.exec_types) != num_dims:
            raise ValueError(
                f"exec_types length ({len(self.exec_types)}) must match "
                f"dim_types length ({num_dims})."
            )
        if len(self.dim_sizes) != num_dims:
            raise ValueError(
                f"dim_sizes length ({len(self.dim_sizes)}) must match "
                f"dim_types length ({num_dims})."
            )

        # Validate dim_sizes are positive
        for i, size in enumerate(self.dim_sizes):
            if size <= 0:
                raise ValueError(f"dim_sizes[{i}] must be positive, got {size}.")

        # Validate strides shape
        if len(self.strides) == 0:
            raise ValueError("strides must have at least one level (level 0).")

        # Validate level 0 strides
        expected_tensors = 3 if is_binary else 2
        if len(self.strides[0]) != expected_tensors:
            raise ValueError(
                f"strides[0] must have {expected_tensors} tensors "
                f"({'in0, in1, out' if is_binary else 'in, out'}), "
                f"got {len(self.strides[0])}."
            )

        # Validate stride dimensions match num_dims
        for tensor_idx, tensor_strides in enumerate(self.strides[0]):
            if len(tensor_strides) != num_dims:
                tensor_names = ['in0', 'in1', 'out'] if is_binary else ['in', 'out']
                raise ValueError(
                    f"strides[0][{tensor_idx}] ({tensor_names[tensor_idx]}) length "
                    f"({len(tensor_strides)}) must match number of dimensions ({num_dims})."
                )

        # Unary-specific validations
        if is_unary:
            # All dim_types must be 'c' for unary operations
            for i, dt in enumerate(self.dim_types):
                if dt != DimType.c:
                    raise ValueError(
                        f"For unary operations, all dim_types must be etops.dim.c. "
                        f"dim_types[{i}] is {dt}."
                    )

            # prim_first and prim_last must be 'none' for unary
            if self.prim_first != PrimType.none:
                raise ValueError(
                    f"For unary operations, prim_first must be etops.prim.none, "
                    f"got {self.prim_first}."
                )
            if self.prim_last != PrimType.none:
                raise ValueError(
                    f"For unary operations, prim_last must be etops.prim.none, "
                    f"got {self.prim_last}."
                )

        # Binary-specific validations
        if is_binary:
            # Validate prim_first is compatible
            if self.prim_first not in [PrimType.none, PrimType.zero]:
                raise ValueError(
                    f"For binary contractions, prim_first must be etops.prim.none or "
                    f"etops.prim.zero, got {self.prim_first}."
                )

            # Validate prim_last is compatible
            if self.prim_last not in [PrimType.none, PrimType.relu]:
                raise ValueError(
                    f"For binary contractions, prim_last must be etops.prim.none or "
                    f"etops.prim.relu, got {self.prim_last}."
                )

    def apply(self, op: _CppOp) -> None:
        """
        Apply this configuration to a TensorOperation instance.
        Args:
            op: The TensorOperation instance to configure
        Raises:
            RuntimeError: If the setup fails.
        """
        err = op.setup(
            self.backend,
            self.data_type,
            self.prim_first,
            self.prim_main,
            self.prim_last,
            tuple(self.dim_types),
            tuple(self.exec_types),
            tuple(self.dim_sizes),
            tuple(tuple(tuple(tensor) for tensor in level) for level in self.strides)
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

# Backend namespace
class _TPPBackend:
    """TPP (Tensor Processing Primitives) backend for tensor operations."""
    name: str = "tpp"

    @staticmethod
    def get_default_optimization_config() -> Dict[str, Union[int, bool]]:
        """Get default optimization configuration for TPP backend.

        Returns:
            Dictionary with keys:
            - target_m (int): Target M block size.
            - target_n (int): Target N block size.
            - target_k (int): Target K block size.
            - num_threads (int): Number of threads.
            - packed_gemm_support (bool): Packed GEMM support.
            - br_gemm_support (bool): Batch-reduce GEMM support.
            - packing_support (bool): Packing support.
            - sfc_support (bool): SFC support.
            - l2_cache_size (int): L2 cache size in bytes (default: 1048576)
        """
        return _CppOp.get_default_optimization_config("tpp")

class backend:
    """Namespace for available backends."""
    tpp = _TPPBackend()

def optimize(
    config: TensorOperationConfig,
    optimization_config: Optional[Dict[str, Union[int, bool]]] = None
) -> TensorOperationConfig:
    """
    Optimize a tensor operation configuration.

    Args:
        config: The tensor operation configuration to optimize
        optimization_config: Backend-specific optimization parameters.
                           If None, uses backend default configuration.

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
        RuntimeError: If optimization fails
        ValueError: If backend is unknown
    """
    # Set backend to 'tpp' if not provided
    actual_backend = config.backend if config.backend is not None else 'tpp'

    # Get default config if not provided
    if optimization_config is None:
        if actual_backend == "tpp":
            optimization_config = backend.tpp.get_default_optimization_config()
        else:
            raise ValueError(f"Unknown backend: {actual_backend}")

    # Call C++ optimize
    result = _CppOp.optimize(
        actual_backend,
        config.data_type,
        config.prim_first,
        config.prim_main,
        config.prim_last,
        config.dim_types,
        config.exec_types,
        config.dim_sizes,
        tuple(tuple(tuple(tensor) for tensor in level) for level in config.strides),
        optimization_config
    )
    
    # Unpack and check error
    (err, opt_data_type, opt_prim_first, opt_prim_main, opt_prim_last,
     opt_dim_types, opt_exec_types, opt_dim_sizes, opt_strides) = result
    
    if err != ErrorType.success:
        raise RuntimeError(f"einsum_ir optimization failed: {err}")
    
    return TensorOperationConfig(
        backend=actual_backend,
        data_type=opt_data_type,
        prim_first=opt_prim_first,
        prim_main=opt_prim_main,
        prim_last=opt_prim_last,
        dim_types=opt_dim_types,
        exec_types=opt_exec_types,
        dim_sizes=opt_dim_sizes,
        strides=opt_strides
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
    "backend",
    "optimize",
    "ErrorType"
]
