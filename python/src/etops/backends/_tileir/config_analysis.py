"""
Config analysis (binary and unary) for the tileir backend.

Transforms a ``TensorOperationConfig`` into a frozen analysis object that
the IR builder and compiler consume.  All tensor layout arithmetic (shapes,
strides, grid decomposition) is computed once here so that downstream code
never touches the raw config directly.

Each tensor gets its **own** view containing only the dimensions where its
stride is non-zero.  This is a hard requirement of the tileiras compiler
which rejects zero strides.
"""

from __future__ import annotations

__all__ = [
    "BinaryConfigAnalysis",
    "UnaryConfigAnalysis",
    "analyze_binary_config",
    "analyze_unary_config",
]

from dataclasses import dataclass
from typing import Sequence, Tuple

from etops.config import TensorOperationConfig
from etops.types import DataType, DimType, ExecType, PrimType


# ---------------------------------------------------------------------------
# Data-type helpers
# ---------------------------------------------------------------------------


def _storage_dtype(dt: DataType) -> str:
    """Return the ``cuda.tile`` datatype attribute name for *dt*.

    TF32 uses FP32 storage; all others map 1-to-1.
    """
    _MAP = {
        DataType.float32: "float32",
        DataType.float64: "float64",
        DataType.float16: "float16",
        DataType.bfloat16: "bfloat16",
        DataType.tfloat32: "float32",
    }
    return _MAP[dt]


def _acc_dtype(dt: DataType) -> str:
    """Return the accumulator datatype attribute name for *dt*.

    FP16 / BF16 / TF32 accumulate in FP32; FP32 in FP32; FP64 in FP64.
    """
    if dt in (DataType.float16, DataType.bfloat16, DataType.tfloat32, DataType.float32):
        return "float32"
    return "float64"


# ---------------------------------------------------------------------------
# Per-tensor index-map entry
# ---------------------------------------------------------------------------

# Each position in a per-tensor view maps to one of three kinds:
#   ("shared", pos_in_shared_list)   → decomposed from block-id
#   ("seq",    offset_in_seq_list)   → sequential loop induction variable
#   ("prim",   0)                    → constant 0 (tile origin)

IndexMapEntry = Tuple[str, int]


# ---------------------------------------------------------------------------
# Analysis dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryConfigAnalysis:
    """Frozen analysis of a binary ``TensorOperationConfig`` for the tileir backend.

    All fields are derived deterministically from *config* during
    construction and are immutable thereafter.

    Each tensor has its own view containing only the dimensions where its
    config stride is non-zero.  The ``index_map_*`` tuples tell the IR
    builder what index variable each view position expects.

    Attributes:
        config: The original configuration.
        storage_dtype_name: ``cuda.tile._ir.typing_support.datatype`` attr
            name for the element storage type.
        acc_dtype_name: Attribute name for the accumulator type.
        prim_first: First-touch primitive (``zero`` or ``none``).
        prim_main: Main primitive (``gemm`` or ``brgemm``).
        prim_last: Last-touch primitive (``relu`` or ``none``).

        shared_loop_ids: Config indices with ``exec == shared``, in config
            order (outermost first).
        seq_loop_ids: Config indices with ``exec == seq``, in config order.
        prim_m_ids: Config indices where ``dim == m`` and ``exec == prim``.
        prim_n_ids: Config indices where ``dim == n`` and ``exec == prim``.
        prim_k_ids: Config indices where ``dim == k`` and ``exec == prim``.

        kernel_shape_m: Product of prim-M dimension sizes.
        kernel_shape_n: Product of prim-N dimension sizes.
        kernel_shape_k: Product of prim-K dimension sizes.

        grid_size: Product of all shared-loop dimension sizes.
        shared_loop_strides: Row-major strides within the linearised grid
            (one entry per ``shared_loop_ids``).

        tensor_shape_in0: Per-tensor static shape tuple for in0 ``ArrayTy``.
        tensor_shape_in1: Per-tensor static shape tuple for in1 ``ArrayTy``.
        tensor_shape_out: Per-tensor static shape tuple for out ``ArrayTy``.
        tensor_strides_in0: Per-tensor static stride tuple for in0 ``ArrayTy``.
        tensor_strides_in1: Per-tensor static stride tuple for in1 ``ArrayTy``.
        tensor_strides_out: Per-tensor static stride tuple for out ``ArrayTy``.
        tensor_order_in0: Identity ordering ``(0, ..., ndim_in0-1)``.
        tensor_order_in1: Identity ordering ``(0, ..., ndim_in1-1)``.
        tensor_order_out: Identity ordering ``(0, ..., ndim_out-1)``.

        index_map_in0: Per-position ``(kind, id)`` for in0 view.
        index_map_in1: Per-position ``(kind, id)`` for in1 view.
        index_map_out: Per-position ``(kind, id)`` for out view.

        load_tile_shape_in0: Tile shape for ``TileLoad`` of in0.
        load_tile_shape_in1: Tile shape for ``TileLoad`` of in1.
        store_tile_shape_out: Tile shape for ``TileStore`` of out.
        mma_x_shape: 2-D shape of the MMA x-operand (N, K).
        mma_y_shape: 2-D shape of the MMA y-operand (K, M).
        acc_shape: 2-D shape of the accumulator tile (N, M).

        seq_k_loop_ids: Subset of ``seq_loop_ids`` with ``dim == k``.
        seq_non_k_loop_ids: Subset of ``seq_loop_ids`` with ``dim != k``.
        brgemm_batch_count: For BRGEMM, the product of prim-K sizes that
            form the batch-reduce dimension (1 for GEMM).
    """

    # -- original config --
    config: TensorOperationConfig

    # -- data types --
    storage_dtype_name: str
    acc_dtype_name: str

    # -- primitives --
    prim_first: PrimType
    prim_main: PrimType
    prim_last: PrimType

    # -- loop classification (config indices) --
    shared_loop_ids: Tuple[int, ...]
    seq_loop_ids: Tuple[int, ...]
    seq_k_loop_ids: Tuple[int, ...]
    seq_non_k_loop_ids: Tuple[int, ...]
    prim_m_ids: Tuple[int, ...]
    prim_n_ids: Tuple[int, ...]
    prim_k_ids: Tuple[int, ...]

    # -- kernel MMA tile sizes --
    kernel_shape_m: int
    kernel_shape_n: int
    kernel_shape_k: int

    # -- grid --
    grid_size: int
    shared_loop_strides: Tuple[int, ...]

    # -- per-tensor static view descriptors (only non-zero-stride dims) --
    tensor_shape_in0: Tuple[int, ...]
    tensor_shape_in1: Tuple[int, ...]
    tensor_shape_out: Tuple[int, ...]
    tensor_strides_in0: Tuple[int, ...]
    tensor_strides_in1: Tuple[int, ...]
    tensor_strides_out: Tuple[int, ...]
    tensor_order_in0: Tuple[int, ...]
    tensor_order_in1: Tuple[int, ...]
    tensor_order_out: Tuple[int, ...]

    # -- per-tensor index maps (one entry per view position) --
    index_map_in0: Tuple[IndexMapEntry, ...]
    index_map_in1: Tuple[IndexMapEntry, ...]
    index_map_out: Tuple[IndexMapEntry, ...]

    # -- tile shapes for load / store / MMA --
    load_tile_shape_in0: Tuple[int, ...]
    load_tile_shape_in1: Tuple[int, ...]
    store_tile_shape_out: Tuple[int, ...]
    mma_x_shape: Tuple[int, int]
    mma_y_shape: Tuple[int, int]
    acc_shape: Tuple[int, int]

    # -- BRGEMM specific --
    brgemm_batch_count: int

    # -- dimension sizes & strides (raw, for convenience) --
    dim_sizes: Tuple[int, ...]
    dim_types: Tuple[DimType, ...]
    exec_types: Tuple[ExecType, ...]
    strides_in0: Tuple[int, ...]
    strides_in1: Tuple[int, ...]
    strides_out: Tuple[int, ...]


def analyze_binary_config(config: TensorOperationConfig) -> BinaryConfigAnalysis:
    """Analyze a binary *config* and return a frozen :class:`BinaryConfigAnalysis`.

    Args:
        config: A fully-specified binary ``TensorOperationConfig`` ready for
            compilation (all exec_types assigned, prim dims innermost).

    Returns:
        A frozen ``BinaryConfigAnalysis`` with all derived values.

    Raises:
        ValueError: If *config* violates tileir binary constraints (e.g. no
            prim M/N/K, shared after seq, K shared, etc.).
    """
    dim_types = tuple(config.dim_types)
    exec_types = tuple(config.exec_types)
    dim_sizes = tuple(config.dim_sizes)
    strides_in0 = tuple(config.strides[0][0])
    strides_in1 = tuple(config.strides[0][1])
    strides_out = tuple(config.strides[0][2])
    num_dims = len(dim_types)

    # -- validate basic constraints -------------------------------------------
    _validate_binary(
        config, dim_types, exec_types, dim_sizes, strides_in0, strides_in1, strides_out
    )

    # -- classify dimensions --------------------------------------------------
    shared_loop_ids = []
    seq_loop_ids = []
    seq_k_loop_ids = []
    seq_non_k_loop_ids = []
    prim_m_ids = []
    prim_n_ids = []
    prim_k_ids = []

    for i in range(num_dims):
        et = exec_types[i]
        dt = dim_types[i]
        if et == ExecType.shared:
            shared_loop_ids.append(i)
        elif et == ExecType.seq:
            seq_loop_ids.append(i)
            if dt == DimType.k:
                seq_k_loop_ids.append(i)
            else:
                seq_non_k_loop_ids.append(i)
        elif et == ExecType.prim:
            if dt == DimType.m:
                prim_m_ids.append(i)
            elif dt == DimType.n:
                prim_n_ids.append(i)
            elif dt == DimType.k:
                prim_k_ids.append(i)

    # -- kernel MMA sizes (padded to next power of 2) --------------------------
    kernel_shape_m = _next_pow2(_product(dim_sizes[i] for i in prim_m_ids))
    kernel_shape_n = _next_pow2(_product(dim_sizes[i] for i in prim_n_ids))
    kernel_shape_k = _next_pow2(_product(dim_sizes[i] for i in prim_k_ids))

    # -- grid -----------------------------------------------------------------
    grid_size = _product(dim_sizes[i] for i in shared_loop_ids)

    shared_loop_strides = _compute_shared_loop_strides(shared_loop_ids, dim_sizes)

    # -- canonical dim ordering (shared → seq → prim[N,K,M]) -----------------
    prim_ids = list(prim_n_ids) + list(prim_k_ids) + list(prim_m_ids)
    view_order = list(shared_loop_ids) + list(seq_loop_ids) + prim_ids

    # -- build per-tensor views -----------------------------------------------
    #
    # For each tensor, iterate through view_order and keep only dims where
    # the tensor's config stride is non-zero.  Prim dims are always kept
    # (their strides should be non-zero for participating tensors), and
    # K-type dims are absent from the output tensor (stride=0), etc.

    shape_in0, strides_in0_tv, tile_in0, imap_in0 = _build_tensor_view(
        strides_in0,
        view_order,
        exec_types,
        dim_sizes,
        shared_loop_ids,
        seq_loop_ids,
    )
    shape_in1, strides_in1_tv, tile_in1, imap_in1 = _build_tensor_view(
        strides_in1,
        view_order,
        exec_types,
        dim_sizes,
        shared_loop_ids,
        seq_loop_ids,
    )
    shape_out, strides_out_tv, tile_out, imap_out = _build_tensor_view(
        strides_out,
        view_order,
        exec_types,
        dim_sizes,
        shared_loop_ids,
        seq_loop_ids,
    )

    order_in0 = tuple(range(len(shape_in0)))
    order_in1 = tuple(range(len(shape_in1)))
    order_out = tuple(range(len(shape_out)))

    # MMA 2-D shapes: after TileReshape from N-D to 2-D
    # x-operand (B / in1): (N, K)
    # y-operand (A / in0): (K, M)
    # acc: (N, M)
    mma_x_shape = (kernel_shape_n, kernel_shape_k)
    mma_y_shape = (kernel_shape_k, kernel_shape_m)
    acc_shape = (kernel_shape_n, kernel_shape_m)

    # -- BRGEMM ---------------------------------------------------------------
    brgemm_batch_count = 1  # GEMM default

    return BinaryConfigAnalysis(
        config=config,
        storage_dtype_name=_storage_dtype(config.data_type),
        acc_dtype_name=_acc_dtype(config.data_type),
        prim_first=config.prim_first,
        prim_main=config.prim_main,
        prim_last=config.prim_last,
        shared_loop_ids=tuple(shared_loop_ids),
        seq_loop_ids=tuple(seq_loop_ids),
        seq_k_loop_ids=tuple(seq_k_loop_ids),
        seq_non_k_loop_ids=tuple(seq_non_k_loop_ids),
        prim_m_ids=tuple(prim_m_ids),
        prim_n_ids=tuple(prim_n_ids),
        prim_k_ids=tuple(prim_k_ids),
        kernel_shape_m=kernel_shape_m,
        kernel_shape_n=kernel_shape_n,
        kernel_shape_k=kernel_shape_k,
        grid_size=grid_size,
        shared_loop_strides=tuple(shared_loop_strides),
        tensor_shape_in0=shape_in0,
        tensor_shape_in1=shape_in1,
        tensor_shape_out=shape_out,
        tensor_strides_in0=strides_in0_tv,
        tensor_strides_in1=strides_in1_tv,
        tensor_strides_out=strides_out_tv,
        tensor_order_in0=order_in0,
        tensor_order_in1=order_in1,
        tensor_order_out=order_out,
        index_map_in0=imap_in0,
        index_map_in1=imap_in1,
        index_map_out=imap_out,
        load_tile_shape_in0=tile_in0,
        load_tile_shape_in1=tile_in1,
        store_tile_shape_out=tile_out,
        mma_x_shape=mma_x_shape,
        mma_y_shape=mma_y_shape,
        acc_shape=acc_shape,
        brgemm_batch_count=brgemm_batch_count,
        dim_sizes=dim_sizes,
        dim_types=dim_types,
        exec_types=exec_types,
        strides_in0=strides_in0,
        strides_in1=strides_in1,
        strides_out=strides_out,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_binary(
    config: TensorOperationConfig,
    dim_types: Tuple[DimType, ...],
    exec_types: Tuple[ExecType, ...],
    dim_sizes: Tuple[int, ...],
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
) -> None:
    """Raise ``ValueError`` if *config* violates tileir binary constraints."""
    num_dims = len(dim_types)

    # Length consistency
    if not (
        len(exec_types)
        == len(dim_sizes)
        == len(strides_in0)
        == len(strides_in1)
        == len(strides_out)
        == num_dims
    ):
        raise ValueError(
            "Lengths of dim_types, exec_types, dim_sizes, and strides "
            "must all be equal."
        )

    # Positive sizes, non-negative strides
    for i in range(num_dims):
        if dim_sizes[i] <= 0:
            raise ValueError(f"dim_sizes[{i}] must be positive.")
        if strides_in0[i] < 0 or strides_in1[i] < 0 or strides_out[i] < 0:
            raise ValueError(f"Strides at index {i} must be non-negative.")

    # Must have at least one prim M, N, K
    prim_m = sum(
        1
        for i in range(num_dims)
        if exec_types[i] == ExecType.prim and dim_types[i] == DimType.m
    )
    prim_n = sum(
        1
        for i in range(num_dims)
        if exec_types[i] == ExecType.prim and dim_types[i] == DimType.n
    )
    prim_k = sum(
        1
        for i in range(num_dims)
        if exec_types[i] == ExecType.prim and dim_types[i] == DimType.k
    )
    if prim_m < 1:
        raise ValueError("At least one prim dimension of type m is required.")
    if prim_n < 1:
        raise ValueError("At least one prim dimension of type n is required.")
    if prim_k < 1:
        raise ValueError("At least one prim dimension of type k is required.")

    # Prim dimensions must be innermost; no shared after seq
    _validate_ordering(exec_types)

    # K dimensions must not be shared
    for i in range(num_dims):
        if dim_types[i] == DimType.k and exec_types[i] == ExecType.shared:
            raise ValueError("K dimensions cannot be shared.")

    # Seq K dims must follow all other seq dims
    seq_k_seen = False
    for i in range(num_dims):
        if exec_types[i] == ExecType.seq:
            if dim_types[i] == DimType.k:
                seq_k_seen = True
            elif seq_k_seen:
                raise ValueError(
                    "All sequential K dimensions must come after all "
                    "other sequential dimensions."
                )


# ---------------------------------------------------------------------------
# Unary analysis dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnaryConfigAnalysis:
    """Frozen analysis of a unary ``TensorOperationConfig`` for the tileir backend.

    All fields are derived deterministically from *config* during
    construction and are immutable thereafter.

    Unary operations have exactly two tensors (in0 and out).  All
    dimension types must be ``dim.c``.  The ``prim_main`` is one of
    ``copy``, ``zero``, or ``relu``.

    Each tensor has its own view containing only the dimensions where its
    config stride is non-zero.  The ``index_map_*`` tuples tell the IR
    builder what index variable each view position expects.

    Attributes:
        config: The original configuration.
        storage_dtype_name: ``cuda.tile._ir.typing_support.datatype`` attr
            name for the element storage type.
        prim_main: Main primitive (``copy`` or ``relu``).

        shared_loop_ids: Config indices with ``exec == shared``, in config
            order (outermost first).
        seq_loop_ids: Config indices with ``exec == seq``, in config order.
        prim_ids: Config indices with ``exec == prim``.

        kernel_shape: Tuple of padded prim dimension sizes (one per prim dim).

        grid_size: Product of all shared-loop dimension sizes.
        shared_loop_strides: Row-major strides within the linearised grid
            (one entry per ``shared_loop_ids``).

        tensor_shape_in0: Per-tensor static shape tuple for in0 ``ArrayTy``.
        tensor_shape_out: Per-tensor static shape tuple for out ``ArrayTy``.
        tensor_strides_in0: Per-tensor static stride tuple for in0 ``ArrayTy``.
        tensor_strides_out: Per-tensor static stride tuple for out ``ArrayTy``.
        tensor_order_in0: Identity ordering ``(0, ..., ndim_in0-1)``.
        tensor_order_out: Identity ordering ``(0, ..., ndim_out-1)``.

        index_map_in0: Per-position ``(kind, id)`` for in0 view.
        index_map_out: Per-position ``(kind, id)`` for out view.

        load_tile_shape_in0: Tile shape for ``TileLoad`` of in0.
        store_tile_shape_out: Tile shape for ``TileStore`` of out.

        dim_sizes: Raw dimension sizes from the config.
        dim_types: Raw dimension types from the config.
        exec_types: Raw exec types from the config.
        strides_in0: Raw strides for in0.
        strides_out: Raw strides for out.
    """

    # -- original config --
    config: TensorOperationConfig

    # -- data types --
    storage_dtype_name: str

    # -- primitives --
    prim_main: PrimType

    # -- loop classification (config indices) --
    shared_loop_ids: Tuple[int, ...]
    seq_loop_ids: Tuple[int, ...]
    prim_ids: Tuple[int, ...]

    # -- kernel tile sizes --
    kernel_shape: Tuple[int, ...]

    # -- grid --
    grid_size: int
    shared_loop_strides: Tuple[int, ...]

    # -- per-tensor static view descriptors (only non-zero-stride dims) --
    tensor_shape_in0: Tuple[int, ...]
    tensor_shape_out: Tuple[int, ...]
    tensor_strides_in0: Tuple[int, ...]
    tensor_strides_out: Tuple[int, ...]
    tensor_order_in0: Tuple[int, ...]
    tensor_order_out: Tuple[int, ...]

    # -- per-tensor index maps (one entry per view position) --
    index_map_in0: Tuple[IndexMapEntry, ...]
    index_map_out: Tuple[IndexMapEntry, ...]

    # -- tile shapes for load / store --
    load_tile_shape_in0: Tuple[int, ...]
    store_tile_shape_out: Tuple[int, ...]

    # -- dimension sizes & strides (raw, for convenience) --
    dim_sizes: Tuple[int, ...]
    dim_types: Tuple[DimType, ...]
    exec_types: Tuple[ExecType, ...]
    strides_in0: Tuple[int, ...]
    strides_out: Tuple[int, ...]


def analyze_unary_config(config: TensorOperationConfig) -> UnaryConfigAnalysis:
    """Analyze a unary *config* and return a frozen :class:`UnaryConfigAnalysis`.

    Args:
        config: A fully-specified unary ``TensorOperationConfig`` ready for
            compilation (all exec_types assigned, prim dims innermost).

    Returns:
        A frozen ``UnaryConfigAnalysis`` with all derived values.

    Raises:
        ValueError: If *config* violates tileir unary constraints (e.g.
            non-c dim types, wrong number of tensors in strides, etc.).
    """
    dim_types = tuple(config.dim_types)
    exec_types = tuple(config.exec_types)
    dim_sizes = tuple(config.dim_sizes)
    strides_in0 = tuple(config.strides[0][0])
    strides_out = tuple(config.strides[0][1])
    num_dims = len(dim_types)

    # -- validate unary constraints -------------------------------------------
    _validate_unary(config, dim_types, exec_types, dim_sizes, strides_in0, strides_out)

    # -- classify dimensions --------------------------------------------------
    shared_loop_ids = []
    seq_loop_ids = []
    prim_ids = []

    for i in range(num_dims):
        et = exec_types[i]
        if et == ExecType.shared:
            shared_loop_ids.append(i)
        elif et == ExecType.seq:
            seq_loop_ids.append(i)
        elif et == ExecType.prim:
            prim_ids.append(i)

    # -- kernel tile sizes (padded to next power of 2) ------------------------
    kernel_shape = tuple(_next_pow2(dim_sizes[i]) for i in prim_ids)

    # -- grid -----------------------------------------------------------------
    grid_size = _product(dim_sizes[i] for i in shared_loop_ids)

    shared_loop_strides = _compute_shared_loop_strides(shared_loop_ids, dim_sizes)

    # -- canonical dim ordering (shared → seq → prim) -------------------------
    view_order = list(shared_loop_ids) + list(seq_loop_ids) + list(prim_ids)

    # -- build per-tensor views -----------------------------------------------
    shape_in0, strides_in0_tv, tile_in0, imap_in0 = _build_tensor_view(
        strides_in0,
        view_order,
        exec_types,
        dim_sizes,
        shared_loop_ids,
        seq_loop_ids,
    )
    shape_out, strides_out_tv, tile_out, imap_out = _build_tensor_view(
        strides_out,
        view_order,
        exec_types,
        dim_sizes,
        shared_loop_ids,
        seq_loop_ids,
    )

    order_in0 = tuple(range(len(shape_in0)))
    order_out = tuple(range(len(shape_out)))

    return UnaryConfigAnalysis(
        config=config,
        storage_dtype_name=_storage_dtype(config.data_type),
        prim_main=config.prim_main,
        shared_loop_ids=tuple(shared_loop_ids),
        seq_loop_ids=tuple(seq_loop_ids),
        prim_ids=tuple(prim_ids),
        kernel_shape=kernel_shape,
        grid_size=grid_size,
        shared_loop_strides=tuple(shared_loop_strides),
        tensor_shape_in0=shape_in0,
        tensor_shape_out=shape_out,
        tensor_strides_in0=strides_in0_tv,
        tensor_strides_out=strides_out_tv,
        tensor_order_in0=order_in0,
        tensor_order_out=order_out,
        index_map_in0=imap_in0,
        index_map_out=imap_out,
        load_tile_shape_in0=tile_in0,
        store_tile_shape_out=tile_out,
        dim_sizes=dim_sizes,
        dim_types=dim_types,
        exec_types=exec_types,
        strides_in0=strides_in0,
        strides_out=strides_out,
    )


# ---------------------------------------------------------------------------
# Unary validation
# ---------------------------------------------------------------------------


def _validate_unary(
    config: TensorOperationConfig,
    dim_types: Tuple[DimType, ...],
    exec_types: Tuple[ExecType, ...],
    dim_sizes: Tuple[int, ...],
    strides_in0: Tuple[int, ...],
    strides_out: Tuple[int, ...],
) -> None:
    """Raise ``ValueError`` if *config* violates tileir unary constraints."""
    num_dims = len(dim_types)

    # Length consistency
    if not (
        len(exec_types)
        == len(dim_sizes)
        == len(strides_in0)
        == len(strides_out)
        == num_dims
    ):
        raise ValueError(
            "Lengths of dim_types, exec_types, dim_sizes, and strides "
            "must all be equal."
        )

    # Strides must have exactly 2 tensors (in0, out)
    if len(config.strides[0]) != 2:
        raise ValueError(
            f"Unary operations require exactly 2 tensors in strides "
            f"(in0, out), got {len(config.strides[0])}."
        )

    # All dim_types must be c
    for i in range(num_dims):
        if dim_types[i] != DimType.c:
            raise ValueError(
                f"Unary operations require all dim_types to be 'c', "
                f"but dim_types[{i}] is {dim_types[i]}."
            )

    # prim_first and prim_last must be none
    if config.prim_first != PrimType.none:
        raise ValueError(
            f"Unary operations require prim_first to be 'none', "
            f"got {config.prim_first}."
        )
    if config.prim_last != PrimType.none:
        raise ValueError(
            f"Unary operations require prim_last to be 'none', got {config.prim_last}."
        )

    # prim_main must be copy or relu
    if config.prim_main not in (PrimType.copy, PrimType.relu):
        raise ValueError(
            f"Unary operations require prim_main to be copy or relu, "
            f"got {config.prim_main}."
        )

    # Positive sizes, non-negative strides
    for i in range(num_dims):
        if dim_sizes[i] <= 0:
            raise ValueError(f"dim_sizes[{i}] must be positive.")
        if strides_in0[i] < 0 or strides_out[i] < 0:
            raise ValueError(f"Strides at index {i} must be non-negative.")

    # Must have at least one prim dimension
    prim_count = sum(1 for i in range(num_dims) if exec_types[i] == ExecType.prim)
    if prim_count < 1:
        raise ValueError("At least one prim dimension is required.")

    # Prim dimensions must be innermost; no shared after seq
    _validate_ordering(exec_types)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_ordering(exec_types: Tuple[ExecType, ...]) -> None:
    """Check that prim dims are innermost and no shared follows seq.

    This validation is shared between ``_validate_binary`` and
    ``_validate_unary``.

    Raises:
        ValueError: If prim dims are not innermost or a shared dim
            follows a seq dim.
    """
    num_dims = len(exec_types)

    # Prim dimensions must be innermost
    non_prim_seen = False
    for i in range(num_dims - 1, -1, -1):
        if exec_types[i] == ExecType.prim:
            if non_prim_seen:
                raise ValueError("Prim dimensions must be the innermost dimensions.")
        else:
            non_prim_seen = True

    # No shared after seq
    seq_seen = False
    for i in range(num_dims):
        if exec_types[i] == ExecType.seq:
            seq_seen = True
        elif exec_types[i] == ExecType.shared and seq_seen:
            raise ValueError("No shared dimension may follow a sequential dimension.")


def _build_tensor_view(
    config_strides: Tuple[int, ...],
    view_order: Sequence[int],
    exec_types: Tuple[ExecType, ...],
    dim_sizes: Tuple[int, ...],
    shared_loop_ids: Sequence[int],
    seq_loop_ids: Sequence[int],
) -> Tuple[
    Tuple[int, ...],  # shape
    Tuple[int, ...],  # strides
    Tuple[int, ...],  # tile_shape (1 for non-prim, padded size for prim)
    Tuple[IndexMapEntry, ...],  # index_map
]:
    """Build a per-tensor view from *config_strides* over *view_order*.

    Dims where the tensor's config stride is zero are dropped.  For each
    kept dimension the function records the static shape, stride, tile
    extent (power-of-2 for prim, 1 otherwise), and an index-map entry
    that tells the IR builder which loop variable feeds this position.

    This helper is shared between ``analyze_binary_config`` and
    ``analyze_unary_config``.

    Args:
        config_strides: Per-dim strides for a single tensor.
        view_order: Ordered config indices (shared → seq → prim).
        exec_types: Exec types from the config.
        dim_sizes: Dimension sizes from the config.
        shared_loop_ids: Config indices with ``exec == shared``.
        seq_loop_ids: Config indices with ``exec == seq``.

    Returns:
        ``(shape, strides, tile_shape, index_map)`` tuples.
    """
    shape: list[int] = []
    strides: list[int] = []
    tile_shape: list[int] = []
    index_map: list[IndexMapEntry] = []

    for _vo_pos, cfg_idx in enumerate(view_order):
        s = config_strides[cfg_idx]
        if s == 0:
            continue  # skip dims where this tensor has stride=0

        et = exec_types[cfg_idx]
        shape.append(dim_sizes[cfg_idx])
        strides.append(s)

        if et == ExecType.prim:
            tile_shape.append(_next_pow2(dim_sizes[cfg_idx]))
            index_map.append(("prim", 0))
        else:
            tile_shape.append(1)
            if et == ExecType.shared:
                shared_pos = list(shared_loop_ids).index(cfg_idx)
                index_map.append(("shared", shared_pos))
            elif et == ExecType.seq:
                seq_offset = list(seq_loop_ids).index(cfg_idx)
                index_map.append(("seq", seq_offset))

    return (
        tuple(shape),
        tuple(strides),
        tuple(tile_shape),
        tuple(index_map),
    )


def _product(iterable) -> int:
    """Return the product of elements in *iterable*, defaulting to 1."""
    result = 1
    for x in iterable:
        result *= x
    return result


def _next_pow2(x: int) -> int:
    """Return the smallest power of 2 that is >= *x*.

    The tileiras compiler requires all tile shape dimensions to be powers
    of 2.  Non-power-of-2 prim dimensions are rounded up here and the
    extra elements are handled by ``PaddingMode.ZERO`` on ``TileLoad``.
    """
    if x <= 0:
        return 1
    power = 1
    while power < x:
        power *= 2
    return power


def _compute_shared_loop_strides(
    shared_loop_ids: Sequence[int],
    dim_sizes: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Compute row-major strides for the linearised grid.

    The innermost shared dimension has stride 1; each outer dimension's
    stride is the product of all inner shared dimension sizes.
    """
    n = len(shared_loop_ids)
    if n == 0:
        return ()
    strides = [0] * n
    stride = 1
    for i in range(n - 1, -1, -1):
        strides[i] = stride
        stride *= dim_sizes[shared_loop_ids[i]]
    return tuple(strides)
