"""
Cutile backend implementation using cuda.tile.

This backend provides GPU-accelerated tensor operations using NVIDIA's
cuda.tile framework. It is optional and requires cuda.tile to be installed.
"""

from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from etops.config import TensorOperationConfig
    from etops.backends.base import CompiledOperation

import etops
from etops.backends._cutile.config_parser import ConfigParser
from etops.backends._cutile.jit_compiler import JitCompiler
from etops.backends._cutile.cache import InMemoryCache


# =============================================================================
# Helper Functions
# =============================================================================

def _select_prim(
    indices: List[int],
    dim_type: "etops.types.DimType",
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
) -> int:
    """
    Select the prim representative from a group of dimension indices.

    Prefers the unit-stride (or smallest non-zero stride) dimension;
    breaks ties by choosing the largest dimension size.

    Args:
        indices: List of dimension indices to select from.
        dim_type: The dimension type (m, n, or k).
        strides_in0: Strides for the first input tensor.
        strides_in1: Strides for the second input tensor.

    Returns:
        The index of the selected prim dimension.
    """
    if not indices:
        raise ValueError("Cannot select prim from empty group")

    if len(indices) == 1:
        return indices[0]

    def primary_stride(i: int) -> Tuple[int, int]:
        """Returns (stride, -size) for lex ordering."""
        if dim_type == etops.dim.m:
            s = strides_in0[i]
        elif dim_type == etops.dim.n:
            s = strides_in1[i]
        else:  # k
            s0 = strides_in0[i]
            s1 = strides_in1[i]
            s = min(s0, s1) if s0 > 0 and s1 > 0 else (s0 if s0 > 0 else s1)
        # Use infinity for zero strides
        effective_stride = s if s > 0 else float('inf')
        return (effective_stride, -1)

    return min(indices, key=primary_stride)


def _sort_by_stride_desc(
    indices: List[int],
    dim_type: "etops.types.DimType",
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
) -> List[int]:
    """
    Sort indices descending by primary-tensor stride.

    Args:
        indices: List of dimension indices to sort.
        dim_type: The dimension type (c, m, n, or k).
        strides_in0: Strides for the first input tensor.
        strides_in1: Strides for the second input tensor.
        strides_out: Strides for the output tensor.

    Returns:
        Sorted list of indices (descending by stride).
    """
    def order_key(i: int) -> int:
        if dim_type == etops.dim.c:
            s0 = strides_in0[i] if strides_in0[i] > 0 else 0
            s1 = strides_in1[i] if strides_in1[i] > 0 else 0
            so = strides_out[i] if strides_out[i] > 0 else 0
            return min(s0, s1, so) if (s0 > 0 or s1 > 0 or so > 0) else 0
        elif dim_type == etops.dim.m:
            return strides_in0[i] if strides_in0[i] > 0 else 0
        elif dim_type == etops.dim.n:
            return strides_in1[i] if strides_in1[i] > 0 else 0
        else:  # k
            s0 = strides_in0[i] if strides_in0[i] > 0 else 0
            s1 = strides_in1[i] if strides_in1[i] > 0 else 0
            return min(s0, s1) if (s0 > 0 or s1 > 0) else 0

    return sorted(indices, key=order_key, reverse=True)


def _interleave_evenly(major: List[int], minor: List[int]) -> List[int]:
    """
    Bresenham-style interleaving of two sequences.

    Distributes the minority type evenly within the majority type.

    Args:
        major: The longer list.
        minor: The shorter list.

    Returns:
        Interleaved list.
    """
    if not major:
        return minor[:]
    if not minor:
        return major[:]

    total = len(major) + len(minor)
    result = []
    i_maj = 0
    i_min = 0

    for pos in range(total):
        # Check if minor should be placed at this position
        # using Bresenham accumulation: floor((pos+1)*|minor|/total) > floor(pos*|minor|/total)
        prev_acc = pos * len(minor) // total
        curr_acc = (pos + 1) * len(minor) // total

        if i_min < len(minor) and curr_acc > prev_acc:
            result.append(minor[i_min])
            i_min += 1
        else:
            result.append(major[i_maj])
            i_maj += 1

    return result


def _interleave_shared_mn(
    shared_m: List[int],
    shared_n: List[int],
) -> List[int]:
    """
    Interleave shared M and N dimensions.

    The minority group is distributed evenly within the majority group.

    Args:
        shared_m: List of M dimension indices with exec.shared.
        shared_n: List of N dimension indices with exec.shared.

    Returns:
        Interleaved list of M and N indices.
    """
    if len(shared_m) >= len(shared_n):
        return _interleave_evenly(shared_m, shared_n)

    # N is majority, M is minority
    # Determine positions for M using Bresenham accumulation
    m_len = len(shared_m)
    n_len = len(shared_n)
    total = m_len + n_len
    minor_count = m_len

    # Find positions where M elements should be placed
    m_positions = []
    m_idx = 0
    for pos in range(total):
        prev_acc = pos * minor_count // total
        curr_acc = (pos + 1) * minor_count // total
        if curr_acc > prev_acc:
            m_positions.append(pos)
            m_idx += 1
            if m_idx >= m_len:
                break

    # Build result: M at m_positions, N fills others
    result = [None] * total
    for i, m_pos in enumerate(m_positions):
        result[m_pos] = shared_m[i]

    n_idx = 0
    for pos in range(total):
        if result[pos] is None:
            result[pos] = shared_n[n_idx]
            n_idx += 1

    return result


def _add_synthetic_prim_dims(
    dim_types: Tuple["etops.types.DimType", ...],
    dim_sizes: Tuple[int, ...],
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
) -> Tuple[
    Tuple["etops.types.DimType", ...],
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
    Tuple[int, ...],
]:
    """
    Add synthetic prim dimensions for missing M, N, or K types.

    Synthetics are size=1 with appropriate strides.

    Args:
        dim_types: Original dimension types.
        dim_sizes: Original dimension sizes.
        strides_in0: Original first input tensor strides.
        strides_in1: Original second input tensor strides.
        strides_out: Original output tensor strides.

    Returns:
        Extended dim_types, dim_sizes, strides_in0, strides_in1, strides_out.
    """
    new_dim_types = list(dim_types)
    new_dim_sizes = list(dim_sizes)
    new_s0 = list(strides_in0)
    new_s1 = list(strides_in1)
    new_so = list(strides_out)

    # Synthetic dimension templates: (DimType, s0, s1, so)
    synthetics = [
        (etops.dim.m, 1, 0, 1),  # Missing M
        (etops.dim.n, 0, 1, 1),  # Missing N
        (etops.dim.k, 1, 1, 0),  # Missing K
    ]

    for dim_t, s0, s1, so in synthetics:
        if dim_t not in dim_types:
            new_dim_types.append(dim_t)
            new_dim_sizes.append(1)
            new_s0.append(s0)
            new_s1.append(s1)
            new_so.append(so)

    return (
        tuple(new_dim_types),
        tuple(new_dim_sizes),
        tuple(new_s0),
        tuple(new_s1),
        tuple(new_so),
    )


def _optimize_config_impl(
    dim_types: Tuple["etops.types.DimType", ...],
    dim_sizes: Tuple[int, ...],
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
    max_grid: int,
) -> Tuple[Tuple[int, ...], Tuple["etops.types.ExecType", ...]]:
    """
    Core optimization algorithm for determining exec_types and dimension ordering.

    Args:
        dim_types: Dimension types.
        dim_sizes: Dimension sizes.
        strides_in0: First input tensor strides.
        strides_in1: Second input tensor strides.
        strides_out: Output tensor strides.
        max_grid: Maximum allowed grid size.

    Returns:
        Tuple of (new_order, exec_types) for the optimized configuration.
    """
    # Step 1: Bucket indices by type
    c_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.c]
    m_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.m]
    n_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.n]
    k_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.k]

    # Step 2: Select prim representatives
    p_m = _select_prim(m_indices, etops.dim.m, strides_in0, strides_in1)
    p_n = _select_prim(n_indices, etops.dim.n, strides_in0, strides_in1)
    p_k = _select_prim(k_indices, etops.dim.k, strides_in0, strides_in1)

    # Remaining indices after prim selection
    shared_m = [i for i in m_indices if i != p_m]
    shared_n = [i for i in n_indices if i != p_n]
    seq_k = [i for i in k_indices if i != p_k]

    # Step 3: Sort shared dims descending by primary-tensor stride
    # Outermost (largest stride) first, innermost (smallest stride) last
    c_ord = _sort_by_stride_desc(
        c_indices, etops.dim.c, strides_in0, strides_in1, strides_out
    )
    m_ord = _sort_by_stride_desc(
        shared_m, etops.dim.m, strides_in0, strides_in1, strides_out
    )
    n_ord = _sort_by_stride_desc(
        shared_n, etops.dim.n, strides_in0, strides_in1, strides_out
    )
    k_ord = _sort_by_stride_desc(
        seq_k, etops.dim.k, strides_in0, strides_in1, strides_out
    )

    # Step 4: Interleave shared M and N
    mn_ord = _interleave_shared_mn(m_ord, n_ord)

    # Step 5: Build shared dimension order (outermost to innermost)
    # c_ord + mn_ord gives shared dims in order they'll appear in new_order
    shared_ord = c_ord + mn_ord

    # Step 6: Demote outermost shared dims to seq if grid would exceed limit
    # We iterate through shared_ord (outermost first) and demote until grid fits
    grid_size = 1
    keep_shared = []
    demote_to_seq = []

    for idx in shared_ord:
        dim_size = dim_sizes[idx]
        if grid_size * dim_size > max_grid:
            # Demote this and all remaining shared dims to seq
            demote_to_seq.append(idx)
        else:
            grid_size *= dim_size
            keep_shared.append(idx)

    # All dims demoted to seq join the existing seq_k dims
    all_seq = demote_to_seq + k_ord

    # Step 7: Build new dimension order
    # [kept shared dims] → [demoted shared dims (now seq)] → [K seq dims] → [prims]
    new_order = keep_shared + all_seq + [p_n, p_k, p_m]

    # Step 8: Assign exec_types based on which indices are prim, shared, seq
    prim_indices = {p_m, p_n, p_k}
    keep_shared_set = set(keep_shared)
    all_seq_set = set(all_seq)

    # Build exec_types in new order
    exec_types = []
    for j in new_order:
        if j in prim_indices:
            exec_types.append(etops.exec.prim)
        elif j in all_seq_set:
            exec_types.append(etops.exec.seq)
        else:  # must be shared
            exec_types.append(etops.exec.shared)

    return tuple(new_order), tuple(exec_types)


# Module-level cache: persists across etops.compile() calls within the same process.
# Kernels with identical configurations are compiled only once.
_global_cache = InMemoryCache()


class CutileOperation:
    """
    Compiled cutile operation.

    This class wraps a compiled cuda.tile kernel and provides the execute()
    method required by the CompiledOperation protocol.
    """

    def __init__(self, kernel_module, grid_size: int):
        """
        Initialize cutile operation.

        Args:
            kernel_module: Compiled kernel module from cuda.tile
            grid_size: Grid size for kernel launch
        """
        self._kernel = kernel_module
        self._grid_size = grid_size

    def execute(
        self,
        in0,
        in1: Optional[object],
        out,
    ) -> None:
        """
        Execute the tensor operation on GPU.

        Args:
            in0: First input tensor (must be on CUDA device)
            in1: Second input tensor (None for unary operations)
            out: Output tensor (must be pre-allocated on CUDA device)
        """
        import cupy as cp
        import cuda.tile as ct

        ct.launch(
            cp.cuda.get_current_stream(),
            (self._grid_size,),
            self._kernel.contraction_kernel,
            (in0, in1, out),
        )


def create_operation(config: "TensorOperationConfig") -> CutileOperation:
    """
    Create and compile a cutile operation from configuration.

    Args:
        config: Tensor operation configuration

    Returns:
        Compiled CutileOperation instance

    Raises:
        ValueError: If config is not executable (e.g. missing prim exec_types)
        ImportError: If cuda.tile or cupy is not installed
    """
    config_parser = ConfigParser(config)
    config_parser.verify_executable_config()

    jit = JitCompiler(config_parser)
    kernel_module = _global_cache.get_or_compile(config_parser, jit)

    return CutileOperation(kernel_module, config_parser.grid_size)


def get_default_optimization_config() -> dict:
    """
    Get default optimization configuration for cutile backend.

    Returns:
        Dictionary with default optimization parameters:
        - max_grid: Maximum CUDA grid size (2^24 - 1)
    """
    return {
        "max_grid": 16777215,  # 2^24 - 1
    }


def optimize_config(
    config: "TensorOperationConfig",
    optimization_config: Optional[dict] = None,
) -> "TensorOperationConfig":
    """
    Optimize a tensor operation configuration for cutile backend.

    This optimizer determines optimal exec_types and dimension ordering
    for GPU execution via cuda.tile. The algorithm:

    1. Adds synthetic prim dimensions for any missing M/N/K types
    2. Selects prim representatives (preferring unit stride)
    3. Sorts shared dims by stride for memory locality
    4. Interleaves shared M and N dimensions
    5. Demotes outermost shared dims to seq if grid would exceed max_grid
    6. Produces ordering: [shared] → [seq] → [prims]

    Args:
        config: Tensor operation configuration to optimize.
        optimization_config: Optional dict with keys:
            - max_grid: Maximum allowed grid size (default: 2^24 - 1)

    Returns:
        Optimized TensorOperationConfig with correct exec_types and dimension order.
    """
    # Merge with defaults
    defaults = get_default_optimization_config()
    if optimization_config is not None:
        opts = {**defaults, **optimization_config}
    else:
        opts = defaults

    max_grid = opts["max_grid"]

    # Unpack strides from level 0
    strides_in0 = config.strides[0][0]
    strides_in1 = config.strides[0][1]
    strides_out = config.strides[0][2]

    # Add synthetic prim dims if M/N/K missing
    (
        new_dim_types,
        new_dim_sizes,
        new_s0,
        new_s1,
        new_so,
    ) = _add_synthetic_prim_dims(
        tuple(config.dim_types),
        tuple(config.dim_sizes),
        strides_in0,
        strides_in1,
        strides_out,
    )

    # Run core optimization algorithm
    new_order, new_exec_types = _optimize_config_impl(
        new_dim_types,
        new_dim_sizes,
        new_s0,
        new_s1,
        new_so,
        max_grid=max_grid,
    )

    # Permute arrays by new order
    # new_order is a list of ORIGINAL indices in the NEW order
    # For dim_types, sizes, strides: we index by original index to get value at that index
    permuted_dim_types = tuple(new_dim_types[i] for i in new_order)
    permuted_dim_sizes = tuple(new_dim_sizes[i] for i in new_order)

    # new_exec_types is already in the correct order (indexed by position in new_order)
    # Each position pos in new_exec_types corresponds to original index new_order[pos]
    permuted_exec_types = new_exec_types

    # Permute all stride levels
    permuted_strides = (
        (tuple(new_s0[i] for i in new_order),
        tuple(new_s1[i] for i in new_order),
        tuple(new_so[i] for i in new_order)),
    )

    # Build new config
    from etops.config import TensorOperationConfig
    return TensorOperationConfig(
        backend=config.backend,
        data_type=config.data_type,
        prim_first=config.prim_first,
        prim_main=config.prim_main,
        prim_last=config.prim_last,
        dim_types=permuted_dim_types,
        exec_types=permuted_exec_types,
        dim_sizes=permuted_dim_sizes,
        strides=permuted_strides,
    )


__all__ = [
    "create_operation",
    "CutileOperation",
    "get_default_optimization_config",
    "optimize_config",
]
