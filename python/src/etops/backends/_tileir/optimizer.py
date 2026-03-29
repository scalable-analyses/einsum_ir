"""
Optimizer for the tileir backend.

Determines optimal ``exec_types`` and dimension ordering for GPU execution
via the tileir (direct TileIR construction) backend.

The optimizer is structured as a transformation pipeline that operates on
``TensorOperationConfig`` objects using general-purpose transforms
(``split_dim``, ``fuse_dims``, ``reorder_dims``) from the
``transforms`` module.

**Binary pipeline** (``_optimize_binary``):

1. Add synthetic size-1 prim dims for any missing M, N, or K type.
2. Split oversized dims (``size > max_prim_dim_size``) via factorization.
3. Select one prim representative per GEMM role (M, N, K).
4. Assign exec types: M/N/C outers -> shared, K outers -> seq.
5. Reorder: shared -> seq -> prim (N, K, M innermost).
6. Fuse adjacent compatible dims.

**Unary pipeline** (``_optimize_unary``):

1. Split oversized dims (``size > max_prim_dim_size``) via factorization.
2. Select prim dims (unit-stride first, then fill by smallest stride,
   subject to ``max_prim_tile_volume`` and ``max_prim_dims``).
3. Apply cuda.tile assembler crash workaround.
4. Reorder: shared -> seq -> prim.
5. Fuse adjacent compatible dims.
"""

from __future__ import annotations

__all__ = ["optimize"]

import math
from dataclasses import replace
from typing import List, Optional, Tuple

from etops.backends._tileir.config_analysis import _next_pow2

import etops

from etops.backends._tileir.transforms import (
    fuse_dims,
    reorder_dims,
    split_dim,
)


def _largest_factor_leq(n: int, cap: int) -> int:
    """Return the largest factor of *n* that is <= *cap*.

    If *n* is prime and greater than *cap*, returns 1 (the trivial
    factor).  This is acceptable because a size-1 "outer" dimension
    acts as a synthetic no-op dimension.

    Args:
        n: Positive integer to factorize.
        cap: Upper bound (inclusive) for the returned factor.

    Returns:
        Largest integer *f* such that ``1 <= f <= cap`` and ``n % f == 0``.

    Raises:
        ValueError: If *n* or *cap* is less than 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    if n <= cap:
        return n

    # Check divisors from cap downward.  For small caps this is fast;
    # for large caps we only need to search up to sqrt(n) from below.
    best = 1
    sqrt_n = int(math.isqrt(n))
    for d in range(1, sqrt_n + 1):
        if n % d == 0:
            if d <= cap:
                best = max(best, d)
            complement = n // d
            if complement <= cap:
                best = max(best, complement)
    return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_prim(
    indices: List[int],
    dim_type: etops.types.DimType,
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
) -> int:
    """Select the prim representative from a group of dimension indices.

    Prefers the dimension with the smallest non-zero stride in its
    primary tensor (M -> in0, N -> in1, K -> min of both).  Ties are
    broken by preferring the smaller index for stability.

    Args:
        indices: Non-empty list of dimension indices in the same group.
        dim_type: The GEMM role of this group (``dim.m``, ``dim.n``,
            or ``dim.k``).
        strides_in0: Level-0 strides for the left input tensor.
        strides_in1: Level-0 strides for the right input tensor.

    Returns:
        Index of the selected prim representative.

    Raises:
        ValueError: If *indices* is empty.
    """
    if not indices:
        raise ValueError("Cannot select prim from empty group")
    if len(indices) == 1:
        return indices[0]

    def primary_stride(i: int) -> Tuple[float, int]:
        if dim_type == etops.dim.m:
            s = strides_in0[i]
        elif dim_type == etops.dim.n:
            s = strides_in1[i]
        else:
            s0 = strides_in0[i]
            s1 = strides_in1[i]
            s = min(s0, s1) if s0 > 0 and s1 > 0 else (s0 if s0 > 0 else s1)
        effective_stride = s if s > 0 else float("inf")
        return (effective_stride, i)

    return min(indices, key=primary_stride)


def _sort_by_stride_desc(
    indices: List[int],
    dim_type: etops.types.DimType,
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
) -> List[int]:
    """Sort *indices* descending by primary-tensor stride.

    Outermost (largest-stride) dimensions come first so they map to
    the outermost grid/loop positions for memory locality.

    For dim.c, the maximum non-zero stride across all three tensors is
    used; for dim.m, the in0 stride; for dim.n, the in1 stride; for
    dim.k, the maximum across in0 and in1.

    Args:
        indices: Dimension indices to sort.
        dim_type: GEMM role (``dim.c``, ``dim.m``, ``dim.n``, or
            ``dim.k``).
        strides_in0: Level-0 strides for the left input tensor.
        strides_in1: Level-0 strides for the right input tensor.
        strides_out: Level-0 strides for the output tensor.

    Returns:
        Sorted copy of *indices* (descending by stride).
    """

    def order_key(i: int) -> int:
        if dim_type == etops.dim.c:
            s0 = strides_in0[i] if strides_in0[i] > 0 else 0
            s1 = strides_in1[i] if strides_in1[i] > 0 else 0
            so = strides_out[i] if strides_out[i] > 0 else 0
            return max(s0, s1, so)
        elif dim_type == etops.dim.m:
            return strides_in0[i] if strides_in0[i] > 0 else 0
        elif dim_type == etops.dim.n:
            return strides_in1[i] if strides_in1[i] > 0 else 0
        else:
            s0 = strides_in0[i] if strides_in0[i] > 0 else 0
            s1 = strides_in1[i] if strides_in1[i] > 0 else 0
            return max(s0, s1) if (s0 > 0 or s1 > 0) else 0

    return sorted(indices, key=order_key, reverse=True)


def _interleave_evenly(
    major: List[int],
    minor: List[int],
) -> List[int]:
    """Bresenham-style interleaving of two sequences.

    Distributes *minor* elements as evenly as possible among *major*
    elements.  When ``len(major) >= len(minor)`` the result is a
    single list where minor elements are spaced roughly evenly within
    the major elements.

    Args:
        major: The dominant (more numerous) sequence.
        minor: The subordinate (fewer) sequence.

    Returns:
        Interleaved list of length ``len(major) + len(minor)``.
    """
    if not major:
        return minor[:]
    if not minor:
        return major[:]
    total = len(major) + len(minor)
    result: List[int] = []
    i_maj = 0
    i_min = 0
    for pos in range(total):
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
    """Interleave shared M and N dimensions for balanced grid mapping.

    M dimensions are always treated as the minor (interleaved) group
    regardless of count, to keep N-strided accesses contiguous in the
    outer grid positions.

    Args:
        shared_m: Shared M dimension indices (sorted by stride desc).
        shared_n: Shared N dimension indices (sorted by stride desc).

    Returns:
        Interleaved list with N as major and M as minor.
    """
    if len(shared_m) >= len(shared_n):
        return _interleave_evenly(shared_m, shared_n)
    return _interleave_evenly(shared_n, shared_m)


def _add_synthetic_prim_dims(
    config: object,
) -> object:
    """Add synthetic size-1 prim dimensions for missing M, N, or K types.

    Returns a new config with any missing GEMM roles appended as size-1
    dimensions.  Stride patterns follow GEMM semantics:

    * M: present in in0 and out (stride 1), absent from in1 (stride 0).
    * N: present in in1 and out (stride 1), absent from in0 (stride 0).
    * K: present in in0 and in1 (stride 1), absent from out (stride 0).

    The initial ``exec_type`` for synthetic dims is ``seq``; the caller
    reassigns all exec types in the subsequent pipeline step.

    Args:
        config: Input configuration (binary contraction).

    Returns:
        New config with any missing M/N/K dims appended.
    """
    dim_types = list(config.dim_types)
    dim_sizes = list(config.dim_sizes)
    exec_types = list(config.exec_types)

    # Extract level-0 strides (only level we manipulate).
    strides_in0 = list(config.strides[0][0])
    strides_in1 = list(config.strides[0][1])
    strides_out = list(config.strides[0][2])

    existing_types = set(dim_types)
    synthetics = [
        # (dim_type, s_in0, s_in1, s_out)
        (etops.dim.m, 1, 0, 1),
        (etops.dim.n, 0, 1, 1),
        (etops.dim.k, 1, 1, 0),
    ]
    for dim_t, s0, s1, so in synthetics:
        if dim_t not in existing_types:
            dim_types.append(dim_t)
            dim_sizes.append(1)
            exec_types.append(etops.exec.seq)
            strides_in0.append(s0)
            strides_in1.append(s1)
            strides_out.append(so)

    return replace(
        config,
        dim_types=tuple(dim_types),
        exec_types=tuple(exec_types),
        dim_sizes=tuple(dim_sizes),
        strides=((tuple(strides_in0), tuple(strides_in1), tuple(strides_out)),),
    )


def _split_oversized_dims(
    config: object,
    max_prim_dim_size: int,
) -> object:
    """Split every dimension whose size exceeds *max_prim_dim_size*.

    Iterates from the last dimension to the first so that earlier
    indices are not invalidated by insertions.

    Args:
        config: Input configuration.
        max_prim_dim_size: Maximum raw size for any single dimension.

    Returns:
        New config with oversized dims split into outer/inner pairs.
    """
    result = config
    # Work backwards so indices remain valid after each split.
    dim_idx = len(result.dim_sizes) - 1
    while dim_idx >= 0:
        if result.dim_sizes[dim_idx] > max_prim_dim_size:
            inner = _largest_factor_leq(result.dim_sizes[dim_idx], max_prim_dim_size)
            if inner == 1:
                # Dimension is prime and > cap — cannot split further.
                # The outer stays at the original size; move on.
                dim_idx -= 1
                continue
            result = split_dim(result, dim_idx, inner)
            # After split, dim_idx is the outer and dim_idx+1 is the
            # inner.  The outer might still be > cap, so re-check it
            # (don't decrement).
            continue
        dim_idx -= 1
    return result


def _try_fuse_adjacent(config: object) -> object:
    """Fuse adjacent dimensions that share type, exec_type, and strides.

    Scans from left to right, fusing greedily.  A successful fuse
    reduces the dimension count by 1 and re-checks the same position.

    Args:
        config: Input configuration.

    Returns:
        New config with compatible adjacent dims fused.
    """
    result = config
    idx = 0
    while idx < len(result.dim_sizes) - 1:
        dim_a = idx
        dim_b = idx + 1

        # Same dim_type and exec_type?
        if (
            result.dim_types[dim_a] != result.dim_types[dim_b]
            or result.exec_types[dim_a] != result.exec_types[dim_b]
        ):
            idx += 1
            continue

        # Stride compatibility check across all tensors at all levels.
        size_b = result.dim_sizes[dim_b]
        compatible = True
        for level in result.strides:
            for tensor_strides in level:
                stride_a = tensor_strides[dim_a]
                stride_b = tensor_strides[dim_b]
                if not (
                    (stride_a == 0 and stride_b == 0) or stride_a == size_b * stride_b
                ):
                    compatible = False
                    break
            if not compatible:
                break

        if compatible:
            result = fuse_dims(result, dim_a, dim_b)
            # Don't advance — the fused dim might be fusable with the
            # next one too.
        else:
            idx += 1

    return result


# ---------------------------------------------------------------------------
# Binary pipeline
# ---------------------------------------------------------------------------


def _optimize_binary_impl(
    config: object,
    max_grid: int,
) -> object:
    """Core binary optimization: assign exec_types and reorder dims.

    Operates on a config that has already been through synthetic-dim
    addition and oversized-dim splitting.

    Steps (continuing from the outer ``_optimize_binary`` pipeline):

    3. Select one prim representative per role (M, N, K).
    4. Assign exec types: prim reps -> prim, K outers -> seq,
       M/N/C outers -> shared (subject to max_grid).
    5. Reorder: shared -> seq -> prim (N, K, M innermost).
    6. Fuse adjacent compatible dims.

    Args:
        config: Input configuration (post-split, post-synthetic).
        max_grid: Maximum CUDA grid size.

    Returns:
        Optimized config with exec types assigned and dims reordered.
    """
    dim_types = tuple(config.dim_types)
    dim_sizes = tuple(config.dim_sizes)
    strides_in0 = tuple(config.strides[0][0])
    strides_in1 = tuple(config.strides[0][1])
    strides_out = tuple(config.strides[0][2])

    # Classify indices by dim_type.
    c_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.c]
    m_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.m]
    n_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.n]
    k_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.k]

    # Step 3: select prim representatives.
    p_m = _select_prim(m_indices, etops.dim.m, strides_in0, strides_in1)
    p_n = _select_prim(n_indices, etops.dim.n, strides_in0, strides_in1)
    p_k = _select_prim(k_indices, etops.dim.k, strides_in0, strides_in1)
    prim_set = {p_m, p_n, p_k}

    # Step 4: classify remaining dims.
    shared_m = [i for i in m_indices if i != p_m]
    shared_n = [i for i in n_indices if i != p_n]
    seq_k = [i for i in k_indices if i != p_k]

    # Sort within groups by stride for memory locality.
    c_ord = _sort_by_stride_desc(
        c_indices,
        etops.dim.c,
        strides_in0,
        strides_in1,
        strides_out,
    )
    m_ord = _sort_by_stride_desc(
        shared_m,
        etops.dim.m,
        strides_in0,
        strides_in1,
        strides_out,
    )
    n_ord = _sort_by_stride_desc(
        shared_n,
        etops.dim.n,
        strides_in0,
        strides_in1,
        strides_out,
    )
    k_ord = _sort_by_stride_desc(
        seq_k,
        etops.dim.k,
        strides_in0,
        strides_in1,
        strides_out,
    )

    # Interleave shared M and N for balanced grid dimensions.
    mn_ord = _interleave_shared_mn(m_ord, n_ord)
    shared_ord = c_ord + mn_ord

    # Apply max_grid limit: demote excess shared dims to seq.
    grid_size = 1
    keep_shared: List[int] = []
    demote_to_seq: List[int] = []
    for idx in shared_ord:
        dim_size = dim_sizes[idx]
        if grid_size * dim_size > max_grid:
            demote_to_seq.append(idx)
        else:
            grid_size *= dim_size
            keep_shared.append(idx)

    all_seq = demote_to_seq + k_ord

    # Step 5: build final ordering — shared -> seq -> prim (N, K, M).
    new_order = tuple(keep_shared + all_seq + [p_n, p_k, p_m])

    # Build exec_type assignments.
    all_seq_set = set(all_seq)
    exec_types: List[etops.types.ExecType] = []
    for j in new_order:
        if j in prim_set:
            exec_types.append(etops.exec.prim)
        elif j in all_seq_set:
            exec_types.append(etops.exec.seq)
        else:
            exec_types.append(etops.exec.shared)

    # Apply reorder via transform, then override exec_types.
    reordered = reorder_dims(config, new_order)
    reordered = replace(reordered, exec_types=tuple(exec_types))

    # Step 6: fuse adjacent compatible dims.
    return _try_fuse_adjacent(reordered)


# ---------------------------------------------------------------------------
# Unary pipeline
# ---------------------------------------------------------------------------


def _optimize_unary_impl(
    config: object,
    max_grid: int,
    max_prim_dims: int,
    max_prim_tile_volume: int,
) -> object:
    """Core unary optimization: assign exec_types and reorder dims.

    Operates on a config that has already been through oversized-dim
    splitting.

    Steps (continuing from the outer ``_optimize_unary`` pipeline):

    2. Select prim dims: unit-stride first, then fill by smallest
       min-stride subject to volume cap.
    3. Apply cuda.tile assembler crash workaround.
    4. Remaining dims: shared (up to max_grid), then seq.
    5. Reorder: shared -> seq -> prim.
    6. Fuse adjacent compatible dims.

    Args:
        config: Input configuration (post-split).
        max_grid: Maximum CUDA grid size.
        max_prim_dims: Maximum number of prim dimensions.
        max_prim_tile_volume: Maximum padded tile volume for prim dims.

    Returns:
        Optimized config with exec types assigned and dims reordered.
    """
    dim_sizes = tuple(config.dim_sizes)
    # Unary configs have 2 tensors: [0]=in, [1]=out.
    strides_in0 = tuple(config.strides[0][0])
    strides_out = tuple(config.strides[0][1])
    num_dims = len(dim_sizes)
    all_indices = list(range(num_dims))

    # Score each dim by min non-zero stride across in0 and out.
    def _min_stride(i: int) -> float:
        s0 = strides_in0[i] if strides_in0[i] > 0 else float("inf")
        so = strides_out[i] if strides_out[i] > 0 else float("inf")
        return min(s0, so)

    # Step 2: identify must-have prim dims (unit-stride dims).
    must_prim: set = set()
    for i in all_indices:
        if strides_in0[i] == 1:
            must_prim.add(i)
            break
    for i in all_indices:
        if strides_out[i] == 1:
            must_prim.add(i)
            break

    # Committed padded volume from must-have prim dims.
    prim_volume = 1
    for i in must_prim:
        prim_volume *= _next_pow2(dim_sizes[i])

    # Fill up to max_prim_dims, subject to volume cap.
    candidates = sorted(
        [i for i in all_indices if i not in must_prim],
        key=_min_stride,
    )
    prim_set = set(must_prim)
    for idx in candidates:
        if len(prim_set) >= max_prim_dims:
            break
        padded = _next_pow2(dim_sizes[idx])
        if prim_volume * padded > max_prim_tile_volume:
            continue
        prim_volume *= padded
        prim_set.add(idx)

    # Ensure at least 1 prim dim.
    if not prim_set:
        prim_set.add(all_indices[-1])

    # ------------------------------------------------------------------
    # Step 3 — Workaround: cuda.tile assembler crash with padded-64
    # innermost prim dim.
    #
    # The cuda.tile assembler (v1.2.0, sm_121) crashes with return
    # code 5 when ALL of the following hold:
    #   - 4 or more prim dims
    #   - the innermost prim dim pads to exactly 64
    #   - the original size of that dim is divisible by 4
    #   - that dim has unit stride in exactly one of in0/out
    #     (asymmetric gather/scatter)
    #
    # When this is detected the problematic dim is removed from the
    # prim set so it falls through to shared or seq instead.
    #
    # TODO(cuda.tile): Remove when the assembler bug is fixed.
    # ------------------------------------------------------------------
    if len(prim_set) >= 4:
        prim_ord_check = sorted(prim_set, key=_min_stride, reverse=True)
        innermost = prim_ord_check[-1]
        padded_inner = _next_pow2(dim_sizes[innermost])
        if (
            padded_inner == 64
            and dim_sizes[innermost] % 4 == 0
            and (strides_in0[innermost] == 1) != (strides_out[innermost] == 1)
        ):
            prim_set.discard(innermost)

    prim_indices = sorted(prim_set)
    non_prim_indices = [i for i in all_indices if i not in prim_set]

    # Step 4: sort non-prim by stride desc for shared ordering.
    def _max_stride(i: int) -> int:
        s0 = strides_in0[i] if strides_in0[i] > 0 else 0
        so = strides_out[i] if strides_out[i] > 0 else 0
        return max(s0, so)

    shared_ord = sorted(non_prim_indices, key=_max_stride, reverse=True)

    # Apply max_grid limit.
    grid_size = 1
    keep_shared: List[int] = []
    demote_to_seq: List[int] = []
    for idx in shared_ord:
        dim_size = dim_sizes[idx]
        if grid_size * dim_size > max_grid:
            demote_to_seq.append(idx)
        else:
            grid_size *= dim_size
            keep_shared.append(idx)

    # Sort prim dims by stride ascending (unit-stride innermost).
    prim_ord = sorted(prim_indices, key=_min_stride, reverse=True)

    # Step 5: final ordering — shared -> seq -> prim.
    new_order = tuple(keep_shared + demote_to_seq + prim_ord)

    prim_set_final = set(prim_ord)
    seq_set = set(demote_to_seq)
    exec_types: List[etops.types.ExecType] = []
    for j in new_order:
        if j in prim_set_final:
            exec_types.append(etops.exec.prim)
        elif j in seq_set:
            exec_types.append(etops.exec.seq)
        else:
            exec_types.append(etops.exec.shared)

    # Apply reorder via transform, then override exec_types.
    reordered = reorder_dims(config, new_order)
    reordered = replace(reordered, exec_types=tuple(exec_types))

    # Step 6: fuse adjacent compatible dims.
    return _try_fuse_adjacent(reordered)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def optimize(
    config: object,
    optimization_config: Optional[dict] = None,
) -> object:
    """Optimize *config* for tileir execution.

    Dispatches to the binary or unary optimizer based on ``prim_main``.

    Args:
        config: Input ``TensorOperationConfig``.
        optimization_config: Optional overrides.  Keys are documented in
            :func:`~etops.backends.tileir.get_default_optimization_config`.

    Returns:
        Optimized ``TensorOperationConfig``.
    """
    from etops.backends.tileir import get_default_optimization_config
    from etops.types import PrimType

    defaults = get_default_optimization_config()
    opts = {**defaults, **(optimization_config or {})}
    max_grid = opts["max_grid"]

    is_unary = config.prim_main in (PrimType.copy, PrimType.relu)

    if is_unary:
        return _optimize_unary(config, opts, max_grid)
    return _optimize_binary(config, opts, max_grid)


def _optimize_binary(
    config: object,
    opts: dict,
    max_grid: int,
) -> object:
    """Optimize a binary contraction config.

    Pipeline:
        1. Add synthetic prim dims for missing M/N/K.
        2. Split oversized dims.
        3-6. Assign exec types, reorder, and fuse.

    Args:
        config: Input configuration.
        opts: Merged optimization parameters.
        max_grid: Maximum CUDA grid size.

    Returns:
        Optimized config.
    """
    max_prim_dim_size = opts["max_prim_dim_size"]

    # Step 1: add synthetic dims.
    cfg = _add_synthetic_prim_dims(config)

    # Step 2: split oversized dims.
    cfg = _split_oversized_dims(cfg, max_prim_dim_size)

    # Steps 3-6: assign exec types, reorder, fuse.
    return _optimize_binary_impl(cfg, max_grid)


def _optimize_unary(
    config: object,
    opts: dict,
    max_grid: int,
) -> object:
    """Optimize a unary operation config.

    Pipeline:
        1. Split oversized dims.
        2-6. Assign exec types (with workaround), reorder, and fuse.

    Args:
        config: Input configuration.
        opts: Merged optimization parameters.
        max_grid: Maximum CUDA grid size.

    Returns:
        Optimized config.
    """
    max_prim_dims = opts["max_prim_dims"]
    max_prim_tile_volume = opts["max_prim_tile_volume"]
    max_prim_dim_size = opts["max_prim_dim_size"]

    # Step 1: split oversized dims.
    cfg = _split_oversized_dims(config, max_prim_dim_size)

    # Steps 2-6: assign exec types, workaround, reorder, fuse.
    return _optimize_unary_impl(cfg, max_grid, max_prim_dims, max_prim_tile_volume)
