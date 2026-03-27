"""
Optimizer for the tileir backend.

Determines optimal ``exec_types`` and dimension ordering for GPU execution
via the tileir (direct TileIR construction) backend.
"""

from __future__ import annotations

__all__ = ["optimize"]

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from etops.config import TensorOperationConfig

import etops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_prim(
    indices: List[int],
    dim_type: "etops.types.DimType",
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
) -> int:
    """Select the prim representative from a group of dimension indices.

    Prefers the unit-stride (or smallest non-zero stride) dimension.
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
        return (effective_stride, -1)

    return min(indices, key=primary_stride)


def _sort_by_stride_desc(
    indices: List[int],
    dim_type: "etops.types.DimType",
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
) -> List[int]:
    """Sort *indices* descending by primary-tensor stride."""

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
        else:
            s0 = strides_in0[i] if strides_in0[i] > 0 else 0
            s1 = strides_in1[i] if strides_in1[i] > 0 else 0
            return min(s0, s1) if (s0 > 0 or s1 > 0) else 0

    return sorted(indices, key=order_key, reverse=True)


def _interleave_evenly(major: List[int], minor: List[int]) -> List[int]:
    """Bresenham-style interleaving of two sequences."""
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
    """Interleave shared M and N dimensions."""
    if len(shared_m) >= len(shared_n):
        return _interleave_evenly(shared_m, shared_n)

    m_len = len(shared_m)
    n_len = len(shared_n)
    total = m_len + n_len
    minor_count = m_len

    m_positions: List[int] = []
    m_idx = 0
    for pos in range(total):
        prev_acc = pos * minor_count // total
        curr_acc = (pos + 1) * minor_count // total
        if curr_acc > prev_acc:
            m_positions.append(pos)
            m_idx += 1
            if m_idx >= m_len:
                break

    result: List[Optional[int]] = [None] * total
    for i, m_pos in enumerate(m_positions):
        result[m_pos] = shared_m[i]

    n_idx = 0
    for pos in range(total):
        if result[pos] is None:
            result[pos] = shared_n[n_idx]
            n_idx += 1

    return result  # type: ignore[return-value]


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
    """Add synthetic prim dimensions for missing M, N, or K types."""
    new_dt = list(dim_types)
    new_ds = list(dim_sizes)
    new_s0 = list(strides_in0)
    new_s1 = list(strides_in1)
    new_so = list(strides_out)

    synthetics = [
        (etops.dim.m, 1, 0, 1),
        (etops.dim.n, 0, 1, 1),
        (etops.dim.k, 1, 1, 0),
    ]
    for dim_t, s0, s1, so in synthetics:
        if dim_t not in dim_types:
            new_dt.append(dim_t)
            new_ds.append(1)
            new_s0.append(s0)
            new_s1.append(s1)
            new_so.append(so)

    return tuple(new_dt), tuple(new_ds), tuple(new_s0), tuple(new_s1), tuple(new_so)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _optimize_impl(
    dim_types: Tuple["etops.types.DimType", ...],
    dim_sizes: Tuple[int, ...],
    strides_in0: Tuple[int, ...],
    strides_in1: Tuple[int, ...],
    strides_out: Tuple[int, ...],
    max_grid: int,
) -> Tuple[Tuple[int, ...], Tuple["etops.types.ExecType", ...]]:
    """Core optimization: determine exec_types and ordering."""
    c_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.c]
    m_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.m]
    n_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.n]
    k_indices = [i for i, dt in enumerate(dim_types) if dt == etops.dim.k]

    p_m = _select_prim(m_indices, etops.dim.m, strides_in0, strides_in1)
    p_n = _select_prim(n_indices, etops.dim.n, strides_in0, strides_in1)
    p_k = _select_prim(k_indices, etops.dim.k, strides_in0, strides_in1)

    shared_m = [i for i in m_indices if i != p_m]
    shared_n = [i for i in n_indices if i != p_n]
    seq_k = [i for i in k_indices if i != p_k]

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

    mn_ord = _interleave_shared_mn(m_ord, n_ord)
    shared_ord = c_ord + mn_ord

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
    new_order = tuple(keep_shared + all_seq + [p_n, p_k, p_m])

    prim_set = {p_m, p_n, p_k}
    all_seq_set = set(all_seq)
    exec_types = []
    for j in new_order:
        if j in prim_set:
            exec_types.append(etops.exec.prim)
        elif j in all_seq_set:
            exec_types.append(etops.exec.seq)
        else:
            exec_types.append(etops.exec.shared)

    return new_order, tuple(exec_types)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def optimize(
    config: "TensorOperationConfig",
    optimization_config: Optional[dict] = None,
) -> "TensorOperationConfig":
    """Optimize *config* for tileir execution.

    Args:
        config: Input configuration.
        optimization_config: Optional overrides (``max_grid``).

    Returns:
        Optimized ``TensorOperationConfig``.
    """
    from etops.backends.tileir import get_default_optimization_config

    defaults = get_default_optimization_config()
    opts = {**defaults, **(optimization_config or {})}
    max_grid = opts["max_grid"]

    strides_in0 = config.strides[0][0]
    strides_in1 = config.strides[0][1]
    strides_out = config.strides[0][2]

    new_dt, new_ds, new_s0, new_s1, new_so = _add_synthetic_prim_dims(
        tuple(config.dim_types),
        tuple(config.dim_sizes),
        tuple(strides_in0),
        tuple(strides_in1),
        tuple(strides_out),
    )

    new_order, new_exec_types = _optimize_impl(
        new_dt, new_ds, new_s0, new_s1, new_so, max_grid
    )

    permuted_dim_types = tuple(new_dt[i] for i in new_order)
    permuted_dim_sizes = tuple(new_ds[i] for i in new_order)
    permuted_strides = (
        (
            tuple(new_s0[i] for i in new_order),
            tuple(new_s1[i] for i in new_order),
            tuple(new_so[i] for i in new_order),
        ),
    )

    from etops.config import TensorOperationConfig

    return TensorOperationConfig(
        backend=config.backend,
        data_type=config.data_type,
        prim_first=config.prim_first,
        prim_main=config.prim_main,
        prim_last=config.prim_last,
        dim_types=permuted_dim_types,
        exec_types=new_exec_types,
        dim_sizes=permuted_dim_sizes,
        strides=permuted_strides,
    )
