"""Iteration-level passes: chain fusion and primitive/cache tiling.

These passes operate on schedule iteration nodes — fusing contiguous
outer/inner chains, and tiling iteration sites toward role-target or
cache-budget extents. The underlying primitives are `fuse_iterations`
and `split_iteration`; the passes here are heuristic drivers that decide
*which* sites to act on and with *what* parameters.
"""

from __future__ import annotations

import logging
import math

from etops.analyses import dim_roles
from etops.diag import TeirPassError
from etops.ir import Teir
from etops.ir.dtypes import get_dtype
from etops.passes._framework import PassContext
from etops.transforms import fuse_iterations, split_iteration

_LOG = logging.getLogger(__name__)

__all__ = [
    "FuseContiguousIterations",
    "TileForCache",
    "TileForPrimitive",
]


def _node_id_for_axis(teir: Teir, axis_id: str) -> str | None:
    """Return the unique iteration node iterating ``axis_id``, or None.

    The optimization pipeline produces single-iteration-per-axis IRs by
    construction; this helper returns ``None`` if zero or multiple
    iteration sites exist, which the caller treats as "skip this axis."
    """

    node_ids = [
        nid for nid, it in teir.schedule.iterations.items() if it.axis == axis_id
    ]
    return node_ids[0] if len(node_ids) == 1 else None


def FuseContiguousIterations(teir: Teir, ctx: PassContext) -> Teir:
    """Fuse adjacent iterations in the schedule whose strides are contiguous.

    The pass walks every iteration node and repeatedly tries to fuse it
    with its sole child via `fuse_iterations`. The fuse primitive
    validates all preconditions (single-child chain, matching policies,
    stride coherency, axes not in primitive roles, well-scoped guards)
    and raises `TeirPassError` when any fails; the pass treats every
    raise as "this site isn't fusable" and moves on.
    """

    current = teir
    for nid in list(current.schedule.iterations.keys()):
        while True:
            try:
                current = fuse_iterations(current, nid)
            except TeirPassError:
                break
    return current


def TileForPrimitive(teir: Teir, ctx: PassContext) -> Teir:
    """Split schedule-iterated role axes toward the profile's target tile sizes.

    When one of the M/N roles is extent-starved (the largest axis is
    smaller than its target), the partner's target is scaled up by the
    shortfall ratio, clamped to the backend's
    ``[compensation_min, compensation_max]`` band, so the kernel's
    combined footprint stays near the cache budget.
    """

    dtype_name = next(
        (
            p.metadata["data_type"]
            for p in teir.primitives.values()
            if p.operation == "Contraction"
        ),
        None,
    )
    if dtype_name is None:
        return teir
    backend = ctx.profile.backend
    roles = ctx.analyses.get(dim_roles, teir)
    targets = ctx.profile.tile_targets(get_dtype(dtype_name))
    slack = backend.divisor_slack
    base = {"M": targets.m, "N": targets.n, "K": targets.k}

    largest_for_role: dict[str, int] = {}
    for aid, axis in teir.axes.items():
        role = roles.by_axis.get(aid, "_")
        if role in base and axis.extent > largest_for_role.get(role, 0):
            largest_for_role[role] = axis.extent
    target_for_role = dict(base)
    for partner_a, partner_b in (("M", "N"), ("N", "M")):
        shortfall = largest_for_role.get(partner_a, 0)
        t = base.get(partner_a, 0)
        if shortfall <= 0 or t <= 0 or shortfall >= t:
            continue
        boost = max(
            backend.compensation_min,
            min(t / shortfall, backend.compensation_max),
        )
        target_for_role[partner_b] = max(
            int(target_for_role[partner_b] * boost), target_for_role[partner_b]
        )

    current = teir
    candidates: list[tuple[str, str]] = []
    for it in current.schedule.iterations.values():
        role = roles.by_axis.get(it.axis, "_")
        if role in target_for_role:
            candidates.append((it.axis, role))
    for axis_id, role in candidates:
        if axis_id not in current.axes:
            continue
        extent = current.axes[axis_id].extent
        target = target_for_role[role]
        if extent <= target:
            continue
        divisor = _find_divisor_close_to(extent, target, slack)
        if divisor is None or divisor in (extent, 1):
            _LOG.warning(
                f"TileForPrimitive: cannot split axis {axis_id!r}"
                f" (extent={extent}, target={target})"
            )
            continue
        node_id = _node_id_for_axis(current, axis_id)
        if node_id is None:
            continue
        try:
            current = split_iteration(current, node_id, inner_extent=divisor)
            roles = ctx.analyses.get(dim_roles, current)
        except TeirPassError as exc:
            _LOG.warning(f"TileForPrimitive: split of {axis_id!r} failed: {exc}")
    return current


def _find_divisor_close_to(n: int, target: int, slack: float = 1.5) -> int | None:
    """Return the proper divisor of ``n`` whose ratio to ``target`` is closest
    in log space, bounded by ``target * slack`` on the upper side.

    Trivial divisors ``1`` and ``n`` are excluded: ``1`` produces a degenerate
    split (size-1 inner axis), ``n`` is a no-op.
    """

    if target <= 0 or n <= 2:
        return None
    ceiling = max(int(target * slack), 2)
    log_target = math.log(target)
    best: int | None = None
    best_dist = math.inf
    d = 2
    while d * d <= n:
        if n % d == 0:
            for cand in (d, n // d):
                if cand <= 1 or cand >= n or cand > ceiling:
                    continue
                dist = abs(math.log(cand) - log_target)
                if dist < best_dist or (
                    dist == best_dist and (best is None or cand > best)
                ):
                    best_dist = dist
                    best = cand
        d += 1
    return best


def TileForCache(teir: Teir, ctx: PassContext) -> Teir:
    """Introduce a cache-blocking layer over outer M/N and K axes.

    For each tilable axis we ask: at what inner extent does the per-tensor
    footprint touched by varying that axis fit inside the cache budget?
    The answer is ``budget // stride_on(tensor)`` per tensor; the minimum
    across operands is the inner extent we look for a divisor near.
    """

    profile = ctx.profile
    backend = profile.backend
    micro = profile.microarch
    roles = ctx.analyses.get(dim_roles, teir)
    slack = backend.divisor_slack
    l2_per_thread = max(micro.l2_bytes // max(profile.num_threads, 1), 1)
    l2_budget = int(backend.parallel_l2_fraction * l2_per_thread)
    l3_budget = int(backend.cache_block_l3_fraction * micro.l3_bytes)

    current = teir
    for ax_id, role in list(roles.by_axis.items()):
        if ax_id not in current.axes:
            continue
        if role not in ("M", "N"):
            continue
        inner = _cache_block_inner_extent(current, ax_id, l2_budget, slack)
        if inner is None:
            continue
        node_id = _node_id_for_axis(current, ax_id)
        if node_id is None:
            continue
        try:
            current = split_iteration(current, node_id, inner_extent=inner)
            roles = ctx.analyses.get(dim_roles, current)
        except TeirPassError as exc:
            _LOG.warning(f"TileForCache: L2 split of {ax_id!r} failed: {exc}")

    for ax_id, role in list(roles.by_axis.items()):
        if ax_id not in current.axes:
            continue
        if role != "K":
            continue
        inner = _cache_block_inner_extent(current, ax_id, l3_budget, slack)
        if inner is None:
            continue
        node_id = _node_id_for_axis(current, ax_id)
        if node_id is None:
            continue
        try:
            current = split_iteration(current, node_id, inner_extent=inner)
            roles = ctx.analyses.get(dim_roles, current)
        except TeirPassError as exc:
            _LOG.warning(f"TileForCache: L3 split of {ax_id!r} failed: {exc}")
    return current


def _cache_block_inner_extent(
    teir: Teir, axis_id: str, byte_budget: int, slack: float
) -> int | None:
    """Choose an inner-tile extent for ``axis_id`` under ``byte_budget``."""

    axis = teir.axes[axis_id]
    if axis.extent <= 1:
        return None
    bounds: list[int] = []
    for tid in teir.tensor_ids:
        stride = axis.stride_on(tid)
        if stride <= 0:
            continue
        bounds.append(max(byte_budget // stride, 1))
    if not bounds:
        return None
    target_block = min(bounds)
    if target_block >= axis.extent:
        return None
    inner = _find_divisor_close_to(axis.extent, target_block, slack)
    if inner is None or inner in (axis.extent, 1):
        return None
    return inner
