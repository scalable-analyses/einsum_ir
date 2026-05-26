"""Parallelism passes: M/N tile balancing and parallel-iteration assignment."""

from __future__ import annotations

import logging
from typing import Any

from etops.analyses import dim_roles
from etops.diag import TeirPassError, TeirValidationError
from etops.ir import Teir
from etops.ir.cursors import iteration_nodes_outer_first, parent_map
from etops.passes._framework import PassContext
from etops.passes._iteration import _find_divisor_close_to, _node_id_for_axis
from etops.transforms import set_policy, split_iteration

_LOG = logging.getLogger(__name__)

__all__ = ["AssignParallelism", "BalanceMNTiles"]


def BalanceMNTiles(teir: Teir, ctx: PassContext) -> Teir:
    """Joint M/N tile balancing against the per-thread L2 budget.

    For each Contraction, computes the per-thread output-tile footprint
    as the product of every M/N axis the worker traverses — the role-axis
    inner tile times the schedule-iterated M/N ancestors of the invocation,
    multiplied by the output element width. When the footprint exceeds the
    budget, the pass splits the largest schedule-iterated M or N ancestor;
    this shrinks the footprint visibly for the next iteration of the loop.
    """

    profile = ctx.profile
    backend = profile.backend
    budget = max(
        int(
            backend.parallel_l2_fraction
            * (profile.microarch.l2_bytes / max(profile.num_threads, 1))
        ),
        1,
    )
    slack = backend.divisor_slack
    out_tensor = _output_tensor_id(teir)
    if out_tensor is None:
        return teir
    out_width = teir.tensors[out_tensor].dtype.bytes
    roles = ctx.analyses.get(dim_roles, teir)
    current = teir
    for pid, prim in list(current.primitives.items()):
        if prim.operation != "Contraction":
            continue
        kernel_bytes = _per_thread_tile_bytes(current, pid, prim, roles, out_width)
        if kernel_bytes <= budget:
            continue
        axis_id = _largest_schedule_mn_axis_in_cone(current, pid, roles)
        if axis_id is None:
            _LOG.warning(
                f"BalanceMNTiles: primitive {pid!r} per-thread M/N tile"
                f" footprint ({kernel_bytes} bytes) exceeds per-thread L2"
                f" budget ({budget} bytes) and no schedule M/N axis is"
                " available to split further"
            )
            continue
        per_iter = max(kernel_bytes // max(current.axes[axis_id].extent, 1), 1)
        shrink_target = max(budget // per_iter, 2)
        divisor = _find_divisor_close_to(
            current.axes[axis_id].extent, shrink_target, slack
        )
        if divisor is None:
            continue
        node_id = _node_id_for_axis(current, axis_id)
        if node_id is None:
            continue
        try:
            current = split_iteration(current, node_id, inner_extent=divisor)
            roles = ctx.analyses.get(dim_roles, current)
        except TeirPassError as exc:
            _LOG.warning(f"BalanceMNTiles: split of {axis_id!r} failed: {exc}")
    return current


def _output_tensor_id(teir: Teir) -> str | None:
    tensor_ids = teir.tensor_ids
    if not tensor_ids:
        return None
    if "out" in tensor_ids:
        return "out"
    return tensor_ids[-1]


def _ancestor_iter_axes(teir: Teir, primitive_id: str) -> list[str]:
    """Return the axes iterated by every schedule ancestor of any invocation
    that dispatches ``primitive_id``. Each axis appears at most once, in
    outermost-first order of first occurrence."""

    target_invs = {
        nid
        for nid, inv in teir.schedule.invocations.items()
        if inv.primitive == primitive_id
    }
    if not target_invs:
        return []
    parents = parent_map(teir)
    sched = teir.schedule
    seen: set[str] = set()
    ordered: list[str] = []
    for inv_id in target_invs:
        chain: list[str] = []
        cur: str | None = parents.get(inv_id)
        while cur is not None:
            node = sched.iterations.get(cur)
            if node is None:
                break
            chain.append(node.axis)
            cur = parents.get(cur)
        for ax in reversed(chain):
            if ax not in seen:
                seen.add(ax)
                ordered.append(ax)
    return ordered


def _per_thread_tile_bytes(
    teir: Teir,
    primitive_id: str,
    prim: Any,
    roles: Any,
    out_width: int,
) -> int:
    """Bytes touched by one parallel worker for ``primitive_id``'s output
    tile. Sums the role-axis tile times the product of every M/N schedule
    ancestor that has *not* been marked parallel — those become the loop
    nest each worker traverses."""

    role_m = 1
    role_n = 1
    for aid in prim.axes.get("M", ()):
        ax = teir.axes.get(aid)
        if ax is not None:
            role_m *= ax.extent
    for aid in prim.axes.get("N", ()):
        ax = teir.axes.get(aid)
        if ax is not None:
            role_n *= ax.extent

    sched_extent = 1
    for ax_id in _ancestor_iter_axes(teir, primitive_id):
        if roles.by_axis.get(ax_id, "_") not in ("M", "N"):
            continue
        ax = teir.axes.get(ax_id)
        if ax is None:
            continue
        sched_extent *= ax.extent
    return role_m * role_n * sched_extent * out_width


def _largest_schedule_mn_axis_in_cone(
    teir: Teir, primitive_id: str, roles: Any
) -> str | None:
    """Largest M/N schedule axis that is an ancestor of any invocation of
    ``primitive_id``. Splitting one of these reduces the per-worker tile
    footprint reported by `_per_thread_tile_bytes`."""

    best: tuple[int, str] | None = None
    for axis_id in _ancestor_iter_axes(teir, primitive_id):
        if roles.by_axis.get(axis_id, "_") not in ("M", "N"):
            continue
        ax = teir.axes.get(axis_id)
        if ax is None:
            continue
        if best is None or ax.extent > best[0]:
            best = (ax.extent, axis_id)
    return best[1] if best else None


def AssignParallelism(teir: Teir, ctx: PassContext) -> Teir:
    """Mark a chain of outer M/N/C iteration nodes ``policy=parallel``.

    Walks the schedule outer-first looking for an eligible iteration node
    (role ``M``/``N``/``C``, no guard, not already parallel). From that
    head, extends the chain downward through single-child iteration nodes
    that meet the same eligibility criteria, stopping as soon as the chain
    cumulative extent clears ``num_threads * parallel_min_fanout`` or the
    chain hits an ineligible link. The runtime can interpret the chain as a
    single collapsed parallel domain, so emitting a chain rather than a
    single node lets narrow outer axes contribute their extents jointly.

    The chain head's ``threading.num_threads`` metadata is set to
    ``min(num_threads, cumulative_extent)``; intermediate nodes carry the
    ``parallel`` policy but no thread budget of their own (the head's
    budget governs the whole collapsed domain).
    """

    profile = ctx.profile
    if profile.num_threads <= 1:
        return teir
    roles = ctx.analyses.get(dim_roles, teir)
    threshold = profile.num_threads * profile.backend.parallel_min_fanout

    chain = _find_collapse_chain(teir, roles, threshold)
    if chain is None:
        return teir

    cumulative_extent = 1
    for nid in chain:
        cumulative_extent *= teir.axes[teir.schedule.iterations[nid].axis].extent
    chosen = min(profile.num_threads, cumulative_extent)

    current = teir
    try:
        for nid in chain:
            current = set_policy(current, nid, "parallel")
        builder = current.builder()
        head_id = chain[0]
        meta = dict(current.schedule.iterations[head_id].metadata)
        meta["threading.num_threads"] = int(chosen)
        builder.set_iteration_metadata(head_id, meta)
        current = builder.finish(validate=False)
    except (TeirPassError, TeirValidationError) as exc:
        _LOG.warning(f"AssignParallelism: chain marking failed: {exc}")
        return teir
    return current


def _find_collapse_chain(teir: Teir, roles: Any, threshold: int) -> list[str] | None:
    """Return the outermost iteration-node chain whose cumulative extent
    clears ``threshold``, or ``None`` if no chain qualifies.

    A chain link is eligible if it iterates an M/N/C axis, carries no
    guard, and is not already parallel. Extension stops at the first
    non-single-child link or the first ineligible child; the inner
    ``while`` loop also stops as soon as the running product clears
    ``threshold``, so the resulting chain is the smallest collapse domain
    sufficient to feed the threading budget.
    """

    sched = teir.schedule
    for head_id in iteration_nodes_outer_first(teir):
        head = sched.iterations.get(head_id)
        if head is None or not _link_eligible(head, roles):
            continue
        cumulative = teir.axes[head.axis].extent
        if cumulative <= 0:
            continue
        chain = [head_id]
        current_node = head
        while cumulative < threshold and len(current_node.children) == 1:
            child_id = current_node.children[0]
            child = sched.iterations.get(child_id)
            if child is None or not _link_eligible(child, roles):
                break
            chain.append(child_id)
            cumulative *= teir.axes[child.axis].extent
            current_node = child
        if cumulative >= threshold:
            return chain
    return None


def _link_eligible(node: Any, roles: Any) -> bool:
    if node.policy == "parallel" or node.guard is not None:
        return False
    return roles.by_axis.get(node.axis, "_") in ("M", "N", "C")
