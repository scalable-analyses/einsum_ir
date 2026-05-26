"""Schedule-centric axis transformations: split, fuse, set-policy."""

from __future__ import annotations

from etops.diag import TeirPassError
from etops.ir import Guard, Teir
from etops.ir._records import VALID_POLICIES

__all__ = [
    "fuse_iterations",
    "set_policy",
    "split_iteration",
]


def split_iteration(teir: Teir, iteration_node_id: str, *, inner_extent: int) -> Teir:
    """Split an iteration node into outer x inner over two new axes.

    The schedule iteration node identified by ``iteration_node_id`` is
    replaced by an outer-then-inner chain over two new axes derived from
    the original iterated axis (``axis_id + "0"`` and ``axis_id + "1"``,
    with the standard collision disambiguator). The new inner node
    inherits the original children and policy; the original's guard
    stays on the outer node (one check per outer-scope entry, cheaper
    than per-inner-entry, and the guard's referenced axes are ancestors
    of the outer too); the original's metadata is duplicated onto both
    nodes so advisory hints remain visible at either level.

    Schedule-centric semantics: this operation touches only the named
    iteration node. Other iteration nodes that may iterate the same axis
    are left alone, and primitive role lists are not rewritten. The
    original axis is left in ``teir.axes``.

    Args:
        teir: Source IR.
        iteration_node_id: Id of the iteration node to split.
        inner_extent: Inner-half extent. Must divide the iterated axis's
            extent.

    Raises:
        TeirPassError: If the node does not exist, the iterated axis
            appears in any primitive role list, ``inner_extent`` does
            not divide the extent, or the node's guard references the
            iterated axis.
    """

    sched = teir.schedule
    if iteration_node_id not in sched.iterations:
        raise TeirPassError(
            f"split_iteration: unknown iteration node {iteration_node_id!r}"
        )
    node = sched.iterations[iteration_node_id]
    axis_id = node.axis
    axis = teir.axes[axis_id]

    if inner_extent <= 0 or axis.extent % inner_extent != 0:
        raise TeirPassError(
            f"split_iteration: inner_extent {inner_extent} must divide axis"
            f" {axis_id!r} extent {axis.extent}"
        )

    for pid, prim in teir.primitives.items():
        for role, axes in prim.axes.items():
            if axis_id in axes:
                raise TeirPassError(
                    f"split_iteration: axis {axis_id!r} appears in primitive"
                    f" {pid!r} role {role!r}; split_iteration only operates"
                    " on schedule-iterated axes"
                )

    if node.guard is not None and any(
        getattr(term, "axis", None) == axis_id for term in node.guard
    ):
        raise TeirPassError(
            f"split_iteration: node {iteration_node_id!r} guard references"
            f" the axis being split ({axis_id!r})"
        )

    builder = teir.builder()
    outer_extent = axis.extent // inner_extent

    # Path-encode the split history: ``0`` for outer (visited first in
    # pre-order), ``1`` for inner. Recursive splits compose.
    outer_id = builder.claim_unused_axis_id(axis_id + "0")
    inner_id = builder.claim_unused_axis_id(axis_id + "1")

    inner_strides = dict(axis.strides)
    outer_strides = {tid: s * inner_extent for tid, s in axis.strides.items()}
    outer_offsets = dict(axis.offsets)

    builder.add_axis(inner_id, extent=inner_extent, strides_by_tensor=inner_strides)
    builder.add_axis(
        outer_id,
        extent=outer_extent,
        strides_by_tensor=outer_strides,
        offsets_by_tensor=outer_offsets,
    )

    # Repoint the original node at the outer axis as the parent of the
    # new inner node. The original's guard stays on the outer — guards
    # only reference ancestors of the original (now ancestors of the
    # outer), and outer-level checking saves N-fold guard evaluations
    # versus checking on every inner entry. The original's metadata is
    # duplicated to both nodes: advisory hints attach to the original
    # iteration's full extent, which is now the (outer, inner) product;
    # carrying the metadata on both keeps the hint visible at either
    # level.
    inner_node_id = builder.claim_unused_node_id(f"{iteration_node_id}1")
    builder.add_iteration(
        inner_node_id,
        axis=inner_id,
        policy=node.policy,
        children=list(node.children),
        guard=None,
        metadata=dict(node.metadata),
    )
    builder.set_iteration_axis(iteration_node_id, outer_id)
    builder.set_iteration_children(iteration_node_id, [inner_node_id])

    return builder.finish(validate=True)


def fuse_iterations(teir: Teir, outer_iteration_node_id: str) -> Teir:
    """Fuse an outer iteration node with its inner child into one node.

    The outer-then-inner chain (outer at ``outer_iteration_node_id``,
    inner = the outer's sole child) is collapsed into a single iteration
    node over a new fused axis whose extent is the product of the two.
    The new node reuses ``outer_iteration_node_id`` so external
    references survive; its guard is the concatenation of the outer's
    and inner's guards (conjunction).

    Schedule-centric semantics: this operation touches only the named
    pair. Other iteration nodes that may iterate the outer or inner
    axes are left alone, and primitive role lists are not rewritten. The
    original outer and inner axes are left in ``teir.axes``.

    Args:
        teir: Source IR.
        outer_iteration_node_id: Id of the outer iteration node. Its
            sole child must be the inner iteration node forming the
            chain.

    Raises:
        TeirPassError: If the node does not exist, has zero or multiple
            children, the child is not an iteration node, either axis
            appears in a primitive role list, policies differ, stride
            coherency is violated, or the inner's guard references the
            outer's axis (which is removed from this iteration site).
    """

    sched = teir.schedule
    if outer_iteration_node_id not in sched.iterations:
        raise TeirPassError(
            f"fuse_iterations: unknown iteration node {outer_iteration_node_id!r}"
        )
    outer_node = sched.iterations[outer_iteration_node_id]
    if len(outer_node.children) != 1:
        raise TeirPassError(
            f"fuse_iterations: outer node {outer_iteration_node_id!r} must"
            f" have exactly one child; has {len(outer_node.children)}"
        )
    inner_node_id = outer_node.children[0]
    if inner_node_id not in sched.iterations:
        raise TeirPassError(
            f"fuse_iterations: outer node {outer_iteration_node_id!r}'s"
            f" child {inner_node_id!r} is not an iteration node"
        )
    inner_node = sched.iterations[inner_node_id]
    outer_axis_id = outer_node.axis
    inner_axis_id = inner_node.axis
    outer_axis = teir.axes[outer_axis_id]
    inner_axis = teir.axes[inner_axis_id]

    if outer_node.policy != inner_node.policy:
        raise TeirPassError(
            f"fuse_iterations: policy mismatch (outer={outer_node.policy},"
            f" inner={inner_node.policy})"
        )

    for pid, prim in teir.primitives.items():
        for role, axes in prim.axes.items():
            if outer_axis_id in axes or inner_axis_id in axes:
                raise TeirPassError(
                    f"fuse_iterations: axis {outer_axis_id!r} or"
                    f" {inner_axis_id!r} appears in primitive {pid!r} role"
                    f" {role!r}; fuse_iterations only operates on"
                    " schedule-iterated axes"
                )

    # Stride coherency on every tensor:
    # ``stride_outer == extent_inner * stride_inner``.
    for tid in set(outer_axis.strides) | set(inner_axis.strides):
        s_outer = outer_axis.strides.get(tid, 0)
        s_inner = inner_axis.strides.get(tid, 0)
        if s_inner == 0 and s_outer == 0:
            continue
        if s_outer != inner_axis.extent * s_inner:
            raise TeirPassError(
                f"fuse_iterations: stride coherency violated for tensor"
                f" {tid!r}: expected outer={inner_axis.extent}*inner"
                f" ({inner_axis.extent * s_inner}); got {s_outer}"
            )

    # The inner guard cannot reference the outer axis: that axis no
    # longer appears in the schedule at this site after fusion, and our
    # first/last guard language can't encode the equivalent condition
    # against the fused axis. The outer's own guard is fine (it
    # references ancestors of outer, which are also ancestors of the
    # fused node).
    if inner_node.guard is not None:
        for term in inner_node.guard:
            if getattr(term, "axis", None) == outer_axis_id:
                raise TeirPassError(
                    f"fuse_iterations: inner guard term references outer"
                    f" axis {outer_axis_id!r}; outer is removed from this"
                    " iteration site by the fuse"
                )

    combined_guard: Guard | None
    if outer_node.guard is not None and inner_node.guard is not None:
        combined_guard = outer_node.guard + inner_node.guard
    else:
        combined_guard = outer_node.guard or inner_node.guard

    builder = teir.builder()
    fused_id = builder.claim_unused_axis_id(f"{outer_axis_id}_{inner_axis_id}")

    fused_strides = {
        tid: inner_axis.strides.get(tid, 0)
        for tid in set(inner_axis.strides) | set(outer_axis.strides)
        if inner_axis.strides.get(tid, 0) != 0
    }
    fused_offsets = {
        tid: outer_axis.offsets.get(tid, 0) + inner_axis.offsets.get(tid, 0)
        for tid in set(outer_axis.offsets) | set(inner_axis.offsets)
        if (outer_axis.offsets.get(tid, 0) + inner_axis.offsets.get(tid, 0)) != 0
    }
    builder.add_axis(
        fused_id,
        extent=outer_axis.extent * inner_axis.extent,
        strides_by_tensor=fused_strides,
        offsets_by_tensor=fused_offsets,
    )

    # Collapse outer → inner into one iteration over the fused axis;
    # reuse outer's node id so external references hold.
    builder.set_iteration_axis(outer_iteration_node_id, fused_id)
    builder.set_iteration_children(outer_iteration_node_id, list(inner_node.children))
    builder.set_iteration_policy(outer_iteration_node_id, inner_node.policy)
    builder.set_iteration_guard(outer_iteration_node_id, combined_guard)
    builder.set_iteration_metadata(outer_iteration_node_id, dict(inner_node.metadata))
    builder.remove_iteration(inner_node_id, reparent_children=True)
    return builder.finish(validate=True)


def set_policy(teir: Teir, node_id: str, policy: str) -> Teir:
    """Toggle an iteration node's policy between sequential and parallel."""

    if node_id not in teir.schedule.iterations:
        raise TeirPassError(
            f"set_policy: unknown iteration node {node_id!r}",
        )
    if policy not in VALID_POLICIES:
        raise TeirPassError(
            f"set_policy: policy must be one of {sorted(VALID_POLICIES)};"
            f" got {policy!r}",
        )
    builder = teir.builder()
    builder.set_iteration_policy(node_id, policy)
    return builder.finish(validate=True)
