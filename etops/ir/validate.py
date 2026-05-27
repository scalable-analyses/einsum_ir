"""Global validation of a `Teir`.

The builder enforces local invariants at mutation time (ids unique,
strides non-negative, role arities, dtypes). This module verifies the
*global* invariants the builder cannot prove locally:

- Each non-root schedule node appears in exactly one ``children`` list.
- Each root appears in ``roots`` exactly once and not in any children.
- Every iteration node is reachable from a root (cycles, if any, are
  diagnosed as unreachable nodes).
- Guards reference only iteration-node ancestors of the guarded node;
  guard targets must be iteration nodes (not invocation nodes, not
  unknown ids) and may not name the guarded node itself.
- No iteration node nests an axis already iterated by an ancestor
  along the same path (the axis's byte stride would be applied twice).
- Parallel iteration of an axis with zero stride on the output tensor
  is rejected (a race).
- Every primitive-consumed axis byte stride is a multiple of the
  consumer tensor's element width.
- Every axis appearing in a primitive role has zero offsets on every
  tensor (offsets are scoped to schedule iteration).

Every violation is reported, then a single `TeirValidationError`
carrying every diagnostic is raised.
"""

from __future__ import annotations

from etops.diag import TeirValidationError
from etops.ir._records import (
    InvocationNode,
    IterationNode,
    Teir,
)
from etops.ir.cursors import descendant_invocations
from etops.ir.primitives import _OPERATIONS

__all__ = ["validate"]


def validate(teir: Teir) -> None:
    """Validate ``teir`` and raise `TeirValidationError` on failure."""

    errors: list[TeirValidationError] = []
    _check_primitive_operations(teir, errors)
    _check_primitive_role_axes_exist(teir, errors)
    _check_per_tensor_maps_known_tensors(teir, errors)
    _check_role_axis_offsets(teir, errors)
    _check_role_axis_alignment(teir, errors)
    _check_tensor_naming(teir, errors)
    _check_schedule(teir, errors)
    if not errors:
        return
    first = errors[0]
    raise TeirValidationError(
        message=first.message if len(errors) == 1 else "TEIR validation failed",
        suggestion=first.suggestion,
        diagnostics=list(errors),
    )


def _check_primitive_operations(teir: Teir, errors: list) -> None:
    for pid, prim in teir.primitives.items():
        if prim.operation not in _OPERATIONS:
            errors.append(
                TeirValidationError(
                    f"primitive {pid!r} has unknown operation {prim.operation!r};"
                    f" registered operations: {sorted(_OPERATIONS)}"
                )
            )


def _check_primitive_role_axes_exist(teir: Teir, errors: list) -> None:
    for pid, prim in teir.primitives.items():
        for role, axes in prim.axes.items():
            for ax in axes:
                if ax not in teir.axes:
                    errors.append(
                        TeirValidationError(
                            f"primitive {pid!r} role {role!r} references"
                            f" undefined axis {ax!r}"
                        )
                    )


def _check_per_tensor_maps_known_tensors(teir: Teir, errors: list) -> None:
    known = set(teir.tensors)
    for aid, axis in teir.axes.items():
        for tid in axis.strides:
            if tid not in known:
                errors.append(
                    TeirValidationError(
                        f"axis {aid!r} strides reference undefined tensor {tid!r}"
                    )
                )
        for tid in axis.offsets:
            if tid not in known:
                errors.append(
                    TeirValidationError(
                        f"axis {aid!r} offsets reference undefined tensor {tid!r}"
                    )
                )


# --------------------------------------------------------------------------
# Primitive-local invariants
# --------------------------------------------------------------------------


def _check_role_axis_offsets(teir: Teir, errors: list) -> None:
    role_consumers: dict[str, tuple[str, str]] = {}
    for pid, prim in teir.primitives.items():
        for role, axes in prim.axes.items():
            for ax in axes:
                role_consumers.setdefault(ax, (pid, role))

    for aid, axis in teir.axes.items():
        if aid not in role_consumers:
            continue
        consumer_pid, consumer_role = role_consumers[aid]
        for tid, offset in axis.offsets.items():
            if offset != 0:
                errors.append(
                    TeirValidationError(
                        f"axis {aid!r} is consumed by primitive {consumer_pid!r}"
                        f" role {consumer_role!r} and must have zero offsets on"
                        f" every tensor; tensor {tid!r} has offset {offset}"
                    )
                )


def _check_tensor_naming(teir: Teir, errors: list) -> None:
    """The native backends address tensors positionally and treat the last
    declared tensor as the output. When the tensor named ``"out"`` is
    present, require it to be declared last so the backends agree with
    callers that reach for it by name (e.g. the parallelism passes).
    """

    ids = teir.tensor_ids
    if "out" in ids and ids[-1] != "out":
        errors.append(
            TeirValidationError(
                "tensor 'out' must be declared last (the native backends treat"
                f" the last-declared tensor as the output); got tensor order {list(ids)!r}",
                suggestion="Reorder builder.add_tensor calls so 'out' is added"
                " after every input tensor, or rename the output tensor.",
            )
        )


def _check_role_axis_alignment(teir: Teir, errors: list) -> None:
    """Every primitive-consumed axis byte stride must align to element width."""

    width_by_tensor: dict[str, int] = {
        tid: tensor.dtype.bytes for tid, tensor in teir.tensors.items()
    }
    role_axes: set[str] = set()
    for prim in teir.primitives.values():
        for axes in prim.axes.values():
            role_axes.update(axes)

    for aid in sorted(role_axes):
        ax = teir.axes.get(aid)
        if ax is None:
            continue
        for tid, stride in ax.strides.items():
            width = width_by_tensor.get(tid)
            if not width:
                continue
            if stride % width != 0:
                errors.append(
                    TeirValidationError(
                        f"axis {aid!r} byte stride {stride} on tensor {tid!r}"
                        f" is not a multiple of the tensor's element width"
                        f" ({width} bytes); kernels expect element-aligned"
                        " leading dimensions"
                    )
                )


# --------------------------------------------------------------------------
# Schedule shape, acyclicity, guard scope, parallel reductions
# --------------------------------------------------------------------------


def _check_schedule(teir: Teir, errors: list) -> None:
    sched = teir.schedule
    all_node_ids = set(sched.iterations.keys()) | set(sched.invocations.keys())

    if not sched.roots and all_node_ids:
        errors.append(TeirValidationError("schedule has nodes but no roots"))

    root_seen: dict[str, int] = {}
    for rid in sched.roots:
        if rid not in all_node_ids:
            errors.append(
                TeirValidationError(f"root references unknown schedule node {rid!r}")
            )
        root_seen[rid] = root_seen.get(rid, 0) + 1
    for rid, count in root_seen.items():
        if count > 1:
            errors.append(
                TeirValidationError(
                    f"schedule root {rid!r} appears {count} times in 'roots'"
                )
            )

    for nid in sched.iterations.keys() & sched.invocations.keys():
        errors.append(
            TeirValidationError(
                f"node id {nid!r} appears as both iteration and invocation"
            )
        )

    parent_of: dict[str, str] = {}
    for nid, node in sched.iterations.items():
        for cid in node.children:
            if cid not in all_node_ids:
                errors.append(
                    TeirValidationError(
                        f"iteration {nid!r} references unknown child {cid!r}"
                    )
                )
                continue
            if cid in parent_of:
                errors.append(
                    TeirValidationError(
                        f"node {cid!r} is a child of multiple iteration nodes:"
                        f" {parent_of[cid]!r} and {nid!r}"
                    )
                )
            else:
                parent_of[cid] = nid
        if node.axis not in teir.axes:
            errors.append(
                TeirValidationError(
                    f"iteration node {nid!r} references unknown axis {node.axis!r}"
                )
            )

    for nid in all_node_ids:
        is_root = nid in root_seen
        is_child = nid in parent_of
        if is_root and is_child:
            errors.append(
                TeirValidationError(
                    f"node {nid!r} appears in both 'roots' and a children list"
                )
            )
        if not is_root and not is_child:
            errors.append(
                TeirValidationError(
                    f"node {nid!r} is neither a root nor a child of any iteration node"
                )
            )

    for nid, inv in sched.invocations.items():
        if inv.primitive not in teir.primitives:
            errors.append(
                TeirValidationError(
                    f"invocation {nid!r} references unknown primitive {inv.primitive!r}"
                )
            )

    _walk_schedule(teir, all_node_ids, errors)


def _walk_schedule(teir: Teir, all_node_ids: set[str], errors: list) -> None:
    """Single DFS that checks guard scope, axis nesting, parallel reduction.

    Tracks the ordered chain of ancestor iteration-node ids (and their
    axes) from each root; any node unreachable from a root after the
    walk is flagged (covering cycle detection).
    """

    sched = teir.schedule
    out_tensor = teir.tensor_ids[-1] if teir.tensor_ids else None
    visited: set[str] = set()

    # Each stack frame carries the chain of (iter_node_id, axis) pairs
    # from root to (but not including) the node we are about to visit.
    stack: list[tuple[str, tuple[tuple[str, str], ...]]] = [
        (rid, ()) for rid in reversed(sched.roots) if rid in all_node_ids
    ]
    while stack:
        node_id, ancestor_chain = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node: IterationNode | InvocationNode | None
        if node_id in sched.iterations:
            node = sched.iterations[node_id]
        elif node_id in sched.invocations:
            node = sched.invocations[node_id]
        else:
            continue

        guard = node.guard
        if guard is not None:
            ancestor_iter_ids = {nid for nid, _ in ancestor_chain}
            for term in guard:
                target = term.node
                if target == node_id:
                    errors.append(
                        TeirValidationError(
                            f"node {node_id!r} guard references itself"
                            f" ({target!r}); guard targets must be strict"
                            " iteration-node ancestors"
                        )
                    )
                    continue
                if target in ancestor_iter_ids:
                    continue
                if target in sched.iterations:
                    errors.append(
                        TeirValidationError(
                            f"node {node_id!r} guard references iteration node"
                            f" {target!r} which is not an ancestor of"
                            f" {node_id!r}"
                        )
                    )
                elif target in sched.invocations:
                    errors.append(
                        TeirValidationError(
                            f"node {node_id!r} guard references invocation node"
                            f" {target!r}; guard targets must be iteration nodes"
                        )
                    )
                else:
                    errors.append(
                        TeirValidationError(
                            f"node {node_id!r} guard references unknown node {target!r}"
                        )
                    )

        if isinstance(node, IterationNode):
            ancestor_axes = {axis for _, axis in ancestor_chain}
            if node.axis in ancestor_axes:
                errors.append(
                    TeirValidationError(
                        f"iteration node {node_id!r} iterates axis {node.axis!r}"
                        " already iterated by an ancestor along this path; the"
                        " axis's byte stride would be applied twice to every"
                        " tile address"
                    )
                )
            if (
                node.policy == "parallel"
                and out_tensor is not None
                and node.axis in teir.axes
                and teir.axes[node.axis].stride_on(out_tensor) == 0
            ):
                descendant_invs = list(descendant_invocations(teir, node_id))
                if descendant_invs:
                    prim_id = sched.invocations[descendant_invs[0]].primitive
                    errors.append(
                        TeirValidationError(
                            f"iteration node {node_id!r} (policy=parallel) iterates"
                            f" axis {node.axis!r} that has zero stride on output"
                            f" tensor {out_tensor!r}; concurrent iterations would"
                            f" race when primitive {prim_id!r} writes to the"
                            f" output tile",
                            suggestion="Use policy=sequential on this iteration"
                            " node, or choose a free (M / N / batch) axis for"
                            " parallel iteration.",
                        )
                    )
            child_chain = (*ancestor_chain, (node_id, node.axis))
            for cid in reversed(node.children):
                if cid not in visited:
                    stack.append((cid, child_chain))

    for nid in sorted(all_node_ids - visited):
        errors.append(
            TeirValidationError(
                f"schedule contains a cycle or detached subtree involving node {nid!r}"
            )
        )
