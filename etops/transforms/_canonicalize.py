"""Canonicalize identifiers for diff-friendly stable output."""

from __future__ import annotations

from etops.analyses import dim_roles
from etops.diag import TeirPassError
from etops.ir import Teir, TeirBuilder

__all__ = ["canonicalize_ids"]

# Axis roles drive iteration-node naming: an axis classified as M/N/K/C
# names its iteration node ``m{i}`` / ``n{i}`` / ``k{i}`` / ``c{i}`` with a
# per-role counter; unclassified axes fall back to ``_{i}``. Invocations
# carry their own ``i{i}`` counter independent of the iteration prefixes.
ROLE_PREFIX = {"M": "m", "N": "n", "K": "k", "C": "c", "_": "_"}


def canonicalize_ids(teir: Teir) -> Teir:
    """Rename schedule node identifiers to a stable, role-aware form.

    The pre-order traversal of the schedule assigns one identifier per
    visited node. Iteration nodes are named after their axis's role —
    ``m{i}``, ``n{i}``, ``k{i}``, ``c{i}`` for the four classified roles
    and ``_{i}`` for unclassified axes; each role keeps its own dense
    counter. Invocation nodes are named ``i{i}`` with their own counter.
    Axis, primitive, and tensor identifiers are left as they are. The
    function is idempotent.
    """

    sched = teir.schedule
    roles = dim_roles(teir).by_axis
    rename: dict[str, str] = {}
    counters: dict[str, int] = {"m": 0, "n": 0, "k": 0, "c": 0, "_": 0, "i": 0}

    stack: list[str] = list(reversed(sched.roots))
    while stack:
        nid = stack.pop()
        if nid in rename:
            continue
        if nid in sched.iterations:
            it = sched.iterations[nid]
            prefix = ROLE_PREFIX[roles.get(it.axis, "_")]
            rename[nid] = f"{prefix}{counters[prefix]}"
            counters[prefix] += 1
            for cid in reversed(it.children):
                stack.append(cid)
        elif nid in sched.invocations:
            rename[nid] = f"i{counters['i']}"
            counters["i"] += 1
    counter = len(rename)

    builder = TeirBuilder()
    builder.set_name(teir.name)
    for tid, t in teir.tensors.items():
        builder.add_tensor(tid, dtype=t.dtype)
    for aid, axis in teir.axes.items():
        builder.add_axis(
            aid,
            extent=axis.extent,
            strides_by_tensor=dict(axis.strides),
            offsets_by_tensor=dict(axis.offsets),
        )
    for pid, prim in teir.primitives.items():
        builder.add_primitive(
            pid,
            operation=prim.operation,
            axes={role: list(axes) for role, axes in prim.axes.items()},
            metadata=dict(prim.metadata),
        )

    for nid, inv in sched.invocations.items():
        if nid not in rename:
            continue
        builder.add_invocation(
            rename[nid],
            primitive=inv.primitive,
            guard=inv.guard,
            metadata=dict(inv.metadata),
        )

    pending = [
        (rename[nid], it) for nid, it in sched.iterations.items() if nid in rename
    ]
    if len(pending) + sum(1 for nid in sched.invocations if nid in rename) != counter:
        raise TeirPassError("canonicalize_ids: orphan schedule nodes")

    ready_iter_ids: set[str] = set()
    added: set[str] = {rename[nid] for nid in sched.invocations if nid in rename}
    indegree: dict[str, int] = {}
    for new_id, it in pending:
        deps = 0
        for cid in it.children:
            new_cid = rename.get(cid)
            if new_cid is None:
                raise TeirPassError(
                    f"canonicalize_ids: iteration node references unknown child {cid!r}"
                )
            if new_cid not in added:
                deps += 1
        indegree[new_id] = deps
        if deps == 0:
            ready_iter_ids.add(new_id)

    by_new_id = {new_id: it for new_id, it in pending}
    parent_of: dict[str, list[str]] = {nid: [] for nid in by_new_id}
    for new_id, it in pending:
        for cid in it.children:
            new_cid = rename[cid]
            if new_cid in by_new_id:
                parent_of[new_cid].append(new_id)

    ready_stack = list(ready_iter_ids)
    while ready_stack:
        new_id = ready_stack.pop()
        it = by_new_id[new_id]
        builder.add_iteration(
            new_id,
            axis=it.axis,
            policy=it.policy,
            children=[rename[c] for c in it.children],
            guard=it.guard,
            metadata=dict(it.metadata),
        )
        added.add(new_id)
        for parent in parent_of.get(new_id, ()):
            indegree[parent] -= 1
            if indegree[parent] == 0:
                ready_stack.append(parent)

    if any(d > 0 for d in indegree.values()):
        raise TeirPassError("canonicalize_ids: cycle detected in iteration nodes")

    builder.set_roots([rename[r] for r in sched.roots])
    return builder.finish(validate=True)
