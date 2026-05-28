"""Schedule-level passes: role promotion, locality reorder, canonicalization,
kernel-shape inflation, and guarded-init lifting."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from etops.analyses import ROLE_TENSOR_INDICES, dim_roles, stride_patterns
from etops.diag import TeirPassError
from etops.ir import First, Primitive, Teir, TeirBuilder
from etops.ir.cursors import (
    descendant_invocations,
    iteration_nodes_outer_first,
    parent_map,
)
from etops.passes._framework import PassContext
from etops.transforms import canonicalize_ids

_LOG = logging.getLogger(__name__)

__all__ = [
    "Canonicalize",
    "EnsureKernelShape",
    "LiftGuardedInit",
    "PromoteRoleAxes",
    "ReorderForLocality",
]


_ROLE_RANK = {"C": 0, "M": 1, "N": 2, "K": 3, "_": 4}
_UNARY_OPERATIONS = frozenset({"Zero", "Copy", "ReLU"})


def _axis_carries_offset_on_primitive_tensors(
    teir: Teir, axis_id: str, primitive_id: str
) -> bool:
    """True if `axis_id` has a non-zero offset on any tensor reachable from
    `primitive_id`'s role axes. Lifting the invocation past this axis would
    silently drop the offset from the tile base address."""

    axis = teir.axes.get(axis_id)
    if axis is None or not axis.offsets:
        return False
    prim = teir.primitives.get(primitive_id)
    if prim is None:
        return False
    operand_tensors = set(teir.tensor_ids)
    return any(axis.offsets.get(tid, 0) != 0 for tid in operand_tensors)


# ============================================================================
# PromoteRoleAxes
# ============================================================================


def PromoteRoleAxes(teir: Teir, ctx: PassContext) -> Teir:
    """Move schedule-iterated axes into primitive role lists, subject to
    the backend dispatcher's per-tensor unit-stride contract.

    For each Contraction primitive, `_choose_role_assignment` enumerates
    the (M, N, K) axis tuples allowed by ``ctx.profile.role_target_*``
    and picks the most-promoting tuple that the dispatcher will accept
    once `EnsureKernelShape` synth-fills any empty roles. Sibling unary
    primitives (Zero / Copy / ReLU sharing an iter ancestor with the
    Contraction) constrain the choice: a candidate axis is rejected if
    propagating it via `_promote_axis_to_all_descendants` would violate
    a sibling's tile contract.

    Isolated unary primitives (no Contraction in their ancestor cone)
    are then promoted via the unit-stride heuristic in
    `_promote_unary_unit_stride`.

    A final pass moves primitive-native (pre-pass) role axes to the end
    of each role list, matching libxsmm's BRGEMM convention where the
    inner GEMM K sits at ``K[-1]``. Phase 1 promotes new K axes in the
    order chosen by `_choose_role_assignment`, so the originals landing
    at the end naturally place themselves innermost.
    """

    backend = ctx.profile.backend
    target_counts = {
        "M": backend.role_target_m,
        "N": backend.role_target_n,
        "K": backend.role_target_k,
    }
    original_axes_by_role: dict[tuple[str, str], frozenset[str]] = {
        (pid, role): frozenset(axes)
        for pid, prim in teir.primitives.items()
        for role, axes in prim.axes.items()
    }

    current = teir

    # Phase 1: Contractions via generate-and-test.
    contraction_ids = [
        pid
        for pid, prim in current.primitives.items()
        if prim.operation == "Contraction"
    ]
    for pid in contraction_ids:
        assignment = _choose_role_assignment(current, pid, ctx, target_counts)
        for role in ("M", "N", "K"):
            for axis_id in assignment[role]:
                try:
                    current = _promote_axis_to_all_descendants(current, axis_id, role)
                except TeirPassError as exc:
                    _LOG.warning(
                        "PromoteRoleAxes: %r → %s.%s: %s",
                        axis_id,
                        pid,
                        role,
                        exc,
                    )

    # Phase 2: isolated unary primitives via unit-stride heuristic.
    for pid in list(current.primitives.keys()):
        unary_prim = current.primitives.get(pid)
        if unary_prim is None or unary_prim.operation not in _UNARY_OPERATIONS:
            continue
        if _shares_ancestor_with_contraction(current, pid):
            continue
        current = _promote_unary_unit_stride(current, pid)

    return _reorder_originals_to_end(current, original_axes_by_role)


# ----------------------------------------------------------------------------
# Phase 1 — Contraction promotion via generate-and-test
# ----------------------------------------------------------------------------


def _choose_role_assignment(
    teir: Teir,
    primitive_id: str,
    ctx: PassContext,
    target_counts: dict[str, int],
) -> dict[str, tuple[str, ...]]:
    """Pick the (M, N, K) axis tuples to promote into this Contraction.

    Enumerates every (M, N, K) tuple drawn from per-role candidates
    (see `_gather_role_candidates`) within
    ``target_counts[role] - len(prim.axes[role])`` slots — the role
    cardinalities available after honoring any pre-populated axes.
    Each candidate tuple is checked against:

    - `_dispatch_satisfies`: the dispatcher's per-tensor unit-stride
      contract on the post-reorder role layout (promoted first,
      pre-populated last — see `_reorder_originals_to_end`).
    - `_unary_siblings_accept`: every sibling unary primitive in the
      promoted-axes' iter cone receives the axis on Phase-1 propagation
      and must still satisfy its tile contract.

    Among feasible tuples, the chosen one maximizes the count of
    promoted axes; ties are broken by the product of promoted-axis
    extents (a proxy for kernel-internal work). The empty assignment
    ``{M: (), N: (), K: ()}`` is always feasible — `EnsureKernelShape`
    synth-fills empty roles — so the picker falls back to it only when
    every non-empty assignment is infeasible.
    """

    prim = teir.primitives[primitive_id]
    roles = ctx.analyses.get(dim_roles, teir).by_axis
    candidates = _gather_role_candidates(teir, primitive_id, roles)
    effective = {
        role: max(0, target_counts[role] - len(prim.axes.get(role, ())))
        for role in ("M", "N", "K")
    }
    pre = {role: tuple(prim.axes.get(role, ())) for role in ("M", "N", "K")}

    best: tuple[tuple[int, int], dict[str, tuple[str, ...]]] | None = None
    for m_axes in _enumerate_subsets(candidates["M"], effective["M"]):
        for n_axes in _enumerate_subsets(candidates["N"], effective["N"]):
            for k_axes in _enumerate_subsets(candidates["K"], effective["K"]):
                final_m = (*m_axes, *pre["M"])
                final_n = (*n_axes, *pre["N"])
                final_k = (*k_axes, *pre["K"])
                if not _dispatch_satisfies(teir, final_m, final_n, final_k):
                    continue
                if not _unary_siblings_accept(
                    teir, primitive_id, m_axes, n_axes, k_axes
                ):
                    continue
                score = (
                    len(m_axes) + len(n_axes) + len(k_axes),
                    _kernel_work_score(teir, m_axes, n_axes, k_axes),
                )
                if best is None or score > best[0]:
                    best = (score, {"M": m_axes, "N": n_axes, "K": k_axes})

    if best is None:
        return {"M": (), "N": (), "K": ()}
    return best[1]


def _gather_role_candidates(
    teir: Teir,
    primitive_id: str,
    roles: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    """Return the iterated axes pickable for each M / N / K role on the
    Contraction.

    A candidate axis must:

    1. classify as the role under `dim_roles`,
    2. be iterated by an iter node common to every invocation of
       ``primitive_id`` — only common-ancestor axes can hoist into a
       role without breaking other invocations of the same primitive,
       and
    3. have the role on every primitive invocation reachable from that
       iter node (otherwise `_promote_axis_to_all_descendants` would
       raise on propagation).

    Output is outermost-first by schedule order, so the K-tuple
    enumeration produces orderings whose final element is the inner
    GEMM K in BRGEMM dispatch.
    """

    sched = teir.schedule
    invocation_ids = [
        nid for nid, inv in sched.invocations.items() if inv.primitive == primitive_id
    ]
    if not invocation_ids:
        return {"M": (), "N": (), "K": ()}
    parent_of = parent_map(teir)

    common: set[str] | None = None
    for nid in invocation_ids:
        chain_axes: set[str] = set()
        cur = parent_of.get(nid)
        while cur is not None:
            it = sched.iterations.get(cur)
            if it is not None:
                chain_axes.add(it.axis)
            cur = parent_of.get(cur)
        common = chain_axes if common is None else common & chain_axes
    common_axes = common or set()

    candidates: dict[str, list[str]] = {"M": [], "N": [], "K": []}
    for iter_node_id in iteration_nodes_outer_first(teir):
        ax = sched.iterations[iter_node_id].axis
        role = roles.get(ax)
        if role not in candidates:
            continue
        if ax not in common_axes:
            continue
        if not all(
            role in teir.primitives[sched.invocations[iid].primitive].axes
            for iid in descendant_invocations(teir, iter_node_id)
        ):
            continue
        candidates[role].append(ax)
    return {r: tuple(axes) for r, axes in candidates.items()}


def _enumerate_subsets(
    candidates: tuple[str, ...], target: int
) -> Iterator[tuple[str, ...]]:
    """Yield every ordered tuple of distinct elements from ``candidates``
    with length in ``[0, target]``.

    Sizes 0-2 are exhaustively emitted, matching the current backends'
    role-cardinality targets (TPP / BLAS GEMM = 1, TPP BRGEMM = 2 with
    order significant). A target greater than 2 raises — extend here if
    a new backend ever requires larger role lists.
    """

    yield ()
    if target <= 0 or not candidates:
        return
    for c in candidates:
        yield (c,)
    if target == 1:
        return
    for c0 in candidates:
        for c1 in candidates:
            if c0 != c1:
                yield (c0, c1)
    if target > 2:
        raise TeirPassError(
            f"PromoteRoleAxes: target cardinality {target} exceeds the"
            " supported maximum (2); extend `_enumerate_subsets`."
        )


def _dispatch_satisfies(
    teir: Teir,
    m_axes: tuple[str, ...],
    n_axes: tuple[str, ...],
    k_axes: tuple[str, ...],
) -> bool:
    """True iff the proposed role assignment satisfies the dispatcher's
    per-tensor unit-stride contract.

    Per tensor ``T`` ∈ ``(in0, in1, out)``, at least one role in ``T``'s
    dispatch domain (`ROLE_TENSOR_INDICES`) must either be empty
    (`EnsureKernelShape` synth-fills with a unit-stride placeholder) or
    have its dispatch axis — ``M[0]``, ``N[0]``, ``K[-1]`` — at unit
    byte stride on ``T``.
    """

    tensor_ids = teir.tensor_ids
    if len(tensor_ids) != 3:
        return True
    role_axes = {"M": m_axes, "N": n_axes, "K": k_axes}
    for tensor_idx, tensor_id in enumerate(tensor_ids):
        elem_bytes = teir.tensors[tensor_id].dtype.bytes
        covered = False
        for role, indices in ROLE_TENSOR_INDICES.items():
            if tensor_idx not in indices:
                continue
            axes = role_axes[role]
            if not axes:
                covered = True
                break
            dispatch_axis = axes[-1] if role == "K" else axes[0]
            if teir.axes[dispatch_axis].stride_on(tensor_id) == elem_bytes:
                covered = True
                break
        if not covered:
            return False
    return True


def _unary_siblings_accept(
    teir: Teir,
    primitive_id: str,
    m_axes: tuple[str, ...],
    n_axes: tuple[str, ...],
    k_axes: tuple[str, ...],
) -> bool:
    """True iff propagating the proposed axes to sibling unary primitives
    doesn't violate any operand's tile contract.

    `_promote_axis_to_all_descendants` adds each promoted axis to every
    invocation in the iter node's descendant cone that has the role.
    Sibling unary primitives (Zero / Copy / ReLU) accept the additions
    only if the resulting tile classifies under `_tile_is_classifiable`
    on every operand the unary dispatcher inspects.
    """

    iter_for_axis = {it.axis: nid for nid, it in teir.schedule.iterations.items()}
    sibling_added: dict[str, dict[str, list[str]]] = {}
    for role, axes in (("M", m_axes), ("N", n_axes), ("K", k_axes)):
        for axis_id in axes:
            iter_node_id = iter_for_axis.get(axis_id)
            if iter_node_id is None:
                continue
            for inv_id in descendant_invocations(teir, iter_node_id):
                inv = teir.schedule.invocations[inv_id]
                p = teir.primitives[inv.primitive]
                if p.operation not in _UNARY_OPERATIONS:
                    continue
                sibling_added.setdefault(p.id, {}).setdefault(role, []).append(axis_id)

    for prim_id, role_to_added in sibling_added.items():
        unary_prim = teir.primitives[prim_id]
        operand_ids = _unary_operand_tensor_ids(unary_prim, teir)
        if not operand_ids:
            continue
        proposed: list[str] = []
        for r in ("M", "N"):
            proposed.extend(unary_prim.axes.get(r, ()))
            proposed.extend(role_to_added.get(r, ()))
        for tensor_id in operand_ids:
            elem_bytes = teir.tensors[tensor_id].dtype.bytes
            if not _tile_is_classifiable(teir, tuple(proposed), tensor_id, elem_bytes):
                return False
    return True


def _kernel_work_score(
    teir: Teir,
    m_axes: tuple[str, ...],
    n_axes: tuple[str, ...],
    k_axes: tuple[str, ...],
) -> int:
    """Product of extents of all proposed promoted axes — a tie-break
    among feasible assignments. Larger means more kernel-internal SIMD
    work, generally preferable to schedule-loop iteration."""

    work = 1
    for axes in (m_axes, n_axes, k_axes):
        for ax in axes:
            work *= teir.axes[ax].extent
    return work


# ----------------------------------------------------------------------------
# Phase 2 — Isolated unary promotion via unit-stride heuristic
# ----------------------------------------------------------------------------


def _promote_unary_unit_stride(teir: Teir, primitive_id: str) -> Teir:
    """Promote an ancestor axis with unit byte stride on the output (N) and
    a distinct ancestor with unit byte stride on the input (M)."""

    prim = teir.primitives[primitive_id]
    n_done = len(prim.axes.get("N", ())) > 0
    m_done = len(prim.axes.get("M", ())) > 0
    if n_done and m_done:
        return teir

    out_tensor, in_tensor = _unary_io_tensors(teir)
    if out_tensor is None:
        return teir
    out_width = teir.tensors[out_tensor].dtype.bytes

    ancestors = _common_ancestor_axes(teir, primitive_id)
    if not ancestors:
        return teir

    candidate_n: str | None = None
    for aid in reversed(ancestors):
        ax = teir.axes.get(aid)
        if ax is not None and ax.stride_on(out_tensor) == out_width:
            candidate_n = aid
            break

    candidate_m: str | None = None
    if in_tensor is not None:
        in_width = teir.tensors[in_tensor].dtype.bytes
        for aid in reversed(ancestors):
            if aid == candidate_n:
                continue
            ax = teir.axes.get(aid)
            if ax is not None and ax.stride_on(in_tensor) == in_width:
                candidate_m = aid
                break

    current = teir
    if not n_done and candidate_n is not None:
        with contextlib.suppress(TeirPassError):
            current = _promote_axis_to_all_descendants(current, candidate_n, "N")
    if (
        not m_done
        and candidate_m is not None
        and candidate_m != candidate_n
        and any(it.axis == candidate_m for it in current.schedule.iterations.values())
    ):
        with contextlib.suppress(TeirPassError):
            current = _promote_axis_to_all_descendants(current, candidate_m, "M")
    return current


def _shares_ancestor_with_contraction(teir: Teir, primitive_id: str) -> bool:
    """True iff a Contraction invocation shares any ancestor iter node with
    an invocation of ``primitive_id``."""

    sched = teir.schedule
    unary_inv_ids = {
        nid for nid, inv in sched.invocations.items() if inv.primitive == primitive_id
    }
    contraction_inv_ids = {
        nid
        for nid, inv in sched.invocations.items()
        if (prim := teir.primitives.get(inv.primitive)) is not None
        and prim.operation == "Contraction"
    }
    if not unary_inv_ids or not contraction_inv_ids:
        return False
    parent_of = parent_map(teir)

    def ancestors(nid: str) -> set[str]:
        out: set[str] = set()
        cur = parent_of.get(nid)
        while cur is not None:
            out.add(cur)
            cur = parent_of.get(cur)
        return out

    unary_anc: set[str] = set()
    for nid in unary_inv_ids:
        unary_anc |= ancestors(nid)
    return any(unary_anc & ancestors(nid) for nid in contraction_inv_ids)


def _unary_io_tensors(teir: Teir) -> tuple[str | None, str | None]:
    """Return ``(out_tensor_id, in_tensor_id)`` for a unary IR."""

    tensor_ids = teir.tensor_ids
    if not tensor_ids:
        return (None, None)
    out_id = "out" if "out" in tensor_ids else tensor_ids[-1]
    inputs = [t for t in tensor_ids if t != out_id]
    return (out_id, inputs[0] if inputs else None)


def _common_ancestor_axes(teir: Teir, primitive_id: str) -> list[str]:
    """Axes iterated by every common ancestor of the primitive's
    invocations, ordered outermost-first."""

    sched = teir.schedule
    inv_ids = [
        nid for nid, inv in sched.invocations.items() if inv.primitive == primitive_id
    ]
    if not inv_ids:
        return []
    parent_of = parent_map(teir)

    chains: list[list[str]] = []
    for nid in inv_ids:
        chain: list[str] = []
        cur = parent_of.get(nid)
        while cur is not None:
            it = sched.iterations.get(cur)
            if it is not None:
                chain.append(it.axis)
            cur = parent_of.get(cur)
        chains.append(list(reversed(chain)))

    common: list[str] = []
    for items in zip(*chains, strict=False):
        if all(x == items[0] for x in items):
            common.append(items[0])
        else:
            break
    return common


# ----------------------------------------------------------------------------
# Unary tile classification
# ----------------------------------------------------------------------------


def _unary_operand_tensor_ids(prim: Primitive, teir: Teir) -> tuple[str, ...]:
    """Return the tensor ids whose tile layout the unary dispatcher inspects.

    - ``Zero`` touches only the output tensor (the last in the ``Teir``).
    - ``Copy`` / ``ReLU`` touch the first (input) and last (output).

    Returns ``()`` for unsupported operations or degenerate teirs.
    """

    tensor_ids = teir.tensor_ids
    if not tensor_ids:
        return ()
    if prim.operation == "Zero":
        return (tensor_ids[-1],)
    if prim.operation in ("Copy", "ReLU") and len(tensor_ids) >= 2:
        return (tensor_ids[0], tensor_ids[-1])
    return ()


def _tile_is_classifiable(
    teir: Teir,
    axes: tuple[str, ...],
    tensor_id: str,
    elem_bytes: int,
) -> bool:
    """Python mirror of the C++ ``classify_operand_layout`` contract.

    Returns True iff, restricted to extent>1 axes (the active ones), the
    tile has at most two axes and at least one unit-stride axis on
    ``tensor_id``. Size-1 axes are inactive and never count against
    either limit.
    """

    active = tuple(a for a in axes if teir.axes[a].extent > 1)
    if len(active) > 2:
        return False
    if not active:
        return True
    return any(teir.axes[a].stride_on(tensor_id) == elem_bytes for a in active)


# ----------------------------------------------------------------------------
# Promotion mechanics + originals reorder
# ----------------------------------------------------------------------------


def _promote_axis_to_all_descendants(teir: Teir, axis_id: str, role: str) -> Teir:
    """Append ``axis_id`` to ``role`` of every invocation reachable from the
    iteration node iterating that axis, then drop the iteration node."""

    iter_node_id: str | None = None
    for nid, it in teir.schedule.iterations.items():
        if it.axis == axis_id:
            iter_node_id = nid
            break
    if iter_node_id is None:
        raise TeirPassError(f"no iteration node for axis {axis_id!r}")

    builder = teir.builder()
    for inv_id in descendant_invocations(teir, iter_node_id):
        inv = teir.schedule.invocations[inv_id]
        prim = teir.primitives[inv.primitive]
        if role not in prim.axes:
            raise TeirPassError(
                f"primitive {inv.primitive!r} ({prim.operation}) cannot accept"
                f" axis {axis_id!r} into role {role!r}"
            )
        role_axes = builder.primitive_role_axes(inv.primitive, role)
        if axis_id not in role_axes:
            builder.set_primitive_role_axes(inv.primitive, role, [*role_axes, axis_id])

    builder.remove_iteration(iter_node_id, reparent_children=True)
    return builder.finish(validate=False)


def _reorder_originals_to_end(
    teir: Teir,
    original_axes_by_role: dict[tuple[str, str], frozenset[str]],
) -> Teir:
    """Move primitive-native (pre-pass) role axes to the end of each
    role list.

    Phase-1-promoted axes retain the order chosen by
    `_choose_role_assignment` — they appear first; pre-populated axes
    follow. This matches libxsmm's BRGEMM convention: when an inner
    GEMM K is pre-populated as part of the IR fixture, batch-reduce
    axes added by Phase 1 land at ``K[0]`` (outer) and the original at
    ``K[-1]`` (inner).
    """

    builder = teir.builder()
    rewrote = False
    for pid, prim in teir.primitives.items():
        for role, axes in prim.axes.items():
            originals = original_axes_by_role.get((pid, role), frozenset())
            if not originals:
                continue
            promoted = [a for a in axes if a not in originals]
            native = [a for a in axes if a in originals]
            new_axes = (*promoted, *native)
            if tuple(new_axes) != tuple(axes):
                builder.set_primitive_role_axes(pid, role, list(new_axes))
                rewrote = True
    if not rewrote:
        return teir
    return builder.finish(validate=False)


# ============================================================================
# ReorderForLocality
# ============================================================================


def ReorderForLocality(teir: Teir, ctx: PassContext) -> Teir:
    """Reorder iteration nodes along single-child schedule chains by stride affinity.

    Within each maximal single-child chain of iteration nodes (no
    intervening invocations, no guards on inner links), nodes whose axis
    has unit stride on the output sink to the innermost position so the
    output is written contiguously; the remaining nodes are grouped by
    dim role so `FuseContiguousIterations` sees adjacent axes that fuse
    legally. The reorder is implemented by rewiring children pointers:
    each node keeps its id, axis, guard, metadata, and policy and moves
    to its new position in the chain.
    """

    roles = ctx.analyses.get(dim_roles, teir)
    strides = ctx.analyses.get(stride_patterns, teir)
    sched = teir.schedule
    out_tensor = teir.tensor_ids[-1] if teir.tensor_ids else None
    parents = parent_map(teir)

    def sort_key(nid: str) -> tuple[int, int, int]:
        axis = sched.iterations[nid].axis
        role = roles.by_axis.get(axis, "_")
        role_rank = _ROLE_RANK.get(role, 4)
        unit_on_out = (
            1
            if (
                out_tensor is not None
                and strides.by_pair.get((axis, out_tensor)) == "unit"
            )
            else 0
        )
        extent = teir.axes[axis].extent if axis in teir.axes else 0
        return (unit_on_out, role_rank, -extent)

    builder = teir.builder()
    rewrote = False
    used_in_chain: set[str] = set()

    for start in list(sched.iterations):
        if start in used_in_chain or start not in sched.iterations:
            continue
        chain: list[str] = []
        current_id: str | None = start
        while (
            current_id is not None
            and current_id in sched.iterations
            and current_id not in used_in_chain
        ):
            node = sched.iterations[current_id]
            if node.guard is not None and chain:
                break
            chain.append(current_id)
            used_in_chain.add(current_id)
            if len(node.children) != 1:
                break
            next_id = node.children[0]
            if next_id not in sched.iterations:
                break
            current_id = next_id
        if len(chain) < 2:
            continue
        sorted_chain = sorted(chain, key=sort_key)
        if sorted_chain == chain:
            continue

        # Rewire by children pointers. The chain's parent now points at
        # the new chain root; each consecutive pair in sorted_chain
        # becomes a parent-child link; the innermost picks up the
        # original tail (whatever hung below the pre-reorder innermost).
        chain_tail = list(sched.iterations[chain[-1]].children)
        parent_id = parents[chain[0]]
        if parent_id is not None:
            old_children = list(builder.iteration_children(parent_id))
            new_children = [
                sorted_chain[0] if c == chain[0] else c for c in old_children
            ]
            builder.set_iteration_children(parent_id, new_children)
        else:
            new_roots = [sorted_chain[0] if r == chain[0] else r for r in sched.roots]
            builder.set_roots(new_roots)
        for i, nid in enumerate(sorted_chain):
            if i == len(sorted_chain) - 1:
                builder.set_iteration_children(nid, chain_tail)
            else:
                builder.set_iteration_children(nid, [sorted_chain[i + 1]])
        rewrote = True
    if not rewrote:
        return teir
    return builder.finish(validate=False)


# ============================================================================
# Canonicalize
# ============================================================================


def Canonicalize(teir: Teir, ctx: PassContext) -> Teir:
    """Rename schedule nodes to a dense canonical form."""

    return canonicalize_ids(teir)


# ============================================================================
# EnsureKernelShape
# ============================================================================


def EnsureKernelShape(teir: Teir, ctx: PassContext) -> Teir:
    """Inflate empty Contraction M / N / K roles with size-1 placeholder axes
    that the TPP and BLAS dispatchers can use as a unit-stride role axis.

    Both libxsmm (TPP) and cblas (BLAS) route every Contraction through a
    GEMM kernel that requires, *per operand*, one role axis at unit byte
    stride and the other at an element-aligned stride. The dispatcher's
    inspection domain per operand is fixed (see
    `etops.analyses.ROLE_TENSOR_INDICES`):

    - ``in0`` is dispatched on ``{M, K}``.
    - ``in1`` is dispatched on ``{K, N}``.
    - ``out`` is dispatched on ``{M, N}``.

    Einsums whose role schema is incomplete — pure-C Hadamard schedules
    (no M / N / K), outer products (no K), dot products (no M / N) — would
    leave one or more of these role lists empty after `PromoteRoleAxes`,
    and the dispatcher would have nothing to inspect. This pass closes
    that gap by emitting a single size-1 axis per missing role, recorded
    with strides on exactly the tensors the dispatcher cares about.

    Stride choice: the placeholder's byte stride on tensor ``t`` is the
    product of the extents of the role axes already at unit stride on
    ``t``, times the dtype's byte width. That number is the row-major
    "next outer" offset on ``t``: when ``t`` has no real unit-stride role
    axis, the product is 1 and the placeholder itself becomes unit-stride;
    when ``t`` has a real unit-stride M (or N, or K) of extent ``e``, the
    placeholder lands at byte stride ``e * sizeof(dtype)``, which is the
    leading dimension libxsmm cross-checks against
    ``max(1, m_or_k_lib)`` on dispatch. Because the placeholder has
    extent one, the stride is a layout label the dispatcher reads — no
    address ever actually offsets by it.
    """

    builder = teir.builder()
    rewrote = False
    for pid, prim in list(teir.primitives.items()):
        if prim.operation != "Contraction":
            continue
        if _synthesize_missing_role_axes(teir, builder, pid, prim):
            rewrote = True
    if not rewrote:
        return teir
    return builder.finish(validate=False)


def _synthesize_missing_role_axes(
    teir: Teir,
    builder: TeirBuilder,
    pid: str,
    prim: Primitive,
) -> bool:
    """Add a size-1 placeholder axis for every empty M / N / K role of ``prim``.

    Returns True iff any placeholder was emitted.
    """

    tensor_ids = teir.tensor_ids
    bytes_per_tensor = {tid: teir.tensors[tid].dtype.bytes for tid in tensor_ids}
    leading_strides = _dispatch_leading_strides(teir, prim, bytes_per_tensor)

    emitted = False
    for role in ("M", "N", "K"):
        if role not in prim.axes or len(prim.axes[role]) > 0:
            continue
        strides: dict[str, int] = {}
        for tensor_idx in ROLE_TENSOR_INDICES[role]:
            if tensor_idx >= len(tensor_ids):
                continue
            tid = tensor_ids[tensor_idx]
            strides[tid] = leading_strides[tid]
        axis_id = builder.claim_unused_axis_id(f"_synth_{pid}_{role.lower()}")
        builder.add_axis(
            axis_id,
            extent=1,
            strides_by_tensor=strides,
            offsets_by_tensor={},
        )
        builder.set_primitive_role_axes(pid, role, [axis_id])
        emitted = True
    return emitted


def _dispatch_leading_strides(
    teir: Teir,
    prim: Primitive,
    bytes_per_tensor: dict[str, int],
) -> dict[str, int]:
    """Byte stride a synthesized placeholder axis should carry on each tensor.

    For each tensor ``t``, only the *dispatch axis* of each role
    contributes — the single role axis the libxsmm/cblas dispatcher
    inspects for unit-stride: ``M[0]`` for M, ``N[0]`` for N, and
    ``K[-1]`` (the inner GEMM K, as opposed to the BRGEMM outer
    batch-reduce axis at ``K[0]``) for K.

    Result per tensor::

        bytes_per_tensor[t] * product(
            axis.extent
            for axis in dispatch_axes_of(prim)
            if axis.stride_on(t) == bytes_per_tensor[t]
        )

    The product is ``1`` (so the placeholder itself becomes unit-stride
    on ``t``) when no real dispatch axis is unit-stride on ``t``;
    otherwise the placeholder lands at the row-major next-outer slot
    and supplies libxsmm with a valid leading dimension while the
    existing role axis carries the unit-stride contract.
    """

    leading: dict[str, int] = {}
    for tid, bytes_per in bytes_per_tensor.items():
        extent_product = 1
        for axis_id in _dispatch_axes(prim):
            axis = teir.axes[axis_id]
            if axis.stride_on(tid) == bytes_per:
                extent_product *= axis.extent
        leading[tid] = extent_product * bytes_per
    return leading


def _dispatch_axes(prim: Primitive) -> tuple[str, ...]:
    """Return the role axes the backend dispatcher inspects for unit stride.

    For an empty role this is empty (no contribution). For non-empty M
    or N (always cardinality 1 in current backends), this is the single
    axis. For non-empty K this is the innermost element — the GEMM K
    in BRGEMM, which carries the unit-stride/transpose decision per
    libxsmm convention.
    """

    out: list[str] = []
    for role in ("M", "N"):
        axes = prim.axes.get(role, ())
        if axes:
            out.append(axes[0])
    k_axes = prim.axes.get("K", ())
    if k_axes:
        out.append(k_axes[-1])
    return tuple(out)


# ============================================================================
# LiftGuardedInit
# ============================================================================


def LiftGuardedInit(teir: Teir, ctx: PassContext) -> Teir:
    """Reposition guarded init / finalize invocations to free the reduction axis.

    Two rewrite shapes are recognized:

    1. ``Zero[first(X)] | Copy[first(X)] | ReLU[last(X)]`` as a direct
       sibling of the X loop — the guard is dropped and the invocation
       moves out of the loop (``first`` lifts above, ``last`` below).
    2. A "scalar batched GEMM" pattern where the X
       iteration is the outermost loop and the guarded sibling lives
       several free-axis loops deeper. The X iteration is interchanged
       past the intervening free-axis loops until it wraps only the
       contraction, with the lifted invocation as its sibling.

    Both rewrites preserve schedule semantics and eliminate the guard so
    the reduction axis can be promoted into a Contraction primitive's K
    role — a precondition for libxsmm dispatch on the TPP backend and for
    single-``cblas_*gemm`` dispatch on the BLAS backend.
    """

    previous = teir
    while True:
        after = _lift_first_guarded_siblings(previous)
        after = _lower_reduction_inside_init(after)
        if after is previous:
            return after
        previous = after


def _lift_first_guarded_siblings(teir: Teir) -> Teir:
    """Lift ``first(X)`` / ``last(X)``-guarded sibling invocations out of the X loop."""

    builder = teir.builder()
    sched = teir.schedule
    parent_of = parent_map(teir)

    changed = False
    for nid, node in list(sched.iterations.items()):
        loop_axis = node.axis
        if len(node.children) < 2:
            continue
        new_children = list(node.children)
        for child_id in list(node.children):
            if child_id not in sched.invocations:
                continue
            inv = sched.invocations[child_id]
            guard = inv.guard
            if guard is None or len(guard) != 1:
                continue
            term = guard[0]
            if term.axis != loop_axis:
                continue
            prim = teir.primitives.get(inv.primitive)
            if prim is None:
                continue
            if prim.operation not in _UNARY_OPERATIONS:
                raise TeirPassError(
                    f"cannot lift invocation {child_id!r} (operation"
                    f" {prim.operation!r}) guarded by {term!r}: only"
                    f" {sorted(_UNARY_OPERATIONS)} are safe to lift"
                )
            if _axis_carries_offset_on_primitive_tensors(
                teir, loop_axis, inv.primitive
            ):
                continue
            kind = "before" if isinstance(term, First) else "after"
            builder.set_invocation_guard(child_id, None)
            new_children = [c for c in new_children if c != child_id]
            if not new_children:
                # Iteration node would lose its only child; revert.
                builder.set_invocation_guard(child_id, guard)
                continue
            builder.set_iteration_children(nid, new_children)
            parent_id = parent_of.get(nid)
            if parent_id is None:
                roots = list(builder.roots())
                idx = roots.index(nid)
                roots.insert(idx if kind == "before" else idx + 1, child_id)
                builder.set_roots(roots)
            else:
                parent_kids = list(builder.iteration_children(parent_id))
                idx = parent_kids.index(nid)
                parent_kids.insert(idx if kind == "before" else idx + 1, child_id)
                builder.set_iteration_children(parent_id, parent_kids)
            parent_of[child_id] = parent_id
            changed = True
    if not changed:
        return teir
    return builder.finish(validate=False)


def _lower_reduction_inside_init(teir: Teir) -> Teir:
    """Pull an outermost reduction iteration inside an inner loop that hosts
    the ``first(X)``-guarded init, so the reduction axis can be promoted
    into the contraction primitive."""

    sched = teir.schedule
    parent_of = parent_map(teir)

    for x_id, x_node in sched.iterations.items():
        if len(x_node.children) != 1:
            continue
        chain_ids: list[str] = [x_id]
        current_id = x_id
        while True:
            cur_node = sched.iterations.get(current_id)
            if cur_node is None or len(cur_node.children) != 1:
                break
            only_child = cur_node.children[0]
            child_node = sched.iterations.get(only_child)
            if child_node is None:
                break
            current_id = only_child
            chain_ids.append(current_id)
            if len(child_node.children) > 1:
                break
        inner_id = chain_ids[-1]
        if inner_id == x_id:
            continue
        inner = sched.iterations.get(inner_id)
        if inner is None or len(inner.children) < 2:
            continue
        guard_axis = x_node.axis
        guarded_id: str | None = None
        kind: str | None = None
        for child_id in inner.children:
            inv = sched.invocations.get(child_id)
            if inv is None or inv.guard is None or len(inv.guard) != 1:
                continue
            term = inv.guard[0]
            if term.axis != guard_axis:
                continue
            prim = teir.primitives.get(inv.primitive)
            if prim is None or prim.operation not in _UNARY_OPERATIONS:
                continue
            guarded_id = child_id
            kind = "first" if isinstance(term, First) else "last"
            break
        if guarded_id is None or kind is None:
            continue

        guarded_inv = sched.invocations[guarded_id]
        if _axis_carries_offset_on_primitive_tensors(
            teir, x_node.axis, guarded_inv.primitive
        ):
            continue

        builder = teir.builder()
        x_axis = builder.iteration_axis(x_id)
        x_policy = builder.iteration_policy(x_id)
        inner_kids = list(builder.iteration_children(inner_id))
        siblings = [c for c in inner_kids if c != guarded_id]
        if not siblings:
            continue

        x_parent = parent_of.get(x_id)
        x_only_child = builder.iteration_children(x_id)[0]
        if x_parent is None:
            new_roots = [x_only_child if r == x_id else r for r in builder.roots()]
            builder.set_roots(new_roots)
        else:
            new_children = [
                x_only_child if c == x_id else c
                for c in builder.iteration_children(x_parent)
            ]
            builder.set_iteration_children(x_parent, new_children)
        builder.remove_iteration(x_id, reparent_children=True)

        new_x_id = builder.claim_unused_node_id(x_id)
        builder.add_iteration(
            new_x_id,
            axis=x_axis,
            policy=x_policy,
            children=siblings,
        )
        remaining = [c for c in inner_kids if c == guarded_id]
        if kind == "first":
            new_inner_children = [*remaining, new_x_id]
        else:
            new_inner_children = [new_x_id, *remaining]
        builder.set_iteration_children(inner_id, new_inner_children)
        builder.set_invocation_guard(guarded_id, None)
        return builder.finish(validate=False)
    return teir
