"""Tests for `etops.transforms`."""

from __future__ import annotations

import numpy as np
import pytest

import etops
from etops.emit import einsum
from etops.ir import First, Last, TeirBuilder, guard
from etops.transforms import (
    canonicalize_ids,
    fuse_iterations,
    set_policy,
    split_iteration,
)


def _make_gemm(a: int = 8, b: int = 8, c: int = 8) -> etops.Teir:
    return einsum("ab,bc->ac", dim_sizes=dict(a=a, b=b, c=c))


def _node_iterating(teir: etops.Teir, axis_id: str) -> str:
    return next(
        nid for nid, it in teir.schedule.iterations.items() if it.axis == axis_id
    )


class TestSplitIteration:
    """`split_iteration` rewrites one iteration node into outer/inner."""

    def test_divisor_split(self) -> None:
        """An axis of extent 8 splits into outer=4, inner=2, with the inner
        inheriting the original stride and the outer landing at
        ``inner_extent * original_stride`` on every tensor the axis spans."""

        t = _make_gemm(b=8)
        t2 = split_iteration(t, _node_iterating(t, "b"), inner_extent=2)
        assert set(t2.axis_ids) == {"a", "b", "b0", "b1", "c"}
        assert t2.axes["b0"].extent == 4
        assert t2.axes["b1"].extent == 2
        # b is on in0 (ab, stride=4) and in1 (bc, stride=32).
        assert t2.axes["b1"].stride_on("in0") == 4
        assert t2.axes["b0"].stride_on("in0") == 4 * 2
        assert t2.axes["b1"].stride_on("in1") == 32
        assert t2.axes["b0"].stride_on("in1") == 32 * 2

    def test_rejects_non_divisor(self) -> None:
        """Non-divisor inner extent is rejected."""
        t = _make_gemm(b=8)
        with pytest.raises(etops.TeirPassError, match="divide"):
            split_iteration(t, _node_iterating(t, "b"), inner_extent=3)

    def test_preserves_semantics(self) -> None:
        """The split IR executes identically under NumPy."""
        t = _make_gemm(a=4, b=8, c=4)
        rng = np.random.default_rng(0)
        a_arr = rng.standard_normal((4, 8)).astype(np.float32)
        b_arr = rng.standard_normal((8, 4)).astype(np.float32)
        c_arr = np.zeros((4, 4), dtype=np.float32)
        c_arr2 = np.zeros((4, 4), dtype=np.float32)
        etops.compile(t, backend="numpy").execute(a_arr, b_arr, c_arr)
        t2 = split_iteration(t, _node_iterating(t, "b"), inner_extent=2)
        etops.compile(t2, backend="numpy").execute(a_arr, b_arr, c_arr2)
        np.testing.assert_allclose(c_arr2, c_arr, atol=1e-5)


class TestCanonicalizeIds:
    """`canonicalize_ids` produces dense, stable identifiers."""

    def test_preserves_semantics(self) -> None:
        """Canonicalized IR executes equivalently."""
        t = _make_gemm(a=4, b=8, c=4)
        canonical = canonicalize_ids(t)
        rng = np.random.default_rng(1)
        a_arr = rng.standard_normal((4, 8)).astype(np.float32)
        b_arr = rng.standard_normal((8, 4)).astype(np.float32)
        c1 = np.zeros((4, 4), dtype=np.float32)
        c2 = np.zeros((4, 4), dtype=np.float32)
        etops.compile(t, backend="numpy").execute(a_arr, b_arr, c1)
        etops.compile(canonical, backend="numpy").execute(a_arr, b_arr, c2)
        np.testing.assert_allclose(c2, c1, atol=1e-5)

    def test_iteration_nodes_use_axis_role_prefix(self) -> None:
        """Iteration nodes are named by their axis role (m/n/k/c)."""
        # `ab,bc->ac` classifies a as M, c as N, b as K.
        canonical = canonicalize_ids(_make_gemm(a=2, b=4, c=2))
        sched = canonical.schedule
        role_of_iter = {
            it.id: {"a": "m", "c": "n", "b": "k"}[it.axis]
            for it in sched.iterations.values()
        }
        for node_id, expected_prefix in role_of_iter.items():
            assert node_id.startswith(expected_prefix), (
                f"iter on role {expected_prefix.upper()} got id {node_id!r}"
            )

    def test_invocation_nodes_use_i_prefix(self) -> None:
        """Both invocations (Zero, Contraction) get canonical ``i*`` ids."""
        canonical = canonicalize_ids(_make_gemm())
        assert set(canonical.schedule.invocations) == {"i0", "i1"}

    def test_per_role_counters_are_dense(self) -> None:
        """Each role's counter starts at 0 and is dense within the role."""
        # `trus,pqtu->pqrs` gives multiple axes per role: M={s}, N={p,q},
        # K={t,u}, C={r} — but the original einsum schedule places K
        # axes inside the contraction subtree, so the canonical names
        # should still be dense per role across the whole schedule.
        teir = einsum("trus,pqtu->pqrs", dim_sizes=dict(t=2, r=2, u=2, s=2, p=2, q=2))
        canonical = canonicalize_ids(teir)
        ids = {it.id for it in canonical.schedule.iterations.values()}
        for prefix in ("m", "n", "k", "c"):
            indices = sorted(int(i[len(prefix) :]) for i in ids if i.startswith(prefix))
            if indices:
                assert indices == list(range(len(indices))), (
                    f"role {prefix.upper()} counter is not dense: {indices}"
                )


class TestSetPolicy:
    """`set_policy` toggles sequential / parallel."""

    def test_makes_parallel(self) -> None:
        """An iteration node over a free axis can be flipped to parallel."""
        t = _make_gemm()
        # Pick an iteration node whose axis has non-zero stride on the output
        # tensor; parallel iteration over a reduction axis (zero stride on
        # `out`) is rejected by the validator (see I-05).
        free_axes = {aid for aid, axis in t.axes.items() if axis.stride_on("out") != 0}
        node_id = next(
            nid for nid, node in t.schedule.iterations.items() if node.axis in free_axes
        )
        t2 = set_policy(t, node_id, "parallel")
        assert t2.schedule.iterations[node_id].policy == "parallel"

    def test_parallel_on_reduction_axis_rejected(self) -> None:
        """`set_policy` refuses to mark a reduction axis as parallel."""
        t = _make_gemm()
        # Find an iteration over the K axis (`b`) — zero stride on out.
        node_id = next(
            nid for nid, node in t.schedule.iterations.items() if node.axis == "b"
        )
        with pytest.raises(
            etops.TeirValidationError, match="zero stride on output tensor"
        ):
            set_policy(t, node_id, "parallel")


def _guarded_zero_then_add(*, axis_extent: int) -> etops.Teir:
    """Build a single-axis IR: iter @loop runs ``axis_extent`` trips over
    axis ``a``; on the first trip the invocation ``init`` runs (a Zero),
    then ``upd`` runs unconditionally. Both invocations share the same
    primitive instance; the guard is the moving piece under test.
    """

    b = TeirBuilder().set_name("guarded_zero_then_add")
    b.add_tensor("out", dtype="f32")
    b.add_axis("a", extent=axis_extent, strides_by_tensor={"out": 4})
    b.add_primitive(
        "zero",
        operation="Zero",
        axes={"M": [], "N": []},
        metadata={"data_type": "f32"},
    )
    init = b.add_invocation("init", primitive="zero", guard=guard(First("loop")))
    upd = b.add_invocation("upd", primitive="zero")
    b.add_iteration("loop", axis="a", children=[init, upd])
    b.set_roots(["loop"])
    return b.finish(validate=True)


class TestSplitGuardRewrite:
    """`split_iteration` rewrites descendant guards naming the split node."""

    def test_first_term_expands_to_conjunction(self) -> None:
        """``first(@loop)`` becomes ``first(@loop) and first(@loop1)``."""
        t = _guarded_zero_then_add(axis_extent=4)
        t2 = split_iteration(t, "loop", inner_extent=2)
        init_guard = t2.schedule.invocations["init"].guard
        assert init_guard is not None
        # Outer kept its id; inner was named ``loop1``.
        node_ids = tuple(term.node for term in init_guard)
        assert node_ids == ("loop", "loop1")
        assert all(isinstance(term, First) for term in init_guard)

    def test_last_term_expands_to_conjunction(self) -> None:
        """``last(@loop)`` becomes ``last(@loop) and last(@loop1)``."""
        b = TeirBuilder().set_name("last_guarded")
        b.add_tensor("out", dtype="f32")
        b.add_axis("a", extent=4, strides_by_tensor={"out": 4})
        b.add_primitive(
            "zero",
            operation="Zero",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        fini = b.add_invocation("fini", primitive="zero", guard=guard(Last("loop")))
        upd = b.add_invocation("upd", primitive="zero")
        b.add_iteration("loop", axis="a", children=[upd, fini])
        b.set_roots(["loop"])
        t = b.finish(validate=True)
        t2 = split_iteration(t, "loop", inner_extent=2)
        fini_guard = t2.schedule.invocations["fini"].guard
        assert fini_guard is not None
        assert tuple(term.node for term in fini_guard) == ("loop", "loop1")
        assert all(isinstance(term, Last) for term in fini_guard)

    def test_split_preserves_guard_semantics(self) -> None:
        """The guarded init fires exactly once before and after a split."""
        import numpy as np

        t = _guarded_zero_then_add(axis_extent=6)
        out = np.full((6,), 1.0, dtype=np.float32)
        etops.compile(t, backend="numpy").execute(out)
        baseline = out.copy()

        out2 = np.full((6,), 1.0, dtype=np.float32)
        t2 = split_iteration(t, "loop", inner_extent=2)
        etops.compile(t2, backend="numpy").execute(out2)
        np.testing.assert_array_equal(out2, baseline)


class TestFuseGuardCollapse:
    """`fuse_iterations` collapses symmetric guard pairs into one term on
    the fused node and rejects everything else that names outer or inner."""

    @staticmethod
    def _two_loop_fixture(*, init_guard) -> etops.Teir:
        b = TeirBuilder().set_name("two_loop")
        b.add_tensor("out", dtype="f32")
        # outer axis ``a`` has stride 2 elements (= 2*2 elements * 2 inner trips)
        # = 8 bytes; inner axis ``b`` has stride 4 bytes. Together they walk
        # a contiguous 4-element output.
        b.add_axis("a", extent=2, strides_by_tensor={"out": 8})
        b.add_axis("b", extent=2, strides_by_tensor={"out": 4})
        b.add_primitive(
            "zero",
            operation="Zero",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        init = b.add_invocation("init", primitive="zero", guard=init_guard)
        upd = b.add_invocation("upd", primitive="zero")
        b.add_iteration("inner", axis="b", children=[init, upd])
        b.add_iteration("outer", axis="a", children=["inner"])
        b.set_roots(["outer"])
        return b.finish(validate=True)

    def test_symmetric_first_pair_collapses(self) -> None:
        """``first(@outer) and first(@inner)`` becomes ``first(@outer)``."""
        t = self._two_loop_fixture(init_guard=guard(First("outer"), First("inner")))
        t2 = fuse_iterations(t, "outer")
        init_guard = t2.schedule.invocations["init"].guard
        assert init_guard is not None
        assert tuple(term.node for term in init_guard) == ("outer",)
        assert isinstance(init_guard[0], First)

    def test_symmetric_last_pair_collapses(self) -> None:
        """``last(@outer) and last(@inner)`` becomes ``last(@outer)``."""
        t = self._two_loop_fixture(init_guard=guard(Last("outer"), Last("inner")))
        t2 = fuse_iterations(t, "outer")
        fused_guard = t2.schedule.invocations["init"].guard
        assert fused_guard is not None
        assert tuple(term.node for term in fused_guard) == ("outer",)
        assert isinstance(fused_guard[0], Last)

    def test_unpaired_outer_term_rejected(self) -> None:
        """An unpaired ``first(@outer)`` has no first/last form on the fused
        axis (would fire ``inner_extent`` consecutive fused trips)."""
        t = self._two_loop_fixture(init_guard=guard(First("outer")))
        with pytest.raises(etops.TeirPassError, match="cannot be expressed"):
            fuse_iterations(t, "outer")

    def test_cross_pair_rejected(self) -> None:
        """``first(@outer) and last(@inner)`` picks a mid-sequence fused trip
        which has no first/last representation."""
        t = self._two_loop_fixture(init_guard=guard(First("outer"), Last("inner")))
        with pytest.raises(etops.TeirPassError, match="cannot be expressed"):
            fuse_iterations(t, "outer")
