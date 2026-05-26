"""Tests for `etops.transforms`."""

from __future__ import annotations

import numpy as np
import pytest

import etops
from etops.emit import einsum
from etops.transforms import (
    canonicalize_ids,
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
