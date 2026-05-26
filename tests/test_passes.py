"""Tests for the pass framework and passes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import etops
from etops.analyses import AnalysisManager, dim_roles
from etops.emit import einsum
from etops.ir import TeirBuilder
from etops.passes import (
    AssignParallelism,
    EnsureKernelShape,
    PassContext,
)
from etops.passes._schedule import (
    _choose_role_assignment,
)

# Two GEMM and two tensor-contraction einsum variants whose output axis
# orderings exercise different branches of `ReorderForLocality` and the
# selective-promotion picker. Sizes are fixed and shared across the
# parametrized tests below.
_GEMM_EXPRS: tuple[str, ...] = ("ab,bc->ac", "ab,bc->ca")
_GEMM_SIZES: dict[str, int] = dict(a=8, b=16, c=8)
_CONTRACTION_EXPRS: tuple[str, ...] = ("trus,pqtu->pqrs", "trus,pqtu->qprs")
_CONTRACTION_SIZES: dict[str, int] = dict(t=4, r=3, u=2, s=8, p=4, q=2)


def _tpp_pass_ctx() -> PassContext:
    """Default-profile `PassContext` for tests that don't tune the profile."""

    return PassContext(
        analyses=AnalysisManager(),
        profile=etops.default_profile("tpp"),
    )


class TestCardinalityTargets:
    """TPP / BLAS profiles produce the expected primitive-role cardinalities."""

    @pytest.mark.parametrize("expr", _CONTRACTION_EXPRS)
    def test_tpp_brgemm_shape(self, expr: str) -> None:
        """TPP produces a contraction with M=1, N=1, K=2."""
        teir = einsum(expr, dim_sizes=_CONTRACTION_SIZES)
        ct = etops.optimize(teir, backend="tpp").primitives["contraction"]
        assert len(ct.role("M")) == 1
        assert len(ct.role("N")) == 1
        assert len(ct.role("K")) == 2

    @pytest.mark.parametrize("expr", _CONTRACTION_EXPRS)
    def test_blas_gemm_shape(self, expr: str) -> None:
        """BLAS produces a contraction with M=1, N=1, K=1."""
        teir = einsum(expr, dim_sizes=_CONTRACTION_SIZES)
        ct = etops.optimize(teir, backend="blas").primitives["contraction"]
        assert len(ct.role("M")) == 1
        assert len(ct.role("N")) == 1
        assert len(ct.role("K")) == 1


class TestSemanticPreservation:
    """The optimizer never changes the numerical result."""

    @pytest.mark.parametrize("expr", _GEMM_EXPRS)
    def test_gemm_matches_baseline(self, expr: str) -> None:
        """Optimized IR produces the same output as the unoptimized IR."""
        teir = einsum(expr, dim_sizes=_GEMM_SIZES)
        rng = np.random.default_rng(0)
        a = rng.standard_normal((8, 16)).astype(np.float32)
        b = rng.standard_normal((16, 8)).astype(np.float32)
        out_shape = np.einsum(expr, a, b).shape
        c1 = np.zeros(out_shape, dtype=np.float32)
        c2 = np.zeros(out_shape, dtype=np.float32)
        etops.compile(teir, backend="numpy").execute(a, b, c1)
        opt = etops.optimize(teir, backend="numpy")
        etops.compile(opt, backend="numpy").execute(a, b, c2)
        np.testing.assert_allclose(c2, c1, atol=1e-5)


class TestAnalysisCaching:
    """`AnalysisManager` caches by IR identity."""

    @staticmethod
    def _teir() -> etops.Teir:
        return einsum("ab,bc->ac", dim_sizes=dict(a=4, b=8, c=4))

    def test_cache_hit(self) -> None:
        """Two lookups against the same `Teir` reuse the cached result."""
        t = self._teir()
        mgr = AnalysisManager()
        first = mgr.get(dim_roles, t)
        second = mgr.get(dim_roles, t)
        assert first is second

    def test_invalidate(self) -> None:
        """Invalidation drops the cache entry for that IR."""
        t = self._teir()
        mgr = AnalysisManager()
        first = mgr.get(dim_roles, t)
        mgr.invalidate_all_for(t)
        second = mgr.get(dim_roles, t)
        assert first is not second


class TestAssignParallelismCollapse:
    """`AssignParallelism` emits a collapsed chain of parallel iterations.

    The walker's `dispatch_parallel` interprets a chain of nested
    parallel iteration nodes as one collapsed parallel domain, so the
    pass is allowed to mark several outer M/N/C axes parallel together
    when no single axis clears `num_threads * parallel_min_fanout`.
    """

    @staticmethod
    def _make_chain_ir(d: int, b: int, c: int) -> etops.Teir:
        # Unary Copy with three nested single-child iterations: the role
        # analyzer classifies every axis on a 2-tensor IR as 'C', so all
        # three are chain-eligible.
        builder = TeirBuilder()
        builder.add_tensor("in0", dtype="f32")
        builder.add_tensor("out", dtype="f32")
        # Row-major byte strides: c innermost, b middle, d outermost.
        c_stride = 4
        b_stride = c_stride * c
        d_stride = b_stride * b
        builder.add_axis(
            "d", extent=d, strides_by_tensor={"in0": d_stride, "out": d_stride}
        )
        builder.add_axis(
            "b", extent=b, strides_by_tensor={"in0": b_stride, "out": b_stride}
        )
        builder.add_axis(
            "c", extent=c, strides_by_tensor={"in0": c_stride, "out": c_stride}
        )
        builder.add_primitive(
            "copy",
            operation="Copy",
            axes={"M": [], "N": []},
            metadata={"data_type": "f32"},
        )
        builder.add_invocation("inv", primitive="copy")
        builder.add_iteration("iter_c", axis="c", children=["inv"])
        builder.add_iteration("iter_b", axis="b", children=["iter_c"])
        builder.add_iteration("iter_d", axis="d", children=["iter_b"])
        builder.set_roots(["iter_d"])
        return builder.finish(validate=True)

    @staticmethod
    def _ctx(num_threads: int, min_fanout: int) -> PassContext:
        base = etops.default_profile("tpp")
        backend = dataclasses.replace(base.backend, parallel_min_fanout=min_fanout)
        profile = dataclasses.replace(base, backend=backend, num_threads=num_threads)
        return PassContext(analyses=AnalysisManager(), profile=profile)

    def test_extends_chain_until_threshold_clears(self) -> None:
        """The chain stops as soon as cumulative extent clears the budget."""

        # d*b = 16 reaches the threshold; c is not absorbed into the chain.
        teir = self._make_chain_ir(d=4, b=4, c=2)
        ctx = self._ctx(num_threads=4, min_fanout=4)
        result = AssignParallelism(teir, ctx)
        sched = result.schedule.iterations
        assert sched["iter_d"].policy == "parallel"
        assert sched["iter_b"].policy == "parallel"
        assert sched["iter_c"].policy == "sequential"
        assert sched["iter_d"].metadata["threading.num_threads"] == 4

    def test_no_parallelism_when_chain_cannot_clear_threshold(self) -> None:
        """Cumulative extent below `num_threads * min_fanout` emits nothing."""

        # 2*2*2 = 8 < 16, and the chain can extend no further (the leaf's
        # only child is the invocation).
        teir = self._make_chain_ir(d=2, b=2, c=2)
        ctx = self._ctx(num_threads=4, min_fanout=4)
        result = AssignParallelism(teir, ctx)
        for nid in ("iter_d", "iter_b", "iter_c"):
            assert result.schedule.iterations[nid].policy == "sequential"

    def test_single_thread_target_is_noop(self) -> None:
        """A 1-thread target short-circuits without rewriting the IR."""

        teir = self._make_chain_ir(d=4, b=4, c=2)
        ctx = self._ctx(num_threads=1, min_fanout=4)
        result = AssignParallelism(teir, ctx)
        assert result is teir


class TestEnsureKernelShape:
    """`EnsureKernelShape` synthesizes dispatchable placeholder role axes."""

    @staticmethod
    def _build_contraction(
        *,
        role_axes: dict[str, list[str]],
        axes: dict[str, dict[str, int]],
        extents: dict[str, int],
        dtype: str = "f32",
    ) -> etops.Teir:
        """Build a single-Contraction IR with a fixed role-axis assignment."""

        b = TeirBuilder().set_name("synth_test")
        b.add_tensor("in0", dtype=dtype)
        b.add_tensor("in1", dtype=dtype)
        b.add_tensor("out", dtype=dtype)
        for ax_id, strides in axes.items():
            b.add_axis(ax_id, extent=extents[ax_id], strides_by_tensor=strides)
        b.add_primitive(
            "contraction",
            operation="Contraction",
            axes={role: list(axes_ids) for role, axes_ids in role_axes.items()},
            metadata={"data_type": dtype},
        )
        inv = b.add_invocation("inv", primitive="contraction")
        chain = inv
        # Wrap with one iteration per axis so the IR validates as a
        # well-formed schedule. The exact nesting is irrelevant — the pass
        # never touches the schedule.
        for ax_id in axes:
            iter_id = f"iter_{ax_id}"
            chain = b.add_iteration(iter_id, axis=ax_id, children=[chain])
        b.set_roots([chain])
        return b.finish(validate=False)

    def test_pure_c_synthesizes_three_unit_axes(self) -> None:
        """All three roles empty → one size-1 axis per role, each unit-stride
        on the tensors `ROLE_TENSOR_INDICES` assigns to it."""

        elem = 4
        teir = self._build_contraction(
            role_axes={"M": [], "N": [], "K": []},
            axes={
                # One pure-C axis present on every tensor; not a role axis.
                "c": {"in0": elem, "in1": elem, "out": elem},
            },
            extents={"c": 5},
        )
        result = EnsureKernelShape(teir, _tpp_pass_ctx())
        prim = result.primitives["contraction"]
        m_id, n_id, k_id = prim.axes["M"][0], prim.axes["N"][0], prim.axes["K"][0]

        for ax_id in (m_id, n_id, k_id):
            assert result.axes[ax_id].extent == 1

        # M lives on in0 and out; N on in1 and out; K on in0 and in1.
        assert result.axes[m_id].stride_on("in0") == elem
        assert result.axes[m_id].stride_on("out") == elem
        assert result.axes[m_id].stride_on("in1") == 0

        assert result.axes[n_id].stride_on("in1") == elem
        assert result.axes[n_id].stride_on("out") == elem
        assert result.axes[n_id].stride_on("in0") == 0

        assert result.axes[k_id].stride_on("in0") == elem
        assert result.axes[k_id].stride_on("in1") == elem
        assert result.axes[k_id].stride_on("out") == 0

    def test_outer_product_synth_k_carries_leading_dim(self) -> None:
        """M and N already unit-stride → synth K stride equals
        ``M_extent * bytes`` on in0 and ``N_extent * bytes`` on in1.
        """

        elem = 4
        teir = self._build_contraction(
            role_axes={"M": ["a"], "N": ["b"], "K": []},
            axes={
                "a": {"in0": elem, "out": 4 * elem},
                "b": {"in1": elem, "out": elem},
            },
            extents={"a": 3, "b": 4},
        )
        result = EnsureKernelShape(teir, _tpp_pass_ctx())
        prim = result.primitives["contraction"]
        k_id = prim.axes["K"][0]
        assert result.axes[k_id].extent == 1
        assert result.axes[k_id].stride_on("in0") == 3 * elem
        assert result.axes[k_id].stride_on("in1") == 4 * elem
        assert result.axes[k_id].stride_on("out") == 0

    def test_non_unit_partner_yields_unit_synth(self) -> None:
        """A real M axis with non-unit stride leaves in0 without a
        unit-stride role axis, so synth K must itself become unit on in0."""

        elem = 4
        teir = self._build_contraction(
            role_axes={"M": ["m"], "N": ["n"], "K": []},
            axes={
                "m": {"in0": 3 * elem, "out": elem},
                "n": {"in1": elem, "out": 4 * elem},
            },
            extents={"m": 4, "n": 3},
        )
        result = EnsureKernelShape(teir, _tpp_pass_ctx())
        k_id = result.primitives["contraction"].axes["K"][0]
        # in0 has no unit-stride role axis (M is at 12 bytes); synth K
        # must claim unit stride for the dispatcher to find one.
        assert result.axes[k_id].stride_on("in0") == elem
        # in1 has unit-stride N (extent 3) → synth K sits at the next-outer
        # slot.
        assert result.axes[k_id].stride_on("in1") == 3 * elem


_TPP_TARGETS: dict[str, int] = {
    "M": etops.default_profile("tpp").backend.role_target_m,
    "N": etops.default_profile("tpp").backend.role_target_n,
    "K": etops.default_profile("tpp").backend.role_target_k,
}


def _tpp_role_assignment(teir: etops.Teir) -> dict[str, tuple[str, ...]]:
    return _choose_role_assignment(teir, "contraction", _tpp_pass_ctx(), _TPP_TARGETS)


class TestChooseRoleAssignment:
    """`_choose_role_assignment` selects the (M, N, K) promotion tuples
    that maximize kernel work under the dispatcher's per-tensor
    unit-stride contract."""

    def test_canonical_gemm_promotes_real_role_axes(self) -> None:
        """A row-major GEMM ``ab,bc->ac`` promotes ``a`` to M, ``c`` to N,
        ``b`` to K — each role has exactly one candidate."""

        teir = einsum("ab,bc->ac", dim_sizes=dict(a=4, b=8, c=6))
        assignment = _tpp_role_assignment(teir)
        assert assignment == {"M": ("a",), "N": ("c",), "K": ("b",)}

    def test_mkc_ck_to_cm_leaves_k_unpromoted(self) -> None:
        """For ``mkc,ck->cm`` the C-axis is unit-stride on in0. The picker
        leaves K empty (and M promoted) so `EnsureKernelShape`'s synth K
        supplies the in0 unit-stride; the real ``k`` reduction stays in the
        schedule."""

        teir = einsum("mkc,ck->cm", dim_sizes=dict(m=3, k=2, c=4))
        assignment = _tpp_role_assignment(teir)
        assert assignment == {"M": ("m",), "N": (), "K": ()}

    def test_outer_product_promotes_m_and_n(self) -> None:
        """``a,b->ab`` has no K candidate; M and N are each real and
        unit-stride. The picker promotes both and leaves K to the synth."""

        teir = einsum("a,b->ab", dim_sizes=dict(a=4, b=6))
        assignment = _tpp_role_assignment(teir)
        assert assignment == {"M": ("a",), "N": ("b",), "K": ()}

    def test_hadamard_promotes_nothing(self) -> None:
        """Pure-C ``abc,abc->abc`` has no M / N / K candidates. The
        picker returns the empty assignment; `EnsureKernelShape` will
        synth all three placeholder axes."""

        teir = einsum("abc,abc->abc", dim_sizes=dict(a=2, b=3, c=4))
        assignment = _tpp_role_assignment(teir)
        assert assignment == {"M": (), "N": (), "K": ()}
