"""Functional tests for every registered backend.

The three backend test triples (``numpy`` / ``tpp`` / ``blas``) share
the same workload coverage — only the backend label, availability skip,
and numerical tolerance differ. This module parametrizes the common
tests over backend and groups the few backend-specific tests
(BLAS-rejects-BRGEMM, numpy-only ReLU + runtime checks) into dedicated
classes.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

import etops
from etops.emit import einsum
from etops.ir.dtypes import resolve_numpy_dtype
from etops.textir import parse
from tests._helpers import backend_available, backend_param, tol_for
from tests.fixtures import (
    BRGEMM_LAYOUT_SIZES,
    GEMM_LAYOUT_SIZES,
    SPEC_TEIR_FILES,
    build_brgemm_with_layout,
    build_gemm_with_layout,
    build_hadamard_contraction,
    build_outer_product_contraction,
    gemm_layout_einsum_expr,
    gemm_layout_shapes,
)

_ALL_BACKENDS = backend_param("numpy", "tpp", "blas")
_NATIVE_BACKENDS = backend_param("tpp", "blas")


# --------------------------------------------------------------------------
# Permutations
# --------------------------------------------------------------------------


@_ALL_BACKENDS
class TestPermutation:
    """Both permutation schedules match `np.transpose` on every backend."""

    SIZES: ClassVar[dict[str, int]] = {"a": 2, "b": 3, "c": 4, "d": 5}

    @pytest.mark.parametrize(
        "teir_path",
        [SPEC_TEIR_FILES["scalar_permutation"], SPEC_TEIR_FILES["tiled_permutation"]],
        ids=["scalar", "tiled"],
    )
    def test_match(self, backend: str, teir_path) -> None:
        teir = parse(teir_path.read_text())
        op = etops.compile(teir, backend=backend)
        in_shape = tuple(self.SIZES[k] for k in "abcd")
        out_shape = tuple(self.SIZES[k] for k in "dcba")
        a_in = np.random.default_rng(0).standard_normal(in_shape).astype(np.float32)
        a_out = np.zeros(out_shape, dtype=np.float32)
        op.execute(a_in, a_out)
        np.testing.assert_allclose(
            a_out, a_in.transpose((3, 2, 1, 0)), **tol_for("exact", "f32")
        )


# --------------------------------------------------------------------------
# Batched GEMM
# --------------------------------------------------------------------------


@_ALL_BACKENDS
class TestBatchedGemm:
    """Batched GEMM schedules match NumPy on every backend.

    For the native backends the IR is run through ``etops.optimize`` first
    so the scalar contraction is rewritten into the libxsmm / cblas
    dispatch shape; the numpy oracle runs the un-optimized IR directly.
    """

    SIZES: ClassVar[dict[str, int]] = {"d": 2, "b": 3, "c": 4, "a": 5}

    @pytest.mark.parametrize(
        "teir_path",
        [SPEC_TEIR_FILES["scalar_bgemm"], SPEC_TEIR_FILES["reordered_scalar_bgemm"]],
        ids=["scalar", "reordered"],
    )
    def test_match(self, backend: str, teir_path) -> None:
        teir = parse(teir_path.read_text())
        if backend != "numpy":
            teir = etops.optimize(teir, backend=backend)
        op = etops.compile(teir, backend=backend)
        rng = np.random.default_rng(1)
        d, b, c, a = (self.SIZES[k] for k in "dbca")
        a_arr = rng.standard_normal((d, b, a)).astype(np.float32)
        b_arr = rng.standard_normal((d, a, c)).astype(np.float32)
        c_arr = np.zeros((d, b, c), dtype=np.float32)
        op.execute(a_arr, b_arr, c_arr)
        expected = np.einsum("dba,dac->dbc", a_arr, b_arr, optimize=False)
        np.testing.assert_allclose(c_arr, expected, **tol_for("mul", "f32"))


# --------------------------------------------------------------------------
# Tensor contraction (trus,pqtu -> pqrs)
# --------------------------------------------------------------------------


_CONTRACTION_SIZES: dict[str, int] = {"t": 2, "r": 3, "u": 2, "s": 4, "p": 3, "q": 2}

_TC_TEIR_PATHS = {
    "scalar": SPEC_TEIR_FILES["scalar_tensor_contraction"],
    "gemm": SPEC_TEIR_FILES["gemm_tensor_contraction"],
    "brgemm": SPEC_TEIR_FILES["brgemm_tensor_contraction"],
}


@_ALL_BACKENDS
@pytest.mark.parametrize("schedule", ["scalar", "gemm", "brgemm"])
class TestTensorContraction:
    """``trus,pqtu -> pqrs`` matches ``np.einsum`` across schedule shapes."""

    def test_match(self, backend: str, schedule: str) -> None:
        if schedule == "scalar" and backend != "numpy":
            pytest.skip("scalar contraction lowers only through the numpy oracle")
        if schedule == "brgemm" and backend == "blas":
            pytest.skip("BLAS backend refuses BRGEMM-shaped contractions")
        teir = parse(_TC_TEIR_PATHS[schedule].read_text())
        op = etops.compile(teir, backend=backend)
        rng = np.random.default_rng(2)
        t, r, u, s, p, q = (
            _CONTRACTION_SIZES[k] for k in ("t", "r", "u", "s", "p", "q")
        )
        in0 = rng.standard_normal((t, r, u, s)).astype(np.float32)
        in1 = rng.standard_normal((p, q, t, u)).astype(np.float32)
        out = np.zeros((p, q, r, s), dtype=np.float32)
        op.execute(in0, in1, out)
        expected = np.einsum("trus,pqtu->pqrs", in0, in1, optimize=False)
        np.testing.assert_allclose(out, expected, **tol_for("contract", "f32"))


# --------------------------------------------------------------------------
# Selective promotion (native backends only — numpy oracle has no
# promotion pass)
# --------------------------------------------------------------------------


@_NATIVE_BACKENDS
class TestSelectivePromotion:
    """Selective promotion + synth fill supply the dispatcher's unit-stride contract.

    For TPP the K-cardinality target is 2 and BLAS targets 1. When
    promoting a candidate would leave some tensor with no unit-stride-
    carrying role (typically because that tensor's unit-stride axis is
    a pure-C axis outside any dispatch role), the picker leaves the
    offending role unpromoted and ``EnsureKernelShape`` synthesizes the
    placeholder.
    """

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_mkc_ck_to_cm(self, backend: str, dtype: str) -> None:
        """``mkc,ck->cm`` — the C-axis ``c`` is unit-stride on in0, so neither
        M={m} nor K={k} is unit-stride on in0. K is left unpromoted so synth
        K supplies in0's unit stride; the real ``k`` reduction stays in the
        schedule loop."""

        teir = einsum("mkc,ck->cm", dim_sizes=dict(m=3, k=2, c=4), dtype=dtype)
        op = etops.compile(etops.optimize(teir, backend=backend), backend=backend)
        np_dtype = resolve_numpy_dtype(dtype)
        rng = np.random.default_rng(0)
        in0 = rng.standard_normal((3, 2, 4)).astype(np_dtype)
        in1 = rng.standard_normal((4, 2)).astype(np_dtype)
        out = np.zeros((4, 3), dtype=np_dtype)
        op.execute(in0, in1, out)
        expected = np.einsum("mkc,ck->cm", in0, in1, optimize=False)
        np.testing.assert_allclose(out, expected, **tol_for("contract", dtype))

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_ma_na_to_mna(self, backend: str, dtype: str) -> None:
        """``ma,na->mna`` — every operand's unit-stride axis is the pure-C
        ``a``. No M / N / K promotion satisfies any tensor's contract; all
        three roles stay empty and synth fills each, reducing the kernel to
        a scalar multiply-accumulate iterated by the schedule."""

        teir = einsum("ma,na->mna", dim_sizes=dict(m=3, n=4, a=2), dtype=dtype)
        op = etops.compile(etops.optimize(teir, backend=backend), backend=backend)
        np_dtype = resolve_numpy_dtype(dtype)
        rng = np.random.default_rng(0)
        in0 = rng.standard_normal((3, 2)).astype(np_dtype)
        in1 = rng.standard_normal((4, 2)).astype(np_dtype)
        out = np.zeros((3, 4, 2), dtype=np_dtype)
        op.execute(in0, in1, out)
        expected = np.einsum("ma,na->mna", in0, in1, optimize=False)
        np.testing.assert_allclose(out, expected, **tol_for("contract", dtype))


# --------------------------------------------------------------------------
# Synthesized role placeholders (native backends only)
# --------------------------------------------------------------------------


@_NATIVE_BACKENDS
class TestSynthesizedRoles:
    """Contractions whose einsum lacks M / N / K roles still dispatch after
    ``EnsureKernelShape`` synthesizes size-1 placeholder axes.
    """

    HADAMARD_SIZES: ClassVar[dict[str, int]] = {"a": 2, "b": 3, "c": 4}
    OUTER_SIZES: ClassVar[dict[str, int]] = {"a": 3, "b": 4}

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_hadamard_matches_numpy(self, backend: str, dtype: str) -> None:
        teir = build_hadamard_contraction(self.HADAMARD_SIZES, dtype=dtype)
        op = etops.compile(etops.optimize(teir, backend=backend), backend=backend)
        np_dtype = resolve_numpy_dtype(dtype)
        rng = np.random.default_rng(0)
        shape = tuple(self.HADAMARD_SIZES[k] for k in ("a", "b", "c"))
        in0 = rng.standard_normal(shape).astype(np_dtype)
        in1 = rng.standard_normal(shape).astype(np_dtype)
        out = np.zeros(shape, dtype=np_dtype)
        op.execute(in0, in1, out)
        expected = np.einsum("abc,abc->abc", in0, in1, optimize=False)
        np.testing.assert_allclose(out, expected, **tol_for("mul", dtype))

    @pytest.mark.parametrize("dtype", ["f32", "f64"])
    def test_outer_product_matches_numpy(self, backend: str, dtype: str) -> None:
        teir = build_outer_product_contraction(self.OUTER_SIZES, dtype=dtype)
        op = etops.compile(etops.optimize(teir, backend=backend), backend=backend)
        np_dtype = resolve_numpy_dtype(dtype)
        rng = np.random.default_rng(1)
        a_size, b_size = self.OUTER_SIZES["a"], self.OUTER_SIZES["b"]
        in0 = rng.standard_normal((a_size,)).astype(np_dtype)
        in1 = rng.standard_normal((b_size,)).astype(np_dtype)
        out = np.zeros((a_size, b_size), dtype=np_dtype)
        op.execute(in0, in1, out)
        np.testing.assert_allclose(out, np.outer(in0, in1), **tol_for("mul", dtype))


# --------------------------------------------------------------------------
# Layout-matrix coverage: all 8 (view x trans_a x trans_b) combinations
# --------------------------------------------------------------------------


@_NATIVE_BACKENDS
class TestLayoutMatrix:
    """Exhaustive GEMM dispatch corners. BRGEMM is TPP-only since BLAS
    has no batch-reduce ``cblas`` entry point and refuses BRGEMM-shaped
    contractions at compile time."""

    @pytest.mark.parametrize("out_unit", ["M", "N"])
    @pytest.mark.parametrize("in1_unit", ["K", "N"])
    @pytest.mark.parametrize("in0_unit", ["M", "K"])
    def test_gemm_layout(
        self, backend: str, in0_unit: str, in1_unit: str, out_unit: str
    ) -> None:
        teir = build_gemm_with_layout(
            in0_unit=in0_unit,
            in1_unit=in1_unit,
            out_unit=out_unit,
            sizes=GEMM_LAYOUT_SIZES,
        )
        op = etops.compile(teir, backend=backend, optimize=False)
        in0_shape, in1_shape, out_shape = gemm_layout_shapes(
            in0_unit, in1_unit, out_unit, GEMM_LAYOUT_SIZES
        )
        rng = np.random.default_rng(0)
        in0 = rng.standard_normal(in0_shape).astype(np.float32)
        in1 = rng.standard_normal(in1_shape).astype(np.float32)
        out = np.zeros(out_shape, dtype=np.float32)
        expr = gemm_layout_einsum_expr(in0_unit, in1_unit, out_unit)
        expected = np.einsum(expr, in0, in1)
        op.execute(in0, in1, out)
        np.testing.assert_allclose(out, expected, **tol_for("contract", "f32"))

    @pytest.mark.parametrize("out_unit", ["M", "N"])
    @pytest.mark.parametrize("in1_unit", ["K", "N"])
    @pytest.mark.parametrize("in0_unit", ["M", "K"])
    def test_brgemm_layout(
        self, backend: str, in0_unit: str, in1_unit: str, out_unit: str
    ) -> None:
        if backend == "blas":
            pytest.skip("BLAS backend refuses BRGEMM-shaped contractions")
        teir = build_brgemm_with_layout(
            in0_unit=in0_unit,
            in1_unit=in1_unit,
            out_unit=out_unit,
            sizes=BRGEMM_LAYOUT_SIZES,
        )
        op = etops.compile(teir, backend=backend, optimize=False)
        in0_shape, in1_shape, out_shape = gemm_layout_shapes(
            in0_unit, in1_unit, out_unit, BRGEMM_LAYOUT_SIZES
        )
        kb = BRGEMM_LAYOUT_SIZES["kb"]
        rng = np.random.default_rng(0)
        in0 = rng.standard_normal((kb, *in0_shape)).astype(np.float32)
        in1 = rng.standard_normal((kb, *in1_shape)).astype(np.float32)
        out = np.zeros(out_shape, dtype=np.float32)
        expr = gemm_layout_einsum_expr(in0_unit, in1_unit, out_unit)
        expected = sum(np.einsum(expr, in0[i], in1[i]) for i in range(kb))
        op.execute(in0, in1, out)
        np.testing.assert_allclose(out, expected, **tol_for("contract", "f32"))


# --------------------------------------------------------------------------
# BLAS-specific negative test
# --------------------------------------------------------------------------


@pytest.mark.blas
@pytest.mark.skipif(not backend_available("blas"), reason="BLAS backend unavailable")
class TestBlasRejections:
    """The BLAS backend refuses to lower BRGEMM-shaped contractions."""

    def test_brgemm_rejected(self) -> None:
        teir = parse(SPEC_TEIR_FILES["brgemm_tensor_contraction"].read_text())
        with pytest.raises(etops.TeirLoweringError, match="BRGEMM"):
            etops.compile(teir, backend="blas")
