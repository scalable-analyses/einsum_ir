"""Tests for the einsum emitter."""

from __future__ import annotations

import pytest

import etops
from etops.emit import einsum


class TestEinsumBasics:
    """Einsum strings produce the expected IR shape."""

    def test_unary_permutation_axes(self) -> None:
        """An ``abcd->dcba`` permutation produces four axes and one Copy."""
        t = einsum("abcd->dcba", dim_sizes=dict(a=2, b=3, c=4, d=5))
        assert t.axis_ids == ("a", "b", "c", "d")
        assert t.tensor_ids == ("in0", "out")
        prim_ops = [p.operation for p in t.primitives.values()]
        assert prim_ops == ["Copy"]

    def test_binary_contraction_classifies_axes(self) -> None:
        """Each axis ends up in M, N, K, or C correctly."""
        t = einsum("trus,pqtu->pqrs", dim_sizes=dict(t=2, r=3, u=2, s=4, p=3, q=2))
        iter_axes = [it.axis for it in t.schedule.iterations.values()]
        # The default emitter iterates every axis (free p, q, r, s and
        # contracted t, u); promotions happen only during `etops.optimize`.
        assert set(iter_axes) == {"p", "q", "r", "s", "t", "u"}

    def test_byte_strides_for_unary(self) -> None:
        """Strides for ``abcd->dcba`` are row-major byte strides."""
        t = einsum("abcd->dcba", dim_sizes=dict(a=2, b=3, c=4, d=5))
        # axis 'd' is innermost on in0 (stride 4 bytes for FP32) and outermost
        # on out (stride = c*b*a*4 = 4*3*2*4 = 96 bytes).
        d = t.axes["d"]
        assert d.stride_on("in0") == 4
        assert d.stride_on("out") == 4 * 4 * 3 * 2


class TestEinsumBracketedForm:
    """The bracketed form accepts multi-character axis names."""

    def test_bracketed_multi_character_axes(self) -> None:
        """The bracketed form accepts multi-character axis names; emitter axis
        order is deterministic."""
        t = einsum(
            "[m_outer,m_inner,k],[k,n]->[m_outer,m_inner,n]",
            dim_sizes=dict(m_outer=2, m_inner=2, k=3, n=4),
        )
        assert t.axis_ids == ("m_outer", "m_inner", "n", "k")

    def test_mixing_bracketed_and_bare_rejected(self) -> None:
        """Mixing ``[a,b]`` and ``cd`` operands is rejected."""
        with pytest.raises(etops.TeirEmissionError, match="mixing"):
            einsum("[a,b],cd->[a,b,c,d]", dim_sizes=dict(a=2, b=3, c=4, d=5))

    def test_unbalanced_bracket_rejected(self) -> None:
        """An unbalanced ``[`` is rejected."""
        with pytest.raises(etops.TeirEmissionError, match="unbalanced"):
            einsum("[a,b,c->[c,b,a]", dim_sizes=dict(a=2, b=3, c=4))

    def test_empty_bracketed_operand_allowed(self) -> None:
        """A scalar (rank-0) bracketed operand parses as the empty axis list."""

        t = einsum(
            "[a,b],[a,b]->[]",
            dim_sizes=dict(a=2, b=3),
        )
        # `a` and `b` are both K axes (in both inputs, in neither output).
        assert set(t.axis_ids) == {"a", "b"}


class TestEinsumDiagnostics:
    """The emitter rejects malformed strings with informative errors."""

    def test_missing_output(self) -> None:
        """Missing ``->`` is rejected."""
        with pytest.raises(etops.TeirEmissionError, match="explicit output"):
            einsum("ab", dim_sizes=dict(a=2, b=3))

    def test_missing_dim_size(self) -> None:
        """Missing ``dim_sizes`` entry is rejected."""
        with pytest.raises(etops.TeirEmissionError, match="missing dim_size"):
            einsum("ab->ba", dim_sizes={"a": 2})

    def test_unary_axis_set_must_match(self) -> None:
        """A unary expression cannot drop or add axes."""
        with pytest.raises(etops.TeirEmissionError, match="same axis set"):
            einsum("abc->ba", dim_sizes=dict(a=2, b=3, c=4))

    def test_unknown_dtype(self) -> None:
        """An unknown dtype name is rejected."""
        with pytest.raises(KeyError):
            einsum("ab->ba", dim_sizes=dict(a=2, b=3), dtype="not-a-type")
