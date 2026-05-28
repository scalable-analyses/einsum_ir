"""TCCG ``default`` correctness gate.

Every contraction in ``examples.contraction.tccg.CORPUS["default"]`` is
emitted, compiled, executed, and compared element-wise against
``np.einsum(..., optimize=False)`` on every available production
backend.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

import etops
from etops.emit import einsum
from etops.ir.dtypes import resolve_numpy_dtype
from examples.contraction.tccg import CORPUS, verify_tolerance
from tests._helpers import backend_available

_DEFAULT_ENTRIES = list(CORPUS["default"])


def _split(expr: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    lhs, rhs = expr.split("->")
    in0, in1 = lhs.split(",")
    return tuple(in0), tuple(in1), tuple(rhs)


def _allocate(
    expr: str, extents: Mapping[str, int], dtype: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in0, in1, out = _split(expr)
    np_dtype = resolve_numpy_dtype(dtype)
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(tuple(extents[a] for a in in0)).astype(
            np_dtype, copy=False
        ),
        rng.standard_normal(tuple(extents[a] for a in in1)).astype(
            np_dtype, copy=False
        ),
        np.zeros(tuple(extents[a] for a in out), dtype=np_dtype),
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("expr", "extents"),
    _DEFAULT_ENTRIES,
    ids=[expr for expr, _ in _DEFAULT_ENTRIES],
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("tpp", marks=pytest.mark.tpp),
        pytest.param("blas", marks=pytest.mark.blas),
    ],
)
def test_tccg_default_matches_numpy(
    expr: str, extents: Mapping[str, int], backend: str
) -> None:
    """Each TCCG ``default`` contraction matches ``np.einsum`` on every backend."""

    if not backend_available(backend):
        pytest.skip(f"backend {backend!r} unavailable in this build")

    teir = einsum(expr, dim_sizes=extents, dtype="f32")
    in0, in1, out = _allocate(expr, extents, dtype="f32", seed=hash(expr) & 0xFFFFFFFF)
    op = etops.compile(teir, backend=backend)
    op.execute(in0, in1, out)
    expected = np.einsum(expr, in0, in1, optimize=False)
    np.testing.assert_allclose(
        out, expected, **verify_tolerance(expr, extents, in0, in1, "f32")
    )
