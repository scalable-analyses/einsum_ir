"""End-to-end TPP backend dispatch tests across canonical workloads."""

from __future__ import annotations

import numpy as np
import pytest

import etops
from etops.emit import einsum
from tests._helpers import backend_available

pytestmark = [
    pytest.mark.tpp,
    pytest.mark.skipif(not backend_available("tpp"), reason="TPP backend unavailable"),
]


def test_square_sgemm_dispatches_through_tpp() -> None:
    """A representative square SGEMM optimizes to TPP BRGEMM shape and matches
    NumPy. The TPP dispatch is strict — a non-dispatchable IR would raise
    from ``etops.compile`` long before execution."""

    dim = 256
    teir = einsum("ab,bc->ac", dim_sizes={"a": dim, "b": dim, "c": dim})
    opt = etops.optimize(teir, backend="tpp")
    ct = next(p for p in opt.primitives.values() if p.operation == "Contraction")
    assert (len(ct.role("M")), len(ct.role("N")), len(ct.role("K"))) == (1, 1, 2)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((dim, dim)).astype(np.float32)
    b = rng.standard_normal((dim, dim)).astype(np.float32)
    c = np.zeros((dim, dim), dtype=np.float32)
    etops.compile(opt, backend="tpp").execute(a, b, c)
    np.testing.assert_allclose(c, a @ b, atol=1e-2 * dim, rtol=1e-3)


def test_brgemm_k_role_order_matches_spec() -> None:
    """For a workload that tiles the K axis, the resulting BRGEMM primitive
    must list K = [outer, inner] so that K[-1] is the unit-stride GEMM K axis
    on `in0`."""
    t = einsum("ab,bc->ac", dim_sizes={"a": 1024, "b": 1024, "c": 1024})
    t_opt = etops.optimize(t, backend="tpp")
    contraction = next(
        p for p in t_opt.primitives.values() if p.operation == "Contraction"
    )
    k_axes = contraction.role("K")
    if len(k_axes) < 2:
        pytest.skip(
            f"optimizer produced K cardinality {len(k_axes)}; this test"
            " verifies the BRGEMM-shaped (2-axis K) layout"
        )
    inner_k = t_opt.axes[k_axes[-1]]
    in0_stride = inner_k.stride_on("in0")
    elem_bytes = t_opt.tensors["in0"].dtype.bytes
    assert in0_stride == elem_bytes, (
        f"K[-1] = {k_axes[-1]!r} must be unit-stride on in0;"
        f" got byte-stride {in0_stride} (expected {elem_bytes})"
    )
