"""Observation passes: well-formedness validation and kernel-eligibility check."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from etops.analyses import kernel_eligibility
from etops.diag import TeirPassError
from etops.ir import Teir, validate
from etops.passes._framework import Pass, PassContext

_LOG = logging.getLogger(__name__)

__all__ = [
    "CheckBackendShape",
    "CheckKernelEligibility",
    "Validate",
]


def CheckBackendShape(
    *,
    allowed_shapes: Sequence[tuple[int, int, int]],
) -> Pass:
    """Build a pass that rejects Contractions whose (|M|, |N|, |K|) tuple is
    not in ``allowed_shapes``.
    """

    allowed = frozenset(allowed_shapes)

    def _check(teir: Teir, ctx: PassContext) -> Teir:
        backend_name = ctx.profile.backend.name
        for pid, prim in teir.primitives.items():
            if prim.operation != "Contraction":
                continue
            shape = (
                len(prim.role("M")),
                len(prim.role("N")),
                len(prim.role("K")),
            )
            if shape not in allowed:
                allowed_str = ", ".join(repr(s) for s in sorted(allowed))
                raise TeirPassError(
                    f"primitive {pid!r} has (|M|, |N|, |K|) = {shape};"
                    f" the {backend_name!r} backend only supports {allowed_str}"
                )
        return teir

    _check.__name__ = "CheckBackendShape"
    return _check


def Validate(teir: Teir, ctx: PassContext) -> Teir:
    """Final well-formedness check; returns the input unchanged on success."""

    validate(teir)
    return teir


def CheckKernelEligibility(teir: Teir, ctx: PassContext) -> Teir:
    """Warn when a Contraction's role cardinalities match no backend kernel.

    Pure observation pass: returns the IR unchanged.
    """

    eligibility = ctx.analyses.get(kernel_eligibility, teir)
    for pid, kind in eligibility.by_primitive.items():
        if kind == "Unsupported":
            prim = teir.primitives[pid]
            _LOG.warning(
                "CheckKernelEligibility: primitive %r has role"
                " cardinalities (M=%d, N=%d, K=%d); no kernel"
                " classification (Scalar / GEMM / BRGEMM) fits",
                pid,
                len(prim.role("M")),
                len(prim.role("N")),
                len(prim.role("K")),
            )
    return teir
