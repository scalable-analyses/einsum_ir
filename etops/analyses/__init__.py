"""Read-only analyses of a `Teir`.

Each analysis is a module-level pure function ``(Teir) -> result``. The
`AnalysisManager` caches results by ``(id(teir), function)``; passes
fetch via ``ctx.analyses.get(function, teir)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from etops.ir import Teir

__all__ = [
    "ROLE_TENSOR_INDICES",
    "AnalysisManager",
    "DimRoles",
    "KernelEligibility",
    "StridePatterns",
    "dim_roles",
    "kernel_eligibility",
    "stride_patterns",
]


#: Tensor positions (in the conventional ``(in0, in1, out)`` ordering) that
#: each Contraction role axis appears on.
#:
#: This is the inverse of the classification performed by `dim_roles`: M
#: axes live on ``in0`` and ``out``, N axes on ``in1`` and ``out``, K axes
#: on ``in0`` and ``in1``. The TPP and BLAS dispatchers encode the same
#: mapping in C++ (`teir/src/internal/primitive_utils.h`) when collecting
#: per-role byte strides for libxsmm / cblas; this Python copy is the
#: source of truth on the optimizer side.
ROLE_TENSOR_INDICES: dict[str, tuple[int, ...]] = {
    "M": (0, 2),
    "N": (1, 2),
    "K": (0, 1),
}


class AnalysisManager:
    """Lightweight per-pipeline cache for analyses keyed by `Teir` identity.

    Entries live for the lifetime of the enclosing `PassContext`; the pass
    framework calls `invalidate_all_for(prev)` when a pass returns a new
    `Teir`, which prevents stale lookups across IR rewrites.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, Callable[..., Any]], Any] = {}

    def get(self, analysis: Callable[[Teir], Any], teir: Teir) -> Any:
        """Return the analysis result for ``teir``, computing it on demand."""

        key = (id(teir), analysis)
        cached = self._cache.get(key)
        if cached is None:
            cached = analysis(teir)
            self._cache[key] = cached
        return cached

    def invalidate_all_for(self, teir: Teir) -> None:
        teir_id = id(teir)
        for key in list(self._cache):
            if key[0] == teir_id:
                del self._cache[key]


# --------------------------------------------------------------------------
# Dimension roles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DimRoles:
    """Effective role of each axis (M, N, K, C, or unclassified)."""

    by_axis: dict[str, str]


def dim_roles(teir: Teir) -> DimRoles:
    """Classify each axis as M, N, K, C, or ``_``.

    For a binary contraction (``in0``, ``in1``, ``out``) the classification
    is derived from the zero-stride pattern:

    - Stride zero on ``out`` only           → ``K`` (contracted).
    - Stride zero on ``in1`` and out has it → ``M`` (left-only free).
    - Stride zero on ``in0`` and out has it → ``N`` (right-only free).
    - Nonzero on all three                  → ``C`` (batch).
    - Anything else                         → ``_`` (unclassified).

    For a unary operation (single input + output) every axis is classified
    ``C``; the dedicated `PromoteRoleAxes` unary phase handles unary
    role-axis selection by stride pattern. Operations with more than three
    tensors are rejected because the M/N/K classification rests on the
    ``(in0, in1, out)`` convention.
    """

    tensors = teir.tensor_ids
    if len(tensors) > 3:
        from etops.diag import TeirPassError

        raise TeirPassError(
            f"dim_roles: tensor count {len(tensors)} exceeds the"
            " binary-contraction convention (in0, in1, out); the optimizer"
            " does not yet support fused multi-input contractions"
        )
    roles: dict[str, str] = {}
    for aid, axis in teir.axes.items():
        s = [axis.stride_on(t) != 0 for t in tensors]
        if len(tensors) == 3:
            in0, in1, out = s
            if in0 and in1 and out:
                roles[aid] = "C"
            elif in0 and not in1 and out:
                roles[aid] = "M"
            elif not in0 and in1 and out:
                roles[aid] = "N"
            elif in0 and in1 and not out:
                roles[aid] = "K"
            else:
                roles[aid] = "_"
        elif len(tensors) == 2:
            roles[aid] = "C"
        else:
            roles[aid] = "_"
    return DimRoles(by_axis=roles)


# --------------------------------------------------------------------------
# Stride patterns
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StridePatterns:
    """Per-axis stride pattern per tensor: unit / zero / generic."""

    by_pair: dict[tuple[str, str], str]


def stride_patterns(teir: Teir) -> StridePatterns:
    """Classify the stride pattern of every (axis, tensor) pair."""

    by_pair: dict[tuple[str, str], str] = {}
    for tid, tensor in teir.tensors.items():
        elem = tensor.dtype.bytes
        for aid, axis in teir.axes.items():
            s = axis.stride_on(tid)
            if s == 0:
                by_pair[(aid, tid)] = "zero"
            elif s == elem:
                by_pair[(aid, tid)] = "unit"
            else:
                by_pair[(aid, tid)] = "generic"
    return StridePatterns(by_pair=by_pair)


# --------------------------------------------------------------------------
# Kernel eligibility
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelEligibility:
    """Per-primitive kernel-dispatch eligibility."""

    by_primitive: dict[str, str]


def kernel_eligibility(teir: Teir) -> KernelEligibility:
    """Decide for each `Contraction` primitive whether it targets scalar,
    GEMM, or BRGEMM."""

    by_primitive: dict[str, str] = {}
    for pid, prim in teir.primitives.items():
        if prim.operation != "Contraction":
            continue
        m, n, k = len(prim.role("M")), len(prim.role("N")), len(prim.role("K"))
        if m == 0 and n == 0 and k == 0:
            by_primitive[pid] = "Scalar"
        elif m == 1 and n == 1 and k == 1:
            by_primitive[pid] = "GEMM"
        elif m == 1 and n == 1 and k == 2:
            by_primitive[pid] = "BRGEMM"
        else:
            by_primitive[pid] = "Unsupported"
    return KernelEligibility(by_primitive=by_primitive)
