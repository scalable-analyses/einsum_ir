"""Default per-backend pass pipeline factories.

The TPP and BLAS backends share most pipeline stages but differ in the
final shape they accept: TPP dispatches via libxsmm GEMM (K=1) or BRGEMM
(K=2); BLAS dispatches via cblas GEMM only (K=1). The shape gate is
backend-specific; everything else is shared.
"""

from __future__ import annotations

from etops.passes._axis import DropTrivialAxes, DropUnusedAxes
from etops.passes._framework import Pass, PassPipeline
from etops.passes._iteration import (
    FuseContiguousIterations,
    TileForCache,
    TileForPrimitive,
)
from etops.passes._parallel import AssignParallelism, BalanceMNTiles
from etops.passes._schedule import (
    Canonicalize,
    EnsureKernelShape,
    LiftGuardedInit,
    PromoteRoleAxes,
    ReorderForLocality,
)
from etops.passes._validate import CheckBackendShape, CheckKernelEligibility, Validate


def _native_pipeline_with_shape(shape_check: Pass) -> PassPipeline:
    return PassPipeline(
        [
            DropTrivialAxes,
            LiftGuardedInit,
            ReorderForLocality,
            FuseContiguousIterations,
            TileForPrimitive,
            TileForCache,
            PromoteRoleAxes,
            EnsureKernelShape,
            BalanceMNTiles,
            AssignParallelism,
            CheckKernelEligibility,
            shape_check,
            DropUnusedAxes,
            Canonicalize,
            Validate,
        ]
    )


def _tpp_pipeline() -> PassPipeline:
    """TPP backend pipeline. libxsmm dispatches GEMM (1, 1, 1) and BRGEMM (1, 1, 2)."""

    return _native_pipeline_with_shape(
        CheckBackendShape(allowed_shapes=[(1, 1, 1), (1, 1, 2)])
    )


def _blas_pipeline() -> PassPipeline:
    """BLAS backend pipeline. cblas dispatches GEMM only (1, 1, 1); no BRGEMM."""

    return _native_pipeline_with_shape(CheckBackendShape(allowed_shapes=[(1, 1, 1)]))
