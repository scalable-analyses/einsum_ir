"""Pass framework and passes for the TEIR optimizer."""

from __future__ import annotations

from etops.passes._axis import DropTrivialAxes, DropUnusedAxes
from etops.passes._framework import Pass, PassContext, PassPipeline
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

__all__ = [
    "AssignParallelism",
    "BalanceMNTiles",
    "Canonicalize",
    "CheckBackendShape",
    "CheckKernelEligibility",
    "DropTrivialAxes",
    "DropUnusedAxes",
    "EnsureKernelShape",
    "FuseContiguousIterations",
    "LiftGuardedInit",
    "Pass",
    "PassContext",
    "PassPipeline",
    "PromoteRoleAxes",
    "ReorderForLocality",
    "TileForCache",
    "TileForPrimitive",
    "Validate",
]
