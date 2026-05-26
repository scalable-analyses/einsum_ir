"""Pipeline and optimization profile for the NumPy oracle.

The shipping ``etops/`` package only registers the production backends
(``tpp`` and ``blas``); the NumPy reference backend lives entirely in the
test tree. Its pipeline is intentionally minimal — the oracle is a
correctness reference, not a performance target — so the only passes are
the universal ``Canonicalize`` + ``DropTrivialAxes`` clean-up plus the
final ``Validate`` gate.
"""

from __future__ import annotations

from etops.optimization import (
    ISA_EXTENSIONS,
    BackendStrategy,
    Microarchitecture,
    OptimizationProfile,
)
from etops.passes import (
    Canonicalize,
    DropTrivialAxes,
    PassPipeline,
    Validate,
)

__all__ = ["NumpyOptimizationProfile", "numpy_pipeline"]


def numpy_pipeline() -> PassPipeline:
    """Minimal pipeline for the NumPy reference backend."""

    return PassPipeline([Canonicalize, DropTrivialAxes, Validate])


def NumpyOptimizationProfile() -> OptimizationProfile:
    """Minimal NumPy reference profile (oracle, not a performance target)."""

    return OptimizationProfile(
        microarch=Microarchitecture(
            name="numpy",
            l1_bytes=32 * 1024,
            l2_bytes=512 * 1024,
            l3_bytes=2 * 1024 * 1024,
        ),
        extension=ISA_EXTENSIONS["neon"],
        backend=BackendStrategy(
            name="numpy",
            role_target_m=1,
            role_target_n=1,
            role_target_k=1,
            tile_policy="register",
            library_dim_elements=0,
            k_over_mn=4,
            parallel_l2_fraction=0.5,
            cache_block_l3_fraction=0.5,
            divisor_slack=1.5,
            parallel_min_fanout=4,
            compensation_min=0.5,
            compensation_max=4.0,
        ),
        num_threads=1,
    )
