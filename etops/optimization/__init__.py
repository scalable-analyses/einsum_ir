"""Optimization profiles and hardware/backend descriptors."""

from __future__ import annotations

from etops.optimization.profiles import (
    ISA_EXTENSIONS,
    BackendStrategy,
    BlasStrategy,
    EffectiveTileTargets,
    ISAExtension,
    Microarchitecture,
    OptimizationProfile,
    TppStrategy,
    blas_profile,
    detect_isa_extension,
    detect_microarch,
    detect_num_threads,
    tpp_profile,
)

__all__ = [
    "ISA_EXTENSIONS",
    "BackendStrategy",
    "BlasStrategy",
    "EffectiveTileTargets",
    "ISAExtension",
    "Microarchitecture",
    "OptimizationProfile",
    "TppStrategy",
    "blas_profile",
    "detect_isa_extension",
    "detect_microarch",
    "detect_num_threads",
    "tpp_profile",
]
