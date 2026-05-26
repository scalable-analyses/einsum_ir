"""Top-level compile / execute façade.

A single `Backend` record bundles the compile callable, the default pass
pipeline, and the default optimization profile for each registered
backend; `etops.compile(teir, backend=...)` dispatches through that
registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from etops.diag import TeirLoweringError
from etops.ir import Teir
from etops.lowering._native import NativeOperation
from etops.optimization import (
    OptimizationProfile,
    blas_profile,
    tpp_profile,
)
from etops.passes._framework import PassPipeline
from etops.passes.pipelines import _blas_pipeline, _tpp_pipeline

__all__ = [
    "Backend",
    "compile",
    "default_profile",
    "list_backends",
    "optimize",
    "register_backend",
]


@dataclass(frozen=True)
class Backend:
    """A registered backend's compile entry point and defaults."""

    name: str
    compile_fn: Callable[[Teir], Any]
    pipeline_factory: Callable[[], PassPipeline]
    profile_factory: Callable[[], OptimizationProfile]


_BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    """Register a backend."""

    _BACKENDS[backend.name] = backend


def list_backends() -> tuple[str, ...]:
    """Return the registered backend names."""

    return tuple(_BACKENDS.keys())


def _require(backend: str) -> Backend:
    try:
        return _BACKENDS[backend]
    except KeyError as exc:
        known = ", ".join(sorted(_BACKENDS.keys())) or "<none>"
        raise TeirLoweringError(
            f"unknown backend {backend!r}; registered backends: {known}"
        ) from exc


def default_profile(backend: str) -> OptimizationProfile:
    """Return the default `OptimizationProfile` for ``backend``.

    Raises:
        KeyError: When ``backend`` has no registered profile.
    """

    entry = _BACKENDS.get(backend)
    if entry is None:
        raise KeyError(
            f"no default profile for backend {backend!r};"
            f" registered: {sorted(_BACKENDS.keys()) or '<none>'}"
        )
    return entry.profile_factory()


def compile(
    teir: Teir,
    *,
    backend: str,
    optimize: bool = True,
    profile: OptimizationProfile | None = None,
) -> Any:
    """Compile a `Teir` for execution by ``backend``."""

    entry = _require(backend)
    if optimize:
        if profile is None:
            profile = entry.profile_factory()
        teir = entry.pipeline_factory().run(teir, profile=profile)
    return entry.compile_fn(teir)


def optimize(
    teir: Teir,
    *,
    backend: str,
    profile: OptimizationProfile | None = None,
) -> Teir:
    """Run ``backend``'s default pipeline over ``teir`` and return the result."""

    entry = _require(backend)
    if profile is None:
        profile = entry.profile_factory()
    return entry.pipeline_factory().run(teir, profile=profile)


for _name, _pipeline, _profile in (
    ("tpp", _tpp_pipeline, tpp_profile),
    ("blas", _blas_pipeline, blas_profile),
):
    register_backend(
        Backend(
            name=_name,
            compile_fn=partial(NativeOperation, backend=_name),
            pipeline_factory=_pipeline,
            profile_factory=_profile,
        )
    )
