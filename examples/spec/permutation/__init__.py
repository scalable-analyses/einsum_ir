"""Permutation examples — schedule artifacts for unary tensor copy / transpose.

Artifacts live under one subdirectory per einsum family
(``input_output``). Each subdirectory's filenames describe the schedule
shape applied to that family:

- ``abcd_dcba/{scalar,tiled}.teir``: two schedules for the canonical
  reverse permutation ``abcd -> dcba``.
- ``behi_ehib/scalar.teir``: the scalar schedule for the 4-D rotation
  ``[b,e,h,i] -> [e,h,i,b]``.
- ``acdb_abcd/scalar.teir``: the scalar schedule for the partial-sort
  permutation ``[a,c,d,b] -> [a,b,c,d]``.
- ``_cases.py``: programmatic regressions whose IR doesn't round-trip
  through a textual artifact. Keys use the same
  ``<einsum_family>/<descriptor>`` shape (``a_a/scalar_f64``,
  ``abc_cba/scalar_f64``, ``rank9_arbitrary/scalar``,
  ``rank9_same_inner/scalar``, ``abc_bca/strided_input``); the last
  entry's input view carries non-contiguous byte strides the einsum
  emitter cannot express and is skipped on the TPP backend.

``run`` selects an artifact by name (defaults to ``abcd_dcba/tiled``),
applies a minimal pre-lowering pipeline containing only
``EnsureKernelShape``, compiles, executes, and asserts numerical
agreement against an independent NumPy reference. Pass ``show=True`` to
print both the parsed and post-``EnsureKernelShape`` IR trees first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import etops
from etops import textir
from etops.ir import Teir
from etops.passes import EnsureKernelShape, PassPipeline
from tests._helpers import allocate_operands

__all__ = ["ARTIFACTS", "DEFAULT_NAME", "run"]

_HERE = Path(__file__).parent
DEFAULT_NAME = "abcd_dcba/tiled"


def _build_artifacts() -> dict[str, str]:
    """``{name: kind}`` with kind in ``{"teir", "case"}``.

    ``.teir`` files are discovered recursively; any subdirectory grouping
    surfaces as a ``"<subdir>/<stem>"`` artifact name.
    """

    out: dict[str, str] = {}
    for path in sorted(_HERE.rglob("*.teir")):
        out[str(path.relative_to(_HERE).with_suffix(""))] = "teir"
    cases_path = _HERE / "_cases.py"
    if cases_path.exists():
        from . import _cases

        for key in _cases.CASES:
            if key in out:
                raise RuntimeError(
                    f"artifact name collision between .teir and _cases: {key!r}"
                )
            out[key] = "case"
    return out


ARTIFACTS: dict[str, str] = _build_artifacts()


def _load(name: str) -> Teir:
    kind = ARTIFACTS[name]
    if kind == "teir":
        return textir.parse((_HERE / f"{name}.teir").read_text())
    from ._cases import CASES

    return CASES[name].build()


def run(
    backend: str = "tpp",
    name: str | None = None,
    show: bool = False,
) -> None:
    """Execute artifact ``name`` on ``backend`` and verify against the oracle."""

    name = name or DEFAULT_NAME
    teir = _load(name)
    prepared = PassPipeline([EnsureKernelShape]).run(
        teir, profile=etops.default_profile(backend)
    )

    if show:
        print("Parsed IR:")
        print(etops.show(teir))
        print("After EnsureKernelShape (what executes):")
        print(etops.show(prepared))

    flats, views, out_index = allocate_operands(teir)
    op = etops.compile(prepared, backend=backend, optimize=False)
    op.execute(*flats)

    from ._oracles import REFERENCES

    inputs = [v for i, v in enumerate(views) if i != out_index]
    expected = REFERENCES[name](*inputs)
    np.testing.assert_allclose(views[out_index], expected, atol=1e-4, rtol=1e-4)
