"""Tensor contraction examples.

Files are grouped under one subdirectory per einsum family
(``input0_input1_output``); each subdirectory's filenames
describe the schedule shape applied to that family:

- ``trus_pqtu_pqrs/{scalar,gemm,brgemm}.teir``: three schedules for the
  canonical ``trus,pqtu -> pqrs`` contraction.
- ``abcd_efab_efcd/brgemm.teir``: a BRGEMM schedule for
  ``abcd,efab -> efcd`` with K axes ``(a, b)`` contiguous in ``in0``.
- ``acbd_eafb_ecfd/brgemm.teir``: a BRGEMM schedule for
  ``acbd,eafb -> ecfd`` with K axes interleaved with M axes in ``in0``.
- ``dba_dac_dbc/{scalar,scalar_reordered}.teir``: two scalar schedules
  for the batched GEMM ``dba,dac -> dbc`` (``d`` is the batch axis);
  ``scalar`` uses a ``first(a)`` guard on the Zero, ``scalar_reordered``
  hoists the Zero out of the K loop.
- ``_cases.py`` carries one programmatic entry
  ``yxgcaei_yxhfca_yhgfxei/scalar``: the all-roles-populated rank-7/6/7
  contraction.

``run`` selects an artifact by name (defaults to
``trus_pqtu_pqrs/gemm``), applies a minimal pre-lowering pipeline
containing only ``EnsureKernelShape``, compiles, executes, and asserts
numerical agreement against an independent NumPy reference. The BLAS
backend does not implement BRGEMM, so BRGEMM-shaped artifacts only run
on TPP. Pass ``show=True`` to print both the parsed and
post-``EnsureKernelShape`` IR trees first.
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
DEFAULT_NAME = "trus_pqtu_pqrs/gemm"


def _build_artifacts() -> dict[str, str]:
    """``{name: kind}`` with kind in ``{"teir", "case"}``.

    ``.teir`` files are discovered recursively so per-einsum-family
    subdirectories like ``trus_pqtu_pqrs/`` show up as artifact names
    of the form ``"<family>/<schedule>"`` (e.g.
    ``"trus_pqtu_pqrs/gemm"``).
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
