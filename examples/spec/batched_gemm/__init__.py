"""Scalar batched GEMM examples (``dba,dac -> dbc``).

Two schedules for the same computation:

- ``scalar.teir``: the original schedule with a ``first(a)`` guard
  initializing the output before each accumulation.
- ``scalar_reordered.teir``: the equivalent reordered schedule where
  ``Zero`` is hoisted out of the contraction loop.

``run`` parses the named schedule (defaults to ``scalar_reordered``)
and applies a minimal pre-lowering pipeline containing only
``EnsureKernelShape``, which synth-fills any empty M/N/K role lists so
the backend can dispatch. It then compiles, executes on the requested
backend (TPP by default; BLAS via ``backend="blas"`` or the
``--backend`` CLI flag), and compares the result to ``np.einsum``.
Pass ``show=True`` or ``--show`` to print both the parsed and the
post-EnsureKernelShape IR trees first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import etops
from etops import textir
from etops.passes import EnsureKernelShape, PassPipeline

__all__ = ["DEFAULT_SCHEDULE", "SCHEDULES", "run"]

_HERE = Path(__file__).parent

SCHEDULES: dict[str, Path] = {
    "scalar": _HERE / "scalar.teir",
    "scalar_reordered": _HERE / "scalar_reordered.teir",
}
DEFAULT_SCHEDULE = "scalar_reordered"


def run(
    backend: str = "tpp",
    schedule: str | None = None,
    show: bool = False,
) -> None:
    """Execute the named batched GEMM schedule on ``backend`` and check."""

    teir = textir.parse(SCHEDULES[schedule or DEFAULT_SCHEDULE].read_text())
    prepared = PassPipeline([EnsureKernelShape]).run(
        teir, profile=etops.default_profile(backend)
    )

    if show:
        print("Parsed IR:")
        print(etops.show(teir))
        print("After EnsureKernelShape (what executes):")
        print(etops.show(prepared))

    # Axis extents match every shipped batched-GEMM .teir: d=2, b=3, a=5, c=4.
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((2, 3, 5)).astype(np.float32)  # dba
    in1 = rng.standard_normal((2, 5, 4)).astype(np.float32)  # dac
    out = np.zeros((2, 3, 4), dtype=np.float32)  # dbc

    op = etops.compile(prepared, backend=backend, optimize=False)
    op.execute(in0, in1, out)

    expected = np.einsum("dba,dac->dbc", in0, in1, optimize=False)
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)
