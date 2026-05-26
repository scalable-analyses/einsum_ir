"""Permutation examples (``abcd -> dcba``).

This package owns two permutation schedules:

- ``scalar.teir``: the scalar-Copy schedule (one element per
  innermost iteration).
- ``tiled.teir``: the tile-Copy schedule consuming axes ``d`` and ``a``
  in the primitive's role lists.

``run`` parses the named schedule (defaults to ``tiled``) and applies
a minimal pre-lowering pipeline containing only ``EnsureKernelShape``,
which synth-fills any empty M/N/K role lists so the backend can
dispatch. It then compiles, executes on the requested backend (TPP by
default; BLAS via ``backend="blas"`` or the ``--backend`` CLI flag),
and checks the result against ``np.transpose``. Pass ``show=True`` or
``--show`` to print both the parsed and the post-EnsureKernelShape IR
trees first.
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
    "tiled": _HERE / "tiled.teir",
}
DEFAULT_SCHEDULE = "tiled"


def run(
    backend: str = "tpp",
    schedule: str | None = None,
    show: bool = False,
) -> None:
    """Execute the named permutation schedule on ``backend`` and check."""

    teir = textir.parse(SCHEDULES[schedule or DEFAULT_SCHEDULE].read_text())
    prepared = PassPipeline([EnsureKernelShape]).run(
        teir, profile=etops.default_profile(backend)
    )

    if show:
        print("Parsed IR:")
        print(etops.show(teir))
        print("After EnsureKernelShape (what executes):")
        print(etops.show(prepared))

    # Axis extents match every shipped permutation .teir: a=2, b=3, c=4, d=5.
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)  # abcd
    out = np.zeros((5, 4, 3, 2), dtype=np.float32)  # dcba

    op = etops.compile(prepared, backend=backend, optimize=False)
    op.execute(in0, out)

    expected = np.transpose(in0, (3, 2, 1, 0))
    np.testing.assert_allclose(out, expected, atol=0.0, rtol=0.0)
