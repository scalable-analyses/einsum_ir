"""Tensor contraction examples (``trus,pqtu -> pqrs``).

Three schedules for the same computation:

- ``scalar.teir``: pure scalar nesting; ``Zero`` hoisted to the ``s``
  loop ahead of the inner ``t`` / ``u`` contraction loops.
- ``gemm.teir``: inner GEMM primitive (``M=s``, ``N=q``, ``K=u``)
  inside ``(p, r, t)`` nesting; ``Zero`` hoisted to ``(p, r)``.
- ``brgemm.teir``: BRGEMM primitive (``M=s``, ``N=q``, ``K=(t, u)``)
  collapsing both contraction axes into one batch-reduce call; outer
  ``(p, r)`` iteration runs in parallel.

``run`` parses the named schedule (defaults to ``gemm``) and applies
a minimal pre-lowering pipeline containing only ``EnsureKernelShape``,
which synth-fills any empty M/N/K role lists so the backend can
dispatch. It then compiles, executes on the requested backend (TPP by
default; BLAS via ``backend="blas"`` or the ``--backend`` CLI flag),
and checks the result against ``np.einsum``. The BLAS backend does
not implement BRGEMM, so the ``brgemm`` schedule runs only on TPP.
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
    "gemm": _HERE / "gemm.teir",
    "brgemm": _HERE / "brgemm.teir",
}
DEFAULT_SCHEDULE = "gemm"


def run(
    backend: str = "tpp",
    schedule: str | None = None,
    show: bool = False,
) -> None:
    """Execute the named tensor contraction schedule on ``backend`` and check."""

    teir = textir.parse(SCHEDULES[schedule or DEFAULT_SCHEDULE].read_text())
    prepared = PassPipeline([EnsureKernelShape]).run(
        teir, profile=etops.default_profile(backend)
    )

    if show:
        print("Parsed IR:")
        print(etops.show(teir))
        print("After EnsureKernelShape (what executes):")
        print(etops.show(prepared))

    # Axis extents match every shipped contraction .teir:
    # p=3, q=2, r=3, s=4, t=2, u=2.
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((2, 3, 2, 4)).astype(np.float32)  # trus
    in1 = rng.standard_normal((3, 2, 2, 2)).astype(np.float32)  # pqtu
    out = np.zeros((3, 2, 3, 4), dtype=np.float32)  # pqrs

    op = etops.compile(prepared, backend=backend, optimize=False)
    op.execute(in0, in1, out)

    expected = np.einsum("trus,pqtu->pqrs", in0, in1, optimize=False)
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)
