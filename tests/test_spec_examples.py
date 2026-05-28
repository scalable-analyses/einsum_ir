"""Canonical spec examples must execute end-to-end on every backend.

Each ``.teir`` file under ``examples/spec/`` is read straight from disk
and each ``CASES`` entry in any sibling ``_cases.py`` is built fresh;
both kinds of artifact are compiled for every registered backend and
their result is compared element-wise against an independent NumPy
oracle (``REFERENCES`` in each directory's ``_oracles.py``) written
directly from the spec semantics. The oracle is deliberately *not* the
etops NumPy backend, so a bug shared between the backend and the oracle
cannot hide.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import etops
from etops.ir import Teir
from etops.textir import dump, parse
from tests._helpers import allocate_operands, backend_available

_SPEC_ROOT = Path(__file__).resolve().parents[1] / "examples" / "spec"


@dataclass(frozen=True)
class Artifact:
    """One discoverable artifact: a parsed ``.teir`` file or a CASES entry."""

    kind: str  # "teir" or "case"
    dir_name: str
    name: str
    source: Any  # Path for kind="teir", Case for kind="case"

    @property
    def label(self) -> str:
        return f"{self.dir_name}/{self.name}"


def _all_artifacts() -> list[Artifact]:
    artifacts: list[Artifact] = []
    for teir_path in sorted(_SPEC_ROOT.rglob("*.teir")):
        relative = teir_path.relative_to(_SPEC_ROOT)
        spec_dir = relative.parts[0]
        name = str(relative.relative_to(spec_dir).with_suffix(""))
        artifacts.append(
            Artifact(
                kind="teir",
                dir_name=spec_dir,
                name=name,
                source=teir_path,
            )
        )
    for cases_path in sorted(_SPEC_ROOT.rglob("_cases.py")):
        module = importlib.import_module(
            f"examples.spec.{cases_path.parent.name}._cases"
        )
        for case_id, case in module.CASES.items():
            artifacts.append(
                Artifact(
                    kind="case",
                    dir_name=cases_path.parent.name,
                    name=case_id,
                    source=case,
                )
            )
    return artifacts


_ARTIFACTS = _all_artifacts()


def _load(artifact: Artifact) -> Teir:
    if artifact.kind == "teir":
        return parse(artifact.source.read_text())
    return artifact.source.build()


def _references_for(dir_name: str) -> dict[str, Callable[..., np.ndarray]]:
    module = importlib.import_module(f"examples.spec.{dir_name}._oracles")
    return module.REFERENCES


def _is_brgemm_primitive(teir: Teir) -> bool:
    for prim in teir.primitives.values():
        if prim.operation == "Contraction" and len(prim.role("K")) >= 2:
            return True
    return False


@pytest.fixture(params=_ARTIFACTS, ids=lambda a: a.label)
def artifact(request: pytest.FixtureRequest) -> Artifact:
    return request.param


def test_spec_artifact_round_trips(artifact: Artifact) -> None:
    """Every committed ``.teir`` parses and round-trips through textir.

    ``_cases.py`` entries skip this test because they go straight from
    ``build()`` to ``execute()`` without a textual round-trip.
    """

    if artifact.kind != "teir":
        pytest.skip("_cases.py entries do not round-trip a textual artifact")
    text = artifact.source.read_text()
    teir = parse(text)
    assert parse(dump(teir)) == teir


@pytest.mark.parametrize("backend", ["numpy", "tpp", "blas"])
def test_spec_artifact_executes(artifact: Artifact, backend: str) -> None:
    """Each artifact executes correctly under every available backend.

    The BLAS backend does not implement BRGEMM (CBLAS has no batch-reduce
    API), so BRGEMM-shaped artifacts skip the BLAS backend.
    """

    if not backend_available(backend):
        pytest.skip(f"backend {backend!r} unavailable in this build")

    teir = _load(artifact)
    if backend == "blas" and _is_brgemm_primitive(teir):
        pytest.skip("BLAS backend does not implement BRGEMM dispatch")
    if artifact.kind == "case":
        case = artifact.source
        if backend in getattr(case, "skip_backends", ()):
            reason = getattr(case, "skip_reason", "") or (
                f"{artifact.label} skipped on {backend} backend"
            )
            pytest.skip(reason)

    references = _references_for(artifact.dir_name)
    if artifact.name not in references:
        pytest.skip(f"no oracle registered for {artifact.label!r}")

    flats, views, out_index = allocate_operands(teir)
    inputs = [v for i, v in enumerate(views) if i != out_index]
    expected = references[artifact.name](*inputs)

    flats[out_index][...] = 0
    op = etops.compile(teir, backend=backend)
    op.execute(*flats)
    np.testing.assert_allclose(views[out_index], expected, atol=1e-4, rtol=1e-4)


def test_artifacts_are_discovered() -> None:
    """Sanity: discovery picks up every spec directory and its artifacts."""

    discovered = {a.label for a in _ARTIFACTS}
    expected_min = {
        "permutation/abcd_dcba/scalar",
        "permutation/abcd_dcba/tiled",
        "permutation/behi_ehib/scalar",
        "permutation/acdb_abcd/scalar",
        "permutation/a_a/scalar_f64",
        "permutation/abc_cba/scalar_f64",
        "permutation/rank9_arbitrary/scalar",
        "permutation/rank9_same_inner/scalar",
        "permutation/abc_bca/strided_input",
        "tensor_contraction/trus_pqtu_pqrs/scalar",
        "tensor_contraction/trus_pqtu_pqrs/gemm",
        "tensor_contraction/trus_pqtu_pqrs/brgemm",
        "tensor_contraction/abcd_efab_efcd/brgemm",
        "tensor_contraction/acbd_eafb_ecfd/brgemm",
        "tensor_contraction/dba_dac_dbc/scalar",
        "tensor_contraction/dba_dac_dbc/scalar_reordered",
        "tensor_contraction/yxgcaei_yxhfca_yhgfxei/scalar",
    }
    missing = expected_min - discovered
    assert not missing, f"discovery missed artifacts: {sorted(missing)}"
