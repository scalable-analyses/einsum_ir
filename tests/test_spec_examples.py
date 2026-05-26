"""Canonical spec examples must execute end-to-end on every backend.

Each ``.teir`` file under ``examples/`` is read straight from disk,
compiled for each registered backend, and the result is compared
element-wise against an independent NumPy oracle written directly from
the spec semantics (``np.einsum`` / ``np.transpose``). The oracle is
deliberately *not* the etops NumPy backend, so a bug shared between the
backend and the oracle cannot hide.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import etops
from etops.textir import parse
from tests._helpers import backend_available, tensor_shape_from_strides

_HERE = Path(__file__).resolve().parents[1] / "examples"


def _all_teir_files() -> list[Path]:
    return sorted(_HERE.rglob("*.teir"))


def _permutation_oracle(in0: np.ndarray) -> np.ndarray:
    return np.transpose(in0, (3, 2, 1, 0))


def _bgemm_oracle(in0: np.ndarray, in1: np.ndarray) -> np.ndarray:
    return np.einsum("dba,dac->dbc", in0, in1)


def _contraction_oracle(in0: np.ndarray, in1: np.ndarray) -> np.ndarray:
    return np.einsum("trus,pqtu->pqrs", in0, in1)


_ORACLES: dict[str, Callable[..., np.ndarray]] = {
    "permutation": _permutation_oracle,
    "batched_gemm": _bgemm_oracle,
    "tensor_contraction": _contraction_oracle,
}


def _operand_buffers(
    teir: etops.Teir,
) -> tuple[list[np.ndarray], int]:
    """Allocate per-tensor buffers shaped per the IR's byte strides."""

    from etops.ir.dtypes import resolve_numpy_dtype

    buffers: list[np.ndarray] = []
    out_index = -1
    for tid, tensor in teir.tensors.items():
        if tid == "out":
            out_index = len(buffers)
        shape = tensor_shape_from_strides(teir, tid)
        rng = np.random.default_rng(seed=hash((teir.name, tid)) & 0xFFFFFFFF)
        buf = rng.standard_normal(shape).astype(
            resolve_numpy_dtype(tensor.dtype.name), copy=False
        )
        buffers.append(np.ascontiguousarray(buf))
    if out_index == -1:
        out_index = len(buffers) - 1
    buffers[out_index][:] = 0
    return buffers, out_index


@pytest.fixture(
    params=_all_teir_files(),
    ids=lambda p: f"{p.parent.name}/{p.stem}",
)
def teir_path(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_spec_example_round_trips(teir_path: Path) -> None:
    """Every shipped ``.teir`` file parses and round-trips through textir."""

    from etops.textir import dump

    text = teir_path.read_text()
    teir = parse(text)
    redumped = dump(teir)
    teir2 = parse(redumped)
    assert teir == teir2


@pytest.mark.parametrize("backend", ["numpy", "tpp", "blas"])
def test_spec_example_executes(teir_path: Path, backend: str) -> None:
    """Each ``.teir`` file executes correctly under every available backend.

    The BLAS backend does not implement BRGEMM (documented limitation —
    CBLAS has no batch-reduce API), so the BRGEMM-shaped spec example
    is skipped for BLAS only.
    """

    if not backend_available(backend):
        pytest.skip(f"backend {backend!r} unavailable in this build")
    teir = parse(teir_path.read_text())
    if backend == "blas" and _is_brgemm_primitive(teir):
        pytest.skip("BLAS backend does not implement BRGEMM dispatch")
    operands, out_index = _operand_buffers(teir)
    inputs = [arr for i, arr in enumerate(operands) if i != out_index]

    oracle = _ORACLES.get(teir_path.parent.name)
    if oracle is None:
        pytest.skip(f"no independent oracle for {teir_path.parent.name!r}")
    expected = oracle(*inputs)

    operands[out_index][:] = 0
    op = etops.compile(teir, backend=backend)
    op.execute(*operands)
    np.testing.assert_allclose(operands[out_index], expected, atol=1e-4, rtol=1e-4)


def _is_brgemm_primitive(teir: etops.Teir) -> bool:
    for prim in teir.primitives.values():
        if prim.operation == "Contraction" and len(prim.role("K")) >= 2:
            return True
    return False
