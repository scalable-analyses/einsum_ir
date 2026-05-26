"""Subprocess tests for ``python -m etops.textir`` subcommands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from etops.emit import einsum
from etops.textir import dump
from tests._helpers import backend_available


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "etops.textir", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _write_sample(tmp_path: Path) -> Path:
    teir = einsum("ab,bc->ac", dim_sizes=dict(a=2, b=3, c=4), name="sample")
    p = tmp_path / "sample.teir"
    p.write_text(dump(teir))
    return p


class TestTextirCli:
    """Subprocess-level coverage of the textual-IR CLI."""

    def test_cat_round_trips(self, tmp_path: Path) -> None:
        result = _run("cat", str(_write_sample(tmp_path)))
        assert "teir-format" in result.stdout
        assert "tensor %in0" in result.stdout

    def test_validate_clean(self, tmp_path: Path) -> None:
        assert _run("validate", str(_write_sample(tmp_path))).returncode == 0

    def test_validate_quiet(self, tmp_path: Path) -> None:
        assert _run("validate", "--quiet", str(_write_sample(tmp_path))).stdout == ""

    def test_cat_rejects_invalid_by_default(self, tmp_path: Path) -> None:
        """``cat`` without ``--no-validate`` refuses semantically broken files."""

        path = tmp_path / "broken.teir"
        # An iteration node referencing an orphan child fails validation.
        path.write_text(
            "teir @broken {\n"
            "  tensor %out : f32\n"
            "  axis @a extent 4 strides { out: 4 }\n"
            "  schedule { roots [@iter_a] iter @iter_a axis @a policy sequential"
            " children [@orphan] }\n"
            "}\n"
        )
        assert _run("cat", str(path), check=False).returncode != 0

    def test_cat_no_validate_accepts_unvalidated(self, tmp_path: Path) -> None:
        """``cat --no-validate`` pretty-prints a malformed but parseable file.

        Patch the K iteration node (axis ``b`` is K in ``ab,bc->ac``, canonical
        id ``k0``) to ``policy=parallel`` — a race the validator rejects.
        """

        teir = einsum("ab,bc->ac", dim_sizes=dict(a=2, b=2, c=2), name="bad")
        text = dump(teir).replace(
            "iter @k0 axis @b policy sequential",
            "iter @k0 axis @b policy parallel",
        )
        path = tmp_path / "bad.teir"
        path.write_text(text)
        assert _run("cat", str(path), check=False).returncode != 0
        ok = _run("cat", "--no-validate", str(path))
        assert "@bad" in ok.stdout
        assert "policy parallel" in ok.stdout

    def test_validate_fails_on_bad_input(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.teir"
        path.write_text("teir @bad { garbage }\n")
        result = _run("validate", str(path), check=False)
        assert result.returncode != 0
        assert "error" in result.stderr.lower()

    @pytest.mark.tpp
    @pytest.mark.skipif(not backend_available("tpp"), reason="TPP backend unavailable")
    def test_run_tpp(self, tmp_path: Path) -> None:
        """``run --backend tpp`` executes the IR end-to-end via libxsmm."""

        import numpy as np

        teir = einsum("ab->ba", dim_sizes=dict(a=3, b=4))
        in_path = tmp_path / "x.teir"
        in_path.write_text(dump(teir))

        rng = np.random.default_rng(0)
        in0 = rng.standard_normal((3, 4)).astype(np.float32)
        out = np.zeros((4, 3), dtype=np.float32)
        in0_path = tmp_path / "in0.npy"
        out_path = tmp_path / "out_in.npy"
        np.save(in0_path, in0)
        np.save(out_path, out)
        out_dst = tmp_path / "out_dst.npy"

        _run(
            "run",
            str(in_path),
            "--backend",
            "tpp",
            "--inputs",
            str(in0_path),
            str(out_path),
            "--output",
            str(out_dst),
        )
        np.testing.assert_allclose(np.load(out_dst), in0.T)
