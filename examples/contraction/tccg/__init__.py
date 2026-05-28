"""TCCG pairwise tensor-contraction benchmark driver.

The corpus reproduces the Tensor Contraction Code Generator (TCCG)
benchmark — 24 contractions in ``CORPUS["default"]`` and a 48-entry
superset in ``CORPUS["full"]``. Each entry pairs an einsum string with
an axis-extent map. The upstream ``testcases_transC`` entry
(``"abc-bk-akc"``) is the input-swap of the ccsd entry
``"adc,bd->abc"`` already present in ``full``, and is therefore not
duplicated.

Reference:
    Paul Springer and Paolo Bientinesi. 2018. Design of a
    High-Performance GEMM-like Tensor-Tensor Multiplication. ACM
    Trans. Math. Softw. 44, 3, Article 28 (2018), 29 pages.
    https://doi.org/10.1145/3157733

``run_setting`` verifies every contraction in a setting against
``np.einsum(..., optimize=False)`` on each requested backend and, when
``perf=True``, additionally measures wall-clock time across the listed
backends, computing GFLOPS via the standard convention
(``2 x output_volume x K_volume``).
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np

import etops
from etops.emit import einsum
from etops.ir.dtypes import resolve_numpy_dtype

__all__ = [
    "CORPUS",
    "DEFAULT_SETTING",
    "gflops",
    "run_setting",
    "verify_tolerance",
]


DEFAULT_SETTING = "default"

CORPUS: dict[str, tuple[tuple[str, Mapping[str, int]], ...]] = {
    "default": (
        ("efbad,cf->abcde", {"a": 48, "b": 36, "c": 24, "d": 36, "e": 48, "f": 36}),
        ("efcad,bf->abcde", {"a": 48, "b": 24, "c": 36, "d": 36, "e": 48, "f": 36}),
        ("dbea,ec->abcd", {"a": 96, "b": 84, "c": 24, "d": 96, "e": 96}),
        ("ecbfa,fd->abcde", {"a": 48, "b": 36, "c": 36, "d": 24, "e": 48, "f": 48}),
        ("deca,be->abcd", {"a": 96, "b": 24, "c": 84, "d": 96, "e": 84}),
        ("bda,dc->abc", {"a": 384, "b": 384, "c": 24, "d": 384}),
        ("ebad,ce->abcd", {"a": 96, "b": 84, "c": 24, "d": 84, "e": 96}),
        (
            "dega,gfbc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "dfgb,geac->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "degb,gfac->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "degc,gfab->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        ("dca,bd->abc", {"a": 384, "b": 24, "c": 376, "d": 384}),
        ("ea,ebcd->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("eb,aecd->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("ec,abed->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("adec,ebd->abc", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("cad,dcb->ab", {"a": 384, "b": 376, "c": 384, "d": 384}),
        ("acd,dbc->ab", {"a": 384, "b": 376, "c": 376, "d": 384}),
        ("acd,db->abc", {"a": 384, "b": 376, "c": 376, "d": 384}),
        ("adc,bd->abc", {"a": 384, "b": 384, "c": 376, "d": 376}),
        ("ac,cb->ab", {"a": 7248, "b": 7240, "c": 7248}),
        ("aebf,fdec->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 84, "f": 96}),
        ("eafd,fbec->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96, "f": 96}),
        ("aebf,dfce->abcd", {"a": 96, "b": 84, "c": 84, "d": 96, "e": 84, "f": 84}),
    ),
    "full": (
        ("bda,dc->abc", {"a": 384, "b": 384, "c": 24, "d": 384}),
        ("dca,bd->abc", {"a": 384, "b": 24, "c": 376, "d": 384}),
        ("dbea,ec->abcd", {"a": 96, "b": 84, "c": 24, "d": 96, "e": 96}),
        ("deca,be->abcd", {"a": 96, "b": 24, "c": 84, "d": 96, "e": 84}),
        ("ebad,ce->abcd", {"a": 96, "b": 84, "c": 24, "d": 84, "e": 96}),
        ("efbad,cf->abcde", {"a": 48, "b": 36, "c": 24, "d": 36, "e": 48, "f": 36}),
        ("ecbfa,fd->abcde", {"a": 48, "b": 36, "c": 36, "d": 24, "e": 48, "f": 48}),
        ("efcad,bf->abcde", {"a": 48, "b": 24, "c": 36, "d": 36, "e": 48, "f": 36}),
        ("ea,ebcd->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("eb,aecd->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("ec,abed->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("ac,cb->ab", {"a": 7248, "b": 7240, "c": 7248}),
        ("acd,dbc->ab", {"a": 384, "b": 376, "c": 376, "d": 384}),
        ("cad,dcb->ab", {"a": 384, "b": 376, "c": 384, "d": 384}),
        ("acd,db->abc", {"a": 384, "b": 376, "c": 376, "d": 384}),
        ("ad,bdc->abc", {"a": 384, "b": 384, "c": 376, "d": 376}),
        ("adc,bd->abc", {"a": 384, "b": 384, "c": 376, "d": 376}),
        ("adc,db->abc", {"a": 384, "b": 376, "c": 376, "d": 384}),
        ("adec,ebd->abc", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96}),
        ("aebf,dfce->abcd", {"a": 96, "b": 84, "c": 84, "d": 96, "e": 84, "f": 84}),
        ("aebf,fdec->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 84, "f": 96}),
        ("aecf,bfde->abcd", {"a": 96, "b": 96, "c": 84, "d": 84, "e": 84, "f": 84}),
        ("aecf,fbed->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 84, "f": 96}),
        ("aedf,bfce->abcd", {"a": 96, "b": 96, "c": 84, "d": 84, "e": 84, "f": 84}),
        ("aedf,fbec->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 84, "f": 96}),
        ("aefb,fdce->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 84, "f": 96}),
        ("aefc,fbed->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 84, "f": 96}),
        ("eafb,fdec->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96, "f": 96}),
        ("eafc,bfde->abcd", {"a": 96, "b": 96, "c": 84, "d": 84, "e": 96, "f": 84}),
        ("eafd,fbec->abcd", {"a": 96, "b": 84, "c": 84, "d": 84, "e": 96, "f": 96}),
        (
            "dega,gfbc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "degb,gfac->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "degc,gfab->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "dfga,gebc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "dfgb,geac->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "dfgc,geab->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "efga,gdbc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 20, "e": 24, "f": 20, "g": 24},
        ),
        (
            "efgb,gdac->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 20, "e": 24, "f": 20, "g": 24},
        ),
        (
            "efgc,gdab->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 20, "e": 24, "f": 20, "g": 24},
        ),
        (
            "gdab,efgc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 20, "e": 24, "f": 20, "g": 24},
        ),
        (
            "gdac,efgb->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 20, "e": 24, "f": 20, "g": 24},
        ),
        (
            "gdbc,efga->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 20, "e": 24, "f": 20, "g": 24},
        ),
        (
            "geab,dfgc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "geac,dfgb->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "gebc,dfga->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "gfab,degc->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "gfac,degb->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
        (
            "gfbc,dega->abcdef",
            {"a": 24, "b": 20, "c": 20, "d": 24, "e": 20, "f": 20, "g": 24},
        ),
    ),
}


_STAT_FUNCS: dict[str, Callable[[Sequence[float]], float]] = {
    "median": statistics.median,
    "min": min,
    "mean": statistics.fmean,
}


def _split_expr(expr: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return ``(in0_axes, in1_axes, out_axes)`` for a pairwise einsum string."""

    lhs, rhs = expr.split("->")
    in0, in1 = lhs.split(",")
    return tuple(in0), tuple(in1), tuple(rhs)


def _k_axes(expr: str) -> tuple[str, ...]:
    """Return the contraction K axes (present on both inputs, absent from output)."""

    in0, in1, out = _split_expr(expr)
    return tuple(a for a in in0 if a in set(in1) and a not in set(out))


def _k_volume(expr: str, extents: Mapping[str, int]) -> int:
    """Product of extents over the K axes; 1 if there are none."""

    return math.prod(extents[a] for a in _k_axes(expr)) or 1


def gflops(expr: str, extents: Mapping[str, int], execute_seconds: float) -> float:
    """Compute GFLOPS for a pairwise contraction as ``2 * out_volume * K_volume``."""

    _, _, out = _split_expr(expr)
    out_volume = math.prod(extents[a] for a in out)
    return 2 * out_volume * _k_volume(expr, extents) / execute_seconds / 1e9


#: Empirical safety factor over the probabilistic ``sqrt(K)*eps*|in|^2`` noise
#: floor for an f32 dot product of length ``K``. Calibrated against the
#: ``default`` corpus on tpp: the worst observed ratio is 22.30 on
#: ``acd,dbc->ab`` (K=144 384).
_FLOAT_TOLERANCE_SAFETY = 50.0


def verify_tolerance(
    expr: str,
    extents: Mapping[str, int],
    in0: np.ndarray,
    in1: np.ndarray,
    dtype: str,
) -> dict[str, float]:
    """Return ``atol`` / ``rtol`` kwargs for the verification ``assert_allclose``.

    ``atol`` floats with the dot-product noise floor of the contraction —
    ``safety * sqrt(K) * eps * |in0|_inf * |in1|_inf`` — so high-K entries
    don't fail on f32 accumulation rounding while low-K entries still gate
    at the small fixed floor.
    """

    eps = float(np.finfo(resolve_numpy_dtype(dtype)).eps)
    k_volume = _k_volume(expr, extents)
    in0_max = float(np.abs(in0).max())
    in1_max = float(np.abs(in1).max())
    noise_floor = math.sqrt(k_volume) * eps * in0_max * in1_max
    return {"atol": max(1e-5, _FLOAT_TOLERANCE_SAFETY * noise_floor), "rtol": 1e-4}


def _allocate_pair(
    expr: str, extents: Mapping[str, int], dtype: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate ``(in0, in1, out)`` for an einsum-emitted IR.

    Returns row-major ND arrays sized per the einsum string. Inputs carry
    pseudo-random values from a seeded PCG64; the output is zero-filled.
    The einsum emitter produces row-major byte strides per tensor, so
    these contiguous arrays match the IR's strides byte-for-byte.
    """

    in0, in1, out = _split_expr(expr)
    np_dtype = resolve_numpy_dtype(dtype)
    rng = np.random.default_rng(seed)
    in0_arr = rng.standard_normal(tuple(extents[a] for a in in0)).astype(
        np_dtype, copy=False
    )
    in1_arr = rng.standard_normal(tuple(extents[a] for a in in1)).astype(
        np_dtype, copy=False
    )
    out_arr = np.zeros(tuple(extents[a] for a in out), dtype=np_dtype)
    return in0_arr, in1_arr, out_arr


def _verify(
    expr: str,
    extents: Mapping[str, int],
    backend: str,
    dtype: str = "f32",
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compile and execute ``expr`` on ``backend``; return ``(compile_seconds, op, in0, in1, out)``."""

    teir = einsum(expr, dim_sizes=extents, dtype=dtype)
    in0, in1, out = _allocate_pair(
        expr, extents, dtype=dtype, seed=hash(expr) & 0xFFFFFFFF
    )
    compile_start = time.perf_counter()
    op = etops.compile(teir, backend=backend)
    compile_seconds = time.perf_counter() - compile_start
    op.execute(in0, in1, out)
    expected = np.einsum(expr, in0, in1, optimize=False)
    np.testing.assert_allclose(
        out, expected, **verify_tolerance(expr, extents, in0, in1, dtype)
    )
    return compile_seconds, op, in0, in1, out


def _measure(
    op: object,
    in0: np.ndarray,
    in1: np.ndarray,
    out: np.ndarray,
    warmup: int,
    measure: int,
) -> list[float]:
    """Time ``op.execute`` over ``warmup`` discarded + ``measure`` recorded runs."""

    for _ in range(warmup):
        op.execute(in0, in1, out)
    samples: list[float] = []
    for _ in range(measure):
        start = time.perf_counter()
        op.execute(in0, in1, out)
        samples.append(time.perf_counter() - start)
    return samples


def _format_extents(extents: Mapping[str, int]) -> str:
    """Space-joined ``axis=value`` rendering, sorted by axis name."""

    return " ".join(f"{ax}={extents[ax]}" for ax in sorted(extents))


def _resolve_backends(spec: str) -> tuple[str, ...]:
    if spec == "both":
        return ("tpp", "blas")
    if spec in ("tpp", "blas"):
        return (spec,)
    raise ValueError(f"unknown backend selector {spec!r}; expected tpp/blas/both")


def _csv_writer(path: Path) -> tuple[IO[str], Any]:
    """Open ``path`` for write and emit the canonical header row.

    Line buffering (``buffering=1``) flushes the underlying file after
    every ``\\n``-terminated write so a ``tail -f`` of the CSV reflects
    each completed ``(contraction, backend)`` row instead of buffering
    them all until ``run_setting`` returns.
    """

    fp = path.open("w", newline="", buffering=1)
    writer = csv.writer(fp)
    writer.writerow(
        [
            "contraction",
            "extents",
            "backend",
            "dtype",
            "compile_seconds",
            "execute_median_seconds",
            "execute_min_seconds",
            "execute_gflops_median",
            "warmup_runs",
            "measure_runs",
        ]
    )
    return fp, writer


def run_setting(
    setting: str = DEFAULT_SETTING,
    *,
    backends: str | Sequence[str] = "both",
    dtype: str = "f32",
    perf: bool = False,
    csv_path: Path | None = None,
    warmup: int = 3,
    measure: int = 10,
    stat: str = "median",
    verbose: bool = False,
    stream: IO[str] | None = None,
) -> int:
    """Verify (and optionally measure) every contraction in ``setting``.

    Returns the number of failed contractions; zero means every
    ``(contraction, backend)`` pair verified.
    """

    if setting not in CORPUS:
        raise ValueError(f"unknown corpus setting {setting!r}; known: {sorted(CORPUS)}")
    if stat not in _STAT_FUNCS:
        raise ValueError(f"unknown statistic {stat!r}; known: {sorted(_STAT_FUNCS)}")
    if csv_path is not None and not perf:
        raise ValueError("--csv requires --perf")

    out_stream = stream if stream is not None else sys.stdout
    backend_list = (
        _resolve_backends(backends) if isinstance(backends, str) else tuple(backends)
    )

    csv_fp: IO[str] | None = None
    csv_writer = None
    if csv_path is not None:
        csv_fp, csv_writer = _csv_writer(csv_path)

    failures = 0
    try:
        for expr, extents in CORPUS[setting]:
            for backend in backend_list:
                try:
                    compile_seconds, op, in0, in1, out = _verify(
                        expr, extents, backend, dtype=dtype
                    )
                except AssertionError as exc:
                    failures += 1
                    print(
                        f"[FAIL] {expr} on {backend}: {exc}",
                        file=out_stream,
                    )
                    continue
                except etops.TeirError as exc:
                    failures += 1
                    print(
                        f"[ERROR] {expr} on {backend}: {type(exc).__name__}: {exc}",
                        file=out_stream,
                    )
                    continue

                if verbose:
                    print(
                        f"[OK] {expr:36s}  backend={backend:5s}"
                        f"  extents=({_format_extents(extents)})",
                        file=out_stream,
                    )

                if not perf:
                    continue

                samples = _measure(op, in0, in1, out, warmup, measure)
                median_s = statistics.median(samples)
                min_s = min(samples)
                gf_median = gflops(expr, extents, median_s)
                if verbose:
                    chosen = _STAT_FUNCS[stat](samples)
                    print(
                        f"       compile={compile_seconds * 1e3:8.3f} ms"
                        f"  exec_{stat}={chosen * 1e3:8.3f} ms"
                        f"  gflops_median={gf_median:7.2f}",
                        file=out_stream,
                    )
                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            expr,
                            _format_extents(extents),
                            backend,
                            dtype,
                            f"{compile_seconds:.6f}",
                            f"{median_s:.6f}",
                            f"{min_s:.6f}",
                            f"{gf_median:.3f}",
                            warmup,
                            measure,
                        ]
                    )
    finally:
        if csv_fp is not None:
            csv_fp.close()

    total = len(CORPUS[setting]) * len(backend_list)
    passed = total - failures
    print(
        f"{passed}/{total} verified on backends={','.join(backend_list)}"
        f"  setting={setting}  dtype={dtype}",
        file=out_stream,
    )
    if csv_path is not None:
        print(f"wrote {csv_path}", file=out_stream)
    return failures
