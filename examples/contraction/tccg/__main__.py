"""CLI entry: ``python -m examples.contraction.tccg [...]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from examples.contraction.tccg import CORPUS, DEFAULT_SETTING, run_setting


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify (and optionally measure) the TCCG pairwise tensor"
            " contraction benchmark corpus."
        ),
    )
    parser.add_argument(
        "--setting",
        default=DEFAULT_SETTING,
        choices=sorted(CORPUS),
        help=f"Corpus subset to run (default: {DEFAULT_SETTING}).",
    )
    parser.add_argument(
        "--backend",
        default="both",
        choices=["tpp", "blas", "both"],
        help="Backend(s) to verify against (default: both).",
    )
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=["f32", "f64"],
        help="Element type used for every contraction (default: f32).",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Measure execute() wall-time after verification.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write per-(contraction, backend) timing rows to PATH (requires --perf).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of discarded warmup runs per contraction (default: 3).",
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=10,
        help="Number of measured runs per contraction (default: 10).",
    )
    parser.add_argument(
        "--stat",
        default="median",
        choices=["median", "min", "mean"],
        help="Statistic shown in --verbose output (default: median).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print a line per verified contraction (default: pass/fail summary only).",
    )
    args = parser.parse_args()

    if args.csv is not None and not args.perf:
        parser.error("--csv requires --perf")

    failures = run_setting(
        setting=args.setting,
        backends=args.backend,
        dtype=args.dtype,
        perf=args.perf,
        csv_path=args.csv,
        warmup=args.warmup,
        measure=args.measure,
        stat=args.stat,
        verbose=args.verbose,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
