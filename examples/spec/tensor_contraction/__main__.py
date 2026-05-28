"""CLI entry: ``python -m examples.spec.tensor_contraction [--backend {tpp,blas}] [--name NAME] [--show]``."""

from __future__ import annotations

import argparse

from examples.spec.tensor_contraction import ARTIFACTS, DEFAULT_NAME, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tensor-contraction spec artifact.",
    )
    parser.add_argument(
        "--backend",
        default="tpp",
        choices=["tpp", "blas"],
        help="Backend to compile and execute against (default: tpp).",
    )
    parser.add_argument(
        "--name",
        default=None,
        choices=sorted(ARTIFACTS),
        help=f"Artifact name — any .teir stem or CASES key (default: {DEFAULT_NAME}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the parsed and prepared IR trees before executing.",
    )
    args = parser.parse_args()
    run(backend=args.backend, name=args.name, show=args.show)


if __name__ == "__main__":
    main()
