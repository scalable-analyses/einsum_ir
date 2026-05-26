"""CLI entry: ``python -m examples.spec.permutation [--backend {tpp,blas}] [--schedule {scalar,tiled}] [--show]``."""

from __future__ import annotations

import argparse

from examples.spec.permutation import DEFAULT_SCHEDULE, SCHEDULES, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the abcd->dcba permutation example.",
    )
    parser.add_argument(
        "--backend",
        default="tpp",
        choices=["tpp", "blas"],
        help="Backend to compile and execute against (default: tpp).",
    )
    parser.add_argument(
        "--schedule",
        default=None,
        choices=sorted(SCHEDULES),
        help=f"Which .teir schedule to parse and execute (default: {DEFAULT_SCHEDULE}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the parsed IR tree before executing.",
    )
    args = parser.parse_args()
    run(backend=args.backend, schedule=args.schedule, show=args.show)


if __name__ == "__main__":
    main()
