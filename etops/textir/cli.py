"""CLI entry points for the textual TEIR.

Subcommands:

- ``cat``      — parse and re-emit. ``--format tree`` renders the
                 human-friendly schedule tree instead of canonical text.
- ``validate`` — well-formedness check (exit non-zero on failure).
- ``run``      — execute the IR end-to-end against ``.npy`` inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from etops.diag import TeirError
from etops.ir import validate
from etops.textir.parser import parse
from etops.textir.printer import dump

__all__ = ["main"]


def _cmd_cat(args: argparse.Namespace) -> int:
    text = Path(args.path).read_text()
    teir = parse(text, validate=not args.no_validate)
    if args.format == "tree":
        from etops.visualize import show

        sys.stdout.write(show(teir))
    else:
        sys.stdout.write(dump(teir))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    text = Path(args.path).read_text()
    teir = parse(text)
    validate(teir)
    if not args.quiet:
        sys.stdout.write("OK\n")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import etops
    from etops.diag import TeirEmissionError

    text = Path(args.path).read_text()
    teir = parse(text)
    if args.backend not in etops.list_backends():
        raise TeirEmissionError(
            f"backend {args.backend!r} is not registered; available: "
            + ", ".join(etops.list_backends())
        )
    inputs = [np.load(p) for p in args.inputs]
    if len(inputs) != len(teir.tensors):
        raise TeirEmissionError(
            f"--inputs expected {len(teir.tensors)} entries; got {len(inputs)}"
        )
    if args.output_tensor is not None:
        if args.output_tensor not in teir.tensor_ids:
            raise TeirEmissionError(
                f"--output-tensor {args.output_tensor!r} is not a declared"
                f" tensor; known: {list(teir.tensor_ids)}"
            )
        out_index = list(teir.tensor_ids).index(args.output_tensor)
    else:
        # Convention: the last tensor in declaration order is the output.
        out_index = len(teir.tensors) - 1
    op = etops.compile(teir, backend=args.backend)
    op.execute(*inputs)
    if args.output:
        np.save(args.output, inputs[out_index])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m etops.textir`` and ``etops-textir``."""

    parser = argparse.ArgumentParser(
        prog="etops-textir",
        description="Textual TEIR utility (cat, show, validate, run).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cat = sub.add_parser("cat", help="Parse and re-emit a textual TEIR file.")
    p_cat.add_argument("path", help="Path to a textual TEIR file.")
    p_cat.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip global validation; useful for pretty-printing files that"
            " are still being authored."
        ),
    )
    p_cat.add_argument(
        "--format",
        choices=("canonical", "tree"),
        default="canonical",
        help="Output format: 'canonical' (round-trippable text) or 'tree'.",
    )
    p_cat.set_defaults(func=_cmd_cat)

    p_val = sub.add_parser("validate", help="Validate a textual TEIR file.")
    p_val.add_argument("path", help="Path to a textual TEIR file.")
    p_val.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress 'OK' on success."
    )
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Execute a textual TEIR file.")
    p_run.add_argument("path", help="Path to a textual TEIR file.")
    p_run.add_argument("--backend", required=True, help="Backend name.")
    p_run.add_argument(
        "--inputs",
        nargs="+",
        default=[],
        help="NumPy .npy files for each tensor in IR order.",
    )
    p_run.add_argument(
        "--output", help="If set, write the output tensor's contents to this .npy path."
    )
    p_run.add_argument(
        "--output-tensor",
        default=None,
        help=(
            "Declared tensor id whose contents are written to --output;"
            " defaults to the last tensor in IR declaration order."
        ),
    )
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TeirError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
