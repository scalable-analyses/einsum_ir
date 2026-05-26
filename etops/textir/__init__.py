"""Textual TEIR: canonical serialization with round-trip guarantees.

`parse(dump(teir)) == teir` for every well-formed `Teir`. The format is
MLIR-flavored — structured operations, sigils, region-style nesting — but
takes no MLIR dependency.
"""

from __future__ import annotations

from etops.textir.parser import parse
from etops.textir.printer import dump

__all__ = ["dump", "parse"]
