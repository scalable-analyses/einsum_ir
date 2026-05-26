"""Canonical textual TEIR printer."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from etops.diag import TeirEmissionError
from etops.ir import Teir
from etops.ir._records import format_guard as _format_guard

__all__ = ["dump"]

FORMAT_VERSION = "1.0"

# A metadata string value is printed bare (no quotes) iff it matches this
# regex. Anything else, plus `true` / `false` (which would otherwise round-
# trip as booleans), is double-quoted with backslash escapes.
_BARE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_RESERVED_BARE = frozenset({"true", "false"})


def dump(teir: Teir, *, include_header: bool = True) -> str:
    """Serialize a `Teir` to the canonical textual format.

    The output is intentionally stable: tensors, axes, and primitives emit
    in their insertion order; strides and offsets within an axis emit in
    lexicographic tensor order; iteration nodes emit first, then invocation
    nodes, each group preserving insertion order.
    """

    lines: list[str] = []
    if include_header:
        lines.append(f"teir-format {FORMAT_VERSION}")
        lines.append("")
    head = f"teir @{teir.name} {{" if teir.name else "teir {"
    lines.append(head)

    # Tensors
    for tid, tensor in teir.tensors.items():
        lines.append(f"  tensor %{tid} : {tensor.dtype.name}")
    if teir.tensors:
        lines.append("")

    # Axes
    for aid, axis in teir.axes.items():
        parts = [f"  axis @{aid} extent {axis.extent}"]
        if axis.strides:
            entries = ", ".join(
                f"{tid}: {value}" for tid, value in sorted(axis.strides.items())
            )
            parts.append(f"strides {{ {entries} }}")
        if axis.offsets:
            entries = ", ".join(
                f"{tid}: {value}" for tid, value in sorted(axis.offsets.items())
            )
            parts.append(f"offsets {{ {entries} }}")
        lines.append(" ".join(parts))
    if teir.axes:
        lines.append("")

    # Primitives
    for pid, prim in teir.primitives.items():
        roles_entries = []
        for role, axes in prim.axes.items():
            inner = ", ".join(f"@{a}" for a in axes)
            roles_entries.append(f"{role}: [{inner}]")
        roles = ", ".join(roles_entries)
        head = f"  primitive @{pid} : {prim.operation} axes {{ {roles} }}"
        if prim.metadata:
            head += f" metadata {{ {_format_metadata_map(prim.metadata)} }}"
        lines.append(head)
    if teir.primitives:
        lines.append("")

    # Schedule
    lines.append("  schedule {")
    roots_inner = ", ".join(f"@{r}" for r in teir.schedule.roots)
    lines.append(f"    roots [{roots_inner}]")
    lines.append("")
    for nid, node in teir.schedule.iterations.items():
        kids = ", ".join(f"@{c}" for c in node.children)
        line = (
            f"    iter @{nid} axis @{node.axis} policy {node.policy} children [{kids}]"
        )
        if node.guard is not None:
            line += f" guard {_format_guard(node.guard)}"
        if node.metadata:
            line += f" metadata {{ {_format_metadata_map(node.metadata)} }}"
        lines.append(line)
    for nid, inv in teir.schedule.invocations.items():
        line = f"    invoke @{nid} primitive @{inv.primitive}"
        if inv.guard is not None:
            line += f" guard {_format_guard(inv.guard)}"
        if inv.metadata:
            line += f" metadata {{ {_format_metadata_map(inv.metadata)} }}"
        lines.append(line)
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _format_metadata_map(metadata: Mapping[str, Any]) -> str:
    """Format ``metadata`` as a comma-separated ``k: v`` list, sorted by key."""

    items = sorted(metadata.items())
    return ", ".join(f"{k}: {_format_metadata_value(v)}" for k, v in items)


def _format_metadata_value(value: object) -> str:
    """Format a metadata scalar as its textual-IR literal.

    The textual IR supports four concrete value kinds: ``bool`` (rendered
    as ``true`` / ``false``), ``int`` (decimal), ``float`` (Python
    ``repr``: always contains a fractional point or exponent), and
    ``str`` (bare identifier where unambiguous, double-quoted with
    backslash escapes otherwise).
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TeirEmissionError(
                f"cannot serialize non-finite metadata float {value!r};"
                " textual TEIR metadata floats must be finite"
            )
        text = repr(value)
        if "." not in text and "e" not in text and "E" not in text:
            text = text + ".0"
        return text
    if isinstance(value, str):
        if value not in _RESERVED_BARE and _BARE_IDENT_RE.match(value):
            return value
        escaped = (
            value.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
            .replace("\0", "\\0")
        )
        return f'"{escaped}"'
    raise TypeError(
        f"cannot serialize metadata value of type {type(value).__name__};"
        " supported textual-IR metadata types are bool, int, float, str"
    )
