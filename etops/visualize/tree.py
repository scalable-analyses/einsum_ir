"""Indent-based pretty-printer for `Teir` configurations."""

from __future__ import annotations

from etops.ir import Teir
from etops.ir._records import format_guard as _format_guard
from etops.ir.cursors import pre_order

__all__ = ["show"]


def show(teir: Teir) -> str:
    """Return a human-friendly summary of ``teir``.

    The schedule is rendered with two-space indentation per nesting level.
    Output is not round-trippable; use `etops.textir.dump` for that.
    """

    lines: list[str] = []
    head = f"teir @{teir.name}" if teir.name else "teir"
    lines.append(head)
    if teir.tensors:
        lines.append(
            "  tensors: "
            + ", ".join(f"{t.id} ({t.dtype.name})" for t in teir.tensors.values())
        )
    if teir.axes:
        lines.append("  axes:")
        for a in teir.axes.values():
            parts = [f"{a.id} [{a.extent}]"]
            if a.strides:
                entries = ", ".join(f"{t}: {v}" for t, v in a.strides.items())
                parts.append(f"strides{{{entries}}}")
            if a.offsets:
                entries = ", ".join(f"{t}: {v}" for t, v in a.offsets.items())
                parts.append(f"offsets{{{entries}}}")
            lines.append("    " + "  ".join(parts))
    if teir.primitives:
        lines.append("  primitives:")
        for pid, prim in teir.primitives.items():
            roles = ", ".join(
                f"{r}=[{','.join(prim.axes.get(r, ()))}]" for r in prim.axes
            )
            meta = ", ".join(f"{k}={v}" for k, v in prim.metadata.items())
            tail = f"  ({meta})" if meta else ""
            lines.append(f"    @{pid}: {prim.operation} {{{roles}}}{tail}")
    lines.append("  schedule:")
    sched = teir.schedule
    depth: dict[str, int] = {nid: 0 for nid in sched.roots}
    for nid in pre_order(teir):
        indent = "  " * (2 + depth.get(nid, 0))
        if nid in sched.iterations:
            it = sched.iterations[nid]
            descr = (
                f"iter @{nid} axis=@{it.axis} policy={it.policy}"
                f" extent={teir.axes[it.axis].extent}"
            )
            if it.guard is not None:
                descr += f"  [guard {_format_guard(it.guard)}]"
            lines.append(indent + descr)
            for cid in it.children:
                depth[cid] = depth.get(nid, 0) + 1
        elif nid in sched.invocations:
            inv = sched.invocations[nid]
            descr = f"invoke @{nid} primitive=@{inv.primitive}"
            if inv.guard is not None:
                descr += f"  [guard {_format_guard(inv.guard)}]"
            lines.append(indent + descr)
    return "\n".join(lines) + "\n"
