"""Convert an `etops.Teir` into the plain-dict form ``_native`` consumes."""

from __future__ import annotations

from typing import Any

from etops.ir import (
    First,
    Guard,
    InvocationNode,
    IterationNode,
    Teir,
)

__all__ = ["teir_to_dict"]


def teir_to_dict(teir: Teir) -> dict[str, Any]:
    """Marshal a `Teir` into the dict form expected by ``_native``."""

    return {
        "tensors": [
            {
                "id": tid,
                "dtype": {"name": t.dtype.name, "bits": int(t.dtype.bits)},
            }
            for tid, t in teir.tensors.items()
        ],
        "axes": [
            {
                "id": aid,
                "extent": int(a.extent),
                "strides": dict(a.strides),
                "offsets": dict(a.offsets),
            }
            for aid, a in teir.axes.items()
        ],
        "primitives": [
            {
                "id": pid,
                "operation": p.operation,
                "axes": {role: list(axes) for role, axes in p.axes.items()},
                "metadata": {k: _metadata_str(v) for k, v in p.metadata.items()},
            }
            for pid, p in teir.primitives.items()
        ],
        "schedule": {
            "roots": list(teir.schedule.roots),
            "iterations": [
                _iter_to_dict(node) for node in teir.schedule.iterations.values()
            ],
            "invocations": [
                _inv_to_dict(node) for node in teir.schedule.invocations.values()
            ],
        },
    }


def _iter_to_dict(node: IterationNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "axis": node.axis,
        "policy": node.policy,
        "children": list(node.children),
        "guard": _guard_to_list(node.guard),
        "num_threads": int(node.metadata.get("threading.num_threads", 0)),
    }


def _inv_to_dict(node: InvocationNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "primitive": node.primitive,
        "guard": _guard_to_list(node.guard),
    }


def _guard_to_list(guard: Guard | None) -> list[dict[str, str]] | None:
    if guard is None:
        return None
    out: list[dict[str, str]] = []
    for term in guard:
        kind = "first" if isinstance(term, First) else "last"
        out.append({"kind": kind, "axis": term.axis})
    return out


def _metadata_str(value: Any) -> str:
    """Serialize a metadata value as a string.

    The C++ runtime stores metadata as the encoded string (the same
    representation the textual IR emits); backends decode it as needed.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    raise TypeError(
        f"cannot marshal metadata value of type {type(value).__name__};"
        " supported metadata types are bool, int, float, str"
    )
