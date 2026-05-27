"""Axis-level passes: trivial-axis drop and orphan-axis cleanup."""

from __future__ import annotations

import logging

from etops.ir import Teir
from etops.passes._framework import PassContext

_LOG = logging.getLogger(__name__)

__all__ = [
    "DropTrivialAxes",
    "DropUnusedAxes",
]


def DropTrivialAxes(teir: Teir, ctx: PassContext) -> Teir:
    """Remove size-1 axes that no primitive consumes."""

    consumed: set[str] = set()
    for prim in teir.primitives.values():
        for axes in prim.axes.values():
            consumed.update(axes)
    to_drop = [
        aid
        for aid, axis in teir.axes.items()
        if axis.extent == 1
        and aid not in consumed
        and not any(off != 0 for off in axis.offsets.values())
    ]
    if not to_drop:
        return teir
    builder = teir.builder()
    for aid in to_drop:
        target_node_ids = [
            nid
            for nid, node in teir.schedule.iterations.items()
            if node.axis == aid and builder.has_node(nid)
        ]
        for nid in target_node_ids:
            children = builder.iteration_children(nid)
            if len(children) != 1:
                _LOG.warning(
                    f"DropTrivialAxes: axis {aid!r} iteration has"
                    f" {len(children)} children; keeping it"
                )
                continue
            builder.remove_iteration(nid, reparent_children=True)
        still_used = any(
            builder.iteration_axis(nid) == aid
            for nid in teir.schedule.iterations
            if builder.has_node(nid)
        )
        if not still_used and builder.has_axis(aid):
            builder.remove_axis(aid)
    return builder.finish(validate=False)


def DropUnusedAxes(teir: Teir, ctx: PassContext) -> Teir:
    """Remove axes that no schedule iteration or primitive role references."""

    used: set[str] = set()
    for node in teir.schedule.iterations.values():
        used.add(node.axis)
    for prim in teir.primitives.values():
        for axes in prim.axes.values():
            used.update(axes)
    orphans = [aid for aid in teir.axes if aid not in used]
    if not orphans:
        return teir
    builder = teir.builder()
    for aid in orphans:
        builder.remove_axis(aid)
    return builder.finish(validate=False)
