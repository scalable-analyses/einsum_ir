"""Schedule traversal helpers.

Ordered walks (`pre_order`, `post_order`, `iteration_nodes_outer_first`,
`descendant_invocations`) plus a parent-map utility shared across passes
and transforms.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from etops.ir._records import Teir

__all__ = [
    "descendant_invocations",
    "iteration_nodes_outer_first",
    "parent_map",
    "post_order",
    "pre_order",
]


def parent_map(teir: Teir) -> dict[str, str | None]:
    """Return ``node_id → parent_id`` for every schedule node.

    Roots map to ``None``.
    """

    schedule = teir.schedule
    parent: dict[str, str | None] = {nid: None for nid in schedule.iterations}
    for nid in schedule.invocations:
        parent.setdefault(nid, None)
    for pid, node in schedule.iterations.items():
        for cid in node.children:
            parent[cid] = pid
    return parent


def pre_order(teir: Teir) -> Iterator[str]:
    """Yield every schedule node id in pre-order (parents before children)."""

    schedule = teir.schedule
    iterations = schedule.iterations
    seen: set[str] = set()
    stack: list[str] = list(reversed(schedule.roots))
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        yield nid
        iter_node = iterations.get(nid)
        if iter_node is not None:
            stack.extend(reversed(iter_node.children))


def post_order(teir: Teir) -> Iterator[str]:
    """Yield every schedule node id in post-order (children before parents)."""

    schedule = teir.schedule
    iterations = schedule.iterations
    seen: set[str] = set()
    stack: list[tuple[str, bool]] = [(r, False) for r in reversed(schedule.roots)]
    while stack:
        nid, visited = stack.pop()
        if visited:
            yield nid
            continue
        if nid in seen:
            continue
        seen.add(nid)
        stack.append((nid, True))
        iter_node = iterations.get(nid)
        if iter_node is not None:
            for cid in reversed(iter_node.children):
                stack.append((cid, False))


def iteration_nodes_outer_first(teir: Teir) -> Iterator[str]:
    """Yield iteration node ids in breadth-first, outer-first order."""

    sched = teir.schedule
    seen: set[str] = set()
    queue: deque[str] = deque(sched.roots)
    while queue:
        nid = queue.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        iter_node = sched.iterations.get(nid)
        if iter_node is not None:
            yield nid
            queue.extend(iter_node.children)


def descendant_invocations(teir: Teir, root: str) -> Iterator[str]:
    """Yield every invocation reachable from ``root`` in the schedule forest.

    Yields ``root`` itself when it is an invocation; otherwise walks the
    iteration subtree in pre-order, yielding invocation leaves.
    """

    sched = teir.schedule
    if root in sched.invocations:
        yield root
        return
    iterations = sched.iterations
    if root not in iterations:
        return
    seen: set[str] = set()
    stack: list[str] = list(iterations[root].children)
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        if nid in sched.invocations:
            yield nid
        elif nid in iterations:
            stack.extend(reversed(iterations[nid].children))
