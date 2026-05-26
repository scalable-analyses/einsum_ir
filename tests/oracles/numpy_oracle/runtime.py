"""Pure-Python schedule walker for the NumPy reference backend."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from etops.diag import TeirLoweringError, TeirRuntimeError
from etops.ir import (
    First,
    Guard,
    InvocationNode,
    IterationNode,
    Last,
    Primitive,
    Teir,
    validate,
)
from etops.ir.dtypes import resolve_numpy_dtype
from tests.oracles.numpy_oracle import primitives as numpy_prims
from tests.oracles.numpy_oracle._tile import TileViewBuilder

__all__ = ["NumpyOperation", "compile_to_numpy"]

_LOG = logging.getLogger(__name__)


PrimitiveExecutor = Callable[
    [Teir, Primitive, TileViewBuilder, dict],
    None,
]


_REGISTRY: dict[str, PrimitiveExecutor] = {
    "Zero": numpy_prims.execute_zero,
    "Copy": numpy_prims.execute_copy,
    "ReLU": numpy_prims.execute_relu,
    "Contraction": numpy_prims.execute_contraction,
}


def compile_to_numpy(teir: Teir) -> NumpyOperation:
    """Compile ``teir`` for execution by the NumPy reference backend.

    Validation runs prior to compilation. The returned operation is
    callable via ``execute(*tensors)`` where each tensor is a NumPy array
    whose dtype matches the IR declaration.
    """

    validate(teir)
    return NumpyOperation(teir)


class NumpyOperation:
    """Compiled `Teir` ready for execution against NumPy tensors."""

    def __init__(self, teir: Teir) -> None:
        self._teir = teir

    @property
    def teir(self) -> Teir:
        """Return the underlying `Teir`."""

        return self._teir

    @property
    def backend(self) -> str:
        """Return the backend identifier."""

        return "numpy"

    def execute(self, *tensors: np.ndarray) -> None:
        """Execute the schedule against ``tensors`` in IR declaration order."""

        teir = self._teir
        if len(tensors) != len(teir.tensors):
            msg = (
                f"NumpyOperation.execute: expected {len(teir.tensors)} tensors;"
                f" got {len(tensors)}"
            )
            raise TeirRuntimeError(msg)

        flats: dict[str, np.ndarray] = {}
        for tid, arr in zip(teir.tensor_ids, tensors, strict=True):
            expected_dtype = resolve_numpy_dtype(teir.tensors[tid].dtype.name)
            if arr.dtype != expected_dtype:
                msg = (
                    f"tensor {tid!r}: dtype {arr.dtype} does not match"
                    f" IR-declared dtype {expected_dtype}"
                )
                raise TeirRuntimeError(msg)
            flats[tid] = arr.reshape(-1)

        builder = TileViewBuilder(teir, flats)
        _walk_iter(teir, builder)


# --------------------------------------------------------------------------
# Iterative walker
# --------------------------------------------------------------------------


def _evaluate_guard(
    guard: Guard | None, ancestor_indices: Mapping[str, int], teir: Teir
) -> bool:
    """Return True if the guard's terms all hold for the current state."""

    if guard is None:
        return True
    for term in guard:
        if term.axis not in ancestor_indices:
            msg = f"guard references non-ancestor axis {term.axis!r}"
            raise TeirLoweringError(msg)
        idx = ancestor_indices[term.axis]
        extent = teir.axes[term.axis].extent
        if isinstance(term, First):
            if idx != 0:
                return False
        elif isinstance(term, Last):
            if idx != extent - 1:
                return False
        else:  # pragma: no cover - validated upstream
            msg = f"unknown guard term: {type(term).__name__}"
            raise TeirLoweringError(msg)
    return True


def _walk_iter(teir: Teir, builder: TileViewBuilder) -> None:
    """Walk the schedule forest iteratively.

    The work stack contains frames describing both "enter a node with these
    ancestor indices" and "step to the next iteration of an axis at this
    depth" actions; we materialize each leaf invocation only when its
    ancestor chain is finalized.
    """

    sched = teir.schedule
    iterations = sched.iterations
    invocations = sched.invocations
    ancestor_indices: dict[str, int] = {}

    # Frame kinds for the work stack.
    _ENTER = 0
    _CONTINUE = 1
    _EXIT = 2

    # ENTER frames carry just (action, node_id).
    # CONTINUE frames carry (action, (iter_node_id, axis_id, extent, next_idx)).
    # EXIT frames carry (action, axis_id).
    Frame = tuple[int, Any]
    stack: list[Frame] = [(_ENTER, root) for root in reversed(sched.roots)]

    while stack:
        action, payload = stack.pop()
        if action == _ENTER:
            node_id_str: str = payload
            if node_id_str in iterations:
                it_node: IterationNode = iterations[node_id_str]
                if not _evaluate_guard(it_node.guard, ancestor_indices, teir):
                    continue
                extent = teir.axes[it_node.axis].extent
                if it_node.policy not in ("sequential", "parallel"):
                    raise TeirLoweringError(
                        f"unknown iteration policy: {it_node.policy!r}"
                    )
                if it_node.policy == "parallel":
                    _LOG.debug(
                        "numpy backend: walking parallel iteration node %s sequentially",
                        node_id_str,
                    )
                if extent <= 0:
                    continue
                ancestor_indices[it_node.axis] = 0
                stack.append((_EXIT, it_node.axis))
                stack.append((_CONTINUE, (node_id_str, it_node.axis, extent, 1)))
                for cid in reversed(it_node.children):
                    stack.append((_ENTER, cid))
                continue
            if node_id_str in invocations:
                inv_node: InvocationNode = invocations[node_id_str]
                if not _evaluate_guard(inv_node.guard, ancestor_indices, teir):
                    continue
                primitive = teir.primitives[inv_node.primitive]
                executor = _REGISTRY.get(primitive.operation)
                if executor is None:
                    raise TeirLoweringError(
                        f"numpy backend has no lowering for operation {primitive.operation!r}"
                    )
                bases = {
                    tid: builder.base_byte_address(tid, ancestor_indices)
                    for tid in teir.tensor_ids
                }
                executor(teir, primitive, builder, bases)
                continue
            raise TeirLoweringError(f"unknown schedule node id: {node_id_str!r}")

        if action == _CONTINUE:
            cont_node_id, cont_axis, cont_extent, cont_next = payload
            if cont_next >= cont_extent:
                continue
            ancestor_indices[cont_axis] = cont_next
            stack.append(
                (_CONTINUE, (cont_node_id, cont_axis, cont_extent, cont_next + 1))
            )
            cont_it_node: IterationNode = iterations[cont_node_id]
            for cid in reversed(cont_it_node.children):
                stack.append((_ENTER, cid))
            continue

        # _EXIT
        exit_axis: str = payload
        ancestor_indices.pop(exit_axis, None)
