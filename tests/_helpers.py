"""Shared utilities for the etops test suite.

Backend availability + parametrize machinery, numerical tolerance lookup,
IR-stride reflection, and the stride-honoring operand allocator used by
``tests/test_spec_examples.py`` and the spec-directory CLIs.
"""

from __future__ import annotations

import numpy as np
import pytest

from etops.ir import Teir
from etops.ir.dtypes import resolve_numpy_dtype

__all__ = [
    "allocate_operands",
    "backend_available",
    "backend_param",
    "operand_view",
    "tensor_axes_in_storage_order",
    "tensor_shape_from_strides",
    "tol_for",
]


def backend_available(name: str) -> bool:
    """Return True if backend ``name`` is registered and dispatchable.

    ``numpy`` is registered unconditionally by ``conftest.py``. The native
    backends (``tpp``, ``blas``) depend on the C++ extension and on the
    primitive registry advertising ``Contraction`` for the backend.
    """

    if name == "numpy":
        return True
    try:
        from etops import _native
    except ImportError:
        return False
    return _native.config.has_primitive(name, "Contraction")


def backend_param(*backends: str) -> pytest.MarkDecorator:
    """Build a parametrize decorator over ``backends`` with skip + marker support.

    Each backend gets its own pytest marker (``numpy`` / ``tpp`` / ``blas``)
    plus a skip-if-unavailable guard for the native ones.
    """

    params = []
    for b in backends:
        marks: list[pytest.MarkDecorator] = [getattr(pytest.mark, b)]
        if b != "numpy":
            marks.append(
                pytest.mark.skipif(
                    not backend_available(b),
                    reason=f"{b.upper()} backend unavailable",
                )
            )
        params.append(pytest.param(b, marks=marks))
    return pytest.mark.parametrize("backend", params)


def tol_for(kind: str, dtype: str) -> dict[str, float]:
    """Return ``atol`` / ``rtol`` kwargs for ``np.testing.assert_allclose``.

    ``kind`` selects the operation class — ``exact`` for byte-equivalent
    transforms (permutation copies), ``mul`` for single multiplications
    (BGEMM), and ``contract`` for longer reductions (tensor contraction).
    """

    if kind == "exact":
        return {"atol": 0.0}
    if dtype == "f64":
        return {"atol": 1e-12, "rtol": 1e-12}
    if kind == "mul":
        return {"atol": 1e-5, "rtol": 1e-5}
    if kind == "contract":
        return {"atol": 1e-4, "rtol": 1e-4}
    raise ValueError(f"unknown tolerance kind {kind!r}")


def tensor_axes_in_storage_order(teir: Teir, tensor_id: str) -> list[str]:
    """Axis ids used by ``tensor_id``, ordered by descending byte stride.

    The einsum emitter produces row-major strides, so descending stride
    matches the left-to-right axis order in the original expression.
    """

    used = [
        (aid, axis.stride_on(tensor_id))
        for aid, axis in teir.axes.items()
        if axis.stride_on(tensor_id) > 0
    ]
    used.sort(key=lambda pair: -pair[1])
    return [aid for aid, _ in used]


def tensor_shape_from_strides(teir: Teir, tensor_id: str) -> tuple[int, ...]:
    """Recover a tensor's logical row-major shape from its byte strides."""

    return tuple(
        teir.axes[a].extent for a in tensor_axes_in_storage_order(teir, tensor_id)
    )


def _operand_capacity(teir: Teir, tensor_id: str) -> int:
    """Element count needed to back the IR's per-tensor strides without overrun."""

    elem_bytes = teir.tensors[tensor_id].dtype.bytes
    used = [
        (axis.extent, axis.stride_on(tensor_id))
        for axis in teir.axes.values()
        if axis.stride_on(tensor_id) > 0
    ]
    if not used:
        return 1
    max_off = sum((extent - 1) * stride for extent, stride in used)
    return max_off // elem_bytes + 1


def operand_view(teir: Teir, tensor_id: str, flat: np.ndarray) -> np.ndarray:
    """Strided ND view of ``flat`` shaped per ``tensor_id``'s byte strides.

    When the IR's strides correspond to a contiguous row-major layout this
    returns a view whose memory layout matches a plain ``reshape`` of
    ``flat``; otherwise it produces a non-contiguous strided view via
    ``np.lib.stride_tricks.as_strided`` over the same backing memory.
    """

    used = [
        (aid, axis.stride_on(tensor_id))
        for aid, axis in teir.axes.items()
        if axis.stride_on(tensor_id) > 0
    ]
    if not used:
        return flat.reshape(())
    used.sort(key=lambda pair: -pair[1])
    shape = tuple(teir.axes[aid].extent for aid, _ in used)
    bstrides = tuple(stride for _, stride in used)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=bstrides)


def allocate_operands(
    teir: Teir,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Per-tensor ``(flats, views, out_index)`` honoring the IR's byte strides.

    ``flats[i]`` is a 1-D backing buffer sized to cover every byte the IR's
    strides on ``teir.tensor_ids[i]`` can reach; the backends interpret
    arrays as flat memory addressed by the IR's strides, so passing the
    1-D buffers keeps the contract uniform regardless of the per-tensor
    stride pattern. ``views[i]`` is a per-tensor ND view of the same
    backing buffer suitable for oracle inputs and for comparing the
    post-execute output. ``out_index`` is the index into both lists of
    the output tensor (the literal ``"out"`` if present, otherwise the
    last declared tensor). Per-tensor seeds are derived from
    ``hash((teir.name, tid))`` so two runs of the same IR produce the
    same operand values.
    """

    flats: list[np.ndarray] = []
    views: list[np.ndarray] = []
    out_index = -1
    for tid in teir.tensor_ids:
        if tid == "out":
            out_index = len(flats)
        np_dtype = resolve_numpy_dtype(teir.tensors[tid].dtype.name)
        rng = np.random.default_rng(seed=hash((teir.name, tid)) & 0xFFFFFFFF)
        flat = rng.standard_normal(_operand_capacity(teir, tid)).astype(
            np_dtype, copy=False
        )
        flats.append(flat)
        views.append(operand_view(teir, tid, flat))
    if out_index == -1:
        out_index = len(flats) - 1
    flats[out_index][...] = 0
    return flats, views, out_index
