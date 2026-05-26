"""Shared utilities for the etops test suite.

Backend availability + parametrize machinery, numerical tolerance lookup,
and IR-stride reflection used across several test modules.
"""

from __future__ import annotations

import pytest

from etops.ir import Teir

__all__ = [
    "backend_available",
    "backend_param",
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
