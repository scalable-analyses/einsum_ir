"""TEIR element data types.

A closed catalog keyed by canonical name. The C++ runtime agrees with
the Python side via the same canonical names; no enum sync is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etops.diag import TeirRuntimeError

__all__ = ["DataType", "all_dtypes", "get_dtype", "resolve_numpy_dtype"]


@dataclass(frozen=True)
class DataType:
    """Canonical specification of a TEIR element type."""

    name: str
    bits: int

    @property
    def bytes(self) -> int:
        """Storage width in bytes (rounded up)."""

        return (self.bits + 7) // 8


_CATALOG: tuple[DataType, ...] = (
    DataType(name="f32", bits=32),
    DataType(name="f64", bits=64),
)


_DTYPES: dict[str, DataType] = {dtype.name: dtype for dtype in _CATALOG}

_NUMPY_NAME: dict[str, str] = {"f32": "float32", "f64": "float64"}


def get_dtype(name: str) -> DataType:
    """Look up a `DataType` by canonical name.

    Raises:
        KeyError: If no `DataType` is registered for ``name``.
    """

    return _DTYPES[name]


def all_dtypes() -> tuple[DataType, ...]:
    """Return all registered `DataType`s in canonical order."""

    return _CATALOG


def resolve_numpy_dtype(name: str) -> np.dtype:
    """Return the NumPy dtype that mirrors a TEIR canonical dtype name."""

    np_name = _NUMPY_NAME.get(name)
    if np_name is None:
        raise TeirRuntimeError(
            f"no NumPy mapping for TEIR dtype {name!r}; known: {sorted(_NUMPY_NAME)}"
        )
    return np.dtype(np_name)
