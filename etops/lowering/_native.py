"""Driver for the libteir-backed BLAS and TPP backends.

`NativeOperation(teir, backend=...)` translates an `etops.Teir` to the
dict form ``etops._native`` consumes and asks the C++ runtime to compile
it under a named backend. The resulting object re-raises C++ exceptions
as the typed Python equivalents.
"""

from __future__ import annotations

import numpy as np

from etops.diag import TeirLoweringError, TeirRuntimeError, TeirValidationError
from etops.ir import Teir
from etops.ir.dtypes import resolve_numpy_dtype
from etops.lowering._marshal import teir_to_dict

__all__ = ["NativeOperation", "is_available"]


class NativeOperation:
    """Compiled `Teir` executable by the C++ runtime."""

    def __init__(self, teir: Teir, *, backend: str) -> None:
        try:
            from etops import _native
        except ImportError as exc:  # pragma: no cover
            raise TeirLoweringError(
                "etops._native is not available in this build"
            ) from exc
        if not _native.config.has_primitive(backend, "Contraction"):
            raise TeirLoweringError(
                f"{backend} primitive lowerings are not compiled into this build"
            )
        self._teir = teir
        self._backend = backend
        self._native = _native
        self._expected_dtypes = [
            resolve_numpy_dtype(teir.tensors[tid].dtype.name) for tid in teir.tensor_ids
        ]
        try:
            self._op = _native.compile_teir(teir_to_dict(teir), backend)
        except _native.NativeLoweringError as exc:
            raise TeirLoweringError(str(exc)) from exc
        except _native.NativeValidationError as exc:
            raise TeirValidationError(str(exc)) from exc
        except _native.NativeRuntimeError as exc:
            raise TeirLoweringError(str(exc)) from exc

    @property
    def teir(self) -> Teir:
        return self._teir

    @property
    def backend(self) -> str:
        return self._backend

    def execute(self, *tensors: np.ndarray) -> None:
        """Execute the schedule against ``tensors`` in IR declaration order."""

        if len(tensors) != len(self._teir.tensors):
            msg = (
                f"NativeOperation.execute: expected {len(self._teir.tensors)} tensors;"
                f" got {len(tensors)}"
            )
            raise TeirRuntimeError(msg)
        bases: list[int] = []
        for tid, arr, expected_dtype in zip(
            self._teir.tensor_ids, tensors, self._expected_dtypes, strict=True
        ):
            if arr.dtype != expected_dtype:
                msg = (
                    f"tensor {tid!r}: dtype {arr.dtype} does not match"
                    f" IR-declared dtype {expected_dtype}"
                )
                raise TeirRuntimeError(msg)
            bases.append(int(arr.ctypes.data))
        try:
            self._op.execute(bases)
        except self._native.NativeLoweringError as exc:
            raise TeirLoweringError(str(exc)) from exc
        except self._native.NativeValidationError as exc:
            raise TeirValidationError(str(exc)) from exc
        except self._native.NativeRuntimeError as exc:
            raise TeirRuntimeError(str(exc)) from exc


def is_available(backend: str) -> bool:
    """Return True if ``backend`` is registered in the current build."""

    try:
        from etops import _native
    except ImportError:
        return False
    return _native.config.has_primitive(backend, "Contraction")
