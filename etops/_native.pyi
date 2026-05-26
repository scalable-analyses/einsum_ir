"""Type stubs for the `libteir` C++ extension exposed as ``etops._native``.

The runtime module is produced by pybind11 from
``teir/bindings/python/module.cpp``. These stubs describe the public surface
the rest of ``etops`` depends on; runtime details (refcounting, GIL
handling, etc.) are intentionally omitted.
"""

from typing import Any

__version__: str

class NativeValidationError(Exception): ...
class NativeLoweringError(Exception): ...
class NativeRuntimeError(Exception): ...

class Operation:
    def execute(self, base_addresses: list[int]) -> None: ...

def compile_teir(teir_dict: dict[str, Any], backend: str) -> Operation: ...

class _Config:
    threading_backend: str
    blas_vendor: str
    libxsmm_available: bool
    blas_available: bool

    def num_threads_available(self) -> int: ...
    def has_primitive(self, backend: str, operation: str) -> bool: ...

config: _Config
