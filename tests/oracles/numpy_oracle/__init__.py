"""Pure-Python NumPy reference backend.

The NumPy backend is the correctness oracle for every other backend. It is
implemented in Python — independent of the C++ runtime.
"""

from __future__ import annotations

from tests.oracles.numpy_oracle._pipeline import (
    NumpyOptimizationProfile,
    numpy_pipeline,
)
from tests.oracles.numpy_oracle.runtime import NumpyOperation, compile_to_numpy

__all__ = [
    "NumpyOperation",
    "NumpyOptimizationProfile",
    "compile_to_numpy",
    "numpy_pipeline",
]
