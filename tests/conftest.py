"""Top-level pytest configuration for the etops test suite.

Hypothesis profile selection:

- ``dev``     — 50 examples, fast (per-step verification).
- ``ci``      — 200 examples, the default (phase-boundary verification).
- ``nightly`` — 1000 examples, manual or scheduled runs only.

Override with ``pytest --hypothesis-profile=<name>``. The default profile is
``ci`` unless overridden by ``ETOPS_HYPOTHESIS_PROFILE``.

The NumPy reference backend lives in ``tests/oracles/numpy_oracle/`` and
is registered as the ``numpy`` backend for the duration of the test
session; it is the correctness oracle that every production backend is
compared against.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, Verbosity, settings

from etops.runtime import Backend, list_backends, register_backend
from tests.oracles.numpy_oracle import (
    NumpyOptimizationProfile,
    compile_to_numpy,
    numpy_pipeline,
)

for _name, _max in (("dev", 50), ("ci", 200), ("nightly", 1000)):
    settings.register_profile(
        _name,
        max_examples=_max,
        deadline=None,
        verbosity=Verbosity.verbose if _name == "nightly" else Verbosity.normal,
        suppress_health_check=[HealthCheck.too_slow],
    )

settings.load_profile(os.environ.get("ETOPS_HYPOTHESIS_PROFILE", "ci"))

if "numpy" not in list_backends():
    register_backend(
        Backend(
            name="numpy",
            compile_fn=compile_to_numpy,
            pipeline_factory=numpy_pipeline,
            profile_factory=NumpyOptimizationProfile,
        )
    )
