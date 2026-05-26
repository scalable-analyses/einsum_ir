"""etops — Python frontend for the Tiled Execution Intermediate Representation.

The package is structured around an immutable IR (:class:`Teir`), a
programmatic builder (:class:`TeirBuilder`) that exposes a documented
mutator surface (`add_*`, `remove_*`, `replace_*`, `set_*`, `append_*`),
an einsum emitter (:func:`etops.emit.einsum`), a textual serialization
(:mod:`etops.textir`), the :func:`etops.optimize` driver that runs the
backend's default pass pipeline, and a compile / execute facade backed by
the C++ runtime via :mod:`etops._native` (the ``tpp`` and ``blas``
backends).

The top-level surface is intentionally narrow. Advanced users opt into
submodules explicitly:

>>> import etops
>>> import etops.transforms as t
>>> import etops.passes as p
"""

from __future__ import annotations

import logging

# Silence framework logging by default. The application is responsible for
# configuring handlers.
logging.getLogger("etops").addHandler(logging.NullHandler())

try:
    from etops._version import __version__
except ImportError:  # pragma: no cover - present after a build
    __version__ = "0.0.0+unknown"

from etops.diag import (  # noqa: E402  (re-exports follow the version block)
    TeirEmissionError,
    TeirError,
    TeirLoweringError,
    TeirPassError,
    TeirRuntimeError,
    TeirValidationError,
)
from etops.ir import (  # noqa: E402
    Axis,
    DataType,
    First,
    Guard,
    InvocationNode,
    IterationNode,
    Last,
    Primitive,
    Schedule,
    Teir,
    TeirBuilder,
    Tensor,
    validate,
)
from etops.optimization import (  # noqa: E402
    BackendStrategy,
    BlasStrategy,
    ISAExtension,
    Microarchitecture,
    OptimizationProfile,
    TppStrategy,
    blas_profile,
    tpp_profile,
)
from etops.runtime import (  # noqa: E402
    compile,
    default_profile,
    list_backends,
    optimize,
)
from etops.visualize import show  # noqa: E402

__all__ = [
    "Axis",
    "BackendStrategy",
    "BlasStrategy",
    "DataType",
    "First",
    "Guard",
    "ISAExtension",
    "InvocationNode",
    "IterationNode",
    "Last",
    "Microarchitecture",
    "OptimizationProfile",
    "Primitive",
    "Schedule",
    "Teir",
    "TeirBuilder",
    "TeirEmissionError",
    "TeirError",
    "TeirLoweringError",
    "TeirPassError",
    "TeirRuntimeError",
    "TeirValidationError",
    "Tensor",
    "TppStrategy",
    "__version__",
    "blas_profile",
    "compile",
    "default_profile",
    "list_backends",
    "optimize",
    "show",
    "tpp_profile",
    "validate",
]
