etops — Tiled Execution IR for Tensor Operations
=================================================

``etops`` is a Python package and a C++ runtime that compile and execute
tensor operations described by the *Tiled Execution Intermediate
Representation* (TEIR). The package owns the IR, the einsum and textual
emitters, the transformation library, and the pass-based optimizer. The
companion library ``libteir`` (under ``teir/``) provides the production
C++ runtime with TPP (libxsmm) and BLAS lowerings.

Quickstart
----------

.. code-block:: python

   import numpy as np
   import etops
   from etops.emit import einsum

   dim_sizes = dict(d=4, b=64, a=128, c=32)
   teir = einsum("dba,dac->dbc", dim_sizes=dim_sizes)
   print(etops.show(teir))
   teir = etops.optimize(teir, backend="tpp")
   print(etops.show(teir))
   op   = etops.compile(teir, backend="tpp")

   rng = np.random.default_rng(0)
   in0 = rng.standard_normal((4, 64, 128), dtype=np.float32)
   in1 = rng.standard_normal((4, 128, 32), dtype=np.float32)
   out = np.zeros((4, 64, 32), dtype=np.float32)
   op.execute(in0, in1, out)

   np.testing.assert_allclose(
       out, np.einsum("dba,dac->dbc", in0, in1), atol=1e-4
   )

Installation
------------

The recommended path uses the Nix-pinned dev shell::

   nix develop
   uv venv .venv && source .venv/bin/activate
   uv pip install -e ".[test]"

Plain ``pip`` works on any platform with Python 3.10+, CMake 3.20+, and a
C++20 compiler::

   pip install -e ".[test]"

``libxsmm`` is auto-fetched at build time via CMake ``FetchContent``;
``LIBXSMM_ROOT=<path>`` overrides this. A ``cblas``-compatible BLAS
implementation is found via ``find_package(BLAS)``.

Testing
-------

::

   pytest tests/                                       # full Python suite
   cmake -S teir -B build/teir -DETOPS_ENABLE_TESTS=ON
   cmake --build build/teir -j
   ctest --test-dir build/teir --output-on-failure     # C++ Catch2 tests

   # Sanitizer build
   cmake -S teir -B build/sanitize -DCMAKE_BUILD_TYPE=Debug \
     -DETOPS_ENABLE_TESTS=ON -DETOPS_ENABLE_SANITIZERS=ON \
     -DETOPS_THREADING_BACKEND=SEQUENTIAL

See ``AGENTS.md`` for the operating manual.

License
-------

``MIT AND BSD-3-Clause``. ``etops`` itself is MIT-licensed; the bundled
libxsmm runtime is BSD-3-Clause. See ``LICENSE`` for the full text.
