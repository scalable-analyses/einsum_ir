# AGENTS.md — etops / teir

## Project

`etops` is the Python package for the Tiled Execution Intermediate
Representation (TEIR). It owns the IR data model, einsum and textual
emitters, transforms, passes, optimization profiles, and lowering
drivers. The NumPy reference backend lives in
`tests/oracles/numpy_oracle/` and is registered as the `numpy` backend
by `tests/conftest.py`; every production backend is property-tested
against it.

`libteir` is the C++20 runtime under `teir/`. It owns the IR mirror
types, schedule walker, threading abstraction (Apple Dispatch / OpenMP
/ sequential), and the TPP (libxsmm) and BLAS (cblas) primitive
lowerings. The pybind11 binding at `teir/bindings/python/` exposes
`etops._native`.

## Build

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -ve ".[test]"             # editable install + tests

cmake -S teir -B build/teir -DETOPS_ENABLE_TESTS=ON
cmake --build build/teir -j               # standalone C++ build

cmake -S teir -B build/sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DETOPS_ENABLE_TESTS=ON \
  -DETOPS_ENABLE_SANITIZERS=ON \
  -DETOPS_THREADING_BACKEND=SEQUENTIAL
cmake --build build/sanitize -j           # ASan+UBSan correctness gate

nix develop                               # reproducible toolchain
```

CMake knobs: `ETOPS_THREADING_BACKEND={AUTO|DISPATCH|OPENMP|SEQUENTIAL}`,
`ETOPS_USE_SYSTEM_LIBXSMM=ON|OFF`, `LIBXSMM_ROOT=<path>`,
`ETOPS_ENABLE_SANITIZERS=ON|OFF` (shorthand for ASan+UBSan),
`ETOPS_SANITIZERS=<comma-list>` (e.g. `address,undefined`),
`ETOPS_ENABLE_TESTS=ON|OFF`, `ETOPS_BUILD_PYTHON=ON|OFF`.

## Test

```bash
pytest tests/                                       # Hypothesis ci (200 examples)
pytest tests/ --hypothesis-profile=dev              # 50 examples, fast iteration
pytest -m property                                  # property-based tests only
pytest -m tpp                                       # libxsmm-required tests
pytest -m blas                                      # cblas-required tests

# CI selects the Hypothesis profile via env var; defaults to `ci` in conftest.
ETOPS_HYPOTHESIS_PROFILE=ci pytest tests/

ctest --test-dir build/teir --output-on-failure
ASAN_OPTIONS=abort_on_error=1:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build/sanitize --output-on-failure

python -m etops.textir cat     examples/spec/permutation/scalar.teir
python -m etops.textir validate examples/spec/permutation/scalar.teir

ruff check etops/ tests/ examples/
ruff format --check etops/ tests/ examples/
find teir \( -name '*.h' -o -name '*.cpp' \) \
  | grep -v '_deps/' \
  | xargs clang-format --dry-run -Werror
mypy etops/
```

IMPORTANT: never lower the Hypothesis `ci` example count. Use
`--hypothesis-profile=dev` for iteration.

## Code style

Defer to `.clang-format`, `.clang-tidy`, and `pyproject.toml`. The
project-specific rules below are not the standard defaults.

### C++20

- 2-space indent, K&R braces.
- `snake_case` functions / methods; `PascalCase` types;
  `SCREAMING_SNAKE_CASE` only for compile-time constants.
- Non-static private / protected members carry a trailing underscore
  (`compiled_`, not `m_compiled` or `_compiled`).
- Integer types:
  - `int64_t` for TEIR-semantic values: tensor extents, byte strides,
    byte offsets, element counts, iteration counters, byte address
    arithmetic.
  - `int32_t` for bounded interned indices: axis indices, node indices,
    tensor slot indices, ancestor depth.
  - `std::size_t` for STL container indexing.
  - Third-party APIs (libxsmm, dispatch, pybind11, cblas) use the type
    the API requires; cast at the boundary.
  - Fixed-width unsigned (`uint8_t`, `uint16_t`, …) is permitted for
    bitmask storage and `enum class : uintN_t` underlying types, where
    the unsigned wrap-around semantics are required (signed left-shift
    UB) and a tight width is justified by the value range.
- `[[nodiscard]]` on every builder, factory, and accessor whose return
  value the caller is expected to use.
- `#pragma once`, not include guards.
- YOU MUST throw typed `teir::Exception` subclasses
  (`ValidationException`, `LoweringException`, `RuntimeException`).
  Never bare `throw;` or `std::runtime_error`.
- American English in identifiers, messages, and comments.

### Python 3.10+

- `@dataclass(frozen=True)` for IR records and configuration objects.
- `IntEnum` (explicit integer values) for enum-like sets.
- Google-style docstrings; type annotations on every public signature.
- `logging.getLogger(__name__)` for diagnostics; never `print`.
- YOU MUST chain exceptions (`raise X(...) from y`).
- `mypy etops/` must stay clean.
- American English.

## IR invariants

1. **Immutability.** `Teir` and every record it owns are immutable;
   mutation goes through `TeirBuilder`. Passes call `teir.builder()`
   then `builder.finish()`.
2. **Byte units.** All TEIR-Axes strides and offsets are raw bytes.
   The einsum emitter multiplies element strides by `dtype.bytes` once
   at construction; the IR never stores element strides.
3. **Per-tensor maps.** Every axis stores strides and offsets per
   tensor. Missing entries imply zero.
4. **Shared id namespace.** Iteration and invocation nodes share one
   node id namespace. Children order is encoded by list position.
5. **Forest, not tree.** The schedule may have multiple roots; the
   order of `roots` is semantic.
6. **Conjunctive guards.** Guards are conjunctions of `first(node)` /
   `last(node)` terms, where each `node` is the id of an iteration-node
   ancestor of the guarded node. The term is true on that ancestor's
   first / last trip.
7. **Closed catalogs.** ``etops.ir.dtypes`` and ``etops.ir.primitives``
   carry a closed set of data types and primitive operations. Adding a
   new dtype or operation is a core-source edit. Backends register
   lowerings against the existing operation names at module init time
   (via `PYBIND11_MODULE` and test setup).
8. **Profile-driven passes.** Passes read tuning constants from
   `ctx.profile`. Hard-coded constants in a pass are a bug.
9. **Textual IR is canonical.** `parse(dump(teir)) == teir` for every
   well-formed `Teir`. JSON serialization is deferred.

## Common pitfalls

- **Stride confusion.** TEIR stores **byte** strides. The einsum
  emitter multiplies element strides by `dtype.bytes` once at
  construction. Do not multiply or divide a second time.
- **Mutating IR.** `Teir` records are `@dataclass(frozen=True)` — attribute
  rebinding raises `FrozenInstanceError`. Mapping fields are plain dicts;
  do not mutate them. Always go through `Teir.builder()`.
- **Builder finalization.** `builder.finish()` defaults to
  `validate=True`. Pass `validate=False` only when a pass needs a
  partial `Teir` mid-rewrite.

## Extending TEIR

See existing examples for the layout: `tests/fixtures/`,
`etops/transforms/`, `etops/passes/`, `teir/src/backends/`.

## TEIR specification

The TEIR specification lives in the TEIR chapter of the tnzr.compile
book: https://tnzr.org/compile/

## Do not edit without supervision

- `flake.lock` — update only via `nix flake update`.
- `etops/_version.py` — generated by `setuptools-scm` at build time.
- `teir/include/teir/config.h.in` — CMake configures it into
  `<build>/generated/teir/config.h`. Do not hand-edit the generated
  copy.
- `build/`, `dist/`, `*.egg-info/` — build artifacts.
