# AGENTS.md — einsum_ir

High-performance C++17 einsum (tensor contraction) execution engine with
Python bindings (`etops`). Supports TPP (libxsmm), BLAS, TBLIS, and scalar
backends with OpenMP/GCD parallelism.

---

## Build Commands

### Primary build (SCons)

```bash
# Minimal build (auto-detects libxsmm, BLAS, TBLIS)
scons

# Full build with explicit paths (most common in CI)
scons libtorch=/path/to/torch libxsmm=$(pwd)/libxsmm blas=$(pwd)/openblas -j4

# Debug build with AddressSanitizer + UBSan
scons mode=debug+san libtorch=${LIBTORCH_PATH} libxsmm=${LIBXSMM_PATH} blas=yes

# Build outputs:
#   build/tests               — unit test executable
#   build/bench_binary        — binary contraction benchmark
#   build/bench_unary         — unary permutation benchmark
#   build/bench_expression    — expression benchmark
#   build/bench_tree          — tree benchmark
#   build/bench_mlp           — MLP inference benchmark
```

Key SCons options: `mode=release|debug|release+san|debug+san`,
`parallel=auto|dispatch|omp|none`, `libxsmm=yes|no|<path>`,
`blas=yes|no|<path>`, `tblis=yes|no|<path>`, `libtorch=yes|no|<path>`.

### Standalone library build (CMake — `src/basic/` only)

```bash
mkdir build && cd build
cmake ../src/basic                          # auto-installs libxsmm
# or with manual path:
cmake ../src/basic -DLIBXSMM_ROOT=${LIBXSMM_PATH} -DEINSUM_IR_AUTO_INSTALL_LIBXSMM=OFF
make -j
cmake --install . --prefix $(pwd)/../install
```

### Python package (`etops`)

```bash
cd python
pip install -ve .
# If SCM version detection fails:
SETUPTOOLS_SCM_PRETEND_VERSION=$(python version_cli.py) pip install -ve .
```

### Prerequisites

```bash
# libxsmm
git clone https://github.com/libxsmm/libxsmm.git
cd libxsmm && make BLAS=0 -j8 && cd ..

# OpenBLAS
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.26.tar.gz
tar -xvf v0.3.26.tar.gz && cd OpenBLAS-0.3.26
make -j8 && make PREFIX=$(pwd)/../openblas install && cd ..

# Catch2 v2 single-header test framework
wget https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp
```

---

## Test Commands

Test framework: **Catch2 v2** (single-header `catch.hpp`, `TEST_CASE`/`REQUIRE`).

```bash
# Run all tests
./build/tests

# Run a single test by Catch2 tag
./build/tests "[bin_cont_dim_types]"
./build/tests "[einsum_exp]"
./build/tests "[filter_dim_ids]"
./build/tests "[contraction_optimizer]"

# Run a test by name substring
./build/tests "Derives the dimension types"

# List all test cases
./build/tests --list-tests

# Run tests with sanitizers (as in CI)
ASAN_OPTIONS="abort_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
LSAN_OPTIONS="exitcode=1" \
./build/tests
```

Test files are named `ClassName.test.cpp` (no libtorch required) or
`ClassName.test.torch.cpp` (requires libtorch). New tests must be added to
`src/SConscript` or `src/basic/SConscript` to be compiled.

---

## Code Style

### C++ Standard and Indentation

- **C++17** throughout.
- **2 spaces** for indentation — no tabs.
- Opening brace on the same line as the statement (K&R style); bodies of
  `if`/`for`/`while` always use braces even for single statements.
- Spaces inside parentheses for multi-argument calls: `foo( a, b, c )`.

### Naming Conventions

| Scope | Convention | Example |
|---|---|---|
| Local variable | `l_` prefix, `snake_case` | `l_dim_sizes`, `l_err` |
| Member variable | `m_` prefix, `snake_case` | `m_num_dims`, `m_compiled` |
| Input parameter | `i_` prefix, `snake_case` | `i_num_dims_left`, `i_dtype` |
| Output parameter | `o_` prefix, `snake_case` | `o_dim_types`, `o_strides` |
| In/out parameter | `io_` prefix, `snake_case` | `io_tensor_out` |
| Class | `PascalCase` | `BinaryContraction`, `EinsumExpression` |
| Method / free function | `snake_case` | `compile()`, `dim_types()` |
| Enum type | `snake_case` + `_t` suffix | `dim_t`, `data_t`, `err_t` |
| Enum value | `ALL_CAPS` | `FP32`, `MADD`, `SUCCESS`, `UNDEFINED_DIM` |
| Preprocessor macro | `PP_` prefix | `PP_EINSUM_IR_HAS_LIBXSMM` |
| Feature flags | `EINSUM_IR_USE_*` / `EINSUM_IR_ENABLE_*` | `EINSUM_IR_ENABLE_TPP` |
| Namespace | all lowercase | `einsum_ir`, `einsum_ir::backend` |

### Types

- Use `int64_t` for all integer values (sizes, IDs, strides, loop bounds).
  **Never use bare `int`.**
- Use `uint64_t` for kernel dimension parameters (`m_m`, `m_n`, `m_lda`).
- Use `char *` for raw byte-level memory pointers; `void *` / `void const *`
  for generic tensor data across the API boundary.
- Use `std::map< int64_t, int64_t >` for dimension-id-to-size/stride maps.
- **Do not use `auto`** — always write out the explicit type.
- Prefer pointer-to-const for non-owning input data members:
  `std::map< int64_t, int64_t > const * m_dim_sizes = nullptr;`
- Initialize all member variables at the point of declaration:
  `int64_t m_num_dims = 0;`, `bool m_compiled = false;`.

### Enums and Namespaces

- Define enums inside namespaces, not as `enum class`:
  ```cpp
  namespace einsum_ir {
    enum data_t : int { UNDEFINED_DTYPE = 0, FP32 = 1, FP64 = 2 };
  }
  ```
- The `src/basic/` sub-library has its own parallel `einsum_ir::basic`
  namespace with its own `constants.h`.

### Include Guards and Headers

- Use traditional `#ifndef` guards (not `#pragma once`):
  ```cpp
  #ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION
  #define EINSUM_IR_BACKEND_BINARY_CONTRACTION
  // ...
  #endif
  ```
  Guard names follow `EINSUM_IR_<NAMESPACE>_<CLASS>`.
- Include order in `.cpp` files:
  1. Own header (`#include "ClassName.h"`)
  2. Standard library (`<cstdint>`, `<vector>`, `<map>`, …)
  3. Third-party (`<libxsmm.h>`, `"ATen/ATen.h"`, `"catch.hpp"`)
  4. Other project headers (relative paths with `../`)
- Use `"double quotes"` for project and third-party headers placed on the
  include path; `<angle brackets>` for system headers.

### Comments and Documentation

- Document all public methods and class members in headers using Doxygen:
  ```cpp
  /**
   * Derives the dimension types of tensor t2.
   *
   * @param i_num_dims_t0 number of t0 dimensions.
   * @param o_dim_types_t2 will be set to the dimension types of t2.
   **/
  static void dim_types( int64_t         i_num_dims_t0,
                         int64_t const * i_dim_ids_t0,
                         dim_t         * o_dim_types_t2 );
  ```
- Document member variables with `//!`:
  ```cpp
  //! number of dimensions of the left tensor
  int64_t m_num_dims_left = 0;
  ```
- Use `//` for inline implementation notes.

### Error Handling

- Return `err_t` enum codes from `compile()` / `compile_base()`:
  `SUCCESS`, `COMPILATION_FAILED`, `TENSOR_BLOCKING_FAILED`, etc.
- **Never throw exceptions** in the core C++ library. Check return codes at
  every call site and propagate errors upward:
  ```cpp
  einsum_ir::err_t l_err = l_cont.compile();
  if( l_err != einsum_ir::SUCCESS ) return l_err;
  ```
- Use `assert(...)` for internal invariants in implementation files.
- Guard nullable pointers before dereferencing:
  ```cpp
  if( m_dim_sizes_outer_out_aux != nullptr ) { ... }
  ```
- Python layer converts C++ error codes into `RuntimeError` exceptions.

---

## Project Structure

```
einsum_ir/
├── SConstruct                  # Root SCons build
├── src/
│   ├── constants.h             # Top-level enums (dim_t, data_t, err_t, …)
│   ├── tests.cpp               # Catch2 main()
│   ├── basic/                  # Standalone contraction library (also CMake)
│   │   ├── constants.h         # basic-layer enums
│   │   ├── binary/             # ContractionBackend*, ContractionOptimizer, …
│   │   └── unary/              # UnaryBackend*, UnaryOptimizer, …
│   ├── backend/                # High-level wrappers (BinaryContraction*, EinsumNode, …)
│   └── frontend/               # User APIs (EinsumExpression*, EinsumTree*, parsers)
├── python/                     # etops Python package (pybind11 / scikit-build-core)
│   └── src/
│       ├── TensorOperation.{h,cpp}
│       ├── bindings.cpp
│       └── etops/__init__.py
└── samples/                    # Benchmark scripts and example configs
```

---

## CI Notes

CI runs on Ubuntu, macOS, and Fedora (self-hosted). The debug+sanitizer build
(`mode=debug+san`) is the primary correctness gate. When adding new source
files, register them in `src/SConscript` (or `src/basic/SConscript`) so they
are picked up by both the SCons build and the CI.
