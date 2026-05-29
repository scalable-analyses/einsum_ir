#!/usr/bin/env bash
# Configure, build, and test libteir (the C++ runtime).
#
# Knobs (env, with defaults):
#   ETOPS_BUILD_TYPE         Debug | Release                  (default: Release)
#   ETOPS_THREADING_BACKEND  AUTO|DISPATCH|OPENMP|SEQUENTIAL  (default: AUTO)
#   ETOPS_SANITIZERS         "address,undefined" | "undefined" | ""  (default: "")
#   ETOPS_ASAN_DETECT_LEAKS  0 | 1                            (default: 1)
#   ETOPS_CLANG_TIDY         0 | 1   run clang-tidy after the build (default: 0)
#   ETOPS_BUILD_DIR          build directory                  (default: build/teir)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

build_type="${ETOPS_BUILD_TYPE:-Release}"
threading="${ETOPS_THREADING_BACKEND:-AUTO}"
sanitizers="${ETOPS_SANITIZERS:-}"
build_dir="${ETOPS_BUILD_DIR:-build/teir}"

cmake -S teir -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DETOPS_ENABLE_TESTS=ON \
  -DETOPS_BUILD_PYTHON=OFF \
  -DETOPS_THREADING_BACKEND="${threading}" \
  -DETOPS_SANITIZERS="${sanitizers}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "${build_dir}" -j

if [[ -n "${sanitizers}" ]]; then
  export ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=${ETOPS_ASAN_DETECT_LEAKS:-1}"
  export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"
  export LSAN_OPTIONS="exitcode=1"
fi

ctest --test-dir "${build_dir}" --output-on-failure

# clang-tidy reuses this build's compile_commands.json (built once). Policy
# (incl. warnings-as-errors) lives in .clang-tidy, so local == CI.
if [[ "${ETOPS_CLANG_TIDY:-0}" == "1" ]]; then
  # bindings/ need pybind11 (a BUILD_PYTHON=ON build); skip them in this
  # BUILD_PYTHON=OFF lint build so clang-tidy parses cleanly.
  find teir -name '*.cpp' | grep -v '_deps/' | grep -v '/bindings/' \
    | xargs clang-tidy -p "${build_dir}"
fi
