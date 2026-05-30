#!/usr/bin/env bash
# Build + test etops under a chosen Python.
#
# Knobs (env):
#   ETOPS_PYTHON              Python version to provision via uv (e.g. "3.12").
#                             If unset, uses the already-active environment.
#   ETOPS_HYPOTHESIS_PROFILE  dev|ci|nightly  (read by tests/conftest.py; default ci)
#   ETOPS_RUN_SLOW            0|1  also run the slow TCCG corpus (default: 0)
#   CMAKE_ARGS                forwarded to the build by scikit-build-core
#                             (e.g. sanitizer / threading flags)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Bootstrap uv if absent (some runners / fresh dev boxes).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [[ -n "${ETOPS_PYTHON:-}" ]]; then
  uv python install "${ETOPS_PYTHON}"
  uv venv --python "${ETOPS_PYTHON}" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

uv pip install -ve ".[test]"

# Resolve the ASan runtime the built extension actually links (compiler- and
# distro-agnostic: GCC libasan or clang libclang_rt.asan). `find_spec` locates the
# .so without importing it -- importing is what ASan-aborts. Linux-only.
resolve_asan_runtime() {
  local ext
  ext=$(python -c 'import importlib.util as u; s = u.find_spec("etops._native"); print(getattr(s, "origin", "") or "")' 2>/dev/null) || return 1
  [[ -n "$ext" && -e "$ext" ]] || return 1
  ldd "$ext" 2>/dev/null | awk '/libasan|libclang_rt\.asan/ { print $3; exit }'
}

# Build the pytest argument list. ETOPS_RUN_SLOW=1 also runs the slow TCCG
# corpus; normal runs honor pyproject's `-m 'not slow'`.
pytest_args=(tests/)
if [[ "${ETOPS_RUN_SLOW:-0}" == "1" ]]; then
  pytest_args+=(-m "slow or not slow")
fi

if [[ -n "${ETOPS_ASAN_PRELOAD:-}" || -n "${ETOPS_LD_PRELOAD:-}" ]]; then
  asan_rt="${ETOPS_LD_PRELOAD:-$(resolve_asan_runtime)}"
  if [[ -z "$asan_rt" || ! -e "$asan_rt" ]]; then
    echo "ERROR: ASan preload requested but the sanitizer runtime could not be resolved" >&2
    exit 1
  fi
  echo "ASan preload: $asan_rt"
  export LD_PRELOAD="$asan_rt"
  setarch "$(uname -m)" -R python -c "import etops._native"
  setarch "$(uname -m)" -R pytest -s "${pytest_args[@]}"
else
  pytest "${pytest_args[@]}"
fi
