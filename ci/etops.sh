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

# Build the pytest argument list. ETOPS_RUN_SLOW=1 also runs the slow TCCG
# corpus; normal runs honor pyproject's `-m 'not slow'`.
pytest_args=(tests/)
if [[ "${ETOPS_RUN_SLOW:-0}" == "1" ]]; then
  pytest_args+=(-m "slow or not slow")
fi

pytest "${pytest_args[@]}"
