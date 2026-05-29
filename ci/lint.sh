#!/usr/bin/env bash
# Build-free lint: ruff, mypy, clang-format, and textir example validation.
# Assumes ruff, mypy, numpy, and clang-format are on PATH (the lint workflow
# installs them; locally, use your dev environment). textir imports cleanly
# without the native extension, so this needs no build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ruff check etops/ tests/ examples/
ruff format --check etops/ tests/ examples/
mypy etops/

find teir \( -name '*.h' -o -name '*.cpp' \) | grep -v '_deps/' \
  | xargs clang-format --dry-run -Werror

find examples -name '*.teir' -print0 \
  | xargs -0 -t -n1 python -m etops.textir validate
