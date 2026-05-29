#!/usr/bin/env bash
# Install the system toolchain etops needs to build and test.
# Runs identically locally and in CI. Supports the Fedora fleet, macOS
# (Homebrew), and Debian/Ubuntu (local dev boxes).
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  brew update >/dev/null
  brew install cmake ninja openblas libomp pkg-config
  exit 0
fi

if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y \
    git make curl cmake ninja-build gcc-c++ pkgconf-pkg-config \
    openblas-devel libomp-devel clang-tools-extra \
    libasan libubsan
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential curl cmake ninja-build pkg-config \
    libopenblas-dev libomp-dev clang-tools clang-format clang-tidy
else
  echo "ci/install-deps.sh: no supported package manager (need dnf, apt-get, or brew)" >&2
  exit 1
fi
