#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/cuda_completion_bench"}
BIN=${BIN:-"${BUILD_DIR}/cuda_completion_bench"}

if [[ ! -x "${BIN}" ]]; then
  echo "missing binary: ${BIN}" >&2
  echo "build it first, for example:" >&2
  echo "  cmake -S exp/cuda_completion_bench -B build/cuda_completion_bench -DCMAKE_CXX_COMPILER=${REPO_ROOT}/.local/toolchains/llvm-22.1.1/bin/clang++" >&2
  echo "  cmake --build build/cuda_completion_bench -j 8" >&2
  exit 1
fi

exec "${BIN}" "$@"
