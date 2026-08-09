#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_command cmake
require_command ninja
ensure_record_root
record="${KATAGO_RECORD_ROOT}/build-matrix-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "${record}") 2>&1

build_root="${KATAGO_BUILD_ROOT:-${KATAGO_ENV_ROOT}/katago-builds}"
assert_safe_managed_path "${build_root}"
mkdir -p -- "${build_root}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

build_cuda() {
  local build_dir="${build_root}/cuda"
  local system_multiarch system_zlib
  system_multiarch="$(gcc -print-multiarch)"
  system_zlib="/usr/lib/${system_multiarch}/libz.so"
  [[ -r "${system_zlib}" ]] || die "Ubuntu system zlib development library missing: ${system_zlib}"
  local -a cmake_args=(
    -S "${REPO_ROOT}/cpp"
    -B "${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DUSE_BACKEND=CUDA
    -DBUILD_DISTRIBUTED=0
    -DUSE_TCMALLOC=0
    "-DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc"
    -DCUDNN_ROOT_DIR=/usr
    -DZLIB_INCLUDE_DIR=/usr/include
    "-DZLIB_LIBRARY=${system_zlib}"
    "-DZLIB_LIBRARY_RELEASE=${system_zlib}"
  )

  log "configuring CUDA backend in ${build_dir}"
  cmake "${cmake_args[@]}"
  log "building CUDA backend with ${KATAGO_BUILD_JOBS} jobs"
  cmake --build "${build_dir}" --parallel "${KATAGO_BUILD_JOBS}"
  [[ -x "${build_dir}/katago" ]] || die "CUDA build did not produce katago"
  "${build_dir}/katago" version
}

build_cuda
log "KataGo CUDA backend build completed"
printf 'record=%s\n' "${record}"
