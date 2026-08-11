#!/usr/bin/env bash

set -Eeuo pipefail

ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MIGRATION_ROOT="$(cd -- "${ENV_SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${MIGRATION_ROOT}/.." && pwd -P)"

KATAGO_ENV_ROOT="${KATAGO_ENV_ROOT:-${REPO_ROOT}/.final-migration-env}"
KATAGO_LOCAL_ARCHIVE="${KATAGO_LOCAL_ARCHIVE:-${MIGRATION_ROOT}/archive}"
KATAGO_THIRD_PARTY_ROOT="${KATAGO_THIRD_PARTY_ROOT:-${KATAGO_ENV_ROOT}/third_party}"
KATAGO_FINAL_VENV="${KATAGO_FINAL_VENV:-${KATAGO_ENV_ROOT}/venv}"
KATAGO_PYPI_MIRROR="${KATAGO_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
KATAGO_CUDA_ROOT=""
KATAGO_CUDNN_ROOT=""

default_build_jobs() {
  local cpu_jobs available_bytes cgroup_max cgroup_current cgroup_available memory_jobs
  cpu_jobs="$(nproc)"
  available_bytes="$(awk '$1 == "MemAvailable:" { print $2 * 1024; exit }' /proc/meminfo)"
  cgroup_max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
  cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)"
  if [[ "${cgroup_max}" =~ ^[0-9]+$ && "${cgroup_current}" =~ ^[0-9]+$ ]]; then
    cgroup_available=$((cgroup_max - cgroup_current))
    (( cgroup_available < 0 )) && cgroup_available=0
    if [[ -z "${available_bytes}" ]] || (( cgroup_available < available_bytes )); then
      available_bytes="${cgroup_available}"
    fi
  fi
  if [[ "${available_bytes}" =~ ^[0-9]+$ ]]; then
    memory_jobs=$((available_bytes * 3 / 4 / (2 * 1024 * 1024 * 1024)))
    (( memory_jobs < 1 )) && memory_jobs=1
    (( memory_jobs < cpu_jobs )) && cpu_jobs="${memory_jobs}"
  fi
  (( cpu_jobs > 8 )) && cpu_jobs=8
  printf '%s\n' "${cpu_jobs}"
}

KATAGO_BUILD_JOBS="${KATAGO_BUILD_JOBS:-$(default_build_jobs)}"
KATAGO_RECORD_ROOT="${KATAGO_RECORD_ROOT:-${MIGRATION_ROOT}/records}"

export KATAGO_ENV_ROOT KATAGO_LOCAL_ARCHIVE KATAGO_THIRD_PARTY_ROOT
export KATAGO_FINAL_VENV KATAGO_PYPI_MIRROR KATAGO_BUILD_JOBS KATAGO_RECORD_ROOT
export KATAGO_CUDA_ROOT KATAGO_CUDNN_ROOT

log() {
  printf '[final-migration] %s\n' "$*"
}

warn() {
  printf '[final-migration] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[final-migration] ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

ensure_record_root() {
  mkdir -p -- "${KATAGO_RECORD_ROOT}"
}

activate_venv() {
  [[ -f "${KATAGO_FINAL_VENV}/bin/activate" ]] || die "Python venv missing: ${KATAGO_FINAL_VENV}; run setup.sh install"
  # shellcheck disable=SC1091
  source "${KATAGO_FINAL_VENV}/bin/activate"
}

activate_toolchain() {
  local purelib target link name
  [[ -x "${KATAGO_FINAL_VENV}/bin/python" ]] \
    || die "Python environment missing: ${KATAGO_FINAL_VENV}; run setup.sh install"
  purelib="$("${KATAGO_FINAL_VENV}/bin/python" -c \
    'import sysconfig; print(sysconfig.get_path("purelib"))')"
  KATAGO_CUDA_ROOT="${purelib}/nvidia/cu13"
  KATAGO_CUDNN_ROOT="${purelib}/nvidia/cudnn"
  export KATAGO_CUDA_ROOT KATAGO_CUDNN_ROOT
  [[ -x "${KATAGO_CUDA_ROOT}/bin/nvcc" ]] \
    || die "PyPI CUDA compiler missing: ${KATAGO_CUDA_ROOT}; run setup.sh install"
  [[ -r "${KATAGO_CUDNN_ROOT}/include/cudnn.h" ]] \
    || die "PyPI cuDNN headers missing: ${KATAGO_CUDNN_ROOT}; run setup.sh install"
  [[ -r "${KATAGO_CUDNN_ROOT}/lib/libcudnn.so.9" ]] \
    || die "PyPI cuDNN library missing: ${KATAGO_CUDNN_ROOT}; run setup.sh install"

  # NVIDIA's PyPI layout intentionally carries versioned DSOs in lib/. nvcc
  # and CMake use the conventional lib64/ and unversioned developer names, so
  # add a small symlink-only compatibility view in the installed wheel tree.
  if [[ ! -e "${KATAGO_CUDA_ROOT}/lib64" ]]; then
    ln -s lib "${KATAGO_CUDA_ROOT}/lib64"
  fi
  [[ -d "${KATAGO_CUDA_ROOT}/lib64" ]] \
    || die "invalid PyPI CUDA lib64 layout: ${KATAGO_CUDA_ROOT}/lib64"
  for name in libcudart libcublas libcublasLt libnvrtc libnvJitLink; do
    link="${KATAGO_CUDA_ROOT}/lib/${name}.so"
    [[ -e "${link}" ]] && continue
    target="$(find "${KATAGO_CUDA_ROOT}/lib" -maxdepth 1 -type f \
      -name "${name}.so.*" ! -name '*.alt.*' -printf '%f\n' | sort -V | tail -n 1)"
    [[ -n "${target}" ]] || die "PyPI CUDA library missing: ${name}"
    ln -s "${target}" "${link}"
  done
  link="${KATAGO_CUDNN_ROOT}/lib/libcudnn.so"
  if [[ ! -e "${link}" ]]; then
    ln -s libcudnn.so.9 "${link}"
  fi

  export CUDA_HOME="${KATAGO_CUDA_ROOT}"
  export CUDA_PATH="${KATAGO_CUDA_ROOT}"
  export CUDNN_ROOT="${KATAGO_CUDNN_ROOT}"
  export CUDA_VERSION=13.0
  export PATH="${KATAGO_CUDA_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${KATAGO_CUDNN_ROOT}/lib:${KATAGO_CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

github_fallback_warning() {
  warn "accessing GitHub for $1"
  warn "GitHub access may be affected by the network environment; configure HTTPS_PROXY/https_proxy if needed"
}

assert_safe_managed_path() {
  local path
  path="$(readlink -m -- "$1")"
  case "${path}" in
    "${KATAGO_ENV_ROOT}"|"${KATAGO_ENV_ROOT}"/*) ;;
    *) die "refusing to manage path outside KATAGO_ENV_ROOT: ${path}" ;;
  esac
}
