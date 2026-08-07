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
detected_jobs="$(nproc)"
if (( detected_jobs > 8 )); then
  detected_jobs=8
fi
memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || printf 0)"
memory_jobs=$((memory_kib / 4194304))
if (( memory_jobs < 1 )); then
  memory_jobs=1
fi
if (( detected_jobs > memory_jobs )); then
  detected_jobs="${memory_jobs}"
fi
KATAGO_BUILD_JOBS="${KATAGO_BUILD_JOBS:-${detected_jobs}}"
KATAGO_RECORD_ROOT="${KATAGO_RECORD_ROOT:-${MIGRATION_ROOT}/records}"

export KATAGO_ENV_ROOT KATAGO_LOCAL_ARCHIVE KATAGO_THIRD_PARTY_ROOT
export KATAGO_FINAL_VENV KATAGO_PYPI_MIRROR KATAGO_BUILD_JOBS KATAGO_RECORD_ROOT

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
