#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
# shellcheck source=python-runtime.lock.sh
source "${SCRIPT_DIR}/python-runtime.lock.sh"

runtime_only=0
case "${1:-}" in
  "") ;;
  --runtime-only) runtime_only=1 ;;
  *) die "usage: install-python.sh [--runtime-only]" ;;
esac

for command_name in curl readlink sha256sum tar; do
  require_command "${command_name}"
done
mkdir -p -- "${KATAGO_ENV_ROOT}/state" "${KATAGO_ENV_ROOT}/downloads"
assert_safe_managed_path "${KATAGO_FINAL_VENV}"
python_root="${KATAGO_ENV_ROOT}/python"
assert_safe_managed_path "${python_root}"

python_archive="${KATAGO_PYTHON_ARCHIVE:-}"
if [[ -z "${python_archive}" ]]; then
  for candidate in \
    "${KATAGO_LOCAL_ARCHIVE}/python/${KATAGO_PYTHON_RUNTIME_ARCHIVE}" \
    "${KATAGO_LOCAL_ARCHIVE}/${KATAGO_PYTHON_RUNTIME_ARCHIVE}" \
    "${KATAGO_ENV_ROOT}/downloads/${KATAGO_PYTHON_RUNTIME_ARCHIVE}"; do
    if [[ -r "${candidate}" ]]; then
      python_archive="${candidate}"
      break
    fi
  done
fi
if [[ -z "${python_archive}" ]]; then
  python_archive="${KATAGO_ENV_ROOT}/downloads/${KATAGO_PYTHON_RUNTIME_ARCHIVE}"
  github_fallback_warning "the pinned Python ${KATAGO_PYTHON_RUNTIME_VERSION} runtime"
  curl --fail --location --retry 3 \
    --output "${python_archive}.partial" "${KATAGO_PYTHON_RUNTIME_URL}"
  mv -- "${python_archive}.partial" "${python_archive}"
fi
python_archive="$(readlink -e -- "${python_archive}")"
[[ -r "${python_archive}" ]] || die "pinned Python archive is missing"
actual_python_sha="$(sha256sum "${python_archive}" | awk '{print $1}')"
[[ "${actual_python_sha}" == "${KATAGO_PYTHON_RUNTIME_SHA256}" ]] \
  || die "pinned Python archive SHA-256 mismatch: ${actual_python_sha}"

runtime_marker="${KATAGO_ENV_ROOT}/state/python-runtime.sha256"
if [[ ! -x "${python_root}/bin/python3" ]] || \
   [[ "$(cat "${runtime_marker}" 2>/dev/null || true)" != "${KATAGO_PYTHON_RUNTIME_SHA256}" ]]; then
  if [[ -e "${python_root}" ]]; then
    log "replacing an unverified Python runtime below the managed environment"
    find "${python_root}" -mindepth 1 -delete
    rmdir -- "${python_root}"
  fi
  log "extracting pinned Python ${KATAGO_PYTHON_RUNTIME_VERSION} into ${KATAGO_ENV_ROOT}"
  tar --extract --gzip --file "${python_archive}" --directory "${KATAGO_ENV_ROOT}"
  [[ -x "${python_root}/bin/python3" ]] \
    || die "pinned Python archive did not produce python/bin/python3"
  printf '%s\n' "${KATAGO_PYTHON_RUNTIME_SHA256}" > "${runtime_marker}"
fi
actual_python_version="$("${python_root}/bin/python3" -c 'import platform; print(platform.python_version())')"
[[ "${actual_python_version}" == "${KATAGO_PYTHON_RUNTIME_VERSION}" ]] \
  || die "Python runtime version ${actual_python_version} != ${KATAGO_PYTHON_RUNTIME_VERSION}"

venv_marker="${KATAGO_ENV_ROOT}/state/venv-python-runtime.sha256"
if [[ ! -x "${KATAGO_FINAL_VENV}/bin/python" ]] || \
   [[ "$(cat "${venv_marker}" 2>/dev/null || true)" != "${KATAGO_PYTHON_RUNTIME_SHA256}" ]]; then
  if [[ -e "${KATAGO_FINAL_VENV}" ]]; then
    log "replacing a partial or system-Python virtual environment"
    find "${KATAGO_FINAL_VENV}" -mindepth 1 -delete
    rmdir -- "${KATAGO_FINAL_VENV}"
  fi
  log "creating Python ${KATAGO_PYTHON_RUNTIME_VERSION} environment: ${KATAGO_FINAL_VENV}"
  "${python_root}/bin/python3" -m venv --copies "${KATAGO_FINAL_VENV}"
  printf '%s\n' "${KATAGO_PYTHON_RUNTIME_SHA256}" > "${venv_marker}"
fi
activate_venv
if (( runtime_only == 1 )); then
  python --version
  python -m ensurepip --version
  log "pinned Python runtime and virtual environment are ready"
  exit 0
fi

build_requirements="${MIGRATION_ROOT}/autotune/python-build-requirements.txt"
binary_requirements="${MIGRATION_ROOT}/autotune/python-binary-requirements.txt"
wheelhouse="${KATAGO_LOCAL_ARCHIVE}/wheels"
python_packages_ready=0
requirements_marker="${KATAGO_ENV_ROOT}/state/python-requirements.sha256"
requirements_identity="$({ sha256sum "${build_requirements}"; sha256sum "${binary_requirements}"; } | sha256sum | awk '{print $1}')"

if [[ "$(cat "${requirements_marker}" 2>/dev/null || true)" == "${requirements_identity}" ]] && \
   python - "${build_requirements}" "${binary_requirements}" <<'PY'
import importlib.metadata
import pathlib
import sys

for requirements in map(pathlib.Path, sys.argv[1:]):
    for raw in requirements.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        operator = "===" if "===" in line else "=="
        if operator not in line:
            raise SystemExit(1)
        name, expected = line.split(operator, 1)
        if importlib.metadata.version(name) != expected:
            raise SystemExit(1)
PY
then
  python_packages_ready=1
  log "reusing the already verified Python dependency set"
fi

if (( python_packages_ready == 0 )) && [[ -d "${wheelhouse}" ]]; then
  log "installing the fixed Python dependency set from the local wheel archive"
  if python -m pip install --no-index --find-links "${wheelhouse}" \
    --upgrade --no-deps --requirement "${build_requirements}" \
    --requirement "${binary_requirements}"; then
    python_packages_ready=1
  else
    warn "local wheel archive did not contain the complete fixed Python stack"
  fi
fi

if (( python_packages_ready == 0 )); then
  log "resolving the fixed Python dependency set from domestic mirror: ${KATAGO_PYPI_MIRROR}"
  if ! python -m pip install \
    --index-url "${KATAGO_PYPI_MIRROR}" \
    --upgrade --no-deps \
    --requirement "${build_requirements}" \
    --requirement "${binary_requirements}"; then
    die "could not resolve the fixed Python packages from the domestic mirror; populate archive/wheels or configure a proxy"
  fi
fi

printf '%s\n' "${requirements_identity}" > "${requirements_marker}"

log "pinned Python environment complete; source components are installed next"
