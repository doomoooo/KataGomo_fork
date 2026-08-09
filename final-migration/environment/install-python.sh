#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

system_python="${KATAGO_SYSTEM_PYTHON:-/usr/bin/python3}"
[[ -x "${system_python}" ]] || die "system Python is missing: ${system_python}"
mkdir -p -- "${KATAGO_ENV_ROOT}/state"
assert_safe_managed_path "${KATAGO_FINAL_VENV}"

if [[ ! -x "${KATAGO_FINAL_VENV}/bin/python" ]]; then
  log "creating Python environment: ${KATAGO_FINAL_VENV}"
  "${system_python}" -m venv "${KATAGO_FINAL_VENV}"
fi
activate_venv

requirements="${SCRIPT_DIR}/python-bootstrap-requirements.txt"
wheelhouse="${KATAGO_LOCAL_ARCHIVE}/wheels"

if [[ -d "${wheelhouse}" ]]; then
  log "seeding Python bootstrap tools and packages from the local wheel archive"
  python -m pip install --no-index --find-links "${wheelhouse}" \
    --upgrade pip setuptools wheel || warn "local wheel archive did not contain every bootstrap tool"
  python -m pip install --no-index --find-links "${wheelhouse}" \
    --upgrade --upgrade-strategy eager --requirement "${requirements}" \
    || warn "local wheel archive did not contain the complete Python stack"
fi

log "resolving current Python bootstrap releases from domestic mirror: ${KATAGO_PYPI_MIRROR}"
if ! python -m pip install \
  --index-url "${KATAGO_PYPI_MIRROR}" \
  --upgrade --upgrade-strategy eager \
  pip setuptools wheel \
  --requirement "${requirements}"; then
  if [[ "${KATAGO_ALLOW_STALE_BINARY:-0}" != "1" ]]; then
    die "could not resolve current binary packages from the domestic mirror; configure a proxy, or set KATAGO_ALLOW_STALE_BINARY=1 explicitly after populating archive/wheels"
  fi
  warn "using locally cached binary packages because KATAGO_ALLOW_STALE_BINARY=1"
fi

log "current Python bootstrap environment complete; source components are installed next"
