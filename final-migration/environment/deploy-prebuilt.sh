#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

[[ $# -eq 1 ]] || die "usage: deploy-prebuilt.sh BUNDLE"
bundle="$(readlink -e -- "$1")"
[[ -d "${bundle}" ]] || die "distribution bundle is not a directory: $1"
[[ -r "${bundle}/SHA256SUMS" ]] || die "distribution checksum manifest missing"
[[ -r "${bundle}/metadata/source-build-manifest.tsv" ]] || die "source build manifest missing"
[[ -r "${bundle}/metadata/python-binary-resolved.txt" ]] || die "resolved Python manifest missing"
[[ -r "${bundle}/metadata/system-requirements.env" ]] || die "system requirement manifest missing"
[[ -x "${bundle}/bin/katago" ]] || die "KataGo CUDA binary missing"

log "verifying every prebuilt distribution artifact"
(cd -- "${bundle}" && sha256sum --check SHA256SUMS)

# Keep the build toolkit's major.minor ABI instead of moving a compiled binary
# to a future CUDA major release. Parse rather than source bundle content.
while IFS='=' read -r requirement_name requirement_value; do
  [[ "${requirement_value}" =~ ^[a-zA-Z0-9.+:-]+$ ]] \
    || die "invalid system package value for ${requirement_name}"
  case "${requirement_name}" in
    KATAGO_CUDA_TOOLKIT_PACKAGE) KATAGO_CUDA_TOOLKIT_PACKAGE="${requirement_value}" ;;
    KATAGO_CUDNN_PACKAGE) KATAGO_CUDNN_PACKAGE="${requirement_value}" ;;
    *) die "unknown system requirement: ${requirement_name}" ;;
  esac
done < "${bundle}/metadata/system-requirements.env"
[[ -n "${KATAGO_CUDA_TOOLKIT_PACKAGE:-}" && -n "${KATAGO_CUDNN_PACKAGE:-}" ]] \
  || die "incomplete system requirement manifest"
export KATAGO_CUDA_TOOLKIT_PACKAGE KATAGO_CUDNN_PACKAGE

# Permit the bundle to seed .deb/wheel installation if those directories are
# present. Missing system packages still come from Ubuntu/NVIDIA repositories.
export KATAGO_LOCAL_ARCHIVE="${bundle}"
if [[ "${KATAGO_SKIP_SYSTEM_BOOTSTRAP:-0}" == "1" ]]; then
  log "using the pre-provisioned system CUDA stack; package bootstrap explicitly skipped"
  for system_package in "${KATAGO_CUDA_TOOLKIT_PACKAGE}" "${KATAGO_CUDNN_PACKAGE}"; do
    dpkg-query -W -f='${Status}' "${system_package}" 2>/dev/null \
      | grep -q 'install ok installed' \
      || die "required pre-provisioned package is missing: ${system_package}"
  done
else
  "${SCRIPT_DIR}/bootstrap-ubuntu.sh"
fi

mkdir -p -- "${KATAGO_ENV_ROOT}"
assert_safe_managed_path "${KATAGO_FINAL_VENV}"
if [[ ! -x "${KATAGO_FINAL_VENV}/bin/python" ]]; then
  system_python="${KATAGO_SYSTEM_PYTHON:-/usr/bin/python3}"
  [[ -x "${system_python}" ]] || die "system Python is missing: ${system_python}"
  "${system_python}" -m venv "${KATAGO_FINAL_VENV}"
fi
activate_venv

log "installing exact binary dependency closure without network access"
python -m pip install \
  --no-index --no-deps --find-links "${bundle}/wheels" \
  --requirement "${bundle}/metadata/python-binary-resolved.txt"

while IFS=$'\t' read -r name distribution version commit builder wheel wheel_hash; do
  [[ "${name}" != "name" ]] || continue
  [[ "${wheel}" != "-" ]] || continue
  [[ -r "${bundle}/wheels/${wheel}" ]] || die "locally built wheel missing: ${wheel}"
  actual_hash="$(sha256sum "${bundle}/wheels/${wheel}" | awk '{print $1}')"
  [[ "${actual_hash}" == "${wheel_hash}" ]] || die "source wheel hash mismatch: ${wheel}"
  python -m pip install --no-index --no-deps --force-reinstall "${bundle}/wheels/${wheel}"
done < "${bundle}/metadata/source-build-manifest.tsv"

install_root="${KATAGO_ENV_ROOT}/installed"
assert_safe_managed_path "${install_root}"
mkdir -p -- "${install_root}/bin" "${KATAGO_ENV_ROOT}/state"
cp -- "${bundle}/bin/katago" "${install_root}/bin/katago"
cp -- "${bundle}/metadata/source-manifest.tsv" "${KATAGO_ENV_ROOT}/state/source-manifest.tsv"
printf '%s\n' "${bundle}" > "${KATAGO_ENV_ROOT}/state/deployed-bundle"

python "${SCRIPT_DIR}/check-python-environment.py"
"${install_root}/bin/katago" version
log "prebuilt deployment complete: ${install_root}"
