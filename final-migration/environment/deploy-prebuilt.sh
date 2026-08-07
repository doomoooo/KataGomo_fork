#!/usr/bin/env bash

set -Eeuo pipefail

die() {
  printf '[katago-tar] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || die "usage: deploy-prebuilt.sh BUNDLE [PYTHON_ENV]"
bundle="$(readlink -e -- "$1")"
[[ -d "${bundle}" ]] || die "extracted bundle does not exist: $1"
python_env="$(readlink -m -- "${2:-${bundle}/python-env}")"
case "${python_env}" in
  /|/bin|/boot|/dev|/etc|/lib|/lib64|/proc|/run|/sbin|/sys|/usr|/var)
    die "refusing to create a Python environment in a system root: ${python_env}"
    ;;
esac

"${bundle}/verify.sh"
[[ -r "${bundle}/metadata/source-build-manifest.tsv" ]] || die "source build manifest missing"
[[ -r "${bundle}/metadata/python-binary-resolved.txt" ]] || die "resolved Python manifest missing"

system_python="${KATAGO_SYSTEM_PYTHON:-/usr/bin/python3}"
[[ -x "${system_python}" ]] || die "system Python is missing: ${system_python}"
expected_abi="$(sed -n 's/^python_abi=//p' "${bundle}/metadata/build-platform.txt")"
actual_abi="$("${system_python}" -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"
[[ -n "${expected_abi}" && "${actual_abi}" == "${expected_abi}" ]] \
  || die "Python ABI ${actual_abi} does not match bundle ABI ${expected_abi}; the KataGo executable remains directly usable"

if [[ ! -x "${python_env}/bin/python" ]]; then
  "${system_python}" -m venv "${python_env}"
fi
# shellcheck disable=SC1091
source "${python_env}/bin/activate"

printf '[katago-tar] installing the exact Python closure without network access\n'
python -m pip install \
  --no-index --no-deps --find-links "${bundle}/wheels" \
  --requirement "${bundle}/metadata/python-binary-resolved.txt"

while IFS=$'\t' read -r name distribution version commit builder wheel wheel_hash; do
  [[ "${name}" != "name" ]] || continue
  [[ "${wheel}" != "-" ]] || continue
  [[ -r "${bundle}/wheels/${wheel}" ]] || die "source wheel missing: ${wheel}"
  actual_hash="$(sha256sum "${bundle}/wheels/${wheel}" | awk '{print $1}')"
  [[ "${actual_hash}" == "${wheel_hash}" ]] || die "source wheel hash mismatch: ${wheel}"
  python -m pip install --no-index --no-deps --force-reinstall "${bundle}/wheels/${wheel}"
done < "${bundle}/metadata/source-build-manifest.tsv"

python "${bundle}/installer/check-python-environment.py"
printf '[katago-tar] Python tools installed only below %s\n' "${python_env}"
