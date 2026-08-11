#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

if [[ -d "${KATAGO_FINAL_VENV}/bin" ]]; then
  export PATH="${KATAGO_FINAL_VENV}/bin:${PATH}"
fi
ensure_record_root
record="${KATAGO_RECORD_ROOT}/environment-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "${record}") 2>&1

failures=0
check_command() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    printf 'OK command %-16s %s\n' "${command_name}" "$(command -v "${command_name}")"
  else
    printf 'FAIL command %s\n' "${command_name}"
    failures=$((failures + 1))
  fi
}

printf 'KataGo final migration environment audit\n'
printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
printf 'repo=%s\n' "${REPO_ROOT}"
printf 'commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
printf 'env_root=%s\n' "${KATAGO_ENV_ROOT}"
printf 'local_archive=%s\n' "${KATAGO_LOCAL_ARCHIVE}"
printf 'third_party_root=%s\n' "${KATAGO_THIRD_PARTY_ROOT}"

printf '\n[operating-system]\n'
sed -n -E 's/^(PRETTY_NAME|VERSION_ID)=/\0/p' /etc/os-release
uname -a

printf '\n[commands]\n'
for command_name in gcc g++ cmake ninja git nvcc nvidia-smi nsys ncu; do
  check_command "${command_name}"
done
gcc --version 2>/dev/null | head -n 1 || true
cmake --version 2>/dev/null | head -n 1 || true
ninja --version 2>/dev/null || true
"${KATAGO_FINAL_VENV}/bin/python" --version 2>/dev/null || {
  printf 'FAIL pinned Python environment missing: %s\n' "${KATAGO_FINAL_VENV}"
  failures=$((failures + 1))
}
nvcc --version 2>/dev/null | tail -n 1 || true
nsys --version 2>/dev/null | head -n 1 || true
ncu --version 2>/dev/null | tail -n 1 || true

printf '\n[nvidia-driver-and-devices]\n'
if nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,compute_cap,memory.total --format=csv,noheader 2>/dev/null; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  min_driver="${KATAGO_MIN_DRIVER:-595.45}"
  if [[ "$(printf '%s\n%s\n' "${min_driver}" "${driver_version}" | sort -V | head -n 1)" != "${min_driver}" ]]; then
    printf 'FAIL driver %s is older than required %s\n' "${driver_version}" "${min_driver}"
    failures=$((failures + 1))
  fi
else
  printf 'FAIL no operational NVIDIA device/driver; reboot may be required\n'
  failures=$((failures + 1))
fi

printf '\n[system-packages]\n'
dpkg-query -W -f='${binary:Package}\t${Version}\n' 2>/dev/null \
  | grep -E '^(cuda-toolkit|cuda-compiler|libcublas|libcudnn|nsight|zlib1g-dev)' \
  | sort || true
required_packages=(
  build-essential git zlib1g-dev
  libzip-dev libcudnn9-dev-cuda-13
)
for package_name in "${required_packages[@]}"; do
  if dpkg-query -W -f='${Status}' "${package_name}" 2>/dev/null | grep -q 'install ok installed'; then
    printf 'OK package %s\n' "${package_name}"
  else
    printf 'FAIL package %s\n' "${package_name}"
    failures=$((failures + 1))
  fi
done

printf '\n[dynamic-libraries]\n'
ldconfig -p 2>/dev/null | grep -E 'lib(cudnn|cublas|cuda|cudart)\.so' | sort || true

printf '\n[resolved-source-checkouts]\n'
source_manifest="${KATAGO_ENV_ROOT}/state/source-manifest.tsv"
deployed_bundle_file="${KATAGO_ENV_ROOT}/state/deployed-bundle"
if [[ ! -r "${source_manifest}" ]]; then
  printf 'FAIL resolved source manifest missing: %s\n' "${source_manifest}"
  failures=$((failures + 1))
else
while IFS=$'\t' read -r name tier url requested_ref resolved_commit describe submodule_commits; do
  [[ -n "${name}" && "${name}" != \#* ]] || continue
  source_path="${KATAGO_THIRD_PARTY_ROOT}/${name}"
  if [[ -d "${source_path}/.git" ]]; then
    actual="$(git -C "${source_path}" rev-parse HEAD)"
    state=OK
    [[ "${actual}" == "${resolved_commit}" ]] || { state=FAIL; failures=$((failures + 1)); }
    printf '%s %-20s requested=%s resolved=%s actual=%s describe=%s\n' \
      "${state}" "${name}" "${requested_ref}" "${resolved_commit}" "${actual}" "${describe}"
    if [[ -n "$(git -C "${source_path}" status --porcelain)" ]]; then
      printf 'WARN %-20s checkout is dirty\n' "${name}"
    fi
  elif [[ -r "${deployed_bundle_file}" ]]; then
    printf 'OK %-20s packaged source=%s (deployment has no checkout)\n' "${name}" "${resolved_commit}"
  else
    printf 'FAIL missing source %s\n' "${source_path}"
    failures=$((failures + 1))
  fi
done < <(tail -n +2 "${source_manifest}")
fi

printf '\n[python-stack]\n'
if [[ -x "${KATAGO_FINAL_VENV}/bin/python" ]]; then
  activate_venv
  if ! python - <<'PY'
import importlib
import importlib.metadata
import os
import sys

modules = [
    "torch", "triton", "tilelang", "flash_attn.cute", "cutlass", "cuda",
    "onnx", "onnx2torch", "numpy", "psutil", "yaml", "quack", "tvm_ffi",
]
failed = False
for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "<unknown>")
        print(f"OK module {name:20s} version={version} path={getattr(module, '__file__', None)}")
    except Exception as exc:
        print(f"FAIL module {name}: {type(exc).__name__}: {exc}")
        failed = True

import torch
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.backends.cudnn.version={torch.backends.cudnn.version()}")
print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
sys.exit(1 if failed else 0)
PY
  then
    failures=$((failures + 1))
  fi
  if ! python "${SCRIPT_DIR}/check-python-environment.py"; then
    failures=$((failures + 1))
  fi
else
  printf 'FAIL Python environment missing: %s\n' "${KATAGO_FINAL_VENV}"
  failures=$((failures + 1))
fi

printf '\n[result]\n'
printf 'failures=%d\n' "${failures}"
printf 'record=%s\n' "${record}"
(( failures == 0 )) || exit 1
