#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

activate_venv
ensure_record_root
require_command sha256sum

latest_build_file="${KATAGO_ENV_ROOT}/state/latest-source-build"
[[ -r "${latest_build_file}" ]] || die "completed source build missing; run setup.sh install"
source_build="$(<"${latest_build_file}")"
[[ -r "${source_build}/MANIFEST.tsv" ]] || die "source build manifest missing: ${source_build}"
katago_binary="${KATAGO_BUILD_ROOT:-${KATAGO_ENV_ROOT}/katago-builds}/cuda/katago"
[[ -x "${katago_binary}" ]] || die "KataGo CUDA binary missing; run setup.sh build before packaging"
katago_build_dir="$(dirname -- "${katago_binary}")"
unexpected_runtime_paths="$(ldd "${katago_binary}" | grep -E '=> /(workspace|opt/nvidia)/' || true)"
[[ -z "${unexpected_runtime_paths}" ]] \
  || die "KataGo binary contains host-specific runtime paths:${unexpected_runtime_paths//$'\n'/; }"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
bundle="${KATAGO_DISTRIBUTION_ROOT:-${KATAGO_ENV_ROOT}/distributions}/${timestamp}"
assert_safe_managed_path "${bundle}"
wheelhouse="${bundle}/wheels"
metadata_dir="${bundle}/metadata"
binary_dir="${bundle}/bin"
installer_dir="${bundle}/installer"
mkdir -p -- "${wheelhouse}" "${metadata_dir}" "${binary_dir}" "${installer_dir}/lib"

cp -- "${source_build}/MANIFEST.tsv" "${metadata_dir}/source-build-manifest.tsv"
cp -- "${source_build}/SOURCE-MANIFEST.tsv" "${metadata_dir}/source-manifest.tsv"
cp -- "${source_build}/runtime-requirements.txt" "${metadata_dir}/runtime-requirements.txt"
cp -- "${SCRIPT_DIR}/python-bootstrap-requirements.txt" "${metadata_dir}/python-bootstrap-requirements.txt"
cp -- "${source_build}"/wheels/*.whl "${wheelhouse}/"
cp -- "${katago_binary}" "${binary_dir}/katago"
cp -- "${SCRIPT_DIR}/distribution-README.md" "${bundle}/README.md"
cp -- "${SCRIPT_DIR}/deploy-prebuilt.sh" "${installer_dir}/deploy-prebuilt.sh"
cp -- "${SCRIPT_DIR}/bootstrap-ubuntu.sh" "${installer_dir}/bootstrap-ubuntu.sh"
cp -- "${SCRIPT_DIR}/check-python-environment.py" "${installer_dir}/check-python-environment.py"
cp -- "${SCRIPT_DIR}/apt-packages.txt" "${installer_dir}/apt-packages.txt"
cp -- "${SCRIPT_DIR}/lib/common.sh" "${installer_dir}/lib/common.sh"

copy_latest_record() {
  local pattern="$1" destination="$2" candidate
  candidate="$(find "${KATAGO_RECORD_ROOT}" -maxdepth 1 -type f -name "${pattern}" \
    -printf '%T@\t%p\n' | sort -n | tail -n 1 | cut -f2-)"
  if [[ -n "${candidate}" ]]; then
    cp -- "${candidate}" "${metadata_dir}/${destination}"
  fi
}
source_build_id="$(basename -- "${source_build}")"
if [[ -r "${KATAGO_RECORD_ROOT}/source-build-${source_build_id}.log" ]]; then
  cp -- "${KATAGO_RECORD_ROOT}/source-build-${source_build_id}.log" \
    "${metadata_dir}/source-build.log"
fi
copy_latest_record 'environment-*.log' environment-audit.log
copy_latest_record 'third-party-verify-*.log' third-party-verify.log
copy_latest_record 'build-matrix-*.log' katago-build.log

python "${SCRIPT_DIR}/distribution-requirements.py" \
  "${source_build}/MANIFEST.tsv" > "${metadata_dir}/python-binary-resolved.txt"

cuda_release="$(nvcc --version | sed -n -E 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | tail -n 1)"
[[ "${cuda_release}" =~ ^[0-9]+\.[0-9]+$ ]] || die "could not resolve nvcc major.minor"
cuda_package_suffix="${cuda_release/./-}"
{
  printf 'KATAGO_CUDA_TOOLKIT_PACKAGE=cuda-toolkit-%s\n' "${cuda_package_suffix}"
  printf 'KATAGO_CUDNN_PACKAGE=libcudnn9-dev-cuda-%s\n' "${cuda_release%%.*}"
} > "${metadata_dir}/system-requirements.env"

log "downloading the exact binary/runtime wheel closure into the distribution"
python -m pip download \
  --index-url "${KATAGO_PYPI_MIRROR}" \
  --only-binary=:all: \
  --no-deps \
  --dest "${wheelhouse}" \
  --requirement "${metadata_dir}/python-binary-resolved.txt"

{
  printf 'created_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'katago_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  printf 'katago_describe=%s\n' "$(git -C "${REPO_ROOT}" describe --tags --always 2>/dev/null || printf unknown)"
  printf 'python=%s\n' "$(python --version 2>&1)"
  printf 'python_abi=%s\n' "$(python -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"
  printf 'platform=%s\n' "$(python -c 'import platform; print(platform.platform())')"
  printf 'cuda_compiler=%s\n' "$(nvcc --version | tail -n 1)"
  printf 'driver=%s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sort -u | paste -sd, - || printf unavailable)"
  printf 'gpu_architectures=%s\n' "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -Vu | paste -sd, - || printf unavailable)"
} > "${metadata_dir}/build-platform.txt"
ldd "${binary_dir}/katago" > "${metadata_dir}/katago-ldd.txt"
"${binary_dir}/katago" version > "${metadata_dir}/katago-version.txt"
cp -- "${katago_build_dir}/CMakeCache.txt" "${metadata_dir}/katago-CMakeCache.txt"
dpkg-query -W -f='${binary:Package}\t${Version}\n' 2>/dev/null \
  | grep -E '^(cuda-|libcublas|libcudnn|libzip|zlib|nsight-|nvidia-|gcc|g\+\+|cmake|ninja-build)' \
  | sort > "${metadata_dir}/system-packages.tsv"

(
  cd -- "${bundle}"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
mkdir -p -- "${KATAGO_ENV_ROOT}/state"
printf '%s\n' "${bundle}" > "${KATAGO_ENV_ROOT}/state/latest-distribution"

if [[ "${KATAGO_PACKAGE_TAR:-0}" == "1" ]]; then
  require_command zstd
  tarball="${bundle}.tar.zst"
  tar --zstd -C "$(dirname -- "${bundle}")" -cf "${tarball}" "$(basename -- "${bundle}")"
  sha256sum "${tarball}" > "${tarball}.sha256"
  log "compressed distribution=${tarball}"
fi

log "prebuilt distribution complete: ${bundle}"
printf 'bundle=%s\nfiles=%s\n' "${bundle}" "$(find "${bundle}" -type f | wc -l)"
