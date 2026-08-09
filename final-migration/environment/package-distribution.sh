#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

activate_venv
ensure_record_root
for command_name in dpkg-query ldd readelf sha256sum tar; do
  require_command "${command_name}"
done

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
runtime_lib_dir="${bundle}/lib"
mkdir -p -- "${wheelhouse}" "${metadata_dir}" "${runtime_lib_dir}" \
  "${bundle}/bin" "${bundle}/libexec" "${bundle}/installer" "${bundle}/licenses"

cp -- "${source_build}/MANIFEST.tsv" "${metadata_dir}/source-build-manifest.tsv"
cp -- "${source_build}/SOURCE-MANIFEST.tsv" "${metadata_dir}/source-manifest.tsv"
cp -- "${source_build}/runtime-requirements.txt" "${metadata_dir}/runtime-requirements.txt"
cp -- "${SCRIPT_DIR}/python-bootstrap-requirements.txt" "${metadata_dir}/python-bootstrap-requirements.txt"
cp -- "${source_build}"/wheels/*.whl "${wheelhouse}/"
cp -- "${katago_binary}" "${bundle}/libexec/katago"
cp -- "${SCRIPT_DIR}/runtime-wrapper.sh" "${bundle}/bin/katago"
cp -- "${SCRIPT_DIR}/verify-prebuilt.sh" "${bundle}/verify.sh"
cp -- "${SCRIPT_DIR}/deploy-prebuilt.sh" "${bundle}/installer/deploy-prebuilt.sh"
cp -- "${SCRIPT_DIR}/install-tar.sh" "${bundle}/installer/install-tar.sh"
cp -- "${SCRIPT_DIR}/check-python-environment.py" "${bundle}/installer/check-python-environment.py"
cp -- "${SCRIPT_DIR}/distribution-README.md" "${bundle}/README.md"
cp -- "${REPO_ROOT}/final-migration/README.md" "${bundle}/FORK-README.md"
cp -- "${REPO_ROOT}/final-migration/README.zh-CN.md" "${bundle}/FORK-README.zh-CN.md"
cp -a -- "${REPO_ROOT}/final-migration/plans" "${bundle}/plans"
chmod 0755 "${bundle}/bin/katago" "${bundle}/verify.sh" \
  "${bundle}/installer/deploy-prebuilt.sh" "${bundle}/installer/install-tar.sh"

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

seed_wheels() {
  local seed_dir="$1"
  [[ -d "${seed_dir}" ]] || return 0
  while IFS= read -r wheel; do
    cp --no-clobber -- "${wheel}" "${wheelhouse}/"
  done < <(find "${seed_dir}" -maxdepth 1 -type f -name '*.whl' -print | sort)
}

# Reuse a verified previous distribution and the explicit local archive before
# consulting the configured mirror. pip only fetches pinned files still absent.
previous_distribution_file="${KATAGO_ENV_ROOT}/state/latest-distribution"
if [[ -r "${previous_distribution_file}" ]]; then
  previous_distribution="$(<"${previous_distribution_file}")"
  if [[ -d "${previous_distribution}" && -r "${previous_distribution}/SHA256SUMS" ]]; then
    (cd -- "${previous_distribution}" && sha256sum --check --quiet SHA256SUMS)
    seed_wheels "${previous_distribution}/wheels"
  fi
fi
seed_wheels "${KATAGO_LOCAL_ARCHIVE}/wheels"

log "downloading the exact binary/runtime wheel closure into the tar bundle"
python -m pip download \
  --index-url "${KATAGO_PYPI_MIRROR}" \
  --only-binary=:all: \
  --no-deps \
  --dest "${wheelhouse}" \
  --requirement "${metadata_dir}/python-binary-resolved.txt"

cuda_release="$(nvcc --version | sed -n -E 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | tail -n 1)"
[[ "${cuda_release}" =~ ^[0-9]+\.[0-9]+$ ]] || die "could not resolve nvcc major.minor"
cuda_major="${cuda_release%%.*}"
system_multiarch="$(gcc -print-multiarch)"
cudnn_path="$(ldd "${katago_binary}" | awk '$1 ~ /^libcudnn\.so/ {print $3; exit}')"
[[ -r "${cudnn_path}" ]] || die "could not resolve the linked cuDNN library"
cudnn_lib_dir="$(dirname -- "${cudnn_path}")"
cudnn_soname="$(readelf -d "$(readlink -e -- "${cudnn_path}")" \
  | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p' | head -n 1)"
[[ "${cudnn_soname}" =~ ^libcudnn\.so\.[0-9]+$ ]] \
  || die "could not resolve the cuDNN SONAME"
printf '%s\n' "${cuda_release}" > "${metadata_dir}/cuda-release.txt"
printf '%s\n' "${cuda_major}" > "${metadata_dir}/cuda-major.txt"
printf '%s\n' "${cudnn_soname}" > "${metadata_dir}/cudnn-soname.txt"

runtime_manifest="${metadata_dir}/runtime-libraries.tsv"
printf 'soname\tpackaged_file\tsource_path\tpackage\tsha256\n' > "${runtime_manifest}"
declare -a runtime_queue=()
declare -A runtime_seen=()

enqueue_runtime() {
  local candidate="$1"
  [[ -n "${candidate}" && -e "${candidate}" ]] || return 0
  runtime_queue+=("${candidate}")
}

while IFS= read -r dependency; do
  enqueue_runtime "${dependency}"
done < <(ldd "${katago_binary}" | awk '/=> \// {print $3}')
elf_interpreter="$(readelf -l "${katago_binary}" \
  | sed -n 's@.*Requesting program interpreter: \(.*\)]@\1@p')"
[[ -x "${elf_interpreter}" ]] || die "could not resolve the ELF interpreter"
enqueue_runtime "${elf_interpreter}"

for cuda_library in \
  "${CUDA_HOME:-/usr/local/cuda}/lib64/libcublas.so.${cuda_major}" \
  "${CUDA_HOME:-/usr/local/cuda}/lib64/libcublasLt.so.${cuda_major}" \
  "${CUDA_HOME:-/usr/local/cuda}/lib64/libcudart.so.${cuda_major}" \
  "${CUDA_HOME:-/usr/local/cuda}/lib64/libnvJitLink.so.${cuda_major}" \
  "${CUDA_HOME:-/usr/local/cuda}/lib64/libnvrtc.so.${cuda_major}" \
  "${CUDA_HOME:-/usr/local/cuda}/lib64/libnvrtc-builtins.so.${cuda_release}"; do
  enqueue_runtime "${cuda_library}"
done
while IFS= read -r cudnn_library; do
  enqueue_runtime "${cudnn_library}"
done < <(find "${cudnn_lib_dir}" -maxdepth 1 -name 'libcudnn*.so.9' -print | sort)

queue_index=0
while (( queue_index < ${#runtime_queue[@]} )); do
  candidate="${runtime_queue[queue_index]}"
  queue_index=$((queue_index + 1))
  resolved="$(readlink -e -- "${candidate}")"
  [[ -n "${resolved}" ]] || die "runtime library vanished: ${candidate}"
  resolved_name="$(basename -- "${resolved}")"
  case "${resolved_name}" in
    libcuda.so.*)
      continue
      ;;
  esac
  [[ -z "${runtime_seen[${resolved}]:-}" ]] || continue
  runtime_seen["${resolved}"]=1

  packaged_file="${runtime_lib_dir}/${resolved_name}"
  cp -L --preserve=mode,timestamps -- "${resolved}" "${packaged_file}"
  soname="$(readelf -d "${resolved}" | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p' | head -n 1)"
  [[ -n "${soname}" ]] || soname="${resolved_name}"
  if [[ "${soname}" != "${resolved_name}" ]]; then
    ln -sfn -- "${resolved_name}" "${runtime_lib_dir}/${soname}"
  fi
  requested_name="$(basename -- "${candidate}")"
  if [[ "${requested_name}" != "${resolved_name}" && "${requested_name}" != "${soname}" ]]; then
    ln -sfn -- "${resolved_name}" "${runtime_lib_dir}/${requested_name}"
  fi
  package_owner="$(dpkg-query -S "${resolved}" 2>/dev/null | head -n 1 | cut -d: -f1 || true)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${soname}" "${resolved_name}" "${resolved}" "${package_owner:--}" \
    "$(sha256sum "${packaged_file}" | awk '{print $1}')" >> "${runtime_manifest}"

  while IFS= read -r dependency; do
    enqueue_runtime "${dependency}"
  done < <(ldd "${resolved}" 2>/dev/null | awk '/=> \// {print $3}')
done

[[ -r "${CUDA_HOME:-/usr/local/cuda}/EULA.txt" ]] \
  && cp -- "${CUDA_HOME:-/usr/local/cuda}/EULA.txt" "${bundle}/licenses/NVIDIA-CUDA-EULA.txt"
while IFS= read -r package_owner; do
  copyright_file="/usr/share/doc/${package_owner}/copyright"
  [[ -r "${copyright_file}" ]] || continue
  cp -- "${copyright_file}" "${bundle}/licenses/${package_owner}.copyright"
done < <(tail -n +2 "${runtime_manifest}" | cut -f4 | grep -v '^-$' | sort -u)

runtime_symlinks="${metadata_dir}/runtime-symlinks.tsv"
printf 'name\ttarget\n' > "${runtime_symlinks}"
find "${runtime_lib_dir}" -maxdepth 1 -type l \
  -printf '%f\t%l\n' | sort >> "${runtime_symlinks}"

bundled_loader="${runtime_lib_dir}/ld-linux-x86-64.so.2"
[[ -x "${bundled_loader}" ]] || die "bundled ELF loader is missing"
bundled_ldd="$("${bundled_loader}" --library-path "${runtime_lib_dir}" \
  --list "${bundle}/libexec/katago")"
printf '%s\n' "${bundled_ldd}" > "${metadata_dir}/katago-loader-list.txt"
for library in "libcublas.so.${cuda_major}" "${cudnn_soname}" "libnvrtc.so.${cuda_major}"; do
  resolved_from="$(awk -v library="${library}" '$1 == library {print $3; exit}' <<< "${bundled_ldd}")"
  [[ "${resolved_from}" == "${runtime_lib_dir}/"* ]] \
    || die "${library} did not resolve from the bundled runtime: ${resolved_from:-missing}"
done
for library in libc.so.6 libm.so.6 libstdc++.so.6; do
  resolved_from="$(awk -v library="${library}" '$1 == library {print $3; exit}' <<< "${bundled_ldd}")"
  [[ "${resolved_from}" == "${runtime_lib_dir}/"* ]] \
    || die "${library} did not resolve from the bundled runtime: ${resolved_from:-missing}"
done

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
printf '%s\n' "${KATAGO_MIN_DRIVER:-595.45}" > "${metadata_dir}/minimum-driver.txt"
"${bundle}/bin/katago" version > "${metadata_dir}/katago-version.txt"
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

tarball="${bundle}.tar"
tar --create --file "${tarball}" --directory "$(dirname -- "${bundle}")" "$(basename -- "${bundle}")"
(
  cd -- "$(dirname -- "${tarball}")"
  sha256sum "$(basename -- "${tarball}")" > "$(basename -- "${tarball}").sha256"
)
cp -- "${SCRIPT_DIR}/install-tar.sh" "${tarball}.install.sh"
chmod 0755 "${tarball}.install.sh"
(
  cd -- "$(dirname -- "${tarball}")"
  sha256sum "$(basename -- "${tarball}").install.sh" \
    > "$(basename -- "${tarball}").install.sh.sha256"
)

log "non-invasive tar distribution complete: ${tarball}"
printf 'bundle=%s\ntar=%s\nfiles=%s\n' \
  "${bundle}" "${tarball}" "$(find "${bundle}" -type f | wc -l)"
