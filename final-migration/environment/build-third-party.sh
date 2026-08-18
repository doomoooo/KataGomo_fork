#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

activate_venv
activate_toolchain
ensure_record_root
[[ -r "${KATAGO_ENV_ROOT}/state/source-manifest.tsv" ]] || die "source manifest missing; run acquire-third-party.sh"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "${KATAGO_RESUME_SOURCE_BUILD:-}" ]]; then
  build_dir="$(readlink -e -- "${KATAGO_RESUME_SOURCE_BUILD}")"
  [[ -d "${build_dir}" ]] || die "source build to resume does not exist"
  assert_safe_managed_path "${build_dir}"
  timestamp="$(basename -- "${build_dir}")"
else
  build_dir="${KATAGO_ENV_ROOT}/source-builds/${timestamp}"
fi
wheel_dir="${build_dir}/wheels"
manifest="${build_dir}/MANIFEST.tsv"
record="${KATAGO_RECORD_ROOT}/source-build-${timestamp}.log"
assert_safe_managed_path "${build_dir}"
mkdir -p -- "${wheel_dir}"
if [[ -n "${KATAGO_RESUME_SOURCE_BUILD:-}" ]]; then
  [[ -r "${manifest}" ]] || die "resume manifest missing: ${manifest}"
  exec > >(tee -a "${record}") 2>&1
else
  printf 'name\tdistribution\tversion\tcommit\tbuilder\twheel\tsha256\n' > "${manifest}"
  exec > >(tee "${record}") 2>&1
fi

configure_pip_environment
export MAX_JOBS="${KATAGO_BUILD_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${KATAGO_BUILD_JOBS}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;10.3;12.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-89;103;120}"
export XDG_CACHE_HOME="${KATAGO_ENV_ROOT}/cache"
mkdir -p -- "${XDG_CACHE_HOME}"

manifest_field() {
  local component="$1" field="$2"
  awk -F '\t' -v component="${component}" -v field="${field}" \
    '$1 == component {print $field; exit}' "${manifest}"
}

cleanup_component_build() {
  local name="$1" source_path="$2"
  if [[ "${KATAGO_KEEP_SOURCE_BUILD_TREES:-0}" != "1" && -d "${source_path}/build" ]]; then
    assert_safe_managed_path "${source_path}/build"
    log "discarding reproducible intermediate build tree for ${name}"
    find "${source_path}/build" -mindepth 1 -delete
    rmdir "${source_path}/build"
  fi
}

prepare_patched_flash_source() {
  local upstream="${KATAGO_THIRD_PARTY_ROOT}/flash-attention"
  local target="${KATAGO_ENV_ROOT}/patched-sources/flash-attention"
  local marker="${target}/.katago-patch-inputs"
  local upstream_revision cutlass_revision expected
  local sm89_patch="${REPO_ROOT}/cpp/neuralnet/flash-attention-sm89.patch"
  local sm120_patch="${MIGRATION_ROOT}/autotune/patches/flash-attention-sm120-both16.patch"
  upstream_revision="$(git -C "${upstream}" rev-parse HEAD)"
  cutlass_revision="$(git -C "${upstream}/csrc/cutlass" rev-parse HEAD)"
  expected="${upstream_revision} ${cutlass_revision} $(sha256sum "${sm89_patch}" | awk '{print $1}') $(sha256sum "${sm120_patch}" | awk '{print $1}')"
  if [[ -r "${marker}" && "$(cat "${marker}")" == "${expected}" ]]; then
    printf '%s\n' "${target}"
    return
  fi
  assert_safe_managed_path "${target}"
  if [[ -e "${target}" ]]; then
    find "${target}" -mindepth 1 -delete
    rmdir -- "${target}"
  fi
  mkdir -p -- "${target}"
  tar --create --file - --directory "${upstream}" \
    --exclude=.git --exclude=build --exclude=dist --exclude='*.egg-info' . \
    | tar --extract --file - --directory "${target}"
  (
    cd /tmp
    git apply --unsafe-paths --directory="${target}" "${sm89_patch}"
    git apply --unsafe-paths --directory="${target}" "${sm120_patch}"
  )
  printf '%s\n' "${upstream_revision}" > "${target}/.katago-source-revision"
  printf '%s\n' "${cutlass_revision}" > "${target}/csrc/cutlass/.katago-source-revision"
  printf '%s\n' "${expected}" > "${marker}"
  printf '%s\n' "${target}"
}

build_python_source_component() {
  local name="$1" source_path="$2" commit="$3" builder="$4" distribution="$5"
  local temp_wheels wheel wheel_name wheel_hash version expected_hash
  [[ -f "${source_path}/pyproject.toml" || -f "${source_path}/setup.py" ]] \
    || die "no Python build metadata for ${name}: ${source_path}"
  wheel_name="$(manifest_field "${name}" 6)"
  if [[ -n "${wheel_name}" ]]; then
    expected_hash="$(manifest_field "${name}" 7)"
    [[ -r "${wheel_dir}/${wheel_name}" ]] || die "resume wheel missing for ${name}: ${wheel_name}"
    wheel_hash="$(sha256sum "${wheel_dir}/${wheel_name}" | awk '{print $1}')"
    [[ "${wheel_hash}" == "${expected_hash}" ]] || die "resume wheel hash mismatch for ${name}"
    log "reusing verified ${name} wheel from interrupted build"
    python -m pip install --no-deps --force-reinstall "${wheel_dir}/${wheel_name}"
    cleanup_component_build "${name}" "${source_path}"
    return
  fi
  temp_wheels="$(mktemp -d "${build_dir}/${name}.wheels.XXXXXX")"
  log "building ${name} from ${commit}"
  if [[ "${name}" == "flash-attention" ]]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4=0.0.1.dev1+katago
  fi
  python -m pip wheel --no-build-isolation --no-deps \
    --wheel-dir "${temp_wheels}" "${source_path}"
  wheel="$(find "${temp_wheels}" -maxdepth 1 -type f -name '*.whl' -print -quit)"
  [[ -n "${wheel}" ]] || die "${name} did not produce a wheel"
  wheel_name="$(basename -- "${wheel}")"
  mv -- "${wheel}" "${wheel_dir}/${wheel_name}"
  rmdir -- "${temp_wheels}"
  python -m pip install --no-deps --force-reinstall "${wheel_dir}/${wheel_name}"
  wheel_hash="$(sha256sum "${wheel_dir}/${wheel_name}" | awk '{print $1}')"
  version="$(python -c 'import importlib.metadata,sys; print(importlib.metadata.version(sys.argv[1]))' "${distribution}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${distribution}" "${version}" "${commit}" "${builder}" "${wheel_name}" "${wheel_hash}" \
    >> "${manifest}"
  cleanup_component_build "${name}" "${source_path}"
}

build_python_component() {
  local name="$1" source_subdir="$2" commit="$3" builder="$4" distribution="$5"
  local source_root="${KATAGO_THIRD_PARTY_ROOT}/${name}"
  if [[ "${name}" == "flash-attention" ]]; then
    source_root="$(prepare_patched_flash_source)"
  fi
  build_python_source_component \
    "${name}" "${source_root}/${source_subdir}" \
    "${commit}" "${builder}" "${distribution}"
}

while IFS=$'\t' read -r name tier url requested_ref builder distribution submodules; do
  [[ -n "${name}" && "${name}" != \#* ]] || continue
  case "${tier}" in
    core) ;;
    research)
      [[ "${KATAGO_INCLUDE_RESEARCH:-0}" == "1" ]] || continue
      ;;
    *) die "unknown source tier for ${name}: ${tier}" ;;
  esac
  commit="$(git -C "${KATAGO_THIRD_PARTY_ROOT}/${name}" rev-parse HEAD)"
  case "${builder}" in
    python:*)
      build_python_component "${name}" "${builder#python:}" "${commit}" "${builder}" "${distribution}"
      ;;
    header)
      if [[ -z "$(manifest_field "${name}" 1)" ]]; then
        printf '%s\t-\t-\t%s\t%s\t-\t-\n' "${name}" "${commit}" "${builder}" >> "${manifest}"
      fi
      ;;
    rust)
      log "${name} requires its Rust-specific build and remains archive/reference-only"
      if [[ -z "$(manifest_field "${name}" 1)" ]]; then
        printf '%s\t%s\t-\t%s\t%s\t-\t-\n' "${name}" "${distribution}" "${commit}" "${builder}" >> "${manifest}"
      fi
      ;;
    *) die "unknown builder for ${name}: ${builder}" ;;
  esac
done < "${SCRIPT_DIR}/third-party.lock.tsv"

cp -- "${KATAGO_ENV_ROOT}/state/source-manifest.tsv" "${build_dir}/SOURCE-MANIFEST.tsv"
printf '%s\n' "${build_dir}" > "${KATAGO_ENV_ROOT}/state/latest-source-build"
python "${SCRIPT_DIR}/check-python-environment.py"
log "source builds complete; distributable wheels=${wheel_dir}"
printf 'manifest=%s\nrecord=%s\n' "${manifest}" "${record}"
