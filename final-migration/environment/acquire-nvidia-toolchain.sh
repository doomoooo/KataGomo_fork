#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

activate_venv
for command_name in awk cp curl find grep ln mktemp mv readlink sha256sum stat tar wc; do
  require_command "${command_name}"
done

lock="${SCRIPT_DIR}/nvidia-toolchain.lock.tsv"
downloads="${KATAGO_ENV_ROOT}/downloads/nvidia-redist"
mkdir -p -- "${KATAGO_TOOLCHAIN_ROOT}" "${downloads}" "${KATAGO_ENV_ROOT}/state"

cuda_valid() {
  [[ -x "${KATAGO_CUDA_ROOT}/bin/nvcc" ]] || return 1
  [[ -r "${KATAGO_CUDA_ROOT}/include/cuda.h" ]] || return 1
  [[ -r "${KATAGO_CUDA_ROOT}/lib64/libcudart.so" ]] || return 1
  [[ -r "${KATAGO_CUDA_ROOT}/lib64/libcublas.so" ]] || return 1
  [[ -r "${KATAGO_CUDA_ROOT}/lib64/libnvrtc.so" ]] || return 1
  "${KATAGO_CUDA_ROOT}/bin/nvcc" --version | grep -q 'release 13\.2'
}

cudnn_valid() {
  [[ -r "${KATAGO_CUDNN_ROOT}/include/cudnn.h" ]] || return 1
  [[ -r "${KATAGO_CUDNN_ROOT}/include/cudnn_version.h" ]] || return 1
  [[ -r "${KATAGO_CUDNN_ROOT}/lib/libcudnn.so" ]] || return 1
  grep -Eq '^#define CUDNN_MAJOR[[:space:]]+9$' "${KATAGO_CUDNN_ROOT}/include/cudnn_version.h"
  grep -Eq '^#define CUDNN_MINOR[[:space:]]+25$' "${KATAGO_CUDNN_ROOT}/include/cudnn_version.h"
}

toolchain_valid() {
  cuda_valid && cudnn_valid
}

if toolchain_valid; then
  log "reusing managed CUDA 13.2 and cuDNN 9.25 toolchain"
  activate_toolchain
  nvcc --version | tail -n 1
  exit 0
fi

manifest_for_product() {
  awk -F '\t' -v product="$1" '$1 == product {print $3 "\t" $4; exit}' "${lock}"
}

obtain_file() {
  local filename="$1" url="$2" expected_sha="$3" expected_size="$4"
  local candidate actual_sha actual_size
  for candidate in \
    "${KATAGO_LOCAL_ARCHIVE}/nvidia/${filename}" \
    "${KATAGO_LOCAL_ARCHIVE}/toolchains/${filename}" \
    "${KATAGO_LOCAL_ARCHIVE}/${filename}" \
    "${downloads}/${filename}"; do
    if [[ -r "${candidate}" ]]; then
      actual_sha="$(sha256sum "${candidate}" | awk '{print $1}')"
      actual_size="$(stat -c %s "${candidate}")"
      if [[ "${actual_sha}" == "${expected_sha}" && "${actual_size}" == "${expected_size}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      warn "ignoring corrupt cached NVIDIA archive: ${candidate}"
    fi
  done
  candidate="${downloads}/${filename}"
  warn "downloading pinned NVIDIA redistributable: ${filename}"
  curl --fail --location --retry 3 --continue-at - \
    --output "${candidate}.partial" "${url}"
  mv -- "${candidate}.partial" "${candidate}"
  actual_sha="$(sha256sum "${candidate}" | awk '{print $1}')"
  actual_size="$(stat -c %s "${candidate}")"
  [[ "${actual_sha}" == "${expected_sha}" ]] \
    || die "NVIDIA archive SHA-256 mismatch for ${filename}: ${actual_sha}"
  [[ "${actual_size}" == "${expected_size}" ]] \
    || die "NVIDIA archive size mismatch for ${filename}: ${actual_size}"
  printf '%s\n' "${candidate}"
}

prepare_manifest() {
  local product="$1" manifest_url expected_sha filename candidate actual_sha
  IFS=$'\t' read -r manifest_url expected_sha < <(manifest_for_product "${product}")
  [[ -n "${manifest_url}" && -n "${expected_sha}" ]] \
    || die "missing NVIDIA manifest lock for ${product}"
  filename="$(basename -- "${manifest_url}")"
  for candidate in \
    "${KATAGO_LOCAL_ARCHIVE}/nvidia/${filename}" \
    "${KATAGO_LOCAL_ARCHIVE}/toolchains/${filename}" \
    "${KATAGO_LOCAL_ARCHIVE}/${filename}" \
    "${downloads}/${filename}"; do
    if [[ -r "${candidate}" ]]; then
      actual_sha="$(sha256sum "${candidate}" | awk '{print $1}')"
      if [[ "${actual_sha}" == "${expected_sha}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
      warn "ignoring NVIDIA manifest with unexpected hash: ${candidate}"
    fi
  done
  candidate="${downloads}/${filename}"
  warn "downloading pinned NVIDIA ${product} redistributable manifest"
  curl --fail --location --retry 3 --output "${candidate}.partial" "${manifest_url}"
  mv -- "${candidate}.partial" "${candidate}"
  actual_sha="$(sha256sum "${candidate}" | awk '{print $1}')"
  [[ "${actual_sha}" == "${expected_sha}" ]] \
    || die "NVIDIA manifest SHA-256 mismatch for ${product}: ${actual_sha}"
  printf '%s\n' "${candidate}"
}

resolve_artifacts() {
  local product="$1" manifest="$2"
  "${KATAGO_FINAL_VENV}/bin/python" - "${lock}" "${product}" "${manifest}" <<'PY'
import json
import pathlib
import sys
import urllib.parse

lock_path, product, manifest_path = map(pathlib.Path, sys.argv[1:])
product_name = str(product)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
with lock_path.open(encoding="utf-8") as handle:
    for raw in handle:
        if not raw.strip() or raw.startswith("#"):
            continue
        row = raw.rstrip("\n").split("\t")
        if row[0] != product_name:
            continue
        _, _, manifest_url, _, platform, variant, component = row
        entry = manifest.get(component, {}).get(platform)
        if variant != "-":
            entry = entry.get(variant) if isinstance(entry, dict) else None
        if not isinstance(entry, dict):
            raise SystemExit(f"locked NVIDIA component is absent: {component}/{platform}/{variant}")
        relative = entry.get("relative_path")
        sha256 = entry.get("sha256")
        size = str(entry.get("size", ""))
        if not relative or not sha256 or not size.isdigit():
            raise SystemExit(f"incomplete NVIDIA manifest entry: {component}")
        url = urllib.parse.urljoin(manifest_url, relative)
        print("\t".join((component, url, sha256, size, pathlib.PurePosixPath(relative).name)))
PY
}

assemble_product() {
  local product="$1" destination="$2" manifest stage extract_root merged
  local component url expected_sha expected_size filename archive top count
  manifest="$(prepare_manifest "${product}")"
  stage="$(mktemp -d "${KATAGO_TOOLCHAIN_ROOT}/.${product}.XXXXXXXX")"
  extract_root="${stage}/extract"
  merged="${stage}/merged"
  mkdir -p -- "${extract_root}" "${merged}"
  while IFS=$'\t' read -r component url expected_sha expected_size filename; do
    log "resolving NVIDIA ${product} component ${component}"
    archive="$(obtain_file "${filename}" "${url}" "${expected_sha}" "${expected_size}")"
    find "${extract_root}" -mindepth 1 -delete
    tar --extract --xz --file "${archive}" --directory "${extract_root}"
    count="$(find "${extract_root}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "${count}" == "1" ]] || die "unexpected NVIDIA archive layout: ${archive}"
    top="$(find "${extract_root}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    cp -a -- "${top}/." "${merged}/"
  done < <(resolve_artifacts "${product}" "${manifest}")
  if [[ "${product}" == cuda && ! -e "${merged}/lib64" && -d "${merged}/lib" ]]; then
    ln -s lib "${merged}/lib64"
  fi
  cp -- "${manifest}" "${merged}/.katago-redist-manifest.json"
  sha256sum "${lock}" | awk '{print $1}' > "${merged}/.katago-redist-lock.sha256"
  if [[ -e "${destination}" ]]; then
    assert_safe_managed_path "${destination}"
    find "${destination}" -mindepth 1 -delete
    rmdir -- "${destination}"
  fi
  mv -- "${merged}" "${destination}"
  find "${stage}" -mindepth 1 -delete
  rmdir -- "${stage}"
}

cuda_valid || assemble_product cuda "${KATAGO_CUDA_ROOT}"
cudnn_valid || assemble_product cudnn "${KATAGO_CUDNN_ROOT}"
toolchain_valid || die "assembled NVIDIA toolchain did not pass validation"
activate_toolchain
nvcc --version | tail -n 1
log "managed CUDA/cuDNN toolchain complete"
