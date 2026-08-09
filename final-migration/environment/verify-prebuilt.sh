#!/usr/bin/env bash

set -Eeuo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

die() {
  printf '[katago-tar] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$(uname -s)" == "Linux" ]] || die "this bundle targets Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "this bundle targets x86_64"
[[ -r "${bundle_root}/SHA256SUMS" ]] || die "SHA256SUMS is missing"
[[ -x "${bundle_root}/lib/ld-linux-x86-64.so.2" ]] \
  || die "the bundled ELF loader is missing"

printf '[katago-tar] verifying bundle files\n'
(cd -- "${bundle_root}" && sha256sum --check --quiet SHA256SUMS)
[[ -r "${bundle_root}/metadata/runtime-symlinks.tsv" ]] \
  || die "runtime symlink manifest is missing"
if ! cmp --silent "${bundle_root}/metadata/runtime-symlinks.tsv" <(
  printf 'name\ttarget\n'
  find "${bundle_root}/lib" -maxdepth 1 -type l -printf '%f\t%l\n' | sort
); then
  die "runtime symlinks do not match the manifest"
fi

runtime_ldd="$("${bundle_root}/lib/ld-linux-x86-64.so.2" \
  --library-path "${bundle_root}/lib" \
  --list "${bundle_root}/libexec/katago")"
cuda_major="$(<"${bundle_root}/metadata/cuda-major.txt")"
cudnn_soname="$(<"${bundle_root}/metadata/cudnn-soname.txt")"
[[ "${cuda_major}" =~ ^[0-9]+$ ]] || die "invalid recorded CUDA major"
[[ "${cudnn_soname}" =~ ^libcudnn\.so\.[0-9]+$ ]] || die "invalid recorded cuDNN SONAME"
for library in "libcublas.so.${cuda_major}" "${cudnn_soname}" "libnvrtc.so.${cuda_major}"; do
  resolved="$(awk -v library="${library}" '$1 == library {print $3; exit}' <<< "${runtime_ldd}")"
  [[ "${resolved}" == "${bundle_root}/lib/"* ]] \
    || die "${library} did not resolve from the tar bundle: ${resolved:-missing}"
done
for library in libc.so.6 libm.so.6 libstdc++.so.6; do
  resolved="$(awk -v library="${library}" '$1 == library {print $3; exit}' <<< "${runtime_ldd}")"
  [[ "${resolved}" == "${bundle_root}/lib/"* ]] \
    || die "${library} did not resolve from the tar bundle: ${resolved:-missing}"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
  [[ -n "${driver_version}" ]] || die "NVIDIA driver is not operational"
  min_driver="$(<"${bundle_root}/metadata/minimum-driver.txt")"
  if [[ "$(printf '%s\n%s\n' "${min_driver}" "${driver_version}" | sort -V | head -n 1)" != "${min_driver}" ]]; then
    die "NVIDIA driver ${driver_version} is older than required ${min_driver}"
  fi
else
  die "nvidia-smi is unavailable; install a compatible host driver"
fi

"${bundle_root}/bin/katago" version
printf '[katago-tar] bundle verification passed; no system files were changed\n'
