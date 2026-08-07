#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

if [[ "$(id -u)" -ne 0 ]]; then
  # Create the managed root as the invoking user before elevating only the
  # package transaction. The remaining setup must not inherit root-owned build
  # or virtual-environment directories.
  mkdir -p -- "${KATAGO_ENV_ROOT}/state"
  require_command sudo
  exec sudo -E -- "$0" "$@"
fi

[[ -r /etc/os-release ]] || die "cannot identify operating system"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID}" == "ubuntu" ]] || die "only Ubuntu is currently supported (found ${ID})"
[[ "${VERSION_ID}" == "24.04" ]] || warn "baseline is Ubuntu 24.04; found ${VERSION_ID}"

mkdir -p -- "${KATAGO_ENV_ROOT}/state"

mapfile -t ubuntu_packages < <(sed -E '/^[[:space:]]*(#|$)/d' "${SCRIPT_DIR}/apt-packages.txt")

install_local_debs() {
  local deb_dir="${KATAGO_LOCAL_ARCHIVE}/apt"
  local -a debs=()
  local deb package_name requested
  [[ -d "${deb_dir}" ]] || return 0
  while IFS= read -r deb; do
    package_name="$(dpkg-deb -f "${deb}" Package 2>/dev/null || true)"
    requested=0
    for ubuntu_package in "${ubuntu_packages[@]}"; do
      if [[ "${package_name}" == "${ubuntu_package}" ]]; then
        requested=1
        break
      fi
    done
    case "${package_name}" in
      cuda-*|libcublas*|libcudnn*|libcufft*|libcurand*|libcusolver*|libcusparse*|libcufile*|libcuobjclient*|libnpp*|libnvjpeg*|libnvjitlink*|libnvvm*|libnvfatbin*|nsight-*|nvidia-*) requested=1 ;;
      tensorrt*|libnvinfer*|libnvonnx*) requested=0 ;;
    esac
    if [[ "${requested}" == "1" ]]; then
      debs+=("${deb}")
    fi
  done < <(find "${deb_dir}" -maxdepth 1 -type f -name '*.deb' -print | sort)
  if (( ${#debs[@]} > 0 )); then
    log "installing CUDA-scope local archive .deb files before using repositories"
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${debs[@]}"
  fi
}

log "refreshing Ubuntu package metadata"
apt-get update
install_local_debs
log "installing Ubuntu build dependencies"
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${ubuntu_packages[@]}"

ensure_cuda_repository() {
  if [[ -r /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]]; then
    return 0
  fi

  local keyring_deb=""
  local local_keyring
  local_keyring="$(find "${KATAGO_LOCAL_ARCHIVE}/apt" -maxdepth 1 -type f -name 'cuda-keyring_*.deb' -print -quit 2>/dev/null || true)"
  if [[ -n "${local_keyring}" ]]; then
    keyring_deb="${local_keyring}"
    log "using local CUDA repository keyring: ${keyring_deb}"
  else
    local temp_dir
    temp_dir="$(mktemp -d)"
    keyring_deb="${temp_dir}/cuda-keyring.deb"
    warn "CUDA repository keyring not present in local archive; downloading from NVIDIA"
    curl -fL --retry 3 \
      -o "${keyring_deb}" \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  fi
  dpkg -i "${keyring_deb}"
  apt-get update
}

ensure_cuda_repository

latest_versioned_package() {
  local prefix="$1"
  apt-cache pkgnames | grep -E "^${prefix}[0-9]" | sort -V | tail -n 1
}

# Unversioned NVIDIA meta packages intentionally resolve the current repository
# release on a fresh install. Every resolved package version is captured by the
# environment audit and by the prebuilt distribution manifest.
CUDA_TOOLKIT_PACKAGE="${KATAGO_CUDA_TOOLKIT_PACKAGE:-cuda-toolkit}"
CUDNN_PACKAGE="${KATAGO_CUDNN_PACKAGE:-libcudnn9-dev-cuda-13}"
NSIGHT_SYSTEMS_PACKAGE="${KATAGO_NSIGHT_SYSTEMS_PACKAGE:-$(latest_versioned_package nsight-systems-)}"
NSIGHT_COMPUTE_PACKAGE="${KATAGO_NSIGHT_COMPUTE_PACKAGE:-$(latest_versioned_package nsight-compute-)}"
[[ -n "${NSIGHT_SYSTEMS_PACKAGE}" ]] || die "could not resolve current Nsight Systems package"
[[ -n "${NSIGHT_COMPUTE_PACKAGE}" ]] || die "could not resolve current Nsight Compute package"

log "installing current CUDA toolkit, cuDNN, and profiling tools"
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  "${CUDA_TOOLKIT_PACKAGE}" \
  "${CUDNN_PACKAGE}" \
  "${NSIGHT_SYSTEMS_PACKAGE}" \
  "${NSIGHT_COMPUTE_PACKAGE}"

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  log "an operational NVIDIA driver is already present; leaving it unchanged"
elif [[ "${KATAGO_INSTALL_DRIVER:-1}" == "1" ]]; then
  log "installing NVIDIA open kernel driver package"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "${KATAGO_DRIVER_PACKAGE:-nvidia-open}"
  touch "${KATAGO_ENV_ROOT}/state/reboot-required"
  warn "NVIDIA driver was installed but is not active; reboot, then rerun setup.sh all"
  exit 75
else
  die "no operational NVIDIA driver and KATAGO_INSTALL_DRIVER=0"
fi

log "Ubuntu bootstrap complete"
