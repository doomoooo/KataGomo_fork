#!/usr/bin/env bash

# Source this file in an interactive shell:
#   source final-migration/environment/activate-sm103.sh
#
# It deliberately does not import a CUDA-facing Python module or query a GPU.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'source this script instead of executing it\n' >&2
  exit 2
fi

KATAGO_SM103_ACTIVATE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${KATAGO_SM103_ACTIVATE_DIR}/lib/common.sh"

activate_venv
activate_toolchain

export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="${KATAGO_ENV_ROOT}/cache"
export TORCH_EXTENSIONS_DIR="${KATAGO_ENV_ROOT}/cache/torch-extensions"
export CUDA_CACHE_PATH="${KATAGO_ENV_ROOT}/cache/cuda"
export TRITON_CACHE_DIR="${KATAGO_ENV_ROOT}/cache/triton"
export TRITON_PTXAS_PATH="${KATAGO_CUDA_ROOT}/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="${KATAGO_CUDA_ROOT}/bin/ptxas"
export CUTE_DSL_ARCH="sm_103a"
export CUTE_DSL_CACHE_DIR="${KATAGO_ENV_ROOT}/cache/cute-dsl-sm103"
export CUTE_DSL_PTXAS_PATH="${KATAGO_CUDA_ROOT}/bin/ptxas"
export FLASHINFER_WORKSPACE_BASE="${KATAGO_ENV_ROOT}/cache"
export FLASHINFER_CUBIN_DIR="${KATAGO_ENV_ROOT}/cache/flashinfer/cubins"
export FLASHINFER_CUDA_ARCH_LIST="10.3a"
export FLASHINFER_NVCC="${KATAGO_CUDA_ROOT}/bin/nvcc"
export FLASHINFER_NO_DOWNLOAD=1

mkdir -p -- \
  "${TORCH_EXTENSIONS_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${TRITON_CACHE_DIR}" \
  "${CUTE_DSL_CACHE_DIR}" \
  "${FLASHINFER_CUBIN_DIR}"

unset KATAGO_SM103_ACTIVATE_DIR

printf 'KataGo SM103 environment active\n'
printf 'python=%s\n' "$(command -v python)"
printf 'cuda_home=%s\n' "${CUDA_HOME}"
printf 'target=%s flashinfer_arch=%s\n' \
  "${CUTE_DSL_ARCH}" "${FLASHINFER_CUDA_ARCH_LIST}"
