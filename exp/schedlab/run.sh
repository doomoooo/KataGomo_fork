#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/schedlab-trt-redesign"}
BIN=${BIN:-"${BUILD_DIR}/schedlab_run"}

if [[ ! -x "${BIN}" ]]; then
  echo "missing binary: ${BIN}" >&2
  echo "build it first, for example:" >&2
  echo "  cmake -S exp/schedlab -B build/schedlab-trt-redesign -DCMAKE_CXX_COMPILER=${REPO_ROOT}/.local/toolchains/llvm-22.1.1/bin/clang++" >&2
  echo "  cmake --build build/schedlab-trt-redesign -j 8" >&2
  exit 1
fi

for candidate in \
  "${HOME}/.katago/tensorrt/lib" \
  "${HOME}/.katago/tensorrt-cu13-1015/lib"
do
  if [[ ! -d "${candidate}" ]]; then
    continue
  fi
  if [[ ! -f "${candidate}/libnvinfer.so" ]] && [[ -z "$(compgen -G "${candidate}/libnvinfer.so*")" ]]; then
    continue
  fi
  if [[ ":${LD_LIBRARY_PATH:-}:" != *":${candidate}:"* ]]; then
    export LD_LIBRARY_PATH="${candidate}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  break
done

pick_idle_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  mapfile -t gpu_rows < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null)
  if [[ ${#gpu_rows[@]} -eq 0 ]]; then
    return 1
  fi

  declare -A live_compute_count=()
  declare -A any_compute_count=()
  while IFS=',' read -r raw_uuid raw_name; do
    local uuid name
    uuid=$(echo "${raw_uuid}" | xargs)
    name=$(echo "${raw_name}" | xargs)
    [[ -n "${uuid}" ]] || continue
    any_compute_count["${uuid}"]=$(( ${any_compute_count["${uuid}"]:-0} + 1 ))
    if [[ "${name}" != "[Not Found]" ]]; then
      live_compute_count["${uuid}"]=$(( ${live_compute_count["${uuid}"]:-0} + 1 ))
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader,nounits 2>/dev/null || true)

  local best_idle_index=""
  local best_idle_mem=999999999
  local best_any_index=""
  local best_any_live=999999999
  local best_any_count=999999999
  local best_any_mem=999999999

  for row in "${gpu_rows[@]}"; do
    IFS=',' read -r raw_index raw_uuid raw_mem <<<"${row}"
    local index uuid mem live_count any_count
    index=$(echo "${raw_index}" | xargs)
    uuid=$(echo "${raw_uuid}" | xargs)
    mem=$(echo "${raw_mem}" | xargs)
    live_count=${live_compute_count["${uuid}"]:-0}
    any_count=${any_compute_count["${uuid}"]:-0}

    [[ -n "${index}" && -n "${uuid}" && -n "${mem}" ]] || continue

    if (( live_count < best_any_live )) ||
       (( live_count == best_any_live && any_count < best_any_count )) ||
       (( live_count == best_any_live && any_count == best_any_count && mem < best_any_mem )); then
      best_any_live=${live_count}
      best_any_count=${any_count}
      best_any_mem=${mem}
      best_any_index=${index}
    fi

    if (( any_count != 0 )); then
      continue
    fi

    if (( mem < best_idle_mem )); then
      best_idle_mem=${mem}
      best_idle_index=${index}
    fi
  done

  if [[ -n "${best_idle_index}" ]]; then
    printf '%s\n' "${best_idle_index}"
    return 0
  fi
  if [[ -n "${best_any_index}" ]]; then
    printf '%s\n' "${best_any_index}"
    return 0
  fi
  return 1
}

has_cuda_devices_arg=0
for arg in "$@"; do
  if [[ "${arg}" == "--cuda-devices" ]] || [[ "${arg}" == --cuda-devices=* ]]; then
    has_cuda_devices_arg=1
    break
  fi
done

args=("$@")
if [[ ${has_cuda_devices_arg} -eq 0 && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if idle_gpu=$(pick_idle_gpu); then
    echo "run.sh: auto-selected GPU ${idle_gpu}" >&2
    args+=(--cuda-devices "${idle_gpu}")
  fi
fi

exec "${BIN}" "${args[@]}"
