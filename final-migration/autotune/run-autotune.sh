#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
device=0
allow_unlocked=0
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --device)
      (( index + 1 < ${#arguments[@]} )) || { printf 'missing --device value\n' >&2; exit 2; }
      device="${arguments[index+1]}"
      ;;
    --allow-unlocked) allow_unlocked=1 ;;
  esac
done

filtered=()
for argument in "$@"; do
  [[ "${argument}" == --allow-unlocked ]] || filtered+=("${argument}")
done

pointer="${SCRIPT_DIR}/runtime-prefix.txt"
prefix="${SCRIPT_DIR}/runtime"
[[ -r "${pointer}" ]] && prefix="$(<"${pointer}")"
[[ -x "${prefix}/venv/bin/python" ]] || {
  printf '[autotune] environment missing; run ./setup.sh first\n' >&2
  exit 1
}

if [[ "${AUTOTUNE_GPU_LOCK_HELD:-0}" != 1 && ${allow_unlocked} -eq 0 ]]; then
  gpu_lock="${SCRIPT_DIR}/gpu-lock"
  command -v gpu-lock >/dev/null 2>&1 && gpu_lock="$(command -v gpu-lock)"
  [[ -x "${gpu_lock}" ]] || {
    printf '[autotune] gpu-lock is unavailable; pass --allow-unlocked only on an exclusively assigned GPU\n' >&2
    exit 1
  }
  smi_device="$("${prefix}/venv/bin/python" "${SCRIPT_DIR}/detect_gpu.py" \
    --repo "${prefix}/repo" --device "${device}" --print-smi-index)"
  exec "${gpu_lock}" with --gpu "smi:${smi_device}" -- env AUTOTUNE_GPU_LOCK_HELD=1 \
    "${prefix}/venv/bin/python" "${SCRIPT_DIR}/autotune.py" "${filtered[@]}"
fi

exec "${prefix}/venv/bin/python" "${SCRIPT_DIR}/autotune.py" "${filtered[@]}"
