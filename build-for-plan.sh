#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

prefix="${KATAGO_PREFIX:-}"
if [[ -z "${prefix}" && -r "${SCRIPT_DIR}/runtime-prefix.txt" ]]; then
  prefix="$(<"${SCRIPT_DIR}/runtime-prefix.txt")"
fi
if [[ -z "${prefix}" && -d "${SCRIPT_DIR}/repo" ]]; then
  prefix="${SCRIPT_DIR}"
fi
if [[ -z "${prefix}" && -d "${SCRIPT_DIR}/runtime" ]]; then
  prefix="${SCRIPT_DIR}/runtime"
fi
if [[ -z "${prefix}" && -d "${SCRIPT_DIR}/.final-migration-env" ]]; then
  prefix="${SCRIPT_DIR}/.final-migration-env"
fi
[[ -n "${prefix}" && -x "${prefix}/venv/bin/python" ]] || {
  printf '[build-for-plan] ERROR: environment missing; run ./setup.sh first\n' >&2
  exit 1
}

repo="${prefix}/repo"
[[ -r "${repo}/python/cuda_tactic_workflow.py" ]] || repo="${SCRIPT_DIR}"
helper="${SCRIPT_DIR}/build_for_plan.py"
autotune="${SCRIPT_DIR}/autotune.py"
if [[ ! -r "${helper}" ]]; then
  helper="${SCRIPT_DIR}/final-migration/autotune/build_for_plan.py"
fi
if [[ ! -r "${autotune}" ]]; then
  autotune="${SCRIPT_DIR}/final-migration/autotune/autotune.py"
fi
[[ -r "${helper}" && -r "${autotune}" ]] || {
  printf '[build-for-plan] ERROR: build-only workflow files are missing\n' >&2
  exit 1
}

declare -a command=(
  "${prefix}/venv/bin/python" "${helper}"
  --prefix "${prefix}" --repo "${repo}" --autotune "${autotune}"
)
for root in "${SCRIPT_DIR}/plans" "${repo}/final-migration/plans"; do
  if [[ -d "${root}" ]]; then
    command+=(--plans-root "${root}")
    break
  fi
done
exec "${command[@]}" "$@"
