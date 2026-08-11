#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

pointer="${SCRIPT_DIR}/runtime-prefix.txt"
prefix="${SCRIPT_DIR}/runtime"
[[ -r "${pointer}" ]] && prefix="$(<"${pointer}")"
autotune="${SCRIPT_DIR}/autotune.py"
declare -a layout_args=()
if [[ -r "${SCRIPT_DIR}/final-migration/autotune/autotune.py" ]]; then
  prefix="${KATAGO_PREFIX:-${SCRIPT_DIR}/.final-migration-env}"
  autotune="${SCRIPT_DIR}/final-migration/autotune/autotune.py"
  layout_args=(--prefix "${prefix}" --repo "${SCRIPT_DIR}")
fi
[[ -x "${prefix}/venv/bin/python" ]] || {
  printf '[autotune] environment missing; run ./setup.sh first\n' >&2
  exit 1
}

exec "${prefix}/venv/bin/python" "${autotune}" "${layout_args[@]}" "$@"
