#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

pointer="${SCRIPT_DIR}/runtime-prefix.txt"
prefix="${SCRIPT_DIR}/runtime"
[[ -r "${pointer}" ]] && prefix="$(<"${pointer}")"
[[ -x "${prefix}/venv/bin/python" ]] || {
  printf '[autotune] environment missing; run ./setup.sh first\n' >&2
  exit 1
}

exec "${prefix}/venv/bin/python" "${SCRIPT_DIR}/autotune.py" "$@"
