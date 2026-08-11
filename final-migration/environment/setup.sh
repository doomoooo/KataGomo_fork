#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: setup.sh {install|audit|verify|build|package|extract ARCHIVE PREFIX|deploy BUNDLE [PYTHON_ENV]|all}

  install  Install pinned Python and build latest dependencies below the managed root.
  audit    Record and validate tool/library/device versions.
  verify   Compile/import third-party dependency smokes.
  build    Build the KataGo CUDA backend.
  package  Package a non-invasive tar containing the compiled runtime.
  extract  Verify and extract a tar into one empty, isolated prefix.
  deploy   Optionally install archived Python tools below an extracted bundle.
  all      Run install, audit, verify, and build in order.
EOF
}

install_environment() {
  "${SCRIPT_DIR}/acquire-third-party.sh"
  "${SCRIPT_DIR}/install-python.sh"
  "${SCRIPT_DIR}/acquire-nvidia-toolchain.sh"
  "${SCRIPT_DIR}/build-third-party.sh"
}

command_name="${1:-}"
case "${command_name}" in
  install)
    install_environment
    ;;
  audit)
    "${SCRIPT_DIR}/audit-environment.sh"
    ;;
  verify)
    "${SCRIPT_DIR}/verify-third-party.sh"
    ;;
  build)
    "${SCRIPT_DIR}/build-matrix.sh"
    ;;
  package)
    "${SCRIPT_DIR}/package-distribution.sh"
    ;;
  extract)
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    "${SCRIPT_DIR}/install-tar.sh" "$2" "$3"
    ;;
  deploy)
    [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
    "${SCRIPT_DIR}/deploy-prebuilt.sh" "$2" "${3:-$2/python-env}"
    ;;
  all)
    install_environment
    "${SCRIPT_DIR}/audit-environment.sh"
    "${SCRIPT_DIR}/verify-third-party.sh"
    "${SCRIPT_DIR}/build-matrix.sh"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
