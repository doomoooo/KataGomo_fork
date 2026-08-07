#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
  cat <<'EOF'
Usage: setup.sh {install|audit|verify|build|package|deploy BUNDLE|all}

  install  Install Ubuntu/Python and build the latest dependency sources.
  audit    Record and validate tool/library/device versions.
  verify   Compile/import third-party dependency smokes.
  build    Build the KataGo CUDA backend.
  package  Package compiled wheels/binaries for distribution.
  deploy   Install a previously packaged bundle without source builds.
  all      Run install, audit, verify, and build in order.
EOF
}

install_environment() {
  "${SCRIPT_DIR}/bootstrap-ubuntu.sh"
  "${SCRIPT_DIR}/acquire-third-party.sh"
  "${SCRIPT_DIR}/install-python.sh"
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
  deploy)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    "${SCRIPT_DIR}/deploy-prebuilt.sh" "$2"
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
