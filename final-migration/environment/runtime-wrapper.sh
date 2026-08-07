#!/usr/bin/env bash

set -Eeuo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
runtime_lib="${bundle_root}/lib"
runtime_loader="${runtime_lib}/ld-linux-x86-64.so.2"
[[ -x "${runtime_loader}" ]] || {
  printf '[katago-tar] ERROR: bundled ELF loader is missing: %s\n' "${runtime_loader}" >&2
  exit 1
}

# Invoke the bundled loader explicitly. This keeps the tar independent of the
# target distribution's glibc version while still loading libcuda from the
# target's driver installation. All paths stay relative to this extracted tar.
exec "${runtime_loader}" \
  --library-path "${runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${bundle_root}/libexec/katago" "$@"
