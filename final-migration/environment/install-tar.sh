#!/usr/bin/env bash

set -Eeuo pipefail

die() {
  printf '[katago-tar] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 2 ]] || die "usage: install-tar.sh ARCHIVE.tar PREFIX"
archive="$(readlink -e -- "$1")"
prefix="$(readlink -m -- "$2")"
[[ -f "${archive}" ]] || die "tar archive does not exist: $1"
[[ "${archive}" == *.tar ]] || die "expected an uncompressed .tar archive"
case "${prefix}" in
  /|/bin|/boot|/dev|/etc|/lib|/lib64|/proc|/run|/sbin|/sys|/usr|/var)
    die "refusing to install into a system root: ${prefix}"
    ;;
esac

checksum_file="${archive}.sha256"
[[ -r "${checksum_file}" ]] || die "archive checksum file is missing: ${checksum_file}"
(cd -- "$(dirname -- "${archive}")" && sha256sum --check "$(basename -- "${checksum_file}")")

if [[ -e "${prefix}" ]] && find "${prefix}" -mindepth 1 -print -quit | grep -q .; then
  die "installation prefix is not empty: ${prefix}"
fi
mkdir -p -- "${prefix}"

top_level="$(tar -tf "${archive}" | sed -n '1{s:/*$::;p;}')"
[[ -n "${top_level}" && "${top_level}" != /* && "${top_level}" != *'..'* ]] \
  || die "archive has an unsafe top-level path"
if tar -tf "${archive}" | awk -F/ -v top="${top_level}" \
  '$1 != top || $0 ~ /(^|\/)\.\.?(\/|$)/ {exit 1}'; then
  :
else
  die "archive contains more than one top-level path"
fi

tar --extract --file "${archive}" --directory "${prefix}" \
  --strip-components=1 --no-same-owner --no-same-permissions
"${prefix}/verify.sh"
printf '[katago-tar] installed only below %s\n' "${prefix}"
