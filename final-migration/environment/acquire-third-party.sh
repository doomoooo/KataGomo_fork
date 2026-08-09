#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

require_command git
mkdir -p -- "${KATAGO_THIRD_PARTY_ROOT}"
assert_safe_managed_path "${KATAGO_THIRD_PARTY_ROOT}"
ensure_record_root

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
manifest="${KATAGO_RECORD_ROOT}/source-sync-${timestamp}.tsv"
printf 'name\ttier\turl\trequested_ref\tresolved_commit\tdescribe\tsubmodule_commits\n' > "${manifest}"

sync_latest() {
  local name="$1" tier="$2" url="$3" requested_ref="$4" submodules="$5"
  local target="${KATAGO_THIRD_PARTY_ROOT}/${name}"
  local bundle="${KATAGO_LOCAL_ARCHIVE}/git/${name}.bundle"
  local resolved describe submodule_commits="-"
  local new_checkout=0

  if [[ ! -e "${target}" ]]; then
    new_checkout=1
    if [[ -f "${bundle}" ]]; then
      log "seeding ${name} from local Git bundle"
      git clone --no-checkout "${bundle}" "${target}"
      git -C "${target}" remote set-url origin "${url}"
    else
      github_fallback_warning "latest ${name} source"
      git clone --filter=blob:none --no-checkout "${url}" "${target}"
    fi
  elif ! find "${target}" -mindepth 1 -maxdepth 1 ! -name .git -print -quit | grep -q .; then
    # Recover an interrupted --no-checkout clone before treating missing
    # worktree files as user edits.
    new_checkout=1
  fi

  [[ -d "${target}/.git" ]] || die "source target is not a Git checkout: ${target}"
  if [[ "${new_checkout}" == "0" ]]; then
    [[ -z "$(git -C "${target}" status --porcelain)" ]] || die "managed source checkout is dirty: ${target}"
  fi

  github_fallback_warning "latest ${name} commit check"
  if ! git -C "${target}" fetch --depth=1 origin "${requested_ref}"; then
    if [[ "${KATAGO_ALLOW_STALE_SOURCE:-0}" != "1" ]]; then
      die "could not fetch latest ${name}; configure a GitHub proxy or set KATAGO_ALLOW_STALE_SOURCE=1 to use a cached checkout explicitly"
    fi
    warn "using stale cached ${name} because KATAGO_ALLOW_STALE_SOURCE=1"
  else
    git -C "${target}" checkout --detach --force FETCH_HEAD
  fi

  if [[ "${submodules}" == "recursive" && -f "${target}/.gitmodules" ]]; then
    github_fallback_warning "${name} submodules"
    git -C "${target}" submodule sync --recursive
    git -C "${target}" submodule update --init --recursive --depth=1

    submodule_commits="$(git -C "${target}" submodule status --recursive | sed -E 's/^[ +U-]//' | tr '\n' ',' | sed 's/,$//')"
  fi

  [[ -z "$(git -C "${target}" status --porcelain)" ]] || die "source checkout remained dirty after sync: ${target}"

  resolved="$(git -C "${target}" rev-parse HEAD)"
  describe="$(git -C "${target}" describe --tags --always 2>/dev/null || printf unknown)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${tier}" "${url}" "${requested_ref}" "${resolved}" "${describe}" "${submodule_commits}" \
    >> "${manifest}"
  log "synced ${name} at ${resolved} (${describe})"
}

while IFS=$'\t' read -r name tier url requested_ref builder distribution submodules; do
  [[ -n "${name}" && "${name}" != \#* ]] || continue
  if [[ "${tier}" == "research" && "${KATAGO_INCLUDE_RESEARCH:-0}" != "1" ]]; then
    log "skipping research-only source ${name}; set KATAGO_INCLUDE_RESEARCH=1 to include it"
    continue
  fi
  sync_latest "${name}" "${tier}" "${url}" "${requested_ref}" "${submodules}"
done < "${SCRIPT_DIR}/third-party.lock.tsv"

mkdir -p -- "${KATAGO_ENV_ROOT}/state"
cp -- "${manifest}" "${KATAGO_ENV_ROOT}/state/source-manifest.tsv"
log "latest source sync complete; manifest=${manifest}"
