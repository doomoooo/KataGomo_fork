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

  if [[ "${new_checkout}" == "1" || "${KATAGO_REFRESH_SOURCES:-0}" == "1" ]]; then
    github_fallback_warning "${name} source"
    if ! git -C "${target}" fetch --depth=1 origin "${requested_ref}"; then
      if [[ "${new_checkout}" == "1" ]]; then
        die "could not acquire ${name}; populate archive/git or configure a GitHub proxy"
      fi
      die "could not refresh ${name}; rerun without KATAGO_REFRESH_SOURCES=1 to use the clean cached source"
    fi
    git -C "${target}" checkout --detach --force FETCH_HEAD
  else
    log "reusing clean cached ${name}; set KATAGO_REFRESH_SOURCES=1 to refresh upstream"
  fi

  if [[ "${submodules}" != "none" && -f "${target}/.gitmodules" ]]; then
    local -a submodule_args submodule_paths
    local needs_submodule_update=1
    submodule_args=(--init --depth=1)
    if [[ "${submodules}" == "recursive" ]]; then
      submodule_args+=(--recursive)
      git -C "${target}" submodule sync --recursive
    else
      IFS=',' read -r -a submodule_paths <<< "${submodules}"
      submodule_args+=(-- "${submodule_paths[@]}")
      git -C "${target}" submodule sync -- "${submodule_paths[@]}"
      if ! git -C "${target}" submodule status -- "${submodule_paths[@]}" \
        | grep -qE '^[-+U]'; then
        needs_submodule_update=0
      fi
    fi
    if (( needs_submodule_update == 1 )); then
      github_fallback_warning "${name} required submodule(s)"
      git -C "${target}" submodule update "${submodule_args[@]}"
    else
      log "reusing initialized ${name} required submodule(s)"
    fi

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
