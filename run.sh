#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_NAME="b11c768h12nbt3tflrs-fson-silu.bin.gz"

usage() {
  cat <<'EOF'
Usage: ./run.sh [OPTIONS] [-- KATAGO_GTP_ARGS...]

Start KataGo GTP with the certified plan for one visible CUDA device.

Options:
  --device N          CUDA Runtime device ordinal (default: 0)
  --model PATH        model file (default: bundled 70M model)
  --config PATH       GTP config (default: repository gtp_example.cfg)
  --plan PATH         certified tactic plan (default: auto-detect)
  --katago PATH       plan-matching executable (default: auto-detect)
  --search-threads N  search threads (default: batch * (streams + 1) + 8)
  --print-command     validate and print the final command without starting GTP
  -h, --help          show this help

The same values may be supplied through KATAGO_DEVICE, KATAGO_MODEL,
KATAGO_CONFIG, KATAGO_PLAN, KATAGO_BIN, and KATAGO_SEARCH_THREADS.
EOF
}

die() {
  printf '[katago-run] ERROR: %s\n' "$*" >&2
  exit 1
}

device="${KATAGO_DEVICE:-0}"
model="${KATAGO_MODEL:-}"
config="${KATAGO_CONFIG:-}"
plan="${KATAGO_PLAN:-}"
katago="${KATAGO_BIN:-}"
search_threads="${KATAGO_SEARCH_THREADS:-}"
print_command=0
declare -a gtp_args=()

while (( $# > 0 )); do
  case "$1" in
    --device) [[ $# -ge 2 ]] || die "--device requires a value"; device="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || die "--model requires a path"; model="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config requires a path"; config="$2"; shift 2 ;;
    --plan) [[ $# -ge 2 ]] || die "--plan requires a path"; plan="$2"; shift 2 ;;
    --katago) [[ $# -ge 2 ]] || die "--katago requires a path"; katago="$2"; shift 2 ;;
    --search-threads)
      [[ $# -ge 2 ]] || die "--search-threads requires a value"
      search_threads="$2"
      shift 2
      ;;
    --print-command) print_command=1; shift ;;
    --) shift; gtp_args+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) gtp_args+=("$1"); shift ;;
  esac
done

[[ "${device}" =~ ^[0-9]+$ ]] || die "device ordinal must be a non-negative integer"

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
if [[ -z "${prefix}" && "$(basename -- "${SCRIPT_DIR}")" == repo ]]; then
  parent="$(dirname -- "${SCRIPT_DIR}")"
  if [[ -d "${parent}/assets" && -d "${parent}/results" ]]; then
    prefix="${parent}"
  fi
fi

repo=""
for candidate in "${SCRIPT_DIR}" "${prefix:+${prefix}/repo}"; do
  if [[ -n "${candidate}" && -r "${candidate}/python/cuda_tactic_workflow.py" ]]; then
    repo="$(cd -- "${candidate}" && pwd -P)"
    break
  fi
done
[[ -n "${repo}" ]] || die "repository runtime files are missing; run ./setup.sh first"

python="${KATAGO_PYTHON:-}"
if [[ -z "${python}" ]]; then
  for candidate in \
    "${prefix:+${prefix}/venv/bin/python}" \
    "${SCRIPT_DIR}/.final-migration-env/venv/bin/python"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      python="${candidate}"
      break
    fi
  done
fi
if [[ -z "${python}" ]]; then
  python="$(command -v python3 || true)"
fi
[[ -x "${python}" ]] || die "Python is missing; run ./setup.sh first"

if [[ -z "${model}" ]]; then
  for candidate in \
    "${prefix:+${prefix}/assets/${MODEL_NAME}}" \
    "${SCRIPT_DIR}/assets/${MODEL_NAME}" \
    "${SCRIPT_DIR}/models/${MODEL_NAME}"; do
    if [[ -n "${candidate}" && -r "${candidate}" ]]; then
      model="${candidate}"
      break
    fi
  done
fi
[[ -r "${model}" ]] || die "model not found; pass --model PATH or set KATAGO_MODEL"
model="$(readlink -m -- "${model}")"

if [[ -z "${config}" ]]; then
  config="${repo}/cpp/configs/gtp_example.cfg"
fi
[[ -r "${config}" ]] || die "GTP config not found: ${config}"
config="$(readlink -m -- "${config}")"

workflow="${repo}/python/cuda_tactic_workflow.py"
declare -a plan_roots=()
for candidate in \
  "${prefix:+${prefix}/results}" \
  "${SCRIPT_DIR}/final-migration/plans" \
  "${SCRIPT_DIR}/plans" \
  "${prefix:+${prefix}/plans}" \
  "${prefix:+${prefix}/repo/final-migration/plans}"; do
  [[ -n "${candidate}" && -d "${candidate}" ]] && plan_roots+=("${candidate}")
done

validate_plan() {
  "${python}" "${workflow}" validate \
    --plan "$1" --model "${model}" --device "${device}" >/dev/null 2>&1
}

if [[ -n "${plan}" ]]; then
  plan="$(readlink -m -- "${plan}")"
  [[ -r "${plan}" ]] || die "plan not found: ${plan}"
  validate_plan "${plan}" || die "plan is incompatible with device ${device} or model ${model}"
else
  declare -a matching_plans=()
  declare -A seen_plans=()
  declare -A seen_plan_hashes=()
  for root in "${plan_roots[@]}"; do
    while IFS= read -r candidate; do
      resolved="$(readlink -m -- "${candidate}")"
      [[ -z "${seen_plans[${resolved}]:-}" ]] || continue
      seen_plans["${resolved}"]=1
      [[ -r "$(dirname -- "${resolved}")/SHA256SUMS" ]] || continue
      candidate_plan_sha="$(sha256sum "${resolved}" | awk '{print $1}')"
      [[ -z "${seen_plan_hashes[${candidate_plan_sha}]:-}" ]] || continue
      seen_plan_hashes["${candidate_plan_sha}"]=1
      if validate_plan "${resolved}"; then
        matching_plans+=("${resolved}")
      fi
    done < <(find "${root}" -type f -name best-tactic-plan.json -print | sort)
  done
  (( ${#matching_plans[@]} > 0 )) \
    || die "no certified plan matches CUDA device ${device} and the bundled model"
  if (( ${#matching_plans[@]} == 1 )); then
    plan="${matching_plans[0]}"
  else
    # A freshly generated plan lives beside its measured binary. Prefer that
    # self-contained pair over a registry plan that needs a packaged binary.
    best_metric=""
    for candidate in "${matching_plans[@]}"; do
      candidate_binary="$(dirname -- "${candidate}")/build/katago"
      [[ -x "${candidate_binary}" ]] || continue
      mapfile -t candidate_fields < <("${python}" -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
b = str(int(p["batches"][0]))
print(p["final_joint"][b]["binary_sha256"])
print(float(p["final_joint"][b]["stable_long_nn_evals_per_sec"]))
' "${candidate}")
      candidate_hash="${candidate_fields[0]:-}"
      candidate_metric="${candidate_fields[1]:-}"
      if [[ "$(sha256sum "${candidate_binary}" | awk '{print $1}')" == "${candidate_hash}" ]] && \
         { [[ -z "${best_metric}" ]] || awk -v a="${candidate_metric}" -v b="${best_metric}" 'BEGIN { exit !(a > b) }'; }; then
        plan="${candidate}"
        best_metric="${candidate_metric}"
      fi
    done
    [[ -n "${plan}" ]] \
      || die "multiple plans match device ${device}; select one with --plan"
  fi
fi

checksum_file="$(dirname -- "${plan}")/SHA256SUMS"
[[ -r "${checksum_file}" ]] || die "plan checksum manifest is missing: ${checksum_file}"
(cd -- "$(dirname -- "${plan}")" && sha256sum --check --quiet SHA256SUMS) \
  || die "plan checksum verification failed"

mapfile -t plan_fields < <("${python}" -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if len(p["batches"]) != 1:
    raise SystemExit("run.sh requires a single-batch production plan")
batch = int(p["batches"][0])
joint = p["final_joint"][str(batch)]
print(batch)
print(int(p["target"]["streams"]))
print(joint["binary_sha256"])
print(p["plan_id"])
print(p["target"]["gpu_class"])
' "${plan}")
(( ${#plan_fields[@]} == 5 )) || die "could not read plan runtime fields"
batch="${plan_fields[0]}"
streams="${plan_fields[1]}"
binary_sha256="${plan_fields[2]}"
plan_id="${plan_fields[3]}"
gpu_class="${plan_fields[4]}"

binary_identity_path() {
  local candidate="$1" bundle_root
  bundle_root="$(cd -- "$(dirname -- "${candidate}")/.." 2>/dev/null && pwd -P || true)"
  if [[ "$(basename -- "$(dirname -- "${candidate}")")" == bin && \
        -x "${bundle_root}/libexec/katago" ]]; then
    printf '%s\n' "${bundle_root}/libexec/katago"
  else
    printf '%s\n' "${candidate}"
  fi
}

binary_matches_plan() {
  local candidate="$1" identity
  [[ -x "${candidate}" ]] || return 1
  identity="$(binary_identity_path "${candidate}")"
  [[ "$(sha256sum "${identity}" | awk '{print $1}')" == "${binary_sha256}" ]]
}

plan_apply_matches() {
  "${python}" -c '
import json, sys
a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))
keys = ("architecture", "gpu_class", "model_sha256", "fixed_board", "precision", "streams")
if any(a["target"].get(k) != b["target"].get(k) for k in keys):
    raise SystemExit(1)
if a.get("batches") != b.get("batches"):
    raise SystemExit(1)
if a.get("apply", {}).get("per_batch_tactic_overrides") != b.get("apply", {}).get("per_batch_tactic_overrides"):
    raise SystemExit(1)
' "$1" "$2" >/dev/null
}

plan_build_binary() {
  "${python}" -c '
import hashlib, json, pathlib, sys

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest_path = pathlib.Path(sys.argv[1]).resolve()
plan_path = pathlib.Path(sys.argv[2]).resolve()
manifest = json.load(manifest_path.open(encoding="utf-8"))
plan = json.load(plan_path.open(encoding="utf-8"))
if manifest.get("schema") != 1 or manifest.get("kind") != "cuda-plan-build":
    raise SystemExit(1)
if manifest.get("plan_id") != plan.get("plan_id"):
    raise SystemExit(1)
if manifest.get("plan_sha256") != plan.get("plan_sha256"):
    raise SystemExit(1)
if manifest.get("source_plan_file_sha256") != sha256(plan_path):
    raise SystemExit(1)
if manifest.get("batch") != int(plan["batches"][0]):
    raise SystemExit(1)
target = plan["target"]
for key in ("architecture", "gpu_class", "streams"):
    if manifest.get(key) != target.get(key):
        raise SystemExit(1)
root = manifest_path.parent
paths = {}
for key, hash_key in (
    ("binary", "binary_sha256"),
    ("artifact_bundle", "artifact_bundle_sha256"),
    ("space", "space_sha256"),
):
    relative = pathlib.Path(manifest[key])
    if relative.is_absolute():
        raise SystemExit(1)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise SystemExit(1)
    if not path.is_file() or sha256(path) != manifest.get(hash_key):
        raise SystemExit(1)
    paths[key] = path
print(paths["binary"])
' "$1" "$2" 2>/dev/null
}

if [[ -n "${katago}" ]]; then
  katago="$(readlink -m -- "${katago}")"
  binary_matches_plan "${katago}" \
    || die "--katago executable does not match the plan binary SHA-256"
else
  if [[ -n "${prefix}" && -d "${prefix}/results" ]]; then
    while IFS= read -r build_manifest; do
      candidate="$(plan_build_binary "${build_manifest}" "${plan}" || true)"
      [[ -x "${candidate}" ]] || continue
      katago="${candidate}"
      printf '[katago-run] using locally compiled build-only binary for %s\n' \
        "${plan_id}" >&2
      break
    done < <(find "${prefix}/results" -type f -name plan-build.json -print | sort)
  fi
  declare -a binary_candidates=(
    "$(dirname -- "${plan}")/build/katago"
    "${SCRIPT_DIR}/katago"
    "${SCRIPT_DIR}/bin/katago"
    "${prefix:+${prefix}/bin/katago}"
    "${prefix:+${prefix}/katago-builds/cuda/katago}"
  )
  if [[ -n "${prefix}" && -d "${prefix}/results" ]]; then
    while IFS= read -r candidate; do
      binary_candidates+=("${candidate}")
    done < <(find "${prefix}/results" -type f -path '*/build/katago' -print | sort)
  fi
  if [[ -z "${katago}" ]]; then
    for candidate in "${binary_candidates[@]}"; do
      if [[ -n "${candidate}" ]] && binary_matches_plan "${candidate}"; then
        katago="$(readlink -m -- "${candidate}")"
        break
      fi
    done
  fi
  if [[ -z "${katago}" && -n "${prefix}" && -d "${prefix}/results" ]]; then
    while IFS= read -r candidate_plan; do
      candidate="$(dirname -- "${candidate_plan}")/build/katago"
      [[ -x "${candidate}" ]] || continue
      if plan_apply_matches "${plan}" "${candidate_plan}"; then
        katago="$(readlink -m -- "${candidate}")"
        printf '[katago-run] WARNING: using a plan-equivalent rebuilt binary; measured binary SHA-256 is unavailable\n' >&2
        break
      fi
    done < <(find "${prefix}/results" -type f -name best-tactic-plan.json -print | sort)
  fi
fi
[[ -n "${katago}" ]] || die \
  "no exact or plan-equivalent executable is available; run autotune first"

if [[ -z "${search_threads}" ]]; then
  search_threads=$((batch * (streams + 1) + 8))
fi
[[ "${search_threads}" =~ ^[1-9][0-9]*$ ]] || die "search threads must be positive"

overrides="cudaTacticPlanFile=${plan},cudaTacticPlanBatch=${batch},nnMaxBatchSize=${batch},numNNServerThreadsPerModel=${streams},nnBatchAwareDispatch=true,cudaWarmupOnlyMaxBatchSize=true,cudaAsyncInferPipeline=true,cudaEventPipelineUseGraph=false,requireMaxBoardSize=true,numSearchThreads=${search_threads}"
for (( lane=0; lane<streams; lane++ )); do
  overrides+=",cudaDeviceToUseThread${lane}=${device}"
done

declare -a command=(
  "${katago}" gtp -model "${model}" -config "${config}"
  -override-config "${overrides}" "${gtp_args[@]}"
)
printf '[katago-run] device=%s gpu=%s plan=%s B%s streams=%s searchThreads=%s\n' \
  "${device}" "${gpu_class}" "${plan_id}" "${batch}" "${streams}" \
  "${search_threads}" >&2
if (( print_command )); then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
