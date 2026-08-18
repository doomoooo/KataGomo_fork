#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
# shellcheck source=../environment/python-runtime.lock.sh
source "${SCRIPT_DIR}/../environment/python-runtime.lock.sh"
# shellcheck source=corpus.lock.sh
source "${SCRIPT_DIR}/corpus.lock.sh"
ENV_ROOT="${KATAGO_ENV_ROOT:-${REPO_ROOT}/.final-migration-env}"
SOURCE_ROOT="${AUTOTUNE_SOURCE_ROOT:-${ENV_ROOT}/third_party}"
OUTPUT_ROOT="${AUTOTUNE_OUTPUT_ROOT:-${ENV_ROOT}/autotune-distributions}"
FLASH_CUTLASS_ROOT="${AUTOTUNE_FLASH_CUTLASS_ROOT:-${SOURCE_ROOT}/flash-attention/csrc/cutlass}"
ZLIB_ROOT="${AUTOTUNE_ZLIB_ROOT:-${SOURCE_ROOT}/zlib}"
MODEL="${AUTOTUNE_MODEL:-/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz}"
CORPUS="${AUTOTUNE_CORPUS:-}"
CORPUS_MANIFEST="${AUTOTUNE_CORPUS_MANIFEST:-}"
CORPUS_PYTHON="${AUTOTUNE_CORPUS_PYTHON:-${ENV_ROOT}/venv/bin/python}"
CORPUS_OUTPUT_ROOT="${AUTOTUNE_CORPUS_OUTPUT_ROOT:-/workspace/trainingdata/accuracy}"
TRAINING_DATA_CACHE="${AUTOTUNE_TRAINING_DATA_CACHE:-/workspace/trainingdata}"
PYTHON_ARCHIVE="${AUTOTUNE_PYTHON_ARCHIVE:-${ENV_ROOT}/downloads/${KATAGO_PYTHON_RUNTIME_ARCHIVE}}"
PYPI_MIRROR="${KATAGO_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_EXTRA_INDEX_URLS="${KATAGO_PIP_EXTRA_INDEX_URLS:-https://pypi.nvidia.com https://download.pytorch.org/whl/cu132}"
PIP_CACHE_DIR="${KATAGO_PIP_CACHE_DIR:-${ENV_ROOT}/cache/pip}"
DEFAULT_SEED_WHEELS="${ENV_ROOT}/distributions/20260807T205459Z/wheels:${ENV_ROOT}/autotune-wheel-seed"
SEED_WHEELS="${AUTOTUNE_SEED_WHEELS:-${DEFAULT_SEED_WHEELS}}"
CORPUS_RESULT="${ENV_ROOT}/accuracy-corpus/current.json"

log() { printf '[autotune-package] %s\n' "$*"; }
die() { printf '[autotune-package] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "required command missing: $1"; }
for command_name in curl find git gzip python3 sha256sum tar; do need "${command_name}"; done

[[ -z "${CORPUS}" && -z "${CORPUS_MANIFEST}" ]] || \
  [[ -n "${CORPUS}" && -n "${CORPUS_MANIFEST}" ]] \
  || die "AUTOTUNE_CORPUS and AUTOTUNE_CORPUS_MANIFEST must be set together"
if [[ -z "${CORPUS}" ]]; then
  for corpus_state in \
    "${ENV_ROOT}/state/accuracy-corpus.json" \
    "${ENV_ROOT}/accuracy-corpus/current.json"; do
    [[ -r "${corpus_state}" ]] || continue
    mapfile -t frozen_pair < <("${CORPUS_PYTHON}" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print(d["corpus"]); print(d["manifest"])' \
      "${corpus_state}")
    if [[ -r "${frozen_pair[0]}" && -r "${frozen_pair[1]}" ]]; then
      CORPUS="${frozen_pair[0]}"
      CORPUS_MANIFEST="${frozen_pair[1]}"
      break
    fi
  done
fi
if [[ -z "${CORPUS}" ]]; then
  [[ -x "${CORPUS_PYTHON}" ]] || CORPUS_PYTHON="$(command -v python3)"
  "${CORPUS_PYTHON}" -c 'import numpy' \
    || die "accuracy-corpus Python lacks NumPy; run the environment setup first"
  log "reconstructing the frozen 8192-row correctness corpus"
  "${CORPUS_PYTHON}" "${SCRIPT_DIR}/prepare_accuracy_corpus.py" \
    --repo "${REPO_ROOT}" --python "${CORPUS_PYTHON}" \
    --output-dir "${CORPUS_OUTPUT_ROOT}" \
    --work-dir "${ENV_ROOT}/accuracy-corpus" \
    --archive-cache-dir "${TRAINING_DATA_CACHE}" \
    --archive-url "${AUTOTUNE_CORPUS_ARCHIVE_URL}" \
    --archive-sha256 "${AUTOTUNE_CORPUS_ARCHIVE_SHA256}" \
    --result-json "${CORPUS_RESULT}"
  CORPUS="$("${CORPUS_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus"])' "${CORPUS_RESULT}")"
  CORPUS_MANIFEST="$("${CORPUS_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest"])' "${CORPUS_RESULT}")"
else
  log "validating the frozen 8192-row corpus and source identity"
  "${CORPUS_PYTHON}" "${SCRIPT_DIR}/prepare_accuracy_corpus.py" \
    --repo "${REPO_ROOT}" --python "${CORPUS_PYTHON}" \
    --output-dir "${CORPUS_OUTPUT_ROOT}" \
    --work-dir "${ENV_ROOT}/accuracy-corpus" \
    --corpus "${CORPUS}" --manifest "${CORPUS_MANIFEST}" \
    --archive-sha256 "${AUTOTUNE_CORPUS_ARCHIVE_SHA256}" \
    --result-json "${CORPUS_RESULT}"
fi
[[ "$(sha256sum "${CORPUS}" | awk '{print $1}')" == "${AUTOTUNE_CORPUS_SHA256}" ]] \
  || die "corpus differs from the maintained correctness gate"
locked_source_archive="$("${CORPUS_PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_archive"])' \
  "${CORPUS_MANIFEST}")"
[[ "${locked_source_archive}" == "${AUTOTUNE_CORPUS_ARCHIVE}" ]] \
  || die "corpus came from an unmaintained training archive: ${locked_source_archive}"

[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] \
  || die "package only from a clean committed final-migration tree"
for path in "${FLASH_CUTLASS_ROOT}" "${ZLIB_ROOT}" "${MODEL}" "${CORPUS}" "${CORPUS_MANIFEST}"; do
  [[ -e "${path}" ]] || die "required payload input missing: ${path}"
done
mkdir -p -- "${OUTPUT_ROOT}" "$(dirname -- "${PYTHON_ARCHIVE}")" "${PIP_CACHE_DIR}"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URLS}"
export PIP_CACHE_DIR

if [[ ! -r "${PYTHON_ARCHIVE}" ]]; then
  log "downloading the pinned Python source-independent runtime for release construction"
  curl --fail --location --retry 3 \
    --output "${PYTHON_ARCHIVE}.partial" "${KATAGO_PYTHON_RUNTIME_URL}"
  mv -- "${PYTHON_ARCHIVE}.partial" "${PYTHON_ARCHIVE}"
fi
[[ "$(sha256sum "${PYTHON_ARCHIVE}" | awk '{print $1}')" == "${KATAGO_PYTHON_RUNTIME_SHA256}" ]] \
  || die "Python archive checksum mismatch"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
bundle_name="katago-sm89-sm103-sm120-autotune-${timestamp}"
stage="$(mktemp -d "${OUTPUT_ROOT}/.${bundle_name}.XXXXXX")"
bundle="${stage}/${bundle_name}"
source_stage="${stage}/source-stage/sources"
cleanup() {
  case "${stage}" in
    "${OUTPUT_ROOT}"/.katago-sm89-sm103-sm120-autotune-*)
      find "${stage}" -mindepth 1 -delete
      rmdir -- "${stage}"
      ;;
  esac
}
trap cleanup EXIT
mkdir -p -- "${bundle}/payload/wheels" "${bundle}/patches" "${bundle}/metadata" \
  "${bundle}/plans" "${bundle}/records" "${source_stage}"

cp -- "${REPO_ROOT}/setup.sh" "${REPO_ROOT}/run-autotune.sh" \
  "${REPO_ROOT}/build-for-plan.sh" \
  "${SCRIPT_DIR}/autotune.py" "${SCRIPT_DIR}/build_for_plan.py" \
  "${SCRIPT_DIR}/detect_gpu.py" \
  "${SCRIPT_DIR}/prepare_accuracy_corpus.py" \
  "${REPO_ROOT}/python/build_parallelism.py" "${bundle}/"
cp -- "${REPO_ROOT}/run.sh" "${bundle}/run.sh"
cp -- "${SCRIPT_DIR}/README.md" "${SCRIPT_DIR}/SPEC.md" "${SCRIPT_DIR}/source-lock.tsv" \
  "${SCRIPT_DIR}/corpus.lock.sh" \
  "${bundle}/metadata/"
cp -- "${REPO_ROOT}/final-migration/README.md" "${bundle}/README.md"
cp -- "${REPO_ROOT}/final-migration/README.zh-CN.md" "${bundle}/README.zh-CN.md"
cp -- "${REPO_ROOT}/final-migration/RUNTIME.md" "${bundle}/RUNTIME.md"
cp -a -- "${REPO_ROOT}/final-migration/plans/." "${bundle}/plans/"
cp -- "${REPO_ROOT}/final-migration/records/rtx4090d-official-backend-baselines-20260811.md" \
  "${REPO_ROOT}/final-migration/records/rtx5080-official-backend-baselines-20260811.md" \
  "${bundle}/records/"
cp -- "${REPO_ROOT}/cpp/neuralnet/flash-attention-sm89.patch" \
  "${SCRIPT_DIR}/patches/flash-attention-sm120-both16.patch" "${bundle}/patches/"
chmod 0755 "${bundle}/setup.sh" "${bundle}/run-autotune.sh" \
  "${bundle}/run.sh" "${bundle}/build-for-plan.sh" \
  "${bundle}/autotune.py" "${bundle}/build_for_plan.py" \
  "${bundle}/detect_gpu.py" \
  "${bundle}/prepare_accuracy_corpus.py"

copy_source() {
  local name="$1" source="$2" actual target
  grep -q "^${name}"$'\t' "${SCRIPT_DIR}/source-lock.tsv" \
    || die "no compatibility entry for ${name}"
  actual="$(git -C "${source}" rev-parse HEAD)"
  [[ -z "$(git -C "${source}" status --porcelain)" ]] || die "dirty source tree: ${source}"
  target="${source_stage}/${name}"
  mkdir -p -- "${target}"
  tar --create --file - --directory "${source}" --exclude=.git --exclude=build \
      --exclude=dist --exclude='*.egg-info' . | tar --extract --file - --directory "${target}"
  printf '%s\n' "${actual}" > "${target}/.katago-source-revision"
  printf '%s\t%s\n' "${name}" "${actual}" >> "${bundle}/metadata/resolved-sources.tsv"
}

printf 'name\tresolved_revision\n' > "${bundle}/metadata/resolved-sources.tsv"
copy_source cutlass "${SOURCE_ROOT}/cutlass"
copy_source flash-attention "${SOURCE_ROOT}/flash-attention"
copy_source zlib "${ZLIB_ROOT}"

flash_cutlass_actual="$(git -C "${FLASH_CUTLASS_ROOT}" rev-parse HEAD)"
[[ -z "$(git -C "${FLASH_CUTLASS_ROOT}" status --porcelain)" ]] \
  || die "dirty FlashAttention CUTLASS source"
flash_cutlass_target="${source_stage}/flash-attention/csrc/cutlass"
find "${flash_cutlass_target}" -mindepth 1 -delete 2>/dev/null || true
tar --create --file - --directory "${FLASH_CUTLASS_ROOT}" --exclude=.git . \
  | tar --extract --file - --directory "${flash_cutlass_target}"
printf '%s\n' "${flash_cutlass_actual}" > "${flash_cutlass_target}/.katago-source-revision"
printf '%s\t%s\n' flash-cutlass "${flash_cutlass_actual}" \
  >> "${bundle}/metadata/resolved-sources.tsv"

log "applying the two recorded FlashAttention patches to the carried source"
(
  cd /tmp
  git apply --unsafe-paths --directory="${source_stage}/flash-attention" \
    "${REPO_ROOT}/cpp/neuralnet/flash-attention-sm89.patch"
  git apply --unsafe-paths --directory="${source_stage}/flash-attention" \
    "${SCRIPT_DIR}/patches/flash-attention-sm120-both16.patch"
)
printf '%s\t%s\n%s\t%s\n' \
  flash-attention-sm89.patch "$(sha256sum "${REPO_ROOT}/cpp/neuralnet/flash-attention-sm89.patch" | awk '{print $1}')" \
  flash-attention-sm120-both16.patch "$(sha256sum "${SCRIPT_DIR}/patches/flash-attention-sm120-both16.patch" | awk '{print $1}')" \
  > "${source_stage}/flash-attention/.katago-applied-patches.tsv"

tar --create --gzip --file "${bundle}/payload/sources.tar.gz" \
  --directory "${stage}/source-stage" sources
git -C "${REPO_ROOT}" archive --format=tar --prefix=repo/ HEAD \
  | gzip -9 > "${bundle}/payload/repo.tar.gz"
cp -- "${PYTHON_ARCHIVE}" "${bundle}/payload/python.tar.gz"

# Resolve wheels with the bundled interpreter itself.  The packaging host may
# run a different Python and must not accidentally populate a CPython 3.12
# wheelhouse for the locked CPython 3.14 runtime.
wheel_runtime="${stage}/wheel-runtime"
mkdir -p -- "${wheel_runtime}"
tar --extract --gzip --file "${PYTHON_ARCHIVE}" --directory "${wheel_runtime}"
wheel_python="${wheel_runtime}/python/bin/python3"
[[ -x "${wheel_python}" ]] || die "Python archive did not contain python/bin/python3"
[[ "$("${wheel_python}" -c 'import platform; print(platform.python_version())')" == \
   "${KATAGO_PYTHON_RUNTIME_VERSION}" ]] \
  || die "wheel resolver Python does not match the locked runtime"

asset_stage="${stage}/asset-stage/assets"
mkdir -p -- "${asset_stage}"
cp -- "${MODEL}" "${asset_stage}/b11c768h12nbt3tflrs-fson-silu.bin.gz"
cp -- "${CORPUS}" "${asset_stage}/$(basename -- "${CORPUS}")"
cp -- "${CORPUS_MANIFEST}" "${asset_stage}/$(basename -- "${CORPUS_MANIFEST}")"
if [[ -n "${AUTOTUNE_FP32_GOLDEN:-}" ]]; then
  [[ -r "${AUTOTUNE_FP32_GOLDEN}" ]] || die "AUTOTUNE_FP32_GOLDEN is unreadable"
  cp -- "${AUTOTUNE_FP32_GOLDEN}" "${asset_stage}/replay-fixed-fp32-full19.krnn"
  golden_metadata="${AUTOTUNE_FP32_GOLDEN%.krnn}.json"
  [[ -r "${golden_metadata}" ]] \
    || die "AUTOTUNE_FP32_GOLDEN requires its immutable .json sidecar"
  mapfile -t golden_identity < <(python3 -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(d["reference_sha256"]); print(d["model_sha256"]); print(d["corpus_sha256"])' \
    "${golden_metadata}")
  [[ "$(sha256sum "${AUTOTUNE_FP32_GOLDEN}" | awk '{print $1}')" == "${golden_identity[0]}" ]] \
    || die "AUTOTUNE_FP32_GOLDEN differs from its sidecar"
  [[ "$(sha256sum "${MODEL}" | awk '{print $1}')" == "${golden_identity[1]}" ]] \
    || die "AUTOTUNE_FP32_GOLDEN was generated for a different model"
  [[ "$(sha256sum "${CORPUS}" | awk '{print $1}')" == "${golden_identity[2]}" ]] \
    || die "AUTOTUNE_FP32_GOLDEN was generated for a different corpus"
  cp -- "${golden_metadata}" "${asset_stage}/replay-fixed-fp32-full19.json"
fi
tar --create --gzip --file "${bundle}/payload/assets.tar.gz" \
  --directory "${stage}/asset-stage" assets

log "resolving the pinned wheel payload; this is the only packaging step that may use PyPI"
log "using configured additional package indexes"
log "using managed pip cache: ${PIP_CACHE_DIR}"
IFS=: read -r -a seed_wheel_dirs <<< "${SEED_WHEELS}"
find_links=()
for seed_wheel_dir in "${seed_wheel_dirs[@]}"; do
  [[ -d "${seed_wheel_dir}" ]] || die "seed wheel directory missing: ${seed_wheel_dir}"
  find_links+=(--find-links "${seed_wheel_dir}")
done
"${wheel_python}" -m pip download --index-url "${PYPI_MIRROR}" "${find_links[@]}" \
  --only-binary=:all: --no-deps --dest "${bundle}/payload/wheels" \
  --requirement "${SCRIPT_DIR}/python-build-requirements.txt" \
  --requirement "${SCRIPT_DIR}/python-binary-requirements.txt"
python3 "${SCRIPT_DIR}/lock_wheels.py" "${SCRIPT_DIR}/python-build-requirements.txt" \
  "${bundle}/payload/wheels" "${bundle}/payload/python-build-requirements.lock"
python3 "${SCRIPT_DIR}/lock_wheels.py" "${SCRIPT_DIR}/python-binary-requirements.txt" \
  "${bundle}/payload/wheels" "${bundle}/payload/python-binary-requirements.lock"

{
  printf 'created_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'katago_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  printf 'python_version=%s\npython_build_standalone_release=%s\n' \
    "${KATAGO_PYTHON_RUNTIME_VERSION}" "${KATAGO_PYTHON_RUNTIME_RELEASE}"
  printf 'native_cuda_toolkit_pypi=13.3.1\nnative_cudnn_cuda13_pypi=9.25.0.15\n'
  printf 'torch=2.13.0+cu132\ntorch_wheel_cuda=13.2\ntorch_metadata_cuda_toolkit=13.2.1\n'
  printf 'flashinfer_python=0.6.17\nnvidia_cudnn_frontend=1.27.0\nliger_kernel=0.8.1\nmslk=1.3.0+cu132\n'
  printf 'minimum_driver=610.43.02\n'
  printf 'model_sha256=%s\n' "$(sha256sum "${MODEL}" | awk '{print $1}')"
  printf 'corpus_sha256=%s\n' "$(sha256sum "${CORPUS}" | awk '{print $1}')"
  "${CORPUS_PYTHON}" -c 'import json,sys; manifest=json.load(open(sys.argv[1])); result=json.load(open(sys.argv[2])); print("training_data_archive="+manifest["source_archive"]); print("training_data_archive_sha256="+manifest["source_archive_sha256"]); print("training_data_url="+result["source_url"])' "${CORPUS_MANIFEST}" "${CORPUS_RESULT}"
} > "${bundle}/metadata/release.txt"

(
  cd -- "${bundle}"
  find payload patches metadata plans records -type f ! -path payload/SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > payload/SHA256SUMS
  sha256sum README.md README.zh-CN.md RUNTIME.md >> payload/SHA256SUMS
)

tarball="${OUTPUT_ROOT}/${bundle_name}.tar"
tar --create --file "${tarball}" --directory "${stage}" "${bundle_name}"
(
  cd -- "${OUTPUT_ROOT}"
  sha256sum "${bundle_name}.tar" > "${bundle_name}.tar.sha256"
)
log "release complete: ${tarball}"
du -h "${tarball}"
