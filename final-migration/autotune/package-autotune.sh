#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
ENV_ROOT="${KATAGO_ENV_ROOT:-${REPO_ROOT}/.final-migration-env}"
SOURCE_ROOT="${AUTOTUNE_SOURCE_ROOT:-${ENV_ROOT}/third_party}"
OUTPUT_ROOT="${AUTOTUNE_OUTPUT_ROOT:-${ENV_ROOT}/autotune-distributions}"
CUDA_ROOT="${AUTOTUNE_CUDA_ROOT:-${ENV_ROOT}/toolchains/cuda-13.2}"
CUDNN_ROOT="${AUTOTUNE_CUDNN_ROOT:-${ENV_ROOT}/toolchains/cudnn-9.25-cuda13}"
LLVM_ROOT="${AUTOTUNE_TRITON_LLVM_ROOT:-${ENV_ROOT}/toolchains/triton-llvm}"
JSON_ROOT="${AUTOTUNE_TRITON_JSON_ROOT:-${ENV_ROOT}/toolchains/triton-json}"
FLASH_CUTLASS_ROOT="${AUTOTUNE_FLASH_CUTLASS_ROOT:-/workspace/third_party/flash-attention/csrc/cutlass}"
MODEL="${AUTOTUNE_MODEL:-/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz}"
CORPUS="${AUTOTUNE_CORPUS:-}"
CORPUS_MANIFEST="${AUTOTUNE_CORPUS_MANIFEST:-}"
CORPUS_PYTHON="${AUTOTUNE_CORPUS_PYTHON:-${ENV_ROOT}/venv/bin/python}"
CORPUS_OUTPUT_ROOT="${AUTOTUNE_CORPUS_OUTPUT_ROOT:-/workspace/trainingdata/accuracy}"
TRAINING_DATA_CACHE="${AUTOTUNE_TRAINING_DATA_CACHE:-/workspace/trainingdata}"
PYTHON_ARCHIVE="${AUTOTUNE_PYTHON_ARCHIVE:-${ENV_ROOT}/downloads/cpython-3.12.13+20260807-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz}"
PYTHON_URL='https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13%2B20260807-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
PYTHON_SHA256='506191be3ee7bd190a8834dcdc1b3bc70aab50608deccc711935aa007239cabd'
PYPI_MIRROR="${KATAGO_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
DEFAULT_SEED_WHEELS="${ENV_ROOT}/distributions/20260807T205459Z/wheels:${ENV_ROOT}/autotune-wheel-seed"
SEED_WHEELS="${AUTOTUNE_SEED_WHEELS:-${DEFAULT_SEED_WHEELS}}"

log() { printf '[autotune-package] %s\n' "$*"; }
die() { printf '[autotune-package] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "required command missing: $1"; }
for command_name in curl find git gzip python3 sha256sum tar; do need "${command_name}"; done

if [[ -z "${CORPUS}" || -z "${CORPUS_MANIFEST}" ]]; then
  [[ -z "${CORPUS}" && -z "${CORPUS_MANIFEST}" ]] \
    || die "AUTOTUNE_CORPUS and AUTOTUNE_CORPUS_MANIFEST must be set together"
  [[ -x "${CORPUS_PYTHON}" ]] || CORPUS_PYTHON="$(command -v python3)"
  "${CORPUS_PYTHON}" -c 'import numpy' \
    || die "accuracy-corpus Python lacks NumPy; run the environment setup first"
  corpus_result="${ENV_ROOT}/accuracy-corpus/current.json"
  log "resolving the latest public KataGo training archive and the fixed 8192-row corpus"
  "${CORPUS_PYTHON}" "${SCRIPT_DIR}/prepare_accuracy_corpus.py" \
    --repo "${REPO_ROOT}" --python "${CORPUS_PYTHON}" \
    --output-dir "${CORPUS_OUTPUT_ROOT}" \
    --work-dir "${ENV_ROOT}/accuracy-corpus" \
    --archive-cache-dir "${TRAINING_DATA_CACHE}" \
    --refresh-latest --result-json "${corpus_result}"
  CORPUS="$("${CORPUS_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["corpus"])' "${corpus_result}")"
  CORPUS_MANIFEST="$("${CORPUS_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest"])' "${corpus_result}")"
fi

[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] \
  || die "package only from a clean committed final-migration tree"
for path in "${CUDA_ROOT}" "${CUDNN_ROOT}" "${LLVM_ROOT}" "${JSON_ROOT}" \
            "${FLASH_CUTLASS_ROOT}" "${MODEL}" "${CORPUS}" "${CORPUS_MANIFEST}"; do
  [[ -e "${path}" ]] || die "required payload input missing: ${path}"
done
mkdir -p -- "${OUTPUT_ROOT}" "$(dirname -- "${PYTHON_ARCHIVE}")"

if [[ ! -r "${PYTHON_ARCHIVE}" ]]; then
  log "downloading the pinned Python source-independent runtime for release construction"
  curl --fail --location --retry 3 --output "${PYTHON_ARCHIVE}.partial" "${PYTHON_URL}"
  mv -- "${PYTHON_ARCHIVE}.partial" "${PYTHON_ARCHIVE}"
fi
[[ "$(sha256sum "${PYTHON_ARCHIVE}" | awk '{print $1}')" == "${PYTHON_SHA256}" ]] \
  || die "Python archive checksum mismatch"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
bundle_name="katago-sm89-sm120-autotune-${timestamp}"
stage="$(mktemp -d "${OUTPUT_ROOT}/.${bundle_name}.XXXXXX")"
bundle="${stage}/${bundle_name}"
source_stage="${stage}/source-stage/sources"
cleanup() {
  case "${stage}" in
    "${OUTPUT_ROOT}"/.katago-sm89-sm120-autotune-*)
      find "${stage}" -mindepth 1 -delete
      rmdir -- "${stage}"
      ;;
  esac
}
trap cleanup EXIT
mkdir -p -- "${bundle}/payload/wheels" "${bundle}/patches" "${bundle}/metadata" \
  "${bundle}/plans" "${source_stage}"

cp -- "${SCRIPT_DIR}/setup.sh" "${SCRIPT_DIR}/run-autotune.sh" \
  "${SCRIPT_DIR}/autotune.py" "${SCRIPT_DIR}/detect_gpu.py" \
  "${SCRIPT_DIR}/prepare_accuracy_corpus.py" \
  "${REPO_ROOT}/python/build_parallelism.py" "${bundle}/"
cp -- "${SCRIPT_DIR}/README.md" "${SCRIPT_DIR}/SPEC.md" "${SCRIPT_DIR}/source-lock.tsv" \
  "${bundle}/metadata/"
cp -- "${REPO_ROOT}/final-migration/README.md" "${bundle}/README.md"
cp -- "${REPO_ROOT}/final-migration/README.zh-CN.md" "${bundle}/README.zh-CN.md"
cp -a -- "${REPO_ROOT}/final-migration/plans/." "${bundle}/plans/"
cp -- "${REPO_ROOT}/cpp/neuralnet/flash-attention-sm89.patch" \
  "${SCRIPT_DIR}/patches/flash-attention-sm120-both16.patch" "${bundle}/patches/"
chmod 0755 "${bundle}/setup.sh" "${bundle}/run-autotune.sh" \
  "${bundle}/autotune.py" "${bundle}/detect_gpu.py" \
  "${bundle}/prepare_accuracy_corpus.py"

revision_for() {
  awk -F '\t' -v name="$1" '$1 == name {print $2; exit}' "${SCRIPT_DIR}/source-lock.tsv"
}

copy_source() {
  local name="$1" source="${SOURCE_ROOT}/$1" expected actual target
  expected="$(revision_for "${name}")"
  [[ -n "${expected}" ]] || die "no lock entry for ${name}"
  actual="$(git -C "${source}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || die "${name} revision ${actual} != ${expected}"
  [[ -z "$(git -C "${source}" status --porcelain)" ]] || die "dirty source tree: ${source}"
  target="${source_stage}/${name}"
  mkdir -p -- "${target}"
  tar --create --file - --directory "${source}" --exclude=.git --exclude=build \
      --exclude=dist --exclude='*.egg-info' . | tar --extract --file - --directory "${target}"
  printf '%s\n' "${actual}" > "${target}/.katago-source-revision"
}

for name in cutlass flash-attention triton quack TileLang apache-tvm-ffi cudnn-frontend zlib; do
  copy_source "${name}"
done

flash_cutlass_expected="$(revision_for flash-cutlass)"
flash_cutlass_actual="$(git -C "${FLASH_CUTLASS_ROOT}" rev-parse HEAD)"
[[ "${flash_cutlass_actual}" == "${flash_cutlass_expected}" ]] \
  || die "FlashAttention CUTLASS revision mismatch"
flash_cutlass_target="${source_stage}/flash-attention/csrc/cutlass"
find "${flash_cutlass_target}" -mindepth 1 -delete 2>/dev/null || true
tar --create --file - --directory "${FLASH_CUTLASS_ROOT}" --exclude=.git . \
  | tar --extract --file - --directory "${flash_cutlass_target}"
printf '%s\n' "${flash_cutlass_actual}" > "${flash_cutlass_target}/.katago-source-revision"

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

log "packing the CUDA 13.2 build toolkit"
tar --create --gzip --file "${bundle}/payload/cuda-13.2.tar.gz" \
  --directory "${CUDA_ROOT}" \
  --exclude='./nsight*' --exclude='./extras' --exclude='./gds' --exclude='./samples' \
  --transform='flags=r;s#^\.$#cuda#;s#^\./#cuda/#' .
log "packing cuDNN 9.25 CUDA 13"
tar --create --gzip --file "${bundle}/payload/cudnn-9.25-cuda13.tar.gz" \
  --directory "${CUDNN_ROOT}" \
  --transform='flags=r;s#^\.$#cudnn#;s#^\./#cudnn/#' .

toolchain_stage="${stage}/toolchain-stage/toolchains"
mkdir -p -- "${toolchain_stage}/triton-llvm" "${toolchain_stage}/triton-json"
tar --create --file - --directory "${LLVM_ROOT}" . \
  | tar --extract --file - --directory "${toolchain_stage}/triton-llvm"
tar --create --file - --directory "${JSON_ROOT}" . \
  | tar --extract --file - --directory "${toolchain_stage}/triton-json"
tar --create --gzip --file "${bundle}/payload/toolchains.tar.gz" \
  --directory "${stage}/toolchain-stage" toolchains

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
IFS=: read -r -a seed_wheel_dirs <<< "${SEED_WHEELS}"
find_links=()
for seed_wheel_dir in "${seed_wheel_dirs[@]}"; do
  [[ -d "${seed_wheel_dir}" ]] || die "seed wheel directory missing: ${seed_wheel_dir}"
  find_links+=(--find-links "${seed_wheel_dir}")
done
python3 -m pip download --index-url "${PYPI_MIRROR}" "${find_links[@]}" \
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
  printf 'python_version=3.12.13\npython_build_standalone_release=20260807\n'
  printf 'cuda_toolkit=13.2\ncudnn_cuda13=9.25.0.15\n'
  printf 'model_sha256=%s\n' "$(sha256sum "${MODEL}" | awk '{print $1}')"
  printf 'corpus_sha256=%s\n' "$(sha256sum "${CORPUS}" | awk '{print $1}')"
  "${CORPUS_PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("training_data_archive="+d["source_archive"]); print("training_data_archive_sha256="+d["source_archive_sha256"]); print("training_data_url="+d["source_archive_url"])' "${CORPUS_MANIFEST}"
} > "${bundle}/metadata/release.txt"

(
  cd -- "${bundle}"
  find payload patches metadata plans -type f ! -path payload/SHA256SUMS -print0 \
    | sort -z | xargs -0 sha256sum > payload/SHA256SUMS
  sha256sum README.md README.zh-CN.md >> payload/SHA256SUMS
)

tarball="${OUTPUT_ROOT}/${bundle_name}.tar"
tar --create --file "${tarball}" --directory "${stage}" "${bundle_name}"
(
  cd -- "${OUTPUT_ROOT}"
  sha256sum "${bundle_name}.tar" > "${bundle_name}.tar.sha256"
)
log "release complete: ${tarball}"
du -h "${tarball}"
