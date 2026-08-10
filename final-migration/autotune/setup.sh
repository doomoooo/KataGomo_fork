#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PREFIX="${SCRIPT_DIR}/runtime"
JOBS=""

usage() {
  printf 'Usage: %s [--prefix DIR] [--jobs N] [--verify-only]\n' "$0"
}

verify_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$(readlink -m -- "$2")"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --verify-only) verify_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[autotune-setup] %s\n' "$*"; }
die() { printf '[autotune-setup] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "required host command missing: $1"; }

default_build_jobs() {
  local cpu_jobs available_bytes cgroup_max cgroup_current cgroup_available memory_jobs
  cpu_jobs="$(nproc)"
  available_bytes="$(awk '$1 == "MemAvailable:" { print $2 * 1024; exit }' /proc/meminfo)"
  cgroup_max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
  cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)"
  if [[ "${cgroup_max}" =~ ^[0-9]+$ && "${cgroup_current}" =~ ^[0-9]+$ ]]; then
    cgroup_available=$((cgroup_max - cgroup_current))
    (( cgroup_available < 0 )) && cgroup_available=0
    if [[ -z "${available_bytes}" ]] || (( cgroup_available < available_bytes )); then
      available_bytes="${cgroup_available}"
    fi
  fi
  if [[ "${available_bytes}" =~ ^[0-9]+$ ]]; then
    # Keep 25% of currently available memory in reserve and allow 2 GiB for
    # each heavy C++/CUDA compiler process. Triton exceeds 1 GiB per cc1plus.
    memory_jobs=$((available_bytes * 3 / 4 / (2 * 1024 * 1024 * 1024)))
    (( memory_jobs < 1 )) && memory_jobs=1
    (( memory_jobs < cpu_jobs )) && cpu_jobs="${memory_jobs}"
  fi
  (( cpu_jobs > 8 )) && cpu_jobs=8
  printf '%s\n' "${cpu_jobs}"
}

for command_name in bash tar sha256sum readlink find uname getconf gcc g++; do need "${command_name}"; done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] \
  || die "the release supports Linux x86-64 only"
glibc_version="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
IFS=. read -r glibc_major glibc_minor <<< "${glibc_version}"
[[ "${glibc_major}" =~ ^[0-9]+$ && "${glibc_minor}" =~ ^[0-9]+$ ]] \
  || die "unable to parse glibc version ${glibc_version}"
(( glibc_major > 2 || (glibc_major == 2 && glibc_minor >= 28) )) \
  || die "glibc ${glibc_version} is older than the required 2.28"
[[ "${PREFIX}" != / && "${PREFIX}" != /usr && "${PREFIX}" != /opt ]] \
  || die "refusing system prefix ${PREFIX}"
[[ -r "${SCRIPT_DIR}/payload/SHA256SUMS" ]] || die "payload manifest is missing"

log "verifying all carried payloads"
(cd -- "${SCRIPT_DIR}" && sha256sum --check --strict payload/SHA256SUMS)
(( verify_only == 0 )) || exit 0

if [[ -z "${JOBS}" ]]; then
  JOBS="$(default_build_jobs)"
fi
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be positive"

mkdir -p -- "${PREFIX}" "${PREFIX}/state" "${PREFIX}/logs"
exec > >(tee -a "${PREFIX}/logs/setup.log") 2>&1
log "using ${JOBS} parallel build jobs (nproc=$(nproc); default is memory-aware)"

extract_once() {
  local archive="$1" marker="$2"
  if [[ -e "${marker}" ]]; then
    log "reusing extracted $(basename -- "${archive}")"
    return
  fi
  log "extracting $(basename -- "${archive}")"
  tar --extract --gzip --file "${archive}" --directory "${PREFIX}"
  mkdir -p -- "$(dirname -- "${marker}")"
  : > "${marker}"
}

extract_once "${SCRIPT_DIR}/payload/python.tar.gz" "${PREFIX}/state/python.extracted"
extract_once "${SCRIPT_DIR}/payload/cuda-13.2.tar.gz" "${PREFIX}/state/cuda.extracted"
extract_once "${SCRIPT_DIR}/payload/cudnn-9.25-cuda13.tar.gz" "${PREFIX}/state/cudnn.extracted"
extract_once "${SCRIPT_DIR}/payload/sources.tar.gz" "${PREFIX}/state/sources.extracted"
extract_once "${SCRIPT_DIR}/payload/toolchains.tar.gz" "${PREFIX}/state/toolchains.extracted"
extract_once "${SCRIPT_DIR}/payload/repo.tar.gz" "${PREFIX}/state/repo.extracted"
extract_once "${SCRIPT_DIR}/payload/assets.tar.gz" "${PREFIX}/state/assets.extracted"

for tree in "${PREFIX}/cuda" "${PREFIX}/cudnn"; do
  broken_link="$(find "${tree}" -xtype l -print -quit)"
  [[ -z "${broken_link}" ]] || die "carried toolchain contains a broken symlink: ${broken_link}"
done
for library in \
  "${PREFIX}/cuda/targets/x86_64-linux/lib/libcublas.so" \
  "${PREFIX}/cuda/targets/x86_64-linux/lib/libcublasLt.so" \
  "${PREFIX}/cuda/lib64/libcudart.so" \
  "${PREFIX}/cudnn/lib/libcudnn.so"; do
  [[ -r "${library}" ]] || die "carried CUDA library is missing: ${library}"
done

export CUDA_HOME="${PREFIX}/cuda"
export CUDA_PATH="${CUDA_HOME}"
export CUDNN_ROOT="${PREFIX}/cudnn"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export PATH="${PREFIX}/venv/bin:${PREFIX}/python/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${CUDA_HOME}/lib64:${PREFIX}/native/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${PREFIX}/native:${CMAKE_PREFIX_PATH:-}"
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
export MAX_JOBS="${JOBS}"
export XDG_CACHE_HOME="${PREFIX}/cache"
export TRITON_HOME="${PREFIX}/cache/triton"
export TRITON_CACHE_DIR="${PREFIX}/cache/triton-runtime"
export AUTOTUNE_PREFIX="${PREFIX}"
mkdir -p -- "${XDG_CACHE_HOME}" "${TRITON_HOME}" "${TRITON_CACHE_DIR}"

"${CUDA_HOME}/bin/nvcc" --version | tail -n 1
[[ -r "${CUDNN_ROOT}/include/cudnn_version.h" ]] || die "cuDNN headers missing"

if [[ ! -x "${PREFIX}/venv/bin/python" ]]; then
  log "creating the locked Python environment"
  "${PREFIX}/python/bin/python3" -m venv --copies "${PREFIX}/venv"
fi
python_bin="${PREFIX}/venv/bin/python"
wheelhouse="${SCRIPT_DIR}/payload/wheels"

log "installing pinned build and binary prerequisites without an index"
"${python_bin}" -m pip install --no-index --find-links "${wheelhouse}" \
  --require-hashes -r "${SCRIPT_DIR}/payload/python-build-requirements.lock"
"${python_bin}" -m pip install --no-index --find-links "${wheelhouse}" \
  --no-deps --require-hashes -r "${SCRIPT_DIR}/payload/python-binary-requirements.lock"

mapfile -t corpus_files < <(find "${PREFIX}/assets" -maxdepth 1 -type f \
  -name '*-19x19-8192-seed*-full19.npz' -print | sort)
mapfile -t corpus_manifests < <(find "${PREFIX}/assets" -maxdepth 1 -type f \
  -name '*-19x19-8192-seed*-full19.manifest.json' -print | sort)
(( ${#corpus_files[@]} == 1 && ${#corpus_manifests[@]} == 1 )) \
  || die "the release must carry exactly one 8192-row corpus and manifest"
log "validating the frozen latest-at-release training-data corpus"
"${python_bin}" "${SCRIPT_DIR}/prepare_accuracy_corpus.py" \
  --repo "${PREFIX}/repo" --python "${python_bin}" \
  --output-dir "${PREFIX}/assets" --work-dir "${PREFIX}/training-data" \
  --corpus "${corpus_files[0]}" --manifest "${corpus_manifests[0]}" \
  --result-json "${PREFIX}/state/accuracy-corpus.json"

fp32_golden="${PREFIX}/assets/replay-fixed-fp32-full19.krnn"
fp32_metadata="${PREFIX}/assets/replay-fixed-fp32-full19.json"
if [[ -e "${fp32_golden}" || -e "${fp32_metadata}" ]]; then
  [[ -r "${fp32_golden}" && -r "${fp32_metadata}" ]] \
    || die "the FP32 reference and metadata must be carried together"
  mapfile -t fp32_identity < <("${python_bin}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(d["reference_sha256"]); print(d["model_sha256"]); print(d["corpus_sha256"])' \
    "${fp32_metadata}")
  actual_fp32_sha="$(sha256sum "${fp32_golden}" | awk '{print $1}')"
  [[ "${actual_fp32_sha}" == "${fp32_identity[0]}" ]] \
    || die "the immutable FP32 reference checksum differs from its metadata"
  model_asset="${PREFIX}/assets/b11c768h12nbt3tflrs-fson-silu.bin.gz"
  [[ "$(sha256sum "${model_asset}" | awk '{print $1}')" == "${fp32_identity[1]}" ]] \
    || die "the immutable FP32 reference belongs to a different model"
  [[ "$(sha256sum "${corpus_files[0]}" | awk '{print $1}')" == "${fp32_identity[2]}" ]] \
    || die "the immutable FP32 reference belongs to a different corpus"
fi

native_marker="${PREFIX}/state/native-built"
if [[ ! -e "${native_marker}" ]]; then
  log "building the carried zlib source"
  cmake -S "${PREFIX}/sources/zlib" -B "${PREFIX}/build/zlib" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}/native" \
    -DZLIB_BUILD_EXAMPLES=OFF
  cmake --build "${PREFIX}/build/zlib" --parallel "${JOBS}"
  cmake --install "${PREFIX}/build/zlib"
  : > "${native_marker}"
fi

source_wheels="${PREFIX}/built-wheels"
mkdir -p -- "${source_wheels}"

build_source_wheel() {
  local name="$1" source_dir="$2" distribution="$3" marker wheel
  marker="${PREFIX}/state/source-${name}-installed"
  if [[ -e "${marker}" ]]; then
    log "reusing installed source build ${name}"
    return
  fi
  log "building ${name} from carried source"
  find "${source_wheels}" -maxdepth 1 -type f -name "${name}-*.whl" -delete
  "${python_bin}" -m pip wheel --no-index --find-links "${wheelhouse}" \
    --no-build-isolation --no-deps --wheel-dir "${source_wheels}" "${source_dir}"
  wheel=$(find "${source_wheels}" -maxdepth 1 -type f \
    \( -iname "${name//-/_}-*.whl" -o -iname "${name//_/-}-*.whl" \) \
    | sort | tail -n 1)
  [[ -n "${wheel}" ]] || die "${name} did not produce a wheel"
  "${python_bin}" -m pip install --no-index --no-deps --force-reinstall "${wheel}"
  "${python_bin}" -c 'import importlib.metadata,sys; print(sys.argv[1], importlib.metadata.version(sys.argv[1]))' "${distribution}"
  : > "${marker}"
}

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_APACHE_TVM_FFI=0.1.12
build_source_wheel apache_tvm_ffi "${PREFIX}/sources/apache-tvm-ffi" apache-tvm-ffi

export TRITON_OFFLINE_BUILD=1
export LLVM_SYSPATH="${PREFIX}/toolchains/triton-llvm"
export JSON_SYSPATH="${PREFIX}/toolchains/triton-json"
export TRITON_BUILD_PROTON=OFF
build_source_wheel triton "${PREFIX}/sources/triton" triton

export CUDA_VERSION=13.2
export USE_CUDA=1
build_source_wheel tilelang "${PREFIX}/sources/TileLang" tilelang
build_source_wheel quack_kernels "${PREFIX}/sources/quack" quack-kernels

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4=0.0.1.dev1+g69e1bcbe7
build_source_wheel flash_attn_4 "${PREFIX}/sources/flash-attention/flash_attn/cute" flash-attn-4

log "verifying imports and recording exact installed environment"
"${python_bin}" - <<'PY'
import importlib
import importlib.metadata
import json
import pathlib
import torch

mods = ["cuda.bindings.runtime", "cutlass.cute", "tvm_ffi", "triton", "tilelang", "quack", "flash_attn.cute"]
for mod in mods:
    importlib.import_module(mod)
payload = {
    "python": importlib.import_module("sys").version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "distributions": {
        name: importlib.metadata.version(name)
        for name in ("apache-tvm-ffi", "triton", "tilelang", "quack-kernels", "flash-attn-4", "nvidia-cutlass-dsl")
    },
}
prefix = pathlib.Path(importlib.import_module("os").environ["AUTOTUNE_PREFIX"])
(prefix / "state" / "python-environment.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"${python_bin}" -m pip freeze --all > "${PREFIX}/state/pip-freeze.txt"
sha256sum "${source_wheels}"/*.whl > "${PREFIX}/state/source-wheel-sha256.txt"

printf '%s\n' \
  "export AUTOTUNE_PREFIX='${PREFIX}'" \
  "export CUDA_HOME='${CUDA_HOME}'" \
  "export CUDA_PATH='${CUDA_HOME}'" \
  "export CUDNN_ROOT='${CUDNN_ROOT}'" \
  "export CC='${CC}'" \
  "export CXX='${CXX}'" \
  "export PATH='${PREFIX}/venv/bin:${CUDA_HOME}/bin':\"\${PATH}\"" \
  "export LD_LIBRARY_PATH='${CUDNN_ROOT}/lib:${CUDA_HOME}/lib64:${PREFIX}/native/lib':\"\${LD_LIBRARY_PATH:-}\"" \
  "export CMAKE_PREFIX_PATH='${PREFIX}/native':\"\${CMAKE_PREFIX_PATH:-}\"" \
  "export XDG_CACHE_HOME='${PREFIX}/cache'" \
  "export TRITON_HOME='${PREFIX}/cache/triton'" \
  "export TRITON_CACHE_DIR='${PREFIX}/cache/triton-runtime'" \
  > "${PREFIX}/activate"
chmod 0644 "${PREFIX}/activate"
printf '%s\n' "${PREFIX}" > "${SCRIPT_DIR}/runtime-prefix.txt"
log "setup complete; source ${PREFIX}/activate"
