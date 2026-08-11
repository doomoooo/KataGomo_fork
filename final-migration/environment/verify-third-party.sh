#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

activate_venv
activate_toolchain
require_command nvcc
require_command g++

ensure_record_root
record="${KATAGO_RECORD_ROOT}/third-party-verify-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "${record}") 2>&1

smoke_build="${KATAGO_ENV_ROOT}/smoke-build"
assert_safe_managed_path "${smoke_build}"
mkdir -p -- "${smoke_build}"

export XDG_CACHE_HOME="${KATAGO_ENV_ROOT}/cache"
mkdir -p -- "${XDG_CACHE_HOME}"

if [[ -n "${KATAGO_SMOKE_ARCHS:-}" ]]; then
  read -r -a smoke_archs <<< "${KATAGO_SMOKE_ARCHS}"
else
  mapfile -t smoke_archs < <(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | tr -d '. ' | sort -u
  )
  if (( ${#smoke_archs[@]} == 0 )); then
    smoke_archs=(89 120)
  fi
fi

printf 'CUDA smoke architectures:'
printf ' sm_%s' "${smoke_archs[@]}"
printf '\n'

for arch in "${smoke_archs[@]}"; do
  log "compiling CUDA/cuBLAS/cuDNN link smoke for sm_${arch}"
  nvcc -std=c++17 -arch="sm_${arch}" \
    "${SCRIPT_DIR}/smoke/cuda_stack.cu" \
    -I"${KATAGO_CUDNN_ROOT}/include" \
    -L"${CUDA_HOME}/lib64" -L"${KATAGO_CUDNN_ROOT}/lib" \
    -l:libcublas.so.13 -l:libcudnn.so.9 \
    -o "${smoke_build}/cuda-stack-sm${arch}"

  log "compiling CUTLASS/CuTe header smoke for sm_${arch}"
  nvcc -std=c++17 -arch="sm_${arch}" \
    -I"${KATAGO_THIRD_PARTY_ROOT}/cutlass/include" \
    -I"${KATAGO_THIRD_PARTY_ROOT}/cutlass/tools/util/include" \
    -c "${SCRIPT_DIR}/smoke/cutlass_cute.cu" \
    -o "${smoke_build}/cutlass-cute-sm${arch}.o"
done

log "compiling cuDNN frontend header smoke"
g++ -std=c++17 \
  -I"${REPO_ROOT}/cpp/external/cudnn-frontend/include" \
  -I"${CUDA_HOME}/include" \
  -I"${KATAGO_CUDNN_ROOT}/include" \
  "${SCRIPT_DIR}/smoke/cudnn_frontend.cpp" \
  -L"${CUDA_HOME}/lib64" -L"${KATAGO_CUDNN_ROOT}/lib" \
  -l:libcudnn.so.9 -l:libcudart.so.13 \
  -o "${smoke_build}/cudnn-frontend"

log "checking Python CUDA/codegen imports"
python - <<'PY'
import cuda
import cutlass
import flash_attn.cute
import tilelang
import torch
import triton

print("PYTHON_IMPORTS_OK")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("triton", triton.__version__)
print("tilelang", tilelang.__version__)
print("cutlass", cutlass.__version__)
print("flash_attn.cute", flash_attn.cute.__file__)
PY

for arch in "${smoke_archs[@]}"; do
  log "compiling TileLang no-run smoke for sm_${arch}"
  python "${SCRIPT_DIR}/smoke/tilelang_compile.py" --arch "${arch}"
done

log "third-party compile verification complete; no GPU kernels were executed"
printf 'record=%s\n' "${record}"
