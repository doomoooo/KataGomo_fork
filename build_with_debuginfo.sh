#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

BUILD_DIR="cpp/build_dbginfo"
DO_DEPLOY=1
DO_CLEAN=0
NUM_JOBS="${NUM_JOBS:-$(nproc)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
EXTRA_CMAKE_FLAGS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./build_with_debuginfo.sh [options] [-- <extra-cmake-args>...]

Options:
  --clean               Remove the build directory before configure.
  --no-deploy           Build only, do not copy to KATAGO_BIN_PATH.
  --jobs <N>            Parallel build jobs (default: nproc).
  --build-dir <DIR>     Build directory (default: cpp/build_dbginfo).
  --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --no-deploy)
      DO_DEPLOY=0
      shift
      ;;
    --jobs)
      [[ $# -ge 2 ]] || print_error "--jobs requires a value"
      NUM_JOBS="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || print_error "--build-dir requires a value"
      BUILD_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_CMAKE_FLAGS+=("$@")
      break
      ;;
    *)
      print_error "Unknown argument: $1"
      ;;
  esac
done

[[ -d "${SCRIPT_DIR}/cpp" ]] || print_error "Run from repository root: ${SCRIPT_DIR}"
command -v cmake >/dev/null 2>&1 || print_error "cmake not found"
command -v readelf >/dev/null 2>&1 || print_error "readelf not found"
command -v file >/dev/null 2>&1 || print_error "file not found"

if [[ "${DO_CLEAN}" == "1" ]]; then
  print_info "Cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

print_info "Configuring CMake (${CMAKE_BUILD_TYPE})"
cmake -S "${SCRIPT_DIR}/cpp" -B "${SCRIPT_DIR}/${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -fno-omit-frame-pointer -fno-optimize-sibling-calls" \
  -DUSE_BACKEND=TENSORRT \
  -DUSE_AVX2=1 \
  -DTENSORRT_INCLUDE_DIR="${TENSORRT_ROOT}/include" \
  -DTENSORRT_LIBRARY="${TENSORRT_ROOT}/lib/libnvinfer.so" \
  -DTENSORRT_ONNX_LIBRARY="${TENSORRT_ROOT}/lib/libnvonnxparser.so" \
  "${EXTRA_CMAKE_FLAGS[@]}"

print_info "Building with ${NUM_JOBS} jobs"
cmake --build "${SCRIPT_DIR}/${BUILD_DIR}" --parallel "${NUM_JOBS}"

LOCAL_BIN="${SCRIPT_DIR}/${BUILD_DIR}/katago"
[[ -x "${LOCAL_BIN}" ]] || print_error "Built binary not found: ${LOCAL_BIN}"

TARGET_BIN="${LOCAL_BIN}"
if [[ "${DO_DEPLOY}" == "1" ]]; then
  TARGET_BIN="${KATAGO_BIN_PATH}"
  DEPLOY_DIR="$(dirname "${TARGET_BIN}")"
  print_info "Deploying to ${TARGET_BIN}"
  sudo mkdir -p "${DEPLOY_DIR}"
  sudo cp "${LOCAL_BIN}" "${TARGET_BIN}"
  sudo chmod +x "${TARGET_BIN}"
else
  print_warn "Skipping deploy (--no-deploy)"
fi

print_info "Verifying binary: ${TARGET_BIN}"
"${TARGET_BIN}" version
if ! file "${TARGET_BIN}" | grep -q "debug_info"; then
  print_error "Binary is missing debug info according to file(1)"
fi
if ! readelf -S "${TARGET_BIN}" | grep -q ".debug_info"; then
  print_error "Binary is missing .debug_info section"
fi

print_info "Build completed: ${TARGET_BIN}"
