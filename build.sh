#!/usr/bin/env bash
set -euo pipefail

# Build script for KataGomo with TensorRT backend (Gomoku version)
# This script handles compilation and deployment to /opt/katago

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

# Configuration
BUILD_DIR="${SCRIPT_DIR}/cpp/build"
PGO_GEN_BUILD_DIR="${SCRIPT_DIR}/cpp/build_pgo_gen"
PGO_PROFILE_DIR="${SCRIPT_DIR}/cpp/pgo_profiles"
KATAGO_BIN="${KATAGO_BIN_PATH}"
DEPLOY_DIR="$(dirname "${KATAGO_BIN}")"
NUM_JOBS="${NUM_JOBS:-$(nproc)}"
ENABLE_PGO=false

BASE_RELEASE_FLAGS="-O3 -DNDEBUG -march=native -mtune=native -fomit-frame-pointer"
PGO_GENERATE_FLAGS="${BASE_RELEASE_FLAGS} -fprofile-generate=${PGO_PROFILE_DIR}"
PGO_USE_FLAGS="${BASE_RELEASE_FLAGS} -fprofile-use=${PGO_PROFILE_DIR} -fprofile-correction -Wno-error=coverage-mismatch"

# PGO benchmark workload knobs (override via CLI args)
PGO_BENCH_VISITS=2000
PGO_CUDA_STREAMS=4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [--pgo] [--pgo-bench-visits N] [--help]

Options:
  --pgo                Enable PGO build flow (default: disabled)
  --pgo-bench-visits N Visits passed to benchmark workload (default: ${PGO_BENCH_VISITS})
  --help               Show this help message
EOF
}

require_arg_value() {
    local opt_name="$1"
    local opt_value="${2:-}"
    if [ -z "${opt_value}" ]; then
        print_error "Missing value for ${opt_name}"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --pgo)
                ENABLE_PGO=true
                shift
                ;;
            --pgo-bench-visits)
                require_arg_value "$1" "${2:-}"
                PGO_BENCH_VISITS="$2"
                shift 2
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                print_error "Unknown argument: $1 (use --help for usage)"
                ;;
        esac
    done
}

validate_args() {
    if ! [[ "${PGO_BENCH_VISITS}" =~ ^[1-9][0-9]*$ ]]; then
        print_error "--pgo-bench-visits must be a positive integer, got: ${PGO_BENCH_VISITS}"
    fi
}

ensure_tcmalloc() {
    if ldconfig -p | grep -q "libtcmalloc_minimal.so"; then
        print_info "tcmalloc is already installed"
        return
    fi

    print_warning "tcmalloc not found, installing libgoogle-perftools-dev"
    sudo apt-get update
    sudo apt-get install -y libgoogle-perftools-dev

    if [ $? -ne 0 ]; then
        print_error "Failed to install tcmalloc (libgoogle-perftools-dev)"
    fi

    if ! ldconfig -p | grep -q "libtcmalloc_minimal.so"; then
        print_error "tcmalloc installed but libtcmalloc_minimal.so is still missing"
    fi
}

configure_and_build() {
    local build_dir="$1"
    local c_flags_release="$2"
    local cxx_flags_release="$3"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    print_info "Configuring CMake in ${build_dir}"
    cmake -S "${SCRIPT_DIR}/cpp" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DCMAKE_C_FLAGS_RELEASE="${c_flags_release}" \
        -DCMAKE_CXX_FLAGS_RELEASE="${cxx_flags_release}" \
        -DUSE_BACKEND=TENSORRT \
        -DUSE_AVX2=1 \
        -DTENSORRT_INCLUDE_DIR="${TENSORRT_ROOT}/include" \
        -DTENSORRT_LIBRARY="${TENSORRT_ROOT}/lib/libnvinfer.so" \
        -DTENSORRT_ONNX_LIBRARY="${TENSORRT_ROOT}/lib/libnvonnxparser.so"
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed for ${build_dir}"
    fi

    print_info "Compiling KataGomo in ${build_dir} with ${NUM_JOBS} jobs"
    cmake --build "${build_dir}" --parallel "${NUM_JOBS}"
    if [ $? -ne 0 ]; then
        print_error "Compilation failed for ${build_dir}"
    fi
}

run_pgo_training_workload() {
    local pgo_bin="${PGO_GEN_BUILD_DIR}/katago"
    local benchmark_script="${SCRIPT_DIR}/benchmark.sh"
    if [ ! -x "${pgo_bin}" ]; then
        print_error "Instrumented PGO binary not found: ${pgo_bin}"
    fi
    if [ ! -x "${benchmark_script}" ]; then
        print_error "benchmark.sh not found or not executable: ${benchmark_script}"
    fi

    # Keep cudaGraph enabled and reuse existing TensorRT plan cache by default.
    print_info "Running PGO training workload via benchmark.sh (cudaGraph enabled)"
    "${benchmark_script}" \
        --katago-bin "${pgo_bin}" \
        --visits "${PGO_BENCH_VISITS}" \
        --cuda-streams "${PGO_CUDA_STREAMS}"

    # Use -print -quit to avoid pipefail false negatives from find receiving SIGPIPE when grep -q exits early.
    if ! find "${PGO_PROFILE_DIR}" -type f -name "*.gcda" -print -quit | grep -q .; then
        print_error "No PGO profile data (*.gcda) was generated in ${PGO_PROFILE_DIR}"
    fi
}

# Parse CLI args
parse_args "$@"
validate_args

# Check if we're in the right directory
if [ ! -d "${SCRIPT_DIR}/cpp" ]; then
    print_error "This script must be run from the root directory of the KataGomo repository"
fi

# Main build process
main() {
    print_info "Starting KataGomo build process"
    print_info "PGO enabled: ${ENABLE_PGO}"
    # ensure_tcmalloc

    if [ "${ENABLE_PGO}" = "true" ]; then
        # Prepare build directories and fresh PGO profile output
        print_info "Preparing build environment"
        mkdir -p "${BUILD_DIR}" "${PGO_GEN_BUILD_DIR}"
        rm -rf "${PGO_PROFILE_DIR}"
        mkdir -p "${PGO_PROFILE_DIR}"

        # Stage 1: build instrumented binary to generate PGO profiles
        configure_and_build "${PGO_GEN_BUILD_DIR}" "${PGO_GENERATE_FLAGS}" "${PGO_GENERATE_FLAGS}"

        # Stage 2: run representative workload to collect profile data
        run_pgo_training_workload

        # Stage 3: rebuild optimized binary using generated profile + LTO + native
        configure_and_build "${BUILD_DIR}" "${PGO_USE_FLAGS}" "${PGO_USE_FLAGS}"
    else
        # Non-PGO release build with LTO + native tuning
        print_info "Preparing build environment (non-PGO)"
        mkdir -p "${BUILD_DIR}"
        configure_and_build "${BUILD_DIR}" "${BASE_RELEASE_FLAGS}" "${BASE_RELEASE_FLAGS}"
    fi

    # print_info "Checking tcmalloc linkage"
    # ldd katago | grep -q "libtcmalloc_minimal"
    # if [ $? -ne 0 ]; then
    #     print_error "katago is not linked with tcmalloc"
    # fi

    # Deploy
    print_info "Deploying to ${DEPLOY_DIR}"
    sudo mkdir -p "${DEPLOY_DIR}"
    sudo rm -f "${KATAGO_BIN}"
    sudo cp "${BUILD_DIR}/katago" "${KATAGO_BIN}"
    sudo chmod +x "${KATAGO_BIN}"

    # Verify installation
    print_info "Verifying installation"
    "${KATAGO_BIN}" version

    if [ $? -ne 0 ]; then
        print_error "Verification failed"
    fi

    print_info "Build and deployment completed successfully!"
}

# Run the build
main
