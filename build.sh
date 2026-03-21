#!/usr/bin/env bash
set -euo pipefail

# Build script for KataGomo with TensorRT backend (Gomoku version)
# This script handles compilation and deployment to /opt/katago

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

# Configuration
BUILD_DIR="${SCRIPT_DIR}/cpp/build"
KATAGO_BIN="${KATAGO_BIN_PATH}"
DEPLOY_DIR="$(dirname "${KATAGO_BIN}")"
NUM_JOBS="${NUM_JOBS:-$(nproc)}"
LOCAL_TOOLCHAIN_ROOT="${LOCAL_TOOLCHAIN_ROOT:-}"
LOCAL_CC="${LOCAL_CC:-}"
LOCAL_CXX="${LOCAL_CXX:-}"
TOOLCHAIN_BASE_DIR="${TOOLCHAIN_BASE_DIR:-${SCRIPT_DIR}/.local/toolchains}"
DOWNLOAD_BASE_DIR="${DOWNLOAD_BASE_DIR:-${SCRIPT_DIR}/.local/downloads}"
PREFERRED_LLVM_VERSION="${PREFERRED_LLVM_VERSION:-22.1.1}"
MIN_CLANG_MAJOR="${MIN_CLANG_MAJOR:-20}"
MIN_GCC_MAJOR="${MIN_GCC_MAJOR:-15}"
ALLOW_OLD_COMPILER="${ALLOW_OLD_COMPILER:-0}"
BOOTSTRAP_LLVM="${BOOTSTRAP_LLVM:-0}"

BASE_RELEASE_FLAGS="-O3 -DNDEBUG -march=native -mtune=native -fomit-frame-pointer"

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
Usage: $(basename "$0") [--help] [--bootstrap-llvm] [--llvm-version VERSION] [--toolchain-root PATH]

Options:
  --help               Show this help message
  --bootstrap-llvm     Download and unpack a local LLVM toolchain
  --llvm-version VER   LLVM release used by --bootstrap-llvm (default: ${PREFERRED_LLVM_VERSION})
  --toolchain-root DIR Use a specific local toolchain root instead of auto-discovery
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                print_usage
                exit 0
                ;;
            --bootstrap-llvm)
                BOOTSTRAP_LLVM=1
                shift
                ;;
            --llvm-version)
                [[ $# -ge 2 ]] || print_error "--llvm-version requires an argument"
                PREFERRED_LLVM_VERSION="$2"
                shift 2
                ;;
            --toolchain-root)
                [[ $# -ge 2 ]] || print_error "--toolchain-root requires an argument"
                LOCAL_TOOLCHAIN_ROOT="$2"
                shift 2
                ;;
            *)
                print_error "Unknown argument: $1 (use --help for usage)"
                ;;
        esac
    done
}

find_best_compiler_in_root() {
    local root="$1"
    shift
    local -a patterns=("$@")
    local -a candidates=()
    local pattern
    local path

    if [[ ! -d "${root}/bin" ]]; then
        return 1
    fi

    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for path in "${root}/bin"/${pattern}; do
            if [[ -x "${path}" ]]; then
                candidates+=("${path}")
            fi
        done
    done
    shopt -u nullglob

    if [[ ${#candidates[@]} -eq 0 ]]; then
        return 1
    fi

    printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
}

find_latest_local_llvm_root() {
    if [[ ! -d "${TOOLCHAIN_BASE_DIR}" ]]; then
        return 1
    fi

    find "${TOOLCHAIN_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'llvm-*' -printf '%f\n' \
        | sort -V | tail -n 1
}

ensure_local_llvm_toolchain() {
    local version="$1"
    local install_root="${TOOLCHAIN_BASE_DIR}/llvm-${version}"
    local os
    local arch
    local asset
    local url
    local archive
    local partial_archive
    local tmp_extract
    local extracted_root

    if [[ -x "${install_root}/bin/clang++" && -x "${install_root}/bin/clang" ]]; then
        print_info "LLVM ${version} is already available at ${install_root}"
        return 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
        print_error "curl is required to bootstrap a local LLVM toolchain"
    fi

    os="$(uname -s)"
    arch="$(uname -m)"
    case "${os}/${arch}" in
        Linux/x86_64)
            asset="LLVM-${version}-Linux-X64.tar.xz"
            ;;
        Linux/aarch64)
            asset="LLVM-${version}-Linux-ARM64.tar.xz"
            ;;
        *)
            print_error "Automatic LLVM bootstrap is only implemented for Linux x86_64 and Linux aarch64"
            ;;
    esac

    url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${version}/${asset}"
    archive="${DOWNLOAD_BASE_DIR}/${asset}"
    partial_archive="${archive}.partial"
    tmp_extract="${TOOLCHAIN_BASE_DIR}/.extract-llvm-${version}.$$"

    mkdir -p "${TOOLCHAIN_BASE_DIR}" "${DOWNLOAD_BASE_DIR}"

    if [[ ! -f "${archive}" ]]; then
        if [[ -f "${partial_archive}" ]]; then
            print_info "Resuming LLVM ${version} download from ${partial_archive}"
        else
            print_info "Downloading LLVM ${version} from ${url}"
        fi
        curl -L --fail --retry 5 --retry-all-errors -C - --output "${partial_archive}" "${url}"
        mv "${partial_archive}" "${archive}"
    else
        print_info "Reusing downloaded archive ${archive}"
    fi

    print_info "Extracting LLVM ${version} into ${install_root}"
    rm -rf "${tmp_extract}" "${install_root}"
    mkdir -p "${tmp_extract}"
    tar -xJf "${archive}" -C "${tmp_extract}"
    extracted_root="$(find "${tmp_extract}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [[ -z "${extracted_root}" ]]; then
        rm -rf "${tmp_extract}"
        print_error "Failed to extract LLVM ${version}"
    fi

    mv "${extracted_root}" "${install_root}"
    rm -rf "${tmp_extract}"
}

compiler_version() {
    local compiler="$1"
    local version

    version="$("${compiler}" -dumpfullversion -dumpversion 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    if [[ -z "${version}" ]]; then
        version="$("${compiler}" --version | head -n 1 | grep -Eo '[0-9]+([.][0-9]+)+' | head -n 1)"
    fi

    printf '%s' "${version}"
}

compiler_kind() {
    local compiler="$1"
    local first_line

    first_line="$("${compiler}" --version | head -n 1)"
    if [[ "${first_line}" == *clang* ]]; then
        printf '%s' "clang"
    elif [[ "${first_line}" == *gcc* || "${first_line}" == *GCC* || "${first_line}" == *g++* ]]; then
        printf '%s' "gcc"
    else
        printf '%s' "unknown"
    fi
}

check_compiler_floor() {
    local cxx_compiler="$1"
    local kind
    local version
    local major
    local message

    kind="$(compiler_kind "${cxx_compiler}")"
    version="$(compiler_version "${cxx_compiler}")"
    major="${version%%.*}"

    if [[ -z "${version}" || -z "${major}" ]]; then
        print_warning "Could not determine compiler version for ${cxx_compiler}; proceeding without a version gate"
        return 0
    fi

    case "${kind}" in
        clang)
            if (( major < MIN_CLANG_MAJOR )); then
                message="Selected Clang ${version}, but coroutine development should use Clang ${MIN_CLANG_MAJOR}+ or a newer local LLVM toolchain"
                if [[ "${ALLOW_OLD_COMPILER}" == "1" ]]; then
                    print_warning "${message}"
                else
                    print_error "${message}. Run ./build.sh --bootstrap-llvm or set LOCAL_TOOLCHAIN_ROOT."
                fi
            fi
            ;;
        gcc)
            if (( major < MIN_GCC_MAJOR )); then
                message="Selected GCC ${version}, but the compatibility lane should use GCC ${MIN_GCC_MAJOR}+"
                if [[ "${ALLOW_OLD_COMPILER}" == "1" ]]; then
                    print_warning "${message}"
                else
                    print_error "${message}. Prefer a recent local Clang toolchain or set ALLOW_OLD_COMPILER=1 if you need to override."
                fi
            fi
            ;;
        *)
            print_warning "Unrecognized compiler family for ${cxx_compiler}; skipping version gate"
            ;;
    esac
}

resolve_local_toolchain_root() {
    local latest_llvm_dir

    if [[ -n "${LOCAL_TOOLCHAIN_ROOT}" ]]; then
        return 0
    fi

    latest_llvm_dir="$(find_latest_local_llvm_root || true)"
    if [[ -n "${latest_llvm_dir}" ]]; then
        LOCAL_TOOLCHAIN_ROOT="${TOOLCHAIN_BASE_DIR}/${latest_llvm_dir}"
        print_info "Auto-selected local LLVM toolchain: ${LOCAL_TOOLCHAIN_ROOT}"
    fi
}

configure_and_build() {
    local build_dir="$1"
    local c_flags_release="$2"
    local cxx_flags_release="$3"
    local -a compiler_args=()
    local -a toolchain_linker_args=()
    local selected_cxx=""
    local selected_cxx_version=""
    local selected_cxx_kind=""
    local selected_cxx_dir=""

    if [[ -n "${LOCAL_TOOLCHAIN_ROOT}" ]]; then
        if [[ -z "${LOCAL_CC}" ]]; then
            LOCAL_CC="$(find_best_compiler_in_root "${LOCAL_TOOLCHAIN_ROOT}" "clang" "clang-[0-9]*" "gcc" "gcc-[0-9]*" || true)"
        fi

        if [[ -z "${LOCAL_CXX}" ]]; then
            LOCAL_CXX="$(find_best_compiler_in_root "${LOCAL_TOOLCHAIN_ROOT}" "clang++" "clang++-[0-9]*" "g++" "g++-[0-9]*" || true)"
        fi
    fi

    if [[ -n "${LOCAL_CC}" ]]; then
        compiler_args+=("-DCMAKE_C_COMPILER=${LOCAL_CC}")
        print_info "Using local C compiler: ${LOCAL_CC}"
    fi
    if [[ -n "${LOCAL_CXX}" ]]; then
        compiler_args+=("-DCMAKE_CXX_COMPILER=${LOCAL_CXX}")
        print_info "Using local C++ compiler: ${LOCAL_CXX}"
        selected_cxx="${LOCAL_CXX}"
    fi

    if [[ -z "${selected_cxx}" ]]; then
        if [[ -n "${CXX:-}" ]]; then
            selected_cxx="${CXX}"
        elif command -v c++ >/dev/null 2>&1; then
            selected_cxx="$(command -v c++)"
        elif command -v g++ >/dev/null 2>&1; then
            selected_cxx="$(command -v g++)"
        fi
    fi

    if [[ -n "${selected_cxx}" ]]; then
        check_compiler_floor "${selected_cxx}"
        selected_cxx_kind="$(compiler_kind "${selected_cxx}")"
        selected_cxx_version="$(compiler_version "${selected_cxx}")"
        if [[ -n "${selected_cxx_version}" ]]; then
            print_info "Selected C++ compiler version: ${selected_cxx_version}"
        fi
    fi

    if [[ "${selected_cxx_kind}" == "clang" ]]; then
        selected_cxx_dir="$(cd "$(dirname "${selected_cxx}")" && pwd)"
        if [[ -x "${selected_cxx_dir}/ld.lld" ]]; then
            toolchain_linker_args+=(
                "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=-fuse-ld=lld"
                "-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=-fuse-ld=lld"
                "-DCMAKE_MODULE_LINKER_FLAGS_RELEASE=-fuse-ld=lld"
            )
            print_info "Using lld from ${selected_cxx_dir}/ld.lld for Release linking"
        else
            print_warning "Selected Clang compiler does not have a sibling ld.lld; Release IPO may fall back to the system linker"
        fi

        if [[ -x "${selected_cxx_dir}/llvm-ar" ]]; then
            toolchain_linker_args+=(
                "-DCMAKE_AR=${selected_cxx_dir}/llvm-ar"
                "-DCMAKE_C_COMPILER_AR=${selected_cxx_dir}/llvm-ar"
                "-DCMAKE_CXX_COMPILER_AR=${selected_cxx_dir}/llvm-ar"
            )
        fi
        if [[ -x "${selected_cxx_dir}/llvm-ranlib" ]]; then
            toolchain_linker_args+=(
                "-DCMAKE_RANLIB=${selected_cxx_dir}/llvm-ranlib"
                "-DCMAKE_C_COMPILER_RANLIB=${selected_cxx_dir}/llvm-ranlib"
                "-DCMAKE_CXX_COMPILER_RANLIB=${selected_cxx_dir}/llvm-ranlib"
            )
        fi
    fi

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
        -DTENSORRT_ONNX_LIBRARY="${TENSORRT_ROOT}/lib/libnvonnxparser.so" \
        "${compiler_args[@]}" \
        "${toolchain_linker_args[@]}"
    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed for ${build_dir}"
    fi

    print_info "Compiling KataGomo in ${build_dir} with ${NUM_JOBS} jobs"
    cmake --build "${build_dir}" --parallel "${NUM_JOBS}"
    if [ $? -ne 0 ]; then
        print_error "Compilation failed for ${build_dir}"
    fi
}

# Parse CLI args
parse_args "$@"

# Check if we're in the right directory
if [ ! -d "${SCRIPT_DIR}/cpp" ]; then
    print_error "This script must be run from the root directory of the KataGomo repository"
fi

# Main build process
main() {
    print_info "Starting KataGomo build process"

    if [[ "${BOOTSTRAP_LLVM}" == "1" ]]; then
        ensure_local_llvm_toolchain "${PREFERRED_LLVM_VERSION}"
        LOCAL_TOOLCHAIN_ROOT="${TOOLCHAIN_BASE_DIR}/llvm-${PREFERRED_LLVM_VERSION}"
    fi

    resolve_local_toolchain_root

    print_info "Preparing build environment"
    mkdir -p "${BUILD_DIR}"
    configure_and_build "${BUILD_DIR}" "${BASE_RELEASE_FLAGS}" "${BASE_RELEASE_FLAGS}"

    # Deploy
    print_info "Deploying to ${DEPLOY_DIR}"
    mkdir -p "${DEPLOY_DIR}"
    rm -f "${KATAGO_BIN}"
    cp "${BUILD_DIR}/katago" "${KATAGO_BIN}"
    chmod +x "${KATAGO_BIN}"

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
