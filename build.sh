#!/bin/bash

# Build script for KataGomo with TensorRT backend (Gomoku version)
# This script handles compilation and deployment to /opt/katago

# Configuration
BUILD_DIR="cpp/build"
TENSORRT_ROOT="/opt/tensorrt"
DEPLOY_DIR="/opt/katago"

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

# Check if we're in the right directory
if [ ! -d "cpp" ]; then
    print_error "This script must be run from the root directory of the KataGomo repository"
fi

# Main build process
main() {
    print_info "Starting KataGomo build process"

    # Prepare build directory
    print_info "Preparing build environment"
    mkdir -p ${BUILD_DIR}

    # Configure CMake
    print_info "Configuring CMake"
    cd ${BUILD_DIR}
    cmake .. \
        -DUSE_BACKEND=TENSORRT \
        -DUSE_AVX2=1 \
        -DTENSORRT_INCLUDE_DIR=${TENSORRT_ROOT}/include \
        -DTENSORRT_LIBRARY=${TENSORRT_ROOT}/lib/libnvinfer.so \
        -DTENSORRT_ONNX_LIBRARY=${TENSORRT_ROOT}/lib/libnvonnxparser.so

    if [ $? -ne 0 ]; then
        print_error "CMake configuration failed"
    fi

    # Build
    print_info "Compiling KataGomo"
    make -j$(nproc)

    if [ $? -ne 0 ]; then
        print_error "Compilation failed"
    fi

    # Deploy
    print_info "Deploying to ${DEPLOY_DIR}"
    sudo mkdir -p ${DEPLOY_DIR}
    sudo rm ${DEPLOY_DIR}/katago
    sudo cp katago ${DEPLOY_DIR}/
    sudo chmod +x ${DEPLOY_DIR}/katago

    # Verify installation
    print_info "Verifying installation"
    ${DEPLOY_DIR}/katago version

    if [ $? -ne 0 ]; then
        print_error "Verification failed"
    fi

    print_info "Build and deployment completed successfully!"
}

# Run the build
main
