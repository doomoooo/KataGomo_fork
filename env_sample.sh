#!/usr/bin/env bash

# Unified environment variables for local KataGo scripts.
# Edit values directly in this file as needed.

# 1) TensorRT location (used for both tools and runtime libraries)
export TENSORRT_ROOT="/path/to/tensorrt"
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 2) KataGo binary location
export KATAGO_BIN_PATH="/path/to/katago"

# 3) Model location
export KATAGO_MODEL_PATH="/path/to/model.bin.gz/or/model.onnx/"

# 4) Config location
export KATAGO_CONFIG_PATH="/path/to/gtp.cfg"

# 5) Optional local compiler toolchain used by build.sh
# Point this at a locally unpacked compiler toolchain if you do not want to use
# the system compiler. build.sh will auto-detect compilers under
# ${LOCAL_TOOLCHAIN_ROOT}/bin and prefers high-version Clang first.
# As of 2026-03 this branch expects a recent coroutine-capable compiler; the
# default bootstrap target in build.sh is LLVM 22.1.1.
# export TOOLCHAIN_BASE_DIR="/path/to/local/toolchains"
# export PREFERRED_LLVM_VERSION="22.1.1"
# export LOCAL_TOOLCHAIN_ROOT="/path/to/local/toolchains/llvm-22.1.1"
# export LOCAL_CC="/path/to/local/toolchains/llvm-22.1.1/bin/clang"
# export LOCAL_CXX="/path/to/local/toolchains/llvm-22.1.1/bin/clang++"
# export ALLOW_OLD_COMPILER="0"
