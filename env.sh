#!/usr/bin/env bash

# Unified environment variables for local KataGo scripts.
# Edit values directly in this file as needed.

# 1) TensorRT location
export TENSORRT_ROOT=/opt/tensorrt

# 2) KataGo binary location
export KATAGO_BIN_PATH="/opt/katago/katago"

# 3) Runtime library search path (shared by all scripts)
export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 4) Model location
#export KATAGO_MODEL_PATH="/opt/katago/weights/b18tf.onnx"
export KATAGO_MODEL_PATH="/opt/katago/weights/b18tf.onnx"

# 5) Config location
export KATAGO_CONFIG_PATH="/opt/katago/configs/gtp_example.cfg"
