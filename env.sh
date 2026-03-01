#!/usr/bin/env bash

# Unified environment variables for local KataGo scripts.
# Edit values directly in this file as needed.

# 1) TensorRT location
export TENSORRT_ROOT=/opt/tensorrt

# 2) Deploy location
export KATAGO_DEPLOY_DIR=/opt/katago

# 3) Model location
export KATAGO_MODEL_PATH="${KATAGO_DEPLOY_DIR}/weights/b18tf.onnx"

# 4) Config location
export KATAGO_CONFIG_PATH="${KATAGO_DEPLOY_DIR}/configs/gtp_example.cfg"
