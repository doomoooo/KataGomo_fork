#!/usr/bin/env bash
set -euo pipefail

# Default to rebuilding TensorRT engine/tactics for the current tuning knobs.
# Set CLEAR_TRT_CACHE=0 to reuse existing cache.
if [[ "${CLEAR_TRT_CACHE:-1}" == "1" ]]; then
  rm -rf "${HOME}/.katago/trtcache"
  mkdir -p "${HOME}/.katago/trtcache"
fi

/opt/katago/katago benchmark \
  -model /opt/katago/weight/b18tf.onnx \
  -config /opt/katago/config/gtp_example.cfg \
  -v 10000 \
  -t 40 \
  -fixed-batch-size 10 \
  -override-config numNNServerThreadsPerModel=2,trtDeviceToUseThread0=0,trtDeviceToUseThread1=0,trtUseCudaGraph=true,trtBuilderOptimizationLevel=5,trtAvgTimingIterations=8,trtMaxAuxStreams=4
