#!/usr/bin/env bash
set -euo pipefail

# Default to rebuilding TensorRT engine/tactics for the current tuning knobs.
# Set CLEAR_TRT_CACHE=0 to reuse existing cache.
if [[ "${CLEAR_TRT_CACHE:-1}" == "1" ]]; then
  rm -rf "${HOME}/.katago/trtcache"
  mkdir -p "${HOME}/.katago/trtcache"
fi

TRT_BUILDER_OPT_LEVEL=5
TRT_AVG_TIMING_ITERS=8
TRT_MAX_AUX_STREAMS=4
TRT_SET_TACTIC_SOURCES=true
TRT_NUM_OPT_PROFILES=3

OVERRIDE_CONFIG="numNNServerThreadsPerModel=2"
OVERRIDE_CONFIG+=",trtDeviceToUseThread0=0"
OVERRIDE_CONFIG+=",trtDeviceToUseThread1=0"
OVERRIDE_CONFIG+=",trtUseCudaGraph=true"
OVERRIDE_CONFIG+=",trtBuilderOptimizationLevel=${TRT_BUILDER_OPT_LEVEL}"
OVERRIDE_CONFIG+=",trtAvgTimingIterations=${TRT_AVG_TIMING_ITERS}"
OVERRIDE_CONFIG+=",trtMaxAuxStreams=${TRT_MAX_AUX_STREAMS}"
OVERRIDE_CONFIG+=",trtSetTacticSources=${TRT_SET_TACTIC_SOURCES}"
OVERRIDE_CONFIG+=",trtNumOptimizationProfiles=${TRT_NUM_OPT_PROFILES}"

/opt/katago/katago benchmark \
  -model /opt/katago/weight/b18tf.onnx \
  -config /opt/katago/config/gtp_example.cfg \
  -v 10000 \
  -t 40 \
  -fixed-batch-size 10 \
  -override-config "${OVERRIDE_CONFIG}"
