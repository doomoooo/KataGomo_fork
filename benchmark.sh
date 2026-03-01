#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CLEAR_TRT_CACHE=0
TRT_BUILDER_OPT_LEVEL=-1
TRT_AVG_TIMING_ITERS=-1
TRT_MAX_AUX_STREAMS=-1
TRT_SET_TACTIC_SOURCES=true
TRT_MULTI_PROFILE=true
NN_MAX_BATCHSIZE=4
NN_MIN_BATCHSIZE=4
NUM_SEARCH_THREADS=16
TRT_CUDA_STREAMS=4
TRT_DEVICE_ID=0

# Default to rebuilding TensorRT engine/tactics for the current tuning knobs.
# Set CLEAR_TRT_CACHE=0 to reuse existing cache.
if [[ "${CLEAR_TRT_CACHE}" == "1" ]]; then
  rm -rf "${HOME}/.katago/trtcache"
  mkdir -p "${HOME}/.katago/trtcache"
fi

if ! [[ "${TRT_CUDA_STREAMS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "numNNServerThreadsPerModel must be a positive integer, got: ${TRT_CUDA_STREAMS}" >&2
  exit 1
fi

if ! [[ "${TRT_DEVICE_ID}" =~ ^[0-9]+$ ]]; then
  echo "TRT_DEVICE_ID must be a non-negative integer, got: ${TRT_DEVICE_ID}" >&2
  exit 1
fi

if ! [[ "${NN_MAX_BATCHSIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NN_MAX_BATCHSIZE must be a positive integer, got: ${NN_MAX_BATCHSIZE}" >&2
  exit 1
fi

if ! [[ "${NN_MIN_BATCHSIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NN_MIN_BATCHSIZE must be a positive integer, got: ${NN_MIN_BATCHSIZE}" >&2
  exit 1
fi

if (( NN_MIN_BATCHSIZE > NN_MAX_BATCHSIZE )); then
  echo "NN_MIN_BATCHSIZE (${NN_MIN_BATCHSIZE}) must be <= NN_MAX_BATCHSIZE (${NN_MAX_BATCHSIZE})" >&2
  exit 1
fi

OVERRIDE_CONFIG="numNNServerThreadsPerModel=${TRT_CUDA_STREAMS}"
for ((thread_idx=0; thread_idx<TRT_CUDA_STREAMS; thread_idx++)); do
  OVERRIDE_CONFIG+=",trtDeviceToUseThread${thread_idx}=${TRT_DEVICE_ID}"
done

OVERRIDE_CONFIG+=",trtUseCudaGraph=true"
OVERRIDE_CONFIG+=",trtBuilderOptimizationLevel=${TRT_BUILDER_OPT_LEVEL}"
OVERRIDE_CONFIG+=",trtAvgTimingIterations=${TRT_AVG_TIMING_ITERS}"
OVERRIDE_CONFIG+=",trtMaxAuxStreams=${TRT_MAX_AUX_STREAMS}"
OVERRIDE_CONFIG+=",trtSetTacticSources=${TRT_SET_TACTIC_SOURCES}"
OVERRIDE_CONFIG+=",trtMultiProfile=${TRT_MULTI_PROFILE}"
OVERRIDE_CONFIG+=",nnMinBatchSize=${NN_MIN_BATCHSIZE}"

"${KATAGO_DEPLOY_DIR}/katago" benchmark \
  -model "${KATAGO_MODEL_PATH}" \
  -config "${KATAGO_CONFIG_PATH}" \
  -v 10000 \
  -t "${NUM_SEARCH_THREADS}" \
  -fixed-batch-size "${NN_MAX_BATCHSIZE}" \
  -override-config "${OVERRIDE_CONFIG}"
