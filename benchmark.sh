#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

CLEAR_TRT_CACHE=0
TRT_BUILDER_OPT_LEVEL=-1
TRT_AVG_TIMING_ITERS=-1
TRT_MAX_AUX_STREAMS=8
TRT_HOST_WAIT_POLICY=blocking
NN_MAX_BATCHSIZE=10
NUM_SEARCH_THREADS=33
TRT_CUDA_STREAMS=2
BENCH_VISITS=10000

TRT_DEVICE_ID=0
KATAGO_BIN="${KATAGO_BIN_PATH}"
KATAGO_MODEL="${KATAGO_MODEL_PATH}"
KATAGO_CONFIG="${KATAGO_CONFIG_PATH}"

# Optional overrides for PGO profiling runs.
if [[ -n "${PGO_BENCH_VISITS+x}" ]]; then BENCH_VISITS="${PGO_BENCH_VISITS}"; fi
if [[ -n "${PGO_KATAGO_BIN_PATH_OVERRIDE+x}" ]]; then KATAGO_BIN="${PGO_KATAGO_BIN_PATH_OVERRIDE}"; fi

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

TRT_HOST_WAIT_POLICY="$(echo "${TRT_HOST_WAIT_POLICY}" | tr '[:upper:]' '[:lower:]')"
if ! [[ "${TRT_HOST_WAIT_POLICY}" =~ ^(auto|spin|yield|blocking)$ ]]; then
  echo "TRT_HOST_WAIT_POLICY must be one of: auto|spin|yield|blocking, got: ${TRT_HOST_WAIT_POLICY}" >&2
  exit 1
fi

if ! [[ "${NN_MAX_BATCHSIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NN_MAX_BATCHSIZE must be a positive integer, got: ${NN_MAX_BATCHSIZE}" >&2
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
OVERRIDE_CONFIG+=",trtHostWaitPolicy=${TRT_HOST_WAIT_POLICY}"
OVERRIDE_CONFIG+=",useGraphSearch=true"
OVERRIDE_CONFIG+=",useEvalCache=true"

"${KATAGO_BIN}" benchmark \
  -model "${KATAGO_MODEL}" \
  -config "${KATAGO_CONFIG}" \
  -v "${BENCH_VISITS}" \
  -t "${NUM_SEARCH_THREADS}" \
  -fixed-batch-size "${NN_MAX_BATCHSIZE}" \
  -override-config "${OVERRIDE_CONFIG}"
