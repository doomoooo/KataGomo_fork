#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

CLEAR_TRT_CACHE=0
TRT_BUILDER_OPT_LEVEL=-1
TRT_AVG_TIMING_ITERS=-1
TRT_MAX_AUX_STREAMS=-1
TRT_HOST_WAIT_POLICY=blocking
NN_MAX_BATCHSIZE=4
NUM_SEARCH_THREADS=16
# Number of NN server threads (CUDA streams) per GPU.
TRT_CUDA_STREAMS=4
TRT_DEVICE_ID=0
# Optional multi-GPU mapping. Example: "0,1,2".
# If non-empty, this overrides TRT_DEVICE_ID and each listed device gets TRT_CUDA_STREAMS threads.
TRT_DEVICE_IDS=""

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

declare -a TRT_DEVICE_ID_LIST=()
if [[ -n "${TRT_DEVICE_IDS//[[:space:]]/}" ]]; then
  IFS=',' read -r -a RAW_TRT_DEVICE_IDS <<< "${TRT_DEVICE_IDS}"
  for raw_id in "${RAW_TRT_DEVICE_IDS[@]}"; do
    id="${raw_id//[[:space:]]/}"
    if [[ -z "${id}" ]]; then
      echo "TRT_DEVICE_IDS contains an empty item: ${TRT_DEVICE_IDS}" >&2
      exit 1
    fi
    if ! [[ "${id}" =~ ^[0-9]+$ ]]; then
      echo "TRT_DEVICE_IDS must be a comma-separated list of non-negative integers, got: ${TRT_DEVICE_IDS}" >&2
      exit 1
    fi
    TRT_DEVICE_ID_LIST+=("${id}")
  done
else
  TRT_DEVICE_ID_LIST=("${TRT_DEVICE_ID}")
fi

if ! [[ "${NN_MAX_BATCHSIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NN_MAX_BATCHSIZE must be a positive integer, got: ${NN_MAX_BATCHSIZE}" >&2
  exit 1
fi

NUM_NN_SERVER_THREADS=$(( TRT_CUDA_STREAMS * ${#TRT_DEVICE_ID_LIST[@]} ))
OVERRIDE_CONFIG="numNNServerThreadsPerModel=${NUM_NN_SERVER_THREADS}"
for ((thread_idx=0; thread_idx<NUM_NN_SERVER_THREADS; thread_idx++)); do
  # Assign exactly TRT_CUDA_STREAMS threads to each device.
  device_list_idx=$(( thread_idx / TRT_CUDA_STREAMS ))
  mapped_device_id="${TRT_DEVICE_ID_LIST[${device_list_idx}]}"
  OVERRIDE_CONFIG+=",trtDeviceToUseThread${thread_idx}=${mapped_device_id}"
done

OVERRIDE_CONFIG+=",trtUseCudaGraph=true"
OVERRIDE_CONFIG+=",analysisPVLen=99"
OVERRIDE_CONFIG+=",useEvalCache=true"
OVERRIDE_CONFIG+=",trtBuilderOptimizationLevel=${TRT_BUILDER_OPT_LEVEL}"
OVERRIDE_CONFIG+=",trtAvgTimingIterations=${TRT_AVG_TIMING_ITERS}"
OVERRIDE_CONFIG+=",trtMaxAuxStreams=${TRT_MAX_AUX_STREAMS}"
OVERRIDE_CONFIG+=",trtHostWaitPolicy=${TRT_HOST_WAIT_POLICY}"
OVERRIDE_CONFIG+=",nnMaxBatchSize=${NN_MAX_BATCHSIZE}"
OVERRIDE_CONFIG+=",numSearchThreads=${NUM_SEARCH_THREADS}"

"${KATAGO_BIN_PATH}" gtp \
  -model "${KATAGO_MODEL_PATH}" \
  -config "${KATAGO_CONFIG_PATH}" \
  -override-config "${OVERRIDE_CONFIG}"
