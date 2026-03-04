#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

TRT_BUILDER_OPT_LEVEL=-1
TRT_AVG_TIMING_ITERS=-1
TRT_MAX_AUX_STREAMS=8
TRT_HOST_WAIT_POLICY=blocking
TRT_REBUILD_PLAN_CACHE=false
NN_MAX_BATCHSIZE=4
NUM_SEARCH_THREADS=16
TRT_CUDA_STREAMS=4
BENCH_VISITS=100000
SEARCH_RAW_STATS_MAX_ROWS_PER_THREAD=1000000

TRT_DEVICE_ID=0
KATAGO_BIN="${KATAGO_BIN_PATH}"
KATAGO_MODEL="${KATAGO_MODEL_PATH}"
KATAGO_CONFIG="${KATAGO_CONFIG_PATH}"
# Keep no-arg behavior: edit defaults above directly; CLI args only override.

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --katago-bin PATH             Path to katago executable (default: ${KATAGO_BIN})
  --model PATH                  Path to model file (default: ${KATAGO_MODEL})
  --config PATH                 Path to config file (default: ${KATAGO_CONFIG})
  --visits N                    Benchmark visits (default: ${BENCH_VISITS})
  --search-threads N            Number of search threads (default: ${NUM_SEARCH_THREADS})
  --batch-size N                Fixed batch size (default: ${NN_MAX_BATCHSIZE})
  --cuda-streams N              numNNServerThreadsPerModel (default: ${TRT_CUDA_STREAMS})
  --device-id N                 CUDA device id for all NN threads (default: ${TRT_DEVICE_ID})
  --trt-builder-opt-level N     TensorRT builder optimization level (default: ${TRT_BUILDER_OPT_LEVEL})
  --trt-avg-timing-iters N      TensorRT avg timing iters (default: ${TRT_AVG_TIMING_ITERS})
  --trt-max-aux-streams N       TensorRT max aux streams (default: ${TRT_MAX_AUX_STREAMS})
  --trt-host-wait-policy VALUE  TensorRT host wait policy: auto|spin|yield|blocking
  --trt-rebuild-plan-cache      Rebuild the current TensorRT plan cache key (does not clear whole trtcache)
  -h, --help                    Show this help message
EOF
}

require_value() {
  local opt_name="$1"
  local opt_value="${2:-}"
  if [[ -z "${opt_value}" ]]; then
    echo "Missing value for ${opt_name}" >&2
    print_usage >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --katago-bin)
        require_value "$1" "${2:-}"
        KATAGO_BIN="$2"
        shift 2
        ;;
      --model)
        require_value "$1" "${2:-}"
        KATAGO_MODEL="$2"
        shift 2
        ;;
      --config)
        require_value "$1" "${2:-}"
        KATAGO_CONFIG="$2"
        shift 2
        ;;
      --visits)
        require_value "$1" "${2:-}"
        BENCH_VISITS="$2"
        shift 2
        ;;
      --search-threads)
        require_value "$1" "${2:-}"
        NUM_SEARCH_THREADS="$2"
        shift 2
        ;;
      --batch-size)
        require_value "$1" "${2:-}"
        NN_MAX_BATCHSIZE="$2"
        shift 2
        ;;
      --cuda-streams)
        require_value "$1" "${2:-}"
        TRT_CUDA_STREAMS="$2"
        shift 2
        ;;
      --device-id)
        require_value "$1" "${2:-}"
        TRT_DEVICE_ID="$2"
        shift 2
        ;;
      --trt-builder-opt-level)
        require_value "$1" "${2:-}"
        TRT_BUILDER_OPT_LEVEL="$2"
        shift 2
        ;;
      --trt-avg-timing-iters)
        require_value "$1" "${2:-}"
        TRT_AVG_TIMING_ITERS="$2"
        shift 2
        ;;
      --trt-max-aux-streams)
        require_value "$1" "${2:-}"
        TRT_MAX_AUX_STREAMS="$2"
        shift 2
        ;;
      --trt-host-wait-policy)
        require_value "$1" "${2:-}"
        TRT_HOST_WAIT_POLICY="$2"
        shift 2
        ;;
      --trt-rebuild-plan-cache)
        TRT_REBUILD_PLAN_CACHE=true
        shift
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        print_usage >&2
        exit 1
        ;;
    esac
  done
}

if [[ $# -gt 0 ]]; then
  parse_args "$@"
fi

if ! [[ "${TRT_CUDA_STREAMS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "numNNServerThreadsPerModel must be a positive integer, got: ${TRT_CUDA_STREAMS}" >&2
  exit 1
fi

if ! [[ "${BENCH_VISITS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BENCH_VISITS must be a positive integer, got: ${BENCH_VISITS}" >&2
  exit 1
fi

if ! [[ "${NUM_SEARCH_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SEARCH_THREADS must be a positive integer, got: ${NUM_SEARCH_THREADS}" >&2
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

if ! [[ "${TRT_BUILDER_OPT_LEVEL}" =~ ^-?[0-9]+$ ]]; then
  echo "TRT_BUILDER_OPT_LEVEL must be an integer, got: ${TRT_BUILDER_OPT_LEVEL}" >&2
  exit 1
fi

if ! [[ "${TRT_AVG_TIMING_ITERS}" =~ ^-?[0-9]+$ ]]; then
  echo "TRT_AVG_TIMING_ITERS must be an integer, got: ${TRT_AVG_TIMING_ITERS}" >&2
  exit 1
fi

if ! [[ "${TRT_MAX_AUX_STREAMS}" =~ ^-?[0-9]+$ ]]; then
  echo "TRT_MAX_AUX_STREAMS must be an integer, got: ${TRT_MAX_AUX_STREAMS}" >&2
  exit 1
fi

BENCHMARK_OUT_DIR="${SCRIPT_DIR}/benchmark"
mkdir -p "${BENCHMARK_OUT_DIR}"
RAW_STATS_FILE="${BENCHMARK_OUT_DIR}/search_thread_raw_stats_t${NUM_SEARCH_THREADS}_s${TRT_CUDA_STREAMS}_b${NN_MAX_BATCHSIZE}_v${BENCH_VISITS}_$(date +%Y%m%d_%H%M%S).tsv"

OVERRIDE_CONFIG="numNNServerThreadsPerModel=${TRT_CUDA_STREAMS}"
for ((thread_idx=0; thread_idx<TRT_CUDA_STREAMS; thread_idx++)); do
  OVERRIDE_CONFIG+=",trtDeviceToUseThread${thread_idx}=${TRT_DEVICE_ID}"
done

OVERRIDE_CONFIG+=",trtUseCudaGraph=true"
OVERRIDE_CONFIG+=",trtBuilderOptimizationLevel=${TRT_BUILDER_OPT_LEVEL}"
OVERRIDE_CONFIG+=",trtAvgTimingIterations=${TRT_AVG_TIMING_ITERS}"
OVERRIDE_CONFIG+=",trtMaxAuxStreams=${TRT_MAX_AUX_STREAMS}"
OVERRIDE_CONFIG+=",trtHostWaitPolicy=${TRT_HOST_WAIT_POLICY}"
OVERRIDE_CONFIG+=",trtRebuildPlanCache=${TRT_REBUILD_PLAN_CACHE}"
OVERRIDE_CONFIG+=",useGraphSearch=true"
OVERRIDE_CONFIG+=",useEvalCache=true"
OVERRIDE_CONFIG+=",searchThreadRawStatsFile=${RAW_STATS_FILE}"
OVERRIDE_CONFIG+=",searchThreadRawStatsMaxRowsPerThread=${SEARCH_RAW_STATS_MAX_ROWS_PER_THREAD}"

echo "searchThreadRawStatsFile: ${RAW_STATS_FILE}"

"${KATAGO_BIN}" benchmark \
  -model "${KATAGO_MODEL}" \
  -config "${KATAGO_CONFIG}" \
  -v "${BENCH_VISITS}" \
  -t "${NUM_SEARCH_THREADS}" \
  -fixed-batch-size "${NN_MAX_BATCHSIZE}" \
  -override-config "${OVERRIDE_CONFIG}"
