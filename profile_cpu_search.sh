#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "${SCRIPT_DIR}/env.sh"

VISITS=50000
THREADS=20
BATCH_SIZE=4
OUT_DIR="/tmp/katago_logs"
TAG=""
TAG_SET=0

BINARY="${KATAGO_BIN_PATH}"
MODEL="${KATAGO_MODEL_PATH}"
CONFIG="${KATAGO_CONFIG_PATH}"

TRT_CUDA_STREAMS="${TRT_CUDA_STREAMS:-4}"
TRT_DEVICE_ID="${TRT_DEVICE_ID:-0}"
TRT_BUILDER_OPT_LEVEL="${TRT_BUILDER_OPT_LEVEL:--1}"
TRT_AVG_TIMING_ITERS="${TRT_AVG_TIMING_ITERS:--1}"
TRT_MAX_AUX_STREAMS="${TRT_MAX_AUX_STREAMS:--1}"
TRT_SET_TACTIC_SOURCES="${TRT_SET_TACTIC_SOURCES:-true}"
TRT_MULTI_PROFILE="${TRT_MULTI_PROFILE:-true}"

FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-/tmp/FlameGraph}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./profile_cpu_search.sh [options]

Options:
  -v, --visits <N>       Visits for benchmark (default: 50000)
  -t, --threads <N>      Search threads for benchmark (default: 20)
  -b, --batch-size <N>   Fixed NN batch size (default: 4)
  --tag <STR>            Output tag (default: t<threads>_v<visits>)
  --out-dir <DIR>        Output directory (default: /tmp/katago_logs)
  --bin <PATH>           KataGo binary path (default: KATAGO_BIN_PATH)
  --model <PATH>         Model path (default: KATAGO_MODEL_PATH)
  --config <PATH>        Config path (default: KATAGO_CONFIG_PATH)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--visits)
      [[ $# -ge 2 ]] || print_error "$1 requires a value"
      VISITS="$2"
      shift 2
      ;;
    -t|--threads)
      [[ $# -ge 2 ]] || print_error "$1 requires a value"
      THREADS="$2"
      shift 2
      ;;
    -b|--batch-size)
      [[ $# -ge 2 ]] || print_error "$1 requires a value"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --tag)
      [[ $# -ge 2 ]] || print_error "--tag requires a value"
      TAG="$2"
      TAG_SET=1
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || print_error "--out-dir requires a value"
      OUT_DIR="$2"
      shift 2
      ;;
    --bin)
      [[ $# -ge 2 ]] || print_error "--bin requires a value"
      BINARY="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || print_error "--model requires a value"
      MODEL="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || print_error "--config requires a value"
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print_error "Unknown argument: $1"
      ;;
  esac
done

if [[ "${TAG_SET}" == "0" ]]; then
  TAG="t${THREADS}_v${VISITS}"
fi

[[ "${VISITS}" =~ ^[1-9][0-9]*$ ]] || print_error "Invalid visits: ${VISITS}"
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || print_error "Invalid threads: ${THREADS}"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || print_error "Invalid batch size: ${BATCH_SIZE}"
[[ "${TRT_CUDA_STREAMS}" =~ ^[1-9][0-9]*$ ]] || print_error "Invalid TRT_CUDA_STREAMS: ${TRT_CUDA_STREAMS}"
[[ "${TRT_DEVICE_ID}" =~ ^[0-9]+$ ]] || print_error "Invalid TRT_DEVICE_ID: ${TRT_DEVICE_ID}"

command -v rg >/dev/null 2>&1 || print_error "rg not found"
command -v perl >/dev/null 2>&1 || print_error "perl not found"
command -v git >/dev/null 2>&1 || print_error "git not found"

PPROF_BIN=""
if command -v google-pprof >/dev/null 2>&1; then
  PPROF_BIN="$(command -v google-pprof)"
fi

if [[ -z "${PPROF_BIN}" ]]; then
  print_warn "google-pprof not found, trying to install google-perftools"
  sudo apt-get update >/dev/null
  sudo apt-get install -y google-perftools >/dev/null
  if command -v google-pprof >/dev/null 2>&1; then
    PPROF_BIN="$(command -v google-pprof)"
  fi
fi

[[ -n "${PPROF_BIN}" ]] || print_error "google-pprof unavailable"

LIBPROFILER_PATH="$(ldconfig -p 2>/dev/null | awk '/libprofiler\.so/{print $NF; exit}')"
if [[ -z "${LIBPROFILER_PATH}" ]]; then
  for candidate in \
    /usr/lib/x86_64-linux-gnu/libprofiler.so \
    /usr/lib/x86_64-linux-gnu/libprofiler.so.4 \
    /usr/lib/x86_64-linux-gnu/libprofiler.so.0; do
    if [[ -r "${candidate}" ]]; then
      LIBPROFILER_PATH="${candidate}"
      break
    fi
  done
fi
[[ -r "${LIBPROFILER_PATH}" ]] || print_error "libprofiler.so not found"

if [[ ! -f "${FLAMEGRAPH_DIR}/flamegraph.pl" ]]; then
  print_info "Preparing FlameGraph at ${FLAMEGRAPH_DIR}"
  rm -rf "${FLAMEGRAPH_DIR}"
  git clone --depth=1 https://github.com/brendangregg/FlameGraph "${FLAMEGRAPH_DIR}" >/dev/null
fi

[[ -x "${BINARY}" ]] || print_error "Binary not found: ${BINARY}"
[[ -f "${MODEL}" ]] || print_error "Model not found: ${MODEL}"
[[ -f "${CONFIG}" ]] || print_error "Config not found: ${CONFIG}"

mkdir -p "${OUT_DIR}"

PROFILE_PATH="${OUT_DIR}/katago_cpu_${TAG}.prof"
BENCH_LOG="${OUT_DIR}/katago_cpu_${TAG}_benchmark.log"
PPROF_TXT="${OUT_DIR}/katago_cpu_${TAG}_pprof.txt"
COLLAPSED="${OUT_DIR}/katago_cpu_${TAG}.collapsed"
FLAME_SVG="${OUT_DIR}/katago_cpu_${TAG}_flame.svg"
NOREC_COLLAPSED="${OUT_DIR}/katago_cpu_${TAG}_norec.collapsed"
NOREC_FLAME_SVG="${OUT_DIR}/katago_cpu_${TAG}_norec_flame.svg"
FOCUS_SEARCH_TXT="${OUT_DIR}/katago_cpu_${TAG}_focus_search.txt"
FOCUS_LOCKS_TXT="${OUT_DIR}/katago_cpu_${TAG}_focus_locks_maps.txt"
FOCUS_SEARCH_COLLAPSED="${OUT_DIR}/katago_cpu_${TAG}_focus_search.collapsed"
FOCUS_SEARCH_FLAME="${OUT_DIR}/katago_cpu_${TAG}_focus_search_flame.svg"
FOCUS_SEARCH_NOREC_COLLAPSED="${OUT_DIR}/katago_cpu_${TAG}_focus_search_norec.collapsed"
FOCUS_SEARCH_NOREC_FLAME="${OUT_DIR}/katago_cpu_${TAG}_focus_search_norec_flame.svg"

FOCUS_SEARCH_REGEX='Search::|Board::|BoardHistory::|GraphHash::|NNInputs::fillRowV7|iterLadders|NNEvaluator::evaluate|EvalCacheTable|allocateOrFindNode|recomputeNodeStats|selectBestChildToDescend|playoutDescend|SearchNode'
FOCUS_LOCKS_REGEX='EvalCacheTable::|allocateOrFindNode|SearchNodeTable|std::_Rb_tree|std::map|MutexPool::getMutex|pthread_mutex_lock|pthread_mutex_unlock|std::mutex::lock|std::mutex::unlock|std::unique_lock::lock|std::lock_guard'

rm -f \
  "${PROFILE_PATH}" \
  "${BENCH_LOG}" \
  "${PPROF_TXT}" \
  "${COLLAPSED}" \
  "${FLAME_SVG}" \
  "${NOREC_COLLAPSED}" \
  "${NOREC_FLAME_SVG}" \
  "${FOCUS_SEARCH_TXT}" \
  "${FOCUS_LOCKS_TXT}" \
  "${FOCUS_SEARCH_COLLAPSED}" \
  "${FOCUS_SEARCH_FLAME}" \
  "${FOCUS_SEARCH_NOREC_COLLAPSED}" \
  "${FOCUS_SEARCH_NOREC_FLAME}"

OVERRIDE_CONFIG="useGraphSearch=true,useEvalCache=true,numNNServerThreadsPerModel=${TRT_CUDA_STREAMS}"
for ((thread_idx=0; thread_idx<TRT_CUDA_STREAMS; thread_idx++)); do
  OVERRIDE_CONFIG+=",trtDeviceToUseThread${thread_idx}=${TRT_DEVICE_ID}"
done
OVERRIDE_CONFIG+=",trtUseCudaGraph=true"
OVERRIDE_CONFIG+=",trtBuilderOptimizationLevel=${TRT_BUILDER_OPT_LEVEL}"
OVERRIDE_CONFIG+=",trtAvgTimingIterations=${TRT_AVG_TIMING_ITERS}"
OVERRIDE_CONFIG+=",trtMaxAuxStreams=${TRT_MAX_AUX_STREAMS}"
OVERRIDE_CONFIG+=",trtSetTacticSources=${TRT_SET_TACTIC_SOURCES}"
OVERRIDE_CONFIG+=",trtMultiProfile=${TRT_MULTI_PROFILE}"
OVERRIDE_CONFIG+=",nnMinBatchSize=${BATCH_SIZE}"

print_info "Profiling ${BINARY}"
print_info "Output tag: ${TAG}"
print_info "Writing logs to ${OUT_DIR}"

export LD_LIBRARY_PATH="${TENSORRT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LD_PRELOAD="${LIBPROFILER_PATH}" \
CPUPROFILE="${PROFILE_PATH}" \
"${BINARY}" benchmark \
  -model "${MODEL}" \
  -config "${CONFIG}" \
  -v "${VISITS}" \
  -t "${THREADS}" \
  -fixed-batch-size "${BATCH_SIZE}" \
  -override-config "${OVERRIDE_CONFIG}" \
  2>&1 | tee "${BENCH_LOG}"

[[ -s "${PROFILE_PATH}" ]] || print_error "Profile file is empty: ${PROFILE_PATH}"

print_info "Generating pprof reports"
"${PPROF_BIN}" --text "${BINARY}" "${PROFILE_PATH}" > "${PPROF_TXT}"
"${PPROF_BIN}" --collapsed "${BINARY}" "${PROFILE_PATH}" > "${COLLAPSED}"
"${PPROF_BIN}" --text --focus="${FOCUS_SEARCH_REGEX}" "${BINARY}" "${PROFILE_PATH}" > "${FOCUS_SEARCH_TXT}"
"${PPROF_BIN}" --text --focus="${FOCUS_LOCKS_REGEX}" "${BINARY}" "${PROFILE_PATH}" > "${FOCUS_LOCKS_TXT}"

print_info "Generating flamegraphs"
perl "${FLAMEGRAPH_DIR}/flamegraph.pl" "${COLLAPSED}" > "${FLAME_SVG}"
perl "${FLAMEGRAPH_DIR}/stackcollapse-recursive.pl" "${COLLAPSED}" > "${NOREC_COLLAPSED}"
perl "${FLAMEGRAPH_DIR}/flamegraph.pl" "${NOREC_COLLAPSED}" > "${NOREC_FLAME_SVG}"
rg -N "${FOCUS_SEARCH_REGEX}" "${COLLAPSED}" > "${FOCUS_SEARCH_COLLAPSED}" || true
if [[ -s "${FOCUS_SEARCH_COLLAPSED}" ]]; then
  perl "${FLAMEGRAPH_DIR}/flamegraph.pl" "${FOCUS_SEARCH_COLLAPSED}" > "${FOCUS_SEARCH_FLAME}"
  perl "${FLAMEGRAPH_DIR}/stackcollapse-recursive.pl" "${FOCUS_SEARCH_COLLAPSED}" > "${FOCUS_SEARCH_NOREC_COLLAPSED}"
  perl "${FLAMEGRAPH_DIR}/flamegraph.pl" "${FOCUS_SEARCH_NOREC_COLLAPSED}" > "${FOCUS_SEARCH_NOREC_FLAME}"
else
  print_warn "Focused search collapsed profile is empty, skip focused flamegraph"
fi

FINAL_BENCH_LINE="$(tr '\r' '\n' < "${BENCH_LOG}" | rg -N 'visits/s = .* nnEvals/s' | tail -n 1 || true)"
if [[ -n "${FINAL_BENCH_LINE}" ]]; then
  print_info "Benchmark summary: ${FINAL_BENCH_LINE}"
fi

print_info "Done"
print_info "  ${PROFILE_PATH}"
print_info "  ${BENCH_LOG}"
print_info "  ${PPROF_TXT}"
print_info "  ${COLLAPSED}"
print_info "  ${FLAME_SVG}"
print_info "  ${NOREC_FLAME_SVG}"
print_info "  ${FOCUS_SEARCH_TXT}"
print_info "  ${FOCUS_LOCKS_TXT}"
if [[ -s "${FOCUS_SEARCH_FLAME}" ]]; then
  print_info "  ${FOCUS_SEARCH_FLAME}"
  print_info "  ${FOCUS_SEARCH_NOREC_FLAME}"
fi
