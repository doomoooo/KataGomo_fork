#include "../neuralnet/cudabackend_sm120_kernels.h"

#include <cstring>

namespace Sm120Backend {

namespace {

extern "C" int sm120_search_ffn_batch();
extern "C" const char* sm120_search_ffn_id();
extern "C" cudaError_t sm120_search_ffn_launch(
  const half*, const half*, const half*, half*, cudaStream_t);
extern "C" int sm120_search_qkv_batch();
extern "C" const char* sm120_search_qkv_id();
extern "C" cudaError_t sm120_search_qkv_launch(
  const half*, const half*, half*, cudaStream_t);
extern "C" int sm120_search_linear2_batch();
extern "C" const char* sm120_search_linear2_id();
extern "C" cudaError_t sm120_search_linear2_launch(
  const half*, const half*, half*, cudaStream_t);
extern "C" int sm120_search_fa4_batch();
extern "C" const char* sm120_search_fa4_id();
extern "C" cudaError_t sm120_search_fa4_launch(
  void*, void*, void*, void*, int, int, int, int, float, cudaStream_t);

cudaError_t launchFfnCurrent(
  const half* input, const half* linearWeights, const half* gateWeights,
  half* output, cudaStream_t stream
) {
  launchFusedFFNB13(input, linearWeights, gateWeights, output, stream);
  return cudaPeekAtLastError();
}

cudaError_t launchFfnAReuse(
  const half* input, const half* linearWeights, const half* gateWeights,
  half* output, cudaStream_t stream
) {
  launchFusedFFNB13CandidateAReuse(
    input, linearWeights, gateWeights, output, stream);
  return cudaPeekAtLastError();
}

cudaError_t launchFfnSingleStream(
  const half* input, const half* linearWeights, const half* gateWeights,
  half* output, cudaStream_t stream
) {
  return launchFusedFFNB13S1(
    input, linearWeights, gateWeights, output, stream);
}

// Shape values live only in this generated-style registry. Backend operator
// code performs a data-driven lookup and contains no fixed-batch conditions.
// Explicit requested IDs are search candidates on any SM120 GPU. "auto"
// selects only an entry accepted for the exact GPU/batch/stream key.
const FusedFFNAotTactic ffnTactics[] = {
  {13, SM120_GPU_OTHER, 0, "ffn-m128-n64-k32-s2-mb3-exp", false, launchFfnCurrent},
  {13, SM120_GPU_RTX5090D, 2, "ffn-m128-n64-k32-s2-mb3-areuse-exp", true, launchFfnAReuse},
  {13, SM120_GPU_OTHER, 0, "ffn-m128-n64-k32-s3-single-stream-exp", false, launchFfnSingleStream},
};

const WideQKVAotTactic qkvTactics[] = {
  {13, SM120_GPU_RTX5090D, 2, "qkv-m128-n128-k64-s2-tilelang-planar", true, launchWideQKVB13},
  {13, SM120_GPU_OTHER, 0, "qkv-m128-n128-k32-s3-tilelang-planar", false, launchWideQKVB13S1},
};

const ResidualGemmAotTactic residualTactics[] = {
  {13, SM120_GPU_RTX5090D, 2, 1152, "linear2-m128-n128-k32-s4-tilelang-64k", true, launchLinear2ResidualB13},
  {13, SM120_GPU_OTHER, 0, 1152, "linear2-balanced-b13", false, launchLinear2ResidualB13Balanced},
  {13, SM120_GPU_OTHER, 0, 384, "outproj-m128-n128-k32-s4-tilelang-64k", false, launchOutProjectionResidualB13},
};

const FusedFFNAotTactic searchFfnTactic = {
  sm120_search_ffn_batch(), SM120_GPU_OTHER, 0,
  sm120_search_ffn_id(), false, sm120_search_ffn_launch};
const WideQKVAotTactic searchQkvTactic = {
  sm120_search_qkv_batch(), SM120_GPU_OTHER, 0,
  sm120_search_qkv_id(), false, sm120_search_qkv_launch};
const ResidualGemmAotTactic searchLinear2Tactic = {
  sm120_search_linear2_batch(), SM120_GPU_OTHER, 0, 1152,
  sm120_search_linear2_id(), false, sm120_search_linear2_launch};
const FA4AotTactic searchFA4Tactic = {
  sm120_search_fa4_batch(), sm120_search_fa4_id(), sm120_search_fa4_launch};

template<typename T, size_t N>
const T* findTactic(
  const T (&tactics)[N], int batchSize, int gpuClass, int streams,
  const char* requestedId
) {
  const bool automatic = requestedId == nullptr || std::strcmp(requestedId, "auto") == 0;
  for(const T& tactic: tactics) {
    if(tactic.batchSize != batchSize)
      continue;
    if(automatic) {
      if(tactic.automaticWinner && tactic.gpuClass == gpuClass && tactic.streams == streams)
        return &tactic;
    }
    else if(std::strcmp(tactic.id, requestedId) == 0) {
      return &tactic;
    }
  }
  return nullptr;
}

} // namespace

const FusedFFNAotTactic* findFusedFFNAotTactic(
  int batchSize, int gpuClass, int streams, const char* requestedId
) {
  const FusedFFNAotTactic* tactic = findTactic(
    ffnTactics, batchSize, gpuClass, streams, requestedId);
  if(tactic != nullptr)
    return tactic;
  return requestedId != nullptr && searchFfnTactic.batchSize == batchSize &&
    std::strcmp(searchFfnTactic.id, requestedId) == 0 ? &searchFfnTactic : nullptr;
}

const WideQKVAotTactic* findWideQKVAotTactic(
  int batchSize, int gpuClass, int streams, const char* requestedId
) {
  const WideQKVAotTactic* tactic = findTactic(
    qkvTactics, batchSize, gpuClass, streams, requestedId);
  if(tactic != nullptr)
    return tactic;
  return requestedId != nullptr && searchQkvTactic.batchSize == batchSize &&
    std::strcmp(searchQkvTactic.id, requestedId) == 0 ? &searchQkvTactic : nullptr;
}

const ResidualGemmAotTactic* findResidualGemmAotTactic(
  int batchSize, int gpuClass, int streams, int inputChannels,
  const char* requestedId
) {
  if(inputChannels == searchLinear2Tactic.inputChannels &&
     searchLinear2Tactic.batchSize == batchSize && requestedId != nullptr &&
     std::strcmp(searchLinear2Tactic.id, requestedId) == 0)
    return &searchLinear2Tactic;
  const bool automatic = requestedId == nullptr || std::strcmp(requestedId, "auto") == 0;
  for(const ResidualGemmAotTactic& tactic: residualTactics) {
    if(tactic.batchSize != batchSize || tactic.inputChannels != inputChannels)
      continue;
    if(automatic) {
      if(tactic.automaticWinner && tactic.gpuClass == gpuClass && tactic.streams == streams)
        return &tactic;
    }
    else if(std::strcmp(tactic.id, requestedId) == 0) {
      return &tactic;
    }
  }
  return nullptr;
}

const FA4AotTactic* findFA4AotTactic(
  int batchSize, const char* requestedId
) {
  return requestedId != nullptr && searchFA4Tactic.batchSize == batchSize &&
    std::strcmp(searchFA4Tactic.id, requestedId) == 0 ? &searchFA4Tactic : nullptr;
}

} // namespace Sm120Backend
