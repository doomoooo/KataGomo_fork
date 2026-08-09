#include "../neuralnet/cudabackend_sm120_kernels.h"

#include <cstring>
#include <string>

namespace Sm120Backend {

namespace {

extern "C" int sm120_search_ffn_batch();
extern "C" const char* sm120_search_ffn_id();
extern "C" cudaError_t sm120_search_ffn_launch(
  const half*, const half*, const half*, half*, cudaStream_t);
extern "C" int sm120_search_qkv_batch();
extern "C" const char* sm120_search_qkv_id();
extern "C" int sm120_search_qkv_packed();
extern "C" cudaError_t sm120_search_qkv_launch(
  const half*, const half*, half*, cudaStream_t);
extern "C" int sm120_search_linear2_batch();
extern "C" const char* sm120_search_linear2_id();
extern "C" cudaError_t sm120_search_linear2_launch(
  const half*, const half*, half*, int, cudaStream_t);
extern "C" int sm120_search_outproj_batch();
extern "C" const char* sm120_search_outproj_id();
extern "C" cudaError_t sm120_search_outproj_launch(
  const half*, const half*, half*, int, cudaStream_t);
extern "C" int sm120_search_fa4_batch();
extern "C" const char* sm120_search_fa4_id();
extern "C" cudaError_t sm120_search_fa4_launch(
  void*, void*, void*, void*, int, int, int, int, float, bool, cudaStream_t);

// Runtime lookup is explicit-ID only. All historical shapes, including the
// former B13 implementations, are emitted by the B4-B32 fat generators.
const ResidualGemmAotTactic residualTactics[] = {
#define KATAGO_CUTLASS_RESIDUAL_BATCH(B) \
  {B, 1152, "linear2-m128-n128-k32-s3-cutlass", launchLinear2ResidualCutlass}, \
  {B, 384, "outproj-m128-n128-k32-s3-cutlass", launchOutProjectionResidualCutlass}
  KATAGO_CUTLASS_RESIDUAL_BATCH(4),
  KATAGO_CUTLASS_RESIDUAL_BATCH(5),
  KATAGO_CUTLASS_RESIDUAL_BATCH(6),
  KATAGO_CUTLASS_RESIDUAL_BATCH(7),
  KATAGO_CUTLASS_RESIDUAL_BATCH(8),
  KATAGO_CUTLASS_RESIDUAL_BATCH(9),
  KATAGO_CUTLASS_RESIDUAL_BATCH(10),
  KATAGO_CUTLASS_RESIDUAL_BATCH(11),
  KATAGO_CUTLASS_RESIDUAL_BATCH(12),
  KATAGO_CUTLASS_RESIDUAL_BATCH(13),
  KATAGO_CUTLASS_RESIDUAL_BATCH(14),
  KATAGO_CUTLASS_RESIDUAL_BATCH(15),
  KATAGO_CUTLASS_RESIDUAL_BATCH(16),
  KATAGO_CUTLASS_RESIDUAL_BATCH(17),
  KATAGO_CUTLASS_RESIDUAL_BATCH(18),
  KATAGO_CUTLASS_RESIDUAL_BATCH(19),
  KATAGO_CUTLASS_RESIDUAL_BATCH(20),
  KATAGO_CUTLASS_RESIDUAL_BATCH(21),
  KATAGO_CUTLASS_RESIDUAL_BATCH(22),
  KATAGO_CUTLASS_RESIDUAL_BATCH(23),
  KATAGO_CUTLASS_RESIDUAL_BATCH(24),
  KATAGO_CUTLASS_RESIDUAL_BATCH(25),
  KATAGO_CUTLASS_RESIDUAL_BATCH(26),
  KATAGO_CUTLASS_RESIDUAL_BATCH(27),
  KATAGO_CUTLASS_RESIDUAL_BATCH(28),
  KATAGO_CUTLASS_RESIDUAL_BATCH(29),
  KATAGO_CUTLASS_RESIDUAL_BATCH(30),
  KATAGO_CUTLASS_RESIDUAL_BATCH(31),
  KATAGO_CUTLASS_RESIDUAL_BATCH(32),
#undef KATAGO_CUTLASS_RESIDUAL_BATCH
};

const FusedFFNAotTactic searchFfnTactic = {
  sm120_search_ffn_batch(), sm120_search_ffn_id(), false,
  sm120_search_ffn_launch};
const WideQKVAotTactic searchQkvTactic = {
  sm120_search_qkv_batch(), sm120_search_qkv_id(),
  sm120_search_qkv_packed() != 0,
  sm120_search_qkv_launch};
const ResidualGemmAotTactic searchLinear2Tactic = {
  sm120_search_linear2_batch(), 1152,
  sm120_search_linear2_id(), sm120_search_linear2_launch};
const ResidualGemmAotTactic searchOutprojTactic = {
  sm120_search_outproj_batch(), 384,
  sm120_search_outproj_id(), sm120_search_outproj_launch};
const FA4AotTactic searchFA4Tactic = {
  sm120_search_fa4_batch(), sm120_search_fa4_id(), sm120_search_fa4_launch};

template<typename T>
const T* findExplicitFatTactic(
  const T* tactics, std::size_t count, int batchSize, const char* requestedId
) {
  if(requestedId == nullptr)
    return nullptr;
  for(std::size_t index = 0; index < count; index++) {
    const T& tactic = tactics[index];
    if(tactic.batchSize == batchSize && std::strcmp(tactic.id, requestedId) == 0)
      return &tactic;
  }
  return nullptr;
}

} // namespace

const FusedFFNAotTactic* findFusedFFNAotTactic(
  int batchSize, int numSms, int streams, const char* requestedId
) {
  (void)numSms;
  (void)streams;
  std::size_t fatCount = 0;
  const FusedFFNAotTactic* fatTactics = getSm120SearchFfnFatTactics(fatCount);
  const FusedFFNAotTactic* tactic = findExplicitFatTactic(
    fatTactics, fatCount, batchSize, requestedId);
  if(tactic != nullptr)
    return tactic;
  return requestedId != nullptr && searchFfnTactic.batchSize == batchSize &&
    std::strcmp(searchFfnTactic.id, requestedId) == 0 ? &searchFfnTactic : nullptr;
}

const WideQKVAotTactic* findWideQKVAotTactic(
  int batchSize, int numSms, int streams, const char* requestedId
) {
  (void)numSms;
  (void)streams;
  std::size_t fatCount = 0;
  const WideQKVAotTactic* fatTactics = getSm120SearchQkvFatTactics(fatCount);
  const WideQKVAotTactic* tactic = findExplicitFatTactic(
    fatTactics, fatCount, batchSize, requestedId);
  if(tactic != nullptr)
    return tactic;
  return requestedId != nullptr && searchQkvTactic.batchSize == batchSize &&
    std::strcmp(searchQkvTactic.id, requestedId) == 0 ? &searchQkvTactic : nullptr;
}

const WideQKVRopeAotTactic* findWideQKVRopeAotTactic(
  int batchSize, const char* requestedId
) {
  std::size_t fatCount = 0;
  const WideQKVRopeAotTactic* fatTactics =
    getSm120SearchQkvRopeFatTactics(fatCount);
  return findExplicitFatTactic(
    fatTactics,fatCount,batchSize,requestedId);
}

const ResidualGemmAotTactic* findResidualGemmAotTactic(
  int batchSize, int numSms, int streams, int inputChannels,
  const char* requestedId
) {
  (void)numSms;
  (void)streams;
  std::size_t fatCount = 0;
  const ResidualGemmAotTactic* fatTactics = inputChannels == 384 ?
    getSm120SearchOutprojFatTactics(fatCount) :
    getSm120SearchLinear2FatTactics(fatCount);
  const ResidualGemmAotTactic* fatTactic = findExplicitFatTactic(
    fatTactics, fatCount, batchSize, requestedId);
  if(fatTactic != nullptr && fatTactic->inputChannels == inputChannels)
    return fatTactic;
  if(inputChannels == searchLinear2Tactic.inputChannels &&
     searchLinear2Tactic.batchSize == batchSize && requestedId != nullptr &&
     std::strcmp(searchLinear2Tactic.id, requestedId) == 0)
    return &searchLinear2Tactic;
  if(inputChannels == searchOutprojTactic.inputChannels &&
     searchOutprojTactic.batchSize == batchSize && requestedId != nullptr &&
     std::strcmp(searchOutprojTactic.id, requestedId) == 0)
    return &searchOutprojTactic;
  for(const ResidualGemmAotTactic& tactic: residualTactics) {
    if(tactic.batchSize != batchSize || tactic.inputChannels != inputChannels)
      continue;
    if(requestedId != nullptr && std::strcmp(tactic.id, requestedId) == 0) {
      return &tactic;
    }
  }
  return nullptr;
}

const FA4AotTactic* findFA4AotTactic(
  int batchSize, const char* requestedId
) {
  std::size_t fatCount = 0;
  const FA4AotTactic* fatTactics = getSm120SearchFA4FatTactics(fatCount);
  const FA4AotTactic* tactic = findExplicitFatTactic(
    fatTactics, fatCount, batchSize, requestedId);
  if(tactic != nullptr)
    return tactic;

  // FA4's generated IDs include their compile-time batch (fa4-b19-...),
  // unlike the other fat families. A model warms every batch up to its
  // capacity, so selecting packed QKV requires the same FA4 shape at each
  // warmup/runtime batch. Canonicalize only the batch component here.
  if(requestedId != nullptr && std::strncmp(requestedId, "fa4-b", 5) == 0) {
    const char* suffix = requestedId + 5;
    while(*suffix >= '0' && *suffix <= '9')
      suffix++;
    if(*suffix == '-') {
      const std::string batchVariant =
        std::string("fa4-b") + std::to_string(batchSize) + suffix;
      tactic = findExplicitFatTactic(
        fatTactics, fatCount, batchSize, batchVariant.c_str());
      if(tactic != nullptr)
        return tactic;
    }
  }
  return requestedId != nullptr && searchFA4Tactic.batchSize == batchSize &&
    std::strcmp(searchFA4Tactic.id, requestedId) == 0 ? &searchFA4Tactic : nullptr;
}

} // namespace Sm120Backend
