#ifndef KATAGO_CUDA_BACKEND_SM89_TACTIC_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM89_TACTIC_KERNELS_H

#include "../neuralnet/cudaincludes.h"

#include <cstddef>

namespace Sm89Backend {

// Stable generated-bundle ABI. Runtime policy remains in the SM89 backend;
// generated translation units only provide exact-(batch,tactic ID) launchers.
// A launcher is architecture-specific, not product-name-specific. The plan
// records the numeric CUDA device capabilities on which it was tuned.

using FusedFFNAotLaunchFn = cudaError_t (*)(
  const half*, const half*, const half*, half*, cudaStream_t);
using ResidualGemmAotLaunchFn = cudaError_t (*)(
  const half*, const half*, half*, cudaStream_t);

struct FusedFFNAotTactic {
  int batchSize;
  int streams;
  const char* id;
  FusedFFNAotLaunchFn launch;
};

struct ResidualGemmAotTactic {
  int batchSize;
  int streams;
  int inputChannels;
  const char* id;
  ResidualGemmAotLaunchFn launch;
};

const FusedFFNAotTactic* getSm89SearchDualFfnFatTactics(std::size_t& count);
const ResidualGemmAotTactic* getSm89SearchLinear2FatTactics(std::size_t& count);

const FusedFFNAotTactic* findSm89DualFfnTactic(
  int batchSize, int streams, const char* requestedId,
  bool& requestedIdKnown);
const ResidualGemmAotTactic* findSm89Linear2Tactic(
  int batchSize, int streams, int inputChannels,
  const char* requestedId, bool& requestedIdKnown);

} // namespace Sm89Backend

#endif
