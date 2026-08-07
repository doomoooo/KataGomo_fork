#ifndef KATAGO_CUDA_BACKEND_SM120_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM120_KERNELS_H

#include "../neuralnet/cudaincludes.h"

#include <cstddef>

namespace Sm120Backend {

typedef cudaError_t (*FusedFFNAotLaunchFn)(
  const half*, const half*, const half*, half*, cudaStream_t);
typedef cudaError_t (*WideQKVAotLaunchFn)(
  const half*, const half*, half*, cudaStream_t);
typedef cudaError_t (*ResidualGemmAotLaunchFn)(
  const half*, const half*, half*, cudaStream_t);
typedef cudaError_t (*FA4AotLaunchFn)(
  void*, void*, void*, void*, int, int, int, int, float, cudaStream_t);

struct FA4AotTactic {
  int batchSize;
  const char* id;
  FA4AotLaunchFn launch;
};

struct FusedFFNAotTactic {
  int batchSize;
  int requiredNumSms;
  int streams;
  const char* id;
  bool automaticWinner;
  FusedFFNAotLaunchFn launch;
};

struct WideQKVAotTactic {
  int batchSize;
  int requiredNumSms;
  int streams;
  const char* id;
  bool automaticWinner;
  bool packedOutput;
  WideQKVAotLaunchFn launch;
};

struct ResidualGemmAotTactic {
  int batchSize;
  int requiredNumSms;
  int streams;
  int inputChannels;
  const char* id;
  bool automaticWinner;
  ResidualGemmAotLaunchFn launch;
};

// These getters are empty in normal/single-slot builds. A fat-scan build
// replaces one family stub with a generated exact-(batch,tactic ID) table.
const FusedFFNAotTactic* getSm120SearchFfnFatTactics(std::size_t& count);
const WideQKVAotTactic* getSm120SearchQkvFatTactics(std::size_t& count);
const ResidualGemmAotTactic* getSm120SearchLinear2FatTactics(std::size_t& count);
const FA4AotTactic* getSm120SearchFA4FatTactics(std::size_t& count);

const FusedFFNAotTactic* findFusedFFNAotTactic(
  int batchSize, int numSms, int streams, const char* requestedId);
const WideQKVAotTactic* findWideQKVAotTactic(
  int batchSize, int numSms, int streams, const char* requestedId);
const ResidualGemmAotTactic* findResidualGemmAotTactic(
  int batchSize, int numSms, int streams, int inputChannels,
  const char* requestedId);
const FA4AotTactic* findFA4AotTactic(
  int batchSize, const char* requestedId);

void launchFusedFFNB13(
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  cudaStream_t stream
);

void launchFusedFFNB13CandidateAReuse(
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  cudaStream_t stream
);

cudaError_t launchFusedFFNB13S1(
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  cudaStream_t stream
);

cudaError_t launchWideQKVB13(
  const half* input,
  const half* weights,
  half* output,
  cudaStream_t stream
);

cudaError_t launchWideQKVB13S1(
  const half* input,
  const half* weights,
  half* output,
  cudaStream_t stream
);

cudaError_t launchLinear2ResidualB13(
  const half* input,
  const half* weights,
  half* residual,
  cudaStream_t stream
);

cudaError_t launchLinear2ResidualB13Balanced(
  const half* input,
  const half* weights,
  half* residual,
  cudaStream_t stream
);

cudaError_t launchOutProjectionResidualB13(
  const half* input,
  const half* weights,
  half* residual,
  cudaStream_t stream
);

void launchWideSwiGLU(
  const half* wideInput,
  half* output,
  int numTokens,
  int ffnChannels,
  cudaStream_t stream
);

void launchRMSNorm384(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
);

void launchRMSNorm384Vec8(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
);

void launchRMSNorm384TwoWarp(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
);

void launchFusedQKRoPE19(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
);

void launchBatchSharedFusedQKRoPE19(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
);

// Packed QKV rows are [Q384,K384,V384], so consecutive tokens are 1152
// half values apart. qBuf and kBuf point at offsets 0 and 384 respectively.
void launchBatchSharedPackedFusedQKRoPE19(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
);

void launchFusedQKRoPE19Half2(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
);

void launchSwiGLU1152Half8(
  const half* a,
  const half* b,
  half* output,
  int totalElements,
  cudaStream_t stream
);

void launchAffineSiluHalf2(
  const half* input,
  half* output,
  const half* scale,
  const half* bias,
  int totalRows,
  int channels,
  cudaStream_t stream
);

void launchFusedPolicyP1(
  const half* input,
  float* output,
  const float* globalBias,
  const float* scale,
  const float* bias,
  int batchSize,
  cudaStream_t stream
);

} // namespace Sm120Backend

#endif
