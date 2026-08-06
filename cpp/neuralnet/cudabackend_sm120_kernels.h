#ifndef KATAGO_CUDA_BACKEND_SM120_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM120_KERNELS_H

#include "../neuralnet/cudaincludes.h"

namespace Sm120Backend {

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

void launchBatchSharedFusedQKRoPE19B13(
  half* qBuf,
  half* kBuf,
  const float* freqs,
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

void launchFusedPolicyP1B13(
  const half* input,
  float* output,
  const float* globalBias,
  const float* scale,
  const float* bias,
  cudaStream_t stream
);

} // namespace Sm120Backend

#endif
