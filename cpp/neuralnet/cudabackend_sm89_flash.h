#ifndef KATAGO_CUDA_BACKEND_SM89_FLASH_H
#define KATAGO_CUDA_BACKEND_SM89_FLASH_H

#include "../neuralnet/cudaincludes.h"

#include <cstddef>

namespace Sm89Backend {

#ifdef KATAGO_ENABLE_SM89_FLASH_ATTN
size_t sm89FlashAttentionLseBytesD32(int batchSize);

bool sm89FlashAttentionD32(
  const half* q,
  const half* k,
  const half* v,
  half* output,
  float* lse,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  int tactic,
  int numSms,
  cudaStream_t stream
);
#endif

} // namespace Sm89Backend

#endif
