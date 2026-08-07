#ifndef KATAGO_CUDA_BACKEND_SM89_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM89_KERNELS_H

#include "../neuralnet/cudaincludes.h"

// SM89-specific helper kernels. These are intentionally separate from cudahelpers.cu:
// cudahelpers remains the official fallback and is not modified.

// Zero every [n, xy, c] NHWC position whose mask[n, xy] is zero. Used after a beta=1
// residual GEMM so masked padding positions stay exactly zero like the official path.
void sm89MaskZeroNHWC(half* buf, const half* mask, int batchSize, int xySize, int channels, cudaStream_t stream);

// SM89-optimized transformer RMSNorm (NHWC, FP16). Currently specialized for cSize=384:
// one warp per row, 12 halfs per thread, warp-shuffle reduction. Returns true if it handled the
// launch; false means the caller should use the official helper.
bool sm89RMSNormNHWCHalf(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, cudaStream_t stream
);

// Exact-board C768 affine + SiLU path for 19x19. Each thread owns eight
// contiguous half elements while preserving the official per-element arithmetic.
bool sm89ScaleBiasSiluNHWCHalfVec8(
  const half* in, half* out, const half* scale, const half* bias,
  int nSize, int xySize, int cSize, cudaStream_t stream
);

// Exact-board C384 affine + SiLU path for 19x19. Each thread owns four
// contiguous half elements while preserving the official per-element arithmetic.
bool sm89ScaleBiasSiluNHWCHalfVec4C384(
  const half* in, half* out, const half* scale, const half* bias,
  int nSize, int xySize, int cSize, cudaStream_t stream
);

// Exact initial global-feature path for 19x19. Computes the K19->C768
// dot in FP32, rounds once to half, and applies the original half broadcast-add.
bool sm89InitialGlobalMatMulAdd(
  const half* inputGlobal, const half* weights, half* spatial,
  int nSize, int xySize, int inChannels, int outChannels, cudaStream_t stream
);

// Exact policy P1 path for 19x19/C96. Converts the convolution output to
// float and fuses the per-batch global bias with BN affine + SiLU.
bool sm89FusedPolicyP1(
  const half* in, float* out, const float* globalBias,
  const float* scale, const float* bias,
  int nSize, int xySize, int cSize,
  int inputRowStride, int inputChannelOffset, cudaStream_t stream
);

// Reads a C96 or C192 slice from the exact wide C384 head projection and
// writes the original contiguous half BN+SiLU result.
bool sm89HeadBNSiluStrided(
  const half* in, half* out, const half* scale, const half* bias,
  int nSize, int xySize, int cSize,
  int inputRowStride, int inputChannelOffset, cudaStream_t stream
);

// Exact head BN paths for 19x19. Reads either a contiguous head tensor or
// a slice of the wide C384 projection. The SiLU result is rounded to half before
// conversion to float, matching the original half-BN followed by copy kernel.
// C192 additionally writes the rounded half result for the ownership head.
bool sm89HeadBNHalfToFloat(
  const half* in, half* halfOut, float* floatOut,
  const half* scale, const half* bias,
  int nSize, int xySize, int cSize,
  int inputRowStride, int inputChannelOffset, cudaStream_t stream
);

// Splits a combined Bx9 value/score projection and applies the original
// independent FP32 biases while writing the established output layouts.
bool sm89SplitValueTerminal(
  const float* combined, const float* bias,
  float* value, float* scoreValue,
  int batchSize, int valueChannels, int scoreValueChannels,
  cudaStream_t stream
);

// Fused learnable RoPE for Q and K in one kernel (MHA only: numKVHeads == numHeads and same
// totalDim). Returns true if handled, false otherwise.
bool sm89ApplyRoPEQKHalf(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int nnXLen,
  cudaStream_t stream
);

// Builds and consumes a model-lifetime float2 (cos,sin) table for learnable RoPE. The table
// contains seqLen * numHeads * (qHeadDim / 2) entries and is shared across the batch dimension.
bool sm89PrecomputeRoPECosSin(
  const float* freqs, float2* cosSinTable,
  int seqLen, int numHeads, int numKVHeads, int qHeadDim, int nnXLen,
  cudaStream_t stream
);
bool sm89ApplyRoPEQKHalfPrecomputed(
  half* qBuf, half* kBuf, const float2* cosSinTable,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim,
  cudaStream_t stream
);

// Shares each frequency load and sincos evaluation across a fixed group of batch rows.
// Supported AOT group sizes are 2, 3, 4, 7, and 13; false requests caller fallback.
bool sm89ApplyRoPEQKHalfBatchGrouped(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int nnXLen,
  int batchGroup, cudaStream_t stream
);

#endif
