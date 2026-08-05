#ifndef KATAGO_CUDA_BACKEND_SM89_KERNELS_H
#define KATAGO_CUDA_BACKEND_SM89_KERNELS_H

#include "../neuralnet/cudaincludes.h"

// SM89-specific helper kernels. These are intentionally separate from cudahelpers.cu:
// cudahelpers remains the official fallback and is not modified.

// Zero every [n, xy, c] NHWC position whose mask[n, xy] is zero. Used after a beta=1
// residual GEMM so masked padding positions stay exactly zero like the official path.
void sm89MaskZeroNHWC(half* buf, const half* mask, int batchSize, int xySize, int channels, cudaStream_t stream);

#endif
