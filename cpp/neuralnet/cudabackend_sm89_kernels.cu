#include "../neuralnet/cudabackend_sm89_kernels.h"

#include "../neuralnet/cudaerrorcheck.h"

__global__ void sm89MaskZeroNHWCHalfKernel(half* __restrict__ buf, const half* __restrict__ mask, int xySize, int channels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = xySize * channels;
  if(idx >= total)
    return;
  int c = idx % channels;
  int xy = (idx / channels) % xySize;
  int b = idx / total;
  if(mask[(size_t)b * xySize + xy] == __float2half(0.0f))
    buf[idx] = __float2half(0.0f);
}

void sm89MaskZeroNHWC(half* buf, const half* mask, int batchSize, int xySize, int channels, cudaStream_t stream) {
  int total = batchSize * xySize * channels;
  int block = 256;
  int grid = (total + block - 1) / block;
  sm89MaskZeroNHWCHalfKernel<<<grid, block, 0, stream>>>(buf, mask, xySize, channels);
  CUDA_ERR("sm89MaskZeroNHWC",cudaPeekAtLastError());
}
