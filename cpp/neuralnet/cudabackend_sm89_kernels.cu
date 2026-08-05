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

__global__ void sm89RMSNormNHWCHalfKernel(
  const half* __restrict__ in, half* __restrict__ out,
  const half* __restrict__ gamma, const half* __restrict__ beta, const half* __restrict__ mask,
  int totalRows, int xySize, int cSize, float epsilon
) {
  int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if(row >= totalRows)
    return;
  int n = row / xySize;
  int xy = row % xySize;

  float maskVal = 1.0f;
  if(mask != NULL)
    maskVal = __half2float(mask[(size_t)n * xySize + xy]);
  if(maskVal == 0.0f) {
    half* outRow = out + (size_t)row * cSize;
    for(int c = lane; c < cSize; c += 32)
      outRow[c] = __float2half(0.0f);
    return;
  }

  const half* inRow = in + (size_t)row * cSize;
  float vals[12];
  float acc = 0.0f;
#pragma unroll
  for(int e = 0; e < 12; e++) {
    int c = lane + e * 32;
    float v = __half2float(inRow[c]) * maskVal;
    vals[e] = v;
    acc += v * v;
  }
  for(int off = 16; off > 0; off >>= 1)
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  float rms = rsqrtf(acc / (float)cSize + epsilon);

  half* outRow = out + (size_t)row * cSize;
#pragma unroll
  for(int e = 0; e < 12; e++) {
    int c = lane + e * 32;
    float o = vals[e] * rms * __half2float(gamma[c]) + __half2float(beta[c]);
    outRow[c] = __float2half(o * maskVal);
  }
}

bool sm89RMSNormNHWCHalf(
  const half* in, half* out, const half* gamma, const half* beta, const half* mask,
  int nSize, int xySize, int cSize, float epsilon, cudaStream_t stream
) {
  if(cSize != 384)
    return false;
  int totalRows = nSize * xySize;
  int blocks = (totalRows + 3) / 4;
  sm89RMSNormNHWCHalfKernel<<<blocks, 128, 0, stream>>>(
    in, out, gamma, beta, mask, totalRows, xySize, cSize, epsilon
  );
  CUDA_ERR("sm89RMSNormNHWCHalf",cudaPeekAtLastError());
  return true;
}
