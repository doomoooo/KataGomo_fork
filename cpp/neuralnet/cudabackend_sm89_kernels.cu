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

__global__ void sm89ApplyRoPEQKHalfKernel(
  half* __restrict__ qBuf, half* __restrict__ kBuf, const float* __restrict__ freqs,
  int seqLen, int numHeads, int numKVHeads, int qHeadDim, int numPairs, int nnXLen
) {
  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int totalHP = numHeads * numPairs;
  if(xy >= seqLen || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int c0 = h * qHeadDim + 2 * pairIdx;
  int c1 = c0 + 1;
  size_t col = (size_t)n * seqLen + xy;
  size_t totalDim = (size_t)numHeads * qHeadDim;
  size_t idx0 = c0 + col * totalDim;
  size_t idx1 = c1 + col * totalDim;

  int kvh = h * numKVHeads / numHeads;
  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal, sinVal;
  __sincosf(angle, &sinVal, &cosVal);

  float q0 = __half2float(qBuf[idx0]);
  float q1 = __half2float(qBuf[idx1]);
  qBuf[idx0] = __float2half(q0 * cosVal - q1 * sinVal);
  qBuf[idx1] = __float2half(q0 * sinVal + q1 * cosVal);

  float k0 = __half2float(kBuf[idx0]);
  float k1 = __half2float(kBuf[idx1]);
  kBuf[idx0] = __float2half(k0 * cosVal - k1 * sinVal);
  kBuf[idx1] = __float2half(k0 * sinVal + k1 * cosVal);
}

bool sm89ApplyRoPEQKHalf(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int nnXLen,
  cudaStream_t stream
) {
  if(numHeads != numKVHeads)
    return false;
  int numPairs = qHeadDim / 2;
  int totalHP = numHeads * numPairs;
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize);
  sm89ApplyRoPEQKHalfKernel<<<blocks, threads, 0, stream>>>(
    qBuf, kBuf, freqs, seqLen, numHeads, numKVHeads, qHeadDim, numPairs, nnXLen
  );
  CUDA_ERR("sm89ApplyRoPEQKHalf",cudaPeekAtLastError());
  return true;
}
