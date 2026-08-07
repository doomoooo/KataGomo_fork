#include "../neuralnet/cudabackend_sm89_kernels.h"

#include "../neuralnet/cudaerrorcheck.h"

__global__ void sm89MaskZeroNHWCHalfKernel(
  half* __restrict__ buf, const half* __restrict__ mask,
  int totalElements, int xySize, int channels
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= totalElements)
    return;
  int elementsPerBatch = xySize * channels;
  int xy = (idx / channels) % xySize;
  int b = idx / elementsPerBatch;
  if(mask[(size_t)b * xySize + xy] == __float2half(0.0f))
    buf[idx] = __float2half(0.0f);
}

void sm89MaskZeroNHWC(half* buf, const half* mask, int batchSize, int xySize, int channels, cudaStream_t stream) {
  int total = batchSize * xySize * channels;
  int block = 256;
  int grid = (total + block - 1) / block;
  sm89MaskZeroNHWCHalfKernel<<<grid, block, 0, stream>>>(
    buf, mask, total, xySize, channels
  );
  CUDA_ERR("sm89MaskZeroNHWC",cudaPeekAtLastError());
}

union __align__(16) Sm89Half8 {
  uint4 packed;
  half values[8];
};

__forceinline__ __device__ half sm89SiluHalf(half h) {
  float a = __half2float(h);
  return __float2half(a / (1.0f + expf(-a)));
}

__global__ void sm89ScaleBiasSiluNHWCHalfVec8Kernel(
  const half* __restrict__ in,
  half* __restrict__ out,
  const half* __restrict__ scale,
  const half* __restrict__ bias,
  int totalVecs
) {
  constexpr int cVecs = 768 / 8;
  int vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(vecIdx >= totalVecs)
    return;

  int cVec = vecIdx % cVecs;
  Sm89Half8 x;
  Sm89Half8 s;
  Sm89Half8 b;
  Sm89Half8 y;
  x.packed = reinterpret_cast<const uint4*>(in)[vecIdx];
  s.packed = reinterpret_cast<const uint4*>(scale)[cVec];
  b.packed = reinterpret_cast<const uint4*>(bias)[cVec];
#pragma unroll
  for(int i = 0; i < 8; i++) {
    half a = __hfma(x.values[i], s.values[i], b.values[i]);
    y.values[i] = sm89SiluHalf(a);
  }
  reinterpret_cast<uint4*>(out)[vecIdx] = y.packed;
}

bool sm89ScaleBiasSiluNHWCHalfVec8(
  const half* in, half* out, const half* scale, const half* bias,
  int nSize, int xySize, int cSize, cudaStream_t stream
) {
  if(nSize < 1 || xySize != 19 * 19 || cSize != 768)
    return false;
  constexpr int blockSize = 256;
  const int totalVecs = nSize * xySize * (768 / 8);
  const int gridSize = (totalVecs + blockSize - 1) / blockSize;
  sm89ScaleBiasSiluNHWCHalfVec8Kernel<<<gridSize,blockSize,0,stream>>>(
    in, out, scale, bias, totalVecs
  );
  CUDA_ERR("sm89ScaleBiasSiluNHWCHalfVec8",cudaPeekAtLastError());
  return true;
}

union __align__(8) Sm89Half4 {
  uint2 packed;
  half values[4];
};

__global__ void sm89ScaleBiasSiluNHWCHalfVec4C384Kernel(
  const half* __restrict__ in,
  half* __restrict__ out,
  const half* __restrict__ scale,
  const half* __restrict__ bias,
  int totalVecs
) {
  constexpr int cVecs = 384 / 4;
  int vecIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(vecIdx >= totalVecs)
    return;

  int cVec = vecIdx % cVecs;
  Sm89Half4 x;
  Sm89Half4 s;
  Sm89Half4 b;
  Sm89Half4 y;
  x.packed = reinterpret_cast<const uint2*>(in)[vecIdx];
  s.packed = reinterpret_cast<const uint2*>(scale)[cVec];
  b.packed = reinterpret_cast<const uint2*>(bias)[cVec];
#pragma unroll
  for(int i = 0; i < 4; i++) {
    half a = __hfma(x.values[i], s.values[i], b.values[i]);
    y.values[i] = sm89SiluHalf(a);
  }
  reinterpret_cast<uint2*>(out)[vecIdx] = y.packed;
}

bool sm89ScaleBiasSiluNHWCHalfVec4C384(
  const half* in, half* out, const half* scale, const half* bias,
  int nSize, int xySize, int cSize, cudaStream_t stream
) {
  if(nSize < 1 || xySize != 19 * 19 || cSize != 384)
    return false;
  constexpr int blockSize = 256;
  const int totalVecs = nSize * xySize * (384 / 4);
  const int gridSize = (totalVecs + blockSize - 1) / blockSize;
  sm89ScaleBiasSiluNHWCHalfVec4C384Kernel<<<gridSize,blockSize,0,stream>>>(
    in, out, scale, bias, totalVecs
  );
  CUDA_ERR("sm89ScaleBiasSiluNHWCHalfVec4C384",cudaPeekAtLastError());
  return true;
}

template<int xyPerBlock>
__global__ void sm89InitialGlobalMatMulAddKernel(
  const half* __restrict__ inputGlobal,
  const half* __restrict__ weights,
  half* __restrict__ spatial
) {
  constexpr int xySize = 19 * 19;
  constexpr int inChannels = 19;
  constexpr int outChannels = 768;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int xyBase = blockIdx.y * xyPerBlock;
  int n = blockIdx.z;

  float sum = 0.0f;
#pragma unroll
  for(int k = 0; k < inChannels; k++) {
    float w = __half2float(weights[k * outChannels + c]);
    float x = __half2float(inputGlobal[n * inChannels + k]);
    sum = __fmaf_rn(w, x, sum);
  }
  half globalBias = __float2half_rn(sum);
#pragma unroll
  for(int i = 0; i < xyPerBlock; i++) {
    int xy = xyBase + i;
    if(xy < xySize) {
      size_t idx = ((size_t)n * xySize + xy) * outChannels + c;
      spatial[idx] = __hadd(spatial[idx], globalBias);
    }
  }
}

bool sm89InitialGlobalMatMulAdd(
  const half* inputGlobal, const half* weights, half* spatial,
  int nSize, int xySize, int inChannels, int outChannels, cudaStream_t stream
) {
  if(nSize < 1 || xySize != 19 * 19 || inChannels != 19 || outChannels != 768)
    return false;
  constexpr int xyPerBlock = 8;
  constexpr int blockSize = 256;
  dim3 grid(outChannels / blockSize, (xySize + xyPerBlock - 1) / xyPerBlock, nSize);
  sm89InitialGlobalMatMulAddKernel<xyPerBlock><<<grid,blockSize,0,stream>>>(
    inputGlobal, weights, spatial
  );
  CUDA_ERR("sm89InitialGlobalMatMulAdd",cudaPeekAtLastError());
  return true;
}

template<int inputRowStride, int inputChannelOffset>
__global__ void sm89FusedPolicyP1Kernel(
  const half* __restrict__ in,
  float* __restrict__ out,
  const float* __restrict__ globalBias,
  const float* __restrict__ scale,
  const float* __restrict__ bias,
  int nSize
) {
  constexpr int xySize = 19 * 19;
  constexpr int channels = 96;
  int c = threadIdx.x;
  int xy = blockIdx.x * blockDim.y + threadIdx.y;
  int n = blockIdx.y;
  if(c >= channels || xy >= xySize || n >= nSize)
    return;

  size_t row = (size_t)n * xySize + xy;
  size_t inputIdx = row * inputRowStride + inputChannelOffset + c;
  size_t outputIdx = row * channels + c;
  float value = __half2float(in[inputIdx]);
  value = value + globalBias[n * channels + c];
  value = value * scale[c] + bias[c];
  out[outputIdx] = value / (1.0f + expf(-value));
}

bool sm89FusedPolicyP1(
  const half* in, float* out, const float* globalBias,
  const float* scale, const float* bias,
  int nSize, int xySize, int cSize,
  int inputRowStride, int inputChannelOffset, cudaStream_t stream
) {
  if(nSize < 1 || xySize != 19 * 19 || cSize != 96)
    return false;
  dim3 block(96, 5);
  dim3 grid((xySize + block.y - 1) / block.y, nSize);
  if(inputRowStride == 96 && inputChannelOffset == 0)
    sm89FusedPolicyP1Kernel<96,0><<<grid,block,0,stream>>>(in,out,globalBias,scale,bias,nSize);
  else if(inputRowStride == 384 && inputChannelOffset == 0)
    sm89FusedPolicyP1Kernel<384,0><<<grid,block,0,stream>>>(in,out,globalBias,scale,bias,nSize);
  else
    return false;
  CUDA_ERR("sm89FusedPolicyP1",cudaPeekAtLastError());
  return true;
}

template<int channels, int inputChannelOffset>
__global__ void sm89HeadBNSiluStridedKernel(
  const half* __restrict__ in,
  half* __restrict__ out,
  const half* __restrict__ scale,
  const half* __restrict__ bias,
  int nSize
) {
  constexpr int xySize = 19 * 19;
  constexpr int inputRowStride = 384;
  int c = threadIdx.x;
  int xy = blockIdx.x * blockDim.y + threadIdx.y;
  int n = blockIdx.y;
  if(c >= channels || xy >= xySize || n >= nSize)
    return;

  size_t row = (size_t)n * xySize + xy;
  half affine = __hfma(in[row * inputRowStride + inputChannelOffset + c], scale[c], bias[c]);
  out[row * channels + c] = sm89SiluHalf(affine);
}

bool sm89HeadBNSiluStrided(
  const half* in, half* out, const half* scale, const half* bias,
  int nSize, int xySize, int cSize,
  int inputRowStride, int inputChannelOffset, cudaStream_t stream
) {
  if(nSize < 1 || xySize != 19 * 19 || inputRowStride != 384)
    return false;
  if(cSize == 96 && inputChannelOffset == 96) {
    dim3 block(96, 5);
    dim3 grid((xySize + block.y - 1) / block.y, nSize);
    sm89HeadBNSiluStridedKernel<96,96><<<grid,block,0,stream>>>(in,out,scale,bias,nSize);
  }
  else if(cSize == 192 && inputChannelOffset == 192) {
    dim3 block(192, 2);
    dim3 grid((xySize + block.y - 1) / block.y, nSize);
    sm89HeadBNSiluStridedKernel<192,192><<<grid,block,0,stream>>>(in,out,scale,bias,nSize);
  }
  else
    return false;
  CUDA_ERR("sm89HeadBNSiluStrided",cudaPeekAtLastError());
  return true;
}

template<int channels, int rowsPerBlock, bool writeHalf, int inputRowStride, int inputChannelOffset>
__global__ void sm89HeadBNHalfToFloatKernel(
  const half* __restrict__ in,
  half* __restrict__ halfOut,
  float* __restrict__ floatOut,
  const half* __restrict__ scale,
  const half* __restrict__ bias,
  int nSize
) {
  constexpr int xySize = 19 * 19;
  int c = threadIdx.x;
  int xy = blockIdx.x * rowsPerBlock + threadIdx.y;
  int n = blockIdx.y;
  if(c >= channels || xy >= xySize || n >= nSize)
    return;

  size_t row = (size_t)n * xySize + xy;
  size_t outIdx = row * channels + c;
  size_t inIdx = row * inputRowStride + inputChannelOffset + c;
  half affine = __hfma(in[inIdx], scale[c], bias[c]);
  float value = __half2float(affine);
  half activated = __float2half(value / (1.0f + expf(-value)));
  if(writeHalf)
    halfOut[outIdx] = activated;
  floatOut[outIdx] = __half2float(activated);
}

bool sm89HeadBNHalfToFloat(
  const half* in, half* halfOut, float* floatOut,
  const half* scale, const half* bias,
  int nSize, int xySize, int cSize,
  int inputRowStride, int inputChannelOffset, cudaStream_t stream
) {
  if(nSize < 1 || xySize != 19 * 19)
    return false;
  if(cSize == 96 && halfOut == nullptr && inputRowStride == 96 && inputChannelOffset == 0) {
    dim3 block(96, 5);
    dim3 grid((xySize + block.y - 1) / block.y, nSize);
    sm89HeadBNHalfToFloatKernel<96,5,false,96,0><<<grid,block,0,stream>>>(
      in, nullptr, floatOut, scale, bias, nSize
    );
  }
  else if(cSize == 96 && halfOut == nullptr && inputRowStride == 384 && inputChannelOffset == 96) {
    dim3 block(96, 5);
    dim3 grid((xySize + block.y - 1) / block.y, nSize);
    sm89HeadBNHalfToFloatKernel<96,5,false,384,96><<<grid,block,0,stream>>>(
      in, nullptr, floatOut, scale, bias, nSize
    );
  }
  else if(cSize == 192 && halfOut != nullptr && inputRowStride == 192 && inputChannelOffset == 0) {
    dim3 block(192, 2);
    dim3 grid((xySize + block.y - 1) / block.y, nSize);
    sm89HeadBNHalfToFloatKernel<192,2,true,192,0><<<grid,block,0,stream>>>(
      in, halfOut, floatOut, scale, bias, nSize
    );
  }
  else if(cSize == 192 && halfOut != nullptr && inputRowStride == 384 && inputChannelOffset == 192) {
    dim3 block(192, 2);
    dim3 grid((xySize + block.y - 1) / block.y, nSize);
    sm89HeadBNHalfToFloatKernel<192,2,true,384,192><<<grid,block,0,stream>>>(
      in, halfOut, floatOut, scale, bias, nSize
    );
  }
  else
    return false;
  CUDA_ERR("sm89HeadBNHalfToFloat",cudaPeekAtLastError());
  return true;
}

__global__ void sm89SplitValueTerminalKernel(
  const float* __restrict__ combined,
  const float* __restrict__ bias,
  float* __restrict__ value,
  float* __restrict__ scoreValue,
  int total
) {
  constexpr int valueChannels = 3;
  constexpr int scoreValueChannels = 6;
  constexpr int combinedChannels = valueChannels + scoreValueChannels;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= total)
    return;

  int n = idx / combinedChannels;
  int c = idx % combinedChannels;
  float result = combined[idx] + bias[c];
  if(c < valueChannels)
    value[n * valueChannels + c] = result;
  else
    scoreValue[n * scoreValueChannels + c - valueChannels] = result;
}

bool sm89SplitValueTerminal(
  const float* combined, const float* bias,
  float* value, float* scoreValue,
  int batchSize, int valueChannels, int scoreValueChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || valueChannels != 3 || scoreValueChannels != 6)
    return false;
  constexpr int blockSize = 128;
  const int total = batchSize * (valueChannels + scoreValueChannels);
  const int gridSize = (total + blockSize - 1) / blockSize;
  sm89SplitValueTerminalKernel<<<gridSize,blockSize,0,stream>>>(
    combined, bias, value, scoreValue, total
  );
  CUDA_ERR("sm89SplitValueTerminal",cudaPeekAtLastError());
  return true;
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

__global__ void sm89PrecomputeRoPECosSinKernel(
  const float* __restrict__ freqs, float2* __restrict__ cosSinTable,
  int seqLen, int numHeads, int numKVHeads, int qHeadDim, int numPairs, int nnXLen
) {
  int xy = blockIdx.x;
  int hp = threadIdx.x;
  int totalHP = numHeads * numPairs;
  if(xy >= seqLen || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int kvh = h * numKVHeads / numHeads;
  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal, sinVal;
  __sincosf(angle, &sinVal, &cosVal);
  cosSinTable[(size_t)xy * totalHP + hp] = make_float2(cosVal, sinVal);
}

bool sm89PrecomputeRoPECosSin(
  const float* freqs, float2* cosSinTable,
  int seqLen, int numHeads, int numKVHeads, int qHeadDim, int nnXLen,
  cudaStream_t stream
) {
  if(numHeads != numKVHeads)
    return false;
  int numPairs = qHeadDim / 2;
  int totalHP = numHeads * numPairs;
  int threads = ((totalHP + 31) / 32) * 32;
  sm89PrecomputeRoPECosSinKernel<<<seqLen, threads, 0, stream>>>(
    freqs, cosSinTable, seqLen, numHeads, numKVHeads, qHeadDim, numPairs, nnXLen
  );
  CUDA_ERR("sm89PrecomputeRoPECosSin",cudaPeekAtLastError());
  return true;
}

__global__ void sm89ApplyRoPEQKHalfPrecomputedKernel(
  half* __restrict__ qBuf, half* __restrict__ kBuf,
  const float2* __restrict__ cosSinTable,
  int seqLen, int numHeads, int qHeadDim, int numPairs
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
  float2 cosSin = cosSinTable[(size_t)xy * totalHP + hp];
  float cosVal = cosSin.x;
  float sinVal = cosSin.y;

  float q0 = __half2float(qBuf[idx0]);
  float q1 = __half2float(qBuf[idx1]);
  qBuf[idx0] = __float2half(q0 * cosVal - q1 * sinVal);
  qBuf[idx1] = __float2half(q0 * sinVal + q1 * cosVal);

  float k0 = __half2float(kBuf[idx0]);
  float k1 = __half2float(kBuf[idx1]);
  kBuf[idx0] = __float2half(k0 * cosVal - k1 * sinVal);
  kBuf[idx1] = __float2half(k0 * sinVal + k1 * cosVal);
}

bool sm89ApplyRoPEQKHalfPrecomputed(
  half* qBuf, half* kBuf, const float2* cosSinTable,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim,
  cudaStream_t stream
) {
  if(numHeads != numKVHeads)
    return false;
  int numPairs = qHeadDim / 2;
  int totalHP = numHeads * numPairs;
  int threads = ((totalHP + 31) / 32) * 32;
  dim3 blocks(seqLen, batchSize);
  sm89ApplyRoPEQKHalfPrecomputedKernel<<<blocks, threads, 0, stream>>>(
    qBuf, kBuf, cosSinTable, seqLen, numHeads, qHeadDim, numPairs
  );
  CUDA_ERR("sm89ApplyRoPEQKHalfPrecomputed",cudaPeekAtLastError());
  return true;
}

template<int BatchGroup>
__global__ void sm89ApplyRoPEQKHalfBatchGroupedKernel(
  half* __restrict__ qBuf, half* __restrict__ kBuf, const float* __restrict__ freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim,
  int numPairs, int nnXLen
) {
  int xy = blockIdx.x;
  int hp = threadIdx.x;
  int totalHP = numHeads * numPairs;
  if(xy >= seqLen || hp >= totalHP)
    return;

  int h = hp / numPairs;
  int pairIdx = hp % numPairs;
  int c0 = h * qHeadDim + 2 * pairIdx;
  int c1 = c0 + 1;
  int kvh = h * numKVHeads / numHeads;
  int x = xy % nnXLen;
  int y = xy / nnXLen;
  float freqX = freqs[(kvh * numPairs + pairIdx) * 2 + 0];
  float freqY = freqs[(kvh * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal, sinVal;
  __sincosf(angle, &sinVal, &cosVal);
  size_t totalDim = (size_t)numHeads * qHeadDim;

#pragma unroll
  for(int i = 0; i < BatchGroup; i++) {
    int n = blockIdx.y * BatchGroup + i;
    if(n < batchSize) {
      size_t col = (size_t)n * seqLen + xy;
      size_t idx0 = c0 + col * totalDim;
      size_t idx1 = c1 + col * totalDim;
      float q0 = __half2float(qBuf[idx0]);
      float q1 = __half2float(qBuf[idx1]);
      qBuf[idx0] = __float2half(q0 * cosVal - q1 * sinVal);
      qBuf[idx1] = __float2half(q0 * sinVal + q1 * cosVal);
      float k0 = __half2float(kBuf[idx0]);
      float k1 = __half2float(kBuf[idx1]);
      kBuf[idx0] = __float2half(k0 * cosVal - k1 * sinVal);
      kBuf[idx1] = __float2half(k0 * sinVal + k1 * cosVal);
    }
  }
}

template<int BatchGroup>
static void launchSm89ApplyRoPEQKHalfBatchGrouped(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim,
  int numPairs, int nnXLen, int threads, cudaStream_t stream
) {
  dim3 blocks(seqLen, (batchSize + BatchGroup - 1) / BatchGroup);
  sm89ApplyRoPEQKHalfBatchGroupedKernel<BatchGroup><<<blocks, threads, 0, stream>>>(
    qBuf, kBuf, freqs, batchSize, seqLen, numHeads, numKVHeads, qHeadDim,
    numPairs, nnXLen
  );
}

bool sm89ApplyRoPEQKHalfBatchGrouped(
  half* qBuf, half* kBuf, const float* freqs,
  int batchSize, int seqLen, int numHeads, int numKVHeads, int qHeadDim, int nnXLen,
  int batchGroup, cudaStream_t stream
) {
  if(numHeads != numKVHeads)
    return false;
  int numPairs = qHeadDim / 2;
  int totalHP = numHeads * numPairs;
  int threads = ((totalHP + 31) / 32) * 32;
  switch(batchGroup) {
  case 2:
    launchSm89ApplyRoPEQKHalfBatchGrouped<2>(qBuf,kBuf,freqs,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,numPairs,nnXLen,threads,stream);
    break;
  case 3:
    launchSm89ApplyRoPEQKHalfBatchGrouped<3>(qBuf,kBuf,freqs,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,numPairs,nnXLen,threads,stream);
    break;
  case 4:
    launchSm89ApplyRoPEQKHalfBatchGrouped<4>(qBuf,kBuf,freqs,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,numPairs,nnXLen,threads,stream);
    break;
  case 7:
    launchSm89ApplyRoPEQKHalfBatchGrouped<7>(qBuf,kBuf,freqs,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,numPairs,nnXLen,threads,stream);
    break;
  case 13:
    launchSm89ApplyRoPEQKHalfBatchGrouped<13>(qBuf,kBuf,freqs,batchSize,seqLen,numHeads,numKVHeads,qHeadDim,numPairs,nnXLen,threads,stream);
    break;
  default:
    return false;
  }
  CUDA_ERR("sm89ApplyRoPEQKHalfBatchGrouped",cudaPeekAtLastError());
  return true;
}
