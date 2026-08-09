#include "../neuralnet/cudabackend_sm120_kernels.h"

namespace Sm120Backend {

__global__ void wideSwiGLUHalf2Kernel(
  const half* wideInput,
  half* output,
  int pairsPerToken,
  int ffnChannels,
  int pairCount
) {
  int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(pairIdx >= pairCount)
    return;

  int token = pairIdx / pairsPerToken;
  int channelPair = pairIdx - token * pairsPerToken;
  const half2* tokenInput = reinterpret_cast<const half2*>(
    wideInput + (size_t)token * 2 * ffnChannels);
  half2 a = tokenInput[channelPair];
  half2 b = tokenInput[pairsPerToken + channelPair];

  float a0 = __half2float(__low2half(a));
  float a1 = __half2float(__high2half(a));
  float b0 = __half2float(__low2half(b));
  float b1 = __half2float(__high2half(b));
  float s0 = a0 / (1.0f + expf(-a0));
  float s1 = a1 / (1.0f + expf(-a1));
  reinterpret_cast<half2*>(output)[pairIdx] = __halves2half2(
    __float2half(s0 * b0), __float2half(s1 * b1));
}

void launchWideSwiGLU(
  const half* wideInput,
  half* output,
  int numTokens,
  int ffnChannels,
  cudaStream_t stream
) {
  int pairsPerToken = ffnChannels / 2;
  int pairCount = numTokens * pairsPerToken;
  constexpr int threads = 256;
  int blocks = (pairCount + threads - 1) / threads;
  wideSwiGLUHalf2Kernel<<<blocks, threads, 0, stream>>>(
    wideInput, output, pairsPerToken, ffnChannels, pairCount);
}

__global__ void rmsNorm384Half2Kernel(
  const half* __restrict__ input,
  half* __restrict__ output,
  const half* __restrict__ gamma,
  const half* __restrict__ beta,
  int totalRows,
  float epsilon
) {
  int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if(row >= totalRows)
    return;

  constexpr int channels = 384;
  const half2* input2 = reinterpret_cast<const half2*>(input + (size_t)row * channels);
  const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
  const half2* beta2 = reinterpret_cast<const half2*>(beta);
  half2* output2 = reinterpret_cast<half2*>(output + (size_t)row * channels);

  float vals[12];
  float groupSums[6];
#pragma unroll
  for(int e = 0; e < 6; e++) {
    int pair = lane + e * 32;
    half2 v = input2[pair];
    float v0 = __half2float(__low2half(v));
    float v1 = __half2float(__high2half(v));
    vals[2 * e] = v0;
    vals[2 * e + 1] = v1;
    groupSums[e] = v0 * v0 + v1 * v1;
  }
  // Match the official 192-thread tree: six independent 32-thread group
  // reductions, followed by one reduction of the six group sums.
#pragma unroll
  for(int e = 0; e < 6; e++) {
    for(int offset = 16; offset > 0; offset >>= 1)
      groupSums[e] += __shfl_xor_sync(0xffffffff, groupSums[e], offset);
  }
  float sumSquares = lane < 6 ? groupSums[lane] : 0.0f;
  for(int offset = 16; offset > 0; offset >>= 1)
    sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, offset);
  float scale = rsqrtf(sumSquares / (float)channels + epsilon);

#pragma unroll
  for(int e = 0; e < 6; e++) {
    int pair = lane + e * 32;
    half2 g = gamma2[pair];
    half2 b = beta2[pair];
    float o0 = vals[2 * e] * scale * __half2float(__low2half(g)) + __half2float(__low2half(b));
    float o1 = vals[2 * e + 1] * scale * __half2float(__high2half(g)) + __half2float(__high2half(b));
    output2[pair] = __halves2half2(__float2half(o0), __float2half(o1));
  }
}

void launchRMSNorm384(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
) {
  int blocks = (totalRows + 3) / 4;
  rmsNorm384Half2Kernel<<<blocks, 128, 0, stream>>>(
    input, output, gamma, beta, totalRows, epsilon);
}

__global__ void rmsNorm384OrderedEpt3Kernel(
  const half* __restrict__ input,
  half* __restrict__ output,
  const half* __restrict__ gamma,
  const half* __restrict__ beta,
  int totalRows,
  float epsilon
) {
  int row = blockIdx.x;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int channel = threadIdx.x * 3;
  if(row >= totalRows)
    return;

  const half* rowInput = input + (size_t)row * 384;
  half* rowOutput = output + (size_t)row * 384;
  float values[3];
  float sumSquares = 0.0f;
#pragma unroll
  for(int e = 0; e < 3; e++) {
    values[e] = __half2float(rowInput[channel + e]);
    sumSquares += values[e] * values[e];
  }
  for(int offset = 16; offset > 0; offset >>= 1)
    sumSquares += __shfl_down_sync(0xffffffff, sumSquares, offset);
  __shared__ float warpSums[4];
  if(lane == 0)
    warpSums[warp] = sumSquares;
  __syncthreads();
  if(warp == 0) {
    sumSquares = lane < 4 ? warpSums[lane] : 0.0f;
    for(int offset = 16; offset > 0; offset >>= 1)
      sumSquares += __shfl_down_sync(0xffffffff, sumSquares, offset);
    if(lane == 0)
      warpSums[0] = sumSquares;
  }
  __syncthreads();
  float rms = rsqrtf(warpSums[0] / 384.0f + epsilon);
#pragma unroll
  for(int e = 0; e < 3; e++) {
    int c = channel + e;
    float result = values[e] * rms * __half2float(gamma[c]) +
      __half2float(beta[c]);
    rowOutput[c] = __float2half(result);
  }
}

void launchRMSNorm384OrderedEpt3(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
) {
  rmsNorm384OrderedEpt3Kernel<<<totalRows, 128, 0, stream>>>(
    input, output, gamma, beta, totalRows, epsilon);
}

__global__ void rmsNorm384Vec8Kernel(
  const half* __restrict__ input,
  half* __restrict__ output,
  const half* __restrict__ gamma,
  const half* __restrict__ beta,
  int totalRows,
  float epsilon
) {
  int row = blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if(row >= totalRows)
    return;

  constexpr int channels = 384;
  constexpr int halfsPerUint4 = sizeof(uint4) / sizeof(half);
  constexpr int mainHalfs = 32 * halfsPerUint4;

  const half* rowInput = input + (size_t)row * channels;
  half* rowOutput = output + (size_t)row * channels;
  uint4 inputMain = reinterpret_cast<const uint4*>(rowInput)[lane];
  uint2 inputTail = reinterpret_cast<const uint2*>(rowInput + mainHalfs)[lane];
  uint4 gammaMain = reinterpret_cast<const uint4*>(gamma)[lane];
  uint2 gammaTail = reinterpret_cast<const uint2*>(gamma + mainHalfs)[lane];
  uint4 betaMain = reinterpret_cast<const uint4*>(beta)[lane];
  uint2 betaTail = reinterpret_cast<const uint2*>(beta + mainHalfs)[lane];

  const half2* inputMain2 = reinterpret_cast<const half2*>(&inputMain);
  const half2* inputTail2 = reinterpret_cast<const half2*>(&inputTail);
  float vals[12];
  float sumSquares = 0.0f;
#pragma unroll
  for(int e = 0; e < 4; e++) {
    half2 v = inputMain2[e];
    float v0 = __half2float(__low2half(v));
    float v1 = __half2float(__high2half(v));
    vals[2 * e] = v0;
    vals[2 * e + 1] = v1;
    sumSquares += v0 * v0 + v1 * v1;
  }
#pragma unroll
  for(int e = 0; e < 2; e++) {
    half2 v = inputTail2[e];
    float v0 = __half2float(__low2half(v));
    float v1 = __half2float(__high2half(v));
    vals[8 + 2 * e] = v0;
    vals[9 + 2 * e] = v1;
    sumSquares += v0 * v0 + v1 * v1;
  }
  for(int offset = 16; offset > 0; offset >>= 1)
    sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, offset);
  float scale = rsqrtf(sumSquares / (float)channels + epsilon);

  const half2* gammaMain2 = reinterpret_cast<const half2*>(&gammaMain);
  const half2* gammaTail2 = reinterpret_cast<const half2*>(&gammaTail);
  const half2* betaMain2 = reinterpret_cast<const half2*>(&betaMain);
  const half2* betaTail2 = reinterpret_cast<const half2*>(&betaTail);
  uint4 outputMain;
  uint2 outputTail;
  half2* outputMain2 = reinterpret_cast<half2*>(&outputMain);
  half2* outputTail2 = reinterpret_cast<half2*>(&outputTail);
#pragma unroll
  for(int e = 0; e < 4; e++) {
    half2 g = gammaMain2[e];
    half2 b = betaMain2[e];
    float o0 = vals[2 * e] * scale * __half2float(__low2half(g)) + __half2float(__low2half(b));
    float o1 = vals[2 * e + 1] * scale * __half2float(__high2half(g)) + __half2float(__high2half(b));
    outputMain2[e] = __halves2half2(__float2half(o0), __float2half(o1));
  }
#pragma unroll
  for(int e = 0; e < 2; e++) {
    half2 g = gammaTail2[e];
    half2 b = betaTail2[e];
    float o0 = vals[8 + 2 * e] * scale * __half2float(__low2half(g)) + __half2float(__low2half(b));
    float o1 = vals[9 + 2 * e] * scale * __half2float(__high2half(g)) + __half2float(__high2half(b));
    outputTail2[e] = __halves2half2(__float2half(o0), __float2half(o1));
  }
  reinterpret_cast<uint4*>(rowOutput)[lane] = outputMain;
  reinterpret_cast<uint2*>(rowOutput + mainHalfs)[lane] = outputTail;
}

void launchRMSNorm384Vec8(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
) {
  int blocks = (totalRows + 3) / 4;
  rmsNorm384Vec8Kernel<<<blocks, 128, 0, stream>>>(
    input, output, gamma, beta, totalRows, epsilon);
}

__global__ void rmsNorm384TwoWarpHalf2Kernel(
  const half* __restrict__ input,
  half* __restrict__ output,
  const half* __restrict__ gamma,
  const half* __restrict__ beta,
  int totalRows,
  float epsilon
) {
  int row = blockIdx.x;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  if(row >= totalRows)
    return;

  constexpr int channels = 384;
  const half2* input2 = reinterpret_cast<const half2*>(input + (size_t)row * channels);
  const half2* gamma2 = reinterpret_cast<const half2*>(gamma);
  const half2* beta2 = reinterpret_cast<const half2*>(beta);
  half2* output2 = reinterpret_cast<half2*>(output + (size_t)row * channels);

  float vals[6];
  float localSums[3];
#pragma unroll
  for(int e = 0; e < 3; e++) {
    int group = warp * 3 + e;
    int pair = lane + group * 32;
    half2 v = input2[pair];
    float v0 = __half2float(__low2half(v));
    float v1 = __half2float(__high2half(v));
    vals[2 * e] = v0;
    vals[2 * e + 1] = v1;
    localSums[e] = v0 * v0 + v1 * v1;
  }
#pragma unroll
  for(int e = 0; e < 3; e++) {
    for(int offset = 16; offset > 0; offset >>= 1)
      localSums[e] += __shfl_xor_sync(0xffffffff, localSums[e], offset);
  }

  __shared__ float groupSums[6];
  __shared__ float scales[32];
#pragma unroll
  for(int e = 0; e < 3; e++) {
    int group = warp * 3 + e;
    if(lane == group)
      groupSums[group] = localSums[e];
  }
  __syncthreads();

  if(warp == 0) {
    float sumSquares = lane < 6 ? groupSums[lane] : 0.0f;
    for(int offset = 16; offset > 0; offset >>= 1)
      sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, offset);
    scales[lane] = rsqrtf(sumSquares / (float)channels + epsilon);
  }
  __syncthreads();
  float scale = scales[lane];

#pragma unroll
  for(int e = 0; e < 3; e++) {
    int group = warp * 3 + e;
    int pair = lane + group * 32;
    half2 g = gamma2[pair];
    half2 b = beta2[pair];
    float o0 = vals[2 * e] * scale * __half2float(__low2half(g)) + __half2float(__low2half(b));
    float o1 = vals[2 * e + 1] * scale * __half2float(__high2half(g)) + __half2float(__high2half(b));
    output2[pair] = __halves2half2(__float2half(o0), __float2half(o1));
  }
}

void launchRMSNorm384TwoWarp(
  const half* input,
  half* output,
  const half* gamma,
  const half* beta,
  int totalRows,
  float epsilon,
  cudaStream_t stream
) {
  rmsNorm384TwoWarpHalf2Kernel<<<totalRows, 64, 0, stream>>>(
    input, output, gamma, beta, totalRows, epsilon);
}

__global__ void fusedQKRoPE19HalfKernel(
  half* __restrict__ qBuf,
  half* __restrict__ kBuf,
  const float* __restrict__ freqs
) {
  constexpr int seqLen = 361;
  constexpr int totalDim = 384;
  constexpr int numPairs = 16;

  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int h = hp / numPairs;
  int pairIdx = hp - h * numPairs;
  size_t idx0 = (size_t)(n * seqLen + xy) * totalDim + 2 * hp;
  size_t idx1 = idx0 + 1;

  int x = xy % 19;
  int y = xy / 19;
  float freqX = freqs[(h * numPairs + pairIdx) * 2];
  float freqY = freqs[(h * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal;
  float sinVal;
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

__global__ void precomputeQKVRopeTable19HalfKernel(
  const float* __restrict__ freqs,
  half2* __restrict__ table
) {
  constexpr int numPairs = 16;
  int xy = blockIdx.x;
  int hp = threadIdx.x;
  int head = hp / numPairs;
  int pair = hp - head * numPairs;
  int x = xy % 19;
  int y = xy / 19;
  float angle = (float)x * freqs[(head * numPairs + pair) * 2] +
    (float)y * freqs[(head * numPairs + pair) * 2 + 1];
  float cosValue;
  float sinValue;
  __sincosf(angle,&sinValue,&cosValue);
  table[xy * 192 + hp] = __floats2half2_rn(cosValue,sinValue);
}

void launchPrecomputeQKVRopeTable19Half(
  const float* freqs,
  half* table,
  cudaStream_t stream
) {
  precomputeQKVRopeTable19HalfKernel<<<361,192,0,stream>>>(
    freqs,reinterpret_cast<half2*>(table));
}

void launchFusedQKRoPE19(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
) {
  dim3 blocks(361, batchSize);
  fusedQKRoPE19HalfKernel<<<blocks, 192, 0, stream>>>(qBuf, kBuf, freqs);
}

__global__ void batchSharedFusedQKRoPE19HalfKernel(
  half* __restrict__ qBuf,
  half* __restrict__ kBuf,
  const float* __restrict__ freqs,
  int batchSize
) {
  constexpr int seqLen = 361;
  constexpr int totalDim = 384;
  constexpr int numPairs = 16;

  int xy = blockIdx.x;
  int hp = threadIdx.x;
  int h = hp / numPairs;
  int pairIdx = hp - h * numPairs;
  int x = xy % 19;
  int y = xy / 19;
  float freqX = freqs[(h * numPairs + pairIdx) * 2];
  float freqY = freqs[(h * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal;
  float sinVal;
  __sincosf(angle, &sinVal, &cosVal);

  for(int n = 0; n < batchSize; n++) {
    size_t idx0 = (size_t)(n * seqLen + xy) * totalDim + 2 * hp;
    size_t idx1 = idx0 + 1;
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

void launchBatchSharedFusedQKRoPE19(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
) {
  batchSharedFusedQKRoPE19HalfKernel<<<361, 192, 0, stream>>>(
    qBuf, kBuf, freqs, batchSize);
}

__global__ void batchSharedPackedFusedQKRoPE19Half2Kernel(
  half2* __restrict__ qBuf,
  half2* __restrict__ kBuf,
  const float* __restrict__ freqs,
  int batchSize
) {
  constexpr int seqLen = 361;
  constexpr int packedPairsPerRow = 3 * 384 / 2;
  constexpr int numPairs = 16;

  int xy = blockIdx.x;
  int hp = threadIdx.x;
  int h = hp / numPairs;
  int pairIdx = hp - h * numPairs;
  int x = xy % 19;
  int y = xy / 19;
  float freqX = freqs[(h * numPairs + pairIdx) * 2];
  float freqY = freqs[(h * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal;
  float sinVal;
  __sincosf(angle, &sinVal, &cosVal);

  for(int n = 0; n < batchSize; n++) {
    size_t idx = (size_t)(n * seqLen + xy) * packedPairsPerRow + hp;
    half2 q = qBuf[idx];
    float q0 = __half2float(__low2half(q));
    float q1 = __half2float(__high2half(q));
    qBuf[idx] = __halves2half2(
      __float2half(q0 * cosVal - q1 * sinVal),
      __float2half(q0 * sinVal + q1 * cosVal));

    half2 k = kBuf[idx];
    float k0 = __half2float(__low2half(k));
    float k1 = __half2float(__high2half(k));
    kBuf[idx] = __halves2half2(
      __float2half(k0 * cosVal - k1 * sinVal),
      __float2half(k0 * sinVal + k1 * cosVal));
  }
}

void launchBatchSharedPackedFusedQKRoPE19(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
) {
  batchSharedPackedFusedQKRoPE19Half2Kernel<<<361, 192, 0, stream>>>(
    reinterpret_cast<half2*>(qBuf), reinterpret_cast<half2*>(kBuf),
    freqs, batchSize);
}

// Compile one exact-batch form for every supported inference batch. Exposing
// all independent Q/K transactions lets ptxas software-pipeline their global
// loads without a runtime loop while keeping the tactic searchable per batch.
template<int Batch>
__global__ void batchSharedPackedFusedQKRoPEUnrolledHalf2Kernel(
  half2* __restrict__ qBuf,
  half2* __restrict__ kBuf,
  const float* __restrict__ freqs
) {
  constexpr int seqLen = 361;
  constexpr int packedPairsPerRow = 3 * 384 / 2;
  constexpr int numPairs = 16;

  int xy = blockIdx.x;
  int hp = threadIdx.x;
  int h = hp / numPairs;
  int pairIdx = hp - h * numPairs;
  int x = xy % 19;
  int y = xy / 19;
  float freqX = freqs[(h * numPairs + pairIdx) * 2];
  float freqY = freqs[(h * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal;
  float sinVal;
  __sincosf(angle, &sinVal, &cosVal);

#pragma unroll
  for(int n = 0; n < Batch; n++) {
    size_t idx = (size_t)(n * seqLen + xy) * packedPairsPerRow + hp;
    float2 q = __half22float2(qBuf[idx]);
    float2 k = __half22float2(kBuf[idx]);
    qBuf[idx] = __floats2half2_rn(
      q.x * cosVal - q.y * sinVal, q.x * sinVal + q.y * cosVal);
    kBuf[idx] = __floats2half2_rn(
      k.x * cosVal - k.y * sinVal, k.x * sinVal + k.y * cosVal);
  }
}

template<int Batch>
void launchBatchSharedPackedFusedQKRoPEUnrolledExact(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  cudaStream_t stream
) {
  batchSharedPackedFusedQKRoPEUnrolledHalf2Kernel<Batch><<<361, 192, 0, stream>>>(
    reinterpret_cast<half2*>(qBuf), reinterpret_cast<half2*>(kBuf), freqs);
}

void launchBatchSharedPackedFusedQKRoPEUnrolled(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
) {
#define KATAGO_LAUNCH_UNROLLED_ROPE(B) \
  case B: launchBatchSharedPackedFusedQKRoPEUnrolledExact<B>( \
    qBuf,kBuf,freqs,stream); return
  switch(batchSize) {
    KATAGO_LAUNCH_UNROLLED_ROPE(1);
    KATAGO_LAUNCH_UNROLLED_ROPE(2);
    KATAGO_LAUNCH_UNROLLED_ROPE(3);
    KATAGO_LAUNCH_UNROLLED_ROPE(4);
    KATAGO_LAUNCH_UNROLLED_ROPE(5);
    KATAGO_LAUNCH_UNROLLED_ROPE(6);
    KATAGO_LAUNCH_UNROLLED_ROPE(7);
    KATAGO_LAUNCH_UNROLLED_ROPE(8);
    KATAGO_LAUNCH_UNROLLED_ROPE(9);
    KATAGO_LAUNCH_UNROLLED_ROPE(10);
    KATAGO_LAUNCH_UNROLLED_ROPE(11);
    KATAGO_LAUNCH_UNROLLED_ROPE(12);
    KATAGO_LAUNCH_UNROLLED_ROPE(13);
    KATAGO_LAUNCH_UNROLLED_ROPE(14);
    KATAGO_LAUNCH_UNROLLED_ROPE(15);
    KATAGO_LAUNCH_UNROLLED_ROPE(16);
    KATAGO_LAUNCH_UNROLLED_ROPE(17);
    KATAGO_LAUNCH_UNROLLED_ROPE(18);
    KATAGO_LAUNCH_UNROLLED_ROPE(19);
    KATAGO_LAUNCH_UNROLLED_ROPE(20);
    KATAGO_LAUNCH_UNROLLED_ROPE(21);
    KATAGO_LAUNCH_UNROLLED_ROPE(22);
    KATAGO_LAUNCH_UNROLLED_ROPE(23);
    KATAGO_LAUNCH_UNROLLED_ROPE(24);
    KATAGO_LAUNCH_UNROLLED_ROPE(25);
    KATAGO_LAUNCH_UNROLLED_ROPE(26);
    KATAGO_LAUNCH_UNROLLED_ROPE(27);
    KATAGO_LAUNCH_UNROLLED_ROPE(28);
    KATAGO_LAUNCH_UNROLLED_ROPE(29);
    KATAGO_LAUNCH_UNROLLED_ROPE(30);
    KATAGO_LAUNCH_UNROLLED_ROPE(31);
    KATAGO_LAUNCH_UNROLLED_ROPE(32);
    default: return;
  }
#undef KATAGO_LAUNCH_UNROLLED_ROPE
}

__global__ void initialGlobalMatMulAddKernel(
  half* __restrict__ spatialBuf,
  const half* __restrict__ globalInput,
  const half* __restrict__ weights
) {
  constexpr int spatial = 361;
  constexpr int inputChannels = 19;
  constexpr int outputChannels = 768;
  constexpr int channelPairsPerBlock = 32;
  constexpr int warpsPerBlock = 8;

  __shared__ half2 projected[channelPairsPerBlock];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int channelPair = blockIdx.x * channelPairsPerBlock + lane;
  int channel = channelPair * 2;
  int n = blockIdx.y;

  if(warp == 0) {
    float2 value = make_float2(0.0f,0.0f);
#pragma unroll
    for(int k = 0; k < inputChannels; k++) {
      float input = __half2float(globalInput[n * inputChannels + k]);
      half2 weight = *reinterpret_cast<const half2*>(
        weights + k * outputChannels + channel);
      float2 weightFloat = __half22float2(weight);
      value.x = fmaf(weightFloat.x,input,value.x);
      value.y = fmaf(weightFloat.y,input,value.y);
    }
    projected[lane] = __floats2half2_rn(value.x,value.y);
  }
  __syncthreads();

  half2 value = projected[lane];
  half2* spatialPairs = reinterpret_cast<half2*>(spatialBuf);
  constexpr int outputChannelPairs = outputChannels / 2;
  for(int xy = warp; xy < spatial; xy += warpsPerBlock) {
    size_t index = ((size_t)n * spatial + xy) * outputChannelPairs + channelPair;
    spatialPairs[index] = __hadd2(spatialPairs[index],value);
  }
}

void launchInitialGlobalMatMulAdd(
  half* spatialBuf,
  const half* globalInput,
  const half* weights,
  int batchSize,
  cudaStream_t stream
) {
  constexpr int outputChannelPairs = 768 / 2;
  constexpr int channelPairsPerBlock = 32;
  dim3 grid(outputChannelPairs / channelPairsPerBlock,batchSize,1);
  initialGlobalMatMulAddKernel<<<grid,256,0,stream>>>(
    spatialBuf,globalInput,weights);
}

__global__ void fusedQKRoPE19Half2Kernel(
  half2* __restrict__ qBuf,
  half2* __restrict__ kBuf,
  const float* __restrict__ freqs
) {
  constexpr int seqLen = 361;
  constexpr int pairsPerRow = 192;
  constexpr int numPairs = 16;

  int xy = blockIdx.x;
  int n = blockIdx.y;
  int hp = threadIdx.x;
  int h = hp / numPairs;
  int pairIdx = hp - h * numPairs;
  size_t idx = (size_t)(n * seqLen + xy) * pairsPerRow + hp;

  int x = xy % 19;
  int y = xy / 19;
  float freqX = freqs[(h * numPairs + pairIdx) * 2];
  float freqY = freqs[(h * numPairs + pairIdx) * 2 + 1];
  float angle = (float)x * freqX + (float)y * freqY;
  float cosVal;
  float sinVal;
  __sincosf(angle, &sinVal, &cosVal);

  half2 q = qBuf[idx];
  float q0 = __half2float(__low2half(q));
  float q1 = __half2float(__high2half(q));
  qBuf[idx] = __halves2half2(
    __float2half(q0 * cosVal - q1 * sinVal),
    __float2half(q0 * sinVal + q1 * cosVal));

  half2 k = kBuf[idx];
  float k0 = __half2float(__low2half(k));
  float k1 = __half2float(__high2half(k));
  kBuf[idx] = __halves2half2(
    __float2half(k0 * cosVal - k1 * sinVal),
    __float2half(k0 * sinVal + k1 * cosVal));
}

void launchFusedQKRoPE19Half2(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  int batchSize,
  cudaStream_t stream
) {
  dim3 blocks(361, batchSize);
  fusedQKRoPE19Half2Kernel<<<blocks, 192, 0, stream>>>(
    reinterpret_cast<half2*>(qBuf), reinterpret_cast<half2*>(kBuf), freqs);
}

union Half8Pack {
  uint4 packed;
  half2 values[4];
};

__global__ void swiGLU1152Half8Kernel(
  const uint4* __restrict__ a,
  const uint4* __restrict__ b,
  uint4* __restrict__ output,
  int vectorCount
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= vectorCount)
    return;

  Half8Pack av;
  Half8Pack bv;
  Half8Pack ov;
  av.packed = a[idx];
  bv.packed = b[idx];
#pragma unroll
  for(int e = 0; e < 4; e++) {
    float a0 = __half2float(__low2half(av.values[e]));
    float a1 = __half2float(__high2half(av.values[e]));
    float b0 = __half2float(__low2half(bv.values[e]));
    float b1 = __half2float(__high2half(bv.values[e]));
    float s0 = a0 / (1.0f + expf(-a0));
    float s1 = a1 / (1.0f + expf(-a1));
    ov.values[e] = __halves2half2(
      __float2half(s0 * b0), __float2half(s1 * b1));
  }
  output[idx] = ov.packed;
}

void launchSwiGLU1152Half8(
  const half* a,
  const half* b,
  half* output,
  int totalElements,
  cudaStream_t stream
) {
  int vectorCount = totalElements / 8;
  constexpr int threads = 256;
  int blocks = (vectorCount + threads - 1) / threads;
  swiGLU1152Half8Kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const uint4*>(a), reinterpret_cast<const uint4*>(b),
    reinterpret_cast<uint4*>(output), vectorCount);
}

template<int channels>
__global__ void affineSiluHalf2Kernel(
  const half2* __restrict__ input,
  half2* __restrict__ output,
  const half2* __restrict__ scale,
  const half2* __restrict__ bias
) {
  constexpr int pairs = channels / 2;
  int pair = threadIdx.x;
  int idx = blockIdx.x * pairs + pair;
  half2 value = __hfma2(input[idx], scale[pair], bias[pair]);
  float v0 = __half2float(__low2half(value));
  float v1 = __half2float(__high2half(value));
  float s0 = v0 / (1.0f + expf(-v0));
  float s1 = v1 / (1.0f + expf(-v1));
  output[idx] = __halves2half2(__float2half(s0), __float2half(s1));
}

void launchAffineSiluHalf2(
  const half* input,
  half* output,
  const half* scale,
  const half* bias,
  int totalRows,
  int channels,
  cudaStream_t stream
) {
  if(channels == 384) {
    affineSiluHalf2Kernel<384><<<totalRows, 192, 0, stream>>>(
      reinterpret_cast<const half2*>(input), reinterpret_cast<half2*>(output),
      reinterpret_cast<const half2*>(scale), reinterpret_cast<const half2*>(bias));
  }
  else {
    affineSiluHalf2Kernel<768><<<totalRows, 384, 0, stream>>>(
      reinterpret_cast<const half2*>(input), reinterpret_cast<half2*>(output),
      reinterpret_cast<const half2*>(scale), reinterpret_cast<const half2*>(bias));
  }
}

__global__ void affineSiluHalf2x3Kernel(
  const half2* __restrict__ input,
  half2* __restrict__ output,
  const half2* __restrict__ scale,
  const half2* __restrict__ bias,
  int pairCount,
  int pairsPerRow
) {
  int firstPair = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
#pragma unroll
  for(int e = 0; e < 3; e++) {
    int idx = firstPair + e;
    if(idx < pairCount) {
      int channelPair = idx % pairsPerRow;
      half2 value = __hfma2(input[idx], scale[channelPair], bias[channelPair]);
      float v0 = __half2float(__low2half(value));
      float v1 = __half2float(__high2half(value));
      output[idx] = __halves2half2(
        __float2half(v0 / (1.0f + expf(-v0))),
        __float2half(v1 / (1.0f + expf(-v1))));
    }
  }
}

void launchAffineSiluHalf2x3(
  const half* input,
  half* output,
  const half* scale,
  const half* bias,
  int totalRows,
  int channels,
  cudaStream_t stream
) {
  int pairsPerRow = channels / 2;
  int pairCount = totalRows * pairsPerRow;
  constexpr int threads = 512;
  int blocks = (pairCount + threads * 3 - 1) / (threads * 3);
  affineSiluHalf2x3Kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half2*>(input),
    reinterpret_cast<half2*>(output),
    reinterpret_cast<const half2*>(scale),
    reinterpret_cast<const half2*>(bias),
    pairCount, pairsPerRow);
}

union __align__(16) AffineSiluHalf8 {
  uint4 packed;
  half values[8];
};

__global__ void affineSiluFlatVec8C768Kernel(
  const uint4* __restrict__ input,
  uint4* __restrict__ output,
  const uint4* __restrict__ scale,
  const uint4* __restrict__ bias,
  int vectorCount
) {
  constexpr int vectorsPerRow = 768 / 8;
  int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if(vectorIndex >= vectorCount)
    return;
  int channelVector = vectorIndex % vectorsPerRow;
  AffineSiluHalf8 x;
  AffineSiluHalf8 s;
  AffineSiluHalf8 b;
  AffineSiluHalf8 y;
  x.packed = input[vectorIndex];
  s.packed = scale[channelVector];
  b.packed = bias[channelVector];
#pragma unroll
  for(int e = 0; e < 8; e++) {
    half value = __hfma(x.values[e], s.values[e], b.values[e]);
    float valueFloat = __half2float(value);
    y.values[e] = __float2half(
      valueFloat / (1.0f + expf(-valueFloat)));
  }
  output[vectorIndex] = y.packed;
}

void launchAffineSiluFlatVec8C768(
  const half* input,
  half* output,
  const half* scale,
  const half* bias,
  int totalRows,
  cudaStream_t stream
) {
  constexpr int threads = 256;
  int vectorCount = totalRows * (768 / 8);
  int blocks = (vectorCount + threads - 1) / threads;
  affineSiluFlatVec8C768Kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const uint4*>(input),
    reinterpret_cast<uint4*>(output),
    reinterpret_cast<const uint4*>(scale),
    reinterpret_cast<const uint4*>(bias),
    vectorCount);
}

__global__ void fusedPolicyP1Kernel(
  const half* __restrict__ input,
  float* __restrict__ output,
  const float* __restrict__ globalBias,
  const float* __restrict__ scale,
  const float* __restrict__ bias,
  int inputStride,
  int inputOffset
) {
  constexpr int xySize = 19 * 19;
  constexpr int channels = 96;
  int channel = threadIdx.x;
  int xy = blockIdx.x * blockDim.y + threadIdx.y;
  int batch = blockIdx.y;
  if(channel >= channels || xy >= xySize)
    return;

  size_t row = (size_t)batch * xySize + xy;
  size_t outputIdx = row * channels + channel;
  size_t inputIdx = row * inputStride + inputOffset + channel;
  float value = __half2float(input[inputIdx]);
  value = value + globalBias[batch * channels + channel];
  value = value * scale[channel] + bias[channel];
  output[outputIdx] = value / (1.0f + expf(-value));
}

void launchFusedPolicyP1(
  const half* input,
  float* output,
  const float* globalBias,
  const float* scale,
  const float* bias,
  int batchSize,
  int inputStride,
  int inputOffset,
  cudaStream_t stream
) {
  dim3 block(96, 5);
  dim3 grid((361 + block.y - 1) / block.y, batchSize);
  fusedPolicyP1Kernel<<<grid, block, 0, stream>>>(
    input, output, globalBias, scale, bias, inputStride, inputOffset);
}

template<int channels, int rowsPerBlock, bool writeHalf>
__global__ void headBNHalfToFloatKernel(
  const half* __restrict__ input,
  half* __restrict__ halfOutput,
  float* __restrict__ floatOutput,
  const half* __restrict__ scale,
  const half* __restrict__ bias,
  int batchSize,
  int inputStride,
  int inputOffset
) {
  constexpr int xySize = 19 * 19;
  int channel = threadIdx.x;
  int xy = blockIdx.x * rowsPerBlock + threadIdx.y;
  int batch = blockIdx.y;
  if(channel >= channels || xy >= xySize || batch >= batchSize)
    return;
  size_t outputIndex = ((size_t)batch * xySize + xy) * channels + channel;
  size_t inputIndex = ((size_t)batch * xySize + xy) * inputStride + inputOffset + channel;
  half affine = __hfma(input[inputIndex], scale[channel], bias[channel]);
  float value = __half2float(affine);
  half activated = __float2half(value / (1.0f + expf(-value)));
  if(writeHalf)
    halfOutput[outputIndex] = activated;
  floatOutput[outputIndex] = __half2float(activated);
}

bool launchHeadBNHalfToFloat(
  const half* input,
  half* halfOutput,
  float* floatOutput,
  const half* scale,
  const half* bias,
  int batchSize,
  int xySize,
  int channels,
  int inputStride,
  int inputOffset,
  cudaStream_t stream
) {
  if(input == NULL || floatOutput == NULL || scale == NULL || bias == NULL ||
     batchSize < 1 || xySize != 19 * 19 || inputStride < channels ||
     inputOffset < 0 || inputOffset + channels > inputStride)
    return false;
  if(channels == 96 && halfOutput == NULL) {
    dim3 block(96,5);
    dim3 grid((xySize + block.y - 1) / block.y,batchSize);
    headBNHalfToFloatKernel<96,5,false><<<grid,block,0,stream>>>(
      input,NULL,floatOutput,scale,bias,batchSize,inputStride,inputOffset);
  }
  else if(channels == 192 && halfOutput != NULL) {
    dim3 block(192,2);
    dim3 grid((xySize + block.y - 1) / block.y,batchSize);
    headBNHalfToFloatKernel<192,2,true><<<grid,block,0,stream>>>(
      input,halfOutput,floatOutput,scale,bias,batchSize,inputStride,inputOffset);
  }
  else
    return false;
  return cudaPeekAtLastError() == cudaSuccess;
}

__global__ void splitValueTerminalKernel(
  const float* __restrict__ combined,
  const float* __restrict__ bias,
  float* __restrict__ value,
  float* __restrict__ scoreValue,
  int valueChannels,
  int scoreValueChannels
) {
  int batch = blockIdx.x;
  int channel = threadIdx.x;
  int combinedChannels = valueChannels + scoreValueChannels;
  if(channel < valueChannels)
    value[(size_t)batch * valueChannels + channel] =
      combined[(size_t)batch * combinedChannels + channel] + bias[channel];
  else if(channel < combinedChannels) {
    int scoreChannel = channel - valueChannels;
    scoreValue[(size_t)batch * scoreValueChannels + scoreChannel] =
      combined[(size_t)batch * combinedChannels + channel] + bias[channel];
  }
}

bool launchSplitValueTerminal(
  const float* combined,
  const float* bias,
  float* value,
  float* scoreValue,
  int batchSize,
  int valueChannels,
  int scoreValueChannels,
  cudaStream_t stream
) {
  const int combinedChannels = valueChannels + scoreValueChannels;
  if(combined == NULL || bias == NULL || value == NULL || scoreValue == NULL ||
     batchSize < 1 || combinedChannels < 1 || combinedChannels > 1024)
    return false;
  splitValueTerminalKernel<<<batchSize,combinedChannels,0,stream>>>(
    combined,bias,value,scoreValue,valueChannels,scoreValueChannels);
  return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace Sm120Backend
