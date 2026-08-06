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

__global__ void batchSharedFusedQKRoPE19B13HalfKernel(
  half* __restrict__ qBuf,
  half* __restrict__ kBuf,
  const float* __restrict__ freqs
) {
  constexpr int seqLen = 361;
  constexpr int totalDim = 384;
  constexpr int numPairs = 16;
  constexpr int batchSize = 13;

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

void launchBatchSharedFusedQKRoPE19B13(
  half* qBuf,
  half* kBuf,
  const float* freqs,
  cudaStream_t stream
) {
  batchSharedFusedQKRoPE19B13HalfKernel<<<361, 192, 0, stream>>>(
    qBuf, kBuf, freqs);
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

__global__ void fusedPolicyP1B13Kernel(
  const half* __restrict__ input,
  float* __restrict__ output,
  const float* __restrict__ globalBias,
  const float* __restrict__ scale,
  const float* __restrict__ bias
) {
  constexpr int xySize = 19 * 19;
  constexpr int channels = 96;
  int channel = threadIdx.x;
  int xy = blockIdx.x * blockDim.y + threadIdx.y;
  int batch = blockIdx.y;
  if(channel >= channels || xy >= xySize)
    return;

  size_t row = (size_t)batch * xySize + xy;
  size_t idx = row * channels + channel;
  float value = __half2float(input[idx]);
  value = value + globalBias[batch * channels + channel];
  value = value * scale[channel] + bias[channel];
  output[idx] = value / (1.0f + expf(-value));
}

void launchFusedPolicyP1B13(
  const half* input,
  float* output,
  const float* globalBias,
  const float* scale,
  const float* bias,
  cudaStream_t stream
) {
  dim3 block(96, 5);
  dim3 grid((361 + block.y - 1) / block.y, 13);
  fusedPolicyP1B13Kernel<<<grid, block, 0, stream>>>(
    input, output, globalBias, scale, bias);
}

} // namespace Sm120Backend
