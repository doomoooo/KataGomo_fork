#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" cudaError_t sm120_search_qkv_launch(
  const half* input, const half* weights, half* output, cudaStream_t stream
);

namespace {

void cudaCheck(cudaError_t status, const char* operation) {
  if(status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

void cublasCheck(cublasStatus_t status, const char* operation) {
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s: cuBLAS status %d\n", operation, static_cast<int>(status));
    std::exit(2);
  }
}

} // namespace

int main(int argc, char** argv) {
  if(argc != 5) {
    std::fprintf(stderr, "usage: %s DEVICE BATCH WARMUP ITERATIONS\n", argv[0]);
    return 2;
  }
  const int device = std::atoi(argv[1]);
  const int batch = std::atoi(argv[2]);
  const int warmup = std::atoi(argv[3]);
  const int iterations = std::atoi(argv[4]);
  if(batch < 1 || warmup < 0 || iterations < 1)
    return 2;
  cudaCheck(cudaSetDevice(device), "cudaSetDevice");

  constexpr int k = 384;
  constexpr int n = 1152;
  const int m = batch * 361;
  std::vector<half> hostA(static_cast<size_t>(m) * k);
  std::vector<half> hostB(static_cast<size_t>(k) * n);
  for(size_t i = 0; i < hostA.size(); i++)
    hostA[i] = __float2half(0.15f * std::sin(static_cast<float>(i % 1009) * 0.017f));
  for(size_t i = 0; i < hostB.size(); i++)
    hostB[i] = __float2half(0.05f * std::cos(static_cast<float>(i % 1013) * 0.013f));

  half* deviceA = nullptr;
  half* deviceB = nullptr;
  half* deviceCandidate = nullptr;
  half* deviceReference = nullptr;
  cudaCheck(cudaMalloc(&deviceA, hostA.size() * sizeof(half)), "cudaMalloc(A)");
  cudaCheck(cudaMalloc(&deviceB, hostB.size() * sizeof(half)), "cudaMalloc(B)");
  cudaCheck(cudaMalloc(&deviceCandidate, static_cast<size_t>(m) * n * sizeof(half)), "cudaMalloc(candidate)");
  cudaCheck(cudaMalloc(&deviceReference, static_cast<size_t>(m) * n * sizeof(half)), "cudaMalloc(reference)");
  cudaCheck(cudaMemcpy(deviceA, hostA.data(), hostA.size() * sizeof(half), cudaMemcpyHostToDevice), "copy A");
  cudaCheck(cudaMemcpy(deviceB, hostB.data(), hostB.size() * sizeof(half), cudaMemcpyHostToDevice), "copy B");

  cublasHandle_t handle;
  cublasCheck(cublasCreate(&handle), "cublasCreate");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Row-major C[M,N] = A[M,K] * B[K,N] is the equivalent column-major
  // product C^T[N,M] = B^T[N,K] * A^T[K,M].
  cublasCheck(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
    &alpha, deviceB, CUDA_R_16F, n, deviceA, CUDA_R_16F, k,
    &beta, deviceReference, CUDA_R_16F, n,
    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
  ), "reference GEMM");
  cudaCheck(sm120_search_qkv_launch(deviceA, deviceB, deviceCandidate, nullptr), "candidate correctness launch");
  cudaCheck(cudaDeviceSynchronize(), "correctness synchronize");

  std::vector<half> candidate(static_cast<size_t>(m) * n);
  std::vector<half> reference(static_cast<size_t>(m) * n);
  cudaCheck(cudaMemcpy(candidate.data(), deviceCandidate, candidate.size() * sizeof(half), cudaMemcpyDeviceToHost), "copy candidate");
  cudaCheck(cudaMemcpy(reference.data(), deviceReference, reference.size() * sizeof(half), cudaMemcpyDeviceToHost), "copy reference");
  double squared = 0.0;
  double maxAbs = 0.0;
  for(size_t i = 0; i < candidate.size(); i++) {
    const double diff = static_cast<double>(__half2float(candidate[i])) - __half2float(reference[i]);
    squared += diff * diff;
    maxAbs = std::max(maxAbs, std::abs(diff));
  }
  const double rmse = std::sqrt(squared / static_cast<double>(candidate.size()));
  if(!std::isfinite(rmse) || rmse > 0.02 || maxAbs > 0.10) {
    std::fprintf(stderr, "correctness failed: max_abs=%g rmse=%g\n", maxAbs, rmse);
    return 3;
  }

  for(int i = 0; i < warmup; i++)
    cudaCheck(sm120_search_qkv_launch(deviceA, deviceB, deviceCandidate, nullptr), "warmup launch");
  cudaCheck(cudaDeviceSynchronize(), "warmup synchronize");
  std::vector<float> samples;
  for(int sample = 0; sample < 5; sample++) {
    cudaEvent_t begin;
    cudaEvent_t end;
    cudaCheck(cudaEventCreate(&begin), "event begin create");
    cudaCheck(cudaEventCreate(&end), "event end create");
    cudaCheck(cudaEventRecord(begin), "event begin record");
    for(int i = 0; i < iterations; i++)
      cudaCheck(sm120_search_qkv_launch(deviceA, deviceB, deviceCandidate, nullptr), "timed launch");
    cudaCheck(cudaEventRecord(end), "event end record");
    cudaCheck(cudaEventSynchronize(end), "event synchronize");
    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, begin, end), "event elapsed");
    samples.push_back(milliseconds * 1000.0f / static_cast<float>(iterations));
    cudaCheck(cudaEventDestroy(begin), "event begin destroy");
    cudaCheck(cudaEventDestroy(end), "event end destroy");
  }
  std::vector<float> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  std::printf("{\"batch\":%d,\"s1_us_samples\":[", batch);
  for(size_t i = 0; i < samples.size(); i++)
    std::printf("%s%.9g", i == 0 ? "" : ",", samples[i]);
  std::printf(
    "],\"s1_us_median\":%.9g,\"correctness_against_cublas\":{\"max_abs\":%.9g,\"rmse\":%.9g}}\n",
    sorted[sorted.size() / 2], maxAbs, rmse
  );

  cublasDestroy(handle);
  cudaFree(deviceReference);
  cudaFree(deviceCandidate);
  cudaFree(deviceB);
  cudaFree(deviceA);
  return 0;
}
