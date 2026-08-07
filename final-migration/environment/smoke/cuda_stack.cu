#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

__global__ void compile_only_kernel(float* output) {
  if (threadIdx.x == 0) output[0] = 1.0f;
}

int main() {
  // Compile and link validation only. Runtime GPU work is deliberately omitted
  // so environment verification does not interfere with optimization sessions.
  return CUDNN_VERSION > 0 && CUBLAS_VER_MAJOR > 0 ? 0 : 1;
}
