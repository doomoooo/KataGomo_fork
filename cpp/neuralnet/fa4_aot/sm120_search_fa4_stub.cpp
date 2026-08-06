#include <cuda_runtime.h>

extern "C" int sm120_search_fa4_batch() { return 0; }
extern "C" const char* sm120_search_fa4_id() { return "disabled"; }
extern "C" cudaError_t sm120_search_fa4_launch(
  void*, void*, void*, void*, int, int, int, int, float, cudaStream_t
) {
  return cudaErrorInvalidValue;
}
