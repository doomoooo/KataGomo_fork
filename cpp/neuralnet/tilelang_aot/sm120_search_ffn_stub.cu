#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" int sm120_search_ffn_batch() { return 0; }
extern "C" const char* sm120_search_ffn_id() { return "disabled"; }
extern "C" cudaError_t sm120_search_ffn_launch(
  const half*, const half*, const half*, half*, cudaStream_t
) {
  return cudaErrorInvalidValue;
}
