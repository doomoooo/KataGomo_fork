#ifndef KATAGO_SM120_DUAL_FFN_SHARED_A_H_
#define KATAGO_SM120_DUAL_FFN_SHARED_A_H_

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Dynamic-B CUTLASS shared-A dual GEMM followed by the original SwiGLU
// epilogue. The handle owns independent initialized states for each token
// count, so a B4-B32 plan can select this route without a fixed-B fallback.
void* katago_create_dual_ffn_shared_a_sm120();
void katago_destroy_dual_ffn_shared_a_sm120(void* handle);
int katago_launch_dual_ffn_shared_a_sm120(
  void* handle,
  const void* input,
  const void* linearWeights,
  const void* gateWeights,
  void* output,
  int tokens,
  cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
