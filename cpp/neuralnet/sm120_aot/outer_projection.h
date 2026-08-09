#ifndef KATAGO_OUTER_PROJECTION_SM120_H_
#define KATAGO_OUTER_PROJECTION_SM120_H_

#define KATAGO_OUTER_PROJECTION_SM120_HAS_FUSED_SILU 1

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void* katago_create_outer_projection_down_sm120(
  const void* weights, const char* tactic);
void* katago_create_head_projection_sm120(
  const void* weights, int outputChannels, const char* tactic);
void katago_destroy_outer_projection_down_sm120(void* handle);
int katago_launch_outer_projection_down_sm120(
  void* handle,
  const void* input,
  void* output,
  int rows,
  cudaStream_t stream
);

void* katago_create_outer_projection_up_sm120(
  const void* weights, const char* tactic);
void katago_destroy_outer_projection_up_sm120(void* handle);
int katago_launch_outer_projection_up_sm120(
  void* handle,
  const void* input,
  void* residualAndOutput,
  int rows,
  cudaStream_t stream
);
int katago_launch_outer_projection_up_silu_sm120(
  void* handle,
  const void* input,
  void* residualAndOutput,
  const void* scale,
  const void* bias,
  void* activatedOutput,
  int rows,
  cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif  // KATAGO_OUTER_PROJECTION_SM120_H_
