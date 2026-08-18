#include "fa4_sm103_b29_aot_bridge.h"

// Default build: keep the ABI linkable but fail module creation. Therefore a
// configured SM103 FA4 tactic can never silently execute the official SDPA.

extern "C" KatagoFa4Sm103B29Context* katagoFa4Sm103B29Create(
  int device,
  int32_t* status
) {
  (void)device;
  if(status != nullptr)
    *status = KATAGO_FA4_SM103_B29_MODULE_LOAD_FAILED;
  return nullptr;
}

extern "C" void katagoFa4Sm103B29Destroy(
  KatagoFa4Sm103B29Context* context
) {
  (void)context;
}

extern "C" int32_t katagoFa4Sm103B29Launch(
  KatagoFa4Sm103B29Context* context,
  const void* q,
  const void* k,
  const void* v,
  const void* mask,
  void* output,
  float scale,
  cudaStream_t stream,
  int batch,
  int sequence,
  int heads,
  int kvHeads,
  int qkHeadDim,
  int vHeadDim,
  int packedQKV,
  int usingFP16
) {
  (void)context;
  (void)q;
  (void)k;
  (void)v;
  (void)mask;
  (void)output;
  (void)scale;
  (void)stream;
  (void)batch;
  (void)sequence;
  (void)heads;
  (void)kvHeads;
  (void)qkHeadDim;
  (void)vHeadDim;
  (void)packedQKV;
  (void)usingFP16;
  return KATAGO_FA4_SM103_B29_MODULE_LOAD_FAILED;
}
