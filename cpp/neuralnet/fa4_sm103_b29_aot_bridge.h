#ifndef KATAGO_FA4_SM103_B29_AOT_BRIDGE_H
#define KATAGO_FA4_SM103_B29_AOT_BRIDGE_H

#include <cuda_runtime.h>

#include <cstdint>

// Stable exact-shape C ABI around the generated FA4/CuTe SM103a object. The
// ordinary build links a fail-closed stub; an authenticated CMake opt-in swaps
// in the real bridge and object without a Python or TVM-FFI dependency.

struct KatagoFa4Sm103B29Context;

enum KatagoFa4Sm103B29Status : int32_t {
  KATAGO_FA4_SM103_B29_SUCCESS = 0,
  KATAGO_FA4_SM103_B29_INVALID_ARGUMENT = -1,
  KATAGO_FA4_SM103_B29_WRONG_DEVICE = -2,
  KATAGO_FA4_SM103_B29_MODULE_LOAD_FAILED = -3,
  KATAGO_FA4_SM103_B29_LAUNCH_FAILED = -4,
};

extern "C" KatagoFa4Sm103B29Context* katagoFa4Sm103B29Create(
  int device,
  int32_t* status
);

extern "C" void katagoFa4Sm103B29Destroy(
  KatagoFa4Sm103B29Context* context
);

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
);

#endif
