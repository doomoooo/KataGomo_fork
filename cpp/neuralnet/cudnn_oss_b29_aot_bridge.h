#ifndef KATAGO_CUDNN_OSS_B29_AOT_BRIDGE_H
#define KATAGO_CUDNN_OSS_B29_AOT_BRIDGE_H

#include <cuda_runtime.h>

#include <cstdint>

// Stable, architecture-exact C ABI around the generated CuTe-DSL header. The
// default build links the fail-closed stub; an authenticated CMake opt-in swaps
// in this bridge and the exact no-AB12 fast object without Python,
// TVM-FFI, or cuDNN Frontend runtime dependency.

struct KatagoCudnnOssB29Context;

enum KatagoCudnnOssB29Status : int32_t {
  KATAGO_CUDNN_OSS_B29_SUCCESS = 0,
  KATAGO_CUDNN_OSS_B29_INVALID_ARGUMENT = -1,
  KATAGO_CUDNN_OSS_B29_WRONG_DEVICE = -2,
  KATAGO_CUDNN_OSS_B29_MODULE_LOAD_FAILED = -3,
};

extern "C" KatagoCudnnOssB29Context* katagoCudnnOssB29Create(
  int device,
  int32_t* status
);

extern "C" void katagoCudnnOssB29Destroy(KatagoCudnnOssB29Context* context);

extern "C" int32_t katagoCudnnOssB29Launch(
  KatagoCudnnOssB29Context* context,
  const void* input,
  const void* packedWeights,
  void* ab12Scratch,
  void* output,
  float alpha,
  cudaStream_t stream,
  int rows,
  int inputChannels,
  int packedChannels,
  int outputChannels,
  int usingFP16
);

#endif
