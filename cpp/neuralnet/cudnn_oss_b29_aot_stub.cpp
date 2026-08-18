#include "cudnn_oss_b29_aot_bridge.h"

// Default build: keep the stable ABI linkable without a generated CuTe object.
// Selecting the explicit tactic reaches Create(), receives MODULE_LOAD_FAILED,
// and aborts handle construction. No official path can be mislabeled active.

extern "C" KatagoCudnnOssB29Context* katagoCudnnOssB29Create(
  int device,
  int32_t* status
) {
  (void)device;
  if(status != nullptr)
    *status = KATAGO_CUDNN_OSS_B29_MODULE_LOAD_FAILED;
  return nullptr;
}

extern "C" void katagoCudnnOssB29Destroy(
  KatagoCudnnOssB29Context* context
) {
  (void)context;
}

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
) {
  (void)context;
  (void)input;
  (void)packedWeights;
  (void)ab12Scratch;
  (void)output;
  (void)alpha;
  (void)stream;
  (void)rows;
  (void)inputChannels;
  (void)packedChannels;
  (void)outputChannels;
  (void)usingFP16;
  return KATAGO_CUDNN_OSS_B29_MODULE_LOAD_FAILED;
}
