#ifndef KATAGO_CUDA_BACKEND_SM89_QKV_ROPE_GEMM_H
#define KATAGO_CUDA_BACKEND_SM89_QKV_ROPE_GEMM_H

#include "../neuralnet/cudaincludes.h"

#include <memory>

namespace Sm89Backend {

#ifdef KATAGO_ENABLE_SM89_QKV_ROPE_GEMM

// Fixed B13/19x19 FP16 QKV projection. Q and K are rotated in the CUTLASS
// epilogue after FP16 rounding; V is written unchanged.
class Sm89QKVRoPEGemmB13 {
 public:
  Sm89QKVRoPEGemmB13(const half* weights, const float* freqs);
  ~Sm89QKVRoPEGemmB13();
  Sm89QKVRoPEGemmB13(const Sm89QKVRoPEGemmB13&) = delete;
  Sm89QKVRoPEGemmB13& operator=(const Sm89QKVRoPEGemmB13&) = delete;

  bool apply(
    const half* input,
    half* output,
    int batchSize,
    int seqLen,
    int inChannels,
    int qkvChannels,
    int numHeads,
    int headDim,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#endif

} // namespace Sm89Backend

#endif
