#ifndef KATAGO_CUDA_BACKEND_SM89_DUAL_GEMM_H
#define KATAGO_CUDA_BACKEND_SM89_DUAL_GEMM_H

#include "../neuralnet/cudaincludes.h"

#include <memory>

namespace Sm89Backend {

#ifdef KATAGO_ENABLE_SM89_DUAL_GEMM
class Sm89DualGemmSwiGLUB13 {
 public:
  Sm89DualGemmSwiGLUB13(const half* weights, bool useHalf2Tanh);
  ~Sm89DualGemmSwiGLUB13();
  Sm89DualGemmSwiGLUB13(const Sm89DualGemmSwiGLUB13&) = delete;
  Sm89DualGemmSwiGLUB13& operator=(const Sm89DualGemmSwiGLUB13&) = delete;

  bool apply(
    const half* input,
    half* output,
    int batchSize,
    int seqLen,
    int inChannels,
    int ffnChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
#endif

} // namespace Sm89Backend

#endif
