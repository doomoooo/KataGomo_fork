#ifndef KATAGO_CUDA_BACKEND_SM89_DUAL_GEMM_H
#define KATAGO_CUDA_BACKEND_SM89_DUAL_GEMM_H

#include "../neuralnet/cudaincludes.h"

#include <memory>
#include <string>

namespace Sm89Backend {

#ifdef KATAGO_ENABLE_SM89_DUAL_GEMM
class Sm89DualGemmSwiGLU {
 public:
  Sm89DualGemmSwiGLU(const half* weights, const std::string& tactic);
  ~Sm89DualGemmSwiGLU();
  Sm89DualGemmSwiGLU(const Sm89DualGemmSwiGLU&) = delete;
  Sm89DualGemmSwiGLU& operator=(const Sm89DualGemmSwiGLU&) = delete;

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
