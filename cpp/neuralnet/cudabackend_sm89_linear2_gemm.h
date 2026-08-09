#ifndef KATAGO_CUDA_BACKEND_SM89_LINEAR2_GEMM_H
#define KATAGO_CUDA_BACKEND_SM89_LINEAR2_GEMM_H

#include "../neuralnet/cudaincludes.h"

#include <memory>
#include <string>

namespace Sm89Backend {

#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
class Sm89Linear2Gemm {
 public:
  Sm89Linear2Gemm(const half* weights, const std::string& tactic);
  ~Sm89Linear2Gemm();
  Sm89Linear2Gemm(const Sm89Linear2Gemm&) = delete;
  Sm89Linear2Gemm& operator=(const Sm89Linear2Gemm&) = delete;

  bool applyAccumulate(
    const half* input,
    half* output,
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

class Sm89Linear2BnGemm {
 public:
  Sm89Linear2BnGemm(
    const half* weights,
    const half* bnScale,
    const half* bnBias
  );
  ~Sm89Linear2BnGemm();
  Sm89Linear2BnGemm(const Sm89Linear2BnGemm&) = delete;
  Sm89Linear2BnGemm& operator=(const Sm89Linear2BnGemm&) = delete;

  bool applyAccumulateAndActivate(
    const half* input,
    half* residualOutput,
    half* activatedOutput,
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
#endif

#ifdef KATAGO_ENABLE_SM89_OUTPROJ_GEMM
class Sm89OutProjGemm {
 public:
  Sm89OutProjGemm(const half* weights, const std::string& tactic);
  ~Sm89OutProjGemm();
  Sm89OutProjGemm(const Sm89OutProjGemm&) = delete;
  Sm89OutProjGemm& operator=(const Sm89OutProjGemm&) = delete;

  bool applyAccumulate(
    const half* input,
    half* output,
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
#endif

#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
class Sm89PreConvGemm {
 public:
  Sm89PreConvGemm(const half* weights, const std::string& tactic);
  ~Sm89PreConvGemm();
  Sm89PreConvGemm(const Sm89PreConvGemm&) = delete;
  Sm89PreConvGemm& operator=(const Sm89PreConvGemm&) = delete;

  bool apply(
    const half* input,
    half* output,
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
#endif

#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
class Sm89PostConvGemm {
 public:
  Sm89PostConvGemm(const half* weights, const std::string& tactic);
  ~Sm89PostConvGemm();
  Sm89PostConvGemm(const Sm89PostConvGemm&) = delete;
  Sm89PostConvGemm& operator=(const Sm89PostConvGemm&) = delete;

  bool applyAccumulate(
    const half* input,
    half* output,
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

class Sm89PostConvBnGemm {
 public:
  Sm89PostConvBnGemm(
    const half* weights,
    const half* bnScale,
    const half* bnBias
  );
  ~Sm89PostConvBnGemm();
  Sm89PostConvBnGemm(const Sm89PostConvBnGemm&) = delete;
  Sm89PostConvBnGemm& operator=(const Sm89PostConvBnGemm&) = delete;

  bool applyAccumulateAndActivate(
    const half* input,
    half* residualOutput,
    half* activatedOutput,
    int batchSize,
    int seqLen,
    int inChannels,
    int outChannels,
    cudaStream_t stream
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
#endif

} // namespace Sm89Backend

#endif
