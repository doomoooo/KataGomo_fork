#ifndef KATAGO_CUDA_BACKEND_SM89_LINEAR2_GEMM_H
#define KATAGO_CUDA_BACKEND_SM89_LINEAR2_GEMM_H

#include "../neuralnet/cudaincludes.h"

#include <memory>

namespace Sm89Backend {

#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
class Sm89Linear2GemmB13 {
 public:
  explicit Sm89Linear2GemmB13(const half* weights);
  ~Sm89Linear2GemmB13();
  Sm89Linear2GemmB13(const Sm89Linear2GemmB13&) = delete;
  Sm89Linear2GemmB13& operator=(const Sm89Linear2GemmB13&) = delete;

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

class Sm89Linear2BnGemmB13 {
 public:
  Sm89Linear2BnGemmB13(
    const half* weights,
    const half* bnScale,
    const half* bnBias
  );
  ~Sm89Linear2BnGemmB13();
  Sm89Linear2BnGemmB13(const Sm89Linear2BnGemmB13&) = delete;
  Sm89Linear2BnGemmB13& operator=(const Sm89Linear2BnGemmB13&) = delete;

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
class Sm89OutProjGemmB13 {
 public:
  explicit Sm89OutProjGemmB13(const half* weights);
  ~Sm89OutProjGemmB13();
  Sm89OutProjGemmB13(const Sm89OutProjGemmB13&) = delete;
  Sm89OutProjGemmB13& operator=(const Sm89OutProjGemmB13&) = delete;

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
class Sm89PreConvGemmB13 {
 public:
  explicit Sm89PreConvGemmB13(const half* weights);
  ~Sm89PreConvGemmB13();
  Sm89PreConvGemmB13(const Sm89PreConvGemmB13&) = delete;
  Sm89PreConvGemmB13& operator=(const Sm89PreConvGemmB13&) = delete;

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
class Sm89PostConvGemmB13 {
 public:
  explicit Sm89PostConvGemmB13(const half* weights);
  ~Sm89PostConvGemmB13();
  Sm89PostConvGemmB13(const Sm89PostConvGemmB13&) = delete;
  Sm89PostConvGemmB13& operator=(const Sm89PostConvGemmB13&) = delete;

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

class Sm89PostConvBnGemmB13 {
 public:
  Sm89PostConvBnGemmB13(
    const half* weights,
    const half* bnScale,
    const half* bnBias
  );
  ~Sm89PostConvBnGemmB13();
  Sm89PostConvBnGemmB13(const Sm89PostConvBnGemmB13&) = delete;
  Sm89PostConvBnGemmB13& operator=(const Sm89PostConvBnGemmB13&) = delete;

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
