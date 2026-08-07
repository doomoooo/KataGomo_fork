#ifndef KATAGO_CUDA_BACKEND_SM89_FORWARD_H
#define KATAGO_CUDA_BACKEND_SM89_FORWARD_H

#include "../core/simpleallocator.h"
#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/desc.h"

#include <cublas_v2.h>
#include <cudnn.h>
#include <memory>
#include <string>
#include <vector>

// SM89-specific forward implementation.
//
// This is a self-contained Ada-Lovelace forward path: it owns its own cuBLAS/cuDNN handles,
// device weight buffers and scratch allocator, borrows the caller's CUDA stream, and implements
// the forward using ModelDesc.
// cudabackend.cpp / cudahelpers.cu remain untouched and are used only as the official fallback
// when this forward does not support a model/shape.

namespace Sm89Backend {

// Numeric device capabilities queried from the CUDA runtime. Performance
// policy must not depend on marketing names such as "RTX 4090".
struct Sm89DeviceCapabilities {
  int deviceOrdinal;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int numSms;
  int warpSize;
  int maxThreadsPerSm;
  int maxThreadsPerBlock;
  int registersPerSm;
  size_t sharedMemoryPerSm;
  size_t sharedMemoryPerBlockOptin;
  int l2CacheBytes;
};

struct Sm89Ctx {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
  cudaStream_t stream;
  Sm89DeviceCapabilities deviceCaps;
  int serverThreads;
  std::string dualFfnAotTactic;
  std::string linear2AotTactic;

  Sm89Ctx(
    cudaStream_t stream,
    int serverThreads,
    const std::string& dualFfnAotTactic,
    const std::string& linear2AotTactic
  );
  ~Sm89Ctx();
  Sm89Ctx(const Sm89Ctx&) = delete;
  Sm89Ctx& operator=(const Sm89Ctx&) = delete;
};

struct Sm89Scratch {
  SimpleAllocator<void*> allocator;
  void* zeroBuf;
  void* oneBuf;
  void* fullBoardMaskSumBuf;

  Sm89Scratch(bool useFP16, bool useExactMaskElision, int maxBatchSize, int xySize);
  ~Sm89Scratch();
  Sm89Scratch(const Sm89Scratch&) = delete;
  Sm89Scratch& operator=(const Sm89Scratch&) = delete;

  size_t getBufSizeXY(int channels, int maxBatchSize, int xySize, bool useFP16) const;
  size_t getBufSizeXYFloat(int channels, int maxBatchSize, int xySize) const;
  size_t getBufSizeFloat(int channels, int maxBatchSize) const;
  size_t getBufSize(int channels, int maxBatchSize, bool useFP16) const;
};

class Sm89Forward {
 public:
  Sm89Forward(
    const ModelDesc* desc,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC,
    cudaStream_t stream,
    bool useWideQKV,
    bool useWideFFN,
    bool useFusedResidual,
    bool useRMSNormOpt,
    bool useFusedQKRoPE,
    bool usePrecomputedQKRoPE,
    bool useQKVRoPEGemm,
    bool useSplitQKVRoPEGemm,
    int plainQKVVariant,
    int ropeBatchGroup,
    bool useFlashAttention,
    bool useFlashAttentionBoth16,
    bool useDualGemmSwiGLU,
    bool useDualGemmSwiGLUHalf2Tanh,
    bool useLinear2Gemm,
    bool useOutProjGemm,
    bool usePreConvGemm,
    bool usePostConvGemm,
    bool usePostConvBNSilu,
    bool useLinear2PostBNSilu,
    bool usePersistingL2Trunk,
    float persistingL2TrunkHitRatio,
    bool usePersistingL2Inner,
    float persistingL2InnerHitRatio,
    bool useScaleBiasSiluVec8,
    bool useScaleBiasSiluVec4C384,
    bool useInitialConvFrontend,
    bool useInitialGlobalMatMulAdd,
    bool useFusedPolicyP1,
    bool useHeadBNHalfToFloat,
    bool useWideHeadProjection,
    bool useExactMaskElision,
    bool useFusedValueTerminal,
    const std::string& dualFfnAotTactic,
    const std::string& linear2AotTactic,
    int serverThreads,
    bool shareModelWeights
  );
  ~Sm89Forward();
  Sm89Forward(const Sm89Forward&) = delete;
  Sm89Forward& operator=(const Sm89Forward&) = delete;

  // Returns false if this model is not supported by the SM89 forward; the caller must fall back.
  static bool supports(const ModelDesc& desc, bool useFP16, bool useNHWC);

  void apply(
    int batchSize,
    bool requireExactNNLen,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    float* policyPassBuf,
    float* policyBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  );

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace Sm89Backend

#endif
