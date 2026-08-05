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
// device weight buffers and scratch allocator, and implements the forward using ModelDesc.
// cudabackend.cpp / cudahelpers.cu remain untouched and are used only as the official fallback
// when this forward does not support a model/shape.

namespace Sm89Backend {

struct Sm89Ctx {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
  cudaStream_t stream;

  Sm89Ctx();
  ~Sm89Ctx();
  Sm89Ctx(const Sm89Ctx&) = delete;
  Sm89Ctx& operator=(const Sm89Ctx&) = delete;
};

struct Sm89Scratch {
  SimpleAllocator<void*> allocator;
  void* zeroBuf;
  void* oneBuf;

  explicit Sm89Scratch(bool useFP16);
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
    bool useWideQKV,
    bool useWideFFN,
    bool useFusedResidual,
    bool useRMSNormOpt
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
