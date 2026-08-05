#ifndef KATAGO_CUDA_BACKEND_SM89_H
#define KATAGO_CUDA_BACKEND_SM89_H

#include "../core/config_parser.h"
#include "../neuralnet/desc.h"

#include <memory>
#include <string>

// SM89-specific CUDA backend.
//
// All Ada-Lovelace SM89 kernels, AOT handles, weight sharing, persisting-L2 windows and config
// switches live here (cudabackend_sm89.h/cpp). The official backend files (cudabackend.cpp,
// cudahelpers.cu, cudautils.cpp, ...) only contain a thin dispatch: ComputeHandle builds an
// Sm89Model on SM89 and routes apply() through it. cudabackend.cpp remains the official fallback.
//
// Rebuild roadmap (from /workspace/cuda-optimization-history.md and SM120 scaffold):
//   0. scaffold: Sm89Model delegates to the official model (bit-identical)  [current state]
//   1. wide QKV / wide FFN projections, fused residual epilogues
//   2. GEMM / attention AOT with higher-occupancy SM89 kernels
//   3. RMSNorm/silu/head kernels, persisting-L2, weight sharing, initial-conv frontend
//   4. final batch/stream scan + full accuracy regression per stage

struct CudaHandles;    // defined in cudabackend.cpp
struct ScratchBuffers; // defined in cudabackend.cpp
struct Logger;         // defined in core/logger.h

namespace Sm89Backend {

// Trampoline for the official backend apply(). cudabackend.cpp supplies it so Sm89Model never
// needs the internal Model type; ctx is the official Model pointer.
typedef void (*OfficialApplyFn)(
  void* ctx,
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
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

struct Options {
  // Master switch. When false, SM89 keeps the official backend path entirely (A/B control).
  bool enabled = true;

  // Historical optimization switches; defaults are conservative so the scaffold is bit-identical.
  // Each stage lands behind its own switch and is validated before flipping its default.
  bool useWideQKV = false;
  bool useWideFFN = false;
  bool useFusedResidual = false;
  bool useMatmulLt = false;
  bool useFusedQKRoPE = false;
  bool useBatchSharedRoPE = false;
  bool useFusedFFN = false;
  bool useInitialConvFrontend = false;
  bool useInitialGlobalMatMulAdd = false;
  bool useFusedPolicyP1 = false;
  bool useHeadBNHalfToFloat = false;
  bool useWideHeadProjection = false;
  bool usePersistingL2Trunk = false;
  bool usePersistingL2Inner = false;
  bool shareModelWeights = false;
};

bool isSm89Arch(int majorComputeCapability, int minorComputeCapability);

// Reads all cuda*Sm89* / cuda* config keys relevant to the SM89 path.
Options parseOptions(ConfigParser& cfg);

// The SM89 model implementation. The official model is kept alive by the caller and is used as
// the correctness fallback until each stage of the rebuild lands.
class Sm89Model {
 public:
  Sm89Model(
    void* officialApplyContext,
    OfficialApplyFn officialApply,
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC,
    const Options& options
  );
  ~Sm89Model();

  void setLogger(Logger* logger);

  // Mirrors Model::apply exactly so ComputeHandle can dispatch without touching the official
  // getOutput/benchmarkOutput paths.
  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
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
  void* officialApplyContext;
  OfficialApplyFn officialApply;
  CudaHandles* cudaHandles;
  const ModelDesc* desc;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const bool inputsUseNHWC;
  const bool useFP16;
  const bool useNHWC;
  Options options;
  Logger* logger;
  bool loggedFallback;

  // TODO(rebuild): device weight buffers, AOT kernel handles, per-GPU shared-weight caches,
  // persisting-L2 access-policy windows, scratch/workspace plan.
};

} // namespace Sm89Backend

#endif
