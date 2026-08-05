#ifndef KATAGO_CUDA_BACKEND_SM120_H
#define KATAGO_CUDA_BACKEND_SM120_H

#include "../core/config_parser.h"
#include "../neuralnet/desc.h"

#include <memory>
#include <string>

// SM120-specific CUDA backend.
//
// All Blackwell-SM120 kernels, AOT handles, weight sharing, persisting-L2 windows and config
// switches live here (cudabackend_sm120.h/cpp). The official backend files (cudabackend.cpp,
// cudahelpers.cu, cudautils.cpp, ...) are NOT modified with SM120 branches; they only contain a
// thin dispatch: ComputeHandle builds a Sm120Model on SM120 and routes apply() through it.
//
// Rebuild roadmap (from /workspace/cuda-optimization-history.md, final accepted config):
//   0. scaffold: Sm120Model delegates to the official model (bit-identical)  [current state]
//   1. FA4 both16 attention (fixed B19/S361/H12/D32, noncausal, shape fallback to official)
//   2. wide QKV CuTe AOT C384->QKV1152, batch-shared RoPE, fused residual epilogues
//   3. TileLang fused FFN + linear2/out-projection AOT
//   4. RMSNorm/silu/head custom kernels, initial-conv frontend, persisting-L2, weight sharing
//   5. final batch/stream scan + full accuracy regression per stage

struct CudaHandles;    // defined in cudabackend.cpp
struct ScratchBuffers; // defined in cudabackend.cpp
struct Logger;         // defined in core/logger.h

namespace Sm120Backend {

// Trampoline for the official backend apply(). cudabackend.cpp supplies it so Sm120Model never
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
  // Master switch. When false, SM120 keeps the official backend path entirely (A/B control).
  bool enabled = true;

  // Historical optimization switches, defaults = final accepted values from the 5080 history.
  bool useFlashAttention = true;
  std::string flashAttentionAccum = "both16"; // "none","fp32","qk16","pv16","both16"
  bool useWideQKV = true;
  bool useQKVGemmAot = true;
  bool useQKVGemmRopeAot = false;
  bool useFusedQKRoPE = true;
  bool useBatchSharedRoPE = true;
  bool useBatchSharedRoPEUnroll19 = true;
  bool useBatchSharedRoPETwoWay = false;
  bool useFusedResidual = true;
  bool useProjectionGemmLt = false;
  bool useFusedFFN = true;
  bool useFusedRMSNormFFN = false;
  bool useRMSNormQKVGemmAot = false;
  bool useGraph = false;
  bool usePersistingL2Trunk = true;
  bool usePersistingL2Inner = true;
  bool useOuterProjectionAot = true;
  bool shareModelWeights = true;
  bool shareWideQKVWeights = false;
  bool shareOuterProjectionWeights = false;
  bool useInitialConvFrontend = true;
  bool useInitialConvBiasFrontend = false;
  bool useInitialGlobalMatMulAdd = true;
  bool useFusedPolicyP1 = true;
  bool useHeadBNHalfToFloat = true;
  bool useWideHeadProjection = true;
};

bool isSm120Arch(int majorComputeCapability, int minorComputeCapability);

// Reads all cuda*Sm120* / cuda* config keys relevant to the SM120 path. Unknown accum values throw.
Options parseOptions(ConfigParser& cfg);

// The SM120 model implementation. The official model is kept alive by the caller and is used as
// the correctness fallback until each stage of the rebuild lands.
class Sm120Model {
 public:
  Sm120Model(
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
  ~Sm120Model();

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

} // namespace Sm120Backend

#endif
