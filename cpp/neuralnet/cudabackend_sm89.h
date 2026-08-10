#ifndef KATAGO_CUDA_BACKEND_SM89_H
#define KATAGO_CUDA_BACKEND_SM89_H

#include "../core/config_parser.h"
#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/desc.h"

#include <memory>
#include <set>
#include <string>

// SM89-specific CUDA backend.
//
// All Ada-Lovelace SM89 kernels, AOT handles, weight sharing, persisting-L2 windows and config
// switches live here (cudabackend_sm89.h/cpp). The official backend files (cudabackend.cpp,
// cudahelpers.cu, cudautils.cpp, ...) only contain a thin dispatch: ComputeHandle builds an
// Sm89Model on SM89 and routes apply() through it. cudabackend.cpp remains the official fallback.
//
// Supported transformer shapes use Sm89Forward directly. Unsupported models
// keep the official Model traversal as the explicit correctness fallback.

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
  // Stage-1 standalone forward. When true and the model is supported, Sm89Model runs its own
  // SM89-specific forward instead of the official fallback.
  bool useForward = true;

  // Historical optimization coordinates. The generated plan selects exact
  // runtime tactics; unsupported explicit selections fail in Sm89Forward.
  bool useWideQKV = false;
  bool useWideFFN = false;
  bool useFusedResidual = false;
  bool useRMSNormOpt = false;
  // Number of independent row-warps per RMSNorm CTA. Stage39 measured both.
  int rmsNormRowsPerBlock = 4;
  bool useMatmulLt = false;
  bool useFusedQKRoPE = false;
  bool usePrecomputedQKRoPE = false;
  bool useQKVRoPEGemm = false;
  bool useSplitQKVRoPEGemm = false;
  int plainQKVVariant = 0;
  int ropeBatchGroup = 1;
  // 0 disabled; 1 M128N112 fp32; 2 M128N96 fp32; 3 M64N96 packed-GQA
  // fp32; 4 M64N96 unpacked fp32; 5 M64N96 unpacked both16.
  int flashAttentionTactic = 0;
  std::string flashAttentionTacticName = "disabled";
  bool useDualGemmSwiGLU = false;
  bool useLinear2Gemm = false;
  bool useOutProjGemm = false;
  bool usePreConvGemm = false;
  bool usePostConvGemm = false;
  std::string dualFfnCutlassTactic = "disabled";
  std::string linear2CutlassTactic = "disabled";
  std::string outProjCutlassTactic = "disabled";
  std::string preConvCutlassTactic = "disabled";
  std::string postConvCutlassTactic = "disabled";
  bool usePostConvBNSilu = false;
  bool useLinear2PostBNSilu = false;
  bool useBatchSharedRoPE = false;
  bool useFusedFFN = false;
  bool useInitialConvFrontend = false;
  bool useInitialGlobalMatMulAdd = false;
  // 0 disables the fused policy P1 kernel. Stage25 retained both one-row and
  // five-row launch geometries as distinct, numerically identical tactics.
  int policyP1RowsPerBlock = 0;
  bool useHeadBNHalfToFloat = false;
  bool useWideHeadProjection = false;
  bool useFusedValueTerminal = false;
  bool usePersistingL2Trunk = false;
  bool usePersistingL2Inner = false;
  float persistingL2HitRatio = 1.0f;
  bool useScaleBiasSiluVec8 = false;
  bool useScaleBiasSiluVec8C384 = false;
  bool useScaleBiasSiluVec4C384 = false;
  bool shareModelWeights = false;
  std::string dualFfnAotTactic = "disabled";
  std::string linear2AotTactic = "disabled";
  int serverThreads = 1;
};

bool isSm89Arch(int majorComputeCapability, int minorComputeCapability);

// Reads all cuda*Sm89* / cuda* config keys relevant to the SM89 path.
Options parseOptions(ConfigParser& cfg);

class Sm89Forward;

// SM89 forward owner with an explicit whole-model compatibility fallback.
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
    cudaStream_t stream,
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
    size_t workspaceBytes,
    cudaEvent_t inputConsumedEvent = nullptr,
    cudaEvent_t outputConsumedEvent = nullptr
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
  std::set<std::string> loggedActiveTactics;
  std::unique_ptr<Sm89Forward> forward;
  bool forwardActive;
};

} // namespace Sm89Backend

#endif
