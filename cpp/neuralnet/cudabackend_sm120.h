#ifndef KATAGO_CUDA_BACKEND_SM120_H
#define KATAGO_CUDA_BACKEND_SM120_H

#include "../neuralnet/cudaincludes.h"
#include "../core/config_parser.h"
#include "../neuralnet/desc.h"

#include <memory>
#include <string>
#include <unordered_map>

// SM120-specific CUDA backend.
//
// All Blackwell-SM120 kernels, AOT handles, weight sharing, persisting-L2 windows and config
// switches live here (cudabackend_sm120.h/cpp). The official backend files (cudabackend.cpp,
// cudahelpers.cu, cudautils.cpp, ...) are NOT modified with SM120 branches; they only contain a
// thin dispatch: ComputeHandle builds a Sm120Model on SM120 and routes apply() through it.
//
// Rebuild roadmap (from /workspace/cuda-optimization-history.md, final accepted config):
//   0. scaffold: Sm120Model delegates to the official model (bit-identical)  [done]
//   1. FA4 both16 attention (B1-B13/S361/H12/D32, noncausal, shape fallback to official)
//      [done: checked-in AOT artifact uses FP16 QK/PV accumulation]
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

// Hook installed on SM120 compute handles. Called from the official attention
// block (thin dispatch only) with the already-computed Q/K/V buffers in BSHD
// layout. Returns true if the attention output was produced and the official
// SDPA/custom path must be skipped; false means fall back to the official path.
typedef bool (*Sm120AttentionFn)(
  void* ctx,
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  void* qBuf,
  void* kBuf,
  void* vBuf,
  void* maskBuf,
  void* attnOutBuf,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFP16,
  cudaStream_t stream,
  void* workspaceBuf,
  size_t workspaceBytes
);

typedef bool (*Sm120FFNSingleGemmFn)(
  void* ctx,
  cublasHandle_t cublas,
  cudaStream_t stream,
  const void* linear1Weights,
  const void* linearGateWeights,
  const void* inputBuf,
  void* wideScratchBuf,
  void* ffnOutBuf,
  int matBatchSize,
  int numChannels,
  int ffnChannels,
  bool usingFP16
);

typedef bool (*Sm120MatMulFn)(
  void* ctx,
  cudaStream_t stream,
  const void* weights,
  const void* input,
  void* output,
  void* workspace,
  size_t workspaceBytes,
  int matBatchSize,
  int inChannels,
  int outChannels,
  bool usingFP16
);

typedef bool (*Sm120QKVStridedFn)(
  void* ctx,
  cublasHandle_t cublas,
  cudaStream_t stream,
  const void* qWeights,
  const void* kWeights,
  const void* vWeights,
  const void* inputBuf,
  void* qkvBuf,
  int matBatchSize,
  int numChannels,
  int qDim,
  int kDim,
  int vDim,
  bool usingFP16
);

typedef bool (*Sm120FusedResidualGemmFn)(
  void* ctx,
  cublasHandle_t cublas,
  cudaStream_t stream,
  const void* weights,
  const void* inputBuf,
  void* trunkBuf,
  const void* maskBuf,
  int matBatchSize,
  int inputChannels,
  int outputChannels,
  bool usingFP16
);

typedef bool (*Sm120RMSNormFn)(
  void* ctx,
  const void* inputBuf,
  void* outputBuf,
  const void* gammaBuf,
  const void* betaBuf,
  const void* maskBuf,
  int batchSize,
  int xySize,
  int channels,
  float epsilon,
  bool usingFP16,
  cudaStream_t stream
);

typedef bool (*Sm120FusedQKRoPEFn)(
  void* ctx,
  void* qBuf,
  void* kBuf,
  const float* freqs,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int numPairs,
  int nnXLen,
  bool usingFP16,
  cudaStream_t stream
);

typedef bool (*Sm120SwiGLUFn)(
  void* ctx,
  const void* a,
  const void* b,
  void* output,
  int numTokens,
  int channels,
  bool usingFP16,
  cudaStream_t stream
);

typedef bool (*Sm120AffineSiluFn)(
  void* ctx,
  const void* input,
  void* output,
  const void* scale,
  const void* bias,
  const void* mask,
  int batchSize,
  int xySize,
  int channels,
  int activation,
  bool usingFP16,
  cudaStream_t stream
);

typedef bool (*Sm120FusedPolicyP1Fn)(
  void* ctx,
  const void* input,
  float* output,
  const float* globalBias,
  const float* scale,
  const float* bias,
  int batchSize,
  int xySize,
  int channels,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
);

// Installs a stream access-policy window when basePtr is non-null and clears
// it when basePtr is null. The official backend owns the scratch-buffer
// lifetime; SM120 owns only the cache policy.
typedef void (*Sm120PersistingL2Fn)(
  void* ctx,
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes
);

struct Options {
  // Master switch. When false, SM120 keeps the official backend path entirely (A/B control).
  bool enabled = true;

  // Historical optimization switches, defaults = final accepted values from the 5080 history.
  bool useFlashAttention = true;
  std::string flashAttentionAccum = "both16"; // "none","fp32","qk16","pv16","both16"
  bool useWideQKV = true;
  bool useWideQKVSingleStreamSchedule = false;
  bool useQKVStrided = false;
  bool useQKVGemmAot = true;
  bool useQKVGemmRopeAot = false;
  bool useFusedQKRoPE = true;
  bool useFusedQKRoPEHalf2 = false;
  bool useBatchSharedRoPE = false;
  bool useBatchSharedRoPEUnroll19 = true;
  bool useBatchSharedRoPETwoWay = false;
  bool useFusedResidual = true;
  bool useFusedResidualGemm = true;
  bool useProjectionGemmLt = false;
  bool useLinear2ResidualAot = true;
  bool useLinear2ResidualAotBalanced = false;
  bool useOutProjectionResidualAot = false;
  bool useFusedFFN = true;
  bool useFusedFFNAReuse = false;
  bool useFusedFFNSingleStreamSchedule = false;
  bool useWideFFNSingleGemm = false;
  bool useFusedRMSNormFFN = false;
  bool useRMSNorm384 = true;
  bool useRMSNorm384Vec8 = false;
  bool useRMSNorm384TwoWarp = false;
  bool useSwiGLU1152 = true;
  bool useAffineSiluHalf2 = true;
  bool useRMSNormQKVGemmAot = false;
  bool useGraph = false;
  bool usePersistingL2Trunk = false;
  bool usePersistingL2Inner = false;
  bool useOuterProjectionAot = true;
  bool shareModelWeights = true;
  bool shareWideQKVWeights = false;
  bool shareOuterProjectionWeights = false;
  bool useInitialConvFrontend = true;
  bool useInitialConvBiasFrontend = false;
  bool useInitialGlobalMatMulAdd = true;
  bool useFusedPolicyP1 = false;
  bool useHeadBNHalfToFloat = true;
  bool useWideHeadProjection = true;
};

bool isSm120Arch(int majorComputeCapability, int minorComputeCapability);

// Reads all cuda*Sm120* / cuda* config keys relevant to the SM120 path. Unknown accum values throw.
Options parseOptions(ConfigParser& cfg);

// Attention hook implementation (FA4 AOT on SM120, see fa4_aot/).
bool applyAttention(
  void* ctx,
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  void* qBuf,
  void* kBuf,
  void* vBuf,
  void* maskBuf,
  void* attnOutBuf,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFP16,
  cudaStream_t stream,
  void* workspaceBuf,
  size_t workspaceBytes
);

bool applyFFNSingleGemm(
  void* ctx,
  cublasHandle_t cublas,
  cudaStream_t stream,
  const void* linear1Weights,
  const void* linearGateWeights,
  const void* inputBuf,
  void* wideScratchBuf,
  void* ffnOutBuf,
  int matBatchSize,
  int numChannels,
  int ffnChannels,
  bool usingFP16
);

bool applyMatMulLt(
  void* ctx,
  cudaStream_t stream,
  const void* weights,
  const void* input,
  void* output,
  void* workspace,
  size_t workspaceBytes,
  int matBatchSize,
  int inChannels,
  int outChannels,
  bool usingFP16
);

bool applyQKVStrided(
  void* ctx,
  cublasHandle_t cublas,
  cudaStream_t stream,
  const void* qWeights,
  const void* kWeights,
  const void* vWeights,
  const void* inputBuf,
  void* qkvBuf,
  int matBatchSize,
  int numChannels,
  int qDim,
  int kDim,
  int vDim,
  bool usingFP16
);

bool applyFusedResidualGemm(
  void* ctx,
  cublasHandle_t cublas,
  cudaStream_t stream,
  const void* weights,
  const void* inputBuf,
  void* trunkBuf,
  const void* maskBuf,
  int matBatchSize,
  int inputChannels,
  int outputChannels,
  bool usingFP16
);

bool applyRMSNorm(
  void* ctx,
  const void* inputBuf,
  void* outputBuf,
  const void* gammaBuf,
  const void* betaBuf,
  const void* maskBuf,
  int batchSize,
  int xySize,
  int channels,
  float epsilon,
  bool usingFP16,
  cudaStream_t stream
);

bool applyFusedQKRoPE(
  void* ctx,
  void* qBuf,
  void* kBuf,
  const float* freqs,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int numPairs,
  int nnXLen,
  bool usingFP16,
  cudaStream_t stream
);

bool applySwiGLU(
  void* ctx,
  const void* a,
  const void* b,
  void* output,
  int numTokens,
  int channels,
  bool usingFP16,
  cudaStream_t stream
);

bool applyAffineSilu(
  void* ctx,
  const void* input,
  void* output,
  const void* scale,
  const void* bias,
  const void* mask,
  int batchSize,
  int xySize,
  int channels,
  int activation,
  bool usingFP16,
  cudaStream_t stream
);

bool applyFusedPolicyP1(
  void* ctx,
  const void* input,
  float* output,
  const float* globalBias,
  const float* scale,
  const float* bias,
  int batchSize,
  int xySize,
  int channels,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
);

void applyPersistingL2Window(
  void* ctx,
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes
);

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
  bool hasPersistingL2Trunk() const;
  bool hasPersistingL2Inner() const;

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

  // FA4 AOT attention dispatch; called through the Sm120AttentionFn hook.
  bool attention(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    void* qBuf,
    void* kBuf,
    void* vBuf,
    void* maskBuf,
    void* attnOutBuf,
    int batchSize,
    int seqLen,
    int numHeads,
    int numKVHeads,
    int qHeadDim,
    int vHeadDim,
    bool usingFP16,
    cudaStream_t stream,
    void* workspaceBuf,
    size_t workspaceBytes
  );

  bool ffnSingleGemm(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const void* linear1Weights,
    const void* linearGateWeights,
    const void* inputBuf,
    void* wideScratchBuf,
    void* ffnOutBuf,
    int matBatchSize,
    int numChannels,
    int ffnChannels,
    bool usingFP16
  );

  bool matMulLt(
    cudaStream_t stream,
    const void* weights,
    const void* input,
    void* output,
    void* workspace,
    size_t workspaceBytes,
    int matBatchSize,
    int inChannels,
    int outChannels,
    bool usingFP16
  );

  bool qkvStrided(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const void* qWeights,
    const void* kWeights,
    const void* vWeights,
    const void* inputBuf,
    void* qkvBuf,
    int matBatchSize,
    int numChannels,
    int qDim,
    int kDim,
    int vDim,
    bool usingFP16
  );

  bool fusedResidualGemm(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const void* weights,
    const void* inputBuf,
    void* trunkBuf,
    const void* maskBuf,
    int matBatchSize,
    int inputChannels,
    int outputChannels,
    bool usingFP16
  );

  bool rmsNorm(
    const void* inputBuf,
    void* outputBuf,
    const void* gammaBuf,
    const void* betaBuf,
    const void* maskBuf,
    int batchSize,
    int xySize,
    int channels,
    float epsilon,
    bool usingFP16,
    cudaStream_t stream
  );

  bool fusedQKRoPE(
    void* qBuf,
    void* kBuf,
    const float* freqs,
    int batchSize,
    int seqLen,
    int numHeads,
    int numKVHeads,
    int qHeadDim,
    int numPairs,
    int ropeXLen,
    bool usingFP16,
    cudaStream_t stream
  );

  bool swiGLU(
    const void* a,
    const void* b,
    void* output,
    int numTokens,
    int channels,
    bool usingFP16,
    cudaStream_t stream
  );

  bool affineSilu(
    const void* input,
    void* output,
    const void* scale,
    const void* bias,
    const void* mask,
    int batchSize,
    int xySize,
    int channels,
    int activation,
    bool usingFP16,
    cudaStream_t stream
  );

  bool fusedPolicyP1(
    const void* input,
    float* output,
    const float* globalBias,
    const float* scale,
    const float* bias,
    int batchSize,
    int xySize,
    int channels,
    bool usingFP16,
    bool usingNHWC,
    cudaStream_t stream
  );

  void persistingL2Window(
    cudaStream_t stream,
    void* basePtr,
    size_t numBytes
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
  bool loggedFa4;
  bool loggedFusedFFN;
  bool loggedProjectionGemmLt;
  bool loggedWideFFNSingleGemm;
  bool loggedWideQKV;
  bool loggedQKVStrided;
  bool loggedFusedResidualGemm;
  bool loggedRMSNorm384;
  bool loggedFusedQKRoPE;
  bool loggedBatchSharedQKRoPE;
  bool loggedFusedQKRoPEHalf2;
  bool loggedSwiGLU1152;
  bool loggedAffineSiluHalf2;
  bool loggedFusedPolicyP1;
  bool loggedPersistingL2Trunk;
  bool loggedPersistingL2Inner;
  bool persistingL2TrunkActive;
  bool persistingL2InnerActive;
  size_t persistingL2TrunkWindowBytes;
  size_t persistingL2InnerWindowBytes;
  size_t persistingL2RequestedBytes;
  size_t persistingL2ActualBytes;
  float persistingL2TrunkHitRatio;
  float persistingL2InnerHitRatio;
  std::unordered_map<const void*, void*> wideFFNSingleGemmWeights;
  std::unordered_map<const void*, void*> wideQKVWeights;
  std::unordered_map<const void*, void*> qkvStridedWeights;
  struct LtMatmulState;
  std::unique_ptr<LtMatmulState> ltMatmulState;

  // TODO(rebuild): device weight buffers, AOT kernel handles, per-GPU shared-weight caches,
  // scratch/workspace plan.
};

} // namespace Sm120Backend

#endif
