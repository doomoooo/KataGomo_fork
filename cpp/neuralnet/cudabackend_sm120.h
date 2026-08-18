#ifndef KATAGO_CUDA_BACKEND_SM120_H
#define KATAGO_CUDA_BACKEND_SM120_H

#include "../neuralnet/cudaincludes.h"
#include "../core/config_parser.h"
#include "../neuralnet/desc.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// SM120-specific CUDA backend.
//
// All Blackwell-SM120 kernels, AOT handles, weight sharing, persisting-L2 windows and config
// switches live here (cudabackend_sm120.h/cpp). The official backend files (cudabackend.cpp,
// cudahelpers.cu, cudautils.cpp, ...) are NOT modified with SM120 branches; they only contain a
// thin dispatch: ComputeHandle builds a Sm120Model on SM120 and routes apply() through it.
//
// Sm120Model deliberately delegates the outer forward traversal to the
// official Model. Operator hooks installed on CudaHandles replace the selected
// boundaries with SM120 implementations. This keeps one model traversal and
// one buffer layout while the exact-batch plan controls each CUDA tactic.

struct CudaHandles;    // defined in cudabackend.cpp
struct ScratchBuffers; // defined in cudabackend.cpp
struct Logger;         // defined in core/logger.h

namespace Sm120Backend {

struct FusedFFNAotTactic;
struct WideQKVAotTactic;
struct WideQKVRopeAotTactic;
struct ResidualGemmAotTactic;
struct FA4AotTactic;

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
// block (thin dispatch only) with the already-computed Q/K/V buffers. Planar
// buffers are contiguous BSHD; packed buffers use dynamic BSHD strides.
// Returns true if the attention output was produced and the official
// SDPA/custom path must be skipped; false means fall back to the official path.
typedef bool (*Sm120AttentionFn)(
  void* ctx,
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  void* qBuf,
  void* kBuf,
  void* vBuf,
  bool packedQKV,
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

typedef bool (*Sm120Conv1x1Fn)(
  void* ctx,
  const void* weights,
  const void* input,
  void* output,
  int matBatchSize,
  int inChannels,
  int outChannels,
  bool accumulate,
  bool usingFP16,
  cudaStream_t stream
);

typedef bool (*Sm120InitialGlobalFn)(
  void* ctx,
  void* spatialBuf,
  const void* globalInput,
  const void* weights,
  int batchSize,
  int inputChannels,
  int outputChannels,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
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
  bool allowPackedOutput,
  bool* packedOutput,
  const float* ropeFreqs,
  bool* ropeApplied,
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
  bool packedQKV,
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

typedef bool (*Sm120PostConvBNSiluFn)(
  void* ctx,
  const void* input,
  const void* weights,
  void* residual,
  void* activated,
  const void* scale,
  const void* bias,
  const void* mask,
  int batchSize,
  int xySize,
  int inputChannels,
  int outputChannels,
  int activation,
  bool usingFP16,
  bool usingNHWC,
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
  int inputStride,
  int inputOffset,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
);

typedef bool (*Sm120WideHeadProjectionFn)(
  void* ctx,
  const void* input,
  void* output,
  int batchSize,
  int xySize,
  int inputChannels,
  int outputChannels,
  bool usingFP16,
  bool usingNHWC,
  int* outputRowStride,
  int* p1Offset,
  int* g1Offset,
  int* v1Offset,
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

  // Internal discriminator set only by makeSm103PortableOptions(). It is not
  // parsed from configuration. It changes diagnostics and documents that this
  // owner may launch generic CUDA/library tactics only; all architecture-bound
  // choices have already been rejected and cleared.
  bool portableSm103Adapter = false;

  // The plan owns every optimization switch. Defaults are an explicit
  // official-equivalent control state, never an implicit historical winner.
  bool useFlashAttention = false;
  std::string flashAttentionAccum = "none"; // "none","fp32","qk16","pv16","both16"
  std::string fa4AotTactic = "disabled";
  bool fa4AotTacticExplicit = false;
  bool useWideQKV = false;
  bool useQKVStrided = false;
  bool useQKVGemmAot = false;
  std::string qkvRopeAotTactic = "disabled";
  bool qkvRopeAotTacticExplicit = false;
  bool useFusedQKRoPE = false;
  bool useFusedQKRoPEHalf2 = false;
  bool useBatchSharedRoPE = false;
  bool useBatchSharedRoPEUnrolled = false;
  bool useFusedResidual = false;
  bool useFusedResidualGemm = false;
  // Generic shape-autotuned beta=0 MatMul hook. Native SM120 preserves its
  // historical fallback behavior. The SM103 portable B29 owner narrows this
  // to the audited QKV and initial-global shapes and fails closed.
  bool useProjectionGemmLt = false;
  // SM103-only selector: autotune is discovery; explicit B29 choices pin the
  // complete QKV algorithm tuple and leave initial-global on legacy Hgemm.
  std::string projectionGemmLtTacticSm103 = "autotune";
  bool useLinear2ResidualAot = false;
  bool useOutProjectionResidualAot = false;
  std::string outProjectionAotTactic = "disabled";
  bool outProjectionAotTacticExplicit = false;
  bool useFusedFFN = false;
  std::string fusedFFNAotTactic = "disabled";
  bool fusedFFNAotTacticExplicit = false;
  bool useWideFFNSingleGemm = false;
  // disabled, ordered-ept3, one-warp-exact, or warp4-vec8.
  std::string rmsNorm384Tactic = "disabled";
  bool useSwiGLU1152 = false;
  // disabled, half2, half2x3, or flat-vec8-c768.
  std::string affineSiluTactic = "disabled";
  bool usePersistingL2Trunk = false;
  bool usePersistingL2Inner = false;
  int persistingL2Streams = 2;
  double persistingL2HitRatio = 1.0;
  // disabled, warp64x64, or warp64x32.
  std::string outerProjectionDownTactic = "disabled";
  std::string outerProjectionUpTactic = "disabled";
  bool usePostConvBNSilu = false;
  std::string wideQKVAotTactic = "disabled";
  bool wideQKVAotTacticExplicit = false;
  std::string linear2AotTactic = "disabled";
  bool linear2AotTacticExplicit = false;
  bool shareModelWeights = false;
  std::string initialConvFrontendPlan = "disabled";
  bool useInitialGlobalMatMulAdd = false;
  bool useFusedPolicyP1 = false;
  bool useHeadBNHalfToFloat = false;
  // disabled, full-c384, or partial-c288-g1-v1.
  std::string wideHeadProjectionTactic = "disabled";
  bool useFusedValueTerminal = false;
};

bool isSm120Arch(int majorComputeCapability, int minorComputeCapability);

// Retained head boundary used directly by the official forward adapter.
bool launchHeadBNHalfToFloat(
  const half* input,
  half* halfOutput,
  float* floatOutput,
  const half* scale,
  const half* bias,
  int batchSize,
  int xySize,
  int channels,
  int inputStride,
  int inputOffset,
  cudaStream_t stream
);

bool launchSplitValueTerminal(
  const float* combined,
  const float* bias,
  float* value,
  float* scoreValue,
  int batchSize,
  int valueChannels,
  int scoreValueChannels,
  cudaStream_t stream
);

// Reads all cuda*Sm120* / cuda* config keys relevant to the SM120 path. Unknown accum values throw.
Options parseOptions(ConfigParser& cfg);

// Validate an SM120 option set for reuse by the explicit CC10.3 portable
// adapter. Any FA/AOT/CUTLASS or engine-ID selection is a startup error; the
// returned copy is defensively scrubbed as well. Generic CUDA, cuBLAS,
// cuBLASLt and runtime-only choices are preserved.
Options makeSm103PortableOptions(const Options& requested);

// Attention hook implementation (FA4 AOT on SM120, see fa4_aot/).
bool applyAttention(
  void* ctx,
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  void* qBuf,
  void* kBuf,
  void* vBuf,
  bool packedQKV,
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

bool applyConv1x1(
  void* ctx,
  const void* weights,
  const void* input,
  void* output,
  int matBatchSize,
  int inChannels,
  int outChannels,
  bool accumulate,
  bool usingFP16,
  cudaStream_t stream
);

bool applyInitialGlobal(
  void* ctx,
  void* spatialBuf,
  const void* globalInput,
  const void* weights,
  int batchSize,
  int inputChannels,
  int outputChannels,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
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
  bool allowPackedOutput,
  bool* packedOutput,
  const float* ropeFreqs,
  bool* ropeApplied,
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
  bool packedQKV,
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

bool applyPostConvBNSilu(
  void* ctx,
  const void* input,
  const void* weights,
  void* residual,
  void* activated,
  const void* scale,
  const void* bias,
  const void* mask,
  int batchSize,
  int xySize,
  int inputChannels,
  int outputChannels,
  int activation,
  bool usingFP16,
  bool usingNHWC,
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
  int inputStride,
  int inputOffset,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
);

bool applyWideHeadProjection(
  void* ctx,
  const void* input,
  void* output,
  int batchSize,
  int xySize,
  int inputChannels,
  int outputChannels,
  bool usingFP16,
  bool usingNHWC,
  int* outputRowStride,
  int* p1Offset,
  int* g1Offset,
  int* v1Offset,
  cudaStream_t stream
);

void applyPersistingL2Window(
  void* ctx,
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes
);

// SM120 operator owner and official-forward adapter. Plan-explicit AOT routes
// are strict: they must execute or throw rather than silently fall back.
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
  float* getFullBoardAreaBuf() const;

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
    bool packedQKV,
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

  bool conv1x1(
    const void* weights,
    const void* input,
    void* output,
    int matBatchSize,
    int inChannels,
    int outChannels,
    bool accumulate,
    bool usingFP16,
    cudaStream_t stream
  );

  bool initialGlobal(
    void* spatialBuf,
    const void* globalInput,
    const void* weights,
    int batchSize,
    int inputChannels,
    int outputChannels,
    bool usingFP16,
    bool usingNHWC,
    cudaStream_t stream
  );

  bool qkvStrided(
    cublasHandle_t cublas,
    cudaStream_t stream,
    const void* qWeights,
    const void* kWeights,
    const void* vWeights,
    const void* inputBuf,
    void* qkvBuf,
    bool allowPackedOutput,
    bool* packedOutput,
    const float* ropeFreqs,
    bool* ropeApplied,
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
    bool packedQKV,
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

  bool postConvBNSilu(
    const void* input,
    const void* weights,
    void* residual,
    void* activated,
    const void* scale,
    const void* bias,
    const void* mask,
    int batchSize,
    int xySize,
    int inputChannels,
    int outputChannels,
    int activation,
    bool usingFP16,
    bool usingNHWC,
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
    int inputStride,
    int inputOffset,
    bool usingFP16,
    bool usingNHWC,
    cudaStream_t stream
  );

  bool wideHeadProjection(
    const void* input,
    void* output,
    int batchSize,
    int xySize,
    int inputChannels,
    int outputChannels,
    bool usingFP16,
    bool usingNHWC,
    int* outputRowStride,
    int* p1Offset,
    int* g1Offset,
    int* v1Offset,
    cudaStream_t stream
  );

  void persistingL2Window(
    cudaStream_t stream,
    void* basePtr,
    size_t numBytes
  );

 private:
  std::string logMessage(const std::string& message) const;

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
  int sm120NumSms;
  Logger* logger;
  float* fullBoardAreaBuf;
  bool loggedFallback;
  bool loggedFa4;
  bool loggedFa4AtMaxBatch;
  bool loggedFusedFFN;
  bool loggedProjectionGemmLt;
  bool loggedOuterProjectionDown;
  bool loggedOuterProjectionUp;
  bool loggedInitialGlobal;
  bool loggedWideFFNSingleGemm;
  bool loggedWideQKV;
  bool loggedQKVStrided;
  bool loggedQKVRopeAot;
  bool loggedLinear2Aot;
  bool loggedOutProjectionAot;
  bool loggedFusedResidualGemm;
  bool loggedRMSNorm384;
  bool loggedFusedQKRoPE;
  bool loggedBatchSharedQKRoPE;
  bool loggedBatchSharedQKRoPEAtMaxBatch;
  bool loggedFusedQKRoPEHalf2;
  bool loggedSwiGLU1152;
  bool loggedAffineSiluHalf2;
  bool loggedPostConvBNSilu;
  bool loggedFusedPolicyP1;
  bool loggedWideHeadProjection;
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
  std::vector<const FusedFFNAotTactic*> fusedFFNAotByBatch;
  std::vector<const FA4AotTactic*> fa4AotByBatch;
  std::vector<const WideQKVAotTactic*> wideQKVAotByBatch;
  std::vector<const WideQKVRopeAotTactic*> wideQKVRopeAotByBatch;
  std::vector<const ResidualGemmAotTactic*> linear2AotByBatch;
  std::vector<const ResidualGemmAotTactic*> outProjectionAotByBatch;
  std::unordered_map<const void*, void*> wideFFNSingleGemmWeights;
  std::unordered_map<const void*, void*> fusedFFNPairedWeights;
  std::unordered_map<const void*, void*> wideQKVWeights;
  std::unordered_map<const float*, void*> qkvRopeTables;
  std::unordered_map<const void*, void*> qkvStridedWeights;
  std::unordered_map<const void*, void*> outerProjectionDownHandles;
  std::unordered_map<const void*, void*> outerProjectionUpHandles;
  void* wideHeadProjectionWeights;
  void* wideHeadProjectionHandle;
  void* dualFfnSharedAHandle;
  int wideHeadProjectionChannels;
  int wideHeadP1Offset;
  int wideHeadG1Offset;
  int wideHeadV1Offset;
  struct LtMatmulState;
  std::unique_ptr<LtMatmulState> ltMatmulState;

};

} // namespace Sm120Backend

#endif
