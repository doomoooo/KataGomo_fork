#include "../neuralnet/cudabackend_sm120.h"
#include "../neuralnet/cudabackend_sm120_kernels.h"

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudautils.h"
#include "../neuralnet/activations.h"
#include "sm120_aot/outer_projection.h"
#include "sm120_aot/dual_ffn_shared_a.h"

#include "../core/global.h"
#include "../core/logger.h"

#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

using namespace std;

namespace Sm120Backend {

static void setPersistingL2Window(
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes,
  float hitRatio
) {
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = basePtr;
  attr.accessPolicyWindow.num_bytes = numBytes;
  attr.accessPolicyWindow.hitRatio = hitRatio;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CUDA_ERR("Sm120PersistingL2", cudaStreamSetAttribute(
    stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

static void clearPersistingL2Window(cudaStream_t stream) {
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CUDA_ERR("Sm120PersistingL2", cudaStreamSetAttribute(
    stream, cudaStreamAttributeAccessPolicyWindow, &attr));
}

struct Sm120Model::LtMatmulState {
  struct Plan {
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t aDesc;
    cublasLtMatrixLayout_t bDesc;
    cublasLtMatrixLayout_t cDesc;
    cublasLtMatmulAlgo_t algo;
    size_t workspaceBytes;
    bool valid;

    Plan()
      : operationDesc(NULL), aDesc(NULL), bDesc(NULL), cDesc(NULL),
        workspaceBytes(0), valid(false) {}

    ~Plan() {
      if(cDesc != NULL)
        cublasLtMatrixLayoutDestroy(cDesc);
      if(bDesc != NULL)
        cublasLtMatrixLayoutDestroy(bDesc);
      if(aDesc != NULL)
        cublasLtMatrixLayoutDestroy(aDesc);
      if(operationDesc != NULL)
        cublasLtMatmulDescDestroy(operationDesc);
    }
  };

  cublasLtHandle_t handle;
  void* workspace;
  unordered_map<uint64_t,unique_ptr<Plan>> plans;

  static size_t workspaceCapacity() {
    return 64ULL * 1024ULL * 1024ULL;
  }

  LtMatmulState() : handle(NULL), workspace(NULL) {
    CUBLAS_ERR("Sm120Model cuBLASLt create", cublasLtCreate(&handle));
    CUDA_ERR("Sm120Model cuBLASLt workspace", cudaMalloc(&workspace, workspaceCapacity()));
  }

  ~LtMatmulState() {
    plans.clear();
    if(workspace != NULL)
      cudaFree(workspace);
    if(handle != NULL)
      cublasLtDestroy(handle);
  }

  static uint64_t planKey(int m, int n, int k) {
    return (static_cast<uint64_t>(m) << 42) |
           (static_cast<uint64_t>(n) << 21) |
           static_cast<uint64_t>(k);
  }

  Plan* getOrCreatePlan(
    int m,
    int n,
    int k,
    const void* a,
    const void* b,
    void* c,
    cudaStream_t stream
  ) {
    const uint64_t key = planKey(m,n,k);
    const auto existing = plans.find(key);
    if(existing != plans.end())
      return existing->second.get();

    unique_ptr<Plan> plan = make_unique<Plan>();
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F);
    if(status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatrixLayoutCreate(&plan->aDesc, CUDA_R_16F, m, k, m);
    if(status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatrixLayoutCreate(&plan->bDesc, CUDA_R_16F, k, n, k);
    if(status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatrixLayoutCreate(&plan->cDesc, CUDA_R_16F, m, n, m);

    cublasLtMatmulPreference_t preference = NULL;
    if(status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatmulPreferenceCreate(&preference);
    const uint64_t maxWorkspaceBytes = workspaceCapacity();
    if(status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maxWorkspaceBytes,
        sizeof(maxWorkspaceBytes)
      );

    const int requestedAlgoCount = 16;
    vector<cublasLtMatmulHeuristicResult_t> heuristics(requestedAlgoCount);
    int returnedAlgoCount = 0;
    if(status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        plan->operationDesc,
        plan->aDesc,
        plan->bDesc,
        plan->cDesc,
        plan->cDesc,
        preference,
        requestedAlgoCount,
        heuristics.data(),
        &returnedAlgoCount
      );
    if(preference != NULL)
      cublasLtMatmulPreferenceDestroy(preference);

    if(status == CUBLAS_STATUS_SUCCESS && returnedAlgoCount > 0) {
      cudaEvent_t start = NULL;
      cudaEvent_t stop = NULL;
      CUDA_ERR("Sm120Model cuBLASLt tune start event", cudaEventCreate(&start));
      CUDA_ERR("Sm120Model cuBLASLt tune stop event", cudaEventCreate(&stop));

      float bestUs = numeric_limits<float>::infinity();
      const int timingIterations = 8;
      for(int i = 0; i < returnedAlgoCount; i++) {
        if(heuristics[i].state != CUBLAS_STATUS_SUCCESS ||
           heuristics[i].workspaceSize > workspaceCapacity())
          continue;

        status = cublasLtMatmul(
          handle,
          plan->operationDesc,
          &alpha,
          a,
          plan->aDesc,
          b,
          plan->bDesc,
          &beta,
          c,
          plan->cDesc,
          c,
          plan->cDesc,
          &heuristics[i].algo,
          workspace,
          heuristics[i].workspaceSize,
          stream
        );
        if(status != CUBLAS_STATUS_SUCCESS)
          continue;

        CUDA_ERR("Sm120Model cuBLASLt tune record start", cudaEventRecord(start,stream));
        bool launchSucceeded = true;
        for(int iteration = 0; iteration < timingIterations; iteration++) {
          status = cublasLtMatmul(
            handle,
            plan->operationDesc,
            &alpha,
            a,
            plan->aDesc,
            b,
            plan->bDesc,
            &beta,
            c,
            plan->cDesc,
            c,
            plan->cDesc,
            &heuristics[i].algo,
            workspace,
            heuristics[i].workspaceSize,
            stream
          );
          if(status != CUBLAS_STATUS_SUCCESS) {
            launchSucceeded = false;
            break;
          }
        }
        if(!launchSucceeded)
          continue;
        CUDA_ERR("Sm120Model cuBLASLt tune record stop", cudaEventRecord(stop,stream));
        CUDA_ERR("Sm120Model cuBLASLt tune sync", cudaEventSynchronize(stop));
        float elapsedMs = 0.0f;
        CUDA_ERR("Sm120Model cuBLASLt tune elapsed", cudaEventElapsedTime(&elapsedMs,start,stop));
        const float averageUs = elapsedMs * 1000.0f / timingIterations;
        if(averageUs < bestUs) {
          bestUs = averageUs;
          plan->algo = heuristics[i].algo;
          plan->workspaceBytes = heuristics[i].workspaceSize;
          plan->valid = true;
        }
      }

      cudaEventDestroy(stop);
      cudaEventDestroy(start);
    }

    Plan* result = plan.get();
    plans.emplace(key,move(plan));
    return result;
  }
};

bool isSm120Arch(int majorComputeCapability, int minorComputeCapability) {
  return majorComputeCapability == 12 && minorComputeCapability == 0;
}

static bool getBoolOpt(ConfigParser& cfg, const string& key, bool defaultValue) {
  return cfg.contains(key) ? cfg.getBool(key) : defaultValue;
}

Options parseOptions(ConfigParser& cfg) {
  Options o;
  o.enabled = getBoolOpt(cfg, "cudaSm120Backend", true);
  o.useFlashAttention = getBoolOpt(cfg, "cudaUseFlashAttentionSm120", false);
  o.fa4AotTacticExplicit = cfg.contains("cudaFlashAttentionAotTacticSm120");
  o.fa4AotTactic = cfg.contains("cudaFlashAttentionAotTacticSm120") ?
    cfg.getString("cudaFlashAttentionAotTacticSm120") : "disabled";
  if(cfg.contains("cudaFlashAttentionSm120Accum"))
    o.flashAttentionAccum = cfg.getString("cudaFlashAttentionSm120Accum");
  if(o.flashAttentionAccum != "none" && o.flashAttentionAccum != "fp32" &&
     o.flashAttentionAccum != "qk16" && o.flashAttentionAccum != "pv16" &&
     o.flashAttentionAccum != "both16")
    throw StringError("cudaFlashAttentionSm120Accum must be one of none/fp32/qk16/pv16/both16");
  o.useWideQKV = getBoolOpt(cfg, "cudaUseWideQKV", false);
  o.useQKVStrided = getBoolOpt(cfg, "cudaUseQKVStridedSm120", false);
  o.useQKVGemmAot = getBoolOpt(cfg, "cudaUseQKVGemmAot", false);
  o.qkvRopeAotTacticExplicit = cfg.contains("cudaQKVRopeAotTacticSm120");
  o.qkvRopeAotTactic = cfg.contains("cudaQKVRopeAotTacticSm120") ?
    cfg.getString("cudaQKVRopeAotTacticSm120") : "disabled";
  o.useFusedQKRoPE = getBoolOpt(cfg, "cudaUseFusedQKRoPE", false);
  o.useFusedQKRoPEHalf2 = getBoolOpt(cfg, "cudaUseFusedQKRoPEHalf2Sm120", false);
  o.useBatchSharedRoPE = getBoolOpt(cfg, "cudaUseBatchSharedRoPE", false);
  o.useBatchSharedRoPEUnrolled =
    getBoolOpt(cfg, "cudaUseBatchSharedRoPEUnrolledSm120", false);
  o.useFusedResidual = getBoolOpt(cfg, "cudaUseFusedResidual", false);
  o.useFusedResidualGemm = getBoolOpt(cfg, "cudaUseFusedResidualGemmSm120", false);
  o.useProjectionGemmLt = getBoolOpt(cfg, "cudaUseProjectionGemmLt", false);
  o.useLinear2ResidualAot = getBoolOpt(cfg, "cudaUseLinear2ResidualAot", false);
  o.useOutProjectionResidualAot = getBoolOpt(cfg, "cudaUseOutProjectionResidualAot", false);
  o.outProjectionAotTacticExplicit = cfg.contains("cudaOutProjectionAotTacticSm120");
  o.outProjectionAotTactic = cfg.contains("cudaOutProjectionAotTacticSm120") ?
    cfg.getString("cudaOutProjectionAotTacticSm120") :
    "disabled";
  o.useFusedFFN = getBoolOpt(cfg, "cudaUseFusedFFN", false);
  o.fusedFFNAotTacticExplicit = cfg.contains("cudaFusedFFNAotTacticSm120");
  o.fusedFFNAotTactic = cfg.contains("cudaFusedFFNAotTacticSm120") ?
    cfg.getString("cudaFusedFFNAotTacticSm120") :
    "disabled";
  o.useWideFFNSingleGemm = getBoolOpt(cfg, "cudaUseWideFFNSingleGemm", false);
  o.rmsNorm384Tactic = cfg.contains("cudaRMSNormTacticSm120") ?
    cfg.getString("cudaRMSNormTacticSm120") : "disabled";
  if(o.rmsNorm384Tactic != "disabled" &&
     o.rmsNorm384Tactic != "ordered-ept3" &&
     o.rmsNorm384Tactic != "one-warp-exact" &&
     o.rmsNorm384Tactic != "warp4-vec8")
    throw StringError(
      "cudaRMSNormTacticSm120 must be disabled, ordered-ept3, "
      "one-warp-exact, or warp4-vec8");
  o.useSwiGLU1152 = getBoolOpt(cfg, "cudaUseSwiGLU1152Sm120", false);
  o.affineSiluTactic = cfg.contains("cudaAffineSiluTacticSm120") ?
    cfg.getString("cudaAffineSiluTacticSm120") : "disabled";
  if(o.affineSiluTactic != "disabled" &&
     o.affineSiluTactic != "half2" &&
     o.affineSiluTactic != "half2x3" &&
     o.affineSiluTactic != "flat-vec8-c768")
    throw StringError(
      "cudaAffineSiluTacticSm120 must be disabled, half2, half2x3, "
      "or flat-vec8-c768");
  o.useExactMaskElision = getBoolOpt(cfg, "cudaUseExactMaskElisionSm120", false);
  o.usePersistingL2Trunk = getBoolOpt(cfg, "cudaUsePersistingL2Trunk", false);
  o.usePersistingL2Inner = getBoolOpt(cfg, "cudaUsePersistingL2Inner", false);
  o.persistingL2Streams = cfg.contains("cudaPersistingL2StreamsSm120") ?
    cfg.getInt("cudaPersistingL2StreamsSm120", 1, 16) : 2;
  o.persistingL2HitRatio = cfg.contains("cudaPersistingL2HitRatioSm120") ?
    cfg.getDouble("cudaPersistingL2HitRatioSm120", 0.0, 1.0) : 1.0;
  o.outerProjectionDownTactic = cfg.contains("cudaOuterProjectionDownTacticSm120") ?
    cfg.getString("cudaOuterProjectionDownTacticSm120") : "disabled";
  o.outerProjectionUpTactic = cfg.contains("cudaOuterProjectionUpTacticSm120") ?
    cfg.getString("cudaOuterProjectionUpTacticSm120") : "disabled";
  const auto checkOuterProjectionTactic = [](const string& key, const string& value) {
    if(value != "disabled" && value != "warp64x64" && value != "warp64x32")
      throw StringError(key + " must be disabled, warp64x64, or warp64x32");
  };
  checkOuterProjectionTactic(
    "cudaOuterProjectionDownTacticSm120", o.outerProjectionDownTactic);
  checkOuterProjectionTactic(
    "cudaOuterProjectionUpTacticSm120", o.outerProjectionUpTactic);
  o.usePostConvBNSilu = getBoolOpt(cfg, "cudaUsePostConvBNSiluSm120", false);
  o.wideQKVAotTacticExplicit = cfg.contains("cudaWideQKVAotTacticSm120");
  o.wideQKVAotTactic = cfg.contains("cudaWideQKVAotTacticSm120") ?
    cfg.getString("cudaWideQKVAotTacticSm120") :
    "disabled";
  o.linear2AotTacticExplicit = cfg.contains("cudaLinear2AotTacticSm120");
  o.linear2AotTactic = cfg.contains("cudaLinear2AotTacticSm120") ?
    cfg.getString("cudaLinear2AotTacticSm120") :
    "disabled";
  o.shareModelWeights = getBoolOpt(cfg, "cudaShareModelWeights", false);
  o.initialConvFrontendPlan = cfg.contains("cudaInitialConvFrontendPlanSm120") ?
    cfg.getString("cudaInitialConvFrontendPlanSm120") : "disabled";
  if(o.initialConvFrontendPlan != "disabled" &&
     o.initialConvFrontendPlan != "eng45-tile0-stages2" &&
     o.initialConvFrontendPlan != "eng47-k2-2-k6-1-k13-1-k14-0-k22-2")
    throw StringError(
      "cudaInitialConvFrontendPlanSm120 must be disabled, "
      "eng45-tile0-stages2, or eng47-k2-2-k6-1-k13-1-k14-0-k22-2");
  o.useInitialGlobalMatMulAdd = getBoolOpt(cfg, "cudaUseInitialGlobalMatMulAdd", false);
  o.useFusedPolicyP1 = getBoolOpt(cfg, "cudaUseFusedPolicyP1", false);
  o.useHeadBNHalfToFloat = getBoolOpt(cfg, "cudaUseHeadBNHalfToFloat", false);
  o.wideHeadProjectionTactic = cfg.contains("cudaWideHeadProjectionTacticSm120") ?
    cfg.getString("cudaWideHeadProjectionTacticSm120") : "disabled";
  if(o.wideHeadProjectionTactic != "disabled" &&
     o.wideHeadProjectionTactic != "full-c384" &&
     o.wideHeadProjectionTactic != "partial-c288-g1-v1")
    throw StringError(
      "cudaWideHeadProjectionTacticSm120 must be disabled, full-c384, "
      "or partial-c288-g1-v1");
  o.useFusedValueTerminal = getBoolOpt(
    cfg, "cudaUseFusedValueTerminalSm120", false);
  return o;
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->attention(
    cudaHandles, scratch, qBuf, kBuf, vBuf, packedQKV, maskBuf, attnOutBuf,
    batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim,
    usingFP16, stream, workspaceBuf, workspaceBytes
  );
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->ffnSingleGemm(
    cublas, stream, linear1Weights, linearGateWeights, inputBuf,
    wideScratchBuf, ffnOutBuf, matBatchSize, numChannels, ffnChannels, usingFP16);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->matMulLt(
    stream, weights, input, output, workspace, workspaceBytes,
    matBatchSize, inChannels, outChannels, usingFP16);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->conv1x1(
    weights, input, output, matBatchSize, inChannels, outChannels,
    accumulate, usingFP16, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->initialGlobal(
    spatialBuf, globalInput, weights, batchSize, inputChannels,
    outputChannels, usingFP16, usingNHWC, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->qkvStrided(
    cublas, stream, qWeights, kWeights, vWeights, inputBuf, qkvBuf,
    allowPackedOutput, packedOutput, ropeFreqs, ropeApplied,
    matBatchSize, numChannels, qDim, kDim, vDim, usingFP16);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->fusedResidualGemm(
    cublas, stream, weights, inputBuf, trunkBuf, maskBuf, matBatchSize,
    inputChannels, outputChannels, usingFP16);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->rmsNorm(
    inputBuf, outputBuf, gammaBuf, betaBuf, maskBuf, batchSize, xySize,
    channels, epsilon, usingFP16, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->fusedQKRoPE(
    qBuf, kBuf, freqs, batchSize, seqLen, numHeads, numKVHeads,
    qHeadDim, numPairs, nnXLen, packedQKV, usingFP16, stream);
}

bool applySwiGLU(
  void* ctx,
  const void* a,
  const void* b,
  void* output,
  int numTokens,
  int channels,
  bool usingFP16,
  cudaStream_t stream
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->swiGLU(
    a, b, output, numTokens, channels, usingFP16, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->affineSilu(
    input, output, scale, bias, mask, batchSize, xySize, channels,
    activation, usingFP16, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->postConvBNSilu(
    input, weights, residual, activated, scale, bias, mask,
    batchSize, xySize, inputChannels, outputChannels, activation,
    usingFP16, usingNHWC, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->fusedPolicyP1(
    input, output, globalBias, scale, bias, batchSize, xySize, channels,
    inputStride, inputOffset,
    usingFP16, usingNHWC, stream);
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->wideHeadProjection(
    input, output, batchSize, xySize, inputChannels, outputChannels,
    usingFP16, usingNHWC, outputRowStride, p1Offset, g1Offset, v1Offset,
    stream);
}

void applyPersistingL2Window(
  void* ctx,
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self != NULL)
    self->persistingL2Window(stream, basePtr, numBytes);
}

Sm120Model::Sm120Model(
  void* officialApplyContext_,
  OfficialApplyFn officialApply_,
  CudaHandles* cudaHandles_,
  const ModelDesc* desc_,
  int maxBatchSize_,
  int nnXLen_,
  int nnYLen_,
  bool inputsUseNHWC_,
  bool useFP16_,
  bool useNHWC_,
  const Options& options_
) :
  officialApplyContext(officialApplyContext_),
  officialApply(officialApply_),
  cudaHandles(cudaHandles_),
  desc(desc_),
  maxBatchSize(maxBatchSize_),
  nnXLen(nnXLen_),
  nnYLen(nnYLen_),
  inputsUseNHWC(inputsUseNHWC_),
  useFP16(useFP16_),
  useNHWC(useNHWC_),
  options(options_),
  sm120NumSms(0),
  logger(NULL),
  exactMaskSumBuf(NULL),
  loggedFallback(false),
  loggedFa4(false),
  loggedFa4AtMaxBatch(false),
  loggedFusedFFN(false),
  loggedProjectionGemmLt(false),
  loggedOuterProjectionDown(false),
  loggedOuterProjectionUp(false),
  loggedInitialGlobal(false),
  loggedWideFFNSingleGemm(false),
  loggedWideQKV(false),
  loggedQKVStrided(false),
  loggedQKVRopeAot(false),
  loggedLinear2Aot(false),
  loggedOutProjectionAot(false),
  loggedFusedResidualGemm(false),
  loggedRMSNorm384(false),
  loggedFusedQKRoPE(false),
  loggedBatchSharedQKRoPE(false),
  loggedBatchSharedQKRoPEAtMaxBatch(false),
  loggedFusedQKRoPEHalf2(false),
  loggedSwiGLU1152(false),
  loggedAffineSiluHalf2(false),
  loggedPostConvBNSilu(false),
  loggedFusedPolicyP1(false),
  loggedWideHeadProjection(false),
  loggedPersistingL2Trunk(false),
  loggedPersistingL2Inner(false),
  persistingL2TrunkActive(false),
  persistingL2InnerActive(false),
  persistingL2TrunkWindowBytes(0),
  persistingL2InnerWindowBytes(0),
  persistingL2RequestedBytes(0),
  persistingL2ActualBytes(0),
  persistingL2TrunkHitRatio(0.0f),
  persistingL2InnerHitRatio(0.0f)
{
  if(officialApplyContext == NULL || officialApply == NULL || cudaHandles == NULL || desc == NULL)
    throw StringError("Sm120Model: null construction argument");
  int device = 0;
  cudaDeviceProp deviceProp = {};
  CUDA_ERR("Sm120Model", cudaGetDevice(&device));
  CUDA_ERR("Sm120Model", cudaGetDeviceProperties(&deviceProp, device));
  sm120NumSms = deviceProp.multiProcessorCount;
  if(sm120NumSms <= 0)
    throw StringError("Sm120Model: CUDA reported no streaming multiprocessors");
  if(options.useExactMaskElision && nnXLen == 19 && nnYLen == 19 &&
     inputsUseNHWC && useFP16 && useNHWC && maxBatchSize > 0) {
    std::vector<float> fullBoardMaskSums(maxBatchSize, 361.0f);
    CUDA_ERR("Sm120ExactMaskSum", cudaMalloc(
      (void**)&exactMaskSumBuf, fullBoardMaskSums.size() * sizeof(float)));
    CUDA_ERR("Sm120ExactMaskSum", cudaMemcpy(
      exactMaskSumBuf, fullBoardMaskSums.data(),
      fullBoardMaskSums.size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  wideHeadProjectionWeights = NULL;
  wideHeadProjectionHandle = NULL;
  dualFfnSharedAHandle = NULL;
  if(options.fusedFFNAotTactic ==
     "dual_ffn-cutlass-shared-a-m128-n64-k32-s3-swizzle2") {
    dualFfnSharedAHandle = katago_create_dual_ffn_shared_a_sm120();
    if(dualFfnSharedAHandle == NULL)
      throw StringError("SM120 CUTLASS shared-A dual FFN handle creation failed");
  }
  wideHeadProjectionChannels = 0;
  wideHeadP1Offset = -1;
  wideHeadG1Offset = -1;
  wideHeadV1Offset = -1;
  if(options.wideHeadProjectionTactic != "disabled" &&
     options.useFusedPolicyP1 &&
     options.useHeadBNHalfToFloat && useFP16 && useNHWC &&
     nnXLen == 19 && nnYLen == 19) {
    const bool partial =
      options.wideHeadProjectionTactic == "partial-c288-g1-v1";
    const ConvLayerDesc* convs[3] = {
      partial ? NULL : &desc->policyHead.p1Conv,
      &desc->policyHead.g1Conv,
      &desc->valueHead.v1Conv,
    };
    const int offsets[3] = {partial ? -1 : 0, partial ? 0 : 96,
                            partial ? 96 : 192};
    const int expectedOutChannels[3] = {96, 96, 192};
    wideHeadProjectionChannels = partial ? 288 : 384;
    wideHeadP1Offset = offsets[0];
    wideHeadG1Offset = offsets[1];
    wideHeadV1Offset = offsets[2];
    bool compatible = true;
    for(int i = 0; i < 3; i++) {
      if(convs[i] == NULL)
        continue;
      const ConvLayerDesc& conv = *convs[i];
      compatible = compatible &&
        conv.convXSize == 1 && conv.convYSize == 1 &&
        conv.inChannels == 768 && conv.outChannels == expectedOutChannels[i];
    }
    if(compatible) {
      vector<float> weights((size_t)768 * wideHeadProjectionChannels);
      for(int i = 0; i < 3; i++) {
        if(convs[i] == NULL)
          continue;
        const ConvLayerDesc& conv = *convs[i];
        for(int inputChannel = 0; inputChannel < 768; inputChannel++) {
          for(int outputChannel = 0; outputChannel < conv.outChannels; outputChannel++) {
            weights[(size_t)inputChannel * wideHeadProjectionChannels +
                    offsets[i] + outputChannel] =
              conv.weights[(size_t)outputChannel * 768 + inputChannel];
          }
        }
      }
      CudaUtils::mallocAndCopyToDevice(
        "SM120 wide-head projection", weights, wideHeadProjectionWeights, true);
      wideHeadProjectionHandle = katago_create_head_projection_sm120(
        wideHeadProjectionWeights, wideHeadProjectionChannels, "warp64x64");
      if(wideHeadProjectionHandle == NULL)
        throw StringError("SM120 wide-head projection handle creation failed");
    }
  }
  fa4AotByBatch.resize(maxBatchSize + 1, nullptr);
  fusedFFNAotByBatch.resize(maxBatchSize + 1, nullptr);
  wideQKVAotByBatch.resize(maxBatchSize + 1, nullptr);
  wideQKVRopeAotByBatch.resize(maxBatchSize + 1, nullptr);
  linear2AotByBatch.resize(maxBatchSize + 1, nullptr);
  outProjectionAotByBatch.resize(maxBatchSize + 1, nullptr);
  for(int batch = 1; batch <= maxBatchSize; batch++) {
    fa4AotByBatch[batch] = findFA4AotTactic(
      batch, options.fa4AotTactic.c_str());
    if(options.useFusedFFN) {
      fusedFFNAotByBatch[batch] = findFusedFFNAotTactic(
        batch, sm120NumSms, options.persistingL2Streams,
        options.fusedFFNAotTactic.c_str());
    }
    if(options.useWideQKV && options.useQKVGemmAot) {
      wideQKVAotByBatch[batch] = findWideQKVAotTactic(
        batch, sm120NumSms, options.persistingL2Streams,
        options.wideQKVAotTactic.c_str());
    }
    if(options.qkvRopeAotTactic != "disabled") {
      wideQKVRopeAotByBatch[batch] = findWideQKVRopeAotTactic(
        batch, options.qkvRopeAotTactic.c_str());
    }
    if(options.useLinear2ResidualAot) {
      linear2AotByBatch[batch] = findResidualGemmAotTactic(
        batch, sm120NumSms, options.persistingL2Streams, 1152,
        options.linear2AotTactic.c_str());
    }
    if(options.useOutProjectionResidualAot) {
      outProjectionAotByBatch[batch] = findResidualGemmAotTactic(
        batch, sm120NumSms, options.persistingL2Streams, 384,
        options.outProjectionAotTactic.c_str());
    }
  }
  if(options.useProjectionGemmLt)
    ltMatmulState = make_unique<LtMatmulState>();
  if((options.usePersistingL2Trunk || options.usePersistingL2Inner) &&
     nnXLen == 19 && nnYLen == 19 &&
     useFP16 && useNHWC && desc->trunk.trunkNumChannels == 768) {
    int l2Device = 0;
    int maxPersistingBytes = 0;
    int maxWindowBytes = 0;
    CUDA_ERR("Sm120PersistingL2", cudaGetDevice(&l2Device));
    CUDA_ERR("Sm120PersistingL2", cudaDeviceGetAttribute(
      &maxPersistingBytes, cudaDevAttrMaxPersistingL2CacheSize, l2Device));
    CUDA_ERR("Sm120PersistingL2", cudaDeviceGetAttribute(
      &maxWindowBytes, cudaDevAttrMaxAccessPolicyWindowSize, l2Device));

    if(options.usePersistingL2Trunk) {
      persistingL2TrunkWindowBytes =
        (size_t)maxBatchSize * nnXLen * nnYLen *
        desc->trunk.trunkNumChannels * sizeof(half);
    }
    if(options.usePersistingL2Inner && desc->trunk.midNumChannels == 384) {
      persistingL2InnerWindowBytes =
        (size_t)maxBatchSize * nnXLen * nnYLen * desc->trunk.midNumChannels * sizeof(half);
    }
    const size_t windowsPerStream =
      persistingL2TrunkWindowBytes + persistingL2InnerWindowBytes;
    const size_t totalWindowBytes = (size_t)options.persistingL2Streams * windowsPerStream;
    if(totalWindowBytes > 0 &&
       persistingL2TrunkWindowBytes <= (size_t)maxWindowBytes &&
       persistingL2InnerWindowBytes <= (size_t)maxWindowBytes &&
       maxPersistingBytes > 0) {
      persistingL2RequestedBytes = std::min((size_t)maxPersistingBytes, totalWindowBytes);
      CUDA_ERR("Sm120PersistingL2", cudaDeviceSetLimit(
        cudaLimitPersistingL2CacheSize, persistingL2RequestedBytes));
      CUDA_ERR("Sm120PersistingL2", cudaDeviceGetLimit(
        &persistingL2ActualBytes, cudaLimitPersistingL2CacheSize));
      const float grantedHitRatio = std::min(
        1.0f, (float)((double)persistingL2ActualBytes / (double)totalWindowBytes));
      persistingL2TrunkHitRatio = std::min(
        (float)options.persistingL2HitRatio, grantedHitRatio);
      persistingL2TrunkActive =
        persistingL2TrunkWindowBytes > 0 && persistingL2TrunkHitRatio > 0.0f;
      if(persistingL2InnerWindowBytes > 0) {
        persistingL2InnerHitRatio = persistingL2TrunkHitRatio;
        persistingL2InnerActive = persistingL2InnerHitRatio > 0.0f;
      }
    }
  }
  // The official traversal invokes the SM120 operator hooks owned above.
}

Sm120Model::~Sm120Model() {
  if(dualFfnSharedAHandle != NULL)
    katago_destroy_dual_ffn_shared_a_sm120(dualFfnSharedAHandle);
  if(wideHeadProjectionHandle != NULL)
    katago_destroy_outer_projection_down_sm120(wideHeadProjectionHandle);
  if(wideHeadProjectionWeights != NULL)
    cudaFree(wideHeadProjectionWeights);
  if(exactMaskSumBuf != NULL)
    cudaFree(exactMaskSumBuf);
  for(const auto& entry: fusedFFNPairedWeights)
    cudaFree(entry.second);
  for(const auto& entry: wideFFNSingleGemmWeights)
    cudaFree(entry.second);
  for(const auto& entry: wideQKVWeights)
    cudaFree(entry.second);
  for(const auto& entry: qkvRopeTables)
    cudaFree(entry.second);
  for(const auto& entry: qkvStridedWeights)
    cudaFree(entry.second);
  for(const auto& entry: outerProjectionDownHandles)
    katago_destroy_outer_projection_down_sm120(entry.second);
  for(const auto& entry: outerProjectionUpHandles)
    katago_destroy_outer_projection_up_sm120(entry.second);
}

void Sm120Model::setLogger(Logger* logger_) {
  logger = logger_;
}

bool Sm120Model::hasPersistingL2Trunk() const {
  return persistingL2TrunkActive;
}

bool Sm120Model::hasPersistingL2Inner() const {
  return persistingL2InnerActive;
}

float* Sm120Model::getExactMaskSumBuf() const {
  return exactMaskSumBuf;
}

void Sm120Model::apply(
  CudaHandles* cudaHandles_,
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
) {
  (void)cudaHandles_;
  (void)scratch;
  (void)batchSize;
  (void)requireExactNNLen;
  (void)inputBuf;
  (void)inputGlobalBuf;
  (void)inputMetaBuf;
  (void)policyPassBuf;
  (void)policyBuf;
  (void)valueBuf;
  (void)scoreValueBuf;
  (void)ownershipBuf;
  (void)workspaceBuf;
  (void)workspaceBytes;

  if(!loggedFallback) {
    if(logger != NULL)
      logger->write("SM120 backend: official forward adapter active");
    loggedFallback = true;
  }

  // Keep the official traversal; its operator boundaries dispatch to the
  // SM120 hooks installed by ComputeHandle.
  officialApply(
    officialApplyContext,
    cudaHandles,
    scratch,
    batchSize,
    requireExactNNLen,
    inputBuf,
    inputGlobalBuf,
    inputMetaBuf,
    policyPassBuf,
    policyBuf,
    valueBuf,
    scoreValueBuf,
    ownershipBuf,
    workspaceBuf,
    workspaceBytes
  );
}

bool Sm120Model::attention(
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
) {
  (void)scratch;
  (void)workspaceBuf;
  (void)workspaceBytes;

  // Packed QKV has no compatible official fallback because Q/K/V share a
  // token-interleaved allocation. Preserve the ordinary planar fallback, but
  // make any packed-path contract violation explicit rather than reporting a
  // generic failure later in cudabackend.cpp.
  const bool exactTacticRequired =
    options.fa4AotTacticExplicit && options.fa4AotTactic != "disabled";
  const auto rejectUnsupportedPacked = [packedQKV,exactTacticRequired](const char* reason) {
    if(packedQKV || exactTacticRequired)
      throw StringError(string("SM120 backend: requested FA4 precondition failed: ") + reason);
    return false;
  };

  // Stage 1 gate: FP16, MHA (numKVHeads == numHeads), head dim 32, no mask
  // (full-board requireExactNNLen paths), 19x19 sequence length, and the FA4
  // switch on. Everything else falls back to the official attention path.
  if(!options.useFlashAttention)
    return rejectUnsupportedPacked("cudaUseFlashAttentionSm120 is disabled");
  if(!usingFP16)
    return rejectUnsupportedPacked("inference is not FP16");
  if(maskBuf != NULL)
    return rejectUnsupportedPacked("mask is present");
  if(numHeads != numKVHeads || qHeadDim != 32 || vHeadDim != 32 || seqLen != 361)
    return rejectUnsupportedPacked("attention shape is not S361 MHA HxD32");
  if(batchSize < 1 || batchSize > maxBatchSize)
    return rejectUnsupportedPacked("batch is outside the model capacity");

  const FA4AotTactic* searchTactic = fa4AotByBatch[batchSize];
  if(searchTactic != nullptr) {
    float scale = 1.0f / std::sqrt((float)qHeadDim);
    cudaError_t status = searchTactic->launch(
      qBuf, kBuf, vBuf, attnOutBuf, batchSize, seqLen, numHeads,
      qHeadDim, scale, packedQKV, stream);
    if(status != cudaSuccess && exactTacticRequired)
      throw StringError(
        "SM120 backend: requested FA4 AOT launch failed: " +
        string(cudaGetErrorString(status)));
    if(status != cudaSuccess)
      return false;
    const bool shouldLogMaxBatch =
      batchSize == maxBatchSize && !loggedFa4AtMaxBatch;
    if((!loggedFa4 || shouldLogMaxBatch) && logger != NULL)
      logger->write("SM120 backend: FA4 AOT active, tactic=" + string(searchTactic->id));
    loggedFa4 = true;
    if(batchSize == maxBatchSize)
      loggedFa4AtMaxBatch = true;
    return true;
  }
  throw StringError(
    "SM120 backend: selected FA4 tactic '" + options.fa4AotTactic +
    "' is not registered for exact batch " + Global::intToString(batchSize)
  );
}

bool Sm120Model::ffnSingleGemm(
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
) {
  if(!usingFP16)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || matBatchSize % 361 != 0)
    return false;
  int batchSize = matBatchSize / 361;
  if(batchSize < 1 || batchSize > maxBatchSize || numChannels != 384 || ffnChannels != 1152)
    return false;

  auto getPairedWeights = [&]() -> void* {
    auto existing = fusedFFNPairedWeights.find(linear1Weights);
    if(existing != fusedFFNPairedWeights.end())
      return existing->second;
    void* pairedWeights = NULL;
    constexpr int pairChannels = 64;
    const size_t sourcePitch = (size_t)ffnChannels * sizeof(half);
    const size_t pairedPitch = sourcePitch * 2;
    const size_t chunkBytes = (size_t)pairChannels * sizeof(half);
    CUDA_ERR("Sm120FusedFFNPairedWeights", cudaMalloc(
      &pairedWeights, (size_t)2 * ffnChannels * numChannels * sizeof(half)));
    for(int start = 0; start < ffnChannels; start += pairChannels) {
      const size_t pairOffset = (size_t)2 * start * sizeof(half);
      const size_t sourceOffset = (size_t)start * sizeof(half);
      CUDA_ERR("Sm120FusedFFNPairedWeights", cudaMemcpy2DAsync(
        (char*)pairedWeights + pairOffset, pairedPitch,
        (const char*)linear1Weights + sourceOffset, sourcePitch,
        chunkBytes, numChannels, cudaMemcpyDeviceToDevice, stream));
      CUDA_ERR("Sm120FusedFFNPairedWeights", cudaMemcpy2DAsync(
        (char*)pairedWeights + pairOffset + chunkBytes, pairedPitch,
        (const char*)linearGateWeights + sourceOffset, sourcePitch,
        chunkBytes, numChannels, cudaMemcpyDeviceToDevice, stream));
    }
    fusedFFNPairedWeights.emplace(linear1Weights, pairedWeights);
    return pairedWeights;
  };

  const FusedFFNAotTactic* ffnTactic = fusedFFNAotByBatch[batchSize];
  if(dualFfnSharedAHandle != NULL) {
    int status = katago_launch_dual_ffn_shared_a_sm120(
      dualFfnSharedAHandle, inputBuf, linear1Weights, linearGateWeights,
      ffnOutBuf, matBatchSize, stream);
    if(status != 0)
      throw StringError(
        "SM120 CUTLASS shared-A dual FFN launch failed, status=" +
        Global::intToString(status));
    if(!loggedFusedFFN) {
      if(logger != NULL)
        logger->write(
          "SM120 backend: CUTLASS shared-A dual FFN active, tactic=" +
          options.fusedFFNAotTactic);
      loggedFusedFFN = true;
    }
    return true;
  }
  if(ffnTactic != nullptr) {
    const half* firstWeights = (const half*)linear1Weights;
    const half* secondWeights = (const half*)linearGateWeights;
    if(ffnTactic->pairedWeights) {
      firstWeights = (const half*)getPairedWeights();
      secondWeights = NULL;
    }
    CUDA_ERR("Sm120FusedFFNAot", ffnTactic->launch(
      (const half*)inputBuf,
      firstWeights,
      secondWeights,
      (half*)ffnOutBuf,
      stream));
    if(!loggedFusedFFN) {
      if(logger != NULL)
        logger->write("SM120 backend: fused FFN AOT active, tactic=" + string(ffnTactic->id));
      loggedFusedFFN = true;
    }
    return true;
  }

  if(options.fusedFFNAotTacticExplicit &&
     options.fusedFFNAotTactic != "disabled")
    throw StringError(
      "SM120 backend: requested fused FFN tactic '" +
      options.fusedFFNAotTactic + "' is not registered for batch " +
      Global::intToString(batchSize));

  if(!options.useWideFFNSingleGemm)
    return false;

  void* wideWeights = NULL;
  auto existing = wideFFNSingleGemmWeights.find(linear1Weights);
  if(existing != wideFFNSingleGemmWeights.end()) {
    wideWeights = existing->second;
  }
  else {
    size_t rowBytes = (size_t)ffnChannels * sizeof(half);
    size_t widePitch = rowBytes * 2;
    CUDA_ERR("Sm120WideFFNSingleGemm", cudaMalloc(
      &wideWeights, (size_t)2 * ffnChannels * numChannels * sizeof(half)));
    CUDA_ERR("Sm120WideFFNSingleGemm", cudaMemcpy2DAsync(
      wideWeights, widePitch, linear1Weights, rowBytes, rowBytes, numChannels,
      cudaMemcpyDeviceToDevice, stream));
    CUDA_ERR("Sm120WideFFNSingleGemm", cudaMemcpy2DAsync(
      (char*)wideWeights + rowBytes, widePitch,
      linearGateWeights, rowBytes, rowBytes, numChannels,
      cudaMemcpyDeviceToDevice, stream));
    wideFFNSingleGemmWeights.emplace(linear1Weights, wideWeights);
  }

  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  CUBLAS_ERR("Sm120WideFFNSingleGemm", cublasHgemm(
    cublas, CUBLAS_OP_N, CUBLAS_OP_N,
    ffnChannels * 2, matBatchSize, numChannels,
    &alpha, (const half*)wideWeights, ffnChannels * 2,
    (const half*)inputBuf, numChannels,
    &beta, (half*)wideScratchBuf, ffnChannels * 2));
  launchWideSwiGLU(
    (const half*)wideScratchBuf, (half*)ffnOutBuf, matBatchSize, ffnChannels, stream);
  CUDA_ERR("Sm120WideFFNSingleGemm", cudaPeekAtLastError());

  if(!loggedWideFFNSingleGemm) {
    if(logger != NULL)
      logger->write("SM120 backend: single-wide FFN projection active");
    loggedWideFFNSingleGemm = true;
  }
  return true;
}

bool Sm120Model::matMulLt(
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
) {
  (void)workspace;
  (void)workspaceBytes;
  if(!options.useProjectionGemmLt || ltMatmulState == NULL || !usingFP16 ||
     nnXLen != 19 || nnYLen != 19 || matBatchSize <= 0)
    return false;

  LtMatmulState::Plan* plan = ltMatmulState->getOrCreatePlan(
    outChannels, matBatchSize, inChannels, weights, input, output, stream);
  if(plan == NULL || !plan->valid)
    return false;

  const __half alpha = __float2half(1.0f);
  const __half beta = __float2half(0.0f);
  const cublasStatus_t status = cublasLtMatmul(
    ltMatmulState->handle,
    plan->operationDesc,
    &alpha,
    weights,
    plan->aDesc,
    input,
    plan->bDesc,
    &beta,
    output,
    plan->cDesc,
    output,
    plan->cDesc,
    &plan->algo,
    ltMatmulState->workspace,
    plan->workspaceBytes,
    stream
  );
  if(status != CUBLAS_STATUS_SUCCESS)
    return false;

  if(!loggedProjectionGemmLt) {
    if(logger != NULL)
      logger->write("SM120 backend: shape-keyed autotuned cuBLASLt FP16 MatMul active");
    loggedProjectionGemmLt = true;
  }
  return true;
}

bool Sm120Model::conv1x1(
  const void* weights,
  const void* input,
  void* output,
  int matBatchSize,
  int inChannels,
  int outChannels,
  bool accumulate,
  bool usingFP16,
  cudaStream_t stream
) {
  if((options.outerProjectionDownTactic == "disabled" &&
      options.outerProjectionUpTactic == "disabled") ||
     !usingFP16 || weights == NULL ||
     input == NULL || output == NULL || nnXLen != 19 || nnYLen != 19 ||
     matBatchSize <= 0 || matBatchSize % 361 != 0)
    return false;

  void* handle = NULL;
  int status = 0;
  if(options.outerProjectionDownTactic != "disabled" &&
     inChannels == 768 && outChannels == 384 && !accumulate) {
    auto found = outerProjectionDownHandles.find(weights);
    if(found == outerProjectionDownHandles.end()) {
      handle = katago_create_outer_projection_down_sm120(
        weights, options.outerProjectionDownTactic.c_str());
      if(handle == NULL)
        throw StringError("SM120 outer projection down AOT handle creation failed");
      outerProjectionDownHandles.emplace(weights, handle);
    }
    else
      handle = found->second;
    status = katago_launch_outer_projection_down_sm120(
      handle, input, output, matBatchSize, stream);
    if(status != 0)
      throw StringError(
        "SM120 outer projection down AOT launch failed, status=" +
        Global::intToString(status));
    if(!loggedOuterProjectionDown) {
      if(logger != NULL)
        logger->write(
          "SM120 backend: C768->C384 outer projection CUTLASS active, tactic=" +
          options.outerProjectionDownTactic);
      loggedOuterProjectionDown = true;
    }
    return true;
  }

  if(options.outerProjectionUpTactic != "disabled" &&
     inChannels == 384 && outChannels == 768 && accumulate) {
    auto found = outerProjectionUpHandles.find(weights);
    if(found == outerProjectionUpHandles.end()) {
      handle = katago_create_outer_projection_up_sm120(
        weights, options.outerProjectionUpTactic.c_str());
      if(handle == NULL)
        throw StringError("SM120 outer projection up AOT handle creation failed");
      outerProjectionUpHandles.emplace(weights, handle);
    }
    else
      handle = found->second;
    status = katago_launch_outer_projection_up_sm120(
      handle, input, output, matBatchSize, stream);
    if(status != 0)
      throw StringError(
        "SM120 outer projection up AOT launch failed, status=" +
        Global::intToString(status));
    if(!loggedOuterProjectionUp) {
      if(logger != NULL)
        logger->write(
          "SM120 backend: C384->C768 outer projection+residual CUTLASS active, tactic=" +
          options.outerProjectionUpTactic);
      loggedOuterProjectionUp = true;
    }
    return true;
  }
  return false;
}

bool Sm120Model::initialGlobal(
  void* spatialBuf,
  const void* globalInput,
  const void* weights,
  int batchSize,
  int inputChannels,
  int outputChannels,
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
) {
  if(!options.useInitialGlobalMatMulAdd || !usingFP16 || !usingNHWC ||
     spatialBuf == NULL || globalInput == NULL || weights == NULL ||
     batchSize < 1 || batchSize > maxBatchSize ||
     inputChannels != 19 || outputChannels != 768 ||
     nnXLen != 19 || nnYLen != 19)
    return false;
  launchInitialGlobalMatMulAdd(
    (half*)spatialBuf, (const half*)globalInput, (const half*)weights,
    batchSize, stream);
  CUDA_ERR("Sm120InitialGlobal", cudaPeekAtLastError());
  if(!loggedInitialGlobal) {
    if(logger != NULL)
      logger->write(
        "SM120 backend: fused global-feature matmul+broadcast add active");
    loggedInitialGlobal = true;
  }
  return true;
}

bool Sm120Model::qkvStrided(
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
) {
  if(packedOutput == NULL || ropeApplied == NULL)
    return false;
  *packedOutput = false;
  *ropeApplied = false;
  if(!usingFP16)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || matBatchSize % 361 != 0)
    return false;
  int batchSize = matBatchSize / 361;
  if(batchSize < 1 || batchSize > maxBatchSize)
    return false;
  if(numChannels != 384 || qDim != 384 || kDim != qDim || vDim != qDim)
    return false;

  const WideQKVAotTactic* qkvTactic = wideQKVAotByBatch[batchSize];
  const WideQKVRopeAotTactic* qkvRopeTactic =
    wideQKVRopeAotByBatch[batchSize];
  const bool packedAttentionReady = allowPackedOutput &&
    options.useFlashAttention && options.flashAttentionAccum == "both16";
  const bool packedPathReady = packedAttentionReady && options.useFusedQKRoPE &&
    options.useBatchSharedRoPE && options.useFlashAttention &&
    options.flashAttentionAccum == "both16";
  const bool useQKVRopeAot =
    qkvRopeTactic != nullptr && packedAttentionReady && ropeFreqs != NULL;
  const bool useQKVAot =
    qkvTactic != nullptr && (!qkvTactic->packedOutput || packedPathReady);
  if(options.qkvRopeAotTacticExplicit &&
     options.qkvRopeAotTactic != "disabled" && !useQKVRopeAot)
    throw StringError(
      "SM120 backend: requested packed QKV+RoPE tactic '" +
      options.qkvRopeAotTactic +
      "' is unavailable or its packed-attention preconditions are not met for batch " +
      Global::intToString(batchSize));
  if(options.wideQKVAotTacticExplicit &&
     options.wideQKVAotTactic != "disabled" && !useQKVRopeAot && !useQKVAot)
    throw StringError(
      "SM120 backend: requested wide-QKV tactic '" +
      options.wideQKVAotTactic +
      "' is unavailable or its packed-attention preconditions are not met for batch " +
      Global::intToString(batchSize));
  if(useQKVRopeAot || useQKVAot) {
    void* weights = NULL;
    auto existing = wideQKVWeights.find(qWeights);
    if(existing != wideQKVWeights.end()) {
      weights = existing->second;
    }
    else {
      constexpr size_t qkvDim = 384;
      constexpr size_t wideDim = 3 * qkvDim;
      constexpr size_t rows = 384;
      CUDA_ERR("Sm120WideQKV", cudaMalloc(&weights, rows * wideDim * sizeof(half)));
      CUDA_ERR("Sm120WideQKV", cudaMemcpy2DAsync(
        (half*)weights, wideDim * sizeof(half),
        qWeights, qkvDim * sizeof(half), qkvDim * sizeof(half), rows,
        cudaMemcpyDeviceToDevice, stream));
      CUDA_ERR("Sm120WideQKV", cudaMemcpy2DAsync(
        (half*)weights + qkvDim, wideDim * sizeof(half),
        kWeights, qkvDim * sizeof(half), qkvDim * sizeof(half), rows,
        cudaMemcpyDeviceToDevice, stream));
      CUDA_ERR("Sm120WideQKV", cudaMemcpy2DAsync(
        (half*)weights + 2 * qkvDim, wideDim * sizeof(half),
        vWeights, qkvDim * sizeof(half), qkvDim * sizeof(half), rows,
        cudaMemcpyDeviceToDevice, stream));
      wideQKVWeights.emplace(qWeights, weights);
    }

    if(useQKVRopeAot) {
      void* ropeTable = NULL;
      auto existingTable = qkvRopeTables.find(ropeFreqs);
      if(existingTable != qkvRopeTables.end())
        ropeTable = existingTable->second;
      else {
        constexpr size_t tableHalves = 361 * 192 * 2;
        CUDA_ERR("Sm120QKVRopeTable",cudaMalloc(
          &ropeTable,tableHalves * sizeof(half)));
        launchPrecomputeQKVRopeTable19Half(
          ropeFreqs,(half*)ropeTable,stream);
        CUDA_ERR("Sm120QKVRopeTable",cudaPeekAtLastError());
        qkvRopeTables.emplace(ropeFreqs,ropeTable);
      }
      CUDA_ERR("Sm120WideQKVRopeAot",qkvRopeTactic->launch(
        (const half*)inputBuf,(const half*)weights,(const half*)ropeTable,
        (half*)qkvBuf,stream));
      *packedOutput = true;
      *ropeApplied = true;
      if(!loggedQKVRopeAot) {
        if(logger != NULL)
          logger->write(
            "SM120 backend: packed QKV+RoPE AOT active, tactic=" +
            string(qkvRopeTactic->id));
        loggedQKVRopeAot = true;
      }
    }
    else {
      CUDA_ERR("Sm120WideQKVAot", qkvTactic->launch(
        (const half*)inputBuf, (const half*)weights, (half*)qkvBuf, stream));
      *packedOutput = qkvTactic->packedOutput;
      if(!loggedWideQKV) {
        if(logger != NULL)
          logger->write(
            "SM120 backend: wide QKV AOT active, tactic=" +
            string(qkvTactic->id));
        loggedWideQKV = true;
      }
    }
    return true;
  }

  if(!options.useQKVStrided)
    return false;

  void* weights = NULL;
  auto existing = qkvStridedWeights.find(qWeights);
  if(existing != qkvStridedWeights.end()) {
    weights = existing->second;
  }
  else {
    size_t oneWeightBytes = (size_t)qDim * numChannels * sizeof(half);
    CUDA_ERR("Sm120QKVStrided", cudaMalloc(&weights, oneWeightBytes * 3));
    CUDA_ERR("Sm120QKVStrided", cudaMemcpyAsync(
      weights, qWeights, oneWeightBytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_ERR("Sm120QKVStrided", cudaMemcpyAsync(
      (char*)weights + oneWeightBytes, kWeights, oneWeightBytes,
      cudaMemcpyDeviceToDevice, stream));
    CUDA_ERR("Sm120QKVStrided", cudaMemcpyAsync(
      (char*)weights + 2 * oneWeightBytes, vWeights, oneWeightBytes,
      cudaMemcpyDeviceToDevice, stream));
    qkvStridedWeights.emplace(qWeights, weights);
  }

  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0.0f);
  CUBLAS_ERR("Sm120QKVStrided", cublasHgemmStridedBatched(
    cublas, CUBLAS_OP_N, CUBLAS_OP_N,
    qDim, matBatchSize, numChannels,
    &alpha, (const half*)weights, qDim, (int64_t)qDim * numChannels,
    (const half*)inputBuf, numChannels, 0,
    &beta, (half*)qkvBuf, qDim, (int64_t)qDim * matBatchSize, 3));

  if(!loggedQKVStrided) {
    if(logger != NULL)
      logger->write("SM120 backend: strided-batched QKV projection active");
    loggedQKVStrided = true;
  }
  return true;
}

bool Sm120Model::fusedResidualGemm(
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
) {
  if(!options.useFusedResidualGemm || !usingFP16 || maskBuf != NULL)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || matBatchSize % 361 != 0)
    return false;
  int batchSize = matBatchSize / 361;
  if(batchSize < 1 || batchSize > maxBatchSize || outputChannels != 384)
    return false;
  if(inputChannels != 384 && inputChannels != 1152)
    return false;

  const ResidualGemmAotTactic* outProjectionTactic =
    inputChannels == 384 ? outProjectionAotByBatch[batchSize] : nullptr;
  if(outProjectionTactic != nullptr) {
    CUDA_ERR("Sm120OutProjectionResidual", outProjectionTactic->launch(
      (const half*)inputBuf, (const half*)weights, (half*)trunkBuf,
      matBatchSize, stream));
    if(!loggedOutProjectionAot) {
      if(logger != NULL)
        logger->write(
          "SM120 backend: out-projection residual AOT active, tactic=" +
          string(outProjectionTactic->id));
      loggedOutProjectionAot = true;
    }
    if(!loggedFusedResidualGemm) {
      if(logger != NULL)
        logger->write("SM120 backend: GEMM beta residual fusion active");
      loggedFusedResidualGemm = true;
    }
    return true;
  }
  if(inputChannels == 384 && options.outProjectionAotTacticExplicit &&
     options.outProjectionAotTactic != "disabled")
    throw StringError(
      "SM120 backend: requested out-projection tactic '" +
      options.outProjectionAotTactic + "' is not registered for batch " +
      Global::intToString(batchSize));

  const ResidualGemmAotTactic* linear2Tactic =
    inputChannels == 1152 ? linear2AotByBatch[batchSize] : nullptr;
  if(linear2Tactic != nullptr) {
    CUDA_ERR("Sm120Linear2Residual", linear2Tactic->launch(
      (const half*)inputBuf, (const half*)weights, (half*)trunkBuf,
      matBatchSize, stream));
    if(!loggedLinear2Aot) {
      if(logger != NULL)
        logger->write(
          "SM120 backend: linear2 residual AOT active, tactic=" +
          string(linear2Tactic->id));
      loggedLinear2Aot = true;
    }
    if(!loggedFusedResidualGemm) {
      if(logger != NULL)
        logger->write("SM120 backend: GEMM beta residual fusion active");
      loggedFusedResidualGemm = true;
    }
    return true;
  }
  if(inputChannels == 1152 && options.linear2AotTacticExplicit &&
     options.linear2AotTactic != "disabled")
    throw StringError(
      "SM120 backend: requested linear2 tactic '" +
      options.linear2AotTactic + "' is not registered for batch " +
      Global::intToString(batchSize));

  const half alpha = __float2half(1.0f);
  const half beta = __float2half(1.0f);
  CUBLAS_ERR("Sm120FusedResidualGemm", cublasHgemm(
    cublas, CUBLAS_OP_N, CUBLAS_OP_N,
    outputChannels, matBatchSize, inputChannels,
    &alpha, (const half*)weights, outputChannels,
    (const half*)inputBuf, inputChannels,
    &beta, (half*)trunkBuf, outputChannels));

  if(!loggedFusedResidualGemm) {
    if(logger != NULL)
      logger->write("SM120 backend: GEMM beta residual fusion active");
    loggedFusedResidualGemm = true;
  }
  return true;
}

bool Sm120Model::rmsNorm(
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
) {
  if(options.rmsNorm384Tactic == "disabled" || !usingFP16 || maskBuf != NULL)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || xySize != 361 || channels != 384)
    return false;
  if(batchSize < 1 || batchSize > maxBatchSize)
    return false;

  if(options.rmsNorm384Tactic == "warp4-vec8") {
    launchRMSNorm384Vec8(
      (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf,
      (const half*)betaBuf, batchSize * xySize, epsilon, stream);
  }
  else if(options.rmsNorm384Tactic == "ordered-ept3") {
    launchRMSNorm384OrderedEpt3(
      (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf,
      (const half*)betaBuf, batchSize * xySize, epsilon, stream);
  }
  else {
    launchRMSNorm384(
      (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf,
      (const half*)betaBuf, batchSize * xySize, epsilon, stream);
  }
  CUDA_ERR("Sm120RMSNorm384", cudaPeekAtLastError());
  if(!loggedRMSNorm384) {
    if(logger != NULL)
      logger->write(options.rmsNorm384Tactic == "warp4-vec8" ?
        "SM120 backend: vec8 C384 RMSNorm active" :
        (options.rmsNorm384Tactic == "ordered-ept3" ?
          "SM120 backend: ordered-EPT3 C384 RMSNorm active" :
          "SM120 backend: one-warp C384 RMSNorm active"));
    loggedRMSNorm384 = true;
  }
  return true;
}

bool Sm120Model::fusedQKRoPE(
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
) {
  if(!options.useFusedQKRoPE || !usingFP16)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || ropeXLen != 19 || seqLen != 361)
    return false;
  if(batchSize < 1 || batchSize > maxBatchSize)
    return false;
  if(numHeads != 12 || numKVHeads != 12 || qHeadDim != 32 || numPairs != 16)
    return false;

  if(options.useBatchSharedRoPE) {
    if(packedQKV) {
      if(options.useBatchSharedRoPEUnrolled)
        launchBatchSharedPackedFusedQKRoPEUnrolled(
          (half*)qBuf, (half*)kBuf, freqs, batchSize, stream);
      else
        launchBatchSharedPackedFusedQKRoPE19(
          (half*)qBuf, (half*)kBuf, freqs, batchSize, stream);
    }
    else
      launchBatchSharedFusedQKRoPE19(
        (half*)qBuf, (half*)kBuf, freqs, batchSize, stream);
    if(!loggedBatchSharedQKRoPE) {
      if(logger != NULL)
        logger->write(
          options.useBatchSharedRoPEUnrolled && packedQKV ?
          "SM120 backend: unrolled packed batch-shared fused Q/K RoPE active" :
          "SM120 backend: batch-shared fused Q/K RoPE active");
      loggedBatchSharedQKRoPE = true;
    }
    if(options.useBatchSharedRoPEUnrolled && packedQKV &&
       !loggedBatchSharedQKRoPEAtMaxBatch) {
      if(logger != NULL)
        logger->write(
          "SM120 backend: unrolled packed batch-shared fused Q/K RoPE active");
      loggedBatchSharedQKRoPEAtMaxBatch = true;
    }
  }
  else if(options.useFusedQKRoPEHalf2) {
    launchFusedQKRoPE19Half2(
      (half*)qBuf, (half*)kBuf, freqs, batchSize, stream);
    if(!loggedFusedQKRoPEHalf2) {
      if(logger != NULL)
        logger->write("SM120 backend: half2 fused Q/K RoPE active");
      loggedFusedQKRoPEHalf2 = true;
    }
  }
  else {
    launchFusedQKRoPE19(
      (half*)qBuf, (half*)kBuf, freqs, batchSize, stream);
  }
  CUDA_ERR("Sm120FusedQKRoPE", cudaPeekAtLastError());
  if(!loggedFusedQKRoPE) {
    if(logger != NULL)
      logger->write("SM120 backend: fused Q/K learnable RoPE active");
    loggedFusedQKRoPE = true;
  }
  return true;
}

bool Sm120Model::swiGLU(
  const void* a,
  const void* b,
  void* output,
  int numTokens,
  int channels,
  bool usingFP16,
  cudaStream_t stream
) {
  if(!options.useSwiGLU1152 || !usingFP16)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || channels != 1152)
    return false;
  if(numTokens < 361 || numTokens > maxBatchSize * 361 || numTokens % 361 != 0)
    return false;

  launchSwiGLU1152Half8(
    (const half*)a, (const half*)b, (half*)output,
    numTokens * channels, stream);
  CUDA_ERR("Sm120SwiGLU1152", cudaPeekAtLastError());
  if(!loggedSwiGLU1152) {
    if(logger != NULL)
      logger->write("SM120 backend: contiguous half8 C1152 SwiGLU active");
    loggedSwiGLU1152 = true;
  }
  return true;
}

bool Sm120Model::affineSilu(
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
) {
  if(options.affineSiluTactic == "disabled" || !usingFP16 || mask != NULL)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || xySize != 361)
    return false;
  if(channels != 384 && channels != 768)
    return false;
  if(activation != ACTIVATION_SILU || batchSize < 1 || batchSize > maxBatchSize)
    return false;
  if(options.affineSiluTactic == "flat-vec8-c768" && channels != 768)
    return false;

  if(options.affineSiluTactic == "flat-vec8-c768")
    launchAffineSiluFlatVec8C768(
      (const half*)input, (half*)output, (const half*)scale, (const half*)bias,
      batchSize * xySize, stream);
  else if(options.affineSiluTactic == "half2x3")
    launchAffineSiluHalf2x3(
      (const half*)input, (half*)output, (const half*)scale, (const half*)bias,
      batchSize * xySize, channels, stream);
  else
    launchAffineSiluHalf2(
      (const half*)input, (half*)output, (const half*)scale, (const half*)bias,
      batchSize * xySize, channels, stream);
  CUDA_ERR("Sm120AffineSiluHalf2", cudaPeekAtLastError());
  if(!loggedAffineSiluHalf2) {
    if(logger != NULL)
      logger->write(
        options.affineSiluTactic == "flat-vec8-c768" ?
        "SM120 backend: flat vec8 C768 affine SiLU active" :
        options.affineSiluTactic == "half2x3" ?
        "SM120 backend: half2x3 C384/C768 affine SiLU active" :
        "SM120 backend: half2 C384/C768 affine SiLU active");
    loggedAffineSiluHalf2 = true;
  }
  return true;
}

bool Sm120Model::postConvBNSilu(
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
) {
  if(!options.usePostConvBNSilu || !usingFP16 || !usingNHWC || mask != NULL)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || batchSize < 1 ||
     batchSize > maxBatchSize || xySize != 361 ||
     inputChannels != 384 || outputChannels != 768 ||
     activation != ACTIVATION_SILU)
    return false;
  if(input == NULL || weights == NULL || residual == NULL || activated == NULL ||
     scale == NULL || bias == NULL)
    return false;

  CUDA_ERR("Sm120PostConvBNSilu", launchPostConvResidualAffineSilu(
    (const half*)input, (const half*)weights, (half*)residual,
    (half*)activated, (const half*)scale, (const half*)bias,
    batchSize * xySize, stream));
  if(!loggedPostConvBNSilu) {
    if(logger != NULL)
      logger->write(
        "SM120 backend: postConv residual + following C768 affine SiLU active");
    loggedPostConvBNSilu = true;
  }
  return true;
}

bool Sm120Model::fusedPolicyP1(
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
) {
  if(!options.useFusedPolicyP1 || !usingFP16 || !usingNHWC)
    return false;
  if(batchSize < 1 || batchSize > maxBatchSize || nnXLen != 19 || nnYLen != 19 ||
     xySize != 361 || channels != 96 || inputStride < channels ||
     inputOffset < 0 || inputOffset + channels > inputStride)
    return false;

  launchFusedPolicyP1(
    (const half*)input, output, globalBias, scale, bias, batchSize,
    inputStride, inputOffset, stream);
  CUDA_ERR("Sm120FusedPolicyP1", cudaPeekAtLastError());
  if(!loggedFusedPolicyP1) {
    if(logger != NULL)
      logger->write("SM120 backend: fused 19x19 policy P1 active");
    loggedFusedPolicyP1 = true;
  }
  return true;
}

bool Sm120Model::wideHeadProjection(
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
) {
  if(wideHeadProjectionHandle == NULL || input == NULL || output == NULL ||
     outputRowStride == NULL || p1Offset == NULL || g1Offset == NULL ||
     v1Offset == NULL ||
     !usingFP16 || !usingNHWC || batchSize < 1 || batchSize > maxBatchSize ||
     xySize != 361 || inputChannels != 768 ||
     outputChannels < wideHeadProjectionChannels)
    return false;
  int status = katago_launch_outer_projection_down_sm120(
    wideHeadProjectionHandle, input, output, batchSize * xySize, stream);
  if(status != 0)
    throw StringError(
      "SM120 wide-head projection launch failed, status=" +
      Global::intToString(status));
  *outputRowStride = wideHeadProjectionChannels;
  *p1Offset = wideHeadP1Offset;
  *g1Offset = wideHeadG1Offset;
  *v1Offset = wideHeadV1Offset;
  if(!loggedWideHeadProjection) {
    if(logger != NULL)
      logger->write(
        wideHeadProjectionChannels == 288 ?
        "SM120 backend: partial C288 no-split g1+v1 head active" :
        "SM120 backend: full C384 no-split wide head projection active");
    loggedWideHeadProjection = true;
  }
  return true;
}

void Sm120Model::persistingL2Window(
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes
) {
  if(!persistingL2TrunkActive && !persistingL2InnerActive)
    return;

  if(basePtr == NULL) {
    clearPersistingL2Window(stream);
    return;
  }

  if(numBytes == persistingL2TrunkWindowBytes) {
    setPersistingL2Window(
      stream, basePtr, numBytes, persistingL2TrunkHitRatio);
    if(!loggedPersistingL2Trunk) {
      if(logger != NULL) {
        logger->write(
          "SM120 backend: persisting-L2 C768 trunk active, window=" +
          Global::uint64ToString((uint64_t)persistingL2TrunkWindowBytes) +
          " requested=" + Global::uint64ToString((uint64_t)persistingL2RequestedBytes) +
          " actual=" + Global::uint64ToString((uint64_t)persistingL2ActualBytes) +
          " hitRatio=" + Global::doubleToString(persistingL2TrunkHitRatio));
      }
      loggedPersistingL2Trunk = true;
    }
    return;
  }

  if(persistingL2InnerActive && numBytes == persistingL2InnerWindowBytes) {
    setPersistingL2Window(
      stream, basePtr, numBytes, persistingL2InnerHitRatio);
    if(!loggedPersistingL2Inner) {
      if(logger != NULL) {
        logger->write(
          "SM120 backend: persisting-L2 C384 inner active, window=" +
          Global::uint64ToString((uint64_t)persistingL2InnerWindowBytes) +
          " requested=" + Global::uint64ToString((uint64_t)persistingL2RequestedBytes) +
          " actual=" + Global::uint64ToString((uint64_t)persistingL2ActualBytes) +
          " hitRatio=" + Global::doubleToString(persistingL2InnerHitRatio));
      }
      loggedPersistingL2Inner = true;
    }
    return;
  }
}

} // namespace Sm120Backend
