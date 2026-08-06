#include "../neuralnet/cudabackend_sm120.h"
#include "../neuralnet/cudabackend_sm120_kernels.h"

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/activations.h"
#include "fa4_aot/fa4_sm120_b13.h"

#include "../core/global.h"
#include "../core/logger.h"

#include <cublasLt.h>

#include <algorithm>
#include <cmath>
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

// CuTe DSL's exported wrapper keeps its CUDA library/kernel handles in
// process-global generated state. Load it exactly once and keep it alive for
// the process lifetime so replaynn can replace its warmup model safely.
static fa4_Kernel_Module_t fa4Module = {};
static once_flag fa4LoadOnce;

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
  o.useFlashAttention = getBoolOpt(cfg, "cudaUseFlashAttentionSm120", true);
  if(cfg.contains("cudaFlashAttentionSm120Accum"))
    o.flashAttentionAccum = cfg.getString("cudaFlashAttentionSm120Accum");
  if(o.flashAttentionAccum != "none" && o.flashAttentionAccum != "fp32" &&
     o.flashAttentionAccum != "qk16" && o.flashAttentionAccum != "pv16" &&
     o.flashAttentionAccum != "both16")
    throw StringError("cudaFlashAttentionSm120Accum must be one of none/fp32/qk16/pv16/both16");
  o.useWideQKV = getBoolOpt(cfg, "cudaUseWideQKV", true);
  o.useWideQKVSingleStreamSchedule = getBoolOpt(cfg, "cudaUseWideQKVSingleStreamSchedule", false);
  o.useQKVStrided = getBoolOpt(cfg, "cudaUseQKVStridedSm120", false);
  o.useQKVGemmAot = getBoolOpt(cfg, "cudaUseQKVGemmAot", true);
  o.useQKVGemmRopeAot = getBoolOpt(cfg, "cudaUseQKVGemmRopeAot", false);
  o.useFusedQKRoPE = getBoolOpt(cfg, "cudaUseFusedQKRoPE", true);
  o.useFusedQKRoPEHalf2 = getBoolOpt(cfg, "cudaUseFusedQKRoPEHalf2Sm120", false);
  o.useBatchSharedRoPE = getBoolOpt(cfg, "cudaUseBatchSharedRoPE", false);
  o.useBatchSharedRoPEUnroll19 = getBoolOpt(cfg, "cudaUseBatchSharedRoPEUnroll19", true);
  o.useBatchSharedRoPETwoWay = getBoolOpt(cfg, "cudaUseBatchSharedRoPETwoWay", false);
  o.useFusedResidual = getBoolOpt(cfg, "cudaUseFusedResidual", true);
  o.useFusedResidualGemm = getBoolOpt(cfg, "cudaUseFusedResidualGemmSm120", true);
  o.useProjectionGemmLt = getBoolOpt(cfg, "cudaUseProjectionGemmLt", false);
  o.useLinear2ResidualAot = getBoolOpt(cfg, "cudaUseLinear2ResidualAot", true);
  o.useLinear2ResidualAotBalanced = getBoolOpt(cfg, "cudaUseLinear2ResidualAotBalanced", false);
  o.useOutProjectionResidualAot = getBoolOpt(cfg, "cudaUseOutProjectionResidualAot", false);
  o.useFusedFFN = getBoolOpt(cfg, "cudaUseFusedFFN", true);
  o.useFusedFFNAReuse = getBoolOpt(cfg, "cudaUseFusedFFNAReuseSm120", false);
  o.useFusedFFNSingleStreamSchedule = getBoolOpt(cfg, "cudaUseFusedFFNSingleStreamSchedule", false);
  o.useWideFFNSingleGemm = getBoolOpt(cfg, "cudaUseWideFFNSingleGemm", false);
  o.useFusedRMSNormFFN = getBoolOpt(cfg, "cudaUseFusedRMSNormFFN", false);
  o.useRMSNorm384 = getBoolOpt(cfg, "cudaUseRMSNorm384Sm120", true);
  o.useRMSNorm384Vec8 = getBoolOpt(cfg, "cudaUseRMSNorm384Vec8Sm120", false);
  o.useRMSNorm384TwoWarp = getBoolOpt(cfg, "cudaUseRMSNorm384TwoWarpSm120", false);
  o.useSwiGLU1152 = getBoolOpt(cfg, "cudaUseSwiGLU1152Sm120", true);
  o.useAffineSiluHalf2 = getBoolOpt(cfg, "cudaUseAffineSiluHalf2Sm120", true);
  o.useRMSNormQKVGemmAot = getBoolOpt(cfg, "cudaUseRMSNormQKVGemmAot", false);
  o.useGraph = getBoolOpt(cfg, "cudaUseGraph", false);
  o.usePersistingL2Trunk = getBoolOpt(cfg, "cudaUsePersistingL2Trunk", false);
  o.usePersistingL2Inner = getBoolOpt(cfg, "cudaUsePersistingL2Inner", false);
  o.useOuterProjectionAot = getBoolOpt(cfg, "cudaUseOuterProjectionAot", true);
  o.shareModelWeights = getBoolOpt(cfg, "cudaShareModelWeights", true);
  o.shareWideQKVWeights = getBoolOpt(cfg, "cudaShareWideQKVWeights", false);
  o.shareOuterProjectionWeights = getBoolOpt(cfg, "cudaShareOuterProjectionWeights", false);
  o.useInitialConvFrontend = getBoolOpt(cfg, "cudaUseInitialConvFrontend", true);
  o.useInitialConvBiasFrontend = getBoolOpt(cfg, "cudaUseInitialConvBiasFrontend", false);
  o.useInitialGlobalMatMulAdd = getBoolOpt(cfg, "cudaUseInitialGlobalMatMulAdd", true);
  o.useFusedPolicyP1 = getBoolOpt(cfg, "cudaUseFusedPolicyP1", false);
  o.useHeadBNHalfToFloat = getBoolOpt(cfg, "cudaUseHeadBNHalfToFloat", true);
  o.useWideHeadProjection = getBoolOpt(cfg, "cudaUseWideHeadProjection", true);
  return o;
}

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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->attention(
    cudaHandles, scratch, qBuf, kBuf, vBuf, maskBuf, attnOutBuf,
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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->qkvStrided(
    cublas, stream, qWeights, kWeights, vWeights, inputBuf, qkvBuf,
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
  bool usingFP16,
  cudaStream_t stream
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->fusedQKRoPE(
    qBuf, kBuf, freqs, batchSize, seqLen, numHeads, numKVHeads,
    qHeadDim, numPairs, nnXLen, usingFP16, stream);
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
) {
  Sm120Model* self = static_cast<Sm120Model*>(ctx);
  if(self == NULL)
    return false;
  return self->fusedPolicyP1(
    input, output, globalBias, scale, bias, batchSize, xySize, channels,
    usingFP16, usingNHWC, stream);
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
  logger(NULL),
  loggedFallback(false),
  loggedFa4(false),
  loggedFusedFFN(false),
  loggedProjectionGemmLt(false),
  loggedWideFFNSingleGemm(false),
  loggedWideQKV(false),
  loggedQKVStrided(false),
  loggedFusedResidualGemm(false),
  loggedRMSNorm384(false),
  loggedFusedQKRoPE(false),
  loggedBatchSharedQKRoPE(false),
  loggedFusedQKRoPEHalf2(false),
  loggedSwiGLU1152(false),
  loggedAffineSiluHalf2(false),
  loggedFusedPolicyP1(false),
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
  if(options.useProjectionGemmLt)
    ltMatmulState = make_unique<LtMatmulState>();
  if(options.usePersistingL2Trunk && maxBatchSize == 13 && nnXLen == 19 && nnYLen == 19 &&
     useFP16 && useNHWC && desc->trunk.trunkNumChannels == 768) {
    int device = 0;
    int maxPersistingBytes = 0;
    int maxWindowBytes = 0;
    CUDA_ERR("Sm120PersistingL2", cudaGetDevice(&device));
    CUDA_ERR("Sm120PersistingL2", cudaDeviceGetAttribute(
      &maxPersistingBytes, cudaDevAttrMaxPersistingL2CacheSize, device));
    CUDA_ERR("Sm120PersistingL2", cudaDeviceGetAttribute(
      &maxWindowBytes, cudaDevAttrMaxAccessPolicyWindowSize, device));

    persistingL2TrunkWindowBytes =
      (size_t)maxBatchSize * nnXLen * nnYLen * desc->trunk.trunkNumChannels * sizeof(half);
    if(options.usePersistingL2Inner && desc->trunk.midNumChannels == 384) {
      persistingL2InnerWindowBytes =
        (size_t)maxBatchSize * nnXLen * nnYLen * desc->trunk.midNumChannels * sizeof(half);
    }
    const size_t windowsPerStream =
      persistingL2TrunkWindowBytes + persistingL2InnerWindowBytes;
    const size_t totalWindowBytes = 2 * windowsPerStream;
    if(persistingL2TrunkWindowBytes <= (size_t)maxWindowBytes &&
       persistingL2InnerWindowBytes <= (size_t)maxWindowBytes &&
       maxPersistingBytes > 0) {
      persistingL2RequestedBytes = std::min((size_t)maxPersistingBytes, totalWindowBytes);
      CUDA_ERR("Sm120PersistingL2", cudaDeviceSetLimit(
        cudaLimitPersistingL2CacheSize, persistingL2RequestedBytes));
      CUDA_ERR("Sm120PersistingL2", cudaDeviceGetLimit(
        &persistingL2ActualBytes, cudaLimitPersistingL2CacheSize));
      persistingL2TrunkHitRatio = std::min(
        1.0f, (float)((double)persistingL2ActualBytes / (double)totalWindowBytes));
      persistingL2TrunkActive = persistingL2TrunkHitRatio > 0.0f;
      if(persistingL2InnerWindowBytes > 0) {
        persistingL2InnerHitRatio = persistingL2TrunkHitRatio;
        persistingL2InnerActive = persistingL2TrunkActive;
      }
    }
  }
  // Stage 0 scaffold: apply() delegates to the official model until stages land.
}

Sm120Model::~Sm120Model() {
  for(const auto& entry: wideFFNSingleGemmWeights)
    cudaFree(entry.second);
  for(const auto& entry: wideQKVWeights)
    cudaFree(entry.second);
  for(const auto& entry: qkvStridedWeights)
    cudaFree(entry.second);
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
      logger->write("SM120 backend: stage-0 official fallback active (rebuild scaffold)");
    loggedFallback = true;
  }

  // Stage 0: bit-identical delegation to the official backend. Once stage 1 lands, this becomes
  // the SM120-specific forward path with per-stage fallback where shapes/precision are unsupported.
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

  // Stage 1 gate: FP16, MHA (numKVHeads == numHeads), head dim 32, no mask
  // (full-board requireExactNNLen paths), 19x19 sequence length, and the FA4
  // switch on. Everything else falls back to the official attention path.
  if(!options.useFlashAttention)
    return false;
  if(options.flashAttentionAccum != "both16")
    return false;
  if(!usingFP16)
    return false;
  if(maskBuf != NULL)
    return false;
  if(numHeads != numKVHeads || qHeadDim != 32 || vHeadDim != 32 || seqLen != 361)
    return false;
  if(batchSize < 1 || batchSize > maxBatchSize)
    return false;

  call_once(fa4LoadOnce, []() { fa4_Kernel_Module_Load(&fa4Module); });

  fa4_Tensor_mQ_t tq = {qBuf, {batchSize, seqLen, numHeads, qHeadDim}, {seqLen * numHeads * qHeadDim, numHeads * qHeadDim, qHeadDim}};
  fa4_Tensor_mK_t tk = {kBuf, {batchSize, seqLen, numKVHeads, qHeadDim}, {seqLen * numKVHeads * qHeadDim, numKVHeads * qHeadDim, qHeadDim}};
  fa4_Tensor_mV_t tv = {vBuf, {batchSize, seqLen, numKVHeads, vHeadDim}, {seqLen * numKVHeads * vHeadDim, numKVHeads * vHeadDim, vHeadDim}};
  fa4_Tensor_mO_t to = {attnOutBuf, {batchSize, seqLen, numHeads, vHeadDim}, {seqLen * numHeads * vHeadDim, numHeads * vHeadDim, vHeadDim}};

  float scale = 1.0f / std::sqrt((float)qHeadDim);
  int32_t ret = cute_dsl_fa4_wrapper(&fa4Module, &tq, &tk, &tv, &to, scale, stream);

  if(ret != 0) {
    if(logger != NULL)
      logger->write("SM120 backend: FA4 AOT attention launch failed, falling back to official path");
    return false;
  }
  CUDA_ERR("Sm120Attention", cudaPeekAtLastError());

  if(!loggedFa4) {
    if(logger != NULL)
      logger->write("SM120 backend: FA4 AOT attention active (FP16 QK/PV accumulation)");
    loggedFa4 = true;
  }
  return true;
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

  if(options.useFusedFFN && batchSize == 13) {
    if(options.useFusedFFNSingleStreamSchedule) {
      CUDA_ERR("Sm120FusedFFNB13S1", launchFusedFFNB13S1(
        (const half*)inputBuf,
        (const half*)linear1Weights,
        (const half*)linearGateWeights,
        (half*)ffnOutBuf,
        stream));
    }
    else if(options.useFusedFFNAReuse) {
      launchFusedFFNB13CandidateAReuse(
        (const half*)inputBuf,
        (const half*)linear1Weights,
        (const half*)linearGateWeights,
        (half*)ffnOutBuf,
        stream);
    }
    else {
      launchFusedFFNB13(
        (const half*)inputBuf,
        (const half*)linear1Weights,
        (const half*)linearGateWeights,
        (half*)ffnOutBuf,
        stream);
    }
    CUDA_ERR("Sm120FusedFFNB13", cudaPeekAtLastError());
    if(!loggedFusedFFN) {
      if(logger != NULL)
        logger->write(
          options.useFusedFFNSingleStreamSchedule ?
            "SM120 backend: TileLang fused B13 FFN projection active (S1 schedule)" :
          options.useFusedFFNAReuse ?
            "SM120 backend: TileLang fused B13 FFN projection active (A-fragment reuse)" :
            "SM120 backend: TileLang fused B13 FFN projection active");
      loggedFusedFFN = true;
    }
    return true;
  }

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
     nnXLen != 19 || nnYLen != 19 ||
     (matBatchSize != 13 && matBatchSize != 13 * 361))
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
      logger->write("SM120 backend: fixed-B13 autotuned cuBLASLt FP16 MatMul active");
    loggedProjectionGemmLt = true;
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
  int matBatchSize,
  int numChannels,
  int qDim,
  int kDim,
  int vDim,
  bool usingFP16
) {
  if(!usingFP16)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || matBatchSize % 361 != 0)
    return false;
  int batchSize = matBatchSize / 361;
  if(batchSize < 1 || batchSize > maxBatchSize)
    return false;
  if(numChannels != 384 || qDim != 384 || kDim != qDim || vDim != qDim)
    return false;

  if(options.useWideQKV && options.useQKVGemmAot && batchSize == 13) {
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

    if(options.useWideQKVSingleStreamSchedule) {
      CUDA_ERR("Sm120WideQKVS1", launchWideQKVB13S1(
        (const half*)inputBuf, (const half*)weights, (half*)qkvBuf, stream));
    }
    else {
      CUDA_ERR("Sm120WideQKV", launchWideQKVB13(
        (const half*)inputBuf, (const half*)weights, (half*)qkvBuf, stream));
    }
    if(!loggedWideQKV) {
      if(logger != NULL)
        logger->write(options.useWideQKVSingleStreamSchedule ?
          "SM120 backend: TileLang wide B13 QKV projection active (S1 schedule)" :
          "SM120 backend: TileLang wide B13 QKV projection active");
      loggedWideQKV = true;
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

  if(options.useOutProjectionResidualAot && batchSize == 13 && inputChannels == 384) {
    CUDA_ERR("Sm120OutProjectionResidual", launchOutProjectionResidualB13(
      (const half*)inputBuf, (const half*)weights, (half*)trunkBuf, stream));
    return true;
  }

  if(options.useLinear2ResidualAot && batchSize == 13 && inputChannels == 1152) {
    if(options.useLinear2ResidualAotBalanced) {
      CUDA_ERR("Sm120Linear2ResidualBalanced", launchLinear2ResidualB13Balanced(
        (const half*)inputBuf, (const half*)weights, (half*)trunkBuf, stream));
      return true;
    }
    CUDA_ERR("Sm120Linear2Residual", launchLinear2ResidualB13(
      (const half*)inputBuf, (const half*)weights, (half*)trunkBuf, stream));
    return true;
  }

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
  if(!options.useRMSNorm384 || !usingFP16 || maskBuf != NULL)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || xySize != 361 || channels != 384)
    return false;
  if(batchSize < 1 || batchSize > maxBatchSize)
    return false;

  if(options.useRMSNorm384Vec8) {
    launchRMSNorm384Vec8(
      (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf,
      (const half*)betaBuf, batchSize * xySize, epsilon, stream);
  }
  else if(options.useRMSNorm384TwoWarp) {
    launchRMSNorm384TwoWarp(
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
      logger->write(options.useRMSNorm384Vec8 ?
        "SM120 backend: vec8 C384 RMSNorm active" :
        (options.useRMSNorm384TwoWarp ?
          "SM120 backend: two-warp C384 RMSNorm active" :
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

  if(options.useBatchSharedRoPE && batchSize == 13) {
    launchBatchSharedFusedQKRoPE19B13(
      (half*)qBuf, (half*)kBuf, freqs, stream);
    if(!loggedBatchSharedQKRoPE) {
      if(logger != NULL)
        logger->write("SM120 backend: B13-shared fused Q/K RoPE active");
      loggedBatchSharedQKRoPE = true;
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
  if(!options.useAffineSiluHalf2 || !usingFP16 || mask != NULL)
    return false;
  if(nnXLen != 19 || nnYLen != 19 || xySize != 361)
    return false;
  if(channels != 384 && channels != 768)
    return false;
  if(activation != ACTIVATION_SILU || batchSize < 1 || batchSize > maxBatchSize)
    return false;

  launchAffineSiluHalf2(
    (const half*)input, (half*)output, (const half*)scale, (const half*)bias,
    batchSize * xySize, channels, stream);
  CUDA_ERR("Sm120AffineSiluHalf2", cudaPeekAtLastError());
  if(!loggedAffineSiluHalf2) {
    if(logger != NULL)
      logger->write("SM120 backend: half2 C384/C768 affine SiLU active");
    loggedAffineSiluHalf2 = true;
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
  bool usingFP16,
  bool usingNHWC,
  cudaStream_t stream
) {
  if(!options.useFusedPolicyP1 || !usingFP16 || !usingNHWC)
    return false;
  if(batchSize != 13 || nnXLen != 19 || nnYLen != 19 ||
     xySize != 361 || channels != 96)
    return false;

  launchFusedPolicyP1B13(
    (const half*)input, output, globalBias, scale, bias, stream);
  CUDA_ERR("Sm120FusedPolicyP1", cudaPeekAtLastError());
  if(!loggedFusedPolicyP1) {
    if(logger != NULL)
      logger->write("SM120 backend: exact-B13 fused policy P1 active");
    loggedFusedPolicyP1 = true;
  }
  return true;
}

void Sm120Model::persistingL2Window(
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes
) {
  if(!persistingL2TrunkActive)
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
