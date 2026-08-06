#include "../neuralnet/cudabackend_sm89.h"
#include "../neuralnet/cudabackend_sm89_forward.h"

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"

#include "../core/global.h"
#include "../core/logger.h"

#include <algorithm>

using namespace std;

namespace Sm89Backend {

bool isSm89Arch(int majorComputeCapability, int minorComputeCapability) {
  return majorComputeCapability == 8 && minorComputeCapability == 9;
}

static bool getBoolOpt(ConfigParser& cfg, const string& key, bool defaultValue) {
  return cfg.contains(key) ? cfg.getBool(key) : defaultValue;
}

struct PersistingL2Plan {
  float trunkHitRatio;
  float innerHitRatio;
};

static PersistingL2Plan reservePersistingL2(
  const ModelDesc& desc,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool useTrunk,
  bool useInner
) {
  int device = 0;
  int maxPersistingBytes = 0;
  CUDA_ERR("Sm89Model",cudaGetDevice(&device));
  CUDA_ERR("Sm89Model",cudaDeviceGetAttribute(
    &maxPersistingBytes, cudaDevAttrMaxPersistingL2CacheSize, device
  ));

  const size_t spatialRows = (size_t)maxBatchSize * nnXLen * nnYLen;
  const size_t trunkWindowBytes = useTrunk
    ? spatialRows * desc.trunk.trunkNumChannels * sizeof(half)
    : 0;
  const size_t innerWindowBytes = useInner
    ? spatialRows * desc.trunk.midNumChannels * sizeof(half)
    : 0;
  const size_t windowsPerStream = trunkWindowBytes + innerWindowBytes;
  const size_t requestedBytes = std::min(
    (size_t)maxPersistingBytes, 2 * windowsPerStream
  );
  CUDA_ERR("Sm89Model",cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, requestedBytes));

  size_t actualBytes = 0;
  CUDA_ERR("Sm89Model",cudaDeviceGetLimit(&actualBytes, cudaLimitPersistingL2CacheSize));
  const float hitRatio = std::min(
    1.0f, (float)((double)actualBytes / (double)(2 * windowsPerStream))
  );
  return PersistingL2Plan{
    useTrunk ? hitRatio : 0.0f,
    useInner ? hitRatio : 0.0f,
  };
}

Options parseOptions(ConfigParser& cfg) {
  Options o;
  o.enabled = getBoolOpt(cfg, "cudaSm89Backend", true);
  o.useForward = getBoolOpt(cfg, "cudaSm89Forward", true);
  o.useWideQKV = getBoolOpt(cfg, "cudaUseWideQKV", true);
  o.useWideFFN = getBoolOpt(cfg, "cudaUseWideFFN", true);
  o.useFusedResidual = getBoolOpt(cfg, "cudaUseFusedResidual", true);
  o.useRMSNormOpt = getBoolOpt(cfg, "cudaUseRMSNormOpt", true);
  o.useMatmulLt = getBoolOpt(cfg, "cudaUseMatmulLt", false);
  o.useFusedQKRoPE = getBoolOpt(cfg, "cudaUseFusedQKRoPE", false);
  o.usePrecomputedQKRoPE = getBoolOpt(cfg, "cudaUsePrecomputedQKRoPESm89", false);
  o.useQKVRoPEGemm = getBoolOpt(cfg, "cudaUseQKVRoPEGemmSm89", false);
  o.useSplitQKVRoPEGemm = getBoolOpt(cfg, "cudaUseSplitQKVRoPEGemmSm89", false);
  o.plainQKVVariant = cfg.contains("cudaPlainQKVVariantSm89") ?
    cfg.getInt("cudaPlainQKVVariantSm89",0,2) : 0;
  o.ropeBatchGroup = cfg.contains("cudaRoPEBatchGroupSm89") ? cfg.getInt("cudaRoPEBatchGroupSm89",1,13) : 1;
  o.useFlashAttention = getBoolOpt(cfg, "cudaUseFlashAttentionSm89", false);
  o.useFlashAttentionBoth16 = getBoolOpt(cfg, "cudaUseFlashAttentionBoth16Sm89", false);
  o.useDualGemmSwiGLU = getBoolOpt(cfg, "cudaUseDualGemmSwiGLUSm89", false);
  o.useDualGemmSwiGLUHalf2Tanh = getBoolOpt(cfg, "cudaUseDualGemmSwiGLUHalf2TanhSm89", false);
  o.useLinear2Gemm = getBoolOpt(cfg, "cudaUseLinear2GemmSm89", false);
  o.useOutProjGemm = getBoolOpt(cfg, "cudaUseOutProjGemmSm89", false);
  o.usePreConvGemm = getBoolOpt(cfg, "cudaUsePreConvGemmSm89", false);
  o.usePostConvGemm = getBoolOpt(cfg, "cudaUsePostConvGemmSm89", false);
  o.usePostConvBNSilu = getBoolOpt(cfg, "cudaUsePostConvBNSiluSm89", false);
  o.useLinear2PostBNSilu = getBoolOpt(cfg, "cudaUseLinear2PostBNSiluSm89", false);
  o.useBatchSharedRoPE = getBoolOpt(cfg, "cudaUseBatchSharedRoPE", false);
  o.useFusedFFN = getBoolOpt(cfg, "cudaUseFusedFFN", false);
  o.useInitialConvFrontend = getBoolOpt(cfg, "cudaUseInitialConvFrontend", false);
  o.useInitialGlobalMatMulAdd = getBoolOpt(cfg, "cudaUseInitialGlobalMatMulAdd", false);
  o.useFusedPolicyP1 = getBoolOpt(cfg, "cudaUseFusedPolicyP1", false);
  o.useHeadBNHalfToFloat = getBoolOpt(cfg, "cudaUseHeadBNHalfToFloat", false);
  o.useWideHeadProjection = getBoolOpt(cfg, "cudaUseWideHeadProjection", false);
  o.useExactMaskElision = getBoolOpt(cfg, "cudaUseExactMaskElisionSm89", false);
  o.useFusedValueTerminal = getBoolOpt(cfg, "cudaUseFusedValueTerminalSm89", false);
  o.usePersistingL2Trunk = getBoolOpt(cfg, "cudaUsePersistingL2Trunk", false);
  o.usePersistingL2Inner = getBoolOpt(cfg, "cudaUsePersistingL2Inner", false);
  o.useScaleBiasSiluVec8 = getBoolOpt(cfg, "cudaUseScaleBiasSiluVec8Sm89", false);
  o.useScaleBiasSiluVec4C384 = getBoolOpt(cfg, "cudaUseScaleBiasSiluVec4C384Sm89", false);
  o.shareModelWeights = getBoolOpt(cfg, "cudaShareModelWeights", false);
  return o;
}

Sm89Model::Sm89Model(
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
  cudaStream_t stream,
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
  forward(nullptr),
  forwardActive(false)
{
  if(officialApplyContext == NULL || officialApply == NULL || cudaHandles == NULL || desc == NULL)
    throw StringError("Sm89Model: null construction argument");
  if(options.useForward && Sm89Forward::supports(*desc, useFP16, useNHWC)) {
    const PersistingL2Plan persistingL2 =
      (options.usePersistingL2Trunk || options.usePersistingL2Inner)
      ? reservePersistingL2(
          *desc, maxBatchSize, nnXLen, nnYLen,
          options.usePersistingL2Trunk, options.usePersistingL2Inner
        )
      : PersistingL2Plan{0.0f, 0.0f};
    forward = std::make_unique<Sm89Forward>(
      desc, maxBatchSize, nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC, stream,
      options.useWideQKV, options.useWideFFN, options.useFusedResidual, options.useRMSNormOpt,
      options.useFusedQKRoPE, options.usePrecomputedQKRoPE, options.useQKVRoPEGemm,
      options.useSplitQKVRoPEGemm,
      options.plainQKVVariant,
      options.ropeBatchGroup, options.useFlashAttention, options.useFlashAttentionBoth16,
      options.useDualGemmSwiGLU, options.useDualGemmSwiGLUHalf2Tanh,
      options.useLinear2Gemm, options.useOutProjGemm, options.usePreConvGemm,
      options.usePostConvGemm, options.usePostConvBNSilu,
      options.useLinear2PostBNSilu,
      options.usePersistingL2Trunk,
      persistingL2.trunkHitRatio, options.usePersistingL2Inner,
      persistingL2.innerHitRatio, options.useScaleBiasSiluVec8,
      options.useScaleBiasSiluVec4C384,
      options.useInitialConvFrontend,
      options.useInitialGlobalMatMulAdd,
      options.useFusedPolicyP1,
      options.useHeadBNHalfToFloat,
      options.useWideHeadProjection,
      options.useExactMaskElision,
      options.useFusedValueTerminal,
      options.shareModelWeights
    );
    forwardActive = true;
  }
  // Stage 0 scaffold: apply() delegates to the official model until stages land.
}

Sm89Model::~Sm89Model() {
}

void Sm89Model::setLogger(Logger* logger_) {
  logger = logger_;
}

void Sm89Model::apply(
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

  if(forwardActive) {
    forward->apply(
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
    return;
  }

  if(!loggedFallback) {
    if(logger != NULL)
      logger->write("SM89 backend: stage-0 official fallback active (rebuild scaffold)");
    loggedFallback = true;
  }

  // Stage 0: bit-identical delegation to the official backend. Once a stage lands, this becomes
  // the SM89-specific forward path with per-stage fallback where shapes/precision are unsupported.
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

} // namespace Sm89Backend
