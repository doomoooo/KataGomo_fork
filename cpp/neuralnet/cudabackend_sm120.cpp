#include "../neuralnet/cudabackend_sm120.h"

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"

#include "../core/global.h"
#include "../core/logger.h"

using namespace std;

namespace Sm120Backend {

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
  o.useQKVGemmAot = getBoolOpt(cfg, "cudaUseQKVGemmAot", true);
  o.useQKVGemmRopeAot = getBoolOpt(cfg, "cudaUseQKVGemmRopeAot", false);
  o.useFusedQKRoPE = getBoolOpt(cfg, "cudaUseFusedQKRoPE", true);
  o.useBatchSharedRoPE = getBoolOpt(cfg, "cudaUseBatchSharedRoPE", true);
  o.useBatchSharedRoPEUnroll19 = getBoolOpt(cfg, "cudaUseBatchSharedRoPEUnroll19", true);
  o.useBatchSharedRoPETwoWay = getBoolOpt(cfg, "cudaUseBatchSharedRoPETwoWay", false);
  o.useFusedResidual = getBoolOpt(cfg, "cudaUseFusedResidual", true);
  o.useProjectionGemmLt = getBoolOpt(cfg, "cudaUseProjectionGemmLt", false);
  o.useFusedFFN = getBoolOpt(cfg, "cudaUseFusedFFN", true);
  o.useFusedRMSNormFFN = getBoolOpt(cfg, "cudaUseFusedRMSNormFFN", false);
  o.useRMSNormQKVGemmAot = getBoolOpt(cfg, "cudaUseRMSNormQKVGemmAot", false);
  o.useGraph = getBoolOpt(cfg, "cudaUseGraph", false);
  o.usePersistingL2Trunk = getBoolOpt(cfg, "cudaUsePersistingL2Trunk", true);
  o.usePersistingL2Inner = getBoolOpt(cfg, "cudaUsePersistingL2Inner", true);
  o.useOuterProjectionAot = getBoolOpt(cfg, "cudaUseOuterProjectionAot", true);
  o.shareModelWeights = getBoolOpt(cfg, "cudaShareModelWeights", true);
  o.shareWideQKVWeights = getBoolOpt(cfg, "cudaShareWideQKVWeights", false);
  o.shareOuterProjectionWeights = getBoolOpt(cfg, "cudaShareOuterProjectionWeights", false);
  o.useInitialConvFrontend = getBoolOpt(cfg, "cudaUseInitialConvFrontend", true);
  o.useInitialConvBiasFrontend = getBoolOpt(cfg, "cudaUseInitialConvBiasFrontend", false);
  o.useInitialGlobalMatMulAdd = getBoolOpt(cfg, "cudaUseInitialGlobalMatMulAdd", true);
  o.useFusedPolicyP1 = getBoolOpt(cfg, "cudaUseFusedPolicyP1", true);
  o.useHeadBNHalfToFloat = getBoolOpt(cfg, "cudaUseHeadBNHalfToFloat", true);
  o.useWideHeadProjection = getBoolOpt(cfg, "cudaUseWideHeadProjection", true);
  return o;
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
  loggedFallback(false)
{
  if(officialApplyContext == NULL || officialApply == NULL || cudaHandles == NULL || desc == NULL)
    throw StringError("Sm120Model: null construction argument");
  // Stage 0 scaffold: apply() delegates to the official model until stages land.
}

Sm120Model::~Sm120Model() {
}

void Sm120Model::setLogger(Logger* logger_) {
  logger = logger_;
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

} // namespace Sm120Backend
