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

static string getTacticOpt(
  ConfigParser& cfg,
  const string& key,
  std::initializer_list<const char*> allowed
) {
  const string value = cfg.contains(key) ? cfg.getString(key) : "disabled";
  for(const char* candidate : allowed) {
    if(value == candidate)
      return value;
  }
  throw StringError("unknown " + key + ": " + value);
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
  bool useInner,
  float requestedHitRatio
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
  const float availableHitRatio = std::min(
    1.0f, (float)((double)actualBytes / (double)(2 * windowsPerStream))
  );
  const float hitRatio = std::min(requestedHitRatio, availableHitRatio);
  return PersistingL2Plan{
    useTrunk ? hitRatio : 0.0f,
    useInner ? hitRatio : 0.0f,
  };
}

Options parseOptions(ConfigParser& cfg) {
  Options o;
  o.enabled = getBoolOpt(cfg, "cudaSm89Backend", true);
  o.useForward = getBoolOpt(cfg, "cudaSm89Forward", true);
  o.useWideQKV = getBoolOpt(cfg, "cudaUseWideQKV", false);
  o.useWideFFN = getBoolOpt(cfg, "cudaUseWideFFN", false);
  o.useFusedResidual = getBoolOpt(cfg, "cudaUseFusedResidual", false);
  o.useRMSNormOpt = getBoolOpt(cfg, "cudaUseRMSNormOpt", false);
  o.rmsNormRowsPerBlock = cfg.contains("cudaRMSNormRowsPerBlockSm89") ?
    cfg.getInt("cudaRMSNormRowsPerBlockSm89",4,8) : 4;
  if(o.rmsNormRowsPerBlock != 4 && o.rmsNormRowsPerBlock != 8)
    throw StringError("cudaRMSNormRowsPerBlockSm89 must be 4 or 8");
  o.useMatmulLt = getBoolOpt(cfg, "cudaUseMatmulLt", false);
  o.useFusedQKRoPE = getBoolOpt(cfg, "cudaUseFusedQKRoPE", false);
  o.usePrecomputedQKRoPE = getBoolOpt(cfg, "cudaUsePrecomputedQKRoPESm89", false);
  o.useQKVRoPEGemm = getBoolOpt(cfg, "cudaUseQKVRoPEGemmSm89", false);
  o.useSplitQKVRoPEGemm = getBoolOpt(cfg, "cudaUseSplitQKVRoPEGemmSm89", false);
  o.plainQKVVariant = cfg.contains("cudaPlainQKVVariantSm89") ?
    cfg.getInt("cudaPlainQKVVariantSm89",0,1) : 0;
  o.ropeBatchGroup = cfg.contains("cudaRoPEBatchGroupSm89") ?
    cfg.getInt("cudaRoPEBatchGroupSm89",1,32) : 1;
  const string flashTactic = cfg.contains("cudaFlashAttentionTacticSm89") ?
    cfg.getString("cudaFlashAttentionTacticSm89") : "disabled";
  o.flashAttentionTacticName = flashTactic;
  if(flashTactic == "disabled") o.flashAttentionTactic = 0;
  else if(flashTactic == "d32-m128-n112-w4-pack0-fp32") o.flashAttentionTactic = 1;
  else if(flashTactic == "d32-m128-n96-w4-pack0-fp32") o.flashAttentionTactic = 2;
  else if(flashTactic == "d32-m64-n96-w4-pack1-fp32") o.flashAttentionTactic = 3;
  else if(flashTactic == "d32-m64-n96-w4-pack0-fp32") o.flashAttentionTactic = 4;
  else if(flashTactic == "d32-m64-n96-w4-pack0-both16") o.flashAttentionTactic = 5;
  else throw StringError("unknown cudaFlashAttentionTacticSm89: " + flashTactic);
  o.dualFfnCutlassTactic = getTacticOpt(
    cfg, "cudaDualFfnCutlassTacticSm89", {
      "disabled",
      "m128-n64-k32-w64-n32-s3-sw2-exp",
      "m128-n64-k32-w64-n32-s3-sw4-exp",
      "m128-n64-k32-w64-n32-s3-sw2-tanh-half2",
    });
  o.linear2CutlassTactic = getTacticOpt(
    cfg, "cudaLinear2CutlassTacticSm89", {
      "disabled",
      "m128-n128-k32-w64-n32-s3-sw1",
      "m128-n128-k32-w64-n32-s4-sw1",
      "m128-n128-k32-w64-n64-s3-sw1",
      "m128-n128-k32-w64-n64-s4-sw1",
      "m128-n128-k32-w64-n64-s5-sw1",
      "m128-n128-k32-w64-n64-s6-sw1",
    });
  o.outProjCutlassTactic = getTacticOpt(
    cfg, "cudaOutProjCutlassTacticSm89", {
      "disabled",
      "m128-n128-k32-w64-n32-s2-sw1",
      "m128-n128-k32-w64-n32-s3-sw1",
      "m128-n128-k32-w64-n32-s4-sw1",
      "m128-n128-k32-w64-n64-s3-sw1",
      "m128-n128-k32-w64-n64-s4-sw1",
    });
  o.preConvCutlassTactic = getTacticOpt(
    cfg, "cudaPreConvCutlassTacticSm89", {
      "disabled",
      "m128-n128-k32-w64-n32-s3-sw1",
      "m128-n128-k32-w64-n32-s4-sw1",
      "m128-n128-k32-w64-n64-s3-sw1",
      "m128-n128-k32-w64-n64-s4-sw1",
      "m128-n128-k32-w64-n64-s5-sw1",
      "m128-n128-k32-w64-n64-s6-sw1",
    });
  o.postConvCutlassTactic = getTacticOpt(
    cfg, "cudaPostConvCutlassTacticSm89", {
      "disabled",
      "m128-n128-k32-w64-n32-s2-sw1",
      "m128-n128-k32-w64-n32-s3-sw1",
      "m128-n128-k32-w64-n32-s3-sw2",
      "m128-n128-k32-w64-n64-s3-sw1",
      "m128-n128-k32-w64-n64-s3-sw2",
      "m128-n128-k32-w64-n64-s3-sw4",
      "m128-n256-k32-w64-n64-s2-sw2",
      "m256-n128-k32-w64-n64-s2-sw1",
      "m256-n128-k32-w64-n64-s2-sw2",
    });
  o.useDualGemmSwiGLU = o.dualFfnCutlassTactic != "disabled";
  o.useLinear2Gemm = o.linear2CutlassTactic != "disabled";
  o.useOutProjGemm = o.outProjCutlassTactic != "disabled";
  o.usePreConvGemm = o.preConvCutlassTactic != "disabled";
  o.usePostConvGemm = o.postConvCutlassTactic != "disabled";
  o.usePostConvBNSilu = getBoolOpt(cfg, "cudaUsePostConvBNSiluSm89", false);
  o.useLinear2PostBNSilu = getBoolOpt(cfg, "cudaUseLinear2PostBNSiluSm89", false);
  o.useBatchSharedRoPE = getBoolOpt(cfg, "cudaUseBatchSharedRoPE", false);
  o.useFusedFFN = getBoolOpt(cfg, "cudaUseFusedFFN", false);
  o.useInitialConvFrontend = getBoolOpt(cfg, "cudaUseInitialConvFrontend", false);
  o.useInitialGlobalMatMulAdd = getBoolOpt(cfg, "cudaUseInitialGlobalMatMulAdd", false);
  o.policyP1RowsPerBlock = cfg.contains("cudaPolicyP1RowsPerBlockSm89") ?
    cfg.getInt("cudaPolicyP1RowsPerBlockSm89",0,5) : 0;
  if(o.policyP1RowsPerBlock != 0 && o.policyP1RowsPerBlock != 1 &&
     o.policyP1RowsPerBlock != 5)
    throw StringError("cudaPolicyP1RowsPerBlockSm89 must be 0, 1, or 5");
  o.useHeadBNHalfToFloat = getBoolOpt(cfg, "cudaUseHeadBNHalfToFloat", false);
  o.useWideHeadProjection = getBoolOpt(cfg, "cudaUseWideHeadProjection", false);
  o.useFusedValueTerminal = getBoolOpt(cfg, "cudaUseFusedValueTerminalSm89", false);
  o.usePersistingL2Trunk = getBoolOpt(cfg, "cudaUsePersistingL2Trunk", false);
  o.usePersistingL2Inner = getBoolOpt(cfg, "cudaUsePersistingL2Inner", false);
  o.persistingL2HitRatio = cfg.contains("cudaPersistingL2HitRatioSm89") ?
    cfg.getFloat("cudaPersistingL2HitRatioSm89",0.0f,1.0f) : 1.0f;
  o.useScaleBiasSiluVec8 = getBoolOpt(cfg, "cudaUseScaleBiasSiluVec8Sm89", false);
  o.useScaleBiasSiluVec8C384 = getBoolOpt(cfg, "cudaUseScaleBiasSiluVec8C384Sm89", false);
  o.useScaleBiasSiluVec4C384 = getBoolOpt(cfg, "cudaUseScaleBiasSiluVec4C384Sm89", false);
  o.shareModelWeights = getBoolOpt(cfg, "cudaShareModelWeights", false);
  o.dualFfnAotTactic = cfg.contains("cudaFusedFFNAotTacticSm89") ?
    cfg.getString("cudaFusedFFNAotTacticSm89") : "disabled";
  o.linear2AotTactic = cfg.contains("cudaLinear2AotTacticSm89") ?
    cfg.getString("cudaLinear2AotTacticSm89") : "disabled";
  o.serverThreads = cfg.contains("numNNServerThreadsPerModel") ?
    cfg.getInt("numNNServerThreadsPerModel",1,1024) : 1;
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
  loggedActiveTactics(),
  forward(nullptr),
  forwardActive(false)
{
  if(officialApplyContext == NULL || officialApply == NULL || cudaHandles == NULL || desc == NULL)
    throw StringError("Sm89Model: null construction argument");
  if(nnXLen != 19 || nnYLen != 19)
    throw StringError("SM89 optimized backend supports only exact 19x19 inference");
  if(options.useForward && Sm89Forward::supports(*desc, useFP16, useNHWC)) {
    const PersistingL2Plan persistingL2 =
      (options.usePersistingL2Trunk || options.usePersistingL2Inner)
      ? reservePersistingL2(
          *desc, maxBatchSize, nnXLen, nnYLen,
          options.usePersistingL2Trunk, options.usePersistingL2Inner,
          options.persistingL2HitRatio
        )
      : PersistingL2Plan{0.0f, 0.0f};
    forward = std::make_unique<Sm89Forward>(
      desc, maxBatchSize, nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC, stream,
      options.useWideQKV, options.useWideFFN, options.useFusedResidual,
      options.useRMSNormOpt, options.rmsNormRowsPerBlock,
      options.useFusedQKRoPE, options.usePrecomputedQKRoPE, options.useQKVRoPEGemm,
      options.useSplitQKVRoPEGemm,
      options.plainQKVVariant,
      options.ropeBatchGroup, options.flashAttentionTactic,
      options.useDualGemmSwiGLU,
      options.useLinear2Gemm, options.useOutProjGemm, options.usePreConvGemm,
      options.usePostConvGemm, options.usePostConvBNSilu,
      options.useLinear2PostBNSilu,
      options.usePersistingL2Trunk,
      persistingL2.trunkHitRatio, options.usePersistingL2Inner,
      persistingL2.innerHitRatio, options.useScaleBiasSiluVec8,
      options.useScaleBiasSiluVec8C384,
      options.useScaleBiasSiluVec4C384,
      options.useInitialConvFrontend,
      options.useInitialGlobalMatMulAdd,
      options.policyP1RowsPerBlock,
      options.useHeadBNHalfToFloat,
      options.useWideHeadProjection,
      options.useFusedValueTerminal,
      options.dualFfnAotTactic,
      options.linear2AotTactic,
      options.dualFfnCutlassTactic,
      options.linear2CutlassTactic,
      options.outProjCutlassTactic,
      options.preConvCutlassTactic,
      options.postConvCutlassTactic,
      options.serverThreads,
      options.shareModelWeights
    );
    forwardActive = true;
  }
  // Unsupported model shapes retain the official compatibility path.
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
  size_t workspaceBytes,
  cudaEvent_t inputConsumedEvent,
  cudaEvent_t outputConsumedEvent
) {
  (void)cudaHandles_;
  (void)scratch;
  (void)batchSize;
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

  if(!requireExactNNLen)
    throw StringError("SM89 optimized backend supports only exact 19x19 inference");

  if(forwardActive) {
    forward->apply(
      batchSize,
      inputBuf,
      inputGlobalBuf,
      inputMetaBuf,
      policyPassBuf,
      policyBuf,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      workspaceBuf,
      workspaceBytes,
      inputConsumedEvent,
      outputConsumedEvent
    );
    for(string key : forward->getActiveTactics()) {
      if(key == "cudaFlashAttentionTacticSm89")
        key += "=" + options.flashAttentionTacticName;
      if(loggedActiveTactics.insert(key).second && logger != NULL)
        logger->write("SM89 backend: runtime tactic active: " + key);
    }
    return;
  }

  if(!loggedFallback) {
    if(logger != NULL)
      logger->write("SM89 backend: unsupported-model official fallback active");
    loggedFallback = true;
  }

  // The specialized forward rejected this model at construction, so use the
  // explicit whole-model compatibility path.
  if(inputConsumedEvent != nullptr || outputConsumedEvent != nullptr)
    throw StringError("SM89 event-gated inference requires the custom forward path");
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
