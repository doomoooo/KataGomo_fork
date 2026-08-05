#include "../neuralnet/cudabackend_sm89.h"

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"

#include "../core/global.h"
#include "../core/logger.h"

using namespace std;

namespace Sm89Backend {

bool isSm89Arch(int majorComputeCapability, int minorComputeCapability) {
  return majorComputeCapability == 8 && minorComputeCapability == 9;
}

static bool getBoolOpt(ConfigParser& cfg, const string& key, bool defaultValue) {
  return cfg.contains(key) ? cfg.getBool(key) : defaultValue;
}

Options parseOptions(ConfigParser& cfg) {
  Options o;
  o.enabled = getBoolOpt(cfg, "cudaSm89Backend", true);
  o.useWideQKV = getBoolOpt(cfg, "cudaUseWideQKV", false);
  o.useWideFFN = getBoolOpt(cfg, "cudaUseWideFFN", false);
  o.useFusedResidual = getBoolOpt(cfg, "cudaUseFusedResidual", false);
  o.useMatmulLt = getBoolOpt(cfg, "cudaUseMatmulLt", false);
  o.useFusedQKRoPE = getBoolOpt(cfg, "cudaUseFusedQKRoPE", false);
  o.useBatchSharedRoPE = getBoolOpt(cfg, "cudaUseBatchSharedRoPE", false);
  o.useFusedFFN = getBoolOpt(cfg, "cudaUseFusedFFN", false);
  o.useInitialConvFrontend = getBoolOpt(cfg, "cudaUseInitialConvFrontend", false);
  o.useInitialGlobalMatMulAdd = getBoolOpt(cfg, "cudaUseInitialGlobalMatMulAdd", false);
  o.useFusedPolicyP1 = getBoolOpt(cfg, "cudaUseFusedPolicyP1", false);
  o.useHeadBNHalfToFloat = getBoolOpt(cfg, "cudaUseHeadBNHalfToFloat", false);
  o.useWideHeadProjection = getBoolOpt(cfg, "cudaUseWideHeadProjection", false);
  o.usePersistingL2Trunk = getBoolOpt(cfg, "cudaUsePersistingL2Trunk", false);
  o.usePersistingL2Inner = getBoolOpt(cfg, "cudaUsePersistingL2Inner", false);
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
    throw StringError("Sm89Model: null construction argument");
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
