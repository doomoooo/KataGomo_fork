#include "../neuralnet/cudabackend_sm120.h"

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"
#include "fa4_aot/fa4_sm120_b13.h"

#include "../core/global.h"
#include "../core/logger.h"

#include <cmath>

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

struct Sm120Model::Fa4State {
  fa4_Kernel_Module_t module;
  bool loaded;

  Fa4State() : module(), loaded(false) {}
  ~Fa4State() {
    if(loaded) {
      // cudaLibraryUnload is not guaranteed to succeed after a device reset;
      // this runs in the normal destructor path before NeuralNet::globalCleanup.
      cudaLibraryUnload(module.module);
    }
  }
};

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
  fa4State(nullptr)
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

  if(fa4State == NULL)
    fa4State = std::make_unique<Fa4State>();
  if(!fa4State->loaded) {
    fa4_Kernel_Module_Load(&fa4State->module);
    fa4State->loaded = true;
  }

  fa4_Tensor_mQ_t tq = {qBuf, {batchSize, seqLen, numHeads, qHeadDim}, {seqLen * numHeads * qHeadDim, numHeads * qHeadDim, qHeadDim}};
  fa4_Tensor_mK_t tk = {kBuf, {batchSize, seqLen, numKVHeads, qHeadDim}, {seqLen * numKVHeads * qHeadDim, numKVHeads * qHeadDim, qHeadDim}};
  fa4_Tensor_mV_t tv = {vBuf, {batchSize, seqLen, numKVHeads, vHeadDim}, {seqLen * numKVHeads * vHeadDim, numKVHeads * vHeadDim, vHeadDim}};
  fa4_Tensor_mO_t to = {attnOutBuf, {batchSize, seqLen, numHeads, vHeadDim}, {seqLen * numHeads * vHeadDim, numHeads * vHeadDim, vHeadDim}};

  float scale = 1.0f / std::sqrt((float)qHeadDim);
  int32_t ret = cute_dsl_fa4_wrapper(&fa4State->module, &tq, &tk, &tv, &to, scale, stream);

  if(ret != 0) {
    if(logger != NULL)
      logger->write("SM120 backend: FA4 AOT attention launch failed, falling back to official path");
    return false;
  }
  CUDA_ERR("Sm120Attention", cudaPeekAtLastError());

  if(!loggedFa4) {
    if(logger != NULL)
      logger->write("SM120 backend: FA4 AOT attention active (FP32 accumulator; both16 pending)");
    loggedFa4 = true;
  }
  return true;
}

} // namespace Sm120Backend
