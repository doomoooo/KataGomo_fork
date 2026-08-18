#include "../neuralnet/cudabackend_sm103.h"
#include "../neuralnet/cudnn_oss_b29_aot_bridge.h"
#include "../neuralnet/fa4_sm103_b29_aot_bridge.h"

#include "../core/global.h"
#include "../core/logger.h"

#include <cstdint>

using namespace std;

namespace Sm103Backend {

constexpr const char* CudnnOssB29NoAb12FfnTactic =
  "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-no-ab12-b29";
constexpr const char* Fa4Sm103B29AttentionTactic =
  "fa4-main-sm103a-b29-s361-h12-d32-m128n128-q2-kv24-fp32";

namespace {

constexpr int B29Rows = 10469;
constexpr int InputChannels = 384;
constexpr int PackedChannels = 2304;
constexpr int OutputChannels = 1152;
constexpr int PairChannels = 32;
constexpr int Sm103CudnnOssFfnPackGate = 0;
constexpr int Sm103CudnnOssFfnPackLinear1 = 1;
static_assert(
  Sm103CudnnOssFfnPackLinear1-Sm103CudnnOssFfnPackGate == 1,
  "packed FFN projection pair order must remain gate then linear1"
);

void throwCudaError(const char* operation, cudaError_t error) {
  throw StringError(
    string(operation) + " failed: " + cudaGetErrorString(error)
  );
}

} // namespace

bool isSm103Arch(int majorComputeCapability, int minorComputeCapability) {
  return majorComputeCapability == 10 && minorComputeCapability == 3;
}

static bool getBoolOpt(ConfigParser& cfg, const string& key, bool defaultValue) {
  return cfg.contains(key) ? cfg.getBool(key) : defaultValue;
}

Options parseOptions(ConfigParser& cfg) {
  Options options;
  options.enabled = getBoolOpt(cfg, "cudaSm103Backend", false);
  options.reusePortableTactics =
    getBoolOpt(cfg, "cudaSm103ReusePortableTactics", false);
  options.dualFfnTactic = cfg.contains("cudaSm103DualFFNTactic") ?
    cfg.getString("cudaSm103DualFFNTactic") : "disabled";
  options.attentionTactic = cfg.contains("cudaSm103AttentionTactic") ?
    cfg.getString("cudaSm103AttentionTactic") : "disabled";
  options.qkvAuxTactic = cfg.contains("cudaSm103QKVAuxTactic") ?
    cfg.getString("cudaSm103QKVAuxTactic") : "disabled";
  options.allowOfficialForwardScaffold =
    getBoolOpt(cfg, "cudaSm103AllowOfficialForwardScaffold", false);
  if(options.allowOfficialForwardScaffold && !options.enabled)
    throw StringError(
      "cudaSm103AllowOfficialForwardScaffold requires cudaSm103Backend=true"
    );
  if(options.reusePortableTactics && !options.enabled)
    throw StringError(
      "cudaSm103ReusePortableTactics requires cudaSm103Backend=true"
    );
  if(options.dualFfnTactic != "disabled" && !options.enabled)
    throw StringError(
      "cudaSm103DualFFNTactic requires cudaSm103Backend=true"
    );
  if(options.attentionTactic != "disabled" && !options.enabled)
    throw StringError(
      "cudaSm103AttentionTactic requires cudaSm103Backend=true"
    );
  if(options.qkvAuxTactic != "disabled" && !options.enabled)
    throw StringError(
      "cudaSm103QKVAuxTactic requires cudaSm103Backend=true"
    );
  if(options.qkvAuxTactic != "disabled" && !options.reusePortableTactics)
    throw StringError(
      "cudaSm103QKVAuxTactic requires cudaSm103ReusePortableTactics=true"
    );
  if(options.dualFfnTactic != "disabled" &&
     options.dualFfnTactic != CudnnOssB29NoAb12FfnTactic)
    throw StringError(
      string("cudaSm103DualFFNTactic must be disabled or ") +
      CudnnOssB29NoAb12FfnTactic
    );
  if(options.attentionTactic != "disabled" &&
     options.attentionTactic != Fa4Sm103B29AttentionTactic)
    throw StringError(
      string("cudaSm103AttentionTactic must be disabled or ") +
      Fa4Sm103B29AttentionTactic
    );
  if(options.qkvAuxTactic != "disabled" &&
     options.qkvAuxTactic != B29QKVAuxTactic)
    throw StringError(
      string("cudaSm103QKVAuxTactic must be disabled or ") +
      B29QKVAuxTactic
    );
  return options;
}

Sm103Model::Sm103Model(
  void* officialApplyContext_,
  OfficialApplyFn officialApply_,
  CudaHandles* cudaHandles,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool inputsUseNHWC,
  bool useFP16,
  bool useNHWC,
  int majorComputeCapability,
  int minorComputeCapability,
  const Options& options_
) :
  officialApplyContext(officialApplyContext_),
  officialApply(officialApply_),
  options(options_),
  logger(NULL),
  loggedScaffold(false),
  loggedCudnnOssFfn(false),
  loggedFa4Attention(false),
  device(-1),
  cudnnOssB29Context(nullptr),
  fa4Sm103B29Context(nullptr),
  hasLaunchedFfn(false),
  lastFfnStream(nullptr),
  hasLaunchedAttention(false),
  lastAttentionStream(nullptr)
{
  (void)cudaHandles;
  if(officialApplyContext == NULL || officialApply == NULL)
    throw StringError("Sm103Model: null official-forward adapter");
  if(nnXLen != 19 || nnYLen != 19 || !inputsUseNHWC || !useFP16 || !useNHWC)
    throw StringError("SM103 scaffold requires exact 19x19 FP16 NHWC inference");
  if(!isSm103Arch(majorComputeCapability,minorComputeCapability))
    throw StringError("SM103 optimized backend requires exact CC10.3");
  if(!options.allowOfficialForwardScaffold &&
     !options.reusePortableTactics &&
     options.dualFfnTactic == "disabled" &&
     options.attentionTactic == "disabled")
    throw StringError(
      "SM103 optimized backend has no registered tactics yet; "
      "cudaSm103AllowOfficialForwardScaffold=true is required for wiring tests"
    );

  if(options.dualFfnTactic != "disabled" ||
     options.attentionTactic != "disabled") {
    cudaError_t cudaStatus = cudaGetDevice(&device);
    if(cudaStatus != cudaSuccess)
      throwCudaError("SM103 native AOT cudaGetDevice",cudaStatus);
  }

  if(options.dualFfnTactic == CudnnOssB29NoAb12FfnTactic) {
    if(maxBatchSize != 29 || !useFP16 ||
       !isSm103Arch(majorComputeCapability,minorComputeCapability))
      throw StringError(
        "SM103 cuDNN OSS no-AB12 fast candidate requires exact B29/CC10.3 "
        "R10469/K384/N2304->1152 FP16"
      );
    int32_t status = KATAGO_CUDNN_OSS_B29_MODULE_LOAD_FAILED;
    cudnnOssB29Context = katagoCudnnOssB29Create(device,&status);
    if(cudnnOssB29Context == nullptr || status != KATAGO_CUDNN_OSS_B29_SUCCESS)
      throw StringError(
        "SM103 cuDNN OSS no-AB12 fast AOT module unavailable, status=" +
        Global::intToString(status)
      );
  }
  if(options.attentionTactic == Fa4Sm103B29AttentionTactic) {
    if(maxBatchSize != 29 || !useFP16 ||
       !isSm103Arch(majorComputeCapability,minorComputeCapability))
      throw StringError(
        "SM103 FA4 native candidate requires exact B29/CC10.3 "
        "S361/H12/D32 planar FP16"
      );
    int32_t status = KATAGO_FA4_SM103_B29_MODULE_LOAD_FAILED;
    fa4Sm103B29Context = katagoFa4Sm103B29Create(device,&status);
    if(fa4Sm103B29Context == nullptr ||
       status != KATAGO_FA4_SM103_B29_SUCCESS) {
      katagoFa4Sm103B29Destroy(fa4Sm103B29Context);
      fa4Sm103B29Context = nullptr;
      katagoCudnnOssB29Destroy(cudnnOssB29Context);
      cudnnOssB29Context = nullptr;
      throw StringError(
        "SM103 FA4 native AOT module unavailable, status=" +
        Global::intToString(status)
      );
    }
  }
}

Sm103Model::~Sm103Model() {
  int previousDevice = device;
  if(device >= 0 && cudaGetDevice(&previousDevice) == cudaSuccess &&
     previousDevice != device)
    cudaSetDevice(device);
  if(hasLaunchedFfn)
    cudaStreamSynchronize(lastFfnStream);
  if(hasLaunchedAttention)
    cudaStreamSynchronize(lastAttentionStream);
  for(const auto& entry : packedFfnWeights)
    cudaFree(entry.second);
  packedFfnWeights.clear();
  katagoCudnnOssB29Destroy(cudnnOssB29Context);
  katagoFa4Sm103B29Destroy(fa4Sm103B29Context);
  if(device >= 0 && previousDevice != device)
    cudaSetDevice(previousDevice);
}

bool Sm103Model::attention(
  const void* q,
  const void* k,
  const void* v,
  bool packedQKV,
  const void* mask,
  void* output,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFP16,
  cudaStream_t stream
) {
  if(options.attentionTactic != Fa4Sm103B29AttentionTactic)
    return false;
  if(batchSize != 29 || seqLen != 361 || numHeads != 12 ||
     numKVHeads != 12 || qHeadDim != 32 || vHeadDim != 32 ||
     !usingFP16 || packedQKV || mask != nullptr)
    return false;
  if(q == nullptr || k == nullptr || v == nullptr || output == nullptr)
    throw StringError("SM103 FA4 native attention received a null buffer");

  constexpr float scale = 0.1767766952966369f;
  int32_t status = katagoFa4Sm103B29Launch(
    fa4Sm103B29Context,q,k,v,mask,output,scale,stream,
    batchSize,seqLen,numHeads,numKVHeads,qHeadDim,vHeadDim,
    packedQKV ? 1 : 0,usingFP16 ? 1 : 0
  );
  if(status != KATAGO_FA4_SM103_B29_SUCCESS)
    throw StringError(
      "SM103 FA4 native attention launch failed, status=" +
      Global::intToString(status)
    );
  hasLaunchedAttention = true;
  lastAttentionStream = stream;
  if(!loggedFa4Attention) {
    if(logger != nullptr)
      logger->write(
        "SM103 backend: FA4 native attention active, tactic=" +
        string(Fa4Sm103B29AttentionTactic)
      );
    loggedFa4Attention = true;
  }
  return true;
}

void Sm103Model::setLogger(Logger* logger_) {
  logger = logger_;
}

bool Sm103Model::fusedFFN(
  const void* linear1Weights,
  const void* linearGateWeights,
  const void* input,
  void* ab12Scratch,
  void* output,
  int matBatchSize,
  int inputChannels,
  int ffnChannels,
  bool usingFP16,
  cudaStream_t stream
) {
  if(options.dualFfnTactic != CudnnOssB29NoAb12FfnTactic)
    return false;
  if(matBatchSize != B29Rows || inputChannels != InputChannels ||
     ffnChannels != OutputChannels || !usingFP16)
    return false;
  if(linear1Weights == nullptr || linearGateWeights == nullptr ||
     input == nullptr || ab12Scratch == nullptr || output == nullptr)
    throw StringError("SM103 cuDNN OSS fused FFN received a null buffer");

  // stream==0 is CUDA's valid legacy/default stream and must remain accepted.
  const auto key = make_pair(linearGateWeights,linear1Weights);
  auto found = packedFfnWeights.find(key);
  void* packedWeights = nullptr;
  if(found == packedFfnWeights.end()) {
    constexpr size_t elementBytes = sizeof(uint16_t);
    constexpr size_t sourceRowBytes = OutputChannels * elementBytes;
    constexpr size_t packedRowBytes = PackedChannels * elementBytes;
    cudaError_t status = cudaMalloc(
      &packedWeights,(size_t)InputChannels * packedRowBytes
    );
    if(status != cudaSuccess)
      throwCudaError("SM103 cuDNN OSS packed-weight cudaMalloc",status);
    for(int channel = 0; channel < OutputChannels; channel += PairChannels) {
      constexpr size_t chunkBytes = PairChannels * elementBytes;
      const size_t sourceOffset = (size_t)channel * elementBytes;
      const size_t pairOffset =
        ((size_t)(channel / PairChannels) * 2 + Sm103CudnnOssFfnPackGate) *
        chunkBytes;
      status = cudaMemcpy2DAsync(
        (char*)packedWeights + pairOffset,packedRowBytes,
        (const char*)linearGateWeights + sourceOffset,sourceRowBytes,
        chunkBytes,InputChannels,cudaMemcpyDeviceToDevice,stream
      );
      if(status == cudaSuccess)
        status = cudaMemcpy2DAsync(
          (char*)packedWeights + pairOffset + chunkBytes,
          packedRowBytes,
          (const char*)linear1Weights + sourceOffset,sourceRowBytes,
          chunkBytes,InputChannels,cudaMemcpyDeviceToDevice,stream
        );
      if(status != cudaSuccess) {
        cudaFree(packedWeights);
        throwCudaError("SM103 cuDNN OSS packed-weight copy",status);
      }
    }
    packedFfnWeights.emplace(key,packedWeights);
  }
  else
    packedWeights = found->second;

  int32_t status = katagoCudnnOssB29Launch(
    cudnnOssB29Context,input,packedWeights,ab12Scratch,output,1.0f,stream,
    B29Rows,InputChannels,PackedChannels,OutputChannels,1
  );
  if(status != KATAGO_CUDNN_OSS_B29_SUCCESS)
    throw StringError(
      "SM103 cuDNN OSS fused FFN launch failed, status=" +
      Global::intToString(status)
    );
  hasLaunchedFfn = true;
  lastFfnStream = stream;
  if(!loggedCudnnOssFfn) {
    if(logger != nullptr)
      logger->write(
        "SM103 backend: cuDNN OSS no-AB12 fast fused FFN active, tactic=" +
        string(CudnnOssB29NoAb12FfnTactic)
      );
    loggedCudnnOssFfn = true;
  }
  return true;
}

bool applyFusedFFN(
  void* context,
  const void* linear1Weights,
  const void* linearGateWeights,
  const void* input,
  void* ab12Scratch,
  void* output,
  int matBatchSize,
  int inputChannels,
  int ffnChannels,
  bool usingFP16,
  cudaStream_t stream
) {
  if(context == nullptr)
    return false;
  return ((Sm103Model*)context)->fusedFFN(
    linear1Weights,linearGateWeights,input,ab12Scratch,output,
    matBatchSize,inputChannels,ffnChannels,usingFP16,stream
  );
}

bool applyAttention(
  void* context,
  const void* q,
  const void* k,
  const void* v,
  bool packedQKV,
  const void* mask,
  void* output,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFP16,
  cudaStream_t stream
) {
  if(context == nullptr)
    return false;
  return ((Sm103Model*)context)->attention(
    q,k,v,packedQKV,mask,output,batchSize,seqLen,numHeads,numKVHeads,
    qHeadDim,vHeadDim,usingFP16,stream
  );
}

void Sm103Model::apply(
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
) {
  if(!requireExactNNLen)
    throw StringError("SM103 scaffold supports only exact 19x19 inference");
  if(!loggedScaffold) {
    if(logger != NULL)
      logger->write(options.reusePortableTactics ?
        "SM103 backend: official forward adapter active with portable CUDA tactic hooks" :
        "SM103 backend scaffold: official forward adapter active; no optimized tactics launched");
    loggedScaffold = true;
  }
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

} // namespace Sm103Backend
