#include "../neuralnet/cudabackend_sm103.h"

#include "../core/global.h"
#include "../core/logger.h"

using namespace std;

namespace Sm103Backend {

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
  return options;
}

Sm103Model::Sm103Model(
  void* officialApplyContext_,
  OfficialApplyFn officialApply_,
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
  loggedScaffold(false)
{
  if(officialApplyContext == NULL || officialApply == NULL)
    throw StringError("Sm103Model: null official-forward adapter");
  if(nnXLen != 19 || nnYLen != 19 || !inputsUseNHWC || !useFP16 || !useNHWC)
    throw StringError("SM103 scaffold requires exact 19x19 FP16 NHWC inference");
  if(!isSm103Arch(majorComputeCapability,minorComputeCapability))
    throw StringError("SM103 optimized backend requires exact CC10.3");
  if(!options.allowOfficialForwardScaffold &&
     !options.reusePortableTactics)
    throw StringError(
      "SM103 optimized backend has no registered tactics yet; "
      "cudaSm103AllowOfficialForwardScaffold=true is required for wiring tests"
    );
}

void Sm103Model::setLogger(Logger* logger_) {
  logger = logger_;
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
