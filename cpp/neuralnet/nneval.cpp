#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/globalperf.h"

using namespace std;

namespace {
static int canonicalGpuIdxForScheduling(int gpuIdx) {
  return gpuIdx == -1 ? 0 : gpuIdx;
}

static vector<int> deriveLogicalSlotGpuIdxs(const vector<int>& gpuIdxByServerThread) {
  vector<int> logicalSlots = gpuIdxByServerThread;
  vector<int> seenGpuIdxs;
  for(int gpuIdx: gpuIdxByServerThread) {
    bool found = false;
    for(int seenGpuIdx: seenGpuIdxs) {
      if(seenGpuIdx == gpuIdx) {
        found = true;
        break;
      }
    }
    if(!found)
      seenGpuIdxs.push_back(gpuIdx);
  }
  for(int gpuIdx: seenGpuIdxs)
    logicalSlots.push_back(gpuIdx);
  return logicalSlots;
}
}

struct SchedulerState {
  struct PendingH2DSpan {
    int rowIdx;
    uint64_t preprocessSpanId;
    int64_t enqueueStartNs;

    PendingH2DSpan()
      : rowIdx(-1),
        preprocessSpanId(0),
        enqueueStartNs(0)
    {}
  };

  enum class SlotStage {
    Idle,
    Filling,
    LaunchQueued,
    D2HPending
  };

  struct SlotState {
    int slotIdx;
    int gpuIdx;
    int deviceStateIdx;
    NNServerBuf* serverBuf;
    ComputeHandle* gpuHandle;
    SlotStage stage;
    bool inferStarted;
    int batchSize;
    int markerSlotIdx;
    double plannedWorkMs;
    double remainingWorkMs;
    double accumulatedEquivalentWorkMs;
    uint64_t batchUid;
    uint64_t lastH2DSpanId;
    uint64_t inferDependencySpanId;
    uint64_t inferSpanId;
    uint64_t d2hSpanId;
    int64_t inferStartNs;
    int64_t inferEndNs;
    int64_t lastH2DEndNs;
    std::vector<PendingH2DSpan> pendingH2DSpans;
    std::vector<NNResultBuf*> requests;
    std::vector<NNOutput*> outputs;
    bool isUsingFP16;

    SlotState()
      : slotIdx(-1),
        gpuIdx(-1),
        deviceStateIdx(-1),
        serverBuf(nullptr),
        gpuHandle(nullptr),
        stage(SlotStage::Idle),
        inferStarted(false),
        batchSize(0),
        markerSlotIdx(-1),
        plannedWorkMs(0.0),
        remainingWorkMs(0.0),
        accumulatedEquivalentWorkMs(0.0),
        batchUid(0),
        lastH2DSpanId(0),
        inferDependencySpanId(0),
        inferSpanId(0),
        d2hSpanId(0),
        inferStartNs(0),
        inferEndNs(0),
        lastH2DEndNs(0),
        pendingH2DSpans(),
        requests(),
        outputs(),
        isUsingFP16(false)
    {}
  };

  struct DeviceState {
    int gpuIdx;
    std::vector<int> slotIndices;
    int rrCursor;
    int activeInferCount;
    int64_t lastProgressNs;
    std::vector<double> baseWorkMsByBatch;
    std::vector<std::vector<double>> workSamplesByBatch;

    DeviceState()
      : gpuIdx(-1),
        slotIndices(),
        rrCursor(0),
        activeInferCount(0),
        lastProgressNs(0),
        baseWorkMsByBatch(),
        workSamplesByBatch()
    {}
  };

  struct OpenBatchState {
    bool exists;
    int targetSlotIdx;
    int targetGpuIdx;
    int markerSlotIdx;
    uint64_t batchUid;
    uint64_t lastH2DSpanId;
    std::vector<PendingH2DSpan> pendingH2DSpans;
    std::vector<NNResultBuf*> requests;

    OpenBatchState()
      : exists(false),
        targetSlotIdx(-1),
        targetGpuIdx(-1),
        markerSlotIdx(-1),
        batchUid(0),
        lastH2DSpanId(0),
        pendingH2DSpans(),
        requests()
    {}
  };

  Rand rand;
  std::vector<SlotState> slots;
  std::vector<DeviceState> devices;
  std::map<int,int> deviceStateIdxByGpu;
  OpenBatchState openBatch;
  uint64_t nextBatchUid;
  uint64_t nextTimelineSpanId;
  bool startupFailed;
  std::string startupFailureMessage;

  explicit SchedulerState(const std::string& randSeed)
    : rand(randSeed),
      slots(),
      devices(),
      deviceStateIdxByGpu(),
      openBatch(),
      nextBatchUid(1),
      nextTimelineSpanId(1),
      startupFailed(false),
      startupFailureMessage()
  {}
};

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  : clientWaitingForResult(),
    resultMutex(),
    hasResult(false),
    includeOwnerMap(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    rowSpatialBuf(),
    rowGlobalBuf(),
    rowMetaBuf(),
    hasRowMeta(false),
    result(nullptr),
    submittedToNNServer(false),
    errorLogLockout(false),
    // If no symmetry is specified, it will use default or random based on config.
    symmetry(NNInputs::SYMMETRY_NOTSPECIFIED),
    policyOptimism(0.0)
{}

NNResultBuf::~NNResultBuf() {
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL)
{
  int maxBatchSize = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxBatchSize,nnEval.getNNXLen(),nnEval.getNNYLen());
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  const string& expectedSha256,
  Logger* lg,
  int maxBatchSz,
  int xLen,
  int yLen,
  bool rExactNNLen,
  bool iUseNHWC,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerofTwo,
  bool skipNeuralNet,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const TRTConfigs& trtCfgs,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry,
  int backendNumThr
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   usingNHWCMode(useNHWCMode),
   trtConfigs(trtCfgs),
   numThreads(numThr),
   backendNumThreads(backendNumThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   gpuIdxByLogicalSlot(deriveLogicalSlotGpuIdxs(gpuIdxByServerThr)),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(NULL),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   schedulerState(NULL),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   numGpuBusyClaims(),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   currentBatchSize(maxBatchSz),
   queryQueue()
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
  }

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo, nnMutexPoolSizePowerofTwo);

  if(!debugSkipNeuralNet) {
    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    auto last = std::unique(gpuIdxs.begin(), gpuIdxs.end());
    gpuIdxs.erase(last,gpuIdxs.end());
    loadedModel = NeuralNet::loadModelFile(modelFileName,expectedSha256);
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    if(desc.onnxHeader.isOnnx)
    {
      desc.onnxHeader.maybeChangeNNLen(*this);
      if(nnXLen > NNPos::MAX_BOARD_LEN)
        throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
      if(nnYLen > NNPos::MAX_BOARD_LEN)
        throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
    }
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    postProcessParams = desc.postProcessParams;
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      openCLTunerFile,homeDataDirOverride,openCLReTunePerBoardSize,
      usingFP16Mode,usingNHWCMode,loadedModel,trtConfigs
    );
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  //Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByLogicalSlot.size());
  //Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
  queryQueue.setSizeChangedObserver([](size_t newSize) {
    GlobalPerfProfile::noteQueueLength((int)newSize);
  });

  for(int gpuIdx: gpuIdxByServerThread)
    numGpuBusyClaims[canonicalGpuIdxForScheduling(gpuIdx)] = 0;
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();
  delete schedulerState;
  schedulerState = NULL;

  if(computeContext != NULL)
    NeuralNet::freeComputeContext(computeContext);
  computeContext = NULL;

  if(loadedModel != NULL)
    NeuralNet::freeLoadedModel(loadedModel);
  loadedModel = NULL;

  delete nnCacheTable;
}

string NNEvaluator::getModelName() const {
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  return modelFileName;
}
string NNEvaluator::getInternalModelName() const {
  return internalModelName;
}

static bool tryAbbreviateStepString(const string& input, string& buf) {
  size_t i = 0;
  while(i < input.length() && !Global::isDigit(input[i]))
    i++;
  if(i > 1)
    return false;

  string prefix = input.substr(0, i);
  int64_t number;
  bool suc = Global::tryStringToInt64(input.substr(i),number);
  if(!suc)
    return false;

  if(number >= 10000000000LL)
    buf = prefix + std::to_string(number / 1000000000LL) + "G";
  if(number >= 10000000)
    buf = prefix + std::to_string(number / 1000000) + "M";
  else if(number >= 10000)
    buf = prefix + std::to_string(number / 1000) + "K";
  else
    buf = input;
  return true;
}

string NNEvaluator::getAbbrevInternalModelName() const {
  string name = getInternalModelName();
  std::vector<string> pieces = Global::split(name,'-');
  std::vector<string> newPieces;
  for(const string& piece: pieces) {
    string buf;
    if(piece == "kata1") {
      // skip
    }
    else if(piece.size() > 1 && piece[0] == 's' && tryAbbreviateStepString(piece,buf)) {
      newPieces.push_back(buf);
    }
    else if(piece.size() > 1 && piece[0] == 'd' && tryAbbreviateStepString(piece,buf)) {
      // skip
    }
    else {
      newPieces.push_back(piece);
    }
  }
  return Global::concat(newPieces,"-");
}

Logger* NNEvaluator::getLogger() {
  return logger;
}
bool NNEvaluator::isNeuralNetLess() const {
  return debugSkipNeuralNet;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxBatchSize;
}
int NNEvaluator::getCurrentBatchSize() const {
  return currentBatchSize.load(std::memory_order_acquire);
}
void NNEvaluator::setCurrentBatchSize(int batchSize) {
  if(batchSize <= 0 || batchSize > maxBatchSize)
    throw StringError("Invalid setting for batch size");
  currentBatchSize.store(batchSize,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
#ifdef USE_EIGEN_BACKEND
  return 1;
#else
  std::set<int> gpuIdxs;
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
  return (int)gpuIdxs.size();
#endif
}
int NNEvaluator::getNumServerThreads() const {
#ifdef USE_TENSORRT_BACKEND
  if(!debugSkipNeuralNet)
    return (int)gpuIdxByLogicalSlot.size();
#endif
  return (int)gpuIdxByServerThread.size();
}
std::set<int> NNEvaluator::getGpuIdxs() const {
  std::set<int> gpuIdxs;
#ifdef USE_EIGEN_BACKEND
  gpuIdxs.insert(0);
#else
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
#endif
  return gpuIdxs;
}

int NNEvaluator::getNNXLen() const {
  return nnXLen;
}
int NNEvaluator::getNNYLen() const {
  return nnYLen;
}
int NNEvaluator::getModelVersion() const {
  return modelVersion;
}
double NNEvaluator::getTrunkSpatialConvDepth() const {
  return NeuralNet::getModelDesc(loadedModel).getTrunkSpatialConvDepth();
}

enabled_t NNEvaluator::getUsingFP16Mode() const {
  return usingFP16Mode;
}
enabled_t NNEvaluator::getUsingNHWCMode() const {
  return usingNHWCMode;
}

bool NNEvaluator::supportsShorttermError() const {
  return modelVersion >= 9;
}

bool NNEvaluator::getDoRandomize() const {
  return currentDoRandomize.load(std::memory_order_acquire);
}
int NNEvaluator::getDefaultSymmetry() const {
  return currentDefaultSymmetry.load(std::memory_order_acquire);
}
void NNEvaluator::setDoRandomize(bool b) {
  currentDoRandomize.store(b, std::memory_order_release);
}
void NNEvaluator::setDefaultSymmetry(int s) {
  currentDefaultSymmetry.store(s, std::memory_order_release);
}

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) {
  if(loadedModel == NULL) {
    supported = true;
    return desiredRules;
  }
  return NeuralNet::getModelDesc(loadedModel).getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != NULL)
    nnCacheTable->clear();
}


bool NNEvaluator::isAnyThreadUsingFP16() const {
  lock_guard<std::mutex> lock(bufferMutex);
  for(const int& isUsingFP16: serverThreadsIsUsingFP16) {
    if(isUsingFP16)
      return true;
  }
  return false;
}

#ifdef USE_TENSORRT_BACKEND
void NNEvaluator::serveTrtScheduler(const string& randSeedThisThread) {
  (void)randSeedThisThread;
  SchedulerState* state = schedulerState;
  assert(state != NULL);

  auto clearOpenBatch = [&]() {
    state->openBatch.exists = false;
    state->openBatch.targetSlotIdx = -1;
    state->openBatch.targetGpuIdx = -1;
    state->openBatch.markerSlotIdx = -1;
    state->openBatch.batchUid = 0;
    state->openBatch.lastH2DSpanId = 0;
    state->openBatch.pendingH2DSpans.clear();
    state->openBatch.requests.clear();
  };

  auto clearSlot = [&](SchedulerState::SlotState& slot) {
    for(NNOutput* output: slot.outputs)
      delete output;
    slot.outputs.clear();
    slot.requests.clear();
    slot.stage = SchedulerState::SlotStage::Idle;
    slot.inferStarted = false;
    slot.batchSize = 0;
    slot.markerSlotIdx = -1;
    slot.plannedWorkMs = 0.0;
    slot.remainingWorkMs = 0.0;
    slot.accumulatedEquivalentWorkMs = 0.0;
    slot.batchUid = 0;
    slot.lastH2DSpanId = 0;
    slot.inferDependencySpanId = 0;
    slot.inferSpanId = 0;
    slot.d2hSpanId = 0;
    slot.inferStartNs = 0;
    slot.inferEndNs = 0;
    slot.lastH2DEndNs = 0;
    slot.pendingH2DSpans.clear();
  };

  auto timelineNowNs = [&]() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()
    ).count();
  };
  auto shouldCaptureTimelineSpan = [&](int64_t startNs, int64_t endNs) {
    return GlobalPerfProfile::wantsRealtimeTimelineSpan(startNs, endNs);
  };

  auto nextTimelineSpanId = [&]() {
    return state->nextTimelineSpanId++;
  };

  auto recordBatchSample = [&](SchedulerState::DeviceState& device, int batchSize, double workMs) {
    if(batchSize <= 0 || batchSize >= (int)device.baseWorkMsByBatch.size() || workMs <= 0.0)
      return;
    vector<double>& samples = device.workSamplesByBatch[batchSize];
    samples.push_back(workMs);
    if(samples.size() > 10)
      samples.erase(samples.begin());
    double sum = 0.0;
    for(double sample: samples)
      sum += sample;
    device.baseWorkMsByBatch[batchSize] = sum / samples.size();
  };

  auto advanceDeviceProgress = [&](SchedulerState::DeviceState& device, int64_t nowNs) {
    if(device.lastProgressNs == 0) {
      device.lastProgressNs = nowNs;
      return;
    }
    if(nowNs <= device.lastProgressNs)
      return;
    if(device.activeInferCount <= 0) {
      device.lastProgressNs = nowNs;
      return;
    }
    double dtMs = (double)(nowNs - device.lastProgressNs) / 1e6;
    device.lastProgressNs = nowNs;
    double workAdvanceMs = dtMs / device.activeInferCount;
    for(int slotIdx: device.slotIndices) {
      SchedulerState::SlotState& slot = state->slots[slotIdx];
      if(slot.stage == SchedulerState::SlotStage::LaunchQueued && slot.inferStarted) {
        slot.remainingWorkMs = std::max(0.0, slot.remainingWorkMs - workAdvanceMs);
        slot.accumulatedEquivalentWorkMs += workAdvanceMs;
      }
    }
  };

  auto findIdleSlotOnDevice = [&](SchedulerState::DeviceState& device) -> int {
    if(device.slotIndices.empty())
      return -1;
    int numSlots = (int)device.slotIndices.size();
    for(int offset = 0; offset < numSlots; offset++) {
      int idx = (device.rrCursor + offset) % numSlots;
      int slotIdx = device.slotIndices[idx];
      if(state->slots[slotIdx].stage == SchedulerState::SlotStage::Idle) {
        device.rrCursor = (idx + 1) % numSlots;
        return slotIdx;
      }
    }
    return -1;
  };

  auto selectOpenBatchTarget = [&]() -> bool {
    int emptySlotIdx = -1;
    for(size_t i = 0; i < state->devices.size(); i++) {
      SchedulerState::DeviceState& device = state->devices[i];
      if(device.activeInferCount == 0) {
        int slotIdx = findIdleSlotOnDevice(device);
        if(slotIdx >= 0) {
          emptySlotIdx = slotIdx;
          break;
        }
      }
    }
    if(emptySlotIdx >= 0) {
      SchedulerState::SlotState& slot = state->slots[emptySlotIdx];
      slot.stage = SchedulerState::SlotStage::Filling;
      state->openBatch.exists = true;
      state->openBatch.targetSlotIdx = emptySlotIdx;
      state->openBatch.targetGpuIdx = slot.gpuIdx;
      state->openBatch.markerSlotIdx = -1;
      state->openBatch.batchUid = state->nextBatchUid++;
      state->openBatch.lastH2DSpanId = 0;
      state->openBatch.pendingH2DSpans.clear();
      state->openBatch.requests.clear();
      GlobalPerfProfile::changeInferenceThreadActiveCount(emptySlotIdx,1);
      return true;
    }

    double bestRemainingMs = std::numeric_limits<double>::infinity();
    int bestSlotIdx = -1;
    int bestMarkerSlotIdx = -1;
    for(size_t i = 0; i < state->devices.size(); i++) {
      SchedulerState::DeviceState& device = state->devices[i];
      int idleSlotIdx = findIdleSlotOnDevice(device);
      if(idleSlotIdx < 0)
        continue;
      for(int markerSlotIdx: device.slotIndices) {
        SchedulerState::SlotState& markerSlot = state->slots[markerSlotIdx];
        if(markerSlot.stage == SchedulerState::SlotStage::LaunchQueued && markerSlot.inferStarted) {
          if(markerSlot.remainingWorkMs < bestRemainingMs) {
            bestRemainingMs = markerSlot.remainingWorkMs;
            bestSlotIdx = idleSlotIdx;
            bestMarkerSlotIdx = markerSlotIdx;
          }
        }
      }
    }

    if(bestSlotIdx >= 0) {
      SchedulerState::SlotState& slot = state->slots[bestSlotIdx];
      slot.stage = SchedulerState::SlotStage::Filling;
      state->openBatch.exists = true;
      state->openBatch.targetSlotIdx = bestSlotIdx;
      state->openBatch.targetGpuIdx = slot.gpuIdx;
      state->openBatch.markerSlotIdx = bestMarkerSlotIdx;
      state->openBatch.batchUid = state->nextBatchUid++;
      state->openBatch.lastH2DSpanId = 0;
      state->openBatch.pendingH2DSpans.clear();
      state->openBatch.requests.clear();
      GlobalPerfProfile::changeInferenceThreadActiveCount(bestSlotIdx,1);
      return true;
    }
    return false;
  };

  auto allocateOutputsForSlot = [&](SchedulerState::SlotState& slot) {
    for(NNOutput* output: slot.outputs)
      delete output;
    slot.outputs.clear();
    slot.outputs.reserve(slot.batchSize);
    for(int row = 0; row < slot.batchSize; row++) {
      NNOutput* emptyOutput = new NNOutput();
      emptyOutput->nnXLen = nnXLen;
      emptyOutput->nnYLen = nnYLen;
      if(slot.requests[row]->includeOwnerMap)
        emptyOutput->whiteOwnerMap = new float[nnXLen * nnYLen];
      else
        emptyOutput->whiteOwnerMap = NULL;
      slot.outputs.push_back(emptyOutput);
    }
  };

  auto maybeLaunchOpenBatch = [&]() -> bool {
    if(!state->openBatch.exists)
      return false;
    SchedulerState::SlotState& slot = state->slots[state->openBatch.targetSlotIdx];
    SchedulerState::DeviceState& device = state->devices[slot.deviceStateIdx];
    const int64_t launchNowNs = timelineNowNs();
    int desiredBatchSize = std::min(maxBatchSize, currentBatchSize.load(std::memory_order_acquire));
    bool shouldLaunch = false;
    if(device.activeInferCount == 0)
      shouldLaunch = true;
    else if((int)state->openBatch.requests.size() >= desiredBatchSize)
      shouldLaunch = true;
    if(!shouldLaunch)
      return false;

    slot.requests = state->openBatch.requests;
    slot.batchSize = (int)slot.requests.size();
    slot.markerSlotIdx = state->openBatch.markerSlotIdx;
    slot.plannedWorkMs = device.baseWorkMsByBatch[slot.batchSize];
    slot.remainingWorkMs = slot.plannedWorkMs;
    slot.accumulatedEquivalentWorkMs = 0.0;
    slot.batchUid = state->openBatch.batchUid;
    slot.lastH2DSpanId = state->openBatch.lastH2DSpanId;
    slot.inferDependencySpanId = 0;
    slot.inferSpanId = 0;
    slot.d2hSpanId = 0;
    slot.inferStartNs = 0;
    slot.inferEndNs = 0;
    slot.lastH2DEndNs = 0;
    slot.pendingH2DSpans = state->openBatch.pendingH2DSpans;

    allocateOutputsForSlot(slot);

    const ComputeHandle* dependencyHandle = nullptr;
    bool waitsOnMarker = false;
    if(slot.markerSlotIdx >= 0) {
      const SchedulerState::SlotState& markerSlot = state->slots[slot.markerSlotIdx];
      if(markerSlot.stage == SchedulerState::SlotStage::LaunchQueued)
        waitsOnMarker = true;
      else
        slot.inferDependencySpanId = markerSlot.inferSpanId;
    }
    if(waitsOnMarker)
      dependencyHandle = state->slots[slot.markerSlotIdx].gpuHandle;
    NeuralNet::trtLaunchInferenceAsync(slot.gpuHandle, slot.serverBuf->inputBuffers, slot.batchSize, dependencyHandle);
    NeuralNet::trtEnqueueOutputCopiesAsync(slot.gpuHandle, slot.serverBuf->inputBuffers, slot.batchSize);

    slot.stage = SchedulerState::SlotStage::LaunchQueued;
    slot.inferStarted = !waitsOnMarker;
    if(slot.inferStarted) {
      slot.inferStartNs = launchNowNs;
      device.activeInferCount += 1;
      GlobalPerfProfile::changeGpuStreamActiveCount(slot.slotIdx, slot.gpuIdx, 1);
    }

    clearOpenBatch();
    return true;
  };

  Board benchmarkBoard(nnXLen, nnYLen);
  BoardHistory benchmarkHistory(benchmarkBoard, P_BLACK, Rules::getTrompTaylorish(), 0);
  MiscNNInputParams benchmarkNNInputParams;
  benchmarkNNInputParams.symmetry = 0;
  benchmarkNNInputParams.policyOptimism = 0.0;
  SGFMetadata benchmarkMeta;
  const SGFMetadata* benchmarkSgfMeta = nullptr;
  if(numInputMetaChannels > 0) {
    benchmarkMeta = SGFMetadata::getProfile("rank_1d");
    benchmarkSgfMeta = &benchmarkMeta;
  }

  auto fillBenchmarkRequest = [&](NNResultBuf& request) {
    request.includeOwnerMap = false;
    request.boardXSizeForServer = benchmarkBoard.x_size;
    request.boardYSizeForServer = benchmarkBoard.y_size;
    const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
    if(request.rowSpatialBuf.size() < rowSpatialLen)
      request.rowSpatialBuf.resize(rowSpatialLen);
    const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if(request.rowGlobalBuf.size() < rowGlobalLen)
      request.rowGlobalBuf.resize(rowGlobalLen);
    const int rowMetaLen = numInputMetaChannels;
    if(request.rowMetaBuf.size() < rowMetaLen)
      request.rowMetaBuf.resize(rowMetaLen);

    static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
    if(inputsVersion == 3)
      NNInputs::fillRowV3(benchmarkBoard, benchmarkHistory, P_BLACK, benchmarkNNInputParams, nnXLen, nnYLen, inputsUseNHWC, request.rowSpatialBuf.data(), request.rowGlobalBuf.data());
    else if(inputsVersion == 4)
      NNInputs::fillRowV4(benchmarkBoard, benchmarkHistory, P_BLACK, benchmarkNNInputParams, nnXLen, nnYLen, inputsUseNHWC, request.rowSpatialBuf.data(), request.rowGlobalBuf.data());
    else if(inputsVersion == 5)
      NNInputs::fillRowV5(benchmarkBoard, benchmarkHistory, P_BLACK, benchmarkNNInputParams, nnXLen, nnYLen, inputsUseNHWC, request.rowSpatialBuf.data(), request.rowGlobalBuf.data());
    else if(inputsVersion == 6)
      NNInputs::fillRowV6(benchmarkBoard, benchmarkHistory, P_BLACK, benchmarkNNInputParams, nnXLen, nnYLen, inputsUseNHWC, request.rowSpatialBuf.data(), request.rowGlobalBuf.data());
    else if(inputsVersion == 7)
      NNInputs::fillRowV7(benchmarkBoard, benchmarkHistory, P_BLACK, benchmarkNNInputParams, nnXLen, nnYLen, inputsUseNHWC, request.rowSpatialBuf.data(), request.rowGlobalBuf.data());
    else
      ASSERT_UNREACHABLE;

    if(rowMetaLen > 0) {
      assert(benchmarkSgfMeta != nullptr && benchmarkSgfMeta->initialized);
      SGFMetadata::fillMetadataRow(
        benchmarkSgfMeta,
        request.rowMetaBuf.data(),
        P_BLACK,
        benchmarkBoard.x_size * benchmarkBoard.y_size
      );
      request.hasRowMeta = true;
    }
    else {
      request.hasRowMeta = false;
    }

    request.symmetry = 0;
    request.policyOptimism = 0.0;
  };

  auto initializeDeviceBaseWorkEstimates = [&]() {
    constexpr int warmupRuns = 1;
    constexpr int measuredRuns = 10;
    vector<NNResultBuf> benchmarkRequests((size_t)maxBatchSize);
    for(NNResultBuf& request: benchmarkRequests)
      fillBenchmarkRequest(request);

    for(SchedulerState::DeviceState& device: state->devices) {
      if(device.slotIndices.empty())
        continue;
      SchedulerState::SlotState& slot = state->slots[device.slotIndices[0]];
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        for(int row = 0; row < batchSize; row++) {
          NeuralNet::trtPackInputRow(slot.serverBuf->inputBuffers, &benchmarkRequests[row], row, slot.gpuHandle);
          NeuralNet::trtEnqueueInputRowCopy(slot.gpuHandle, slot.serverBuf->inputBuffers, row);
        }

        auto measureOnceMs = [&]() {
          auto startTime = std::chrono::steady_clock::now();
          NeuralNet::trtLaunchInferenceAsync(slot.gpuHandle, slot.serverBuf->inputBuffers, batchSize, nullptr);
          while(!NeuralNet::trtQueryInferenceDone(slot.gpuHandle))
            std::this_thread::yield();
          auto endTime = std::chrono::steady_clock::now();
          return std::chrono::duration<double,std::milli>(endTime - startTime).count();
        };

        for(int i = 0; i < warmupRuns; i++)
          (void)measureOnceMs();

        vector<double>& samples = device.workSamplesByBatch[batchSize];
        samples.clear();
        samples.reserve(measuredRuns);
        double sumMs = 0.0;
        for(int i = 0; i < measuredRuns; i++) {
          double sampleMs = measureOnceMs();
          samples.push_back(sampleMs);
          sumMs += sampleMs;
        }
        device.baseWorkMsByBatch[batchSize] = sumMs / measuredRuns;
      }
    }
  };

  bool startupComplete = false;
  try {
    state->slots.resize(gpuIdxByLogicalSlot.size());
    for(size_t i = 0; i < gpuIdxByLogicalSlot.size(); i++) {
      int gpuIdx = gpuIdxByLogicalSlot[i];
      int canonicalGpuIdx = canonicalGpuIdxForScheduling(gpuIdx);
      int deviceStateIdx;
      auto iter = state->deviceStateIdxByGpu.find(canonicalGpuIdx);
      if(iter == state->deviceStateIdxByGpu.end()) {
        deviceStateIdx = (int)state->devices.size();
        state->deviceStateIdxByGpu[canonicalGpuIdx] = deviceStateIdx;
        state->devices.push_back(SchedulerState::DeviceState());
        state->devices.back().gpuIdx = gpuIdx;
        state->devices.back().rrCursor = 0;
        state->devices.back().activeInferCount = 0;
        state->devices.back().lastProgressNs = 0;
        state->devices.back().baseWorkMsByBatch.assign(maxBatchSize + 1, 1.0);
        state->devices.back().workSamplesByBatch.resize(maxBatchSize + 1);
      }
      else {
        deviceStateIdx = iter->second;
      }

      SchedulerState::SlotState& slot = state->slots[i];
      slot.slotIdx = (int)i;
      slot.gpuIdx = gpuIdx;
      slot.deviceStateIdx = deviceStateIdx;
      slot.serverBuf = new NNServerBuf(*this, loadedModel);
      slot.gpuHandle = NeuralNet::createComputeHandle(
        computeContext,
        loadedModel,
        logger,
        maxBatchSize,
        requireExactNNLen,
        inputsUseNHWC,
        gpuIdx,
        slot.slotIdx,
        backendNumThreads
      );
      slot.isUsingFP16 = NeuralNet::isUsingFP16(slot.gpuHandle);
      state->devices[deviceStateIdx].slotIndices.push_back(slot.slotIdx);
    }

    initializeDeviceBaseWorkEstimates();

    {
      lock_guard<std::mutex> lock(bufferMutex);
      serverThreadsIsUsingFP16.assign(state->slots.size(), 0);
      for(const SchedulerState::SlotState& slot: state->slots)
        serverThreadsIsUsingFP16[slot.slotIdx] = slot.isUsingFP16 ? 1 : 0;
      numServerThreadsStartingUp--;
      if(numServerThreadsStartingUp <= 0)
        mainThreadWaitingForSpawn.notify_all();
    }
    startupComplete = true;

    NNResultBuf* deferredRequest = nullptr;
    int idleSpinIterations = 0;

    auto handleInferCompletion = [&](SchedulerState::SlotState& slot, int64_t nowNs) {
      SchedulerState::DeviceState& device = state->devices[slot.deviceStateIdx];
      uint64_t completedInferSpanId = 0;
      int64_t recordedInferEndNs = nowNs;
      if(slot.stage == SchedulerState::SlotStage::LaunchQueued && slot.inferStarted) {
        GlobalPerfProfile::changeGpuStreamActiveCount(slot.slotIdx, slot.gpuIdx, -1);
        if(device.activeInferCount > 0)
          device.activeInferCount -= 1;
        double measuredWorkMs = slot.accumulatedEquivalentWorkMs;
        if(measuredWorkMs <= 0.0)
          measuredWorkMs = slot.plannedWorkMs;
        recordBatchSample(device, slot.batchSize, measuredWorkMs);

        uint64_t previousH2DSpanId = 0;
        int64_t previousH2DEndNs = 0;
        slot.lastH2DSpanId = 0;
        slot.lastH2DEndNs = 0;
        for(const SchedulerState::PendingH2DSpan& pendingH2D: slot.pendingH2DSpans) {
          double h2dElapsedMs = NeuralNet::trtGetLastInputRowCopyElapsedMs(slot.gpuHandle, pendingH2D.rowIdx);
          int64_t h2dDurationNs = h2dElapsedMs > 0.0 ? (int64_t)(h2dElapsedMs * 1e6 + 0.5) : 0;
          int64_t recordedH2DStartNs = pendingH2D.enqueueStartNs > 0 ? pendingH2D.enqueueStartNs : nowNs;
          if(recordedH2DStartNs < previousH2DEndNs)
            recordedH2DStartNs = previousH2DEndNs;
          int64_t recordedH2DEndNs = h2dDurationNs > 0 ? recordedH2DStartNs + h2dDurationNs : recordedH2DStartNs;
          uint64_t h2dSpanId = 0;
          if(shouldCaptureTimelineSpan(recordedH2DStartNs, recordedH2DEndNs)) {
            h2dSpanId = nextTimelineSpanId();
            GlobalPerfProfile::recordRealtimeTimelineSpan(
              slot.slotIdx,
              slot.gpuIdx,
              GlobalPerfProfile::TimelineLane::H2DStream,
              GlobalPerfProfile::TimelineStage::H2D,
              h2dSpanId,
              pendingH2D.preprocessSpanId,
              previousH2DSpanId,
              slot.batchUid,
              pendingH2D.rowIdx,
              recordedH2DStartNs,
              recordedH2DEndNs
            );
          }
          previousH2DSpanId = h2dSpanId;
          previousH2DEndNs = recordedH2DEndNs;
          slot.lastH2DSpanId = h2dSpanId;
          slot.lastH2DEndNs = recordedH2DEndNs;
        }
        slot.pendingH2DSpans.clear();

        double inferElapsedMs = NeuralNet::trtGetLastInferenceElapsedMs(slot.gpuHandle);
        int64_t inferDurationNs = inferElapsedMs > 0.0 ? (int64_t)(inferElapsedMs * 1e6 + 0.5) : 0;
        int64_t inferReadyNs = slot.inferStartNs > 0 ? slot.inferStartNs : nowNs;
        if(slot.lastH2DEndNs > inferReadyNs)
          inferReadyNs = slot.lastH2DEndNs;
        int64_t inferredStartNs = inferDurationNs > 0 ? nowNs - inferDurationNs : inferReadyNs;
        if(inferredStartNs < inferReadyNs)
          inferredStartNs = inferReadyNs;
        const int64_t inferStartNs = inferredStartNs;
        recordedInferEndNs = inferDurationNs > 0 ? std::min(nowNs, inferStartNs + inferDurationNs) : nowNs;
        if(shouldCaptureTimelineSpan(inferStartNs, recordedInferEndNs)) {
          completedInferSpanId = nextTimelineSpanId();
          GlobalPerfProfile::recordRealtimeTimelineSpan(
            slot.slotIdx,
            slot.gpuIdx,
            GlobalPerfProfile::TimelineLane::InferStream,
            GlobalPerfProfile::TimelineStage::Infer,
            completedInferSpanId,
            slot.lastH2DSpanId,
            slot.inferDependencySpanId,
            slot.batchUid,
            -1,
            inferStartNs,
            recordedInferEndNs
          );
        }
      }
      slot.inferStarted = false;
      slot.inferSpanId = completedInferSpanId;
      slot.inferEndNs = recordedInferEndNs;
      slot.stage = SchedulerState::SlotStage::D2HPending;
      slot.remainingWorkMs = 0.0;

      for(int dependentSlotIdx: device.slotIndices) {
        SchedulerState::SlotState& dependentSlot = state->slots[dependentSlotIdx];
        if(dependentSlot.stage == SchedulerState::SlotStage::LaunchQueued &&
            !dependentSlot.inferStarted &&
            dependentSlot.markerSlotIdx == slot.slotIdx) {
          dependentSlot.inferStarted = true;
          dependentSlot.inferStartNs = slot.inferEndNs > 0 ? slot.inferEndNs : nowNs;
          dependentSlot.remainingWorkMs = dependentSlot.plannedWorkMs;
          dependentSlot.accumulatedEquivalentWorkMs = 0.0;
          dependentSlot.inferDependencySpanId = completedInferSpanId;
          device.activeInferCount += 1;
          GlobalPerfProfile::changeGpuStreamActiveCount(dependentSlot.slotIdx, dependentSlot.gpuIdx, 1);
        }
      }
    };

    auto finalizeCompletedBatch = [&](SchedulerState::SlotState& slot) {
      const int64_t d2hDoneNs = timelineNowNs();
      const int completedBatchSize = slot.batchSize;
      const double completedInferMs = slot.accumulatedEquivalentWorkMs > 0.0 ? slot.accumulatedEquivalentWorkMs : slot.plannedWorkMs;
      uint64_t d2hSpanId = 0;
      double d2hElapsedMs = NeuralNet::trtGetLastOutputCopiesElapsedMs(slot.gpuHandle);
      int64_t d2hDurationNs = d2hElapsedMs > 0.0 ? (int64_t)(d2hElapsedMs * 1e6 + 0.5) : 0;
      const int64_t d2hStartNs = slot.inferEndNs > 0 ? slot.inferEndNs : d2hDoneNs;
      const int64_t recordedD2HEndNs = (d2hDurationNs > 0 && d2hStartNs <= d2hDoneNs) ? std::min(d2hDoneNs, d2hStartNs + d2hDurationNs) : d2hDoneNs;
      if(shouldCaptureTimelineSpan(d2hStartNs, recordedD2HEndNs)) {
        d2hSpanId = nextTimelineSpanId();
        GlobalPerfProfile::recordRealtimeTimelineSpan(
          slot.slotIdx,
          slot.gpuIdx,
          GlobalPerfProfile::TimelineLane::D2HStream,
          GlobalPerfProfile::TimelineStage::D2H,
          d2hSpanId,
          slot.inferSpanId,
          0,
          slot.batchUid,
          -1,
          d2hStartNs,
          recordedD2HEndNs
        );
      }
      slot.d2hSpanId = d2hSpanId;

      const int64_t postprocessStartNs = std::max(recordedD2HEndNs, timelineNowNs());
      for(int row = 0; row < completedBatchSize; row++) {
        NeuralNet::trtUnpackOutputRow(slot.serverBuf->inputBuffers, slot.requests[row], slot.outputs[row], row, slot.gpuHandle);
      }
      const int64_t postprocessEndNs = timelineNowNs();
      if(shouldCaptureTimelineSpan(postprocessStartNs, postprocessEndNs)) {
        GlobalPerfProfile::recordRealtimeTimelineSpan(
          slot.slotIdx,
          slot.gpuIdx,
          GlobalPerfProfile::TimelineLane::SchedulerThread,
          GlobalPerfProfile::TimelineStage::Postprocess,
          nextTimelineSpanId(),
          d2hSpanId,
          0,
          slot.batchUid,
          -1,
          postprocessStartNs,
          postprocessEndNs
        );
      }

      m_numRowsProcessed.fetch_add(completedBatchSize, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      GlobalPerfProfile::recordRealtimeInferenceBatch(
        slot.slotIdx,
        slot.gpuIdx,
        completedBatchSize,
        completedBatchSize,
        0.0,
        0.0,
        0.0,
        completedInferMs,
        0.0,
        0.0
      );

      for(int row = 0; row < completedBatchSize; row++) {
        NNResultBuf* resultBuf = slot.requests[row];
        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        resultBuf->result = std::shared_ptr<NNOutput>(slot.outputs[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
      }
      slot.outputs.clear();
      slot.requests.clear();
      slot.batchSize = 0;
      slot.stage = SchedulerState::SlotStage::Idle;
      slot.inferStarted = false;
      slot.markerSlotIdx = -1;
      slot.plannedWorkMs = 0.0;
      slot.remainingWorkMs = 0.0;
      slot.accumulatedEquivalentWorkMs = 0.0;
      slot.batchUid = 0;
      slot.lastH2DSpanId = 0;
      slot.inferDependencySpanId = 0;
      slot.inferSpanId = 0;
      slot.d2hSpanId = 0;
      slot.inferStartNs = 0;
      slot.inferEndNs = 0;
      slot.lastH2DEndNs = 0;
      slot.pendingH2DSpans.clear();
      GlobalPerfProfile::changeInferenceThreadActiveCount(slot.slotIdx,-1);

      unique_lock<std::mutex> lock(bufferMutex);
      numOngoingEvals -= completedBatchSize;
      if(numWaitingEvals > 0) {
        numEvalsToAwaken += numWaitingEvals;
        numWaitingEvals = 0;
        waitingForFinish.notify_all();
      }
    };

    while(true) {
      int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
      ).count();
      for(SchedulerState::DeviceState& device: state->devices)
        advanceDeviceProgress(device, nowNs);

      bool didWork = false;

      for(SchedulerState::SlotState& slot: state->slots) {
        if(slot.stage == SchedulerState::SlotStage::LaunchQueued &&
            NeuralNet::trtQueryInferenceDone(slot.gpuHandle)) {
          handleInferCompletion(slot, nowNs);
          didWork = true;
        }
      }

      for(SchedulerState::SlotState& slot: state->slots) {
        if(slot.stage == SchedulerState::SlotStage::D2HPending &&
            NeuralNet::trtQueryOutputCopiesDone(slot.gpuHandle)) {
          finalizeCompletedBatch(slot);
          didWork = true;
        }
      }

      if(state->openBatch.exists)
        didWork = maybeLaunchOpenBatch() || didWork;

      NNResultBuf* request = deferredRequest;
      if(request == nullptr) {
        NNResultBuf* popped = nullptr;
        if(queryQueue.tryPop(popped))
          request = popped;
      }
      deferredRequest = nullptr;

      if(request != nullptr) {
        if(!state->openBatch.exists && !selectOpenBatchTarget()) {
          deferredRequest = request;
        }
        else {
          SchedulerState::SlotState& slot = state->slots[state->openBatch.targetSlotIdx];
          int rowIdx = (int)state->openBatch.requests.size();
          int64_t preprocessStartNs = timelineNowNs();
          bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
          int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);
          if(request->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
            if(doRandomize)
              request->symmetry = state->rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
            else
              request->symmetry = defaultSymmetry;
          }
          NeuralNet::trtPackInputRow(slot.serverBuf->inputBuffers, request, rowIdx, slot.gpuHandle);
          int64_t preprocessEndNs = timelineNowNs();
          uint64_t preprocessSpanId = 0;
          if(shouldCaptureTimelineSpan(preprocessStartNs, preprocessEndNs)) {
            preprocessSpanId = nextTimelineSpanId();
            GlobalPerfProfile::recordRealtimeTimelineSpan(
              slot.slotIdx,
              slot.gpuIdx,
              GlobalPerfProfile::TimelineLane::SchedulerThread,
              GlobalPerfProfile::TimelineStage::Preprocess,
              preprocessSpanId,
              0,
              0,
              state->openBatch.batchUid,
              rowIdx,
              preprocessStartNs,
              preprocessEndNs
            );
          }

          int64_t h2dStartNs = timelineNowNs();
          NeuralNet::trtEnqueueInputRowCopy(slot.gpuHandle, slot.serverBuf->inputBuffers, rowIdx);
          SchedulerState::PendingH2DSpan pendingH2D;
          pendingH2D.rowIdx = rowIdx;
          pendingH2D.preprocessSpanId = preprocessSpanId;
          pendingH2D.enqueueStartNs = h2dStartNs;
          state->openBatch.pendingH2DSpans.push_back(pendingH2D);
          state->openBatch.lastH2DSpanId = 0;
          state->openBatch.requests.push_back(request);
          didWork = true;
          if(maybeLaunchOpenBatch())
            didWork = true;
        }
      }

      bool allSlotsIdle = true;
      for(const SchedulerState::SlotState& slot: state->slots) {
        if(slot.stage != SchedulerState::SlotStage::Idle) {
          allSlotsIdle = false;
          break;
        }
      }

      if(queryQueue.isReadOnly() && deferredRequest == nullptr && !state->openBatch.exists && allSlotsIdle)
        break;

      if(!didWork) {
        idleSpinIterations += 1;
        if((idleSpinIterations & 0x3FF) == 0)
          std::this_thread::yield();
      }
      else {
        idleSpinIterations = 0;
      }
    }
  }
  catch(const std::exception& e) {
    if(!startupComplete) {
      lock_guard<std::mutex> lock(bufferMutex);
      state->startupFailed = true;
      state->startupFailureMessage = e.what();
      numServerThreadsStartingUp = 0;
      mainThreadWaitingForSpawn.notify_all();
    }
    else {
      Global::fatalError(string("TRT scheduler thread failed: ") + e.what());
    }
  }

  for(SchedulerState::SlotState& slot: state->slots) {
    clearSlot(slot);
    if(slot.gpuHandle != nullptr) {
      NeuralNet::freeComputeHandle(slot.gpuHandle);
      slot.gpuHandle = nullptr;
    }
    delete slot.serverBuf;
    slot.serverBuf = nullptr;
  }
}
#endif

static void serveEvals(
  string randSeedThisThread,
  NNEvaluator* nnEval, const LoadedModel* loadedModel,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  NNServerBuf* buf = new NNServerBuf(*nnEval,loadedModel);
  Rand rand(randSeedThisThread);

  //Used to have a try catch around this but actually we're in big trouble if this raises an exception
  //and causes possibly the only nnEval thread to die, so actually go ahead and let the exception escape to
  //toplevel for easier debugging
  nnEval->serve(*buf,rand,gpuIdxForThisThread,serverThreadIdx);
  delete buf;
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
  gpuIdxByLogicalSlot = deriveLogicalSlotGpuIdxs(gpuIdxByServerThr);
  numGpuBusyClaims.clear();
  for(int gpuIdx: gpuIdxByServerThread)
    numGpuBusyClaims[canonicalGpuIdxForScheduling(gpuIdx)] = 0;
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  bool useTrtScheduler = false;
#ifdef USE_TENSORRT_BACKEND
  useTrtScheduler = !debugSkipNeuralNet;
#endif

  GlobalPerfProfile::configureInferenceMode(useTrtScheduler);

  if(GlobalPerfProfile::isRealtimeRunning()) {
    if(useTrtScheduler)
      GlobalPerfProfile::configureInferenceSlots(gpuIdxByLogicalSlot);
    else
      GlobalPerfProfile::configureInferenceSlots(gpuIdxByServerThread);
  }

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(getNumServerThreads(),0);
  }

  delete schedulerState;
  schedulerState = NULL;
  if(useTrtScheduler)
    schedulerState = new SchedulerState(randSeed + ":NNEvalScheduler");

  queryQueue.unsetReadOnly();

  if(useTrtScheduler) {
    numServerThreadsStartingUp = 1;
    string randSeedThisThread = randSeed + ":NNEvalScheduler:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread(&NNEvaluator::serveTrtScheduler, this, randSeedThisThread);
    serverThreads.push_back(thread);
  }
  else {
    numServerThreadsStartingUp = numThreads;
    for(int i = 0; i<numThreads; i++) {
      int gpuIdxForThisThread = gpuIdxByServerThread[i];
      string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
      numServerThreadsEverSpawned++;
      std::thread* thread = new std::thread(
        &serveEvals,randSeedThisThread,this,loadedModel,gpuIdxForThisThread,i
      );
      serverThreads.push_back(thread);
    }
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(numServerThreadsStartingUp > 0)
    mainThreadWaitingForSpawn.wait(lock);
  bool startupFailed = schedulerState != NULL && schedulerState->startupFailed;
  string startupFailureMessage = startupFailed ? schedulerState->startupFailureMessage : string();
  lock.unlock();

  if(startupFailed) {
    killServerThreads();
    throw StringError("Failed to start TRT scheduler: " + startupFailureMessage);
  }
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();
  delete schedulerState;
  schedulerState = NULL;

  //Can unset now that threads are dead
  isKilled = false;

  assert(numOngoingEvals == 0);
  assert(numWaitingEvals == 0);
  assert(numEvalsToAwaken == 0);
}

void NNEvaluator::serve(
  NNServerBuf& buf, Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  struct InferenceThreadActiveGuard {
    bool enabled;
    int threadIdx;
    explicit InferenceThreadActiveGuard(bool enabled_)
      : enabled(enabled_),
        threadIdx(-1)
    {}
    InferenceThreadActiveGuard(bool enabled_, int threadIdx_)
      : enabled(enabled_),
        threadIdx(threadIdx_)
    {
      if(enabled)
        GlobalPerfProfile::changeInferenceThreadActiveCount(threadIdx,1);
    }
    ~InferenceThreadActiveGuard() {
      if(enabled)
        GlobalPerfProfile::changeInferenceThreadActiveCount(threadIdx,-1);
    }
  };

  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  if(loadedModel != NULL)
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx,
      backendNumThreads
    );

  {
    lock_guard<std::mutex> lock(bufferMutex);
    assert(serverThreadIdx < serverThreadsIsUsingFP16.size());
    serverThreadsIsUsingFP16[serverThreadIdx] = gpuHandle == NULL ? 0 : NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    numServerThreadsStartingUp--;
    if(numServerThreadsStartingUp <= 0)
      mainThreadWaitingForSpawn.notify_all();
  }

  vector<NNResultBuf*> resultBufs;
  resultBufs.reserve(maxBatchSize);

  vector<NNOutput*> outputBuf;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    resultBufs.clear();
    int desiredBatchSize = std::min(maxBatchSize, currentBatchSize.load(std::memory_order_acquire));
    const int schedulerGpuIdx = canonicalGpuIdxForScheduling(gpuIdxForThisThread);
    bool claimedIdleGpu = false;
    double waitTaskSubmitMs = 0.0;
    {
      lock.lock();
      int& busyClaimCount = numGpuBusyClaims[schedulerGpuIdx];
      if(busyClaimCount <= 0) {
        busyClaimCount += 1;
        claimedIdleGpu = true;
      }
      lock.unlock();
    }

    if(claimedIdleGpu) {
      auto waitStart = std::chrono::steady_clock::now();
      bool gotAnything = queryQueue.waitPopUpToN(resultBufs,desiredBatchSize);
      waitTaskSubmitMs = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - waitStart).count();
      //Queue being closed is a signal that we're done.
      if(!gotAnything) {
        lock.lock();
        numGpuBusyClaims[schedulerGpuIdx] -= 1;
        lock.unlock();
        break;
      }
    }
    else {
      auto waitStart = std::chrono::steady_clock::now();
      bool gotAnything = queryQueue.waitPopExactN(resultBufs, desiredBatchSize);
      waitTaskSubmitMs = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - waitStart).count();
      if(!gotAnything)
        break;
      lock.lock();
      numGpuBusyClaims[schedulerGpuIdx] += 1;
      lock.unlock();
    }

    int numRows = (int)resultBufs.size();
    assert(numRows > 0);
    InferenceThreadActiveGuard inferenceThreadActiveGuard(GlobalPerfProfile::isEnabled(), serverThreadIdx);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        assert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        assert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;

        //At this point, these aren't probabilities, since this is before the postprocessing
        //that happens for each result. These just need to be unnormalized log probabilities.
        //Illegal move filtering happens later.
        for(int y = 0; y<boardYSize; y++) {
          for(int x = 0; x<boardXSize; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policyProbs[pos] = (float)rand.nextGaussian();
          }
        }
        policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,nnXLen,nnYLen)] = (float)rand.nextGaussian();

        resultBuf->result->nnXLen = nnXLen;
        resultBuf->result->nnYLen = nnYLen;
        if(resultBuf->includeOwnerMap) {
          float* whiteOwnerMap = new float[nnXLen*nnYLen];
          for(int i = 0; i<nnXLen*nnYLen; i++)
            whiteOwnerMap[i] = 0.0;
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              whiteOwnerMap[pos] = (float)rand.nextGaussian() * 0.20f;
            }
          }
          resultBuf->result->whiteOwnerMap = whiteOwnerMap;
        }
        else {
          resultBuf->result->whiteOwnerMap = NULL;
        }

        //These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMean = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMeanSq = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        double varTimeLeft = 0.5 * boardXSize * boardYSize;
        resultBuf->result->whiteWinProb = (float)whiteWinProb;
        resultBuf->result->whiteLossProb = (float)whiteLossProb;
        resultBuf->result->whiteNoResultProb = (float)whiteNoResultProb;
        resultBuf->result->whiteScoreMean = (float)whiteScoreMean;
        resultBuf->result->whiteScoreMeanSq = (float)whiteScoreMeanSq;
        resultBuf->result->whiteLead = (float)whiteScoreMean;
        resultBuf->result->varTimeLeft = (float)varTimeLeft;
        resultBuf->result->shorttermWinlossError = 0.0f;
        resultBuf->result->shorttermScoreError = 0.0f;
        resultBuf->result->policyOptimismUsed = (float)resultBuf->policyOptimism;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
      GlobalPerfProfile::recordRealtimeInferenceBatch(
        serverThreadIdx,
        gpuIdxForThisThread,
        numRows,
        numRows,
        waitTaskSubmitMs,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      );
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = new NNOutput();
        assert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        if(resultBufs[row]->includeOwnerMap)
          emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
        else
          emptyOutput->whiteOwnerMap = NULL;
        outputBuf.push_back(emptyOutput);
      }

      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize)
            resultBufs[row]->symmetry = rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
          else {
            assert(defaultSymmetry >= 0 && defaultSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1);
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      NNPerfInfo perfInfo;
      NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, resultBufs.data(), outputBuf, &perfInfo);
      assert(outputBuf.size() == numRows);

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;
      GlobalPerfProfile::recordRealtimeInferenceBatch(
        serverThreadIdx,
        perfInfo.gpuIdx,
        numRows,
        numRows,
        waitTaskSubmitMs,
        perfInfo.preprocessMs,
        perfInfo.h2dMs,
        perfInfo.inferMs,
        perfInfo.d2hMs,
        perfInfo.postprocessMs
      );

      for(int row = 0; row < numRows; row++) {
        assert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        assert(resultBuf->hasResult == false);
        resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }

    //Lock and update stats before looping again
    lock.lock();
    numGpuBusyClaims[schedulerGpuIdx] -= 1;
    numOngoingEvals -= numRows;

    if(numWaitingEvals > 0) {
      numEvalsToAwaken += numWaitingEvals;
      numWaitingEvals = 0;
      waitingForFinish.notify_all();
    }
    lock.unlock();
    continue;
  }

  NeuralNet::freeComputeHandle(gpuHandle);
  if(logger != NULL) {
    logger->write(
      "GPU " + Global::intToString(gpuIdxForThisThread) + " finishing, processed " +
      Global::int64ToString(numRowsHandledThisThread) + " rows " +
      Global::int64ToString(numBatchesHandledThisThread) + " batches"
    );
  }
}

void NNEvaluator::waitForNextNNEvalIfAny() {
  unique_lock<std::mutex> lock(bufferMutex);
  if(numOngoingEvals <= 0)
    return;

  numWaitingEvals++;
  while(numEvalsToAwaken <= 0 && !isKilled)
    waitingForFinish.wait(lock);
  numEvalsToAwaken--;
}


static double softPlus(double x) {
  //Avoid blowup
  if(x > 40.0)
    return x;
  else
    return log(1.0 + exp(x));
}

static const int daggerPattern[9][8] = {
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,2,1,0,0,0,0,0},
  {0,3,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
};
static bool daggerMatch(const Board& board, Player nextPla, Loc& banned, int symmetry) {
  for(int yi = 0; yi < 9; yi++) {
    for(int xi = 0; xi < 8; xi++) {
      int y = yi;
      int x = xi;
      if((symmetry & 0x1) != 0)
        std::swap(x,y);
      if((symmetry & 0x2) != 0)
        x = board.x_size-1-x;
      if((symmetry & 0x4) != 0)
        y = board.y_size-1-y;
      Loc loc = Location::getLoc(x,y,board.x_size);
      int m = daggerPattern[yi][xi];
      if(m == 0 && board.colors[loc] != C_EMPTY)
        return false;
      if(m == 1 && board.colors[loc] != nextPla)
        return false;
      if(m == 2 && board.colors[loc] != getOpp(nextPla))
        return false;
      if(m == 3)
        banned = loc;
    }
  }
  return true;
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& baseNNInputParams,
  NNResultBuf& buf,
  bool includeOwnerMap,
  Rand& rand,
  int numSymmetriesToSample
) {
  MiscNNInputParams nnInputParams = baseNNInputParams;
  vector<std::shared_ptr<NNOutput>> ptrs;
  std::array<int, SymmetryHelpers::NUM_SYMMETRIES> symmetryIndexes;
  std::iota(symmetryIndexes.begin(), symmetryIndexes.end(), 0);
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,SymmetryHelpers::NUM_SYMMETRIES-1)]);
    nnInputParams.symmetry = symmetryIndexes[i];
    bool skipCacheThisIteration = true; //Skip cache since there's no guarantee which symmetry is in the cache
    evaluate(
      board, history, nextPlayer, sgfMeta,
      nnInputParams,
      buf, skipCacheThisIteration, includeOwnerMap
    );
    ptrs.push_back(std::move(buf.result));
  }
  return new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
}

void NNEvaluator::evaluate(
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  evaluate(
    board,
    history,
    nextPlayer,
    NULL,
    nnInputParams,
    buf,
    skipCache,
    includeOwnerMap
  );
}

void NNEvaluator::evaluate(
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  assert(!isKilled);
  buf.hasResult = false;

  if(board.x_size > nnXLen || board.y_size > nnYLen)
    throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactNNLen) {
    if(board.x_size != nnXLen || board.y_size != nnYLen)
      throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                        " nnYLen = " + Global::intToString(nnYLen) +
                        " and requireExactNNLen, but was asked to evaluate board with different x or y size");
  }

  // Avoid using policy optimism for humanSL
  MiscNNInputParams nnInputParams = nnInputParamsArg;
  if(numInputMetaChannels > 0)
    nnInputParams.policyOptimism = 0.0;

  Hash128 nnHash = NNInputs::getHash(board, history, nextPlayer, nnInputParams);
  if(numInputMetaChannels > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    nnHash ^= sgfMeta->getHash(nextPlayer);
  }

  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL))
    {
      buf.hasResult = true;
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
    }
  }
  buf.includeOwnerMap = includeOwnerMap;

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;

  if(!debugSkipNeuralNet) {
    const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
    if(buf.rowSpatialBuf.size() < rowSpatialLen)
      buf.rowSpatialBuf.resize(rowSpatialLen);
    const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if(buf.rowGlobalBuf.size() < rowGlobalLen)
      buf.rowGlobalBuf.resize(rowGlobalLen);
    const int rowMetaLen = numInputMetaChannels;
    if(buf.rowMetaBuf.size() < rowMetaLen)
      buf.rowMetaBuf.resize(rowMetaLen);

    static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
    if(inputsVersion == 3)
      NNInputs::fillRowV3(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(inputsVersion == 4)
      NNInputs::fillRowV4(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(inputsVersion == 5)
      NNInputs::fillRowV5(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(inputsVersion == 6)
      NNInputs::fillRowV6(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else if(inputsVersion == 7)
      NNInputs::fillRowV7(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
    else
      ASSERT_UNREACHABLE;

    if(rowMetaLen > 0) {
      if(sgfMeta == NULL)
        Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
      if(!sgfMeta->initialized)
        Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
      SGFMetadata::fillMetadataRow(
        sgfMeta,
        buf.rowMetaBuf.data(),
        nextPlayer,
        board.x_size*board.y_size
      );
      buf.hasRowMeta = true;
    }
    else {
      buf.hasRowMeta = false;
    }
  }

  buf.symmetry = nnInputParams.symmetry;
  buf.policyOptimism = nnInputParams.policyOptimism;

  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals += 1;
  lock.unlock();

  buf.submittedToNNServer = true;
  bool suc = queryQueue.forcePush(&buf);
  assert(suc);

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  //Perform postprocessing on the result - turn the nn output into probabilities
  //As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  //and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  //and causing policy weights to be different, which would reduce performance of successive searches in a game
  //by making the successive searches distribute their playouts less coherently and using the cache more poorly.
  if(hadResultWithoutOwnerMap) {
    buf.result->whiteWinProb = resultWithoutOwnerMap->whiteWinProb;
    buf.result->whiteLossProb = resultWithoutOwnerMap->whiteLossProb;
    buf.result->whiteNoResultProb = resultWithoutOwnerMap->whiteNoResultProb;
    buf.result->whiteScoreMean = resultWithoutOwnerMap->whiteScoreMean;
    buf.result->whiteScoreMeanSq = resultWithoutOwnerMap->whiteScoreMeanSq;
    buf.result->whiteLead = resultWithoutOwnerMap->whiteLead;
    buf.result->varTimeLeft = resultWithoutOwnerMap->varTimeLeft;
    buf.result->shorttermWinlossError = resultWithoutOwnerMap->shorttermWinlossError;
    buf.result->shorttermScoreError = resultWithoutOwnerMap->shorttermScoreError;
    std::copy(resultWithoutOwnerMap->policyProbs, resultWithoutOwnerMap->policyProbs + NNPos::MAX_NN_POLICY_SIZE, buf.result->policyProbs);
    buf.result->policyOptimismUsed = (float)resultWithoutOwnerMap->policyOptimismUsed;
    buf.result->nnXLen = resultWithoutOwnerMap->nnXLen;
    buf.result->nnYLen = resultWithoutOwnerMap->nnYLen;
    assert(buf.result->whiteOwnerMap != NULL);
  }
  else {
    float* policy = buf.result->policyProbs;

    float policyOutputScaling = postProcessParams.outputScaleMultiplier / nnInputParams.nnPolicyTemperature;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    assert(nextPlayer == history.presumedNextMovePla);
    for(int i = 0; i<policySize; i++) {
      Loc loc = NNPos::posToLoc(i,xSize,ySize,nnXLen,nnYLen);
      isLegal[i] = history.isLegal(board,loc,nextPlayer);
    }

    if(nnInputParams.avoidMYTDaggerHack && xSize >= 13 && ySize >= 13) {
      for(int symmetry = 0; symmetry < 8; symmetry++) {
        Loc banned = Board::NULL_LOC;
        if(daggerMatch(board, nextPlayer, banned, symmetry)) {
          if(banned != Board::NULL_LOC) {
            isLegal[NNPos::locToPos(banned,xSize,nnXLen,nnYLen)] = false;
          }
        }
      }
    }

    for(int i = 0; i<policySize; i++) {
      float policyValue;
      if(isLegal[i]) {
        legalCount += 1;
        policyValue = policy[i] * policyOutputScaling;
      }
      else
        policyValue = -1e30f;

      policy[i] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }

    assert(legalCount > 0);

    float policySum = 0.0f;

    if(nnInputParams.enablePassingHacks) {
      //Cap passing prior policy at 95% (19x other moves)
      float maxPassPolicySumFactor = 19.0f;

      for(int i = 0; i<policySize-1; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
      int passPos = NNPos::locToPos(Board::PASS_LOC, xSize, nnXLen, nnYLen);
      assert(passPos == policySize-1);
      int i = passPos;
      policy[i] = std::max(1e-20f, std::min(exp(policy[i] - maxPolicy), policySum * maxPassPolicySumFactor));
      policySum += policy[i];
    }
    else {
      for(int i = 0; i<policySize; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
    }

    if(!isfinite(policySum)) {
      cout << "Got nonfinite for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nonfinite for policy sum");
    }

    //Somehow all legal moves rounded to 0 probability
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelFileName));
      }
      float uniform = 1.0f / legalCount;
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? uniform : -1.0f;
      }
    }
    //Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    //Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    buf.result->policyOptimismUsed = (float)nnInputParams.policyOptimism;

    //Fix up the value as well. Note that the neural net gives us back the value from the perspective
    //of the player so we need to negate that to make it the white value.
    if(modelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      //Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;

        //Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        if(!isfinite(probSum) || !isfinite(scoreValue)) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreValue << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }

    }
    else if(modelVersion >= 4) {
      double winProb;
      double lossProb;
      double noResultProb;
      double scoreMean;
      double scoreMeanSq;
      double lead;
      double varTimeLeft;
      double shorttermWinlossError;
      double shorttermScoreError;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;
        double scoreMeanPreScaled = buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier;
        double scoreStdevPreSoftplus = buf.result->whiteScoreMeanSq * postProcessParams.outputScaleMultiplier;
        double leadPreScaled = buf.result->whiteLead * postProcessParams.outputScaleMultiplier;
        double varTimeLeftPreSoftplus = buf.result->varTimeLeft * postProcessParams.outputScaleMultiplier;
        double shorttermWinlossErrorPreSoftplus = buf.result->shorttermWinlossError * postProcessParams.outputScaleMultiplier;
        double shorttermScoreErrorPreSoftplus = buf.result->shorttermScoreError * postProcessParams.outputScaleMultiplier;

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultLogits -= 100000.0;

        //Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultProb = 0.0;

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        scoreMean = scoreMeanPreScaled * postProcessParams.scoreMeanMultiplier;
        double scoreStdev = softPlus(scoreStdevPreSoftplus) * postProcessParams.scoreStdevMultiplier;
        scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
        lead = leadPreScaled * postProcessParams.leadMultiplier;
        varTimeLeft = softPlus(varTimeLeftPreSoftplus) * postProcessParams.varianceTimeMultiplier;

        //scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        //noResult counts as 0 score for scorevalue purposes.
        scoreMean = scoreMean * (1.0-noResultProb);
        scoreMeanSq = scoreMeanSq * (1.0-noResultProb);
        lead = lead * (1.0-noResultProb);

        if(modelVersion >= 14) {
          {
            double s = softPlus(shorttermWinlossErrorPreSoftplus * 0.5);
            shorttermWinlossError = sqrt(s * s * postProcessParams.shorttermValueErrorMultiplier);
          }
          {
            double s = softPlus(shorttermScoreErrorPreSoftplus * 0.5);
            shorttermScoreError = sqrt(s * s * postProcessParams.shorttermScoreErrorMultiplier);
          }
        }
        else if(modelVersion >= 10) {
          shorttermWinlossError = sqrt(softPlus(shorttermWinlossErrorPreSoftplus) * postProcessParams.shorttermValueErrorMultiplier);
          shorttermScoreError = sqrt(softPlus(shorttermScoreErrorPreSoftplus) * postProcessParams.shorttermScoreErrorMultiplier);
        }
        else {
          shorttermWinlossError = softPlus(shorttermWinlossErrorPreSoftplus);
          shorttermScoreError = softPlus(shorttermScoreErrorPreSoftplus) * 10.0;
        }

        if(
          !isfinite(probSum) ||
          !isfinite(scoreMean) ||
          !isfinite(scoreMeanSq) ||
          !isfinite(lead) ||
          !isfinite(varTimeLeft) ||
          !isfinite(shorttermWinlossError) ||
          !isfinite(shorttermScoreError)
        ) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits
               << " " << scoreMean << " " << scoreMeanSq
               << " " << lead << " " << varTimeLeft
               << " " << shorttermWinlossError << " " << shorttermScoreError
               << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = (float)lead;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = -(float)lead;
      }

      if(modelVersion >= 9) {
        buf.result->varTimeLeft = (float)varTimeLeft;
        buf.result->shorttermWinlossError = (float)shorttermWinlossError;
        buf.result->shorttermScoreError = (float)shorttermScoreError;
      }
      else {
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  //Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(modelVersion >= 3) {
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          //Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          //but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
          if(nextPlayer == P_WHITE)
            buf.result->whiteOwnerMap[pos] = tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
          else
            buf.result->whiteOwnerMap[pos] = -tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
        }
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }


  //And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}

//Uncomment this to lower the effective hash size down to one where we get true collisions
//#define SIMULATE_TRUE_HASH_COLLISIONS

NNCacheTable::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  //Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  //Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    //Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  //No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTable::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}
