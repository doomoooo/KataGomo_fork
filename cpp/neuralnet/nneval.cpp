#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/test.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <thread>

using namespace std;

namespace {
class ScopedComputeStream {
 public:
  explicit ScopedComputeStream(int gpuIdx)
    : stream(NeuralNet::createComputeStream(gpuIdx))
  {}
  ~ScopedComputeStream() {
    if(stream != NULL)
      NeuralNet::freeComputeStream(stream);
  }
  void* get() const { return stream; }
  ScopedComputeStream(const ScopedComputeStream&) = delete;
  ScopedComputeStream& operator=(const ScopedComputeStream&) = delete;
 private:
  void* stream;
};
}

//-------------------------------------------------------------------------------------

struct EventPipelineSchedulerState {
  struct BatchState {
    vector<NNResultBuf*> requests;
    vector<NNOutput*> outputs;
  };
  struct SlotState {
    int slotIdx = -1;
    int gpuIdx = -1;
    void* computeStream = NULL;
    ComputeHandle* gpuHandle = NULL;
    NNServerBuf* serverBuf = NULL;
    BatchState* front = NULL;
    BatchState* next = NULL;
    BatchState* submitting = NULL;
    bool usingFP16 = false;
    struct SubmitWorker {
      mutex taskMutex;
      condition_variable taskCondition;
      thread* workerThread = NULL;
      bool stop = false;
      bool hasTask = false;
      atomic<bool> taskDone{false};
      int batchSize = 0;
      exception_ptr error;
    };
    unique_ptr<SubmitWorker> submitWorker;
  };

  Rand rand;
  vector<SlotState> slots;
  BatchState* filling = NULL;
  int fillingSlotIdx = -1;
  int rrCursor = 0;
  bool startupFailed = false;
  string startupFailureMessage;

  explicit EventPipelineSchedulerState(const string& seed) : rand(seed) {}
};

static bool parseCudaAsyncInferPipeline(ConfigParser& cfg) {
  if(!cfg.contains("cudaAsyncInferPipeline"))
    return false;
  bool enabled = cfg.getBool("cudaAsyncInferPipeline");
#ifndef USE_CUDA_BACKEND
  if(enabled)
    throw StringError("cudaAsyncInferPipeline requires the CUDA backend");
#endif
  return enabled;
}

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

NNBatchingDispatcher::NNBatchingDispatcher(
  bool enabled_, const vector<int>& gpuIdxByServerThread_
)
  : enabled(enabled_),
    gpuIdxByServerThread(gpuIdxByServerThread_),
    mutex(),
    condition(),
    serverThreadHasActiveBatch(gpuIdxByServerThread_.size(),false)
{}

bool NNBatchingDispatcher::waitForBatch(
  ThreadSafeQueue<NNResultBuf*>& queue,
  vector<NNResultBuf*>& resultBufs,
  int maxBatchSize,
  const atomic<int>& currentBatchSize,
  int serverThreadIdx
) {
  testAssert(serverThreadIdx >= 0 && serverThreadIdx < (int)gpuIdxByServerThread.size());

  if(!enabled) {
    int desiredBatchSize = std::min(maxBatchSize,currentBatchSize.load(std::memory_order_acquire));
    return queue.waitPopUpToN(resultBufs,desiredBatchSize);
  }

  unique_lock<std::mutex> lock(mutex);
  while(true) {
    size_t queuedRows = queue.size();
    if(queuedRows > 0) {
      int desiredBatchSize = std::min(maxBatchSize,currentBatchSize.load(std::memory_order_acquire));
      bool deviceIsIdle = true;
      int gpuIdx = gpuIdxByServerThread[serverThreadIdx];
      if(gpuIdx < 0)
        gpuIdx = 0;
      for(int i = 0; i < (int)serverThreadHasActiveBatch.size(); i++) {
        int otherGpuIdx = gpuIdxByServerThread[i];
        if(otherGpuIdx < 0)
          otherGpuIdx = 0;
        if(otherGpuIdx == gpuIdx && serverThreadHasActiveBatch[i]) {
          deviceIsIdle = false;
          break;
        }
      }

      if(queuedRows >= (size_t)desiredBatchSize || deviceIsIdle) {
        bool gotAnything = queue.waitPopUpToN(resultBufs,desiredBatchSize);
        if(gotAnything) {
          testAssert(!serverThreadHasActiveBatch[serverThreadIdx]);
          serverThreadHasActiveBatch[serverThreadIdx] = true;
        }
        return gotAnything;
      }
    }
    else if(queue.isReadOnly()) {
      return false;
    }
    condition.wait(lock);
  }
}

void NNBatchingDispatcher::completeBatch(int serverThreadIdx) {
  if(!enabled)
    return;
  {
    lock_guard<std::mutex> lock(mutex);
    testAssert(serverThreadIdx >= 0 && serverThreadIdx < (int)serverThreadHasActiveBatch.size());
    testAssert(serverThreadHasActiveBatch[serverThreadIdx]);
    serverThreadHasActiveBatch[serverThreadIdx] = false;
  }
  condition.notify_all();
}

void NNBatchingDispatcher::notify() {
  if(!enabled)
    return;
  {
    lock_guard<std::mutex> lock(mutex);
  }
  condition.notify_all();
}

void NNBatchingDispatcher::resetGpuIdxByServerThread(
  const vector<int>& gpuIdxByServerThread_
) {
  lock_guard<std::mutex> lock(mutex);
  for(bool active : serverThreadHasActiveBatch)
    testAssert(!active);
  gpuIdxByServerThread = gpuIdxByServerThread_;
  serverThreadHasActiveBatch.assign(gpuIdxByServerThread.size(),false);
  condition.notify_all();
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
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry,
  bool disableWarmup_,
  ConfigParser& cfg
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   numThreads(numThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   disableWarmup(disableWarmup_),
   warmupOnlyMaxBatchSize(
     cfg.contains("cudaWarmupOnlyMaxBatchSize") ? cfg.getBool("cudaWarmupOnlyMaxBatchSize") : false
   ),
   batchAwareDispatch(
     cfg.contains("nnBatchAwareDispatch") ? cfg.getBool("nnBatchAwareDispatch") : false
   ),
   cudaAsyncInferPipeline(parseCudaAsyncInferPipeline(cfg)),
   cudaEventPipelineUseGraph(
     cfg.contains("cudaEventPipelineUseGraph") ? cfg.getBool("cudaEventPipelineUseGraph") : false
   ),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(NULL),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   eventPipelineSchedulerState(NULL),
   maxBatchSize(maxBatchSz),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_numCacheHits(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   serverThreadsIsUsingFP16(),
   m_numRowsProcessedByServerThread(numThr,0),
   m_numBatchesProcessedByServerThread(numThr,0),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   currentBatchSize(maxBatchSz),
   queryQueue(),
   batchingDispatcher(batchAwareDispatch,gpuIdxByServerThread)
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");
  if(cudaEventPipelineUseGraph && !cudaAsyncInferPipeline)
    throw StringError("cudaEventPipelineUseGraph requires cudaAsyncInferPipeline=true");
  if(cudaEventPipelineUseGraph && !batchAwareDispatch)
    throw StringError("cudaEventPipelineUseGraph requires nnBatchAwareDispatch=true");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
    logger->write(
      "NN batch-aware dispatch is " + string(batchAwareDispatch ? "enabled" : "disabled")
    );
    if(batchAwareDispatch) {
      logger->write(
        "NN fixed-shape padding is enabled; backend launches always use batch size " +
        Global::intToString(maxBatchSize)
      );
    }
    logger->write(
      "CUDA event-gated inference pipeline is " +
      string(cudaAsyncInferPipeline ? "enabled" : "disabled")
    );
    if(cudaEventPipelineUseGraph)
      logger->write("CUDA exact-shape event-pipeline graph replay is enabled");
    if(cudaAsyncInferPipeline && numThreads < 2)
      logger->write("WARNING: CUDA event pipeline has one infer stream; overlap is limited");
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
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    postProcessParams = desc.postProcessParams;
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      homeDataDirOverride,
      usingFP16Mode,loadedModel,cfg
    );
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  // Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByServerThread.size());
  // Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();

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
  batchingDispatcher.notify();
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
bool NNEvaluator::getRequireExactNNLen() const {
  return requireExactNNLen;
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

bool NNEvaluator::supportsShorttermError() const {
  return modelVersion >= 9;
}

bool NNEvaluator::modelPreferPassAliveUnderSuicideRules() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferPassAliveUnderSuicideRules;
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

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) const {
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
vector<uint64_t> NNEvaluator::numRowsProcessedByServerThread() const {
  lock_guard<std::mutex> lock(bufferMutex);
  return m_numRowsProcessedByServerThread;
}
vector<uint64_t> NNEvaluator::numBatchesProcessedByServerThread() const {
  lock_guard<std::mutex> lock(bufferMutex);
  return m_numBatchesProcessedByServerThread;
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}
uint64_t NNEvaluator::numCacheHits() const {
  return m_numCacheHits.load(std::memory_order_relaxed);
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
  m_numCacheHits.store(0);
  lock_guard<std::mutex> lock(bufferMutex);
  std::fill(m_numRowsProcessedByServerThread.begin(),m_numRowsProcessedByServerThread.end(),0);
  std::fill(m_numBatchesProcessedByServerThread.begin(),m_numBatchesProcessedByServerThread.end(),0);
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

#ifdef USE_CUDA_BACKEND
void NNEvaluator::serveEventPipelineScheduler(const string& randSeedThisThread) {
  (void)randSeedThisThread;
  EventPipelineSchedulerState* state = eventPipelineSchedulerState;
  testAssert(state != NULL);

  auto canonicalGpuIdx = [](int gpuIdx) { return gpuIdx < 0 ? 0 : gpuIdx; };
  auto deleteBatch = [](EventPipelineSchedulerState::BatchState*& batch) {
    if(batch == NULL)
      return;
    for(NNOutput* output : batch->outputs)
      delete output;
    delete batch;
    batch = NULL;
  };
  auto deviceIsIdle = [&](int gpuIdx) {
    int canonical = canonicalGpuIdx(gpuIdx);
    for(const EventPipelineSchedulerState::SlotState& slot : state->slots) {
      if(canonicalGpuIdx(slot.gpuIdx) == canonical &&
         (slot.front != NULL || slot.next != NULL || slot.submitting != NULL))
        return false;
    }
    return true;
  };
  auto slotCanAccept = [&](EventPipelineSchedulerState::SlotState& slot) {
    return slot.next == NULL && slot.submitting == NULL &&
      NeuralNet::eventPipelineInputHostReusable(slot.gpuHandle);
  };
  auto selectFillingSlot = [&]() {
    int residentBatches = 0;
    for(const EventPipelineSchedulerState::SlotState& slot : state->slots) {
      residentBatches += slot.front != NULL ? 1 : 0;
      residentBatches += slot.next != NULL ? 1 : 0;
      residentBatches += slot.submitting != NULL ? 1 : 0;
    }
    if(residentBatches >= (int)state->slots.size() + 1)
      return -1;

    for(int offset = 0; offset < (int)state->slots.size(); offset++) {
      int idx = (state->rrCursor + offset) % (int)state->slots.size();
      EventPipelineSchedulerState::SlotState& slot = state->slots[idx];
      if(slot.front == NULL && slot.next == NULL && slotCanAccept(slot)) {
        state->rrCursor = (idx + 1) % (int)state->slots.size();
        return idx;
      }
    }
    for(int offset = 0; offset < (int)state->slots.size(); offset++) {
      int idx = (state->rrCursor + offset) % (int)state->slots.size();
      EventPipelineSchedulerState::SlotState& slot = state->slots[idx];
      if(slotCanAccept(slot)) {
        state->rrCursor = (idx + 1) % (int)state->slots.size();
        return idx;
      }
    }
    return -1;
  };
  auto allocateOutputs = [&](EventPipelineSchedulerState::BatchState& batch) {
    batch.outputs.reserve(batch.requests.size());
    for(NNResultBuf* request : batch.requests) {
      NNOutput* output = new NNOutput();
      output->nnXLen = nnXLen;
      output->nnYLen = nnYLen;
      output->whiteOwnerMap = request->includeOwnerMap ? new float[nnXLen*nnYLen] : NULL;
      batch.outputs.push_back(output);
    }
  };
  auto beginInferenceSubmission = [&](EventPipelineSchedulerState::SlotState& slot, int batchSize) {
    EventPipelineSchedulerState::SlotState::SubmitWorker* worker = slot.submitWorker.get();
    testAssert(worker != NULL && worker->workerThread != NULL);
    unique_lock<std::mutex> lock(worker->taskMutex);
    testAssert(!worker->hasTask);
    testAssert(!worker->taskDone.load(std::memory_order_relaxed));
    worker->batchSize = batchSize;
    worker->error = exception_ptr();
    worker->hasTask = true;
    worker->taskCondition.notify_all();
  };
  auto consumeInferenceSubmission = [&](EventPipelineSchedulerState::SlotState& slot) {
    EventPipelineSchedulerState::SlotState::SubmitWorker* worker = slot.submitWorker.get();
    testAssert(worker != NULL && worker->workerThread != NULL);
    unique_lock<std::mutex> lock(worker->taskMutex);
    testAssert(worker->taskDone.load(std::memory_order_acquire));
    exception_ptr error = worker->error;
    worker->error = exception_ptr();
    worker->taskDone.store(false,std::memory_order_relaxed);
    lock.unlock();
    if(error != NULL)
      rethrow_exception(error);
  };
  auto waitForInferenceSubmission = [&](EventPipelineSchedulerState::SlotState& slot) {
    EventPipelineSchedulerState::SlotState::SubmitWorker* worker = slot.submitWorker.get();
    testAssert(worker != NULL && worker->workerThread != NULL);
    {
      unique_lock<std::mutex> lock(worker->taskMutex);
      while(!worker->taskDone.load(std::memory_order_acquire))
        worker->taskCondition.wait(lock);
    }
    consumeInferenceSubmission(slot);
  };
  auto maybeFinishInferenceSubmission = [&](EventPipelineSchedulerState::SlotState& slot) {
    if(slot.submitting == NULL)
      return false;
    EventPipelineSchedulerState::SlotState::SubmitWorker* worker = slot.submitWorker.get();
    testAssert(worker != NULL && worker->workerThread != NULL);
    if(!worker->taskDone.load(std::memory_order_acquire))
      return false;

    consumeInferenceSubmission(slot);
    EventPipelineSchedulerState::BatchState* batch = slot.submitting;
    slot.submitting = NULL;
    if(slot.front == NULL) {
      NeuralNet::enqueueEventPipelineOutput(
        slot.gpuHandle,slot.serverBuf->inputBuffers,(int)batch->requests.size()
      );
      slot.front = batch;
    }
    else {
      testAssert(slot.next == NULL);
      slot.next = batch;
    }
    return true;
  };
  auto launchFillingBatch = [&]() {
    testAssert(state->filling != NULL && state->fillingSlotIdx >= 0);
    EventPipelineSchedulerState::SlotState& slot = state->slots[state->fillingSlotIdx];
    EventPipelineSchedulerState::BatchState* batch = state->filling;
    testAssert(!batch->requests.empty());
    testAssert(slotCanAccept(slot));

    for(NNResultBuf* request : batch->requests) {
      if(request->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
        if(currentDoRandomize.load(std::memory_order_acquire))
          request->symmetry = state->rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
        else
          request->symmetry = currentDefaultSymmetry.load(std::memory_order_acquire);
      }
    }
    allocateOutputs(*batch);
    const int requestBatchSize = (int)batch->requests.size();
    const int inferenceBatchSize = batchAwareDispatch ? maxBatchSize : requestBatchSize;
    vector<NNResultBuf*> inferenceRequests;
    NNResultBuf** inferenceRequestData = batch->requests.data();
    if(inferenceBatchSize > requestBatchSize) {
      inferenceRequests = batch->requests;
      inferenceRequests.resize(inferenceBatchSize,batch->requests.back());
      inferenceRequestData = inferenceRequests.data();
    }
    NeuralNet::prepareEventPipelineInput(
      slot.gpuHandle,slot.serverBuf->inputBuffers,inferenceBatchSize,inferenceRequestData
    );
    // Each lane has a persistent host worker. Submission is deliberately
    // nonblocking so independent CUDA streams can be filled concurrently.
    testAssert(slot.submitting == NULL);
    slot.submitting = batch;
    beginInferenceSubmission(slot,inferenceBatchSize);
    state->filling = NULL;
    state->fillingSlotIdx = -1;
  };
  auto maybeLaunchFillingBatch = [&]() {
    if(state->filling == NULL || state->filling->requests.empty())
      return false;
    EventPipelineSchedulerState::SlotState& slot = state->slots[state->fillingSlotIdx];
    int desiredBatchSize = std::min(maxBatchSize,currentBatchSize.load(std::memory_order_acquire));
    bool full = (int)state->filling->requests.size() >= desiredBatchSize;
    bool shouldLaunch = batchAwareDispatch ?
      full || deviceIsIdle(slot.gpuIdx) : full || queryQueue.size() == 0;
    if(!shouldLaunch || !slotCanAccept(slot))
      return false;
    launchFillingBatch();
    return true;
  };
  auto finalizeFront = [&](EventPipelineSchedulerState::SlotState& slot) {
    EventPipelineSchedulerState::BatchState* completed = slot.front;
    testAssert(completed != NULL);
    int batchSize = (int)completed->requests.size();
    NeuralNet::finishEventPipelineOutput(
      slot.gpuHandle,slot.serverBuf->inputBuffers,batchSize,
      completed->requests.data(),completed->outputs
    );

    m_numRowsProcessed.fetch_add(batchSize,std::memory_order_relaxed);
    m_numBatchesProcessed.fetch_add(1,std::memory_order_relaxed);
    {
      lock_guard<std::mutex> lock(bufferMutex);
      testAssert(slot.slotIdx >= 0 &&
                 slot.slotIdx < (int)m_numRowsProcessedByServerThread.size());
      m_numRowsProcessedByServerThread[slot.slotIdx] += batchSize;
      m_numBatchesProcessedByServerThread[slot.slotIdx] += 1;
    }
    for(int row = 0; row < batchSize; row++) {
      NNResultBuf* resultBuf = completed->requests[row];
      unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
      resultBuf->result = shared_ptr<NNOutput>(completed->outputs[row]);
      completed->outputs[row] = NULL;
      resultBuf->hasResult = true;
      resultBuf->clientWaitingForResult.notify_all();
    }
    deleteBatch(slot.front);

    {
      unique_lock<std::mutex> lock(bufferMutex);
      numOngoingEvals -= batchSize;
      if(numWaitingEvals > 0) {
        numEvalsToAwaken += numWaitingEvals;
        numWaitingEvals = 0;
        waitingForFinish.notify_all();
      }
    }

    if(slot.next != NULL) {
      NeuralNet::enqueueEventPipelineOutput(
        slot.gpuHandle,slot.serverBuf->inputBuffers,(int)slot.next->requests.size()
      );
      slot.front = slot.next;
      slot.next = NULL;
    }
  };

  bool startupComplete = false;
  try {
    state->slots.resize(gpuIdxByServerThread.size());
    for(size_t i = 0; i < gpuIdxByServerThread.size(); i++) {
      EventPipelineSchedulerState::SlotState& slot = state->slots[i];
      slot.slotIdx = (int)i;
      slot.gpuIdx = gpuIdxByServerThread[i];
      slot.serverBuf = new NNServerBuf(*this,loadedModel);
      slot.computeStream = NeuralNet::createComputeStream(slot.gpuIdx);
      slot.gpuHandle = NeuralNet::createComputeHandle(
        computeContext,loadedModel,logger,maxBatchSize,requireExactNNLen,
        inputsUseNHWC,slot.gpuIdx,slot.slotIdx,slot.computeStream
      );
      maybeWarmupComputeHandle(slot.gpuHandle,slot.slotIdx);
      NeuralNet::enableEventGatedPipeline(slot.gpuHandle,slot.serverBuf->inputBuffers);
      slot.usingFP16 = NeuralNet::isUsingFP16(slot.gpuHandle);
      slot.submitWorker = std::make_unique<EventPipelineSchedulerState::SlotState::SubmitWorker>();
      EventPipelineSchedulerState::SlotState::SubmitWorker* worker = slot.submitWorker.get();
      EventPipelineSchedulerState::SlotState* slotPtr = &slot;
      worker->workerThread = new thread([slotPtr,worker]() {
        unique_lock<std::mutex> lock(worker->taskMutex);
        while(true) {
          while(!worker->stop && !worker->hasTask)
            worker->taskCondition.wait(lock);
          if(worker->stop)
            break;
          int batchSize = worker->batchSize;
          worker->hasTask = false;
          lock.unlock();
          try {
            NeuralNet::launchEventPipelineInference(
              slotPtr->gpuHandle,slotPtr->serverBuf->inputBuffers,batchSize
            );
          }
          catch(...) {
            lock.lock();
            worker->error = current_exception();
            worker->taskDone.store(true,std::memory_order_release);
            worker->taskCondition.notify_all();
            continue;
          }
          lock.lock();
          worker->taskDone.store(true,std::memory_order_release);
          worker->taskCondition.notify_all();
        }
      });
      if(cudaEventPipelineUseGraph) {
        beginInferenceSubmission(slot,maxBatchSize);
        waitForInferenceSubmission(slot);
        NeuralNet::enqueueEventPipelineOutput(
          slot.gpuHandle,slot.serverBuf->inputBuffers,maxBatchSize
        );
        while(!NeuralNet::eventPipelineOutputReady(slot.gpuHandle))
          std::this_thread::yield();
      }
    }

    {
      lock_guard<std::mutex> lock(bufferMutex);
      serverThreadsIsUsingFP16.assign(state->slots.size(),0);
      for(const EventPipelineSchedulerState::SlotState& slot : state->slots)
        serverThreadsIsUsingFP16[slot.slotIdx] = slot.usingFP16 ? 1 : 0;
      numServerThreadsStartingUp--;
      mainThreadWaitingForSpawn.notify_all();
    }
    startupComplete = true;
    if(logger != NULL)
      logger->write("CUDA event-gated single-slot scheduler started");

    NNResultBuf* deferredRequest = NULL;
    while(true) {
      for(EventPipelineSchedulerState::SlotState& slot : state->slots)
        (void)maybeFinishInferenceSubmission(slot);

      for(EventPipelineSchedulerState::SlotState& slot : state->slots) {
        if(slot.front != NULL && NeuralNet::eventPipelineOutputReady(slot.gpuHandle))
          finalizeFront(slot);
      }

      (void)maybeLaunchFillingBatch();

      NNResultBuf* request = deferredRequest;
      if(request == NULL) {
        NNResultBuf* popped = NULL;
        if(queryQueue.tryPop(popped))
          request = popped;
      }
      deferredRequest = NULL;
      if(request != NULL) {
        if(state->filling == NULL) {
          int slotIdx = selectFillingSlot();
          if(slotIdx < 0)
            deferredRequest = request;
          else {
            state->filling = new EventPipelineSchedulerState::BatchState();
            state->fillingSlotIdx = slotIdx;
          }
        }
        if(deferredRequest == NULL) {
          int desiredBatchSize = std::min(maxBatchSize,currentBatchSize.load(std::memory_order_acquire));
          if((int)state->filling->requests.size() >= desiredBatchSize) {
            deferredRequest = request;
            (void)maybeLaunchFillingBatch();
          }
          else {
            state->filling->requests.push_back(request);
            (void)maybeLaunchFillingBatch();
          }
        }
      }

      bool allSlotsIdle = true;
      for(const EventPipelineSchedulerState::SlotState& slot : state->slots) {
        if(slot.front != NULL || slot.next != NULL || slot.submitting != NULL) {
          allSlotsIdle = false;
          break;
        }
      }
      if(queryQueue.isReadOnly() && deferredRequest == NULL &&
         state->filling == NULL && allSlotsIdle)
        break;
      std::this_thread::yield();
    }
  }
  catch(const exception& e) {
    if(!startupComplete) {
      lock_guard<std::mutex> lock(bufferMutex);
      state->startupFailed = true;
      state->startupFailureMessage = e.what();
      numServerThreadsStartingUp = 0;
      mainThreadWaitingForSpawn.notify_all();
    }
    else
      Global::fatalError(string("CUDA event pipeline scheduler failed: ") + e.what());
  }

  deleteBatch(state->filling);
  for(EventPipelineSchedulerState::SlotState& slot : state->slots) {
    deleteBatch(slot.front);
    deleteBatch(slot.next);
    deleteBatch(slot.submitting);
    if(slot.submitWorker != NULL && slot.submitWorker->workerThread != NULL) {
      {
        lock_guard<std::mutex> lock(slot.submitWorker->taskMutex);
        slot.submitWorker->stop = true;
        slot.submitWorker->taskCondition.notify_all();
      }
      slot.submitWorker->workerThread->join();
      delete slot.submitWorker->workerThread;
      slot.submitWorker->workerThread = NULL;
    }
    if(slot.gpuHandle != NULL)
      NeuralNet::freeComputeHandle(slot.gpuHandle);
    if(slot.computeStream != NULL)
      NeuralNet::freeComputeStream(slot.computeStream);
    delete slot.serverBuf;
    slot.gpuHandle = NULL;
    slot.computeStream = NULL;
    slot.serverBuf = NULL;
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

  // Used to have a try catch around this but actually we're in big trouble if this raises an exception
  // and causes possibly the only nnEval thread to die, so actually go ahead and let the exception escape to
  // toplevel for easier debugging
  nnEval->serve(*buf,rand,gpuIdxForThisThread,serverThreadIdx);
  delete buf;
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
  batchingDispatcher.resetGpuIdxByServerThread(gpuIdxByServerThr);
  lock_guard<std::mutex> lock(bufferMutex);
  m_numRowsProcessedByServerThread.assign(numThreads,0);
  m_numBatchesProcessedByServerThread.assign(numThreads,0);
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(numThreads,0);
  }

  queryQueue.unsetReadOnly();

  bool useEventPipelineScheduler = false;
#ifdef USE_CUDA_BACKEND
  useEventPipelineScheduler = cudaAsyncInferPipeline && !debugSkipNeuralNet;
#endif
  if(useEventPipelineScheduler)
    eventPipelineSchedulerState = new EventPipelineSchedulerState(randSeed + ":EventPipelineScheduler");

  if(useEventPipelineScheduler) {
    numServerThreadsStartingUp = 1;
    string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread([this,randSeedThisThread]() {
#ifdef USE_CUDA_BACKEND
      serveEventPipelineScheduler(randSeedThisThread);
#endif
    });
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
  bool startupFailed = eventPipelineSchedulerState != NULL && eventPipelineSchedulerState->startupFailed;
  string startupFailureMessage = startupFailed ?
    eventPipelineSchedulerState->startupFailureMessage : string();
  lock.unlock();
  if(startupFailed) {
    killServerThreads();
    throw StringError("Failed to start CUDA event pipeline scheduler: " + startupFailureMessage);
  }
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();
  batchingDispatcher.notify();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();
  delete eventPipelineSchedulerState;
  eventPipelineSchedulerState = NULL;

  // Can unset now that threads are dead
  isKilled = false;

  testAssert(numOngoingEvals == 0);
  testAssert(numWaitingEvals == 0);
  testAssert(numEvalsToAwaken == 0);
}

NNEvalBenchmarkResult NNEvaluator::benchmarkPureForward(
  int numWarmups, int numIterations, int phaseOffsetMicros
) {
  if(numIterations <= 0)
    throw StringError("benchmarknn requires numIterations > 0");
  if(numWarmups < 0)
    throw StringError("benchmarknn requires numWarmups >= 0");
  if(phaseOffsetMicros < -1)
    throw StringError("benchmarknn phase offset must be -1 or nonnegative");
  if(debugSkipNeuralNet || loadedModel == NULL || computeContext == NULL)
    throw StringError("benchmarknn requires a real neural net model");

  const int numServerThreads = (int)gpuIdxByServerThread.size();
  const int batchSize = maxBatchSize;
  if(numServerThreads <= 0 || batchSize <= 0)
    throw StringError("benchmarknn: invalid server/batch topology");

  NNEvalBenchmarkResult result;
  result.batchSize = batchSize;
  result.numServerThreads = numServerThreads;
  result.numIterations = numIterations;
  result.phaseOffsetMicros = phaseOffsetMicros;
  result.perServerIterationSeconds.assign(numServerThreads, {});
  result.perServerMedianSeconds.assign(numServerThreads, 0.0);
  result.perServerNNEvalsPerSec.assign(numServerThreads, 0.0);
  result.combinedWallSeconds = 0.0;
  result.combinedNNEvalsPerSec = 0.0;
  result.actualWallSeconds = 0.0;
  result.actualWallPerForwardMs = 0.0;

  std::exception_ptr firstError;
  std::mutex errorMutex;
  std::atomic<bool> anyError(false);

  std::atomic<int> readyCount(0);
  std::atomic<bool> startFlag(false);
  std::unique_ptr<BenchmarkForwardBarrier> phaseBarrier;
  if(phaseOffsetMicros >= 0)
    phaseBarrier = std::make_unique<BenchmarkForwardBarrier>(numServerThreads);
  std::chrono::steady_clock::time_point wallStart;
  std::chrono::steady_clock::time_point wallEnd;

  std::vector<std::thread> threads;
  threads.reserve(numServerThreads);
  for(int threadIdx = 0; threadIdx < numServerThreads; threadIdx++) {
    threads.emplace_back([&, threadIdx]() {
      try {
        // Mirror NNEvaluator::serve: one externally owned compute stream and one compute handle per
        // configured NN server thread.
        NNServerBuf serverBuf(*this, loadedModel);
        ScopedComputeStream computeStream(gpuIdxByServerThread[threadIdx]);
        ComputeHandle* handle = NULL;
        try {
          handle = NeuralNet::createComputeHandle(
            computeContext,
            loadedModel,
            logger,
            maxBatchSize,
            requireExactNNLen,
            inputsUseNHWC,
            gpuIdxByServerThread[threadIdx],
            threadIdx,
            computeStream.get()
          );
          maybeWarmupComputeHandle(handle, threadIdx);

          // Fill all rows with a fixed empty board so every iteration runs the same real inputs.
          Board board(nnXLen, nnYLen);
          BoardHistory history(
            board, P_BLACK, Rules::getTrompTaylorish(), 0,
            modelPreferPassAliveUnderSuicideRules()
          );
          MiscNNInputParams nnInputParams;
          SGFMetadata sgfMeta;
          const SGFMetadata* sgfMetaPtr = NULL;
          if(numInputMetaChannels > 0) {
            sgfMeta = SGFMetadata::makeDummyWarmupProfile();
            sgfMetaPtr = &sgfMeta;
          }

          std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
          std::vector<NNResultBuf*> resultBufs;
          ownedBufs.reserve(batchSize);
          resultBufs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            ownedBufs.push_back(std::make_unique<NNResultBuf>());
            NNResultBuf* buf = ownedBufs.back().get();
            fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
            buf->symmetry = 0;
            buf->policyOptimism = 0.0f;
            resultBufs.push_back(buf);
          }

          // One full getOutput populates the host input arrays and warms lazy graph compilation.
          // Its H2D/D2H/postprocessing are all excluded from the timed benchmark below.
          std::vector<NNOutput*> outputs;
          outputs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            NNOutput* out = new NNOutput();
            out->nnXLen = nnXLen;
            out->nnYLen = nnYLen;
            out->whiteOwnerMap = NULL;
            outputs.push_back(out);
          }
          NeuralNet::getOutput(handle, serverBuf.inputBuffers, batchSize, resultBufs.data(), outputs);
          for(NNOutput* out : outputs)
            delete out;

          readyCount.fetch_add(1);
          while(!startFlag.load())
            std::this_thread::yield();

          std::vector<double> times;
#if defined(USE_CUDA_BACKEND) || defined(USE_TENSORRT_BACKEND)
          if(!NeuralNet::benchmarkOutput(
               handle, serverBuf.inputBuffers, batchSize, numWarmups, numIterations, times,
               phaseBarrier.get(), threadIdx, phaseOffsetMicros
             )) {
            throw StringError("Current backend does not support pure-device benchmarknn");
          }
#else
          throw StringError("benchmarknn requires a CUDA or TensorRT build");
#endif
          result.perServerIterationSeconds[threadIdx] = std::move(times);
        }
        catch(...) {
          if(handle != NULL)
            NeuralNet::freeComputeHandle(handle);
          throw;
        }
        NeuralNet::freeComputeHandle(handle);
      }
      catch(...) {
        anyError.store(true);
        std::lock_guard<std::mutex> lock(errorMutex);
        if(firstError == nullptr)
          firstError = std::current_exception();
      }
    });
  }

  while(readyCount.load() < numServerThreads && !anyError.load())
    std::this_thread::yield();
  wallStart = std::chrono::steady_clock::now();
  startFlag.store(true);

  for(std::thread& t : threads)
    t.join();
  wallEnd = std::chrono::steady_clock::now();

  if(firstError != nullptr)
    std::rethrow_exception(firstError);

  result.actualWallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
  result.actualWallPerForwardMs =
    result.actualWallSeconds / (double)(numWarmups + numIterations) * 1000.0;

  double combinedNNEvalsPerSec = 0.0;
  double maxMedianSeconds = 0.0;
  for(int threadIdx = 0; threadIdx < numServerThreads; threadIdx++) {
    std::vector<double>& times = result.perServerIterationSeconds[threadIdx];
    if(times.size() != (size_t)numIterations)
      throw StringError("benchmarknn: server thread produced an unexpected number of timings");
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double median =
      sorted.size() % 2 == 1
      ? sorted[sorted.size() / 2]
      : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    result.perServerMedianSeconds[threadIdx] = median;
    result.perServerNNEvalsPerSec[threadIdx] = batchSize / median;
    combinedNNEvalsPerSec += result.perServerNNEvalsPerSec[threadIdx];
    maxMedianSeconds = std::max(maxMedianSeconds, median);
  }

  // Per-server medians are measured while all servers run concurrently, so total throughput is the
  // sum of per-server throughputs. combinedWallSeconds is the per-batch wall time of the slowest
  // server (i.e. one concurrent batch), deliberately excluding model load and warmup.
  result.combinedWallSeconds = maxMedianSeconds;
  result.combinedNNEvalsPerSec = combinedNNEvalsPerSec;

  return result;
}

void NNEvaluator::fillRowBufs(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf
) const {
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

void NNEvaluator::maybeWarmupComputeHandle(ComputeHandle* gpuHandle, int serverThreadIdx) {
  if(disableWarmup || gpuHandle == NULL || debugSkipNeuralNet || loadedModel == NULL)
    return;
  // Warmup currently only matters on CUDA, where cuDNN lazily compiles an SDPA execution plan per
  // batch size on first use. Other backends: nothing to warm up for now.
#if !defined(USE_CUDA_BACKEND)
  (void)serverThreadIdx;
  return;
#else
  // Only transformer models build the lazy SDPA graphs; skip the (otherwise harmless but wasteful)
  // warmup passes for plain convnets.
  if(!NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks())
    return;

  if(logger != NULL) {
    string batchRange = warmupOnlyMaxBatchSize ?
      Global::intToString(maxBatchSize) :
      "1.." + Global::intToString(maxBatchSize);
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) +
      ": warming up transformer graphs for batch sizes " + batchRange
    );
  }

  // Empty board of the configured size, default rules/params. Outputs are discarded; we only want
  // the forward passes to trigger graph compilation for every batch size that will be seen.
  Board board(nnXLen, nnYLen);
  //Featurize the way this model expects (a no-op under Tromp-Taylorish rules, but robust if the
  //warmup rules ever change).
  BoardHistory history(board, P_BLACK, Rules::getTrompTaylorish(), 0, modelPreferPassAliveUnderSuicideRules());
  MiscNNInputParams nnInputParams;
  SGFMetadata sgfMeta;
  const SGFMetadata* sgfMetaPtr = NULL;
  if(numInputMetaChannels > 0) {
    sgfMeta = SGFMetadata::makeDummyWarmupProfile();
    sgfMetaPtr = &sgfMeta;
  }

  // Mark the handle as warming up so the backend treats lazy-graph-compilation failures (e.g. cudnn
  // SDPA) leniently, falling back to a custom kernel instead of failing hard. Restored when done.
  bool prevIsWarmup = NeuralNet::setIsWarmup(gpuHandle, true);

  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);

  // Reusable per-row input; identical for every row since it's an empty board.
  std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
  std::vector<NNResultBuf*> resultBufs;
  ownedBufs.reserve(maxBatchSize);
  resultBufs.reserve(maxBatchSize);
  for(int i = 0; i < maxBatchSize; i++) {
    ownedBufs.push_back(std::make_unique<NNResultBuf>());
    NNResultBuf* buf = ownedBufs.back().get();
    fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
    buf->symmetry = 0;
    buf->policyOptimism = nnInputParams.policyOptimism;
    resultBufs.push_back(buf);
  }

  int firstBatchSize = warmupOnlyMaxBatchSize ? maxBatchSize : 1;
  for(int batchSize = firstBatchSize; batchSize <= maxBatchSize; batchSize++) {
    std::vector<NNOutput*> outputs;
    outputs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      NNOutput* out = new NNOutput();
      out->nnXLen = nnXLen;
      out->nnYLen = nnYLen;
      out->whiteOwnerMap = NULL;
      outputs.push_back(out);
    }
    NeuralNet::getOutput(gpuHandle, inputBuffers, batchSize, resultBufs.data(), outputs);
    for(NNOutput* out : outputs)
      delete out;
  }

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::setIsWarmup(gpuHandle, prevIsWarmup);
#endif
}

void NNEvaluator::serve(
  NNServerBuf& buf, Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  ScopedComputeStream computeStream(gpuIdxForThisThread);
  if(loadedModel != NULL) {
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx,
      computeStream.get()
    );

    // Warm up lazily-compiled backend graphs before reporting this thread as started.
    maybeWarmupComputeHandle(gpuHandle, serverThreadIdx);
  }

  {
    lock_guard<std::mutex> lock(bufferMutex);
    testAssert(serverThreadIdx < serverThreadsIsUsingFP16.size());
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
    bool gotAnything = batchingDispatcher.waitForBatch(
      queryQueue,resultBufs,maxBatchSize,currentBatchSize,serverThreadIdx
    );
    // Queue being closed is a signal that we're done.
    if(!gotAnything)
      break;

    int numRows = (int)resultBufs.size();
    testAssert(numRows > 0);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;

        // At this point, these aren't probabilities, since this is before the postprocessing
        // that happens for each result. These just need to be unnormalized log probabilities.
        // Illegal move filtering happens later.
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

        // These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
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
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = new NNOutput();
        testAssert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        if(resultBufs[row]->includeOwnerMap)
          emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
        else
          emptyOutput->whiteOwnerMap = NULL;
        outputBuf.push_back(emptyOutput);
      }

      const int inferenceRows = batchAwareDispatch ? maxBatchSize : numRows;
      vector<NNResultBuf*> inferenceResultBufs;
      NNResultBuf** inferenceResultData = resultBufs.data();
      if(inferenceRows > numRows) {
        inferenceResultBufs = resultBufs;
        inferenceResultBufs.resize(inferenceRows,resultBufs.back());
        inferenceResultData = inferenceResultBufs.data();
        for(int row = numRows; row < inferenceRows; row++) {
          NNOutput* dummyOutput = new NNOutput();
          dummyOutput->nnXLen = nnXLen;
          dummyOutput->nnYLen = nnYLen;
          dummyOutput->whiteOwnerMap = resultBufs.back()->includeOwnerMap ?
            new float[nnXLen*nnYLen] : NULL;
          outputBuf.push_back(dummyOutput);
        }
      }

      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize)
            resultBufs[row]->symmetry = rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
          else {
            testAssert(defaultSymmetry >= 0 && defaultSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1);
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      NeuralNet::getOutput(
        gpuHandle,buf.inputBuffers,inferenceRows,inferenceResultData,outputBuf
      );
      testAssert(outputBuf.size() == inferenceRows);
      for(int row = numRows; row < inferenceRows; row++)
        delete outputBuf[row];
      outputBuf.resize(numRows);

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }

    // Lock and update stats before looping again
    lock.lock();
    if(!debugSkipNeuralNet) {
      testAssert(serverThreadIdx >= 0 &&
                 serverThreadIdx < (int)m_numRowsProcessedByServerThread.size());
      m_numRowsProcessedByServerThread[serverThreadIdx] += numRows;
      m_numBatchesProcessedByServerThread[serverThreadIdx] += 1;
    }
    numOngoingEvals -= numRows;

    if(numWaitingEvals > 0) {
      numEvalsToAwaken += numWaitingEvals;
      numWaitingEvals = 0;
      waitingForFinish.notify_all();
    }
    lock.unlock();
    batchingDispatcher.completeBatch(serverThreadIdx);
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
  // Avoid blowup
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
  const Board& board,
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
    bool skipCacheThisIteration = true; // Skip cache since there's no guarantee which symmetry is in the cache
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
  const Board& board,
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
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  testAssert(!isKilled);
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
      m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
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
    fillRowBufs(board, history, nextPlayer, sgfMeta, nnInputParams, buf);
  }

  buf.symmetry = nnInputParams.symmetry;
  buf.policyOptimism = nnInputParams.policyOptimism;

  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals += 1;
  lock.unlock();

  bool suc = queryQueue.forcePush(&buf);
  testAssert(suc);
  batchingDispatcher.notify();

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  // Perform postprocessing on the result - turn the nn output into probabilities
  // As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  // and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  // and causing policy weights to be different, which would reduce performance of successive searches in a game
  // by making the successive searches distribute their playouts less coherently and using the cache more poorly.
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
    testAssert(buf.result->whiteOwnerMap != NULL);
  }
  else {
    float* policy = buf.result->policyProbs;

    float policyOutputScaling = postProcessParams.outputScaleMultiplier / nnInputParams.nnPolicyTemperature;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    testAssert(nextPlayer == history.presumedNextMovePla);
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

    testAssert(legalCount > 0);

    float policySum = 0.0f;

    if(nnInputParams.enablePassingHacks) {
      //Cap passing prior policy at 95% (19x other moves)
      float maxPassPolicySumFactor = 19.0f;

      for(int i = 0; i<policySize-1; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
      int passPos = NNPos::locToPos(Board::PASS_LOC, xSize, nnXLen, nnYLen);
      testAssert(passPos == policySize-1);
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

    // Somehow all legal moves rounded to 0 probability
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
    // Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    // Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    buf.result->policyOptimismUsed = (float)nnInputParams.policyOptimism;

    // Fix up the value as well. Note that the neural net gives us back the value from the perspective
    // of the player so we need to negate that to make it the white value.
    if(modelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      // Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;

        // Softmax
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

        // Softmax
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

        // scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        // noResult counts as 0 score for scorevalue purposes.
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

  // Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(modelVersion >= 3) {
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          // Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          // but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
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


  // And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}

// Uncomment this to lower the effective hash size down to one where we get true collisions
// #define SIMULATE_TRUE_HASH_COLLISIONS

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
  // Free ret BEFORE locking, to avoid any expensive operations while locked.
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
  // Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    // Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  // No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
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

void NNEvaluator::evaluatePreparedRaw(NNResultBuf& buf, bool includeOwnerMap) {
  testAssert(!isKilled);
  if(debugSkipNeuralNet)
    throw StringError("evaluatePreparedRaw requires a real neural net");

  const size_t expectedSpatial =
    (size_t)NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
  const size_t expectedGlobal = (size_t)NNModelVersion::getNumGlobalFeatures(modelVersion);
  const size_t expectedMeta = (size_t)numInputMetaChannels;
  if(buf.rowSpatialBuf.size() != expectedSpatial)
    throw StringError("evaluatePreparedRaw received the wrong spatial feature length");
  if(buf.rowGlobalBuf.size() != expectedGlobal)
    throw StringError("evaluatePreparedRaw received the wrong global feature length");
  if(buf.rowMetaBuf.size() != expectedMeta || buf.hasRowMeta != (expectedMeta > 0))
    throw StringError("evaluatePreparedRaw received the wrong metadata feature length");

  {
    lock_guard<std::mutex> resultLock(buf.resultMutex);
    buf.hasResult = false;
    buf.result.reset();
  }
  buf.includeOwnerMap = includeOwnerMap;
  buf.boardXSizeForServer = nnXLen;
  buf.boardYSizeForServer = nnYLen;

  {
    unique_lock<std::mutex> lock(bufferMutex);
    numOngoingEvals += 1;
  }

  bool suc = queryQueue.forcePush(&buf);
  testAssert(suc);
  batchingDispatcher.notify();

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
}
