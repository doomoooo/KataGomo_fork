#include "../tests/tests.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <thread>

#include "../core/rand.h"
#include "../external/nlohmann_json/json.hpp"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/npzreader.h"

using namespace std;
using json = nlohmann::json;

namespace {

constexpr int POS_LEN = 19;
constexpr int POS_AREA = POS_LEN * POS_LEN;
constexpr int POLICY_LEN = POS_AREA + 1;
constexpr int NUM_BIN_FEATURES = 22;
constexpr int NUM_GLOBAL_FEATURES = 19;
constexpr int PACKED_WIDTH = (POS_AREA + 7) / 8;

struct PreparedCorpus {
  int numRows;
  vector<float> spatial;
  vector<float> global;
};

struct FP32Golden {
  int numRows;
  vector<float> policyProb;
  vector<float> valueProb;
  vector<float> scoreRaw;
  vector<float> ownershipProb;
};

struct VerifyItem {
  uint64_t pass;
  int row;
  shared_ptr<NNOutput> output;
};

struct HeadError {
  double maxAbs;
  double rmse;
  int maxIndex;
};

static void readExact(ifstream& in, void* data, size_t bytes, const string& path) {
  in.read((char*)data,(streamsize)bytes);
  if(!in)
    throw StringError("GTP request stress: truncated file " + path);
}

static uint32_t readU32LE(ifstream& in, const string& path) {
  unsigned char b[4];
  readExact(in,b,4,path);
  return
    (uint32_t)b[0] |
    ((uint32_t)b[1] << 8) |
    ((uint32_t)b[2] << 16) |
    ((uint32_t)b[3] << 24);
}

static float sigmoid(float x) {
  if(x >= 0.0f)
    return 1.0f / (1.0f + std::exp(-x));
  float e = std::exp(x);
  return e / (1.0f + e);
}

template<typename Getter>
static void softmaxInto(Getter getLogit, int n, float* dst) {
  double maxLogit = -numeric_limits<double>::infinity();
  for(int i = 0; i < n; i++) {
    double x = getLogit(i);
    if(!std::isfinite(x))
      throw StringError("GTP request stress: FP32 reference contains a nonfinite logit");
    maxLogit = std::max(maxLogit,x);
  }
  double sum = 0.0;
  for(int i = 0; i < n; i++) {
    dst[i] = (float)std::exp((double)getLogit(i) - maxLogit);
    sum += dst[i];
  }
  for(int i = 0; i < n; i++)
    dst[i] = (float)(dst[i] / sum);
}

static FP32Golden loadFP32Golden(const string& path) {
  ifstream in(path,ios::binary);
  if(!in)
    throw StringError("GTP request stress: could not open FP32 reference " + path);
  char magic[4];
  readExact(in,magic,4,path);
  if(string(magic,4) != "KRNN")
    throw StringError("GTP request stress: reference is not a KRNN file: " + path);
  uint32_t metaLen = readU32LE(in,path);
  if(metaLen == 0 || metaLen > 16 * 1024 * 1024)
    throw StringError("GTP request stress: invalid KRNN metadata length");
  string metaText(metaLen,'\0');
  readExact(in,metaText.data(),metaText.size(),path);
  json meta = json::parse(metaText);

  int numRows = meta.at("numRows").get<int>();
  int posLen = meta.at("posLen").get<int>();
  if(numRows <= 0 || posLen != POS_LEN)
    throw StringError("GTP request stress: FP32 reference must contain 19x19 rows");
  const int expectedDims[5] = {2 * POS_AREA,2,3,6,POS_AREA};
  const json& sections = meta.at("sections");
  if(!sections.is_array() || sections.size() < 5)
    throw StringError("GTP request stress: FP32 reference lacks raw output sections");

  vector<vector<float>> raw(5);
  for(int section = 0; section < 5; section++) {
    int dim = sections.at(section).at("dim").get<int>();
    uint64_t bytes = sections.at(section).at("bytes").get<uint64_t>();
    uint64_t expectedBytes = (uint64_t)numRows * expectedDims[section] * sizeof(float);
    if(dim != expectedDims[section] || bytes != expectedBytes)
      throw StringError(
        "GTP request stress: unexpected FP32 reference section " +
        Global::intToString(section)
      );
    raw[section].resize((size_t)numRows * expectedDims[section]);
    readExact(in,raw[section].data(),(size_t)expectedBytes,path);
  }

  FP32Golden golden;
  golden.numRows = numRows;
  golden.policyProb.resize((size_t)numRows * POLICY_LEN);
  golden.valueProb.resize((size_t)numRows * 3);
  golden.scoreRaw = std::move(raw[3]);
  golden.ownershipProb.resize((size_t)numRows * POS_AREA);
  for(int row = 0; row < numRows; row++) {
    const float* policy = raw[0].data() + (size_t)row * 2 * POS_AREA;
    const float* pass = raw[1].data() + (size_t)row * 2;
    softmaxInto(
      [&](int move) { return move == POS_AREA ? pass[0] : policy[move * 2]; },
      POLICY_LEN,
      golden.policyProb.data() + (size_t)row * POLICY_LEN
    );
    const float* value = raw[2].data() + (size_t)row * 3;
    softmaxInto(
      [&](int channel) { return value[channel]; },
      3,
      golden.valueProb.data() + (size_t)row * 3
    );
    const float* ownership = raw[4].data() + (size_t)row * POS_AREA;
    float* ownershipProb = golden.ownershipProb.data() + (size_t)row * POS_AREA;
    for(int pos = 0; pos < POS_AREA; pos++) {
      if(!std::isfinite(ownership[pos]))
        throw StringError("GTP request stress: FP32 reference contains nonfinite ownership");
      ownershipProb[pos] = sigmoid(ownership[pos]);
    }
  }
  return golden;
}

static PreparedCorpus loadPreparedCorpus(const string& path, bool useNHWC) {
  const set<string> requested = {
    "binaryInputNCHW",
    "binaryInputNCHWPacked",
    "globalInputNC"
  };
  NpzReader corpus(path,requested);
  const NpzArray& globalNC = corpus.get("globalInputNC");
  if(globalNC.dtype != "f4" || globalNC.shape.size() != 2 ||
     globalNC.shape[0] <= 0 || globalNC.shape[1] != NUM_GLOBAL_FEATURES)
    throw StringError("GTP request stress: globalInputNC must be (N,19) f4");
  int numRows = (int)globalNC.shape[0];

  PreparedCorpus prepared;
  prepared.numRows = numRows;
  prepared.spatial.resize((size_t)numRows * NUM_BIN_FEATURES * POS_AREA);
  prepared.global.resize((size_t)numRows * NUM_GLOBAL_FEATURES);
  const float* globalSrc = (const float*)globalNC.data.data();
  std::copy(
    globalSrc,globalSrc + (size_t)numRows * NUM_GLOBAL_FEATURES,
    prepared.global.begin()
  );

  auto dstIndex = [&](int row, int channel, int pos) {
    size_t rowStart = (size_t)row * NUM_BIN_FEATURES * POS_AREA;
    return useNHWC ?
      rowStart + (size_t)pos * NUM_BIN_FEATURES + channel :
      rowStart + (size_t)channel * POS_AREA + pos;
  };
  if(corpus.has("binaryInputNCHWPacked")) {
    const NpzArray& packed = corpus.get("binaryInputNCHWPacked");
    if(packed.dtype != "u1" || packed.shape.size() != 3 ||
       packed.shape[0] != numRows || packed.shape[1] != NUM_BIN_FEATURES ||
       packed.shape[2] != PACKED_WIDTH)
      throw StringError("GTP request stress: binaryInputNCHWPacked must be (N,22,46) u1");
    for(int row = 0; row < numRows; row++) {
      for(int channel = 0; channel < NUM_BIN_FEATURES; channel++) {
        const unsigned char* src = packed.data.data() +
          ((size_t)row * NUM_BIN_FEATURES + channel) * PACKED_WIDTH;
        for(int pos = 0; pos < POS_AREA; pos++)
          prepared.spatial[dstIndex(row,channel,pos)] =
            (float)((src[pos / 8] >> (7 - pos % 8)) & 1);
      }
    }
  }
  else if(corpus.has("binaryInputNCHW")) {
    const NpzArray& binary = corpus.get("binaryInputNCHW");
    if(binary.dtype != "f4" || binary.shape.size() != 4 ||
       binary.shape[0] != numRows || binary.shape[1] != NUM_BIN_FEATURES ||
       binary.shape[2] != POS_LEN || binary.shape[3] != POS_LEN)
      throw StringError("GTP request stress: binaryInputNCHW must be (N,22,19,19) f4");
    const float* src = (const float*)binary.data.data();
    for(int row = 0; row < numRows; row++)
      for(int channel = 0; channel < NUM_BIN_FEATURES; channel++)
        for(int pos = 0; pos < POS_AREA; pos++)
          prepared.spatial[dstIndex(row,channel,pos)] =
            src[((size_t)row * NUM_BIN_FEATURES + channel) * POS_AREA + pos];
  }
  else {
    throw StringError("GTP request stress: corpus lacks binary NN inputs");
  }
  return prepared;
}

template<typename Getter>
static HeadError compareSoftmax(
  Getter getLogit,
  int n,
  const float* reference
) {
  double maxLogit = -numeric_limits<double>::infinity();
  for(int i = 0; i < n; i++) {
    double x = getLogit(i);
    if(!std::isfinite(x))
      throw StringError("candidate contains a nonfinite logit");
    maxLogit = std::max(maxLogit,x);
  }
  double sum = 0.0;
  for(int i = 0; i < n; i++)
    sum += std::exp((double)getLogit(i) - maxLogit);
  HeadError err{0.0,0.0,0};
  for(int i = 0; i < n; i++) {
    double probability = std::exp((double)getLogit(i) - maxLogit) / sum;
    double diff = std::fabs(probability - reference[i]);
    if(diff > err.maxAbs) {
      err.maxAbs = diff;
      err.maxIndex = i;
    }
    err.rmse += diff * diff;
  }
  err.rmse = std::sqrt(err.rmse / n);
  return err;
}

template<typename Getter, typename Transform>
static HeadError compareHead(
  Getter getValue,
  Transform transform,
  int n,
  const float* reference
) {
  HeadError err{0.0,0.0,0};
  for(int i = 0; i < n; i++) {
    double x = getValue(i);
    if(!std::isfinite(x))
      throw StringError("candidate contains a nonfinite output");
    double diff = std::fabs(transform((float)x) - reference[i]);
    if(diff > err.maxAbs) {
      err.maxAbs = diff;
      err.maxIndex = i;
    }
    err.rmse += diff * diff;
  }
  err.rmse = std::sqrt(err.rmse / n);
  return err;
}

static void requireWithin(
  const char* head,
  const HeadError& err,
  double maxAbsLimit,
  double rmseLimit,
  int row,
  uint64_t pass
) {
  if(err.maxAbs <= maxAbsLimit && err.rmse <= rmseLimit)
    return;
  throw StringError(
    "GTP request stress FP32 mismatch: pass " + Global::uint64ToString(pass + 1) +
    " corpus row " + Global::intToString(row) + " head " + head +
    " index " + Global::intToString(err.maxIndex) +
    " maxAbs " + Global::doubleToString(err.maxAbs) +
    " (limit " + Global::doubleToString(maxAbsLimit) + ") rmse " +
    Global::doubleToString(err.rmse) + " (limit " +
    Global::doubleToString(rmseLimit) + ")"
  );
}

static void verifyOutput(
  const VerifyItem& item,
  const FP32Golden& golden
) {
  if(item.output == nullptr)
    throw StringError("GTP request stress received a null output");
  const NNOutput& output = *item.output;
  if(output.nnXLen != POS_LEN || output.nnYLen != POS_LEN || output.whiteOwnerMap == nullptr)
    throw StringError("GTP request stress received an output with the wrong shape");
  const float* policyReference = golden.policyProb.data() + (size_t)item.row * POLICY_LEN;
  HeadError policy = compareSoftmax(
    [&](int move) { return output.policyProbs[move]; },
    POLICY_LEN,policyReference
  );
  requireWithin("policy-probability",policy,0.025,0.002,item.row,item.pass);

  const float value[3] = {
    output.whiteWinProb,output.whiteLossProb,output.whiteNoResultProb
  };
  HeadError valueError = compareSoftmax(
    [&](int channel) { return value[channel]; },
    3,golden.valueProb.data() + (size_t)item.row * 3
  );
  requireWithin("value-probability",valueError,0.05,0.04,item.row,item.pass);

  const float score[6] = {
    output.whiteScoreMean,output.whiteScoreMeanSq,output.whiteLead,
    output.varTimeLeft,output.shorttermWinlossError,output.shorttermScoreError
  };
  HeadError scoreError = compareHead(
    [&](int channel) { return score[channel]; },
    [](float x) { return x; },
    6,golden.scoreRaw.data() + (size_t)item.row * 6
  );
  requireWithin("score-raw",scoreError,0.60,0.30,item.row,item.pass);

  HeadError ownershipError = compareHead(
    [&](int pos) { return output.whiteOwnerMap[pos]; },
    [](float x) { return sigmoid(x); },
    POS_AREA,golden.ownershipProb.data() + (size_t)item.row * POS_AREA
  );
  requireWithin("ownership-probability",ownershipError,0.025,0.006,item.row,item.pass);
}

static void updateMax(atomic<uint64_t>& target, uint64_t value) {
  uint64_t previous = target.load(std::memory_order_relaxed);
  while(previous < value &&
        !target.compare_exchange_weak(previous,value,std::memory_order_relaxed)) {}
}

}  // namespace

void Tests::runNNGTPRequestStress(
  NNEvaluator* nnEval,
  const string& corpusFile,
  const string& referenceFile,
  int numRequests,
  int numPasses,
  int numRequestThreads,
  int numVerifyThreads
) {
  if(nnEval == nullptr)
    throw StringError("GTP request stress requires an evaluator");
  if(numRequests <= 0 || numPasses < 0 || numRequestThreads < 0 || numVerifyThreads <= 0)
    throw StringError("GTP request stress received invalid thread/pass counts");
  if(nnEval->getNNXLen() != POS_LEN || nnEval->getNNYLen() != POS_LEN)
    throw StringError("GTP request stress requires a 19x19 evaluator");
  if(NNModelVersion::getNumSpatialFeatures(nnEval->getModelVersion()) != NUM_BIN_FEATURES ||
     NNModelVersion::getNumGlobalFeatures(nnEval->getModelVersion()) != NUM_GLOBAL_FEATURES ||
     nnEval->requiresSGFMetadata())
    throw StringError("GTP request stress corpus is incompatible with this model's inputs");

  cout << "Loading preprocessed request corpus " << corpusFile << endl;
  PreparedCorpus corpus = loadPreparedCorpus(corpusFile,nnEval->getInputsUseNHWC());
  cout << "Loading offline full-FP32 reference " << referenceFile << endl;
  FP32Golden golden = loadFP32Golden(referenceFile);
  if(corpus.numRows != golden.numRows)
    throw StringError("GTP request stress corpus/reference row counts disagree");
  if(numRequests > corpus.numRows)
    throw StringError("GTP request stress requests exceeds the offline corpus row count");

  if(numRequestThreads <= 0) {
    const int tailThreads = 32;
    numRequestThreads =
      nnEval->getCurrentBatchSize() * (nnEval->getNumServerThreads() + 1) + tailThreads;
  }
  numRequestThreads = std::min(numRequestThreads,numRequests);
  uint64_t totalRequests = numeric_limits<uint64_t>::max();
  if(numPasses > 0) {
    if((uint64_t)numPasses > numeric_limits<uint64_t>::max() / (uint64_t)numRequests)
      throw StringError("GTP request stress request count overflow");
    totalRequests = (uint64_t)numPasses * (uint64_t)numRequests;
  }

  vector<int> order(numRequests);
  for(int i = 0; i < numRequests; i++)
    order[i] = i;
  Rand orderRand("runNNGTPRequestStress:offline-fp32-order");
  for(int i = numRequests - 1; i > 0; i--) {
    int j = (int)orderRand.nextUInt((uint32_t)i + 1);
    std::swap(order[i],order[j]);
  }

  cout << "GTP request stress: " << numRequests << " offline rows/pass, "
       << nnEval->getCurrentBatchSize() << " fixed batch, "
       << nnEval->getNumServerThreads() << " inference slots, "
       << numRequestThreads << " request threads, " << numVerifyThreads
       << " asynchronous CPU verifier threads, passes "
       << (numPasses == 0 ? string("forever") : Global::intToString(numPasses)) << endl;
  cout << "FP32 gates: policy(abs .025/rmse .002), value(.05/.04), "
       << "score raw(.60/.30), ownership(.025/.006)" << endl;

  nnEval->setDoRandomize(false);
  const uint64_t rowsBefore = nnEval->numRowsProcessed();
  const uint64_t batchesBefore = nnEval->numBatchesProcessed();
  const vector<uint64_t> slotRowsBefore = nnEval->numRowsProcessedByServerThread();
  const vector<uint64_t> slotBatchesBefore = nnEval->numBatchesProcessedByServerThread();

  ThreadSafeQueue<VerifyItem> verifyQueue;
  verifyQueue.reserve((size_t)numRequestThreads * 4);
  atomic<uint64_t> nextTicket(0);
  atomic<uint64_t> produced(0);
  atomic<uint64_t> verified(0);
  atomic<uint64_t> maxBacklog(0);
  atomic<bool> failed(false);
  atomic<bool> producersDone(false);
  mutex errorMutex;
  exception_ptr firstError;
  mutex progressMutex;
  condition_variable progressCondition;

  auto captureError = [&](exception_ptr error) {
    failed.store(true,std::memory_order_release);
    {
      lock_guard<std::mutex> lock(errorMutex);
      if(firstError == nullptr)
        firstError = error;
    }
    progressCondition.notify_all();
  };

  vector<thread> verifierThreads;
  verifierThreads.reserve(numVerifyThreads);
  for(int threadIdx = 0; threadIdx < numVerifyThreads; threadIdx++) {
    verifierThreads.emplace_back([&]() {
      VerifyItem item;
      while(verifyQueue.waitPop(item)) {
        if(failed.load(std::memory_order_acquire))
          continue;
        try {
          verifyOutput(item,golden);
          verified.fetch_add(1,std::memory_order_release);
        }
        catch(...) {
          captureError(current_exception());
        }
      }
    });
  }

  auto stressStart = chrono::steady_clock::now();
  thread reporter([&]() {
    uint64_t nextReport = (uint64_t)numRequests;
    uint64_t lastProduced = 0;
    uint64_t lastBatches = batchesBefore;
    vector<uint64_t> lastSlotRows = slotRowsBefore;
    vector<uint64_t> lastSlotBatches = slotBatchesBefore;
    auto lastTime = stressStart;
    while(true) {
      unique_lock<std::mutex> lock(progressMutex);
      progressCondition.wait(lock,[&]() {
        return failed.load(std::memory_order_acquire) ||
          produced.load(std::memory_order_acquire) >= nextReport ||
          producersDone.load(std::memory_order_acquire);
      });
      if(failed.load(std::memory_order_acquire))
        break;
      uint64_t currentProduced = produced.load(std::memory_order_acquire);
      if(currentProduced < nextReport) {
        if(producersDone.load(std::memory_order_acquire))
          break;
        continue;
      }
      lock.unlock();

      auto now = chrono::steady_clock::now();
      double seconds = chrono::duration<double>(now - lastTime).count();
      uint64_t currentBatches = nnEval->numBatchesProcessed();
      vector<uint64_t> currentSlotRows = nnEval->numRowsProcessedByServerThread();
      vector<uint64_t> currentSlotBatches = nnEval->numBatchesProcessedByServerThread();
      double requestRate = (currentProduced - lastProduced) / seconds;
      double alignedRate =
        (currentBatches - lastBatches) * nnEval->getCurrentBatchSize() / seconds;
      uint64_t currentVerified = verified.load(std::memory_order_acquire);
      uint64_t backlog = currentProduced - std::min(currentProduced,currentVerified);
      cout << "LOAD " << currentProduced << " requests: " << fixed << setprecision(1)
           << requestRate << " requests/s, " << alignedRate << " aligned nnEval/s, verified "
           << currentVerified << ", backlog " << backlog << "; slots";
      for(size_t slot = 0; slot < currentSlotRows.size(); slot++) {
        cout << " [" << slot << "@gpu" << nnEval->getGpuIdxByServerThread((int)slot)
             << " " << (currentSlotRows[slot] - lastSlotRows[slot]) << "r/"
             << (currentSlotBatches[slot] - lastSlotBatches[slot]) << "b]";
      }
      cout << endl;
      lastTime = now;
      lastProduced = currentProduced;
      lastBatches = currentBatches;
      lastSlotRows = std::move(currentSlotRows);
      lastSlotBatches = std::move(currentSlotBatches);
      nextReport = (currentProduced / (uint64_t)numRequests + 1) * (uint64_t)numRequests;
    }
  });

  vector<thread> requestThreads;
  requestThreads.reserve(numRequestThreads);
  for(int threadIdx = 0; threadIdx < numRequestThreads; threadIdx++) {
    requestThreads.emplace_back([&]() {
      try {
        NNResultBuf buf;
        buf.rowSpatialBuf.resize((size_t)NUM_BIN_FEATURES * POS_AREA);
        buf.rowGlobalBuf.resize(NUM_GLOBAL_FEATURES);
        buf.rowMetaBuf.clear();
        buf.hasRowMeta = false;
        buf.symmetry = 0;
        buf.policyOptimism = 0.0;
        while(!failed.load(std::memory_order_acquire)) {
          uint64_t ticket = nextTicket.fetch_add(1,std::memory_order_relaxed);
          if(ticket >= totalRequests)
            break;
          uint64_t pass = ticket / (uint64_t)numRequests;
          uint64_t withinPass = ticket % (uint64_t)numRequests;
          uint64_t offset = (pass * 7919ULL) % (uint64_t)numRequests;
          uint64_t orderIdx = (pass & 1) ?
            (offset + (uint64_t)numRequests - 1 - withinPass) % (uint64_t)numRequests :
            (offset + withinPass) % (uint64_t)numRequests;
          int row = order[(size_t)orderIdx];

          const float* spatial = corpus.spatial.data() +
            (size_t)row * NUM_BIN_FEATURES * POS_AREA;
          std::copy(
            spatial,spatial + (size_t)NUM_BIN_FEATURES * POS_AREA,
            buf.rowSpatialBuf.begin()
          );
          const float* global = corpus.global.data() + (size_t)row * NUM_GLOBAL_FEATURES;
          std::copy(global,global + NUM_GLOBAL_FEATURES,buf.rowGlobalBuf.begin());
          nnEval->evaluatePreparedRaw(buf,true);
          if(failed.load(std::memory_order_acquire))
            break;

          VerifyItem item{pass,row,std::move(buf.result)};
          if(!verifyQueue.forcePush(std::move(item)))
            throw StringError("GTP request stress verifier queue closed unexpectedly");
          uint64_t currentProduced = produced.fetch_add(1,std::memory_order_release) + 1;
          uint64_t currentVerified = verified.load(std::memory_order_acquire);
          updateMax(maxBacklog,currentProduced - std::min(currentProduced,currentVerified));
          if(currentProduced % (uint64_t)numRequests == 0)
            progressCondition.notify_all();
        }
      }
      catch(...) {
        captureError(current_exception());
      }
    });
  }

  for(thread& requestThread: requestThreads)
    requestThread.join();
  producersDone.store(true,std::memory_order_release);
  progressCondition.notify_all();
  reporter.join();
  verifyQueue.setReadOnly();
  for(thread& verifierThread: verifierThreads)
    verifierThread.join();

  if(firstError != nullptr)
    rethrow_exception(firstError);
  if(numPasses > 0 && produced.load() != totalRequests)
    throw StringError("GTP request stress did not produce every requested result");
  if(verified.load() != produced.load())
    throw StringError("GTP request stress did not verify every produced result");

  const uint64_t rowDelta = nnEval->numRowsProcessed() - rowsBefore;
  const uint64_t batchDelta = nnEval->numBatchesProcessed() - batchesBefore;
  const vector<uint64_t> slotRowsAfter = nnEval->numRowsProcessedByServerThread();
  const vector<uint64_t> slotBatchesAfter = nnEval->numBatchesProcessedByServerThread();
  if(rowDelta != produced.load())
    throw StringError("GTP request stress evaluator row count disagrees with produced requests");
  if(slotRowsAfter.size() != slotRowsBefore.size() ||
     slotBatchesAfter.size() != slotBatchesBefore.size())
    throw StringError("GTP request stress inference-slot topology changed");
  uint64_t slotRowSum = 0;
  for(size_t slot = 0; slot < slotRowsAfter.size(); slot++) {
    uint64_t slotRows = slotRowsAfter[slot] - slotRowsBefore[slot];
    slotRowSum += slotRows;
    if(slotRows == 0)
      throw StringError(
        "GTP request stress did not exercise inference slot " +
        Global::intToString((int)slot)
      );
  }
  if(slotRowSum != rowDelta)
    throw StringError("GTP request stress per-slot rows do not sum to evaluator rows");
  double seconds = chrono::duration<double>(chrono::steady_clock::now() - stressStart).count();
  cout << "PASS: verified " << verified.load() << " results against offline FP32 in "
       << fixed << setprecision(2) << seconds << " s; "
       << (produced.load() / seconds) << " requests/s, "
       << (batchDelta * nnEval->getCurrentBatchSize() / seconds)
       << " aligned nnEval/s, max verifier backlog " << maxBacklog.load() << endl;
}
