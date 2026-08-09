#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../dataio/sgf.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/npzreader.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
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

static string jsonEscape(const string& s) {
  ostringstream out;
  for(char c : s) {
    if(c == '"' || c == '\\')
      out << '\\' << c;
    else if(c == '\n')
      out << "\\n";
    else if(c == '\t')
      out << "\\t";
    else if(c == '\r')
      out << "\\r";
    else
      out << c;
  }
  return out.str();
}

static void writeAll(ofstream& out, const void* data, size_t size) {
  out.write((const char*)data, (std::streamsize)size);
  if(!out)
    throw StringError("replaynn: failed writing output file");
}

int MainCmds::replaynn(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string corpusFile;
  string outputFile;
  int batchSizeOverride = -1;
  int boardSize = 19;

  try {
    KataGoCommandLine cmd(
      "Replay a fixed input corpus through the real NN backend and dump raw per-head logits, "
      "plus the corpus targets. Honors config settings for batch size, NN server threads, and "
      "per-server GPU assignment. No search, no SGF, no randomized augmentation."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> corpusArg(
      "","corpus","Path to the .npz accuracy corpus (required)",true,"","FILE"
    );
    TCLAP::ValueArg<string> outputArg(
      "","output","Path for the raw replay output blob (required)",true,"","FILE"
    );
    TCLAP::ValueArg<int> batchSizeArg(
      "","batch-size","Override nnMaxBatchSize from config (default: use config)",
      false,-1,"N"
    );
    TCLAP::ValueArg<int> boardSizeArg(
      "","boardsize","NN board size: 9, 13, or 19 (default 19)",
      false,19,"N"
    );
    cmd.add(corpusArg);
    cmd.add(outputArg);
    cmd.add(batchSizeArg);
    cmd.add(boardSizeArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);
    modelFile = cmd.getModelFile();
    corpusFile = corpusArg.getValue();
    outputFile = outputArg.getValue();
    batchSizeOverride = batchSizeArg.getValue();
    boardSize = boardSizeArg.getValue();
    cmd.getConfig(cfg);

    if(boardSize != 19)
      throw StringError("replaynn: corpus is fixed 19x19; boardsize must be 19");
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = false;
  Logger logger(NULL, logToStdoutDefault, logToStderrDefault, logTimeDefault);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  logger.write("replaynn model " + modelFile);
  logger.write("replaynn corpus " + corpusFile);
  logger.write("replaynn output " + outputFile);

  const int posLen = 19;
  const int64_t posArea = (int64_t)posLen * posLen;
  const int numBinFeatures = 22;
  const int numGlobalFeatures = 19;
  const int64_t policyLen = posArea + 1;
  const int packedWidth = (posArea + 7) / 8;

  NpzReader corpus(corpusFile);
  NpzArray binNCHW;
  if(corpus.has("binaryInputNCHW")) {
    binNCHW = corpus.get("binaryInputNCHW");
    if(binNCHW.dtype != "f4" || binNCHW.shape.size() != 4 ||
       binNCHW.shape[1] != numBinFeatures || binNCHW.shape[2] != posLen || binNCHW.shape[3] != posLen)
      throw StringError("replaynn: corpus binaryInputNCHW must be (N,22,19,19) f4");
  }
  else if(corpus.has("binaryInputNCHWPacked")) {
    const NpzArray& packed = corpus.get("binaryInputNCHWPacked");
    if(packed.dtype != "u1" || packed.shape.size() != 3 ||
       packed.shape[1] != numBinFeatures || packed.shape[2] != packedWidth)
      throw StringError("replaynn: corpus binaryInputNCHWPacked must be (N,22,46) u1");
    const int64_t n = packed.shape[0];
    binNCHW.shape = {n, numBinFeatures, posLen, posLen};
    binNCHW.dtype = "f4";
    binNCHW.elemSize = 4;
    binNCHW.data.resize((size_t)(n * numBinFeatures * posArea) * 4);
    float* dst = (float*)binNCHW.data.data();
    for(int64_t r = 0; r < n; r++) {
      for(int c = 0; c < numBinFeatures; c++) {
        const unsigned char* srcRow = packed.data.data() + (size_t)((r * numBinFeatures + c) * packedWidth);
        for(int byteIdx = 0; byteIdx < packedWidth; byteIdx++) {
          unsigned char byte = srcRow[byteIdx];
          for(int bit = 0; bit < 8; bit++) {
            const int64_t pos = (int64_t)byteIdx * 8 + bit;
            if(pos >= posArea)
              break;
            dst[(size_t)((r * numBinFeatures + c) * posArea + pos)] = (byte >> (7 - bit)) & 1;
          }
        }
      }
    }
  }
  else {
    throw StringError("replaynn: corpus lacks binaryInputNCHW or binaryInputNCHWPacked");
  }
  const NpzArray& globalNC = corpus.get("globalInputNC");
  const NpzArray& policyTargets = corpus.get("policyTargetsNCMove");
  const NpzArray& globalTargets = corpus.get("globalTargetsNC");
  const NpzArray& scoreTargets = corpus.get("scoreDistrN");
  const NpzArray& valueTargets = corpus.get("valueTargetsNCHW");
  const NpzArray& qValueTargets = corpus.get("qValueTargetsNCMove");

  if(globalNC.dtype != "f4" || globalNC.shape.size() != 2 || globalNC.shape[1] != numGlobalFeatures)
    throw StringError("replaynn: corpus globalInputNC must be (N,19) f4");
  if(globalNC.shape[0] != binNCHW.shape[0])
    throw StringError("replaynn: corpus row counts disagree");

  const int64_t numRows = binNCHW.shape[0];
  const int maxBatchSize =
    batchSizeOverride > 0 ? batchSizeOverride :
    cfg.contains("nnMaxBatchSize") ? cfg.getInt("nnMaxBatchSize",1,65536) :
    19;

  const int expectedConcurrentEvals = maxBatchSize;
  // Full-board corpus rows: exact-NN-length semantics are identical to the
  // masked path for full boards and enable the SM120 FA4 AOT attention path.
  const bool defaultRequireExactNNLen = true;
  const bool disableFP16 = false;

  NNEvaluator* nnEval = NULL;
  try {
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,"",cfg,logger,seedRand,expectedConcurrentEvals,
      boardSize,boardSize,maxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_BENCHMARK
    );
    nnEval->killServerThreads();

    const int numThreads = nnEval->getNumServerThreads();
    logger.write("replaynn rows " + Global::int64ToString(numRows) +
                 " threads " + Global::intToString(numThreads) +
                 " batch " + Global::intToString(maxBatchSize));

    // Blob sections (all float32, numRows x dim):
    //  0 policyLogits        (2*posArea)       policy + optimistic policy logits per position
    //  1 policyPassLogits    (2)
    //  2 valueLogits         (3)
    //  3 scoreValueLogits    (6)
    //  4 ownership           (posArea)
    //  5 targetPolicy        (2*policyLen)
    //  6 targetGlobal        (80)
    //  7 targetScore         (842)
    //  8 targetValue         (5*posArea)
    //  9 targetQValue        (3*policyLen)
    // 10 inputBinary         (numBinFeatures*posArea)  NCHW float32
    // 11 inputGlobal         (numGlobalFeatures)
    const int64_t sectionDims[12] = {
      2LL*posArea, 2, 3, 6, posArea,
      2LL*policyLen, 80, 842, 5LL*posArea, 3LL*policyLen,
      (int64_t)numBinFeatures*posArea, numGlobalFeatures
    };
    const int numSections = 12;
    uint64_t blobBytes = 0;
    for(int s = 0; s < numSections; s++)
      blobBytes += (uint64_t)numRows * (uint64_t)sectionDims[s] * 4;
    vector<unsigned char> blob(blobBytes);

    ostringstream meta;
    meta << "{";
    meta << "\"model\":\"" << jsonEscape(nnEval->getModelFileName()) << "\",";
    meta << "\"revision\":\"" << jsonEscape(Version::getGitRevisionWithBackend()) << "\",";
    meta << "\"numRows\":" << numRows << ",";
    meta << "\"posLen\":" << posLen << ",";
    meta << "\"numBinFeatures\":" << numBinFeatures << ",";
    meta << "\"numGlobalFeatures\":" << numGlobalFeatures << ",";
    meta << "\"numThreads\":" << numThreads << ",";
    meta << "\"maxBatchSize\":" << maxBatchSize << ",";
    meta << "\"fixedBatchTailPadding\":true,";
    meta << "\"corpus\":\"" << jsonEscape(corpusFile) << "\",";
    meta << "\"sections\":[";
    for(int s = 0; s < numSections; s++) {
      if(s > 0)
        meta << ",";
      meta << "{\"dim\":" << sectionDims[s] << ",\"bytes\":" << (uint64_t)numRows * (uint64_t)sectionDims[s] * 4 << "}";
    }
    meta << "]}";
    const string metaStr = meta.str();

    vector<thread> threads;
    vector<exception_ptr> threadErrors((size_t)numThreads);
    for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
      threads.emplace_back([&, threadIdx]() {
        try {
          ScopedComputeStream computeStream(nnEval->getGpuIdxByServerThread(threadIdx));
          ComputeHandle* handle = NeuralNet::createComputeHandle(
            nnEval->getComputeContext(), nnEval->getLoadedModel(), &logger,
            maxBatchSize, nnEval->getRequireExactNNLen(), nnEval->getInputsUseNHWC(),
            nnEval->getGpuIdxByServerThread(threadIdx), threadIdx, computeStream.get()
          );
          InputBuffers* inputBuffers = NeuralNet::createInputBuffers(
            nnEval->getLoadedModel(), maxBatchSize, posLen, posLen
          );

          const int64_t rowsPerThread = (numRows + numThreads - 1) / numThreads;
          const int64_t rowStart = threadIdx * rowsPerThread;
          const int64_t rowEnd = std::min(numRows, rowStart + rowsPerThread);
          if(rowStart >= rowEnd) {
            NeuralNet::freeInputBuffers(inputBuffers);
            NeuralNet::freeComputeHandle(handle);
            return;
          }

          // Pointer to each section's first element of this thread's row slice.
          vector<float*> sectionPtr(numSections);
          {
            uint64_t offset = 0;
            for(int s = 0; s < numSections; s++) {
              uint64_t rowBytes = (uint64_t)sectionDims[s] * 4;
              sectionPtr[s] = (float*)(blob.data() + offset + (uint64_t)rowStart * rowBytes);
              offset += (uint64_t)numRows * rowBytes;
            }
          }

          vector<float> hostSpatial((size_t)maxBatchSize * numBinFeatures * posArea);
          vector<float> hostGlobal((size_t)maxBatchSize * numGlobalFeatures);
          vector<unique_ptr<NNResultBuf>> bufs;
          vector<NNResultBuf*> bufPtrs;
          vector<unique_ptr<NNOutput>> outs;
          vector<NNOutput*> outPtrs;
          bufs.reserve(maxBatchSize);
          outs.reserve(maxBatchSize);
          for(int i = 0; i < maxBatchSize; i++) {
            bufs.push_back(std::make_unique<NNResultBuf>());
            bufPtrs.push_back(bufs.back().get());
            outs.push_back(std::make_unique<NNOutput>());
            outs.back()->nnXLen = posLen;
            outs.back()->nnYLen = posLen;
            outs.back()->whiteOwnerMap = NULL;
            outPtrs.push_back(outs.back().get());
          }

          const float* binData = (const float*)binNCHW.data.data();
          const float* globalData = (const float*)globalNC.data.data();
          const float* policyTargetData = (const float*)policyTargets.data.data();
          const float* globalTargetData = (const float*)globalTargets.data.data();
          const float* scoreTargetData = (const float*)scoreTargets.data.data();
          const float* valueTargetData = (const float*)valueTargets.data.data();
          const float* qValueTargetData = (const float*)qValueTargets.data.data();

          for(int64_t rowGlobal = rowStart; rowGlobal < rowEnd; rowGlobal += maxBatchSize) {
            const int realBatchSize =
              (int)std::min<int64_t>(maxBatchSize, rowEnd - rowGlobal);
            // Exact-batch AOT plans must also be exercised on the final
            // partial corpus chunk. Repeat real rows to fill the physical
            // batch, but serialize only the real rows below.
            const int batchSize = maxBatchSize;
            for(int b = 0; b < batchSize; b++) {
              const int64_t row = rowGlobal + (b % realBatchSize);
              float* dst = hostSpatial.data() + (size_t)b * numBinFeatures * posArea;
              const float* src = binData + row * numBinFeatures * posArea;
              if(nnEval->getInputsUseNHWC()) {
                for(int64_t p = 0; p < posArea; p++)
                  for(int c = 0; c < numBinFeatures; c++)
                    dst[p * numBinFeatures + c] = src[c * posArea + p];
              }
              else {
                std::copy(src, src + numBinFeatures * posArea, dst);
              }
              const float* srcGlobal = globalData + row * numGlobalFeatures;
              float* dstGlobal = hostGlobal.data() + (size_t)b * numGlobalFeatures;
              for(int i = 0; i < numGlobalFeatures; i++)
                dstGlobal[i] = srcGlobal[i];
            }
            // getOutput reads the per-row inputs from NNResultBuf::rowSpatialBuf/rowGlobalBuf
            // (with symmetry applied), so populate those buffers rather than the raw
            // userInputBuffer.
            for(int b = 0; b < batchSize; b++) {
              NNResultBuf* buf = bufPtrs[b];
              buf->rowSpatialBuf.assign(
                hostSpatial.data() + (size_t)b * numBinFeatures * posArea,
                hostSpatial.data() + (size_t)(b + 1) * numBinFeatures * posArea
              );
              buf->rowGlobalBuf.assign(
                hostGlobal.data() + (size_t)b * numGlobalFeatures,
                hostGlobal.data() + (size_t)(b + 1) * numGlobalFeatures
              );
              buf->symmetry = 0;
              buf->policyOptimism = 0.0f;
            }
            // getOutput copies the host inputs to device, runs the model, and copies the raw
            // per-head buffers back into inputBuffers->*Results.
            NeuralNet::getOutput(handle, inputBuffers, batchSize, bufPtrs.data(), outPtrs);

            RawNNOutputs raw;
            NeuralNet::getRawNNOutputs(inputBuffers, raw);
            const int numPolicyChannels = (int)raw.numPolicyChannels;
            const int numValueChannels = (int)raw.numValueChannels;
            const int numScoreValueChannels = (int)raw.numScoreValueChannels;
            // numOwnershipChannels is expressed in elements per row (channels * spatial), so
            // normalize by the spatial area.
            const int numOwnershipChannels = (int)(raw.numOwnershipChannels / (size_t)posArea);
            if(numOwnershipChannels * posArea != (int64_t)raw.numOwnershipChannels)
              throw StringError("replaynn: ownership elements not divisible by posArea");
            if(numPolicyChannels != 2 || numValueChannels != 3 || numScoreValueChannels != 6 || numOwnershipChannels != 1) {
              logger.write(
                "replaynn raw head channels: policy=" + Global::intToString(numPolicyChannels) +
                " value=" + Global::intToString(numValueChannels) +
                " scoreValue=" + Global::intToString(numScoreValueChannels) +
                " ownership=" + Global::intToString(numOwnershipChannels)
              );
              throw StringError("replaynn: unexpected raw head channel counts for this model");
            }

            const int64_t localRow = rowGlobal - rowStart;
            float* policyLogits = sectionPtr[0] + localRow * sectionDims[0];
            float* policyPassLogits = sectionPtr[1] + localRow * sectionDims[1];
            float* valueLogits = sectionPtr[2] + localRow * sectionDims[2];
            float* scoreValueLogits = sectionPtr[3] + localRow * sectionDims[3];
            float* ownership = sectionPtr[4] + localRow * sectionDims[4];
            float* targetPolicy = sectionPtr[5] + localRow * sectionDims[5];
            float* targetGlobal = sectionPtr[6] + localRow * sectionDims[6];
            float* targetScore = sectionPtr[7] + localRow * sectionDims[7];
            float* targetValue = sectionPtr[8] + localRow * sectionDims[8];
            float* targetQValue = sectionPtr[9] + localRow * sectionDims[9];
            float* inputBinary = sectionPtr[10] + localRow * sectionDims[10];
            float* inputGlobal = sectionPtr[11] + localRow * sectionDims[11];

            for(int b = 0; b < realBatchSize; b++) {
              const int64_t row = rowGlobal + b;
              float* dstRow = policyLogits + (int64_t)b * sectionDims[0];
              const float* srcPolicy = raw.policyResults + (size_t)b * numPolicyChannels * posArea;
              // Normalize the per-position (policy, optimistic-policy) layout: CUDA is NHWC
              // (channel-minor within each position), TensorRT is NCHW (channel-major across the
              // whole board). The dump is always position-major, channel-minor.
              if(nnEval->getInputsUseNHWC()) {
                for(int64_t i = 0; i < numPolicyChannels * posArea; i++)
                  dstRow[i] = srcPolicy[i];
              }
              else {
                for(int64_t p = 0; p < posArea; p++)
                  for(int c = 0; c < numPolicyChannels; c++)
                    dstRow[p * numPolicyChannels + c] = srcPolicy[c * posArea + p];
              }

              float* dstPass = policyPassLogits + (int64_t)b * sectionDims[1];
              const float* srcPass = raw.policyPassResults + (size_t)b * numPolicyChannels;
              for(int i = 0; i < numPolicyChannels; i++)
                dstPass[i] = srcPass[i];

              float* dstVal = valueLogits + (int64_t)b * sectionDims[2];
              const float* srcVal = raw.valueResults + (size_t)b * numValueChannels;
              for(int i = 0; i < numValueChannels; i++)
                dstVal[i] = srcVal[i];

              float* dstScore = scoreValueLogits + (int64_t)b * sectionDims[3];
              const float* srcScore = raw.scoreValueResults + (size_t)b * numScoreValueChannels;
              for(int i = 0; i < numScoreValueChannels; i++)
                dstScore[i] = srcScore[i];

              float* dstOwn = ownership + (int64_t)b * sectionDims[4];
              const float* srcOwn = raw.ownershipResults + (size_t)b * posArea;
              for(int64_t i = 0; i < posArea; i++)
                dstOwn[i] = srcOwn[i];

              float* dstTgtPol = targetPolicy + (int64_t)b * sectionDims[5];
              for(int64_t i = 0; i < 2 * policyLen; i++)
                dstTgtPol[i] = policyTargetData[row * 2 * policyLen + i];

              float* dstTgtGlob = targetGlobal + (int64_t)b * sectionDims[6];
              for(int64_t i = 0; i < 80; i++)
                dstTgtGlob[i] = globalTargetData[row * 80 + i];

              float* dstTgtScore = targetScore + (int64_t)b * sectionDims[7];
              for(int64_t i = 0; i < 842; i++)
                dstTgtScore[i] = scoreTargetData[row * 842 + i];

              float* dstTgtVal = targetValue + (int64_t)b * sectionDims[8];
              for(int64_t i = 0; i < 5 * posArea; i++)
                dstTgtVal[i] = valueTargetData[row * 5 * posArea + i];

              float* dstTgtQ = targetQValue + (int64_t)b * sectionDims[9];
              for(int64_t i = 0; i < 3 * policyLen; i++)
                dstTgtQ[i] = qValueTargetData[row * 3 * policyLen + i];

              float* dstInBin = inputBinary + (int64_t)b * sectionDims[10];
              const float* srcBin = binData + row * numBinFeatures * posArea;
              for(int64_t i = 0; i < numBinFeatures * posArea; i++)
                dstInBin[i] = srcBin[i];

              float* dstInGlob = inputGlobal + (int64_t)b * sectionDims[11];
              for(int i = 0; i < numGlobalFeatures; i++)
                dstInGlob[i] = globalData[row * numGlobalFeatures + i];
            }
          }
          NeuralNet::freeInputBuffers(inputBuffers);
          NeuralNet::freeComputeHandle(handle);
        }
        catch(...) {
          threadErrors[(size_t)threadIdx] = std::current_exception();
        }
      });
    }
    for(thread& t : threads)
      t.join();
    for(int i = 0; i < numThreads; i++) {
      if(threadErrors[(size_t)i] != nullptr)
        std::rethrow_exception(threadErrors[(size_t)i]);
    }

    {
      ofstream out(outputFile, ios::binary | ios::trunc);
      if(!out)
        throw StringError("replaynn: could not open output " + outputFile);
      const char magic[4] = {'K','R','N','N'};
      const uint32_t metaLen = (uint32_t)metaStr.size();
      writeAll(out, magic, 4);
      writeAll(out, &metaLen, 4);
      writeAll(out, metaStr.data(), metaStr.size());
      writeAll(out, blob.data(), blob.size());
    }
    logger.write("replaynn wrote " + Global::uint64ToString((uint64_t)(4 + 4 + metaStr.size() + blob.size())) + " bytes");
  }
  catch(...) {
    delete nnEval;
    NeuralNet::globalCleanup();
    ScoreValue::freeTables();
    throw;
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
