#ifndef CORE_GLOBALPERF_H_
#define CORE_GLOBALPERF_H_

#include <string>
#include <vector>

class Logger;

namespace GlobalPerfProfile {
  class SearchThreadScope {
   public:
    explicit SearchThreadScope(int threadIdx);
    ~SearchThreadScope();

    SearchThreadScope(const SearchThreadScope&) = delete;
    SearchThreadScope& operator=(const SearchThreadScope&) = delete;
   private:
    int previousThreadIdx;
  };

  void setEnabled(bool enabled);
  bool isEnabled();

  void clear();

  void startRealtime(const std::string& socketPath, int intervalMs, Logger* logger);
  void stopRealtime();
  bool isRealtimeRunning();

  void configureSearchSlots(int numThreads);
  void configureInferenceSlots(const std::vector<int>& gpuIdxByServerThread);
  void setCurrentSearchThreadCount(int numThreads);
  void searchSessionStarted();
  void searchSessionEnded();

  void beginBenchmarkSample();
  void endBenchmarkSample();

  void recordSearchLoop(
    int threadIdx,
    double totalMilliseconds,
    double processMilliseconds,
    double waitMilliseconds,
    int depth,
    int visitDelta,
    bool submittedNNEval
  );
  void noteQueueLength(int queueLength);
  void changeInferenceThreadActiveCount(int inferenceThreadIdx, int delta);
  void recordInferencePhases(
    double preprocessMs,
    double h2dMs,
    double waitGpuMs,
    double d2hMs,
    double postprocessMs,
    int batchSize
  );
  void recordRealtimeInferenceBatch(
    int inferenceThreadIdx,
    int gpuIdx,
    int batchSize,
    int numRows,
    double waitTaskSubmitMs,
    double preprocessMs,
    double h2dMs,
    double inferMs,
    double d2hMs,
    double postprocessMs
  );
  void changeGpuStreamActiveCount(int inferenceThreadIdx, int gpuIdx, int delta);
  void recordInferenceLaunchInterval(double launchIntervalMs);

  std::string makeReport();
}

#endif  // CORE_GLOBALPERF_H_
