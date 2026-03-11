#ifndef CORE_GLOBALPERF_H_
#define CORE_GLOBALPERF_H_

#include <cstdint>
#include <string>
#include <vector>

class Logger;

namespace GlobalPerfProfile {
  enum class TimelineLane {
    SchedulerThread,
    H2DStream,
    InferStream,
    D2HStream
  };

  enum class TimelineStage {
    Preprocess,
    H2D,
    Infer,
    D2H,
    Postprocess
  };

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
  void configureInferenceMode(bool singleSchedulerLogicalSlots);
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
  void recordRealtimeTimelineSpan(
    int inferenceSlotIdx,
    int gpuIdx,
    TimelineLane lane,
    TimelineStage stage,
    uint64_t spanId,
    uint64_t dependencySpanId0,
    uint64_t dependencySpanId1,
    uint64_t batchUid,
    int rowIdx,
    int64_t startNs,
    int64_t endNs
  );
  void recordSchedulerBusySpan(int64_t startNs, int64_t endNs);
  void recordSchedulerIdlePoll(int64_t startNs, int64_t endNs);
  bool wantsRealtimeTimelineSpan(int64_t startNs, int64_t endNs);
  void changeGpuStreamActiveCount(int inferenceThreadIdx, int gpuIdx, int delta);
  void recordInferenceLaunchInterval(double launchIntervalMs);

  std::string makeReport();
}

#endif  // CORE_GLOBALPERF_H_
