#include "../core/globalperf.h"

#include "../core/global.h"
#include "../core/logger.h"
#include "../external/nlohmann_json/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace std;
using nlohmann::json;

namespace {
  using Clock = std::chrono::steady_clock;

  static constexpr double BENCHMARK_SAMPLE_TRIM_SECONDS = 0.100;
  static constexpr int64_t ONE_SECOND_NS = 1000000000LL;
  static constexpr int64_t ONE_MILLISECOND_NS = 1000000LL;
  static constexpr size_t SEARCH_LOOP_RING_CAPACITY = 4096;
  static constexpr size_t INFERENCE_BATCH_RING_CAPACITY = 2048;
  static constexpr size_t INFERENCE_EVENT_RING_CAPACITY = 4096;
  static constexpr size_t QUEUE_LENGTH_RING_CAPACITY = 4096;
  static constexpr size_t TIMELINE_SPAN_RING_CAPACITY = 8192;
  static constexpr size_t SCHEDULER_BUSY_RING_CAPACITY = 4096;
  static constexpr int64_t TIMELINE_WINDOW_NS = 50 * ONE_MILLISECOND_NS;
  static constexpr size_t TIMELINE_SNAPSHOT_MAX_SPANS = 1024;

  struct CountDurationSegment {
    double durationSeconds;
    int count;
  };

  struct TimedSearchLoop {
    double offsetSeconds;
    double processMs;
    double waitMs;
  };

  struct TimedInferencePhases {
    double offsetSeconds;
    double preprocessMs;
    double h2dMs;
    double waitGpuMs;
    double d2hMs;
    double postprocessMs;
    int batchSize;
  };

  struct TimedScalar {
    double offsetSeconds;
    double value;
  };

  struct BenchmarkPerfState {
    mutex stateMutex;

    vector<double> searchLoopProcessMs;
    vector<double> searchLoopWaitMs;

    vector<double> inferencePreprocessMs;
    vector<double> inferenceH2DMs;
    vector<double> inferenceWaitGpuMs;
    vector<double> inferenceD2HMs;
    vector<double> inferencePostprocessMs;
    vector<double> inferenceLaunchIntervalMs;

    vector<double> gpuWaitGpuMsByBatchSize;

    vector<double> queueLengthSeconds;
    int currentQueueLength = 0;

    vector<double> inferenceThreadActiveSeconds;
    int currentInferenceThreadActiveCount = 0;

    bool benchmarkSampleActive = false;
    Clock::time_point benchmarkSampleStart = Clock::now();
    Clock::time_point sampleQueueLastUpdate = Clock::now();
    Clock::time_point sampleInferenceThreadLastUpdate = Clock::now();
    vector<CountDurationSegment> sampleQueueSegments;
    vector<CountDurationSegment> sampleInferenceThreadSegments;
    vector<TimedSearchLoop> sampleSearchLoops;
    vector<TimedInferencePhases> sampleInferencePhases;
    vector<TimedScalar> sampleLaunchIntervals;
  };

  template<typename T, size_t Capacity>
  struct VersionedRingBuffer {
    struct Cell {
      std::atomic<uint64_t> version;
      T sample;

      Cell() : version(0), sample() {}
    };

    std::atomic<uint64_t> writeCount;
    std::array<Cell, Capacity> cells;

    VersionedRingBuffer() : writeCount(0), cells() {}
  };

  struct SearchLoopRealtimeSample {
    uint64_t sequence = 0;
    int64_t timestampNs = 0;
    double totalMs = 0.0;
    double processMs = 0.0;
    double waitMs = 0.0;
    int depth = 0;
    int visitDelta = 0;
    bool submittedNNEval = false;
  };

  struct DeltaEventSample {
    uint64_t sequence = 0;
    int64_t timestampNs = 0;
    int delta = 0;
    int gpuIdx = -1;
    int valueAfter = -1;
  };

  struct InferenceBatchRealtimeSample {
    uint64_t sequence = 0;
    int64_t timestampNs = 0;
    int gpuIdx = -1;
    int batchSize = 0;
    int numRows = 0;
    double waitTaskSubmitMs = 0.0;
    double preprocessMs = 0.0;
    double h2dMs = 0.0;
    double inferMs = 0.0;
    double d2hMs = 0.0;
    double postprocessMs = 0.0;
  };

  struct TimelineSpanRealtimeSample {
    uint64_t sequence = 0;
    uint64_t spanId = 0;
    int64_t startNs = 0;
    int64_t endNs = 0;
    int inferenceSlotIdx = -1;
    int gpuIdx = -1;
    int lane = 0;
    int stage = 0;
    uint64_t dependencySpanId0 = 0;
    uint64_t dependencySpanId1 = 0;
    uint64_t batchUid = 0;
    int rowIdx = -1;
  };

  struct SchedulerBusyRealtimeSample {
    uint64_t sequence = 0;
    int64_t startNs = 0;
    int64_t endNs = 0;
  };

  struct SearchRealtimeSlot {
    std::atomic<uint64_t> totalVisits;
    VersionedRingBuffer<SearchLoopRealtimeSample, SEARCH_LOOP_RING_CAPACITY> searchLoops;

    SearchRealtimeSlot() : totalVisits(0), searchLoops() {}
  };

  struct InferenceRealtimeSlot {
    int gpuIdx;
    std::shared_ptr<std::atomic<int>> gpuStreamActiveCount;
    std::atomic<uint64_t> totalRows;
    std::atomic<uint64_t> totalBatches;
    std::atomic<uint64_t> totalBatchSizeSum;
    VersionedRingBuffer<InferenceBatchRealtimeSample, INFERENCE_BATCH_RING_CAPACITY> batches;
    VersionedRingBuffer<DeltaEventSample, INFERENCE_EVENT_RING_CAPACITY> activeEvents;
    VersionedRingBuffer<DeltaEventSample, INFERENCE_EVENT_RING_CAPACITY> streamEvents;

    InferenceRealtimeSlot()
      : gpuIdx(-1),
        gpuStreamActiveCount(),
        totalRows(0),
        totalBatches(0),
        totalBatchSizeSum(0),
        batches(),
        activeEvents(),
        streamEvents()
    {}
  };

  struct RealtimeState {
    mutex configMutex;
    vector<unique_ptr<SearchRealtimeSlot>> searchSlots;
    vector<unique_ptr<InferenceRealtimeSlot>> inferenceSlots;
    vector<pair<int, shared_ptr<std::atomic<int>>>> gpuStreamCounts;
    VersionedRingBuffer<DeltaEventSample, QUEUE_LENGTH_RING_CAPACITY> queueLengthEvents;
    VersionedRingBuffer<TimelineSpanRealtimeSample, TIMELINE_SPAN_RING_CAPACITY> timelineSpans;
    VersionedRingBuffer<SchedulerBusyRealtimeSample, SCHEDULER_BUSY_RING_CAPACITY> schedulerBusySpans;

    atomic<uint64_t> retiredSearchVisits;
    atomic<uint64_t> retiredInferenceRows;
    atomic<uint64_t> retiredInferenceBatches;
    atomic<uint64_t> retiredInferenceBatchSizeSum;

    atomic<int> currentSearchThreadCount;
    atomic<int> currentQueueLength;
    atomic<int> currentInferenceThreadActiveCount;
    bool inferenceSingleSchedulerMode;

    atomic<int64_t> totalCompletedSearchNs;
    atomic<int64_t> currentSearchStartNs;
    atomic<int> activeSearchCount;
    atomic<int64_t> sessionStartNs;
    atomic<int64_t> timelineCaptureStartNs;
    atomic<int64_t> timelineCaptureEndNs;

    mutex publisherMutex;
    condition_variable publisherCv;
    thread publisherThread;
    bool publisherStopRequested;
    int publisherIntervalMs;
    string publisherSocketPath;
    Logger* publisherLogger;
    atomic<uint64_t> sendErrorCount;
    string lastSendError;
    int64_t lastErrorLogNs;
    uint64_t snapshotSequence;

    RealtimeState()
      : configMutex(),
        searchSlots(),
        inferenceSlots(),
        gpuStreamCounts(),
        queueLengthEvents(),
        timelineSpans(),
        schedulerBusySpans(),
        retiredSearchVisits(0),
        retiredInferenceRows(0),
        retiredInferenceBatches(0),
        retiredInferenceBatchSizeSum(0),
        currentSearchThreadCount(0),
        currentQueueLength(0),
        currentInferenceThreadActiveCount(0),
        inferenceSingleSchedulerMode(false),
        totalCompletedSearchNs(0),
        currentSearchStartNs(0),
        activeSearchCount(0),
        sessionStartNs(0),
        timelineCaptureStartNs(0),
        timelineCaptureEndNs(0),
        publisherMutex(),
        publisherCv(),
        publisherThread(),
        publisherStopRequested(false),
        publisherIntervalMs(1000),
        publisherSocketPath(),
        publisherLogger(nullptr),
        sendErrorCount(0),
        lastSendError(),
        lastErrorLogNs(0),
        snapshotSequence(0)
    {}
  };

  atomic<bool> g_enabled(false);
  atomic<bool> g_realtimeRunning(false);
  BenchmarkPerfState g_benchmarkState;
  RealtimeState g_realtimeState;
  thread_local int g_currentSearchThreadIdx = -1;

  static int64_t nowSteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()
    ).count();
  }

  static int64_t nowUnixMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()
    ).count();
  }

  template<typename T>
  static void ensureSize(vector<T>& values, size_t size) {
    if(values.size() < size)
      values.resize(size, T());
  }

  static void resetBenchmarkStateLocked(BenchmarkPerfState& state) {
    state.searchLoopProcessMs.clear();
    state.searchLoopWaitMs.clear();
    state.inferencePreprocessMs.clear();
    state.inferenceH2DMs.clear();
    state.inferenceWaitGpuMs.clear();
    state.inferenceD2HMs.clear();
    state.inferencePostprocessMs.clear();
    state.inferenceLaunchIntervalMs.clear();
    state.gpuWaitGpuMsByBatchSize.clear();
    state.queueLengthSeconds.clear();
    state.currentQueueLength = 0;
    state.inferenceThreadActiveSeconds.clear();
    state.currentInferenceThreadActiveCount = 0;
    state.benchmarkSampleActive = false;
    state.benchmarkSampleStart = Clock::now();
    state.sampleQueueLastUpdate = Clock::now();
    state.sampleInferenceThreadLastUpdate = Clock::now();
    state.sampleQueueSegments.clear();
    state.sampleInferenceThreadSegments.clear();
    state.sampleSearchLoops.clear();
    state.sampleInferencePhases.clear();
    state.sampleLaunchIntervals.clear();
  }

  static double sampleOffsetSeconds(const BenchmarkPerfState& state, Clock::time_point now) {
    return std::chrono::duration<double>(now - state.benchmarkSampleStart).count();
  }

  static void updateQueueSampleSegmentsLocked(BenchmarkPerfState& state, Clock::time_point now) {
    if(!state.benchmarkSampleActive)
      return;
    double elapsedSeconds = std::chrono::duration<double>(now - state.sampleQueueLastUpdate).count();
    if(elapsedSeconds > 0) {
      state.sampleQueueSegments.push_back(CountDurationSegment{
        elapsedSeconds,
        state.currentQueueLength
      });
    }
    state.sampleQueueLastUpdate = now;
  }

  static void updateInferenceThreadSampleSegmentsLocked(BenchmarkPerfState& state, Clock::time_point now) {
    if(!state.benchmarkSampleActive)
      return;
    double elapsedSeconds = std::chrono::duration<double>(now - state.sampleInferenceThreadLastUpdate).count();
    if(elapsedSeconds > 0) {
      state.sampleInferenceThreadSegments.push_back(CountDurationSegment{
        elapsedSeconds,
        state.currentInferenceThreadActiveCount
      });
    }
    state.sampleInferenceThreadLastUpdate = now;
  }

  static void accumulateTrimmedCountSegments(
    const vector<CountDurationSegment>& segments,
    vector<double>& outBuckets,
    double trimmedStart,
    double trimmedEnd
  ) {
    double cursor = 0.0;
    for(const CountDurationSegment& seg: segments) {
      double segStart = cursor;
      double segEnd = cursor + seg.durationSeconds;
      double clippedStart = std::max(segStart, trimmedStart);
      double clippedEnd = std::min(segEnd, trimmedEnd);
      if(clippedEnd > clippedStart) {
        ensureSize(outBuckets, (size_t)seg.count + 1);
        outBuckets[seg.count] += (clippedEnd - clippedStart);
      }
      cursor = segEnd;
    }
  }

  static double percentile(vector<double> values, double p) {
    if(values.empty())
      return 0.0;
    sort(values.begin(), values.end());
    double idx = p * (values.size() - 1);
    size_t lo = (size_t)floor(idx);
    size_t hi = (size_t)ceil(idx);
    if(lo == hi)
      return values[lo];
    double frac = idx - lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
  }

  static string formatPercentileLine(const string& label, const vector<double>& values) {
    if(values.empty())
      return "  " + label + ": n/a";
    return Global::strprintf(
      "  %s: P50=%.3f P95=%.3f P99=%.3f",
      label.c_str(),
      percentile(values, 0.50),
      percentile(values, 0.95),
      percentile(values, 0.99)
    );
  }

  static string formatShareHistogram(const string& label, const string& prefix, const vector<double>& buckets) {
    double total = 0.0;
    for(double value: buckets)
      total += value;
    if(total <= 0.0)
      return "  " + label + ": n/a";

    ostringstream out;
    out << "  " << label << ": ";
    bool first = true;
    int printed = 0;
    for(size_t i = 0; i < buckets.size(); i++) {
      if(buckets[i] <= 0.0)
        continue;
      if(!first) {
        if(printed % 8 == 0)
          out << "\n    ";
        else
          out << ", ";
      }
      out << prefix << i << "=" << Global::strprintf("%.2f%%", buckets[i] * 100.0 / total);
      first = false;
      printed++;
    }
    return out.str();
  }

  static string formatDecileLine(const string& label, const vector<double>& values) {
    if(values.empty())
      return "  " + label + ": n/a";
    ostringstream out;
    out << "  " << label << ": ";
    for(int decile = 1; decile <= 9; decile++) {
      if(decile > 1)
        out << " ";
      double p = (double)decile / 10.0;
      out << "D" << (decile * 10) << "=" << Global::strprintf("%.3f", percentile(values, p));
    }
    return out.str();
  }

  template<typename T, size_t Capacity>
  static void resetRing(VersionedRingBuffer<T, Capacity>& ring) {
    ring.writeCount.store(0, std::memory_order_relaxed);
    for(size_t i = 0; i < Capacity; i++) {
      ring.cells[i].version.store(0, std::memory_order_relaxed);
      ring.cells[i].sample = T();
    }
  }

  template<typename T, size_t Capacity>
  static void writeRingSample(VersionedRingBuffer<T, Capacity>& ring, T sample) {
    uint64_t sequence = ring.writeCount.fetch_add(1, std::memory_order_relaxed);
    sample.sequence = sequence;
    typename VersionedRingBuffer<T, Capacity>::Cell& cell = ring.cells[sequence % Capacity];
    uint64_t version = cell.version.load(std::memory_order_relaxed);
    cell.version.store(version + 1, std::memory_order_release);
    cell.sample = sample;
    cell.version.store(version + 2, std::memory_order_release);
  }

  template<typename T, size_t Capacity>
  static bool readRingSample(const VersionedRingBuffer<T, Capacity>& ring, uint64_t sequence, T& out) {
    const typename VersionedRingBuffer<T, Capacity>::Cell& cell = ring.cells[sequence % Capacity];
    uint64_t version1 = cell.version.load(std::memory_order_acquire);
    if(version1 == 0 || (version1 & 1))
      return false;
    T sample = cell.sample;
    uint64_t version2 = cell.version.load(std::memory_order_acquire);
    if(version1 != version2 || (version2 & 1))
      return false;
    if(sample.sequence != sequence)
      return false;
    out = sample;
    return true;
  }

  template<typename T, size_t Capacity, typename Fn>
  static void forEachRingSample(
    const VersionedRingBuffer<T, Capacity>& ring,
    Fn&& fn
  ) {
    uint64_t end = ring.writeCount.load(std::memory_order_acquire);
    uint64_t start = end > Capacity ? end - Capacity : 0;
    for(uint64_t sequence = start; sequence < end; sequence++) {
      T sample;
      if(readRingSample(ring, sequence, sample))
        fn(sample);
    }
  }

  static void retireSearchSlotsLocked(RealtimeState& state) {
    uint64_t retired = 0;
    for(const unique_ptr<SearchRealtimeSlot>& slot: state.searchSlots) {
      retired += slot->totalVisits.load(std::memory_order_relaxed);
    }
    if(retired > 0)
      state.retiredSearchVisits.fetch_add(retired, std::memory_order_relaxed);
  }

  static void retireInferenceSlotsLocked(RealtimeState& state) {
    uint64_t retiredRows = 0;
    uint64_t retiredBatches = 0;
    uint64_t retiredBatchSizeSum = 0;
    for(const unique_ptr<InferenceRealtimeSlot>& slot: state.inferenceSlots) {
      retiredRows += slot->totalRows.load(std::memory_order_relaxed);
      retiredBatches += slot->totalBatches.load(std::memory_order_relaxed);
      retiredBatchSizeSum += slot->totalBatchSizeSum.load(std::memory_order_relaxed);
    }
    if(retiredRows > 0)
      state.retiredInferenceRows.fetch_add(retiredRows, std::memory_order_relaxed);
    if(retiredBatches > 0)
      state.retiredInferenceBatches.fetch_add(retiredBatches, std::memory_order_relaxed);
    if(retiredBatchSizeSum > 0)
      state.retiredInferenceBatchSizeSum.fetch_add(retiredBatchSizeSum, std::memory_order_relaxed);
  }

  static void resetRealtimeStateLocked(RealtimeState& state) {
    state.searchSlots.clear();
    state.inferenceSlots.clear();
    state.gpuStreamCounts.clear();
    resetRing(state.queueLengthEvents);
    resetRing(state.timelineSpans);
    resetRing(state.schedulerBusySpans);
    state.retiredSearchVisits.store(0, std::memory_order_relaxed);
    state.retiredInferenceRows.store(0, std::memory_order_relaxed);
    state.retiredInferenceBatches.store(0, std::memory_order_relaxed);
    state.retiredInferenceBatchSizeSum.store(0, std::memory_order_relaxed);
    state.currentSearchThreadCount.store(0, std::memory_order_relaxed);
    state.currentQueueLength.store(0, std::memory_order_relaxed);
    state.currentInferenceThreadActiveCount.store(0, std::memory_order_relaxed);
    state.inferenceSingleSchedulerMode = false;
    state.totalCompletedSearchNs.store(0, std::memory_order_relaxed);
    state.currentSearchStartNs.store(0, std::memory_order_relaxed);
    state.activeSearchCount.store(0, std::memory_order_relaxed);
    state.sessionStartNs.store(nowSteadyNs(), std::memory_order_relaxed);
    state.timelineCaptureStartNs.store(0, std::memory_order_relaxed);
    state.timelineCaptureEndNs.store(0, std::memory_order_relaxed);
    state.sendErrorCount.store(0, std::memory_order_relaxed);
    state.lastSendError.clear();
    state.lastErrorLogNs = 0;
    state.snapshotSequence = 0;
  }

  static void logRealtimeError(const string& message) {
    RealtimeState& state = g_realtimeState;
    state.sendErrorCount.fetch_add(1, std::memory_order_relaxed);
    state.lastSendError = message;
    int64_t nowNs = nowSteadyNs();
    if(nowNs - state.lastErrorLogNs < 5 * ONE_SECOND_NS)
      return;
    state.lastErrorLogNs = nowNs;
    if(state.publisherLogger != nullptr)
      state.publisherLogger->write(message);
  }

  static SearchRealtimeSlot* getSearchSlotOrNull(int threadIdx) {
    if(threadIdx < 0)
      return nullptr;
    RealtimeState& state = g_realtimeState;
    if((size_t)threadIdx >= state.searchSlots.size())
      return nullptr;
    return state.searchSlots[threadIdx].get();
  }

  static InferenceRealtimeSlot* getInferenceSlotOrNull(int threadIdx) {
    if(threadIdx < 0)
      return nullptr;
    RealtimeState& state = g_realtimeState;
    if((size_t)threadIdx >= state.inferenceSlots.size())
      return nullptr;
    return state.inferenceSlots[threadIdx].get();
  }

  static int addToNonNegativeAtomic(std::atomic<int>& value, int delta) {
    int current = value.load(std::memory_order_relaxed);
    while(true) {
      int next = current + delta;
      if(next < 0)
        next = 0;
      if(value.compare_exchange_weak(current, next, std::memory_order_relaxed, std::memory_order_relaxed))
        return next;
    }
  }

  static void scheduleNextTimelineCaptureWindow(int64_t captureEndNs) {
    RealtimeState& state = g_realtimeState;
    const int64_t captureStartNs = captureEndNs - TIMELINE_WINDOW_NS;
    state.timelineCaptureStartNs.store(captureStartNs, std::memory_order_relaxed);
    state.timelineCaptureEndNs.store(captureEndNs, std::memory_order_relaxed);
  }

  static bool loadTimelineCaptureWindow(int64_t& captureStartNs, int64_t& captureEndNs) {
    RealtimeState& state = g_realtimeState;
    captureStartNs = state.timelineCaptureStartNs.load(std::memory_order_relaxed);
    captureEndNs = state.timelineCaptureEndNs.load(std::memory_order_relaxed);
    return captureEndNs > captureStartNs && captureEndNs > 0;
  }

  static json bucketArrayJson(const vector<double>& values, bool normalize) {
    json buckets = json::array();
    double total = 0.0;
    for(double value: values)
      total += value;
    for(size_t i = 0; i < values.size(); i++) {
      if(values[i] <= 0.0)
        continue;
      json bucket;
      bucket["bucket"] = (int)i;
      bucket["value"] = normalize && total > 0.0 ? values[i] / total : values[i];
      buckets.push_back(bucket);
    }
    return buckets;
  }

  static json percentileSummaryJson(const vector<double>& values) {
    json result;
    result["count"] = (uint64_t)values.size();
    if(values.empty()) {
      result["has_data"] = false;
      result["deciles"] = json::array();
      result["p95"] = nullptr;
      result["p99"] = nullptr;
      result["max"] = nullptr;
      return result;
    }

    result["has_data"] = true;
    json deciles = json::array();
    for(int decile = 1; decile <= 9; decile++) {
      deciles.push_back({
        {"p", decile * 10},
        {"value", percentile(values, (double)decile / 10.0)}
      });
    }
    result["deciles"] = deciles;
    result["p95"] = percentile(values, 0.95);
    result["p99"] = percentile(values, 0.99);
    result["max"] = *max_element(values.begin(), values.end());
    return result;
  }

  static vector<double> computeTimeShareBuckets(
    vector<DeltaEventSample> events,
    int64_t windowStartNs,
    int64_t nowNs,
    int currentValue
  ) {
    if(nowNs <= windowStartNs)
      return vector<double>();

    sort(events.begin(), events.end(), [](const DeltaEventSample& a, const DeltaEventSample& b) {
      if(a.timestampNs != b.timestampNs)
        return a.timestampNs < b.timestampNs;
      return a.sequence < b.sequence;
    });

    int value = -1;
    for(const DeltaEventSample& event: events) {
      if(event.timestampNs > nowNs)
        break;
      if(event.timestampNs <= windowStartNs && event.valueAfter >= 0)
        value = event.valueAfter;
    }
    if(value < 0) {
      int totalDelta = 0;
      for(const DeltaEventSample& event: events) {
        if(event.timestampNs < windowStartNs || event.timestampNs > nowNs)
          continue;
        totalDelta += event.delta;
      }
      value = currentValue - totalDelta;
      if(value < 0)
        value = 0;
    }

    vector<double> buckets;
    int64_t cursor = windowStartNs;
    for(const DeltaEventSample& event: events) {
      if(event.timestampNs < windowStartNs)
        continue;
      if(event.timestampNs > nowNs)
        break;
      int64_t timestampNs = std::max(windowStartNs, std::min(nowNs, event.timestampNs));
      if(timestampNs > cursor) {
        ensureSize(buckets, (size_t)value + 1);
        buckets[value] += (double)(timestampNs - cursor);
      }
      value = event.valueAfter >= 0 ? event.valueAfter : value + event.delta;
      if(value < 0)
        value = 0;
      cursor = timestampNs;
    }
    if(nowNs > cursor) {
      ensureSize(buckets, (size_t)value + 1);
      buckets[value] += (double)(nowNs - cursor);
    }
    return buckets;
  }

  static int64_t computeTotalSearchNs(int64_t nowNs) {
    RealtimeState& state = g_realtimeState;
    int64_t total = state.totalCompletedSearchNs.load(std::memory_order_relaxed);
    if(state.activeSearchCount.load(std::memory_order_relaxed) > 0) {
      int64_t currentStart = state.currentSearchStartNs.load(std::memory_order_relaxed);
      if(currentStart > 0 && nowNs > currentStart)
        total += nowNs - currentStart;
    }
    return total;
  }

  static int64_t computeCoveredDurationNs(
    vector<SchedulerBusyRealtimeSample>& spans,
    int64_t windowStartNs,
    int64_t windowEndNs
  ) {
    if(spans.empty() || windowEndNs <= windowStartNs)
      return 0;
    sort(spans.begin(), spans.end(), [](const SchedulerBusyRealtimeSample& a, const SchedulerBusyRealtimeSample& b) {
      if(a.startNs != b.startNs)
        return a.startNs < b.startNs;
      return a.endNs < b.endNs;
    });
    int64_t coveredNs = 0;
    int64_t cursorStartNs = 0;
    int64_t cursorEndNs = 0;
    bool hasCursor = false;
    for(const SchedulerBusyRealtimeSample& span: spans) {
      int64_t clippedStartNs = std::max(windowStartNs, span.startNs);
      int64_t clippedEndNs = std::min(windowEndNs, span.endNs);
      if(clippedEndNs <= clippedStartNs)
        continue;
      if(!hasCursor) {
        cursorStartNs = clippedStartNs;
        cursorEndNs = clippedEndNs;
        hasCursor = true;
        continue;
      }
      if(clippedStartNs <= cursorEndNs) {
        cursorEndNs = std::max(cursorEndNs, clippedEndNs);
        continue;
      }
      coveredNs += cursorEndNs - cursorStartNs;
      cursorStartNs = clippedStartNs;
      cursorEndNs = clippedEndNs;
    }
    if(hasCursor)
      coveredNs += cursorEndNs - cursorStartNs;
    return coveredNs;
  }

  static json buildRealtimeSnapshot() {
    RealtimeState& state = g_realtimeState;
    const int64_t nowNs = nowSteadyNs();
    const int64_t windowStartNs = nowNs - ONE_SECOND_NS;
    int64_t timelineStartNs = nowNs;
    int64_t timelineEndNs = nowNs;
    if(!loadTimelineCaptureWindow(timelineStartNs, timelineEndNs)) {
      timelineStartNs = nowNs;
      timelineEndNs = nowNs;
    }
    const int64_t unixMs = nowUnixMs();
    const int currentQueueLengthAtSnapshot = state.currentQueueLength.load(std::memory_order_relaxed);
    const int currentInferenceThreadActiveCountAtSnapshot =
      state.currentInferenceThreadActiveCount.load(std::memory_order_relaxed);

    vector<double> searchLoopTotalMs;
    vector<double> searchLoopProcessMs;
    vector<double> searchLoopWaitMs;
    vector<double> inferenceWaitTaskSubmitMs;
    vector<double> inferencePreprocessMs;
    vector<double> inferenceH2DMs;
    vector<double> inferenceInferMs;
    vector<double> inferenceD2HMs;
    vector<double> inferencePostprocessMs;
    vector<double> searchDepthHistogram;
    vector<double> queueTimeShareBuckets;
    vector<double> inferenceThreadActiveBuckets;
    vector<double> gpuBatchTimeBySize;
    vector<DeltaEventSample> queueEvents;
    vector<DeltaEventSample> inferenceThreadActiveEvents;
    uint64_t totalVisits = state.retiredSearchVisits.load(std::memory_order_relaxed);
    uint64_t totalRows = state.retiredInferenceRows.load(std::memory_order_relaxed);
    uint64_t totalBatches = state.retiredInferenceBatches.load(std::memory_order_relaxed);
    uint64_t totalBatchSizeSum = state.retiredInferenceBatchSizeSum.load(std::memory_order_relaxed);
    uint64_t windowVisits = 0;
    uint64_t windowRows = 0;
    uint64_t windowBatches = 0;
    uint64_t windowBatchSizeSum = 0;
    vector<pair<int, shared_ptr<std::atomic<int>>>> gpuCounters;
    vector<pair<int,int>> inferenceSlotConfigs;
    vector<int> gpuStreamCountsAtSnapshot;
    vector<vector<DeltaEventSample>> streamEventsByGpu;
    vector<vector<double>> gpuBatchTimeBySizeByGpu;
    vector<TimelineSpanRealtimeSample> timelineSpans;
    vector<SchedulerBusyRealtimeSample> schedulerBusySpans;

    {
      lock_guard<mutex> lock(state.configMutex);
      gpuCounters = state.gpuStreamCounts;
      gpuStreamCountsAtSnapshot.reserve(gpuCounters.size());
      for(const auto& entry: gpuCounters) {
        gpuStreamCountsAtSnapshot.push_back(
          entry.second == nullptr ? 0 : entry.second->load(std::memory_order_relaxed)
        );
      }
      streamEventsByGpu.resize(gpuCounters.size());
      gpuBatchTimeBySizeByGpu.resize(gpuCounters.size());

      for(const unique_ptr<SearchRealtimeSlot>& slot: state.searchSlots) {
        totalVisits += slot->totalVisits.load(std::memory_order_relaxed);
        forEachRingSample(slot->searchLoops, [&](const SearchLoopRealtimeSample& sample) {
          if(sample.timestampNs < windowStartNs || sample.timestampNs > nowNs)
            return;
          windowVisits += (uint64_t)std::max(0, sample.visitDelta);
          if(sample.submittedNNEval) {
            ensureSize(searchDepthHistogram, (size_t)std::max(0, sample.depth) + 1);
            searchDepthHistogram[sample.depth] += 1.0;
            searchLoopTotalMs.push_back(sample.totalMs);
            searchLoopProcessMs.push_back(sample.processMs);
            searchLoopWaitMs.push_back(sample.waitMs);
          }
        });
      }

      forEachRingSample(state.queueLengthEvents, [&](const DeltaEventSample& event) {
        if(event.timestampNs <= nowNs)
          queueEvents.push_back(event);
      });

      for(const unique_ptr<InferenceRealtimeSlot>& slot: state.inferenceSlots) {
        inferenceSlotConfigs.push_back(make_pair(slot->gpuIdx, (int)inferenceSlotConfigs.size()));
        totalRows += slot->totalRows.load(std::memory_order_relaxed);
        totalBatches += slot->totalBatches.load(std::memory_order_relaxed);
        totalBatchSizeSum += slot->totalBatchSizeSum.load(std::memory_order_relaxed);
        forEachRingSample(slot->batches, [&](const InferenceBatchRealtimeSample& sample) {
          if(sample.timestampNs < windowStartNs || sample.timestampNs > nowNs)
            return;
          windowRows += (uint64_t)std::max(0, sample.numRows);
          windowBatches += 1;
          windowBatchSizeSum += (uint64_t)std::max(0, sample.batchSize);
          inferenceWaitTaskSubmitMs.push_back(sample.waitTaskSubmitMs);
          inferencePreprocessMs.push_back(sample.preprocessMs);
          inferenceH2DMs.push_back(sample.h2dMs);
          inferenceInferMs.push_back(sample.inferMs);
          inferenceD2HMs.push_back(sample.d2hMs);
          inferencePostprocessMs.push_back(sample.postprocessMs);
          ensureSize(gpuBatchTimeBySize, (size_t)std::max(0, sample.batchSize) + 1);
          gpuBatchTimeBySize[sample.batchSize] += sample.inferMs;
          for(size_t gpuIdx = 0; gpuIdx < gpuCounters.size(); gpuIdx++) {
            if(gpuCounters[gpuIdx].first != sample.gpuIdx)
              continue;
            ensureSize(gpuBatchTimeBySizeByGpu[gpuIdx], (size_t)std::max(0, sample.batchSize) + 1);
            gpuBatchTimeBySizeByGpu[gpuIdx][sample.batchSize] += sample.inferMs;
            break;
          }
        });
        forEachRingSample(slot->activeEvents, [&](const DeltaEventSample& event) {
          if(event.timestampNs <= nowNs)
            inferenceThreadActiveEvents.push_back(event);
        });
        forEachRingSample(slot->streamEvents, [&](const DeltaEventSample& event) {
          if(event.timestampNs > nowNs)
            return;
          for(size_t gpuIdx = 0; gpuIdx < gpuCounters.size(); gpuIdx++) {
            if(gpuCounters[gpuIdx].first == event.gpuIdx) {
              streamEventsByGpu[gpuIdx].push_back(event);
              break;
            }
          }
        });
      }

      forEachRingSample(state.timelineSpans, [&](const TimelineSpanRealtimeSample& sample) {
        if(sample.endNs < timelineStartNs || sample.startNs > timelineEndNs)
          return;
        timelineSpans.push_back(sample);
      });
      forEachRingSample(state.schedulerBusySpans, [&](const SchedulerBusyRealtimeSample& sample) {
        if(sample.endNs < windowStartNs || sample.startNs > nowNs)
          return;
        schedulerBusySpans.push_back(sample);
      });
    }

    queueTimeShareBuckets = computeTimeShareBuckets(
      queueEvents,
      windowStartNs,
      nowNs,
      currentQueueLengthAtSnapshot
    );
    inferenceThreadActiveBuckets = computeTimeShareBuckets(
      inferenceThreadActiveEvents,
      windowStartNs,
      nowNs,
      currentInferenceThreadActiveCountAtSnapshot
    );

    json gpuStreamTimeShareByGpu = json::array();
    json gpuBatchTimeShareByGpu = json::array();
    for(size_t gpuIdx = 0; gpuIdx < gpuCounters.size(); gpuIdx++) {
      int currentCount = gpuIdx < gpuStreamCountsAtSnapshot.size() ? gpuStreamCountsAtSnapshot[gpuIdx] : 0;
      vector<double> streamBuckets = computeTimeShareBuckets(
        streamEventsByGpu[gpuIdx],
        windowStartNs,
        nowNs,
        currentCount
      );
      gpuStreamTimeShareByGpu.push_back({
        {"gpu", gpuCounters[gpuIdx].first},
        {"buckets", bucketArrayJson(streamBuckets, true)}
      });
      gpuBatchTimeShareByGpu.push_back({
        {"gpu", gpuCounters[gpuIdx].first},
        {"buckets", bucketArrayJson(gpuBatchTimeBySizeByGpu[gpuIdx], true)}
      });
    }

    json snapshot;
    snapshot["version"] = 1;
    snapshot["timestamp_unix_ms"] = unixMs;
    snapshot["timestamp_monotonic_ns"] = nowNs;
    snapshot["sequence"] = ++state.snapshotSequence;

    json totals;
    totals["visits"] = totalVisits;
    totals["nn_eval"] = totalRows;
    totals["nn_batches"] = totalBatches;
    totals["search_threads"] = state.currentSearchThreadCount.load(std::memory_order_relaxed);
    totals["search_wall_time_s"] = (double)computeTotalSearchNs(nowNs) / 1e9;
    totals["avg_batch_size"] = totalBatches > 0 ? (double)totalBatchSizeSum / (double)totalBatches : 0.0;
    snapshot["totals"] = totals;

    json window1s;
    window1s["visits_per_s"] = (double)windowVisits;
    window1s["nn_eval_per_s"] = (double)windowRows;
    window1s["nn_batches_per_s"] = (double)windowBatches;
    window1s["avg_batch_size"] = windowBatches > 0 ? (double)windowBatchSizeSum / (double)windowBatches : 0.0;
    window1s["search_depth_histogram"] = bucketArrayJson(searchDepthHistogram, false);
    window1s["queue_length_time_share"] = bucketArrayJson(queueTimeShareBuckets, true);
    window1s["inference_thread_active_time_share"] = bucketArrayJson(inferenceThreadActiveBuckets, true);
    window1s["gpu_batch_time_share"] = bucketArrayJson(gpuBatchTimeBySize, true);
    window1s["gpu_batch_time_share_by_gpu"] = gpuBatchTimeShareByGpu;
    window1s["cuda_stream_active_time_share_by_gpu"] = gpuStreamTimeShareByGpu;
    int64_t schedulerBusyNs = computeCoveredDurationNs(schedulerBusySpans, windowStartNs, nowNs);
    double schedulerBusyShare = windowStartNs < nowNs ? (double)schedulerBusyNs / (double)(nowNs - windowStartNs) : 0.0;
    schedulerBusyShare = std::max(0.0, std::min(1.0, schedulerBusyShare));
    window1s["scheduler_busy_time_share"] = schedulerBusyShare;
    window1s["scheduler_idle_time_share"] = 1.0 - schedulerBusyShare;

    json searchLoop;
    searchLoop["total_ms"] = percentileSummaryJson(searchLoopTotalMs);
    searchLoop["search_ms"] = percentileSummaryJson(searchLoopProcessMs);
    searchLoop["wait_nn_ms"] = percentileSummaryJson(searchLoopWaitMs);
    window1s["search_loop"] = searchLoop;

    json inference;
    inference["wait_task_submit_ms"] = percentileSummaryJson(inferenceWaitTaskSubmitMs);
    inference["preprocess_ms"] = percentileSummaryJson(inferencePreprocessMs);
    inference["h2d_ms"] = percentileSummaryJson(inferenceH2DMs);
    inference["infer_ms"] = percentileSummaryJson(inferenceInferMs);
    inference["d2h_ms"] = percentileSummaryJson(inferenceD2HMs);
    inference["postprocess_ms"] = percentileSummaryJson(inferencePostprocessMs);
    window1s["inference"] = inference;
    snapshot["window1s"] = window1s;

    json status;
    status["socket_path"] = state.publisherSocketPath;
    status["interval_ms"] = state.publisherIntervalMs;
    status["send_error_count"] = state.sendErrorCount.load(std::memory_order_relaxed);
    status["last_send_error"] = state.lastSendError;
    status["inference_mode"] = state.inferenceSingleSchedulerMode ? "single_scheduler_slots" : "legacy_worker_threads";
    status["session_age_s"] = (double)(nowNs - state.sessionStartNs.load(std::memory_order_relaxed)) / 1e9;
    snapshot["status"] = status;

    sort(timelineSpans.begin(), timelineSpans.end(), [](const TimelineSpanRealtimeSample& a, const TimelineSpanRealtimeSample& b) {
      if(a.startNs != b.startNs)
        return a.startNs < b.startNs;
      return a.spanId < b.spanId;
    });
    size_t droppedTimelineSpans = 0;
    if(timelineSpans.size() > TIMELINE_SNAPSHOT_MAX_SPANS) {
      droppedTimelineSpans = timelineSpans.size() - TIMELINE_SNAPSHOT_MAX_SPANS;
      timelineSpans.erase(
        timelineSpans.begin(),
        timelineSpans.begin() + (ptrdiff_t)droppedTimelineSpans
      );
    }
    int64_t effectiveTimelineStartNs = timelineStartNs;
    if(!timelineSpans.empty())
      effectiveTimelineStartNs = std::max(timelineStartNs, timelineSpans.front().startNs);
    json timelineSlots = json::array();
    for(size_t slotIdx = 0; slotIdx < inferenceSlotConfigs.size(); slotIdx++) {
      timelineSlots.push_back({
        {"slot", (int)slotIdx},
        {"gpu", inferenceSlotConfigs[slotIdx].first}
      });
    }
    json timelineSamples = json::array();
    for(const TimelineSpanRealtimeSample& sample: timelineSpans) {
      timelineSamples.push_back(json::array({
        sample.spanId,
        sample.inferenceSlotIdx,
        sample.lane,
        sample.stage,
        sample.batchUid,
        sample.rowIdx,
        sample.startNs - effectiveTimelineStartNs,
        sample.endNs - effectiveTimelineStartNs,
        sample.dependencySpanId0,
        sample.dependencySpanId1
      }));
    }
    snapshot["timeline"] = {
      {"encoding", "compact_v1"},
      {"range_start_ns", effectiveTimelineStartNs},
      {"range_end_ns", timelineEndNs},
      {"max_spans", TIMELINE_SNAPSHOT_MAX_SPANS},
      {"slots", timelineSlots},
      {"spans", timelineSamples},
      {"dropped_spans", droppedTimelineSpans}
    };
    return snapshot;
  }

  static bool sendRealtimeSnapshot(int fd, const string& socketPath, const string& payload) {
    if(socketPath.empty()) {
      logRealtimeError("globalPerfProfile realtime sender: empty socket path");
      return false;
    }
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if(socketPath.size() >= sizeof(addr.sun_path)) {
      logRealtimeError("globalPerfProfile realtime sender: socket path too long: " + socketPath);
      return false;
    }
    memcpy(addr.sun_path, socketPath.c_str(), socketPath.size() + 1);

    ssize_t sent = sendto(
      fd,
      payload.data(),
      payload.size(),
      MSG_DONTWAIT,
      reinterpret_cast<const sockaddr*>(&addr),
      sizeof(addr)
    );
    if(sent < 0 || (size_t)sent != payload.size()) {
      logRealtimeError(
        "globalPerfProfile realtime sender: sendto failed for " + socketPath +
        ": " + string(strerror(errno))
      );
      return false;
    }
    return true;
  }

  static string bindRealtimeSenderSocket(int fd) {
    string senderPath = "/tmp/katago_perf_sender_" + Global::intToString((int)getpid()) + ".sock";
    sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if(senderPath.size() >= sizeof(addr.sun_path)) {
      logRealtimeError("globalPerfProfile realtime sender: sender socket path too long: " + senderPath);
      return string();
    }
    memcpy(addr.sun_path, senderPath.c_str(), senderPath.size() + 1);

    unlink(senderPath.c_str());
    if(bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
      logRealtimeError(
        "globalPerfProfile realtime sender: bind failed for " + senderPath +
        ": " + string(strerror(errno))
      );
      unlink(senderPath.c_str());
      return string();
    }
    return senderPath;
  }

  static void realtimePublisherLoop() {
    int fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    string senderPath;
    if(fd < 0)
      logRealtimeError("globalPerfProfile realtime sender: socket creation failed: " + string(strerror(errno)));
    else
      senderPath = bindRealtimeSenderSocket(fd);

    while(true) {
      {
        unique_lock<mutex> lock(g_realtimeState.publisherMutex);
        if(g_realtimeState.publisherStopRequested)
          break;
      }

      json snapshot = buildRealtimeSnapshot();
      string payload = snapshot.dump();
      if(fd >= 0)
        sendRealtimeSnapshot(fd, g_realtimeState.publisherSocketPath, payload);

      const int64_t nextCaptureEndNs =
        nowSteadyNs() + (int64_t)g_realtimeState.publisherIntervalMs * ONE_MILLISECOND_NS;
      scheduleNextTimelineCaptureWindow(nextCaptureEndNs);

      unique_lock<mutex> lock(g_realtimeState.publisherMutex);
      if(g_realtimeState.publisherCv.wait_for(
           lock,
           std::chrono::milliseconds(g_realtimeState.publisherIntervalMs),
           []() { return g_realtimeState.publisherStopRequested; }
         ))
        break;
    }

    if(fd >= 0)
      close(fd);
    if(!senderPath.empty())
      unlink(senderPath.c_str());
  }
}

GlobalPerfProfile::SearchThreadScope::SearchThreadScope(int threadIdx)
  : previousThreadIdx(g_currentSearchThreadIdx)
{
  g_currentSearchThreadIdx = threadIdx;
}

GlobalPerfProfile::SearchThreadScope::~SearchThreadScope() {
  g_currentSearchThreadIdx = previousThreadIdx;
}

void GlobalPerfProfile::setEnabled(bool enabled) {
  if(!enabled)
    stopRealtime();
  g_enabled.store(enabled, std::memory_order_release);
  if(!enabled) {
    lock_guard<mutex> lock(g_benchmarkState.stateMutex);
    resetBenchmarkStateLocked(g_benchmarkState);
    lock_guard<mutex> realtimeLock(g_realtimeState.configMutex);
    resetRealtimeStateLocked(g_realtimeState);
  }
}

bool GlobalPerfProfile::isEnabled() {
  return g_enabled.load(std::memory_order_acquire);
}

void GlobalPerfProfile::clear() {
  if(!isEnabled())
    return;
  lock_guard<mutex> lock(g_benchmarkState.stateMutex);
  resetBenchmarkStateLocked(g_benchmarkState);
}

void GlobalPerfProfile::startRealtime(const string& socketPath, int intervalMs, Logger* logger) {
  if(!isEnabled())
    return;

  stopRealtime();
  {
    lock_guard<mutex> lock(g_realtimeState.configMutex);
    resetRealtimeStateLocked(g_realtimeState);
  }
  {
    lock_guard<mutex> lock(g_realtimeState.publisherMutex);
    g_realtimeState.publisherStopRequested = false;
    g_realtimeState.publisherIntervalMs = std::max(100, intervalMs);
    g_realtimeState.publisherSocketPath = socketPath;
    g_realtimeState.publisherLogger = logger;
  }

  g_realtimeRunning.store(true, std::memory_order_release);
  g_realtimeState.publisherThread = std::thread(realtimePublisherLoop);
}

void GlobalPerfProfile::stopRealtime() {
  if(!g_realtimeRunning.exchange(false, std::memory_order_acq_rel))
    return;

  {
    lock_guard<mutex> lock(g_realtimeState.publisherMutex);
    g_realtimeState.publisherStopRequested = true;
    g_realtimeState.publisherCv.notify_all();
  }
  if(g_realtimeState.publisherThread.joinable())
    g_realtimeState.publisherThread.join();
}

bool GlobalPerfProfile::isRealtimeRunning() {
  return g_realtimeRunning.load(std::memory_order_acquire);
}

void GlobalPerfProfile::configureSearchSlots(int numThreads) {
  if(numThreads < 0)
    numThreads = 0;
  g_realtimeState.currentSearchThreadCount.store(numThreads, std::memory_order_relaxed);
  if(!isRealtimeRunning())
    return;

  lock_guard<mutex> lock(g_realtimeState.configMutex);
  retireSearchSlotsLocked(g_realtimeState);
  g_realtimeState.searchSlots.clear();
  g_realtimeState.searchSlots.reserve((size_t)numThreads);
  for(int i = 0; i < numThreads; i++)
    g_realtimeState.searchSlots.push_back(std::make_unique<SearchRealtimeSlot>());
}

void GlobalPerfProfile::configureInferenceMode(bool singleSchedulerLogicalSlots) {
  lock_guard<mutex> lock(g_realtimeState.configMutex);
  g_realtimeState.inferenceSingleSchedulerMode = singleSchedulerLogicalSlots;
}

void GlobalPerfProfile::configureInferenceSlots(const vector<int>& gpuIdxByServerThread) {
  if(!isRealtimeRunning())
    return;

  lock_guard<mutex> lock(g_realtimeState.configMutex);
  retireInferenceSlotsLocked(g_realtimeState);
  g_realtimeState.inferenceSlots.clear();
  g_realtimeState.gpuStreamCounts.clear();
  g_realtimeState.currentInferenceThreadActiveCount.store(0, std::memory_order_relaxed);

  vector<pair<int, shared_ptr<std::atomic<int>>>> gpuCounters;
  for(int gpuIdx: gpuIdxByServerThread) {
    int normalizedGpuIdx = gpuIdx < 0 ? 0 : gpuIdx;
    bool found = false;
    for(const auto& entry: gpuCounters) {
      if(entry.first == normalizedGpuIdx) {
        found = true;
        break;
      }
    }
    if(!found)
      gpuCounters.push_back(make_pair(normalizedGpuIdx, std::make_shared<std::atomic<int>>(0)));
  }

  g_realtimeState.gpuStreamCounts = gpuCounters;
  g_realtimeState.inferenceSlots.reserve(gpuIdxByServerThread.size());
  for(int gpuIdx: gpuIdxByServerThread) {
    int normalizedGpuIdx = gpuIdx < 0 ? 0 : gpuIdx;
    unique_ptr<InferenceRealtimeSlot> slot = std::make_unique<InferenceRealtimeSlot>();
    slot->gpuIdx = normalizedGpuIdx;
    for(const auto& entry: g_realtimeState.gpuStreamCounts) {
      if(entry.first == normalizedGpuIdx) {
        slot->gpuStreamActiveCount = entry.second;
        break;
      }
    }
    g_realtimeState.inferenceSlots.push_back(std::move(slot));
  }
}

void GlobalPerfProfile::setCurrentSearchThreadCount(int numThreads) {
  if(numThreads < 0)
    numThreads = 0;
  g_realtimeState.currentSearchThreadCount.store(numThreads, std::memory_order_relaxed);
}

void GlobalPerfProfile::searchSessionStarted() {
  if(!isRealtimeRunning())
    return;
  int oldCount = g_realtimeState.activeSearchCount.fetch_add(1, std::memory_order_relaxed);
  if(oldCount == 0)
    g_realtimeState.currentSearchStartNs.store(nowSteadyNs(), std::memory_order_relaxed);
}

void GlobalPerfProfile::searchSessionEnded() {
  if(!isRealtimeRunning())
    return;
  int oldCount = g_realtimeState.activeSearchCount.fetch_sub(1, std::memory_order_relaxed);
  if(oldCount <= 0) {
    g_realtimeState.activeSearchCount.store(0, std::memory_order_relaxed);
    return;
  }
  if(oldCount == 1) {
    int64_t startNs = g_realtimeState.currentSearchStartNs.exchange(0, std::memory_order_relaxed);
    int64_t endNs = nowSteadyNs();
    if(startNs > 0 && endNs > startNs)
      g_realtimeState.totalCompletedSearchNs.fetch_add(endNs - startNs, std::memory_order_relaxed);
  }
}

void GlobalPerfProfile::beginBenchmarkSample() {
  if(!isEnabled())
    return;
  Clock::time_point now = Clock::now();
  lock_guard<mutex> lock(g_benchmarkState.stateMutex);
  g_benchmarkState.benchmarkSampleActive = true;
  g_benchmarkState.benchmarkSampleStart = now;
  g_benchmarkState.sampleQueueLastUpdate = now;
  g_benchmarkState.sampleInferenceThreadLastUpdate = now;
  g_benchmarkState.sampleQueueSegments.clear();
  g_benchmarkState.sampleInferenceThreadSegments.clear();
  g_benchmarkState.sampleSearchLoops.clear();
  g_benchmarkState.sampleInferencePhases.clear();
  g_benchmarkState.sampleLaunchIntervals.clear();
}

void GlobalPerfProfile::endBenchmarkSample() {
  if(!isEnabled())
    return;
  Clock::time_point now = Clock::now();
  lock_guard<mutex> lock(g_benchmarkState.stateMutex);
  if(!g_benchmarkState.benchmarkSampleActive)
    return;

  updateQueueSampleSegmentsLocked(g_benchmarkState, now);
  updateInferenceThreadSampleSegmentsLocked(g_benchmarkState, now);

  double totalSeconds = sampleOffsetSeconds(g_benchmarkState, now);
  double trimmedStart = BENCHMARK_SAMPLE_TRIM_SECONDS;
  double trimmedEnd = totalSeconds - BENCHMARK_SAMPLE_TRIM_SECONDS;
  if(trimmedEnd > trimmedStart) {
    for(const TimedSearchLoop& sample: g_benchmarkState.sampleSearchLoops) {
      if(sample.offsetSeconds >= trimmedStart && sample.offsetSeconds <= trimmedEnd) {
        g_benchmarkState.searchLoopProcessMs.push_back(sample.processMs);
        g_benchmarkState.searchLoopWaitMs.push_back(sample.waitMs);
      }
    }

    for(const TimedInferencePhases& sample: g_benchmarkState.sampleInferencePhases) {
      if(sample.offsetSeconds >= trimmedStart && sample.offsetSeconds <= trimmedEnd) {
        g_benchmarkState.inferencePreprocessMs.push_back(sample.preprocessMs);
        g_benchmarkState.inferenceH2DMs.push_back(sample.h2dMs);
        g_benchmarkState.inferenceWaitGpuMs.push_back(sample.waitGpuMs);
        g_benchmarkState.inferenceD2HMs.push_back(sample.d2hMs);
        g_benchmarkState.inferencePostprocessMs.push_back(sample.postprocessMs);
        if(sample.batchSize >= 0) {
          ensureSize(g_benchmarkState.gpuWaitGpuMsByBatchSize, (size_t)sample.batchSize + 1);
          g_benchmarkState.gpuWaitGpuMsByBatchSize[sample.batchSize] += sample.waitGpuMs;
        }
      }
    }

    for(const TimedScalar& sample: g_benchmarkState.sampleLaunchIntervals) {
      if(sample.offsetSeconds >= trimmedStart && sample.offsetSeconds <= trimmedEnd)
        g_benchmarkState.inferenceLaunchIntervalMs.push_back(sample.value);
    }

    accumulateTrimmedCountSegments(
      g_benchmarkState.sampleQueueSegments,
      g_benchmarkState.queueLengthSeconds,
      trimmedStart,
      trimmedEnd
    );
    accumulateTrimmedCountSegments(
      g_benchmarkState.sampleInferenceThreadSegments,
      g_benchmarkState.inferenceThreadActiveSeconds,
      trimmedStart,
      trimmedEnd
    );
  }

  g_benchmarkState.benchmarkSampleActive = false;
  g_benchmarkState.sampleQueueSegments.clear();
  g_benchmarkState.sampleInferenceThreadSegments.clear();
  g_benchmarkState.sampleSearchLoops.clear();
  g_benchmarkState.sampleInferencePhases.clear();
  g_benchmarkState.sampleLaunchIntervals.clear();
}

void GlobalPerfProfile::recordSearchLoop(
  int threadIdx,
  double totalMilliseconds,
  double processMilliseconds,
  double waitMilliseconds,
  int depth,
  int visitDelta,
  bool submittedNNEval
) {
  if(!isEnabled())
    return;

  Clock::time_point now = Clock::now();
  {
    lock_guard<mutex> lock(g_benchmarkState.stateMutex);
    if(g_benchmarkState.benchmarkSampleActive && submittedNNEval) {
      g_benchmarkState.sampleSearchLoops.push_back(TimedSearchLoop{
        sampleOffsetSeconds(g_benchmarkState, now),
        processMilliseconds,
        waitMilliseconds
      });
    }
  }

  if(!isRealtimeRunning())
    return;
  SearchRealtimeSlot* slot = getSearchSlotOrNull(threadIdx);
  if(slot == nullptr)
    return;
  if(visitDelta > 0)
    slot->totalVisits.fetch_add((uint64_t)visitDelta, std::memory_order_relaxed);
  writeRingSample(slot->searchLoops, SearchLoopRealtimeSample{
    0,
    nowSteadyNs(),
    totalMilliseconds,
    processMilliseconds,
    waitMilliseconds,
    std::max(0, depth),
    std::max(0, visitDelta),
    submittedNNEval
  });
}

void GlobalPerfProfile::noteQueueLength(int queueLength) {
  if(!isEnabled())
    return;
  if(queueLength < 0)
    queueLength = 0;
  if(isRealtimeRunning()) {
    g_realtimeState.currentQueueLength.store(queueLength, std::memory_order_relaxed);
    writeRingSample(g_realtimeState.queueLengthEvents, DeltaEventSample{
      0,
      nowSteadyNs(),
      0,
      -1,
      queueLength
    });
  }
  Clock::time_point now = Clock::now();
  lock_guard<mutex> lock(g_benchmarkState.stateMutex);
  updateQueueSampleSegmentsLocked(g_benchmarkState, now);
  g_benchmarkState.currentQueueLength = queueLength;
}

void GlobalPerfProfile::changeInferenceThreadActiveCount(int inferenceThreadIdx, int delta) {
  if(!isEnabled())
    return;
  Clock::time_point now = Clock::now();
  {
    lock_guard<mutex> lock(g_benchmarkState.stateMutex);
    updateInferenceThreadSampleSegmentsLocked(g_benchmarkState, now);
    g_benchmarkState.currentInferenceThreadActiveCount += delta;
    if(g_benchmarkState.currentInferenceThreadActiveCount < 0)
      g_benchmarkState.currentInferenceThreadActiveCount = 0;
  }

  if(!isRealtimeRunning())
    return;
  int newValue = addToNonNegativeAtomic(g_realtimeState.currentInferenceThreadActiveCount, delta);
  InferenceRealtimeSlot* slot = getInferenceSlotOrNull(inferenceThreadIdx);
  if(slot == nullptr)
    return;
  writeRingSample(slot->activeEvents, DeltaEventSample{
    0,
    nowSteadyNs(),
    delta,
    slot->gpuIdx,
    newValue
  });
}

void GlobalPerfProfile::recordInferencePhases(
  double preprocessMs,
  double h2dMs,
  double waitGpuMs,
  double d2hMs,
  double postprocessMs,
  int batchSize
) {
  if(!isEnabled())
    return;
  if(batchSize < 0)
    batchSize = 0;
  Clock::time_point now = Clock::now();
  lock_guard<mutex> lock(g_benchmarkState.stateMutex);
  if(!g_benchmarkState.benchmarkSampleActive)
    return;
  g_benchmarkState.sampleInferencePhases.push_back(TimedInferencePhases{
    sampleOffsetSeconds(g_benchmarkState, now),
    preprocessMs,
    h2dMs,
    waitGpuMs,
    d2hMs,
    postprocessMs,
    batchSize
  });
}

void GlobalPerfProfile::recordRealtimeInferenceBatch(
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
) {
  if(!isRealtimeRunning())
    return;
  InferenceRealtimeSlot* slot = getInferenceSlotOrNull(inferenceThreadIdx);
  if(slot == nullptr)
    return;
  slot->totalRows.fetch_add((uint64_t)std::max(0, numRows), std::memory_order_relaxed);
  slot->totalBatches.fetch_add(1, std::memory_order_relaxed);
  slot->totalBatchSizeSum.fetch_add((uint64_t)std::max(0, batchSize), std::memory_order_relaxed);
  writeRingSample(slot->batches, InferenceBatchRealtimeSample{
    0,
    nowSteadyNs(),
    gpuIdx < 0 ? slot->gpuIdx : gpuIdx,
    std::max(0, batchSize),
    std::max(0, numRows),
    waitTaskSubmitMs,
    preprocessMs,
    h2dMs,
    inferMs,
    d2hMs,
    postprocessMs
  });
}

void GlobalPerfProfile::recordRealtimeTimelineSpan(
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
) {
  if(!isRealtimeRunning())
    return;
  if(inferenceSlotIdx < 0 || spanId == 0)
    return;
  int64_t captureStartNs = 0;
  int64_t captureEndNs = 0;
  if(!loadTimelineCaptureWindow(captureStartNs, captureEndNs))
    return;
  InferenceRealtimeSlot* slot = getInferenceSlotOrNull(inferenceSlotIdx);
  if(slot == nullptr)
    return;
  if(startNs <= 0)
    startNs = nowSteadyNs();
  if(endNs < startNs)
    endNs = startNs;
  if(endNs < captureStartNs || startNs > captureEndNs)
    return;
  if(startNs < captureStartNs)
    startNs = captureStartNs;
  if(endNs > captureEndNs)
    endNs = captureEndNs;
  writeRingSample(g_realtimeState.timelineSpans, TimelineSpanRealtimeSample{
    0,
    spanId,
    startNs,
    endNs,
    inferenceSlotIdx,
    gpuIdx < 0 ? slot->gpuIdx : gpuIdx,
    (int)lane,
    (int)stage,
    dependencySpanId0,
    dependencySpanId1,
    batchUid,
    rowIdx
  });
}

void GlobalPerfProfile::recordSchedulerBusySpan(int64_t startNs, int64_t endNs) {
  if(!isRealtimeRunning())
    return;
  if(startNs <= 0)
    startNs = nowSteadyNs();
  if(endNs < startNs)
    endNs = startNs;
  writeRingSample(g_realtimeState.schedulerBusySpans, SchedulerBusyRealtimeSample{
    0,
    startNs,
    endNs
  });
}

bool GlobalPerfProfile::wantsRealtimeTimelineSpan(int64_t startNs, int64_t endNs) {
  if(!isRealtimeRunning())
    return false;
  if(startNs <= 0)
    startNs = nowSteadyNs();
  if(endNs < startNs)
    endNs = startNs;
  int64_t captureStartNs = 0;
  int64_t captureEndNs = 0;
  if(!loadTimelineCaptureWindow(captureStartNs, captureEndNs))
    return false;
  return !(endNs < captureStartNs || startNs > captureEndNs);
}

void GlobalPerfProfile::changeGpuStreamActiveCount(int inferenceThreadIdx, int gpuIdx, int delta) {
  if(!isRealtimeRunning())
    return;
  InferenceRealtimeSlot* slot = getInferenceSlotOrNull(inferenceThreadIdx);
  if(slot == nullptr)
    return;
  int newValue = 0;
  if(slot->gpuStreamActiveCount != nullptr)
    newValue = addToNonNegativeAtomic(*slot->gpuStreamActiveCount, delta);
  writeRingSample(slot->streamEvents, DeltaEventSample{
    0,
    nowSteadyNs(),
    delta,
    gpuIdx < 0 ? slot->gpuIdx : gpuIdx,
    newValue
  });
}

void GlobalPerfProfile::recordInferenceLaunchInterval(double launchIntervalMs) {
  if(!isEnabled())
    return;
  Clock::time_point now = Clock::now();
  lock_guard<mutex> lock(g_benchmarkState.stateMutex);
  if(!g_benchmarkState.benchmarkSampleActive)
    return;
  g_benchmarkState.sampleLaunchIntervals.push_back(TimedScalar{
    sampleOffsetSeconds(g_benchmarkState, now),
    launchIntervalMs
  });
}

string GlobalPerfProfile::makeReport() {
  if(!isEnabled())
    return "";

  vector<double> searchProcessMs;
  vector<double> searchWaitMs;
  vector<double> queueLengthSeconds;
  vector<double> inferencePreprocessMs;
  vector<double> inferenceH2DMs;
  vector<double> inferenceWaitGpuMs;
  vector<double> inferenceD2HMs;
  vector<double> inferencePostprocessMs;
  vector<double> inferenceLaunchIntervalMs;
  vector<double> gpuWaitGpuMsByBatchSize;
  vector<double> inferenceThreadActiveSeconds;

  {
    lock_guard<mutex> lock(g_benchmarkState.stateMutex);
    searchProcessMs = g_benchmarkState.searchLoopProcessMs;
    searchWaitMs = g_benchmarkState.searchLoopWaitMs;
    queueLengthSeconds = g_benchmarkState.queueLengthSeconds;
    inferencePreprocessMs = g_benchmarkState.inferencePreprocessMs;
    inferenceH2DMs = g_benchmarkState.inferenceH2DMs;
    inferenceWaitGpuMs = g_benchmarkState.inferenceWaitGpuMs;
    inferenceD2HMs = g_benchmarkState.inferenceD2HMs;
    inferencePostprocessMs = g_benchmarkState.inferencePostprocessMs;
    inferenceLaunchIntervalMs = g_benchmarkState.inferenceLaunchIntervalMs;
    gpuWaitGpuMsByBatchSize = g_benchmarkState.gpuWaitGpuMsByBatchSize;
    inferenceThreadActiveSeconds = g_benchmarkState.inferenceThreadActiveSeconds;
  }

  ostringstream out;
  out << "globalPerfProfile\n";
  out << formatPercentileLine("search_process_ms", searchProcessMs) << "\n";
  out << formatPercentileLine("search_wait_nn_ms", searchWaitMs) << "\n";
  out << formatShareHistogram("queue_length_time_share", "q", queueLengthSeconds) << "\n";
  out << formatPercentileLine("inference_preprocess_ms", inferencePreprocessMs) << "\n";
  out << formatPercentileLine("inference_h2d_ms", inferenceH2DMs) << "\n";
  out << formatPercentileLine("inference_wait_gpu_ms", inferenceWaitGpuMs) << "\n";
  out << formatPercentileLine("inference_d2h_ms", inferenceD2HMs) << "\n";
  out << formatPercentileLine("inference_postprocess_ms", inferencePostprocessMs) << "\n";
  out << formatDecileLine("inference_launch_interval_ms", inferenceLaunchIntervalMs) << "\n";
  out << formatShareHistogram("inference_thread_time_share", "active", inferenceThreadActiveSeconds) << "\n";
  out << formatShareHistogram("gpu_batch_time_share", "b", gpuWaitGpuMsByBatchSize) << "\n";
  return out.str();
}
