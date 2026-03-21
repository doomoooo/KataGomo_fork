#include "../core/globalperf.h"

#include "../core/global.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>
#include <vector>

using namespace std;

namespace {
  using Clock = std::chrono::steady_clock;
  static constexpr double BENCHMARK_SAMPLE_TRIM_SECONDS = 0.100;
  static constexpr int NUM_STREAM_TYPES = 3;
  static constexpr int NUM_SCHEDULER_WORK_TYPES = 12;

  struct TimedSearchLoop {
    int64_t eventNs;
    double processMs;
    double waitMs;
  };

  struct TimedInterval {
    int64_t startNs;
    int64_t endNs;
  };

  struct TimedValue {
    int64_t eventNs;
    double valueMs;
  };

  struct StreamTaskSample {
    int64_t startNs;
    int64_t endNs;
    double submitWaitMs;
  };

  struct GlobalPerfState {
    mutex stateMutex;

    bool singleSchedulerMode = false;
    int numInferenceSlots = 0;

    vector<double> searchLoopProcessMs;
    vector<double> searchLoopWaitMs;

    double schedulerBusySeconds = 0.0;
    double schedulerTotalSeconds = 0.0;
    array<vector<double>, NUM_SCHEDULER_WORK_TYPES> schedulerWorkMs;

    array<double, NUM_STREAM_TYPES> streamActiveSeconds;
    array<double, NUM_STREAM_TYPES> streamCapacitySeconds;
    array<vector<double>, NUM_STREAM_TYPES> streamSubmitWaitMs;

    bool benchmarkSampleActive = false;
    int64_t benchmarkSampleStartNs = 0;
    vector<TimedSearchLoop> sampleSearchLoops;
    vector<TimedInterval> sampleSchedulerBusyIntervals;
    array<vector<TimedValue>, NUM_SCHEDULER_WORK_TYPES> sampleSchedulerWork;
    array<vector<StreamTaskSample>, NUM_STREAM_TYPES> sampleStreamTasks;
    array<vector<int64_t>, NUM_STREAM_TYPES> sampleLastStreamEndNs;

    GlobalPerfState()
      : streamActiveSeconds(),
        streamCapacitySeconds(),
        streamSubmitWaitMs(),
        sampleStreamTasks(),
        sampleLastStreamEndNs()
    {}
  };

  atomic<bool> g_enabled(false);
  GlobalPerfState g_state;

  static int64_t nowSteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()
    ).count();
  }

  static int streamTypeIndex(GlobalPerfProfile::CudaStreamType type) {
    switch(type) {
    case GlobalPerfProfile::CudaStreamType::H2D:
      return 0;
    case GlobalPerfProfile::CudaStreamType::Infer:
      return 1;
    case GlobalPerfProfile::CudaStreamType::D2H:
      return 2;
    default:
      return 0;
    }
  }

  static int schedulerWorkTypeIndex(GlobalPerfProfile::SchedulerWorkType type) {
    switch(type) {
    case GlobalPerfProfile::SchedulerWorkType::SelectBatch:
      return 0;
    case GlobalPerfProfile::SchedulerWorkType::H2DQuery:
      return 1;
    case GlobalPerfProfile::SchedulerWorkType::PreprocessRow:
      return 2;
    case GlobalPerfProfile::SchedulerWorkType::PackRow:
      return 3;
    case GlobalPerfProfile::SchedulerWorkType::H2DSubmitRow:
      return 4;
    case GlobalPerfProfile::SchedulerWorkType::LaunchBatch:
      return 5;
    case GlobalPerfProfile::SchedulerWorkType::InferQuery:
      return 6;
    case GlobalPerfProfile::SchedulerWorkType::D2HSubmitBatch:
      return 7;
    case GlobalPerfProfile::SchedulerWorkType::D2HQuery:
      return 8;
    case GlobalPerfProfile::SchedulerWorkType::PostprocessRow:
      return 9;
    case GlobalPerfProfile::SchedulerWorkType::UnpackRow:
      return 10;
    case GlobalPerfProfile::SchedulerWorkType::PublishRow:
      return 11;
    default:
      return 0;
    }
  }

  static const char* schedulerWorkTypeLabel(GlobalPerfProfile::SchedulerWorkType type) {
    switch(type) {
    case GlobalPerfProfile::SchedulerWorkType::SelectBatch:
      return "scheduler_select_batch_ms";
    case GlobalPerfProfile::SchedulerWorkType::H2DQuery:
      return "scheduler_h2d_query_ms";
    case GlobalPerfProfile::SchedulerWorkType::PreprocessRow:
      return "scheduler_preprocess_row_ms";
    case GlobalPerfProfile::SchedulerWorkType::PackRow:
      return "scheduler_pack_row_ms";
    case GlobalPerfProfile::SchedulerWorkType::H2DSubmitRow:
      return "scheduler_h2d_submit_row_ms";
    case GlobalPerfProfile::SchedulerWorkType::LaunchBatch:
      return "scheduler_launch_batch_ms";
    case GlobalPerfProfile::SchedulerWorkType::InferQuery:
      return "scheduler_infer_query_ms";
    case GlobalPerfProfile::SchedulerWorkType::D2HSubmitBatch:
      return "scheduler_d2h_submit_batch_ms";
    case GlobalPerfProfile::SchedulerWorkType::D2HQuery:
      return "scheduler_d2h_query_ms";
    case GlobalPerfProfile::SchedulerWorkType::PostprocessRow:
      return "scheduler_postprocess_row_ms";
    case GlobalPerfProfile::SchedulerWorkType::UnpackRow:
      return "scheduler_unpack_row_ms";
    case GlobalPerfProfile::SchedulerWorkType::PublishRow:
      return "scheduler_publish_row_ms";
    default:
      return "scheduler_unknown_ms";
    }
  }

  static void clearSampleStateLocked(GlobalPerfState& state) {
    state.benchmarkSampleActive = false;
    state.benchmarkSampleStartNs = 0;
    state.sampleSearchLoops.clear();
    state.sampleSchedulerBusyIntervals.clear();
    for(int i = 0; i < NUM_SCHEDULER_WORK_TYPES; i++)
      state.sampleSchedulerWork[i].clear();
    for(int i = 0; i < NUM_STREAM_TYPES; i++) {
      state.sampleStreamTasks[i].clear();
      state.sampleLastStreamEndNs[i].assign((size_t)std::max(0, state.numInferenceSlots), 0);
    }
  }

  static void resetMetricsLocked(GlobalPerfState& state) {
    state.searchLoopProcessMs.clear();
    state.searchLoopWaitMs.clear();
    state.schedulerBusySeconds = 0.0;
    state.schedulerTotalSeconds = 0.0;
    for(int i = 0; i < NUM_SCHEDULER_WORK_TYPES; i++)
      state.schedulerWorkMs[i].clear();
    for(int i = 0; i < NUM_STREAM_TYPES; i++) {
      state.streamActiveSeconds[i] = 0.0;
      state.streamCapacitySeconds[i] = 0.0;
      state.streamSubmitWaitMs[i].clear();
    }
    clearSampleStateLocked(state);
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

  static string formatShareLine(const string& label, double numeratorSeconds, double denominatorSeconds) {
    if(denominatorSeconds <= 0.0)
      return "  " + label + ": n/a";
    double share = numeratorSeconds / denominatorSeconds;
    share = std::max(0.0, std::min(1.0, share));
    return Global::strprintf("  %s: %.2f%%", label.c_str(), share * 100.0);
  }

  static double computeCoveredSeconds(
    const vector<TimedInterval>& intervals,
    int64_t trimmedStartNs,
    int64_t trimmedEndNs
  ) {
    vector<TimedInterval> clipped;
    clipped.reserve(intervals.size());
    for(const TimedInterval& interval: intervals) {
      int64_t startNs = std::max(interval.startNs, trimmedStartNs);
      int64_t endNs = std::min(interval.endNs, trimmedEndNs);
      if(endNs > startNs)
        clipped.push_back(TimedInterval{startNs, endNs});
    }
    if(clipped.empty())
      return 0.0;

    sort(clipped.begin(), clipped.end(), [](const TimedInterval& a, const TimedInterval& b) {
      if(a.startNs != b.startNs)
        return a.startNs < b.startNs;
      return a.endNs < b.endNs;
    });

    int64_t coveredNs = 0;
    int64_t currentStartNs = clipped[0].startNs;
    int64_t currentEndNs = clipped[0].endNs;
    for(size_t i = 1; i < clipped.size(); i++) {
      if(clipped[i].startNs <= currentEndNs) {
        currentEndNs = std::max(currentEndNs, clipped[i].endNs);
      }
      else {
        coveredNs += currentEndNs - currentStartNs;
        currentStartNs = clipped[i].startNs;
        currentEndNs = clipped[i].endNs;
      }
    }
    coveredNs += currentEndNs - currentStartNs;
    return (double)coveredNs / 1e9;
  }

  static double sumClippedTaskSeconds(
    const vector<StreamTaskSample>& tasks,
    int64_t trimmedStartNs,
    int64_t trimmedEndNs
  ) {
    double totalSeconds = 0.0;
    for(const StreamTaskSample& task: tasks) {
      int64_t startNs = std::max(task.startNs, trimmedStartNs);
      int64_t endNs = std::min(task.endNs, trimmedEndNs);
      if(endNs > startNs)
        totalSeconds += (double)(endNs - startNs) / 1e9;
    }
    return totalSeconds;
  }
}

void GlobalPerfProfile::setEnabled(bool enabled) {
  g_enabled.store(enabled, std::memory_order_release);
  if(!enabled) {
    lock_guard<mutex> lock(g_state.stateMutex);
    resetMetricsLocked(g_state);
  }
}

bool GlobalPerfProfile::isEnabled() {
  return g_enabled.load(std::memory_order_acquire);
}

void GlobalPerfProfile::clear() {
  if(!isEnabled())
    return;
  lock_guard<mutex> lock(g_state.stateMutex);
  resetMetricsLocked(g_state);
}

void GlobalPerfProfile::configureInferenceResources(bool singleSchedulerMode, int numInferenceSlots) {
  if(!isEnabled())
    return;
  lock_guard<mutex> lock(g_state.stateMutex);
  g_state.singleSchedulerMode = singleSchedulerMode;
  g_state.numInferenceSlots = std::max(0, numInferenceSlots);
  if(!g_state.benchmarkSampleActive) {
    for(int i = 0; i < NUM_STREAM_TYPES; i++)
      g_state.sampleLastStreamEndNs[i].assign((size_t)g_state.numInferenceSlots, 0);
  }
}

void GlobalPerfProfile::beginBenchmarkSample() {
  if(!isEnabled())
    return;
  lock_guard<mutex> lock(g_state.stateMutex);
  clearSampleStateLocked(g_state);
  g_state.benchmarkSampleActive = true;
  g_state.benchmarkSampleStartNs = nowSteadyNs();
}

void GlobalPerfProfile::endBenchmarkSample() {
  if(!isEnabled())
    return;
  lock_guard<mutex> lock(g_state.stateMutex);
  if(!g_state.benchmarkSampleActive)
    return;

  int64_t endNs = nowSteadyNs();
  int64_t trimmedStartNs = g_state.benchmarkSampleStartNs + (int64_t)(BENCHMARK_SAMPLE_TRIM_SECONDS * 1e9 + 0.5);
  int64_t trimmedEndNs = endNs - (int64_t)(BENCHMARK_SAMPLE_TRIM_SECONDS * 1e9 + 0.5);
  if(trimmedEndNs > trimmedStartNs) {
    double trimmedDurationSeconds = (double)(trimmedEndNs - trimmedStartNs) / 1e9;

    for(const TimedSearchLoop& sample: g_state.sampleSearchLoops) {
      if(sample.eventNs >= trimmedStartNs && sample.eventNs <= trimmedEndNs) {
        g_state.searchLoopProcessMs.push_back(sample.processMs);
        g_state.searchLoopWaitMs.push_back(sample.waitMs);
      }
    }

    if(g_state.singleSchedulerMode) {
      g_state.schedulerBusySeconds += computeCoveredSeconds(
        g_state.sampleSchedulerBusyIntervals,
        trimmedStartNs,
        trimmedEndNs
      );
      g_state.schedulerTotalSeconds += trimmedDurationSeconds;
    }

    for(int i = 0; i < NUM_SCHEDULER_WORK_TYPES; i++) {
      for(const TimedValue& sample: g_state.sampleSchedulerWork[i]) {
        if(sample.eventNs >= trimmedStartNs && sample.eventNs <= trimmedEndNs)
          g_state.schedulerWorkMs[i].push_back(sample.valueMs);
      }
    }

    for(int i = 0; i < NUM_STREAM_TYPES; i++) {
      if(g_state.numInferenceSlots > 0)
        g_state.streamCapacitySeconds[i] += trimmedDurationSeconds * (double)g_state.numInferenceSlots;
      g_state.streamActiveSeconds[i] += sumClippedTaskSeconds(
        g_state.sampleStreamTasks[i],
        trimmedStartNs,
        trimmedEndNs
      );
      for(const StreamTaskSample& sample: g_state.sampleStreamTasks[i]) {
        if(sample.startNs >= trimmedStartNs && sample.startNs <= trimmedEndNs)
          g_state.streamSubmitWaitMs[i].push_back(sample.submitWaitMs);
      }
    }
  }

  clearSampleStateLocked(g_state);
}

void GlobalPerfProfile::recordSearchLoop(double processMilliseconds, double waitMilliseconds) {
  if(!isEnabled())
    return;
  lock_guard<mutex> lock(g_state.stateMutex);
  if(!g_state.benchmarkSampleActive)
    return;
  g_state.sampleSearchLoops.push_back(TimedSearchLoop{
    nowSteadyNs(),
    processMilliseconds,
    waitMilliseconds
  });
}

void GlobalPerfProfile::recordSchedulerBusySpan(int64_t startNs, int64_t endNs) {
  if(!isEnabled())
    return;
  if(startNs <= 0)
    startNs = nowSteadyNs();
  if(endNs < startNs)
    endNs = startNs;

  lock_guard<mutex> lock(g_state.stateMutex);
  if(!g_state.benchmarkSampleActive)
    return;
  g_state.sampleSchedulerBusyIntervals.push_back(TimedInterval{startNs, endNs});
}

void GlobalPerfProfile::recordSchedulerWork(SchedulerWorkType type, double durationMs) {
  if(!isEnabled() || durationMs < 0.0)
    return;

  lock_guard<mutex> lock(g_state.stateMutex);
  if(!g_state.benchmarkSampleActive)
    return;
  g_state.sampleSchedulerWork[schedulerWorkTypeIndex(type)].push_back(TimedValue{
    nowSteadyNs(),
    durationMs
  });
}

void GlobalPerfProfile::recordCudaStreamTask(
  CudaStreamType type,
  int streamIdx,
  int64_t cpuSubmitEndNs,
  double taskDurationMs
) {
  if(!isEnabled() || streamIdx < 0 || cpuSubmitEndNs <= 0)
    return;

  int typeIdx = streamTypeIndex(type);
  int64_t taskDurationNs = (int64_t)std::llround(std::max(0.0, taskDurationMs) * 1e6);

  lock_guard<mutex> lock(g_state.stateMutex);
  if(!g_state.benchmarkSampleActive)
    return;

  vector<int64_t>& lastStreamEndNs = g_state.sampleLastStreamEndNs[typeIdx];
  if(streamIdx >= (int)lastStreamEndNs.size())
    lastStreamEndNs.resize((size_t)streamIdx + 1, 0);

  int64_t taskStartNs = std::max(lastStreamEndNs[streamIdx], cpuSubmitEndNs);
  int64_t taskEndNs = taskStartNs + taskDurationNs;
  lastStreamEndNs[streamIdx] = taskEndNs;

  g_state.sampleStreamTasks[typeIdx].push_back(StreamTaskSample{
    taskStartNs,
    taskEndNs,
    (double)std::max<int64_t>(0, taskStartNs - cpuSubmitEndNs) / 1e6
  });
}

string GlobalPerfProfile::makeReport() {
  if(!isEnabled())
    return "";

  vector<double> searchProcessMs;
  vector<double> searchWaitMs;
  double schedulerBusySeconds = 0.0;
  double schedulerTotalSeconds = 0.0;
  array<vector<double>, NUM_SCHEDULER_WORK_TYPES> schedulerWorkMs;
  array<double, NUM_STREAM_TYPES> streamActiveSeconds;
  array<double, NUM_STREAM_TYPES> streamCapacitySeconds;
  array<vector<double>, NUM_STREAM_TYPES> streamSubmitWaitMs;

  {
    lock_guard<mutex> lock(g_state.stateMutex);
    searchProcessMs = g_state.searchLoopProcessMs;
    searchWaitMs = g_state.searchLoopWaitMs;
    schedulerBusySeconds = g_state.schedulerBusySeconds;
    schedulerTotalSeconds = g_state.schedulerTotalSeconds;
    schedulerWorkMs = g_state.schedulerWorkMs;
    streamActiveSeconds = g_state.streamActiveSeconds;
    streamCapacitySeconds = g_state.streamCapacitySeconds;
    streamSubmitWaitMs = g_state.streamSubmitWaitMs;
  }

  ostringstream out;
  out << "globalPerfProfile\n";
  out << formatPercentileLine("search_process_ms", searchProcessMs) << "\n";
  out << formatPercentileLine("search_wait_nn_ms", searchWaitMs) << "\n";
  out << formatShareLine("scheduler_busy_time_share", schedulerBusySeconds, schedulerTotalSeconds) << "\n";
  out << formatShareLine(
    "scheduler_idle_time_share",
    schedulerTotalSeconds - schedulerBusySeconds,
    schedulerTotalSeconds
  ) << "\n";
  for(int i = 0; i < NUM_SCHEDULER_WORK_TYPES; i++) {
    out << formatPercentileLine(
      schedulerWorkTypeLabel(static_cast<SchedulerWorkType>(i)),
      schedulerWorkMs[i]
    ) << "\n";
  }
  out << formatShareLine("h2d_stream_occupancy", streamActiveSeconds[0], streamCapacitySeconds[0]) << "\n";
  out << formatShareLine("infer_stream_occupancy", streamActiveSeconds[1], streamCapacitySeconds[1]) << "\n";
  out << formatShareLine("d2h_stream_occupancy", streamActiveSeconds[2], streamCapacitySeconds[2]) << "\n";
  out << formatPercentileLine("h2d_submit_wait_ms", streamSubmitWaitMs[0]) << "\n";
  out << formatPercentileLine("infer_submit_wait_ms", streamSubmitWaitMs[1]) << "\n";
  out << formatPercentileLine("d2h_submit_wait_ms", streamSubmitWaitMs[2]) << "\n";
  return out.str();
}
