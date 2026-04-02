#include "schedlab/config.hpp"
#include "schedlab/scheduler.hpp"
#include "schedlab/utils/poisson.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <vector>

namespace schedlab {
  namespace {
    auto now() noexcept -> TimePoint {
      return std::chrono::steady_clock::now();
    }

    auto to_micros(TimePoint tp) noexcept -> double {
      return std::chrono::duration<double, std::micro>(tp.time_since_epoch()).count();
    }

    // 汇总所有 worker 的 EWMA 分片，得到一个全局搜索侧 requests_per_us。
    auto compute_search_requests_per_us(const std::vector<SearchWorkerCounters>& counters) -> double {
      double total_requests_per_us = 0.0;
      for(const SearchWorkerCounters& shard: counters) {
        total_requests_per_us += shard.requests_per_us_ewma.load(std::memory_order_relaxed);
      }
      return total_requests_per_us;
    }

  }  // namespace

  Ewma::Ewma(double alpha) noexcept : alpha(alpha) {}

  void Ewma::update(double sample) noexcept {
    if(!initialized) {
      current_value = sample;
      initialized = true;
      return;
    }
    current_value = alpha * sample + (1.0 - alpha) * current_value;
  }

  auto Ewma::value(double fallback) const noexcept -> double {
    return initialized ? current_value : fallback;
  }

  void InferEstimator::reset(TimePoint ready_at, std::uint32_t max_batch_size) noexcept {
    // 初始阶段只知道“这条 lane 目前空闲”，还没有 runtime 观测值。
    infer_batch_us.assign(
      static_cast<std::size_t>(max_batch_size) + 1,
      Ewma{schedlab_config().scheduler.infer_ewma_alpha});
    last_infer_done_us = to_micros(ready_at);
    ewma_jitter_us = 0.0;
    last_prediction_error_us = 0.0;
  }

  auto InferEstimator::predict_finish(std::uint32_t batch_size, double ready_at_us, TimePoint now) const noexcept
    -> double {
    // 预测模型故意只看 infer workload 本体。
    // H2D / D2H 的异步重叠和 host 侧收尾不再混进这条 lane 的核心 workload 里。
    const auto& config = schedlab_config().scheduler;
    const double uncertainty = config.infer_prediction_jitter_weight * ewma_jitter_us +
      config.infer_prediction_last_error_weight * std::abs(last_prediction_error_us);
    const double start_us = std::max(to_micros(now), ready_at_us);
    return start_us + infer_batch_us[batch_size].value() + uncertainty;
  }

  void InferEstimator::observe_completion(
    std::uint32_t batch_size,
    double started_us,
    double predicted_finish_us,
    TimePoint finished_at) noexcept {
    // 正确样本是：
    // 起点 = max(本次 submit_infer 前的时刻, 这条 lane 上一次 infer_done 时刻)
    // 终点 = 本次 infer_done 时刻
    const double finished_us = to_micros(finished_at);
    const double sample_us = std::max(1.0, finished_us - started_us);
    infer_batch_us[batch_size].update(sample_us);
    const double error_us = finished_us - predicted_finish_us;
    last_prediction_error_us = error_us;
    ewma_jitter_us =
      ewma_jitter_us == 0.0 ? std::abs(error_us) : 0.8 * ewma_jitter_us + 0.2 * std::abs(error_us);
    last_infer_done_us = finished_us;
  }

  void SearchPlayoutState::start() noexcept {
    cpu_begin_us = to_micros(now());
  }

  void SearchPlayoutState::pause() noexcept {
    const double end_us = to_micros(now());
    accumulated_cpu_us += std::max(1.0, end_us - cpu_begin_us);
  }

  void SearchPlayoutState::finish(bool produced_request) noexcept {
    pause();
    this->produced_request = produced_request;
  }

  SearchScheduler::SearchScheduler(std::uint32_t search_worker_count)
    : worker_counters(search_worker_count) {
    const auto& config = schedlab_config().scheduler;
    const double initial_worker_requests_per_us =
      config.search_initial_requests_per_us /
      static_cast<double>(std::max<std::uint32_t>(1, search_worker_count));
    for(SearchWorkerCounters& counters: worker_counters) {
      counters.requests_per_us_ewma.store(initial_worker_requests_per_us, std::memory_order_relaxed);
    }
  }

  auto SearchScheduler::make_new_state() const noexcept -> SearchPlayoutState {
    SearchPlayoutState state;
    state.start();
    return state;
  }

  void SearchScheduler::submit_state(
    std::uint32_t worker_id,
    const SearchPlayoutState& playout_state) noexcept {
    const double search_ewma_alpha = schedlab_config().scheduler.search_ewma_alpha;
    SearchWorkerCounters& counters = worker_counters[worker_id];
    const double sample_requests_per_us =
      playout_state.produced_request ? 1.0 / std::max(1.0, playout_state.accumulated_cpu_us) : 0.0;
    const double previous = counters.requests_per_us_ewma.load(std::memory_order_relaxed);
    const double updated =
      search_ewma_alpha * sample_requests_per_us + (1.0 - search_ewma_alpha) * previous;
    counters.requests_per_us_ewma.store(updated, std::memory_order_relaxed);
  }

  auto SearchScheduler::requests_per_us() const noexcept -> double {
    return compute_search_requests_per_us(worker_counters);
  }

  void InferScheduler::initialize_infer_lanes() noexcept {
    const auto& infer_config = schedlab_config().infer;
    const double initial_infer_batch_us = schedlab_config().scheduler.infer_initial_batch_us;
    const std::uint32_t group_count = static_cast<std::uint32_t>(infer_config.cuda_device_ids.size());
    const std::uint32_t lanes_per_group = infer_config.lanes_per_device;
    const std::uint32_t max_batch_size = infer_config.batch_size;
    max_batch_burst = 0;
    groups.clear();
    groups.resize(group_count);
    for(std::uint32_t group_id = 0; group_id < group_count; ++group_id) {
      groups[group_id].group_id = group_id;
      groups[group_id].lanes.resize(lanes_per_group);
      groups[group_id].inflight_count = 0;
    }
    const TimePoint current_time = now();
    for(std::uint32_t group_id = 0; group_id < group_count; ++group_id) {
      GroupState& group = groups[group_id];
      for(std::uint32_t lane_id = 0; lane_id < lanes_per_group; ++lane_id) {
        LaneState& lane_state = group.lanes[lane_id];
        lane_state.lane = InferLane{group_id, lane_id};
        lane_state.max_batch_size = max_batch_size;
        max_batch_burst += max_batch_size;
        lane_state.inflight_count = 0;
        lane_state.pending_work.clear();
        lane_state.estimator.reset(current_time, max_batch_size);
        // 所有 lane、所有 batch size 都从统一的经验初值起步。
        for(std::uint32_t batch_size = 1; batch_size <= max_batch_size; ++batch_size) {
          lane_state.estimator.infer_batch_us[batch_size].update(initial_infer_batch_us);
        }
      }
    }
  }

  auto InferScheduler::select_lane(std::uint32_t batch_size) const noexcept -> InferLane {
    const TimePoint current_time = now();
    const GroupState& first_group = groups[0];
    InferLane best = first_group.lanes[0].lane;
    double best_finish_us = std::numeric_limits<double>::infinity();
    for(const GroupState& group: groups) {
      for(const LaneState& lane_state: group.lanes) {
        // 这条 lane 的最早可开工时间，
        // 要么是最近一次 infer_done，要么是当前队尾 workload 的预测完成时刻。
        double ready_at_us = lane_state.estimator.last_infer_done_us;
        if(!lane_state.pending_work.empty()) {
          ready_at_us = lane_state.pending_work.back().predicted_finish_us;
        }
        const double predicted_finish_us =
          lane_state.estimator.predict_finish(batch_size, ready_at_us, current_time);
        if(predicted_finish_us < best_finish_us) {
          best = lane_state.lane;
          best_finish_us = predicted_finish_us;
        }
      }
    }
    return best;
  }

  auto InferScheduler::get_idle_group() const noexcept -> std::optional<std::uint32_t> {
    for(const GroupState& group: groups) {
      if(group.inflight_count == 0) {
        return group.group_id;
      }
    }
    return std::nullopt;
  }

  auto InferScheduler::is_group_idle(std::uint32_t group_id) const noexcept -> bool {
    return groups[group_id].inflight_count == 0;
  }

  void InferScheduler::on_request_ready() noexcept {
    submitted_requests.fetch_add(1);
  }

  void InferScheduler::on_infer_submit(InferLane lane, std::uint32_t batch_size) noexcept {
    LaneState& lane_state = groups[lane.group_id].lanes[lane.lane_id];
    InferEstimator& estimator = lane_state.estimator;
    std::deque<PendingInferWork>& queue = lane_state.pending_work;
    const TimePoint current_time = now();
    double ready_at_us = estimator.last_infer_done_us;
    if(!queue.empty()) {
      ready_at_us = queue.back().predicted_finish_us;
    }
    // submit 的瞬间先把一份预测压进队列，后续 PauseGate 水位会直接用这条时间线。
    const double predicted_finish_us = estimator.predict_finish(batch_size, ready_at_us, current_time);
    queue.push_back(PendingInferWork{batch_size, to_micros(current_time), predicted_finish_us});
    lane_state.inflight_count += 1;
    groups[lane.group_id].inflight_count += 1;
  }

  void InferScheduler::on_infer_done(InferLane lane) noexcept {
    LaneState& lane_state = groups[lane.group_id].lanes[lane.lane_id];
    std::deque<PendingInferWork>& queue = lane_state.pending_work;
    const PendingInferWork work = queue.front();
    queue.pop_front();

    InferEstimator& estimator = lane_state.estimator;
    // 这里就是“真正的 lane workload 样本”：
    // 从本次 submit 与上次 infer_done 的较晚者，量到这次 infer_done。
    const double started_us = std::max(work.submit_us, estimator.last_infer_done_us);
    estimator.observe_completion(work.batch_size, started_us, work.predicted_finish_us, now());
    lane_state.inflight_count -= 1;
    groups[lane.group_id].inflight_count -= 1;
  }

  void InferScheduler::on_request_done() noexcept {
    const double max_starvation_probability = schedlab_config().scheduler.max_starvation_probability;
    const double cpu_requests_per_us = search.requests_per_us();
    auto demand_timeline = timeline();
    const TimelinePoint first_point = demand_timeline.next();
    // 第一个未来提交点的需求是硬约束：这是下一次 infer 之前最后一次能抬 target 的机会。
    std::uint64_t gpu_requests = first_point.demand;

    // 之后沿时间线向后扫，记录最后一个危险点。
    // 一旦当前扫描点相对“当前已选 target”的增量已经超过一轮 burst，
    // 就把更远的风险留给下一次 on_request_done()。
    while(true) {
      const TimelinePoint point = demand_timeline.next();
      const std::uint64_t remaining_demand = point.demand - first_point.demand;
      const double cpu_mean_requests = std::max(0.0, point.tau_us - first_point.tau_us) * cpu_requests_per_us;
      const bool exceeds_target_burst_gate = point.demand - gpu_requests > max_batch_burst;
      const bool is_dangerous =
        poisson_starvation_probability(remaining_demand, cpu_mean_requests) > max_starvation_probability;
      if(is_dangerous) {
        gpu_requests = point.demand;
      }
      if(exceeds_target_burst_gate) {
        break;
      }
    }

    target_requests.fetch_add(gpu_requests);
  }

  InferScheduler::TimelineIterator::TimelineIterator(const InferScheduler& infer) noexcept
    : groups(&infer.groups),
      now_us(to_micros(now())) {
    for(const GroupState& group: infer.groups) {
      for(const LaneState& lane_state: group.lanes) {
        Cursor cursor{lane_state.lane, 0, true, 0.0, lane_state.max_batch_size};
        if(!lane_state.pending_work.empty()) {
          cursor.periodic = false;
          cursor.tau_us = std::max(0.0, lane_state.pending_work.front().predicted_finish_us - now_us);
          cursor.batch_size = lane_state.pending_work.front().batch_size;
        }
        queue.push(cursor);
      }
    }
  }

  auto InferScheduler::TimelineIterator::next() noexcept -> TimelinePoint {
    Cursor cursor = queue.top();
    queue.pop();
    const double tau_us = cursor.tau_us;
    cumulative_demand += cursor.batch_size;
    const LaneState& lane_state = (*groups)[cursor.lane.group_id].lanes[cursor.lane.lane_id];
    // 下一个 cursor 的信息
    cursor.queued_index++;
    if(!cursor.periodic && cursor.queued_index < lane_state.pending_work.size()) {
      cursor.tau_us =
        std::max(0.0, lane_state.pending_work[cursor.queued_index].predicted_finish_us - now_us);
      cursor.batch_size = lane_state.pending_work[cursor.queued_index].batch_size;
    } else {
      cursor.periodic = true;
      cursor.tau_us += lane_state.estimator.infer_batch_us[lane_state.max_batch_size].value();
      cursor.batch_size = lane_state.max_batch_size;
    }
    queue.push(cursor);
    return TimelinePoint{cumulative_demand, tau_us};
  }

  auto InferScheduler::timeline() const noexcept -> TimelineIterator {
    return TimelineIterator{*this};
  }

  Scheduler::Scheduler(std::uint32_t search_worker_count, PauseGate& pause_gate)
    : pause_gate(pause_gate),
      search(search_worker_count),
      infer(search) {}

  void Scheduler::request_stop() noexcept {
    pause_gate.force_open();
  }

  void Scheduler::maybe_close_gate() noexcept {
    if(infer.submitted_requests.load() >= infer.target_requests.load()) {
      pause_gate.set_open(false);
    }
  }

  void Scheduler::maybe_open_gate() noexcept {
    if(infer.submitted_requests.load() < infer.target_requests.load()) {
      pause_gate.set_open(true);
    }
  }

}  // namespace schedlab
