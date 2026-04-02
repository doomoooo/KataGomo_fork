#pragma once

#include "schedlab/infer_backend.hpp"
#include "schedlab/utils/pause_gate.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <queue>
#include <vector>

namespace schedlab {
  using TimePoint = std::chrono::steady_clock::time_point;

  // Scheduler 的整体职责：
  // 1. 接收搜索侧和推理侧上报的离散事件
  // 2. 在线估计“搜索侧 request 生成速率”和“各 infer lane 的服务时间”
  // 3. 根据当前时间线推导 PauseGate 应该卡在哪个 request frontier
  //
  // 这里刻意不把 timer 对象暴露给外部。
  // search.cpp / dispatcher.cpp 只负责在正确时刻调用事件接口，
  // 具体的统计口径、预测模型和 PauseGate 控制策略都收敛在这个类里。

  // 最简单的指数滑动平均。
  // 当前只用于 GPU infer workload 的平滑估计。
  class Ewma {
   public:
    explicit Ewma(double alpha = 0.15) noexcept;

    // 输入一个新样本。
    void update(double sample) noexcept;
    // 读取当前平均值；如果还没初始化，就返回 fallback。
    auto value(double fallback = 0.0) const noexcept -> double;

   private:
    double alpha = 0.15;
    double current_value = 0.0;
    bool initialized = false;
  };

  // 搜索侧每个 worker 的 EWMA 分片。
  // worker 线程只写自己的分片；infer 侧读取时再汇总成全局 requests_per_us。
  struct SearchWorkerCounters {
    // 该 worker 当前的 requests_per_us EWMA。
    std::atomic<double> requests_per_us_ewma{0.0};
  };

  // 单条 infer lane 的在线估计器。
  // 这里只建模真正影响调度的 infer workload：
  // 1. 按 batch size 分开的平均 infer 时长
  // 2. 预测误差带来的抖动项
  // 3. 这条 lane 最近一次 infer_done 的时间
  struct InferEstimator {
    // 下标就是 batch size。
    // infer_batch_us[b] 表示这条 lane 在 batch=b 时的在线平均 infer 时间。
    std::vector<Ewma> infer_batch_us;
    // 这条 lane 最近一次 infer_done 的绝对时间。
    // 预测下一次工作何时能开始时，会和 submit 时刻取较晚者。
    double last_infer_done_us = 0.0;
    // 绝对预测误差的平滑量，用来给 finish 预测加一个保守裕量。
    double ewma_jitter_us = 0.0;
    // 最近一次预测误差，保留方向信息。
    double last_prediction_error_us = 0.0;
    // 用“当前 lane 空闲”语义重置 estimator，并分配 batch-size 维度。
    void reset(TimePoint ready_at, std::uint32_t max_batch_size) noexcept;
    // 预测“如果现在再往这条 lane 上排一个 batch，会在何时完成 infer”。
    auto predict_finish(std::uint32_t batch_size, double ready_at_us, TimePoint now) const noexcept -> double;
    // 用一次真实 infer 完成来更新 estimator。
    // 注意 started_us 不是 submit 时刻本身，而是
    // max(本次 submit_infer 时刻, 上一次 infer_done 时刻)。
    void observe_completion(
      std::uint32_t batch_size,
      double started_us,
      double predicted_finish_us,
      TimePoint finished_at) noexcept;
  };

  // 一条 lane 上已经 submit 但尚未 infer_done 的工作。
  // scheduler 只需要知道 batch size、submit 时刻，以及当时给出的预测完成时刻。
  struct PendingInferWork {
    // 这份工作对应的 batch size。
    std::uint32_t batch_size = 0;
    // dispatcher 调用 submit_infer 之前立刻记录的时间。
    double submit_us = 0.0;
    // submit 当下基于 lane 状态给出的预测完成时间。
    double predicted_finish_us = 0.0;
  };

  // 单个搜索 playout 的本地 CPU 计时状态。
  // 生命周期由 search.cpp 持有，但最终样本由 SearchScheduler 消费。
  struct SearchPlayoutState {
    void start() noexcept;
    void pause() noexcept;
    void finish(bool produced_request) noexcept;

   private:
    friend class SearchScheduler;

    double cpu_begin_us = 0.0;
    double accumulated_cpu_us = 0.0;
    bool produced_request = false;
  };

  // 搜索侧事件汇聚器。
  // 只接收 search.cpp 上报的最终 playout 样本。
  class SearchScheduler {
   public:
    explicit SearchScheduler(std::uint32_t search_worker_count);

    auto make_new_state() const noexcept -> SearchPlayoutState;
    // 整个 playout 结束时，search.cpp 直接把本地累计 CPU 时间上报进来。
    void submit_state(std::uint32_t worker_id, const SearchPlayoutState& playout_state) noexcept;
    // 把所有 worker 的 EWMA 分片汇总成全局搜索侧 requests_per_us。
    auto requests_per_us() const noexcept -> double;

   private:
    // 每个 worker 的累计量分片。
    // 搜索线程只写自己的下标，dispatcher 读时再做汇总。
    std::vector<SearchWorkerCounters> worker_counters;
  };

  // infer / dispatcher 侧事件汇聚器。
  // 只接收 dispatcher.cpp 上报的 lane 选择、submit、done 和 batch-consumed 事件。
  class InferScheduler {
   public:
    explicit InferScheduler(const SearchScheduler& search) noexcept : search(search) {}

    // 按当前全局配置里的 group/lane 拓扑初始化所有 lane。
    // 每个 batch size 的初始 infer workload 统一从 config 里的固定值起步。
    void initialize_infer_lanes() noexcept;

    // 选择当前预测最早完成的 lane。返回的是预期完成时间而不是预期启动时间。
    auto select_lane(std::uint32_t batch_size) const noexcept -> InferLane;
    // 返回任意一个当前完全空闲的 group。
    // dispatcher 会用它来决定是不是值得把当前 host slot 提前 seal 掉发车。
    auto get_idle_group() const noexcept -> std::optional<std::uint32_t>;
    // 指定 group 当前是否完全空闲。
    // infer_done 后只需要看刚刚完成那条 lane 所在的 group。
    auto is_group_idle(std::uint32_t group_id) const noexcept -> bool;
    // 搜索线程已经把一个 request 提交给 dispatcher。
    // 这里递增的是累计 submitted request 序号。
    void on_request_ready() noexcept;
    // 某条 lane 即将 submit 一个 infer。
    // 这里会把一份 PendingInferWork 压到该 lane 的队尾。
    void on_infer_submit(InferLane lane, std::uint32_t batch_size) noexcept;
    // 某条 lane 收到了当前队首 workload 的 infer_done。
    // 这里会弹出队首并更新对应的 InferEstimator。
    void on_infer_done(InferLane lane) noexcept;
    // 本次 request 全部完成。
    // 这里会重算下一轮 target_requests frontier。
    void on_request_done() noexcept;

   private:
    struct LaneState {
      // 这条 lane 的身份就是 (group_id, lane_id)。
      InferLane lane{};
      // 这条 lane 在 steady 预测里默认按满 batch 运行。
      std::uint32_t max_batch_size = 0;
      // 这条 lane 的在线服务时间模型。
      InferEstimator estimator{};
      // 这条 lane 已 submit 但还没 infer_done 的工作。
      std::deque<PendingInferWork> pending_work;
      // 这条 lane 当前在飞 batch 数。
      std::uint32_t inflight_count = 0;
    };

    struct GroupState {
      // 调度层内部的逻辑 device group 编号。
      std::uint32_t group_id = 0;
      // 这个 group 上的所有 infer lane，按 lane_id 下标访问。
      std::vector<LaneState> lanes;
      // 这个 group 当前在飞 batch 总数。
      std::uint32_t inflight_count = 0;
    };

   public:
    struct TimelinePoint {
      // 到这个时间点为止，GPU 累计会吞掉多少个 request。
      std::uint64_t demand = 0;
      // 距离“现在”的时间偏移。
      double tau_us = 0.0;
    };

    class TimelineIterator {
     public:
      explicit TimelineIterator(const InferScheduler& infer) noexcept;

      auto next() noexcept -> TimelinePoint;

     private:
      struct Cursor {
        InferLane lane{};
        std::size_t queued_index = 0;
        bool periodic = false;
        double tau_us = 0.0;
        std::uint32_t batch_size = 0;
      };

      struct CursorCmp {
        auto operator()(const Cursor& lhs, const Cursor& rhs) const noexcept -> bool {
          return lhs.tau_us > rhs.tau_us;
        }
      };

      const std::vector<GroupState>* groups = nullptr;
      std::priority_queue<Cursor, std::vector<Cursor>, CursorCmp> queue;
      double now_us = 0.0;
      std::uint64_t cumulative_demand = 0;
    };

    // 返回一个懒时间线迭代器，每次 next() 产出下一个 (demand, tau) 点。
    auto timeline() const noexcept -> TimelineIterator;
    // 累计已提交给 dispatcher 的 request 数。
    std::atomic<std::uint64_t> submitted_requests{0};
    // 下一次该关闭 request gate 的绝对 frontier。
    std::atomic<std::uint64_t> target_requests{0};

   private:
    const SearchScheduler& search;
    // 真正的 [group][lane] 状态树。
    std::vector<GroupState> groups;
    // 所有 lane 各自一个满 batch 的总和。单次 target 提升的最大值。
    std::uint64_t max_batch_burst = 0;
  };

  // 全局调度器。
  // 自己只保留跨 search / infer 两侧共享的协调状态：
  // 1. 累计 submitted request 序号
  // 2. PauseGate 的 target frontier
  // 3. 基于 search 供给和 infer 时间线的联合控制回路
  class Scheduler {
   public:
    Scheduler(std::uint32_t search_worker_count, PauseGate& pause_gate);
    // 全局停机：停止搜索供给，并永久打开 request gate 放行所有等待者。
    void request_stop() noexcept;
    // request frontier 已追上时关闭 gate。
    void maybe_close_gate() noexcept;
    // request frontier 仍在前方时打开 gate。
    void maybe_open_gate() noexcept;

    // 全局 request gate。search / infer 两侧都直接用它。
    PauseGate& pause_gate;
    // search.cpp 使用的事件汇聚器。
    SearchScheduler search;
    // dispatcher.cpp 使用的事件汇聚器。
    InferScheduler infer;
  };
}  // namespace schedlab
