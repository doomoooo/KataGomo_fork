#pragma once

#include "exec/async_scope.hpp"
#include "exec/single_thread_context.hpp"
#include "exec/task.hpp"
#include "schedlab/dispatcher.hpp"
#include "schedlab/scheduler.hpp"
#include "schedlab/utils/mock_phase_runner.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace schedlab {
  // 搜索运行时。
  // 管理所有搜索 worker，并在每个 worker 上持续自复制 root playout 协程。
  class SearchRuntime {
   public:
    // 构造搜索运行时。
    SearchRuntime(Dispatcher& dispatcher, Scheduler& scheduler, MockPhaseRunner& mock_runner);
    // 确保停机并等待所有 worker 退出。
    ~SearchRuntime();

    // 启动所有 worker 的初始 root playout。
    void start();
    // 请求搜索运行时停止。
    void request_stop();
    // 等待所有 worker 上的 task 清空。
    void wait();

   private:
    // 单个搜索线程 lane 的运行期状态。
    struct WorkerLane {
      exec::single_thread_context context;
      exec::async_scope scope;
      std::uint32_t worker_id = 0;

      auto scheduler() noexcept { return context.get_scheduler(); }
    };

    // 单个 root playout 的完整协程体。
    auto root_playout(WorkerLane& worker) -> exec::task<void>;
    // 在指定 worker 上注入一个新的 root playout。
    // 搜索侧只在启动时显式种一条 root，后续靠 playout 自复制维持供给。
    void spawn_root_playout(WorkerLane& worker);

    // 推理 dispatcher。
    Dispatcher& dispatcher;
    // 全局调度器；单次 playout 在本地持有自己的 CPU timer，再把结果上报给它。
    Scheduler& scheduler;
    // CPU phase 模拟器。
    MockPhaseRunner& mock_runner;
    // 所有搜索 worker。
    std::vector<std::unique_ptr<WorkerLane>> workers;
    // 是否已请求停止。
    std::atomic<bool> stopping{false};
  };
}  // namespace schedlab
