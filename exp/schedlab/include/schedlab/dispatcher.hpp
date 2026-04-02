#pragma once

#include "exec/async_scope.hpp"
#include "exec/single_thread_context.hpp"
#include "exec/task.hpp"
#include "schedlab/infer_backend.hpp"
#include "schedlab/scheduler.hpp"
#include "schedlab/utils/one_shot_event.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace schedlab {
  namespace ex = stdexec;

  struct HostSlotState;

  // dispatcher 暴露给搜索线程的一行 request ticket。
  // 每一行都对应 batch 里的一个 request：
  // 1. inputs_mem_addr / outputs_mem_addr 指向该行每个 tensor 的首地址
  // 2. output_ready 由 dispatcher 在 D2H 完成后触发，唤醒等待结果的搜索线程
  // 3. notify_input_ready() / notify_output_consumed() 则反向汇报搜索线程何时准备好输入、何时消费完输出
  //
  // 这些地址都只是指向 HostSlot 内各个 batch tensor 的切片，不单独拥有内存。
  struct RequestState {
    void init_owner_slot(HostSlotState& owner_slot) noexcept;
    void notify_input_ready() const noexcept;
    void notify_output_consumed() const noexcept;

    // 该行每个输入 tensor 在 host 侧 slab 中的首地址。
    std::vector<void*> inputs_mem_addr;
    // 该行每个输出 tensor 在 host 侧 slab 中的首地址。
    std::vector<void*> outputs_mem_addr;
    // dispatcher 在这行结果可读时触发的一次性事件。
    OneShotEvent output_ready{};

   private:
    HostSlotState* owner_slot = nullptr;
  };

  // dispatcher 对一个 host slot 的运行时封装。
  // 一个 host slot 在任意时刻处于两种状态之一：
  // 1. open: 还在收集新的 request ticket，assigned_rows 可继续增长
  // 2. sealed: 已经交给 infer_coro 处理，不再接收新的 ticket，直到 reset 后重新变回 open
  struct HostSlotState {
    // backend 持有的真实 host 侧 batch 容器；dispatcher 这里只保留引用。
    HostSlot* host_slot = nullptr;
    // dispatcher 自己持有的 per-request 元数据。
    // 这些视图和同步句柄属于 request 生命周期，而不是 backend 的 batch 数据。
    std::unique_ptr<RequestState[]> request_states;
    // 当前这个 host slot 已经分配了多少行 ticket。
    // 对 open slot 来说，这是“已占用行数”；对 sealed slot 来说，这是这次 batch 的真实大小。
    std::uint32_t assigned_rows = 0;
    // 是否已经 seal 并发车。
    // acquire_ticket() 只会向当前未 seal 的 slot 继续分配行。
    bool sealed = false;
    // 已经收到多少次 notify_input_ready()。
    // infer_coro 要等它追平 assigned_rows，才说明这一批输入都准备好了。
    std::atomic<std::uint32_t> input_ready_count{0};
    // 已经收到多少次 notify_output_consumed()。
    // infer_coro 要等它追平 assigned_rows，才会 reset 并复用这个 host slot。
    std::atomic<std::uint32_t> output_consumed_count{0};
  };

  // dispatcher。
  // 所有调度和 completion 轮询都收敛到 dispatcher 自己的单线程 lane 上；
  // 具体的 CUDA/TRT 细节则完全藏在 backend 里。
  //
  // 它在整体系统里的职责可以拆成两半：
  // 1. 给搜索线程分配“当前 open host slot 中的下一行 ticket”
  // 2. 当一个 host slot 值得发车时，把它 seal 成一次真实 batch，并异步推进
  //    H2D -> infer -> D2H -> 搜索线程取回输出 -> host slot 复用 这一整条流程
  //
  // 关键约束是：除了搜索线程往 RequestState 暴露的地址里写输入、读输出之外，
  // host slot 的生命周期管理、seal/reset 状态、以及所有 backend completion 轮询
  // 都只在 dispatcher_context 这条单线程 lane 上进行，因此这里不需要额外的重锁。
  class Dispatcher final {
   public:
    // 构造 dispatcher，并完成一次统一初始化：
    // 1. 分配 host_slot_count 个 host slot
    // 2. 基于 backend 的 BatchLayout 初始化每行 RequestState 的地址视图
    // 3. 按当前配置里的 lane 拓扑初始化 scheduler 里的 infer lane 估计器
    Dispatcher(
      Scheduler& scheduler,
      InferBackend& backend);
    // 析构前会先 wait()，确保所有在飞的 infer_coro 都结束，再让 backend 统一归还 host slot 资源。
    ~Dispatcher();

    // 给搜索线程分配一个 request ticket。
    //
    // 返回值是当前 open host slot 里的一行 RequestState：
    // 1. 搜索线程往 inputs_mem_addr 写输入
    // 2. 写完后调用 notify_input_ready()
    // 3. 等 output_ready
    // 4. 消费完 outputs_mem_addr 后调用 notify_output_consumed()
    //
    // 这个接口本身会切到 dispatcher 线程上执行，因此虽然可由多条搜索线程并发调用，
   // 真正修改 current_host_slot / assigned_rows 的地方仍然是串行的。
    auto acquire_ticket() -> exec::task<RequestState*>;
    // 等待所有已经发车的 host slot 都走完整条 infer 流程。
    // 常用于停机时确保没有悬挂的 backend 任务。
    void wait();

   private:
    // 按 backend 的 batch 布局，把一个 host slot 切成 per-request 地址视图。
    void initialize_request_views(HostSlotState& host_slot) const noexcept;

    // dispatcher 线程上的“忙等式让出调度”小工具。
    // 它反复检查谓词；若未满足，就把控制权让回 dispatcher_context，
    // 以便同一条单线程 lane 上的其它协程继续推进。
    template<typename Predicate>
    auto wait_pred(Predicate predicate) -> exec::task<void> {
      while(!predicate()) {
        co_await ex::schedule(dispatcher_context.get_scheduler());
      }
    }

    // 推进一个 sealed host slot 的完整异步生命周期：
    // 1. 等这一批所有行的 input_ready 都到齐
    // 2. 选择 lane，并通过 backend 提交 H2D / infer / D2H
    // 3. D2H 完成后逐行触发 output_ready，放行搜索线程
    // 4. 等所有行的 output_consumed 都到齐
    // 5. reset 这个 host slot，并通知 scheduler 当前这一批 request 已彻底完成
    //
    // 参数 lane 是一个可选“指定 lane”：
    // - 为空时，说明让 scheduler 自己选预测最优 lane
    // - 非空时，通常表示 dispatcher 想把当前 slot 立即塞给某个刚变空闲的 group
    auto infer_coro(HostSlotState* host_slot, std::optional<InferLane> lane) -> exec::task<void>;
    // 把 current_host_slot 立刻 seal 并发车。
    // 发车后 current_host_slot 会滚动到下一个槽位，供后续 acquire_ticket() 继续装填。
    void launch_current_host_slot(std::optional<InferLane> lane) noexcept;
    // 把一个已经完成整轮生命周期的 host slot 复位成“空的 open slot”。
    // 注意这里只 reset 这次实际使用到的前 assigned_rows 行 one-shot event。
    void reset_host_slot(HostSlotState& host_slot, std::uint32_t assigned_rows) noexcept;

    // 做 dispatcher 的统一初始化：
    // 1. 分配并初始化所有 host slot / RequestState
    // 2. 按当前配置初始化 scheduler 里的各 lane workload 初值
    void init();
    // 全局调度器，dispatcher 通过它更新 request frontier 和 infer lane 在线估计。
    Scheduler& scheduler;
    // 真实 backend 抽象，封装 CUDA / TRT 细节。
    InferBackend& backend;

    // backend 能支持的最大 batch size，同时也是每个 host slot 的最大行数。
    std::uint32_t max_batch_size = 0;
    // host slot 总数；dispatcher 会在这些槽位之间循环复用。
    std::uint32_t host_slot_count = 0;
    // dispatcher 自己的单线程执行上下文。
    // 所有 ticket 分配、seal/reset、completion 轮询都串行地跑在这里。
    exec::single_thread_context dispatcher_context;
    // 承载所有 infer_coro 的 async scope。
    exec::async_scope scope{};

    // 所有 host slot 的运行时数组。
    std::unique_ptr<HostSlotState[]> host_slots;
    // 当前仍处于 open 状态、下一次 acquire_ticket() 会往里塞新 request 的槽位下标。
    std::uint32_t current_host_slot = 0;
  };
}  // namespace schedlab
