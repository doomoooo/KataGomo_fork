// v0_pseudo.cpp
//
// 这不是可编译实现。
// 它的目标不是“把 v0 写得更优雅”，而是：
// - 在仍然保持 v0 逻辑正确的前提下
// - 尽量把热路径上的抽象税剥掉
// - 明确表达一个更接近最终高性能实现的控制流骨架
//
// 这是当前唯一权威的 v0 控制流伪代码。
// 因此这里会刻意做几件不那么好读、但更接近最终高性能实现的事：
//
// 1. 搜索侧不再用“每次 playout 一个独立异步任务再调度出去”的模型。
//    改成：
//    - 每个搜索线程一个长期驻留的 search worker
//    - search worker 先 drain 全局高优先级 tail queue
//    - 若没有 tail 且搜索未暂停，则 fallback 启动新的 root playout
//
//    这样比“在当前任务里再补一个后继任务 / frame”更直，也更接近最终想要的调度语义。
//
// 2. batch completion 不再用共享完成对象的高阶异步封装。
//    改成 handle-owning 的 intrusive waiter 链表：
//    - waiter 节点嵌在搜索 frame 里
//    - 无 heap 分配
//    - notify_all() 时直接把 frame 压进全局高优先级 tail queue
//
// 3. open-batch claim 不再用全局 cur_hid_lock。
//    改成：
//    - 每个搜索 worker 维护自己的 hid_hint
//    - 每个 handle 用一个 packed atomic state 表示 {generation, claimed_rows, sealed}
//    - claim 用 CAS 做
//
//    这样仍然满足 v0.md 对 open-batch 原子性的要求，
//    但把全局串行锁改成了局部 hint + per-handle CAS。
//
// 4. infer 侧不再是“一 batch 一个异步对象 + 事件等待链”。
//    改成：
//    - 一个长期驻留的 infer worker main loop
//    - 一个 active-handle intrusive list
//    - 对每个 active handle 做无分配状态推进
//
//    也就是说，本版更像“单线程事件泵 + 一批小状态机”，
//    而不是“很多高阶异步对象临时拼起来”。
//
// 5. worker 入口也不再伪装成 coroutine。
//    搜索 worker 和 infer worker 都直接写成长期驻留的线程主循环；
//    等待点只保留成显式的 park/wake 原语。
//
// ------------------------------------------------------------
// 重要说明：
//
// - 本文件仍然遵循 v0.md 的语义边界：
//   * pre_process() 仍在搜索线程
//   * post_process() 进入 playout_tail，由任意搜索线程执行
//   * handle 仍要等最后一个消费者离开后才能重用
//   * GPU timeline 的 reserve 和 reconcile 仍是两件不同的事
//
// - 本文件会大量使用“伪类型 / 伪等待原语 / 伪队列”。
//   重点是把最终高性能实现的状态与时序冻结下来，而不是追求这里能编译。

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

// ============================================================
// 伪类型
// ============================================================

struct Node;
struct NnInput;
struct NnOutput;
struct GpuTimeline;
struct Gpu;
struct CudaEvent;
struct PinnedHostBuffer;

struct Slot {
  int gpu_id;
  int stream_id;
};

// 最终实现里，这类 helper 应该尽量用：
// - 固定容量
// - intrusive 节点
// - 无 heap 分配
// - cache-friendly layout

template <class T>
struct IntrusiveQueue {
  void push(T* node);
  T* pop();
  bool empty() const;
  void swap(IntrusiveQueue& other);
};

struct WorkerWakeEvent {
  // 无丢通知版本：
  // - notify_one() 先 seq.fetch_add(1)，再做唤醒
  // - waiter 先读 observed_seq，再检查队列是否为空
  // - 若为空，则 wait(observed_seq)
  // - wait() 只有在 seq 仍等于 observed_seq 时才真正 park 当前线程
  //
  // 这样就算 push+notify 发生在“empty check”和“park”之间，
  // wait(observed_seq) 也会因为看到 seq 已变化而立即返回，不会睡死。
  std::atomic<uint32_t> seq{0};

  uint32_t prepare_wait() const;
  void wait(uint32_t observed_seq);
  void notify_one();
  void notify_all();
};

struct FixedRowFlags {
  // 固定容量，大小按 Runtime::max_global_batch 预分配。
  // 真实实现里应避免 vector<bool>，并保证 mark_ready 与读取之间有 release/acquire 语义。
  void init(int capacity);
  void clear_prefix(int rows);
  void mark_ready(int row);
  bool all_ready(int row_low, int row_high) const;
};

inline void cpu_relax();

template <class Fn>
void launch_bound_thread(Fn&& entry);

// ============================================================
// 搜索 frame / waiter
// ============================================================

struct SearchWorker;
struct SearchPlayoutFrame;
struct Runtime;

struct BatchDoneWaitNode {
  BatchDoneWaitNode* next = nullptr;
  SearchPlayoutFrame* frame = nullptr;
  uint32_t generation = 0;
};

struct BatchDoneEvent {
  // 单调递增：
  // - generation g 完成时 completed_generation = g
  // - generation g+1 的 waiter 只需检查 completed_generation >= g+1 是否成立
  //
  // 这避免了“每代 reset 一个完成对象”的额外对象管理。
  std::atomic<uint32_t> completed_generation{0};
  std::atomic<BatchDoneWaitNode*> waiters{nullptr};

  // 若 generation 已完成，返回 false，caller 直接走 fast path。
  // 若还没完成，则把 waiter 挂进去并返回 true。
  //
  // 真正实现时，这里需要两次检查 completed_generation 以避免 lost wake。
  // notify_all(generation) 的职责不是“恢复回原 worker”，而是：
  // - 从 waiters 链表摘出本代所有 frame
  // - 对每个 frame 先写 frame.state = SearchFrameState::TailReady
  // - 再调用 enqueue_ready_tail(*frame.rt, frame)
  //
  // 也就是说，batch completion 不仅要把 frame 发布到 Runtime::tailq，
  // 还必须同步触发 search_work_ready 的唤醒。
  bool park_or_consume(uint32_t generation,
                       BatchDoneWaitNode& node,
                       SearchPlayoutFrame& frame);

  void notify_all(uint32_t generation);
};

// ============================================================
// packed handle state
// ============================================================

// 热路径只保留 v0 真正需要原子观察的 3 个字段：
// - generation
// - claimed_rows
// - sealed
//
// 其余信息：
// - slot 是否已发布
// - timeline 是否已 reserve
// - batch 是否已 complete
// - 剩余消费者数
//
// 全都拆成别的字段，避免让一个 state word 背太多语义。
struct OpenStateView {
  uint32_t generation = 1;
  uint16_t claimed_rows = 0;
  bool sealed = false;
};

inline uint64_t pack_open_state(OpenStateView v);
inline OpenStateView unpack_open_state(uint64_t raw);

// ============================================================
// 搜索侧共享状态
// ============================================================

struct SearchSharedState {
  std::atomic<int> search_nn_target_num{0};
  std::atomic<int> search_nn_current_num{0};
  std::atomic<bool> head_launch_paused{false};
};

// ============================================================
// InferHandle：性能版布局
// ============================================================

enum class DriverStage : uint8_t {
  PumpOpen,
  WaitLastH2D,
  WaitInfer,
  WaitD2H,
};

struct InferHandle {
  // ----------------------------------------------------------
  // 热字段：尽量放在独立 cache line
  // ----------------------------------------------------------

  alignas(64) std::atomic<uint64_t> open_state{pack_open_state({})};

  // generation g 的 slot/max_rows 发布完成后，slot_ready_generation = g。
  alignas(64) std::atomic<uint32_t> slot_ready_generation{0};

  // 同一 generation 只能 reserve 一次。
  alignas(64) std::atomic<uint32_t> timeline_reserved_generation{0};

  // completion 后固定下来，搜索侧 postprocess 完成时递减。
  alignas(64) std::atomic<int> remaining_consumers{0};

  // ----------------------------------------------------------
  // 冷字段 / infer-thread-only 字段
  // ----------------------------------------------------------

  Slot bound_slot;
  int bound_max_rows = 0;

  PinnedHostBuffer host_mem;
  FixedRowFlags ready;
  BatchDoneEvent batch_done;

  // 下面这些字段只在 infer worker 线程写，因此不需要原子。
  DriverStage stage = DriverStage::PumpOpen;
  int pumped_rows = 0;
  int final_rows = 0;
  CudaEvent last_h2d_event;
  CudaEvent infer_done_event;
  CudaEvent d2h_done_event;

  // intrusive active list hook
  InferHandle* next_active = nullptr;
};

struct ClaimedRow {
  InferHandle* handle = nullptr;
  uint32_t generation = 0;
  int row = -1;
  void* row_ptr = nullptr;
};

enum class SearchFrameState : uint8_t {
  HeadCpu,
  WaitingBatch,
  TailReady,
};

// ============================================================
// 搜索 frame：手写状态机，不走“每次一个 coroutine”
// ============================================================

struct PlayoutStateMachine {
  PlayoutStateMachine();
  explicit PlayoutStateMachine(Node& root);

  struct Boundary {
    bool need_nn = false;
    NnInput* nn_input = nullptr;
    NnOutput* nn_output = nullptr;
  };

  void reset(Node& root);
  auto run_cpu_until_boundary() -> Boundary;
  void finish_after_nn();
};

struct SearchPlayoutFrame {
  // 全局高优先级 tail queue hook
  SearchPlayoutFrame* next_runnable = nullptr;

  Runtime* rt = nullptr;
  SearchWorker* owner_worker = nullptr;

  // 每个 frame 固定代表“一次 root playout 的逻辑状态”
  SearchFrameState state = SearchFrameState::HeadCpu;
  PlayoutStateMachine machine;

  // 命中 NN 后挂起用到的状态
  ClaimedRow claim;
  NnOutput* nn_output = nullptr;
  std::uint64_t head_cpu_ticks = 0;

  // intrusive waiter 节点嵌在 frame 里，避免 heap
  BatchDoneWaitNode batch_wait;
};

template <class T>
struct FramePool {
  // 这里只承载 owner worker 的本地空闲 frame。
  // foreign-thread release 不直接碰它，而是先进入 owner worker 的 remote_freeq。
  T* acquire();
  void release(T* frame);
};

// ============================================================
// worker / runtime
// ============================================================

struct SearchWorker {
  // 每个 SearchWorker 由 bootstrap() 绑定到一条长期驻留的搜索线程。
  FramePool<SearchPlayoutFrame> pool;
  // tail 允许跨线程执行，因此 frame 释放可能来自别的搜索线程。
  // 这些回收先进入 owner worker 的 remote_freeq，再由 owner worker 在本地 drain。
  IntrusiveQueue<SearchPlayoutFrame> remote_freeq;

  Node* root = nullptr;
  std::size_t hid_hint = 0;
};

struct InferWorker {
  // InferWorker 对应唯一的 infer 控制线程。
  IntrusiveQueue<InferHandle> activeq;
  WorkerWakeEvent work_ready;
};

struct Runtime {
  SearchSharedState search_shared;

  InferWorker infer_worker;
  std::vector<SearchWorker> search_workers;
  std::vector<InferHandle> infer_handles;

  // 已完成 NN、等待执行 post_process + finish_after_nn 的高优先级队列。
  IntrusiveQueue<SearchPlayoutFrame> tailq;
  // tailq 入队或 head launch gate 重开时唤醒搜索 worker。
  WorkerWakeEvent search_work_ready;

  std::vector<Gpu> gpus;
  GpuTimeline* gpu_timeline = nullptr;

  int max_global_batch = 0;
};

// ============================================================
// 前置声明
// ============================================================

int max_batch_for(Runtime& rt, Slot slot);
std::size_t total_stream_batch_capacity(std::vector<Gpu> const& gpus);
bool slot_is_idle(Runtime& rt, Slot slot);
Slot get_recent_slot(GpuTimeline& timeline);

void update_gpu_estimate(Runtime& rt, InferHandle& handle, int batch_rows);
void reconcile_gpu_timeline(Runtime& rt);
void update_search_nn_target_num(Runtime& rt);
// 这里的 worker 只表示“当前执行这次统计归并的搜索线程”。
// head_cpu_ticks 可能来自另一条搜索线程先前执行的 playout_head。
// 统计目标是全局总计算量估计，而不是保留 head/tail 的原线程归属。
void update_search_work_stats(SearchWorker& worker,
                              std::uint64_t head_cpu_ticks,
                              std::uint64_t tail_cpu_ticks);

void* row_host_ptr(PinnedHostBuffer& mem, int row);
void pre_process(NnInput& in, void* host_ptr);
void post_process(void* host_ptr, NnOutput& out);
std::uint64_t thread_cpu_now();
std::uint64_t thread_cpu_elapsed_since(std::uint64_t start);

// 高性能版仍然必须保留 v0 的 slot 粒度：
// slot = {gpu_id, stream_id}。
//
// 因此 infer 侧的 idle 检查、H2D、graph launch、D2H
// 都必须以 slot 为参数，而不是只看 gpu。
CudaEvent h2d_async(InferHandle& handle, int row_low, int row_high, Slot slot);
CudaEvent infer_async(InferHandle& handle, int batch_rows, Slot slot);
CudaEvent d2h_async(InferHandle& handle, int batch_rows, Slot slot);
bool cuda_event_finished(CudaEvent const& ev);

// ============================================================
// 小工具
// ============================================================

inline uint64_t pack_open_state(OpenStateView v) {
  // [63:32] generation
  // [31:16] claimed_rows
  // [15]    sealed
  return (uint64_t(v.generation) << 32) |
         (uint64_t(v.claimed_rows) << 16) |
         (v.sealed ? uint64_t(1) << 15 : 0);
}

inline OpenStateView unpack_open_state(uint64_t raw) {
  return {
    .generation = uint32_t(raw >> 32),
    .claimed_rows = uint16_t((raw >> 16) & 0xFFFFu),
    .sealed = ((raw >> 15) & 0x1u) != 0,
  };
}

void enqueue_ready_tail(Runtime& rt, SearchPlayoutFrame& frame) {
  rt.tailq.push(&frame);
  rt.search_work_ready.notify_one();
}

void enqueue_active(InferWorker& worker, InferHandle& handle) {
  worker.activeq.push(&handle);
  worker.work_ready.notify_one();
}

void maybe_resume_head_launch(Runtime& rt) {
  const int current = rt.search_shared.search_nn_current_num.load(std::memory_order_acquire);
  const int target = rt.search_shared.search_nn_target_num.load(std::memory_order_acquire);

  if (current < target) {
    rt.search_shared.head_launch_paused.store(false, std::memory_order_release);
    rt.search_work_ready.notify_all();
  }
}

void reserve_gpu_timeline_once(Runtime& rt,
                               InferHandle& handle,
                               uint32_t generation,
                               int fixed_rows) {
  // generation 标记而不是 bool：
  // - 无需在 reopen 时清空
  // - 自然避免同一 handle 不同代之间的 ABA 混淆
  const uint32_t old =
    handle.timeline_reserved_generation.exchange(generation, std::memory_order_acq_rel);
  if (old == generation)
    return;

  (void) rt;
  (void) fixed_rows;
  // reserve_gpu_timeline(rt.gpu_timeline, handle.bound_slot, fixed_rows, current_estimate);
}

void reopen_handle_after_last_consumer(InferHandle& handle, uint32_t generation) {
  // 到这里保证：
  // - 本代所有等待者都已经完成 postprocess
  // - infer worker 也早已把 handle 从 active list 里摘掉
  //
  // 因此可以直接切到下一 generation。
  handle.ready.clear_prefix(handle.bound_max_rows);
  handle.bound_max_rows = 0;
  handle.pumped_rows = 0;
  handle.final_rows = 0;
  handle.stage = DriverStage::PumpOpen;

  const uint32_t next_generation = generation + 1;
  handle.open_state.store(
    pack_open_state({
      .generation = next_generation,
      .claimed_rows = 0,
      .sealed = false,
    }),
    std::memory_order_release);
}

void release_batch_consumer(InferHandle& handle, uint32_t generation) {
  const int prev = handle.remaining_consumers.fetch_sub(1, std::memory_order_acq_rel);
  if (prev == 1) {
    reopen_handle_after_last_consumer(handle, generation);
  }
}

// ============================================================
// 高性能 claim：无全局 cur_hid_lock
// ============================================================

ClaimedRow claim_and_preprocess_row_fast(Runtime& rt, SearchWorker& worker, NnInput& in) {
  const std::size_t handle_count = rt.infer_handles.size();
  std::size_t hid = worker.hid_hint;

  for (;;) {
    InferHandle& handle = rt.infer_handles[hid];

    uint64_t raw = handle.open_state.load(std::memory_order_acquire);
    OpenStateView cur = unpack_open_state(raw);

    if (!cur.sealed) {
      // ------------------------------------------------------
      // 冷路径：本代第一次 claim
      // ------------------------------------------------------
      if (cur.claimed_rows == 0) {
        OpenStateView want = cur;
        want.claimed_rows = 1;

        if (handle.open_state.compare_exchange_weak(
              raw, pack_open_state(want),
              std::memory_order_acq_rel,
              std::memory_order_acquire)) {
          worker.hid_hint = hid;

          // 只有首个 claimer 能走到这里，因此可以负责：
          // - 绑定 slot
          // - 发布 max_rows
          // - 重置 ready flags
          // - 把 handle 挂进 infer active queue
          handle.bound_slot = get_recent_slot(*rt.gpu_timeline);
          handle.bound_max_rows = max_batch_for(rt, handle.bound_slot);
          handle.ready.clear_prefix(handle.bound_max_rows);
          handle.pumped_rows = 0;
          handle.final_rows = 0;
          handle.stage = DriverStage::PumpOpen;

          // max_batch == 1 时，首个 row 直接封 batch。
          if (handle.bound_max_rows == 1) {
            reserve_gpu_timeline_once(rt, handle, cur.generation, 1);
            handle.open_state.store(
              pack_open_state({
                .generation = cur.generation,
                .claimed_rows = 1,
                .sealed = true,
              }),
              std::memory_order_release);
          }

          // 这里用 generation 发布“slot/max_rows 已可见”。
          // 后续 claimers 看不到这个 generation，就不会读 bound_slot / bound_max_rows。
          handle.slot_ready_generation.store(cur.generation, std::memory_order_release);

          // infer worker 只在 first-row 时激活一次 handle。
          enqueue_active(rt.infer_worker, handle);

          void* row_ptr = row_host_ptr(handle.host_mem, 0);
          pre_process(in, row_ptr);
          handle.ready.mark_ready(0);

          return {
            .handle = &handle,
            .generation = cur.generation,
            .row = 0,
            .row_ptr = row_ptr,
          };
        }
      }

      // ------------------------------------------------------
      // 热路径：本代已有 open batch，继续 claim
      // ------------------------------------------------------
      else {
        // 先等首个 claimer 把 slot/max_rows 发布完。
        if (handle.slot_ready_generation.load(std::memory_order_acquire) == cur.generation) {
          const int max_rows = handle.bound_max_rows;

          if (cur.claimed_rows < max_rows) {
            OpenStateView want = cur;
            const int row = cur.claimed_rows;
            want.claimed_rows += 1;
            if (want.claimed_rows == max_rows) {
              want.sealed = true;
            }

            if (handle.open_state.compare_exchange_weak(
                  raw, pack_open_state(want),
                  std::memory_order_acq_rel,
                  std::memory_order_acquire)) {
              if (want.sealed) {
                reserve_gpu_timeline_once(rt, handle, cur.generation, want.claimed_rows);
              }

              worker.hid_hint = hid;

              void* row_ptr = row_host_ptr(handle.host_mem, row);
              pre_process(in, row_ptr);
              handle.ready.mark_ready(row);

              return {
                .handle = &handle,
                .generation = cur.generation,
                .row = row,
                .row_ptr = row_ptr,
              };
            }
          }
        }
      }
    }

    // 当前 handle 不是可 claim 的目标，探测下一个。
    hid += 1;
    if (hid == handle_count)
      hid = 0;
    cpu_relax();
  }
}

// ============================================================
// 搜索 worker：persistent loop，tail 优先，root fallback
// ============================================================

void init_root_frame(Runtime& rt, SearchWorker& worker, SearchPlayoutFrame& frame) {
  frame.next_runnable = nullptr;
  frame.rt = &rt;
  frame.owner_worker = &worker;
  frame.state = SearchFrameState::HeadCpu;
  frame.claim = {};
  frame.nn_output = nullptr;
  frame.head_cpu_ticks = 0;
  frame.batch_wait = {};
  frame.machine.reset(*worker.root);
}

void drain_remote_frees(SearchWorker& worker) {
  while (SearchPlayoutFrame* frame = worker.remote_freeq.pop()) {
    worker.pool.release(frame);
  }
}

SearchPlayoutFrame* acquire_search_frame(SearchWorker& worker) {
  drain_remote_frees(worker);
  return worker.pool.acquire();
}

void release_search_frame(SearchWorker& executing_worker, SearchPlayoutFrame& frame) {
  SearchWorker& owner = *frame.owner_worker;
  if (&executing_worker == &owner) {
    owner.pool.release(&frame);
    return;
  }

  owner.remote_freeq.push(&frame);
}

void run_playout_tail(SearchWorker& worker, SearchPlayoutFrame& frame) {
  // frame 已经从 WaitingBatch 进入 TailReady；
  // 这一步只执行 post_process + finish_after_nn + GC。
  const std::uint64_t tail_start = thread_cpu_now();

  post_process(frame.claim.row_ptr, *frame.nn_output);
  release_batch_consumer(*frame.claim.handle, frame.claim.generation);
  frame.machine.finish_after_nn();

  frame.state = SearchFrameState::HeadCpu;
  frame.nn_output = nullptr;

  update_search_work_stats(
    worker,
    frame.head_cpu_ticks,
    thread_cpu_elapsed_since(tail_start));
  release_search_frame(worker, frame);
}

void run_playout_head(Runtime& rt, SearchWorker& worker) {
  SearchPlayoutFrame* frame = acquire_search_frame(worker);
  init_root_frame(rt, worker, *frame);

  const std::uint64_t head_start = thread_cpu_now();
  auto boundary = frame->machine.run_cpu_until_boundary();

  if (boundary.need_nn) {
    const int current =
      rt.search_shared.search_nn_current_num.fetch_add(1, std::memory_order_acq_rel) + 1;
    const int target =
      rt.search_shared.search_nn_target_num.load(std::memory_order_acquire);

    if (current >= target) {
      rt.search_shared.head_launch_paused.store(true, std::memory_order_release);
    }

    frame->claim = claim_and_preprocess_row_fast(rt, worker, *boundary.nn_input);
    frame->nn_output = boundary.nn_output;
    frame->state = SearchFrameState::WaitingBatch;
    frame->head_cpu_ticks = thread_cpu_elapsed_since(head_start);

    // 高性能等待：
    // - 若 batch 已完成，则立即走 fast path，不 park
    // - 否则把 frame 节点挂进 handle.batch_done.waiters
    if (frame->claim.handle->batch_done.park_or_consume(
          frame->claim.generation,
          frame->batch_wait,
          *frame)) {
      return;
    }

    frame->state = SearchFrameState::TailReady;
    enqueue_ready_tail(rt, *frame);
    return;
  }

  update_search_work_stats(
    worker,
    thread_cpu_elapsed_since(head_start),
    /* tail_cpu_ticks = */ 0);
  release_search_frame(worker, *frame);
}

void search_worker_loop(Runtime& rt, SearchWorker& worker) {
  for (;;) {
    drain_remote_frees(worker);

    if (SearchPlayoutFrame* frame = rt.tailq.pop()) {
      run_playout_tail(worker, *frame);
      continue;
    }

    if (!rt.search_shared.head_launch_paused.load(std::memory_order_acquire)) {
      run_playout_head(rt, worker);
      continue;
    }

    const uint32_t observed_seq = rt.search_work_ready.prepare_wait();
    if (!rt.tailq.empty())
      continue;
    if (!rt.search_shared.head_launch_paused.load(std::memory_order_acquire))
      continue;
    rt.search_work_ready.wait(observed_seq);
  }
}

// ============================================================
// infer worker：单线程 active-handle 泵
// ============================================================

bool progress_active_handle(Runtime& rt, InferHandle& handle) {
  const OpenStateView s =
    unpack_open_state(handle.open_state.load(std::memory_order_acquire));
  const uint32_t generation = s.generation;

  const Slot slot = handle.bound_slot;

  switch (handle.stage) {
    case DriverStage::PumpOpen: {
      // ------------------------------------------------------
      // 1. 当前 slot idle 封 batch
      // ------------------------------------------------------
      if (!s.sealed && slot_is_idle(rt, slot)) {
        OpenStateView want = s;
        want.sealed = true;

        uint64_t expect = pack_open_state(s);
        const uint64_t desired = pack_open_state(want);

        if (handle.open_state.compare_exchange_strong(
              expect, desired,
              std::memory_order_acq_rel,
              std::memory_order_acquire)) {
          reserve_gpu_timeline_once(rt, handle, generation, want.claimed_rows);
        }
      }

      // ------------------------------------------------------
      // 2. 若有新的 row 区间已全部 ready，则追加 H2D
      // ------------------------------------------------------
      const OpenStateView now =
        unpack_open_state(handle.open_state.load(std::memory_order_acquire));

      if (now.claimed_rows > handle.pumped_rows &&
          handle.ready.all_ready(handle.pumped_rows, now.claimed_rows)) {
        handle.last_h2d_event =
          h2d_async(handle, handle.pumped_rows, now.claimed_rows, slot);
        handle.pumped_rows = now.claimed_rows;
      }

      // ------------------------------------------------------
      // 3. 已 seal 且已把所有 claim 到的 row 都泵完 H2D，
      //    切到 WaitLastH2D
      // ------------------------------------------------------
      if (now.sealed && handle.pumped_rows == now.claimed_rows) {
        handle.final_rows = handle.pumped_rows;
        handle.stage = DriverStage::WaitLastH2D;
      }

      return true;
    }

    case DriverStage::WaitLastH2D: {
      if (!cuda_event_finished(handle.last_h2d_event))
        return true;

      handle.infer_done_event = infer_async(handle, handle.final_rows, slot);
      handle.stage = DriverStage::WaitInfer;
      return true;
    }

    case DriverStage::WaitInfer: {
      if (!cuda_event_finished(handle.infer_done_event))
        return true;

      handle.d2h_done_event = d2h_async(handle, handle.final_rows, slot);
      // 这里以当前 v0 语义为准：
      // infer 一结束、D2H 一发起，就开始 update / reconcile / target update。
      // 真正的 batch completion / notify_all 仍然要等 D2H 完成。
      update_gpu_estimate(rt, handle, handle.final_rows);
      reconcile_gpu_timeline(rt);
      update_search_nn_target_num(rt);
      maybe_resume_head_launch(rt);
      handle.stage = DriverStage::WaitD2H;
      return true;
    }

    case DriverStage::WaitD2H: {
      if (!cuda_event_finished(handle.d2h_done_event))
        return true;

      // 进入 completion / GC 阶段前，固定本代消费者数。
      handle.remaining_consumers.store(handle.final_rows, std::memory_order_release);

      // 唤醒所有等待这一代 batch 的搜索 frame。
      handle.batch_done.notify_all(generation);

      // 从 infer active list 摘掉。
      // 后续 handle 是否可重用，交给最后一个搜索消费者决定。
      return false;
    }
  }

  return false;
}

void infer_worker_loop(Runtime& rt) {
  IntrusiveQueue<InferHandle> local;

  for (;;) {
    // 把新激活的 handle 批量拿到本地，减少共享队列来回碰撞。
    while (InferHandle* h = rt.infer_worker.activeq.pop()) {
      local.push(h);
    }

    if (local.empty()) {
      const uint32_t observed_seq = rt.infer_worker.work_ready.prepare_wait();
      if (!rt.infer_worker.activeq.empty()) {
        continue;
      }
      rt.infer_worker.work_ready.wait(observed_seq);
      continue;
    }

    IntrusiveQueue<InferHandle> next_round;

    while (InferHandle* h = local.pop()) {
      const bool keep_active = progress_active_handle(rt, *h);
      if (keep_active) {
        next_round.push(h);
      }
    }

    local.swap(next_round);

    // 高性能版本不再为每个 handle 建额外的异步包装/timer；
    // infer worker 自己 cooperative tick。
    cpu_relax();
  }
}

// ============================================================
// 启动
// ============================================================

void bootstrap(Runtime& rt, Node& root, int numSearchThreads) {
  // 与 v0 一致：
  // ring 容量 = 3 * sum(gpu.cuda_streams * gpu.max_batch)
  const std::size_t ring_capacity =
    3 * total_stream_batch_capacity(rt.gpus);

  rt.infer_handles.resize(ring_capacity);

  // 高性能版会按“全局最大 batch”预分配每个 handle 的 host_mem / ready 容量，
  // 用空间换掉每个 batch 的 resize / 重新分配。
  for (Gpu const& gpu : rt.gpus) {
    if (gpu.max_batch > rt.max_global_batch)
      rt.max_global_batch = gpu.max_batch;
  }

  for (InferHandle& h : rt.infer_handles) {
    h.ready.init(rt.max_global_batch);
    // h.host_mem.init_pinned_memory(rt.max_global_batch, ...);
    h.open_state.store(
      pack_open_state({
        .generation = 1,
        .claimed_rows = 0,
        .sealed = false,
      }),
      std::memory_order_release);
  }

  rt.search_workers.resize(numSearchThreads);
  for (SearchWorker& worker : rt.search_workers) {
    worker.root = &root;
    SearchWorker* worker_ptr = &worker;
    launch_bound_thread([&rt, worker_ptr] {
      search_worker_loop(rt, *worker_ptr);
    });
  }

  launch_bound_thread([&rt] {
    infer_worker_loop(rt);
  });
}

int main() {
  Runtime rt;
  Node root;

  bootstrap(rt, root, /* numSearchThreads = */ 0);
  return 0;
}

// ============================================================
// 总结：本版的性能取向
// ============================================================
//
// 1. 热路径不再依赖：
//    - global cur_hid_lock
//    - coroutine frame / promise / awaiter glue
//    - 每次等待的泛型异步 op-state
//    - 每次 playout 的任务包装 / 二次调度
//
// 2. 热路径改成：
//    - per-handle packed CAS claim
//    - 全局高优先级 tail queue + search worker fallback root
//    - intrusive waiter
//    - infer active-handle pump
//
// 3. 仍保持 v0 关键语义不变：
//    - 搜索线程做 pre_process
//    - 搜索线程做 post_process
//    - infer 结束、D2H 一发起就做 estimate / reconcile / target update
//    - handle 只在最后一个消费者离开后复用
//
// 4. 这份文件故意不追求“像教程”。
//    它更像最终实现的骨架草图：读起来更硬，但更贴近会被写进高性能代码里的结构。
