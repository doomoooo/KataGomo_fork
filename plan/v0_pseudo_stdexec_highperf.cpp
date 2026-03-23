// v0_pseudo_stdexec_highperf.cpp
//
// 这不是可编译实现。
// 它的目标不是“把 v0 写得更优雅”，而是：
// - 在仍然保持 v0 逻辑正确的前提下
// - 尽量把热路径上的抽象税剥掉
// - 明确表达一个更接近最终高性能实现的控制流骨架
//
// 和 plan/v0_pseudo_stdexec.cpp 的关系：
// - stdexec 版追求“P2300 心智模型清晰”
// - 本版追求“热路径少分配、少调度 hop、少 type-erasure、少全局锁”
//
// 因此本版会刻意做几件不那么好读、但更接近高性能实现的事：
//
// 1. 搜索侧不再用“每次 playout 一个 task + async_scope::spawn”。
//    改成：
//    - 每个搜索线程一个长期驻留的 lane main loop
//    - lane 内部自己维护 runnable playout frame 队列
//    - playout 在命中 NN 叶子后 park 自己，同时把一个新的 root frame 压进本 lane
//
//    这与 v0 的“始终有一个下一次 playout 在排队”逻辑等价，
//    但避免了每次 playout 都新建 task / sender op-state。
//
// 2. batch completion 不再用 ensure_started + split。
//    改成 handle-owning 的 intrusive waiter 链表：
//    - waiter 节点嵌在搜索 frame 里
//    - 无 heap 分配
//    - notify_all() 时直接把 frame 重新压回所属搜索 lane 的 run queue
//
// 3. open-batch claim 不再用全局 cur_hid_lock。
//    改成：
//    - 每个搜索 lane 维护自己的 hid_hint
//    - 每个 handle 用一个 packed atomic state 表示 {generation, claimed_rows, sealed}
//    - claim 用 CAS 做
//
//    这样仍然满足 v0.md 对 open-batch 原子性的要求，
//    但把全局串行锁改成了局部 hint + per-handle CAS。
//
// 4. infer 侧不再是“一 batch 一个 task + sender 等待事件”。
//    改成：
//    - 一个长期驻留的 infer lane main loop
//    - 一个 active-handle intrusive list
//    - 对每个 active handle 做无分配状态推进
//
//    也就是说，本版更像“单线程事件泵 + 一批小状态机”，
//    而不是“很多 sender/task 临时拼起来”。
//
// 5. CUDA event / row-ready 等待不再走泛型 sender bridge。
//    改成 infer lane 自己 polling / cooperative tick。
//    这会牺牲一点抽象美观，但能避免 callback bridge 和额外 hop。
//
// ------------------------------------------------------------
// 重要说明：
//
// - 本文件仍然遵循 v0.md / v0_pseudo.py 的语义边界：
//   * pre_process() 仍在搜索线程
//   * post_process() 仍回到原搜索 lane
//   * handle 仍要等最后一个消费者离开后才能重用
//   * GPU timeline 的 reserve 和 reconcile 仍是两件不同的事
//
// - 本文件会大量使用“伪类型 / 伪 awaitable / 伪队列”。
//   重点是把最终高性能实现的状态与时序冻结下来，而不是追求这里能编译。

#include <stdexec/execution.hpp>

#include <exec/async_scope.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/start_detached.hpp>
#include <exec/task.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ex = stdexec;

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

struct LaneWakeEvent {
  // 伪 awaitable：
  // - 若队列已有工作则立即返回
  // - 否则 park 当前 lane main loop
  auto async_wait() -> /* awaitable<void> */ int;
  void notify_one();
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

// ============================================================
// 搜索 frame / waiter
// ============================================================

struct SearchLane;
struct SearchPlayoutFrame;

struct BatchDoneWaitNode {
  BatchDoneWaitNode* next = nullptr;
  SearchLane* lane = nullptr;
  SearchPlayoutFrame* frame = nullptr;
  uint32_t generation = 0;
};

struct PauseWaitNode {
  PauseWaitNode* next = nullptr;
  SearchLane* lane = nullptr;
  SearchPlayoutFrame* frame = nullptr;
};

struct BatchDoneEvent {
  // 单调递增：
  // - generation g 完成时 completed_generation = g
  // - generation g+1 的 waiter 只需检查 completed_generation >= g+1 是否成立
  //
  // 这避免了“每代 reset 一个完成 sender”的额外对象管理。
  std::atomic<uint32_t> completed_generation{0};
  std::atomic<BatchDoneWaitNode*> waiters{nullptr};

  // 若 generation 已完成，返回 false，caller 直接走 fast path。
  // 若还没完成，则把 waiter 挂进去并返回 true。
  //
  // 真正实现时，这里需要两次检查 completed_generation 以避免 lost wake。
  bool park_or_consume(uint32_t generation,
                       BatchDoneWaitNode& node,
                       SearchLane& lane,
                       SearchPlayoutFrame& frame);

  void notify_all(uint32_t generation);
};

struct PauseGate {
  std::atomic<PauseWaitNode*> waiters{nullptr};

  // 若 paused_flag 已经是 false，则直接返回 false。
  // 否则把 frame park 在 gate 上并返回 true。
  bool park_if_paused(std::atomic<bool>& paused_flag,
                      PauseWaitNode& node,
                      SearchLane& lane,
                      SearchPlayoutFrame& frame);

  // 搜索可继续时，统一把所有 waiter frame 压回各自 lane。
  void notify_all();
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
  std::atomic<bool> search_coro_pause{false};

  PauseGate pause_gate;
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

  // 下面这些字段只在 infer lane 线程写，因此不需要原子。
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

// ============================================================
// 搜索 frame：手写状态机，不走“每次一个 coroutine”
// ============================================================

struct PlayoutMachine {
  PlayoutMachine();
  explicit PlayoutMachine(Node& root);

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
  // runnable queue hook
  SearchPlayoutFrame* next_runnable = nullptr;

  // 每个 frame 固定代表“一次 root playout 的逻辑状态”
  PlayoutMachine machine;

  // 命中 NN 后挂起用到的状态
  bool waiting_for_batch = false;
  ClaimedRow claim;
  NnOutput* nn_output = nullptr;

  // intrusive waiter 节点嵌在 frame 里，避免 heap
  BatchDoneWaitNode batch_wait;
  PauseWaitNode pause_wait;
};

template <class T>
struct FramePool {
  T* acquire();
  void release(T* frame);
};

// ============================================================
// lane / runtime
// ============================================================

struct SearchLane {
  exec::single_thread_context ctx;
  decltype(ctx.get_scheduler()) sched = ctx.get_scheduler();

  IntrusiveQueue<SearchPlayoutFrame> runq;
  LaneWakeEvent work_ready;
  FramePool<SearchPlayoutFrame> pool;

  Node* root = nullptr;

  // 本 lane 下次优先从哪个 handle 开始探测。
  // 这不是 correctness 必需，而是减少 ring 扫描成本。
  std::size_t hid_hint = 0;
};

struct InferLane {
  exec::single_thread_context ctx;
  decltype(ctx.get_scheduler()) sched = ctx.get_scheduler();

  IntrusiveQueue<InferHandle> activeq;
  LaneWakeEvent work_ready;
};

struct Runtime {
  SearchSharedState search_shared;

  InferLane infer_lane;
  std::vector<SearchLane> search_lanes;
  std::vector<InferHandle> infer_handles;

  std::vector<Gpu> gpus;
  GpuTimeline* gpu_timeline = nullptr;

  int max_global_batch = 0;
};

// ============================================================
// 前置声明
// ============================================================

int max_batch_for(Runtime& rt, Slot slot);
std::size_t total_stream_batch_capacity(std::vector<Gpu> const& gpus);
bool gpu_is_idle(Gpu& gpu);
Slot get_recent_slot(GpuTimeline& timeline);

void update_gpu_estimate(Runtime& rt, InferHandle& handle, int batch_rows);
void reconcile_gpu_timeline(Runtime& rt);
void update_search_nn_target_num(Runtime& rt);
void update_search_coro_stats(SearchLane& lane);

void* row_host_ptr(PinnedHostBuffer& mem, int row);
void pre_process(NnInput& in, void* host_ptr);
void post_process(void* host_ptr, NnOutput& out);

CudaEvent h2d_async(InferHandle& handle, int row_low, int row_high, Gpu& gpu);
CudaEvent infer_async(InferHandle& handle, int batch_rows, Gpu& gpu);
CudaEvent d2h_async(InferHandle& handle, int batch_rows, Gpu& gpu);
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

void enqueue_runnable(SearchLane& lane, SearchPlayoutFrame& frame) {
  lane.runq.push(&frame);
  lane.work_ready.notify_one();
}

void enqueue_active(InferLane& lane, InferHandle& handle) {
  lane.activeq.push(&handle);
  lane.work_ready.notify_one();
}

void maybe_resume_root_playout(SearchSharedState& s) {
  const int current = s.search_nn_current_num.load(std::memory_order_acquire);
  const int target = s.search_nn_target_num.load(std::memory_order_acquire);

  if (current < target) {
    s.search_coro_pause.store(false, std::memory_order_release);
    s.pause_gate.notify_all();
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
  // - infer lane 也早已把 handle 从 active list 里摘掉
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

ClaimedRow claim_and_preprocess_row_fast(Runtime& rt, SearchLane& lane, NnInput& in) {
  const std::size_t handle_count = rt.infer_handles.size();
  std::size_t hid = lane.hid_hint;

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
          lane.hid_hint = hid;

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

          // infer lane 只在 first-row 时激活一次 handle。
          enqueue_active(rt.infer_lane, handle);

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

              lane.hid_hint = hid;

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
// 搜索 lane：自管 run queue，不再每次 spawn 一个 task
// ============================================================

void spawn_root_frame(SearchLane& lane) {
  SearchPlayoutFrame* frame = lane.pool.acquire();
  frame->next_runnable = nullptr;
  frame->waiting_for_batch = false;
  frame->claim = {};
  frame->nn_output = nullptr;
  frame->batch_wait = {};
  frame->pause_wait = {};
  frame->machine.reset(*lane.root);
  enqueue_runnable(lane, *frame);
}

void step_search_frame(Runtime& rt, SearchLane& lane, SearchPlayoutFrame& frame) {
  // ----------------------------------------------------------
  // 1. 这是一个从 batch completion 恢复回来的 frame
  // ----------------------------------------------------------
  if (frame.waiting_for_batch) {
    post_process(frame.claim.row_ptr, *frame.nn_output);
    release_batch_consumer(*frame.claim.handle, frame.claim.generation);

    frame.machine.finish_after_nn();
    frame.waiting_for_batch = false;
    frame.nn_output = nullptr;

    update_search_coro_stats(lane);
    lane.pool.release(&frame);
    return;
  }

  // ----------------------------------------------------------
  // 2. root gate：若搜索被暂停，则 park 当前 frame
  // ----------------------------------------------------------
  if (rt.search_shared.search_coro_pause.load(std::memory_order_acquire)) {
    if (rt.search_shared.pause_gate.park_if_paused(
          rt.search_shared.search_coro_pause,
          frame.pause_wait,
          lane,
          frame)) {
      return;
    }
  }

  // ----------------------------------------------------------
  // 3. 继续 CPU 搜索，直到：
  //    - 命中 NN boundary
  //    - 或一次 root playout 结束
  // ----------------------------------------------------------
  auto boundary = frame.machine.run_cpu_until_boundary();

  // ----------------------------------------------------------
  // 4. 命中 NN boundary
  // ----------------------------------------------------------
  if (boundary.need_nn) {
    const int current =
      rt.search_shared.search_nn_current_num.fetch_add(1, std::memory_order_acq_rel) + 1;
    const int target =
      rt.search_shared.search_nn_target_num.load(std::memory_order_acquire);

    if (current >= target) {
      rt.search_shared.search_coro_pause.store(true, std::memory_order_release);
    }

    // v0 要求：当前 frame 即将等待 GPU 前，必须先在同 lane 再补发一个 root playout。
    // 这里不再 spawn task，而是直接向本 lane runq 塞一个新 frame。
    spawn_root_frame(lane);

    // 搜索线程自己 claim row + pre_process。
    frame.claim = claim_and_preprocess_row_fast(rt, lane, *boundary.nn_input);
    frame.nn_output = boundary.nn_output;
    frame.waiting_for_batch = true;

    // 高性能等待：
    // - 若 batch 已完成，则立即走 fast path，不 park
    // - 否则把 frame 节点挂进 handle.batch_done.waiters
    if (frame.claim.handle->batch_done.park_or_consume(
          frame.claim.generation,
          frame.batch_wait,
          lane,
          frame)) {
      return;
    }

    // completion 比 register wait 更早发生时，直接把自己重新入队，
    // 让 post_process 路径仍在 lane main loop 里统一处理。
    enqueue_runnable(lane, frame);
    return;
  }

  // ----------------------------------------------------------
  // 5. 本次 root playout 无 NN，结束并补发后继
  // ----------------------------------------------------------
  update_search_coro_stats(lane);
  spawn_root_frame(lane);
  lane.pool.release(&frame);
}

auto search_lane_main(Runtime& rt, SearchLane& lane) -> exec::task<void> {
  // 每个 lane 开机时先放入一个 root frame。
  spawn_root_frame(lane);

  for (;;) {
    SearchPlayoutFrame* frame = lane.runq.pop();
    if (frame == nullptr) {
      co_await lane.work_ready.async_wait();
      continue;
    }
    step_search_frame(rt, lane, *frame);
  }
}

// ============================================================
// infer lane：单线程 active-handle 泵
// ============================================================

bool progress_active_handle(Runtime& rt, InferHandle& handle) {
  const OpenStateView s =
    unpack_open_state(handle.open_state.load(std::memory_order_acquire));
  const uint32_t generation = s.generation;

  Gpu& gpu = rt.gpus[handle.bound_slot.gpu_id];

  switch (handle.stage) {
    case DriverStage::PumpOpen: {
      // ------------------------------------------------------
      // 1. GPU idle 封 batch
      // ------------------------------------------------------
      if (!s.sealed && gpu_is_idle(gpu)) {
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
          h2d_async(handle, handle.pumped_rows, now.claimed_rows, gpu);
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

      handle.infer_done_event = infer_async(handle, handle.final_rows, gpu);
      handle.stage = DriverStage::WaitInfer;
      return true;
    }

    case DriverStage::WaitInfer: {
      if (!cuda_event_finished(handle.infer_done_event))
        return true;

      handle.d2h_done_event = d2h_async(handle, handle.final_rows, gpu);
      handle.stage = DriverStage::WaitD2H;
      return true;
    }

    case DriverStage::WaitD2H: {
      if (!cuda_event_finished(handle.d2h_done_event))
        return true;

      // 这里明确采用 v0.md 的 completion-point 顺序：
      // D2H 真正完成后，再做 update / reconcile / target update。
      update_gpu_estimate(rt, handle, handle.final_rows);
      reconcile_gpu_timeline(rt);
      update_search_nn_target_num(rt);
      maybe_resume_root_playout(rt.search_shared);

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

auto infer_lane_main(Runtime& rt) -> exec::task<void> {
  IntrusiveQueue<InferHandle> local;

  for (;;) {
    // 把新激活的 handle 批量拿到本地，减少共享队列来回碰撞。
    while (InferHandle* h = rt.infer_lane.activeq.pop()) {
      local.push(h);
    }

    if (local.empty()) {
      co_await rt.infer_lane.work_ready.async_wait();
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

    // 高性能版本不再为每个 handle 建 sender/timer；
    // infer lane 自己 cooperative tick。
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

  rt.search_lanes.resize(numSearchThreads);
  for (SearchLane& lane : rt.search_lanes) {
    lane.root = &root;
    // 真正实现里更适合 start_detached / spawn 一次长期 lane main loop。
    ex::start_detached(ex::starts_on(lane.sched, search_lane_main(rt, lane)));
  }

  ex::start_detached(ex::starts_on(rt.infer_lane.sched, infer_lane_main(rt)));
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
//    - split / ensure_started
//    - 每次 playout 的 async_scope::spawn
//    - 每次等待的泛型 sender op-state
//
// 2. 热路径改成：
//    - per-handle packed CAS claim
//    - per-lane runnable frame 队列
//    - intrusive waiter
//    - infer active-handle pump
//
// 3. 仍保持 v0 关键语义不变：
//    - 搜索线程做 pre_process
//    - 搜索线程做 post_process
//    - batch completion 后才做 estimate / reconcile / target update
//    - handle 只在最后一个消费者离开后复用
//
// 4. 这份文件故意不追求“像教程”。
//    它更像最终实现的骨架草图：读起来更硬，但更贴近会被写进高性能代码里的结构。
