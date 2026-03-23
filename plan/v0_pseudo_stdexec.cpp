// v0_pseudo_stdexec.cpp
//
// 这不是可编译实现。
// 它的目标是把 plan/v0_pseudo.py 的 asyncio 风格伪代码，
// 改写成更贴近 stdexec / P2300 心智模型的 C++ 伪代码。
//
// 本版刻意追求“更纯正的 stdexec 风味”，所以有几个明确取舍：
//
// 1. 长生命周期、强线程亲和、带循环/状态机的控制流：
//    用 exec::task + starts_on(...)。
//
//    例如：
//    - 每个搜索线程上的 root playout 循环
//    - infer 专用线程上的 batch driver
//
// 2. 一次性的异步边界、共享完成信号、回调包装：
//    用 sender 算子。
//
//    例如：
//    - wait CUDA event
//    - wait row-ready
//    - wait root gate
//    - batch 完成后 fan-out 给多个等待者
//
// 3. “暂时去别的执行域做一段事，然后再回来”：
//    优先 on(...) / continues_on(...)。
//
//    本文件里最典型的是：
//    - 搜索协程等待 batch_done 以后，用 continues_on(search_sched)
//      明确回到原搜索线程再做 postprocess。
//
// 4. “已经启动、可共享、可被多个等待者重复连接”的结果：
//    优先 ensure_started(...) + split(...)。
//
//    这正好适合 InferHandle 的 batch completion。
//
// 5. 包装外部异步 API：
//    优先 exec::create，而不是阻塞式 condition_variable / future。
//
// ------------------------------------------------------------
// 这份文件允许比 v0_pseudo.py 更大的结构改动，但必须保持逻辑等价。
//
// 因此这里做了一个重要的“可读性优化”：
// - 不再复用同一个 cur_row 承担两套语义。
// - 改成 claimed_rows / remaining_consumers 两个字段。
//
// 这和 v0.md 是逻辑等价的，只是更像真实 C++ 实现里的 best practice，
// 不容易把 open/inflight 阶段和 completion/GC 阶段的状态混在一起。

#include <stdexec/execution.hpp>

#include <exec/create.hpp>
#include <exec/ensure_started.hpp>
#include <exec/repeat_until.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/split.hpp>
#include <exec/task.hpp>
#include <exec/async_scope.hpp>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
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

// 伪类型：表示“已经启动、可共享、值为 void 的 sender”。
//
// 真正实现时，更推荐：
// - 直接把具体 sender 类型存在 handle 里，避免热路径 type erase；
// - 只有当编译复杂度/头文件耦合太高时，才考虑 any_sender_of。
//
// 语义上它应等价于：
//
//   ex::starts_on(infer_sched, infer_driver(...))
//   | exec::ensure_started()
//   | exec::split()
//
using SharedVoidSender = /* shared sender<void> */ int;

// ============================================================
// sender-first 的等待原语
// ============================================================

// root gate。
//
// 这是一个“通知型 sender 原语”：
// - wait side 返回 sender
// - notify side 直接把所有 waiter 完成
//
// 真正实现时，这类桥接最适合用 exec::create：
// - 如果 gate 已经打开，则在 create 里直接 set_value
// - 否则把 receiver 存进 waiter 链表
// - notify_all() 时依次 set_value
//
// 也就是说，这里虽然看起来像“事件对象”，
// 但它的对外接口是 stdexec 的 sender 风格，不是阻塞式 wait。
//
// 额外注意：
// - notifier 线程不一定是搜索线程
// - 因此 wait side 在恢复业务逻辑前，通常还要显式 continues_on(waiter_scheduler)
struct PauseGate {
  auto async_wait_until_resumed(std::atomic<bool>& paused) -> ex::sender auto;
  void notify_all();
};

// row-ready 状态。
//
// 这里把“位图”和“等待队列”收进同一个抽象里，而不是对外暴露 vector<bool>。
// 这样更符合 stdexec 思路：
// - 生产者调用 mark_ready(row)
// - 消费者拿到一个 sender 来等待某个 row 区间 ready
//
// 真正实现同样优先用 exec::create 包装。
struct RowReadyState {
  void reset(int max_rows);
  void mark_ready(int row);
  void clear();
  auto async_wait_until_ready(int row_low, int row_high) -> ex::sender auto;
};

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
// InferHandle：batch 句柄
// ============================================================

struct InferHandle {
  std::mutex lock;

  // 旧版 v0 用 cur_row 同时承担两种语义：
  // - open / inflight：下一可 claim row
  // - completion / GC：剩余消费者数
  //
  // 本版把它拆开，逻辑完全等价，但更适合真实 C++ 实现。

  // open / inflight 阶段：
  // 已经 claim 的 row 数，同时也是下一可 claim row id。
  int claimed_rows = 0;

  // completion / GC 阶段：
  // 还有多少搜索协程尚未做完 postprocess 并离开。
  int remaining_consumers = 0;

  bool sealed = false;
  bool timeline_reserved = false;

  Slot slot;
  PinnedHostBuffer host_mem;
  RowReadyState row_ready;

  // handle-owning 的 batch completion sender。
  //
  // 这是本版最重要的 stdexec pattern：
  // - 用 ensure_started 让 infer_driver 在“首个 row 被 claim”时立刻启动
  // - 用 split 让多个搜索协程共享同一个完成结果
  //
  // 它是 handle 自己拥有的状态，而不是全局 runtime 帮它托管。
  std::optional<SharedVoidSender> batch_done;
};

// ============================================================
// runtime / 执行域
// ============================================================

struct SearchThreadState {
  exec::single_thread_context ctx;
  exec::async_scope scope;
  decltype(ctx.get_scheduler()) sched = ctx.get_scheduler();

  Node* root = nullptr;
};

struct Runtime {
  SearchSharedState search_shared;

  // infer 专用单线程执行域。
  // infer_driver 是“长期粘在线程上”的控制协程，所以这里用 starts_on + task。
  exec::single_thread_context infer_ctx;
  decltype(infer_ctx.get_scheduler()) infer_sched = infer_ctx.get_scheduler();

  std::vector<SearchThreadState> search_threads;

  std::mutex cur_hid_lock;
  std::size_t cur_hid = 0;
  std::vector<InferHandle> infer_handles;

  std::vector<Gpu> gpus;
  GpuTimeline* gpu_timeline = nullptr;
};

// ============================================================
// 前置声明
// ============================================================

auto infer_driver(Runtime& rt, InferHandle& handle) -> exec::task<void>;

int max_batch_for(Runtime& rt, Slot slot);
bool gpu_is_idle(Gpu& gpu);
Slot get_recent_slot(GpuTimeline& timeline);
std::size_t total_stream_batch_capacity(std::vector<Gpu> const& gpus);

void update_gpu_estimate(Runtime& rt, InferHandle& handle, int batch_rows);
void reconcile_gpu_timeline(Runtime& rt);
void update_search_nn_target_num(Runtime& rt);
void update_search_coro_stats(SearchThreadState& self);

void* row_host_ptr(PinnedHostBuffer& mem, int row);
void pre_process(NnInput& in, void* host_ptr);
void post_process(void* host_ptr, NnOutput& out);

CudaEvent h2d_async(InferHandle& handle, int row_low, int row_high, Gpu& gpu);
CudaEvent infer_async(InferHandle& handle, int batch_rows, Gpu& gpu);
CudaEvent d2h_async(InferHandle& handle, int batch_rows, Gpu& gpu);

// ============================================================
// stdexec 风格 helper
// ============================================================

auto wait_cuda_event(CudaEvent ev) -> ex::sender auto {
  // 真正实现时优先用 exec::create：
  // - 如果 event 已完成，则立即 set_value
  // - 否则注册 CUDA callback / polling bridge
  //
  // 重点是：返回 sender，不阻塞线程。
  //
  // 额外注意：
  // - callback / polling completion 线程不一定是 infer 线程
  // - 所以调用侧在 wait 完以后，通常要显式 continues_on(rt.infer_sched)
  return /* sender<void> */;
}

auto make_batch_done_sender(Runtime& rt, InferHandle& handle) -> SharedVoidSender {
  // 最推荐的表达：
  //
  //   starts_on(infer_sched, infer_driver(...))
  //   | ensure_started()
  //   | split()
  //
  // 为什么不再用 async_scope::spawn_future?
  // - batch completion 是 handle 自己的状态，不是某个全局 scope 的附属产物
  // - ensure_started 更直接表达“现在就启动”
  // - split 更直接表达“多个等待者共享同一个结果”
  //
  return /* ex::starts_on(rt.infer_sched, infer_driver(rt, handle))
            | exec::ensure_started()
            | exec::split() */;
}

void maybe_resume_root_playout(SearchSharedState& s) {
  int current = s.search_nn_current_num.load(std::memory_order_acquire);
  int target  = s.search_nn_target_num.load(std::memory_order_acquire);

  if (current < target) {
    s.search_coro_pause.store(false, std::memory_order_release);
    s.pause_gate.notify_all();
  }
}

void reserve_gpu_timeline_once_locked(Runtime& rt, InferHandle& handle, int fixed_rows) {
  if (handle.timeline_reserved)
    return;

  (void) rt;
  (void) fixed_rows;

  // reserve_gpu_timeline(rt.gpu_timeline, handle.slot, fixed_rows, current_estimate);
  handle.timeline_reserved = true;
}

void reopen_handle_after_last_consumer(InferHandle& handle) {
  handle.claimed_rows = 0;
  handle.remaining_consumers = 0;
  handle.sealed = false;
  handle.timeline_reserved = false;
  handle.row_ready.clear();
  handle.batch_done.reset();
}

// ============================================================
// open batch claim
// ============================================================

struct ClaimedRow {
  InferHandle* handle = nullptr;
  int row = -1;
  void* row_ptr = nullptr;
  SharedVoidSender batch_done = {};
};

ClaimedRow claim_and_preprocess_row(Runtime& rt, NnInput& in) {
  std::unique_lock hid_guard(rt.cur_hid_lock);

  while (true) {
    InferHandle& handle = rt.infer_handles[rt.cur_hid];
    std::unique_lock handle_guard(handle.lock);

    if (handle.sealed) {
      rt.cur_hid = (rt.cur_hid + 1) % rt.infer_handles.size();
      continue;
    }

    const bool first_row = (handle.claimed_rows == 0);

    if (first_row) {
      // 新 batch 的第一次 claim。
      handle.slot = get_recent_slot(*rt.gpu_timeline);
      handle.row_ready.reset(max_batch_for(rt, handle.slot));
      handle.sealed = false;
      handle.timeline_reserved = false;
    }

    const int row = handle.claimed_rows;
    handle.claimed_rows += 1;

    if (handle.claimed_rows == max_batch_for(rt, handle.slot)) {
      reserve_gpu_timeline_once_locked(rt, handle, handle.claimed_rows);
      handle.sealed = true;
    }

    if (first_row) {
      // 注意时序：
      // - batch_done 是 ensure_started + split，因此构造时就会 eager-start infer_driver
      // - 所以必须先让“首行 claim 已经成立”，再创建 batch_done
      //
      // 同时它又必须在释放锁之前发布出去，
      // 否则别的搜索协程可能先 claim 到后续 row，却还看不到 completion sender。
      handle.batch_done.emplace(make_batch_done_sender(rt, handle));
    }

    void* row_ptr = row_host_ptr(handle.host_mem, row);
    SharedVoidSender batch_done = *handle.batch_done;

    handle_guard.unlock();
    hid_guard.unlock();

    // preprocess 明确留在搜索侧执行，不放在 claim 临界区里。
    pre_process(in, row_ptr);
    handle.row_ready.mark_ready(row);

    return {
      .handle = &handle,
      .row = row,
      .row_ptr = row_ptr,
      .batch_done = batch_done,
    };
  }
}

void release_batch_consumer(InferHandle& handle) {
  std::unique_lock lk(handle.lock);
  handle.remaining_consumers -= 1;
  if (handle.remaining_consumers == 0) {
    reopen_handle_after_last_consumer(handle);
  }
}

// ============================================================
// playout_gpu：更纯 sender 风格
// ============================================================

auto playout_gpu(Runtime& rt, SearchThreadState& self, NnInput& in, NnOutput& out)
  -> ex::sender auto {
  // 这里故意不用 task。
  //
  // 因为它本质上是一段“一次性异步管线”：
  // 1. claim row
  // 2. preprocess
  // 3. 等 batch completion
  // 4. 回到搜索线程
  // 5. postprocess
  // 6. GC
  //
  // 这类逻辑用 sender 算子更像 stdexec。
  return ex::just()
       | ex::then([&]() -> ClaimedRow {
           return claim_and_preprocess_row(rt, in);
         })
       | ex::let_value([&](ClaimedRow claim) {
           return claim.batch_done
                // split 自身不 advertise completion scheduler。
                // 而 batch_done 很可能在 infer 线程上完成。
                //
                // 所以这里要显式 continues_on(search_sched)，
                // 保证 postprocess 明确回到原搜索线程。
                | ex::continues_on(self.sched)
                | ex::then([&, claim] {
                    post_process(claim.row_ptr, out);
                    release_batch_consumer(*claim.handle);
                  });
         });
}

// ============================================================
// 显式 playout 状态机
// ============================================================

struct PlayoutMachine {
  explicit PlayoutMachine(Node& root);

  struct Boundary {
    bool need_nn = false;
    NnInput* nn_input = nullptr;
    NnOutput* nn_output = nullptr;
  };

  auto run_cpu_until_boundary() -> Boundary;
  void finish_after_nn();
  bool finished() const;
  bool used_nn() const;
};

// ============================================================
// 搜索侧主流程
// ============================================================

auto playout_root(Runtime& rt, SearchThreadState& self) -> exec::task<void>;

void spawn_next_root_playout(Runtime& rt, SearchThreadState& self) {
  // async_scope::spawn 内部已经会对 child op 做结构化托管。
  // 这里不再额外包一层手写 helper。
  self.scope.spawn(ex::starts_on(self.sched, playout_root(rt, self)));
}

auto playout_root(Runtime& rt, SearchThreadState& self) -> exec::task<void> {
  if (rt.search_shared.search_coro_pause.load(std::memory_order_acquire)) {
    co_await (rt.search_shared.pause_gate.async_wait_until_resumed(
                rt.search_shared.search_coro_pause)
              | ex::continues_on(self.sched));
  }

  PlayoutMachine machine(*self.root);
  bool launched_successor = false;

  auto boundary = machine.run_cpu_until_boundary();

  if (boundary.need_nn) {
    const int current =
      rt.search_shared.search_nn_current_num.fetch_add(1, std::memory_order_acq_rel) + 1;
    const int target =
      rt.search_shared.search_nn_target_num.load(std::memory_order_acquire);

    if (current >= target) {
      rt.search_shared.search_coro_pause.store(true, std::memory_order_release);
    }

    // 命中 NN 叶子时，当前 playout 即将挂起等待 GPU。
    // 所以先补发一个新的 root playout，维持“每个搜索线程恰好有一个下一次 playout 在排队”。
    spawn_next_root_playout(rt, self);
    launched_successor = true;

    // 这里 co_await 的对象是 sender，而不是 task。
    co_await playout_gpu(rt, self, *boundary.nn_input, *boundary.nn_output);
    machine.finish_after_nn();
  }

  update_search_coro_stats(self);

  if (!launched_successor) {
    spawn_next_root_playout(rt, self);
  }
}

// ============================================================
// infer 侧 batch driver
// ============================================================

auto pump_h2d_until_sealed(Runtime& rt,
                           InferHandle& handle,
                           Gpu& gpu,
                           int& row_low,
                           CudaEvent& last_h2d_event) -> ex::sender auto {
  // 这里故意把“spin until sealed”写成 repeat_until 风格，而不是手写 while(true)+sleep(0)。
  //
  // 语义上仍然是 v0 的 spinning driver，
  // 但表达更接近 stdexec：
  // - 每轮迭代都是一个 sender
  // - repeat_until 负责重试
  // - schedule(infer_sched) 负责 cooperative yield
  return ex::schedule(rt.infer_sched)
       | ex::let_value([&]() -> ex::sender auto {
           int row_snapshot = 0;
           bool sealed_snapshot = false;

           {
             std::unique_lock lk(handle.lock);
             row_snapshot = handle.claimed_rows;

             if (gpu_is_idle(gpu) && !handle.sealed) {
               reserve_gpu_timeline_once_locked(rt, handle, row_snapshot);
               handle.sealed = true;
             }

             sealed_snapshot = handle.sealed;
           }

           if (row_snapshot > row_low) {
             return handle.row_ready.async_wait_until_ready(row_low, row_snapshot)
                  // row-ready 可能由搜索线程完成，因此要显式切回 infer scheduler 再发 H2D。
                  | ex::continues_on(rt.infer_sched)
                  | ex::then([&, row_snapshot, sealed_snapshot] {
                      last_h2d_event = h2d_async(handle, row_low, row_snapshot, gpu);
                      row_low = row_snapshot;
                      return sealed_snapshot;
                    });
           }

           return ex::just(sealed_snapshot);
         })
       | exec::repeat_until();
}

auto infer_driver(Runtime& rt, InferHandle& handle) -> exec::task<void> {
  // 这是长期粘在 infer 线程上的控制协程。
  // 因此这里继续用 task，而不是硬把整个 driver 挤成一大串 sender algebra。
  //
  // 原则上：
  // - 长流程、带循环、强线程亲和：task
  // - 一次性边界：sender
  Gpu& gpu = rt.gpus[handle.slot.gpu_id];

  int row_low = 0;
  CudaEvent last_h2d_event{};

  co_await pump_h2d_until_sealed(rt, handle, gpu, row_low, last_h2d_event);

  const int batch_rows = row_low;

  if (batch_rows > 0) {
    co_await (wait_cuda_event(last_h2d_event) | ex::continues_on(rt.infer_sched));
  }

  {
    CudaEvent infer_done = infer_async(handle, batch_rows, gpu);
    co_await (wait_cuda_event(infer_done) | ex::continues_on(rt.infer_sched));
  }

  {
    CudaEvent d2h_done = d2h_async(handle, batch_rows, gpu);
    co_await (wait_cuda_event(d2h_done) | ex::continues_on(rt.infer_sched));
  }

  // 这里明确采用 v0.md 的 completion-point 版本：
  // - 等 D2H 真正完成
  // - 再 update_gpu_estimate()
  // - 再 reconcile_gpu_timeline()
  // - 最后 update_search_nn_target_num()
  //
  // 原 v0_pseudo.py 在 d2h helper 里把 update 写在 await event_complete() 之前，
  // 那里与 v0.md 不一致。本文件在这个冲突点上以 v0.md 为准。
  update_gpu_estimate(rt, handle, batch_rows);
  reconcile_gpu_timeline(rt);
  update_search_nn_target_num(rt);
  maybe_resume_root_playout(rt.search_shared);

  // completion 进入 GC 阶段前，把“还剩多少消费者”固定下来。
  // 这和旧版 cur_row 切语义是完全等价的，只是更显式。
  {
    std::unique_lock lk(handle.lock);
    handle.remaining_consumers = handle.claimed_rows;
  }

  // 不再显式 notify_all()。
  // infer_driver 自己完成这一刻，就是 batch_done sender 的 ready 时刻。
  co_return;
}

// ============================================================
// 启动
// ============================================================

void bootstrap(Runtime& rt, Node& root, int numSearchThreads) {
  // infer_handles 容量仍沿用 v0 结论：
  //   3 * sum(gpu.cuda_streams * gpu.max_batch)
  const std::size_t ring_capacity =
    /* 3 * sum(gpu.cuda_streams * gpu.max_batch) */
    3 * total_stream_batch_capacity(rt.gpus);
  rt.infer_handles.resize(ring_capacity);

  for (InferHandle& h : rt.infer_handles) {
    // h.host_mem.init_pinned_memory(...)
  }

  rt.search_threads.resize(numSearchThreads);
  for (SearchThreadState& st : rt.search_threads) {
    st.root = &root;
    st.scope.spawn(ex::starts_on(st.sched, playout_root(rt, st)));
  }
}

int main() {
  Runtime rt;
  Node root;

  bootstrap(rt, root, /* numSearchThreads = */ 0);

  // 真正实现里的 shutdown 不在本版展开。
  //
  // 这里只补一条 stdexec best practice：
  // - 若将来需要“同时启动若干 nofail 的 join sender，再统一 async_wait()”，
  //   优先考虑 exec::start_now。
  // - 但本文件的两个核心场景并不强行使用它：
  //   1. batch completion 更适合 handle-owning 的 ensure_started + split
  //   2. root playout 是自续接的长期控制流，更适合 async_scope::spawn
  return 0;
}

// ============================================================
// 暂不展开
// ============================================================
//
// 1. root 切换
// 2. 手动暂停 / 恢复搜索
// 3. 多 Search / evaluator 共存
// 4. 具体原子类型、内存序、锁粒度
// 5. exec::create 背后的 waiter 链表与 callback bridge 细节
