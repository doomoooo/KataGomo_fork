#include "schedlab/search.hpp"
#include "schedlab/config.hpp"

#include "stdexec/execution.hpp"

namespace schedlab {
  namespace ex = stdexec;

  SearchRuntime::SearchRuntime(Dispatcher& dispatcher, Scheduler& scheduler, MockPhaseRunner& mock_runner)
    : dispatcher(dispatcher),
      scheduler(scheduler),
      mock_runner(mock_runner) {
    const std::uint32_t worker_count = schedlab_config().runtime.worker_count;
    workers.reserve(worker_count);
    for(std::uint32_t worker_id = 0; worker_id < worker_count; ++worker_id) {
      auto worker = std::make_unique<WorkerLane>();
      worker->worker_id = worker_id;
      workers.push_back(std::move(worker));
    }
  }

  SearchRuntime::~SearchRuntime() {
    request_stop();
    wait();
  }

  void SearchRuntime::start() {
    for(const auto& worker_ptr: workers) {
      WorkerLane& worker = *worker_ptr;
      spawn_root_playout(worker);
    }
  }

  void SearchRuntime::request_stop() {
    stopping.store(true, std::memory_order_release);
    scheduler.request_stop();
  }

  void SearchRuntime::wait() {
    for(const auto& worker: workers) {
      ex::sync_wait(worker->scope.on_empty());
    }
  }

  auto SearchRuntime::root_playout(WorkerLane& worker) -> exec::task<void> {
    // 每个 root playout 先穿过全局暂停门；一旦通过，后续整条 playout
    // 都固定回到自己的 worker lane 上继续，避免在搜索阶段来回切线程。
    co_await ex::continues_on(scheduler.pause_gate.async_wait(), worker.scheduler());
    if(stopping.load(std::memory_order_acquire)) {
      co_return;
    }
    SearchScheduler& search_scheduler = scheduler.search;
    SearchPlayoutState playout_state = search_scheduler.make_new_state();

    const bool need_nn_eval = mock_runner.playout_descend(worker.worker_id);
    RequestState* ticket = nullptr;
    if(need_nn_eval) {
      playout_state.pause();
      // 向 dispatcher 申请一个 row ticket；拿到后，这个 request 的输入/输出
      // 缓冲区和同步句柄就都固定好了。
      ticket = co_await ex::continues_on(dispatcher.acquire_ticket(), worker.scheduler());
      playout_state.start();
      mock_runner.preprocess(worker.worker_id);
      ticket->notify_input_ready();
      scheduler.infer.on_request_ready();
      scheduler.maybe_close_gate();
      // 一旦输入已经交给 dispatcher，就立刻补上新的 root playout，维持搜索侧并发。
      spawn_root_playout(worker);
      playout_state.pause();
      // output_ready 是跨线程 one-shot event：dispatcher 在 D2H 完成后唤醒这里。
      co_await ex::continues_on(ticket->output_ready.async_wait(), worker.scheduler());
      playout_state.start();
      mock_runner.postprocess(worker.worker_id);
    }
    mock_runner.playout_ascend(worker.worker_id);
    playout_state.finish(need_nn_eval);
    search_scheduler.submit_state(worker.worker_id, playout_state);
    if(need_nn_eval) {
      // 故意把 output_consumed 放到 playout_ascend() 之后而不是 postprocess 之后立即调用。
      // 这样 dispatcher 只有在搜索线程真正消费完这一行输出之后，才会 reset 对应 host slot，避免复用过早发生。
      ticket->notify_output_consumed();
    }
    else {
      // 没有走过 need_nn_eval 分支的话就没有提前 spawn 过 playout。
      spawn_root_playout(worker);
    }
  }

  void SearchRuntime::spawn_root_playout(WorkerLane& worker) {
    worker.scope.spawn(ex::starts_on(worker.scheduler(), root_playout(worker)));
  }
}  // namespace schedlab
