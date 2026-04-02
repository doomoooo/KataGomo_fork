#include "schedlab/dispatcher.hpp"
#include "schedlab/config.hpp"

#include "stdexec/execution.hpp"

#include <cstdint>

namespace schedlab {
  namespace ex = stdexec;

  void RequestState::init_owner_slot(HostSlotState& owner_slot) noexcept {
    this->owner_slot = &owner_slot;
  }

  void RequestState::notify_input_ready() const noexcept {
    owner_slot->input_ready_count.fetch_add(1, std::memory_order_release);
  }

  void RequestState::notify_output_consumed() const noexcept {
    owner_slot->output_consumed_count.fetch_add(1, std::memory_order_release);
  }

  Dispatcher::Dispatcher(
    Scheduler& scheduler,
    InferBackend& backend)
    : scheduler(scheduler),
      backend(backend),
      max_batch_size(schedlab_config().infer.batch_size),
      host_slot_count(schedlab_config().runtime.host_slot_count) {
    init();
  }

  Dispatcher::~Dispatcher() {
    wait();
  }

  void Dispatcher::initialize_request_views(HostSlotState& host_slot) const noexcept {
    const BatchLayout& batch_layout = backend.batch_layout;
    for(std::uint32_t row_index = 0; row_index < max_batch_size; ++row_index) {
      RequestState& state = host_slot.request_states[row_index];
      state.inputs_mem_addr.resize(batch_layout.input_row_bytes.size());
      state.outputs_mem_addr.resize(batch_layout.output_row_bytes.size());

      for(std::size_t tensor_index = 0; tensor_index < batch_layout.input_row_bytes.size(); ++tensor_index) {
        state.inputs_mem_addr[tensor_index] =
          static_cast<char*>(host_slot.host_slot->inputs[tensor_index]) +
          static_cast<std::size_t>(row_index) * batch_layout.input_row_bytes[tensor_index];
      }

      for(std::size_t tensor_index = 0; tensor_index < batch_layout.output_row_bytes.size(); ++tensor_index) {
        state.outputs_mem_addr[tensor_index] =
          static_cast<char*>(host_slot.host_slot->outputs[tensor_index]) +
          static_cast<std::size_t>(row_index) * batch_layout.output_row_bytes[tensor_index];
      }
    }
  }

  void Dispatcher::init() {
    backend.allocate_host_slots(host_slot_count);
    host_slots = std::make_unique<HostSlotState[]>(host_slot_count);
    for(std::uint32_t slot_index = 0; slot_index < host_slot_count; ++slot_index) {
      HostSlotState& slot = host_slots[slot_index];
      slot.request_states = std::make_unique<RequestState[]>(max_batch_size);
      for(std::uint32_t row_index = 0; row_index < max_batch_size; ++row_index) {
        slot.request_states[row_index].init_owner_slot(slot);
      }
      slot.host_slot = &backend.host_slots[slot_index];
      initialize_request_views(slot);
    }
    scheduler.infer.initialize_infer_lanes();
  }

  auto Dispatcher::acquire_ticket() -> exec::task<RequestState*> {
    // ticket 分配的关键状态都只在 dispatcher thread 上读写，避免额外加锁。
    InferScheduler& infer_scheduler = scheduler.infer;
    co_await ex::schedule(dispatcher_context.get_scheduler());
    // 确保目前的 host_slot 可用。
    co_await wait_pred([this]() noexcept {
      return !host_slots[current_host_slot].sealed;
    });

    HostSlotState& slot = host_slots[current_host_slot];
    const std::uint32_t row_index = slot.assigned_rows;
    RequestState& state = slot.request_states[row_index];
    slot.assigned_rows = row_index + 1;

    if(slot.assigned_rows == max_batch_size) {
      launch_current_host_slot(std::nullopt);
    } else if(const auto group_id = infer_scheduler.get_idle_group()) {
      launch_current_host_slot(InferLane{*group_id, 0});
    }

    co_return &state;
  }

  void Dispatcher::launch_current_host_slot(std::optional<InferLane> lane) noexcept {
    HostSlotState& slot = host_slots[current_host_slot];
    // 当前 open host_slot 一旦值得发车，就 seal 它并异步起一个 infer 流程；
    // 后续新的请求改去下一个 host_slot。
    slot.sealed = true;
    current_host_slot++;
    if(current_host_slot == host_slot_count) {
      current_host_slot = 0;
    }
    scope.spawn(ex::starts_on(dispatcher_context.get_scheduler(), infer_coro(&slot, lane)));
  }

  auto Dispatcher::infer_coro(HostSlotState* host_slot, std::optional<InferLane> lane) -> exec::task<void> {
    InferScheduler& infer_scheduler = scheduler.infer;
    const std::uint32_t assigned_rows = host_slot->assigned_rows;
    // 等待预处理环节完成，host slot 上的输入就绪
    co_await wait_pred(
      [host_slot, assigned_rows]() noexcept {
        return host_slot->input_ready_count.load(std::memory_order_acquire) == assigned_rows;
      });

    const InferLane selected_lane = lane ? *lane : infer_scheduler.select_lane(assigned_rows);
    // infer_state 是 backend 对“这一次完整 infer 生命周期”的 opaque 封装。
    // dispatcher 只按固定顺序驱动它，不感知内部是否用了 bank、event 或别的机制。
    auto infer_state = backend.make_infer_state(*host_slot->host_slot, assigned_rows, selected_lane);

    // dispatcher 的等待语义本质上都是单线程轮询；co_await 只是把顺序组织写清楚。
    // 如果 backend 需要先做 bank handoff 等前序依赖，也一并收进 submit_h2d() 的谓词里。
    co_await wait_pred(infer_state->submit_h2d());
    infer_scheduler.on_infer_submit(selected_lane, assigned_rows);
    auto infer_done = infer_state->submit_infer();
    co_await wait_pred(std::move(infer_done));
    infer_scheduler.on_infer_done(selected_lane);
    // 某条 lane 的 infer 一结束，只需要看它所在的 group 是否刚好已经空闲。
    // 如果是，并且当前 open host_slot 已经攒到至少 1 行，就立即把它发车；
    // 这样不用等下一次 acquire_ticket() 才顺手触发 seal。
    if(host_slots[current_host_slot].assigned_rows > 0 && infer_scheduler.is_group_idle(selected_lane.group_id)) {
      launch_current_host_slot(selected_lane);
    }

    co_await wait_pred(infer_state->submit_d2h());

    // D2H 完成后逐行放行搜索线程；output_ready 仍然保留为跨线程 one-shot event。
    for(std::uint32_t row_index = 0; row_index < assigned_rows; ++row_index) {
      host_slot->request_states[row_index].output_ready.set();
    }

    // 等待所有后处理收尾
    co_await wait_pred(
      [host_slot, assigned_rows]() noexcept {
        return host_slot->output_consumed_count.load(std::memory_order_acquire) == assigned_rows;
      });
    reset_host_slot(*host_slot, assigned_rows);

    // 在 infer 下降沿决策是否需要放行 pause gate
    scheduler.infer.on_request_done();
    scheduler.maybe_open_gate();
  }

  void Dispatcher::reset_host_slot(HostSlotState& host_slot, std::uint32_t assigned_rows) noexcept {
    for(std::uint32_t row_index = 0; row_index < assigned_rows; ++row_index) {
      host_slot.request_states[row_index].output_ready.reset();
    }
    host_slot.assigned_rows = 0;
    host_slot.input_ready_count.store(0, std::memory_order_relaxed);
    host_slot.output_consumed_count.store(0, std::memory_order_relaxed);
    host_slot.sealed = false;
  }

  void Dispatcher::wait() {
    ex::sync_wait(scope.on_empty());
  }
}  // namespace schedlab
