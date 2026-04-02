#pragma once

#include "stdexec/execution.hpp"

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace schedlab {
  namespace ex = stdexec;

  // 多等待者的 manual-reset gate。
  // 打开时新 waiter 直接通过；gate 析构会强制打开并放行现有/未来 sender。
  // 约定：等待中的 operation 不支持取消。
  class PauseGate final {
    struct WaiterBase {
      explicit WaiterBase(void (*resume)(WaiterBase*) noexcept) noexcept : resume(resume) {}

      WaiterBase* next = nullptr;
      void (*resume)(WaiterBase*) noexcept = nullptr;
      bool linked = false;
    };

    struct State {
      std::mutex mutex;
      std::atomic<bool> is_open{true};
      bool sticky_open = false;
      WaiterBase* waiters = nullptr;
    };

   public:
    ~PauseGate() { open_state(state, true); }

    void set_open(bool should_open) noexcept {
      if(should_open) {
        open();
      } else {
        close();
      }
    }

    void force_open() noexcept { open_state(state, true); }

    auto is_open() const noexcept -> bool {
      return state->is_open.load(std::memory_order_acquire);
    }

    struct Sender {
      using sender_concept = ex::sender_t;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      std::shared_ptr<State> state;

      template<typename Receiver>
      struct Operation : WaiterBase {
        Operation(std::shared_ptr<State> state, Receiver receiver)
          : WaiterBase(&Operation::resume_waiter),
            state(std::move(state)),
            receiver(std::move(receiver)) {}

        ~Operation() {
          if(this->linked) {
            std::terminate();
          }
        }

        Operation(const Operation&) = delete;
        auto operator=(const Operation&) -> Operation& = delete;
        Operation(Operation&&) = delete;
        auto operator=(Operation&&) -> Operation& = delete;

        void start() & noexcept {
          if(started || !state) {
            std::terminate();
          }
          started = true;

          if(state->is_open.load(std::memory_order_acquire)) {
            ex::set_value(std::move(receiver));
            return;
          }

          {
            std::scoped_lock lock(state->mutex);
            if(!state->is_open.load(std::memory_order_relaxed)) {
              this->next = state->waiters;
              state->waiters = this;
              this->linked = true;
              return;
            }
          }

          ex::set_value(std::move(receiver));
        }

        static void resume_waiter(WaiterBase* base) noexcept {
          auto* self = static_cast<Operation*>(base);
          ex::set_value(std::move(self->receiver));
        }

        std::shared_ptr<State> state;
        Receiver receiver;
        bool started = false;
      };

      template<typename Receiver>
      auto connect(this Sender self, Receiver receiver) -> Operation<Receiver> {
        return Operation<Receiver>{std::move(self.state), std::move(receiver)};
      }
    };

    auto async_wait() noexcept -> Sender { return Sender{state}; }

   private:
    void open() noexcept { open_state(state, false); }

    void close() noexcept {
      if(!state->is_open.load(std::memory_order_acquire)) {
        return;
      }

      std::scoped_lock lock(state->mutex);
      if(state->sticky_open || !state->is_open.load(std::memory_order_relaxed)) {
        return;
      }
      state->is_open.store(false, std::memory_order_release);
    }

    static void open_state(const std::shared_ptr<State>& state, bool sticky) noexcept {
      WaiterBase* to_resume = nullptr;
      {
        std::scoped_lock lock(state->mutex);
        state->sticky_open = state->sticky_open || sticky;
        if(state->is_open.load(std::memory_order_relaxed)) {
          return;
        }
        state->is_open.store(true, std::memory_order_release);
        to_resume = std::exchange(state->waiters, nullptr);
      }
      resume_all(to_resume);
    }

    static void resume_all(WaiterBase* head) noexcept {
      while(head != nullptr) {
        WaiterBase* next = head->next;
        head->next = nullptr;
        head->linked = false;
        head->resume(head);
        head = next;
      }
    }

    std::shared_ptr<State> state = std::make_shared<State>();
  };
}  // namespace schedlab
