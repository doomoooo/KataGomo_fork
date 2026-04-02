#pragma once

#include "stdexec/execution.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <utility>

namespace schedlab {
  namespace ex = stdexec;

  // 单等待者、单代次的一次性事件。
  // 约定：sender/operation 不得越过 event 本体生命周期；等待中的 operation 不支持取消。
  struct OneShotEvent {
    struct Waiter {
      Waiter(std::uint64_t generation, void (*resume)(Waiter*) noexcept) noexcept
        : generation(generation),
          resume(resume) {}

      std::uint64_t generation = 0;
      void (*resume)(Waiter*) noexcept = nullptr;
      bool linked = false;
    };

    ~OneShotEvent() {
      std::scoped_lock lock(mutex);
      if(waiter != nullptr) {
        std::terminate();
      }
    }

    void set() noexcept {
      if(signaled.exchange(true, std::memory_order_acq_rel)) {
        return;
      }

      Waiter* to_resume = nullptr;
      {
        std::scoped_lock lock(mutex);
        to_resume = std::exchange(waiter, nullptr);
      }
      if(to_resume != nullptr) {
        to_resume->linked = false;
        to_resume->resume(to_resume);
      }
    }

    void reset() noexcept {
      std::scoped_lock lock(mutex);
      if(waiter != nullptr) {
        std::terminate();
      }
      signaled.store(false, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_release);
    }

    auto is_set() const noexcept -> bool {
      return signaled.load(std::memory_order_acquire);
    }

    struct Sender {
      using sender_concept = ex::sender_t;
      using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

      OneShotEvent* event = nullptr;
      std::uint64_t generation = 0;

      template<typename Receiver>
      struct Operation : Waiter {
        Operation(OneShotEvent* event, std::uint64_t generation, Receiver receiver)
          : Waiter(generation, &Operation::resume_waiter),
            event(event),
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
          if(started || event == nullptr) {
            std::terminate();
          }
          started = true;

          if(this->generation != event->generation.load(std::memory_order_acquire)) {
            std::terminate();
          }
          if(event->signaled.load(std::memory_order_acquire)) {
            ex::set_value(std::move(receiver));
            return;
          }

          {
            std::scoped_lock lock(event->mutex);
            if(this->generation != event->generation.load(std::memory_order_relaxed)) {
              std::terminate();
            }
            if(event->signaled.load(std::memory_order_relaxed)) {
            } else if(event->waiter != nullptr) {
              std::terminate();
            } else {
              event->waiter = this;
              this->linked = true;
              return;
            }
          }

          ex::set_value(std::move(receiver));
        }

        static void resume_waiter(Waiter* base) noexcept {
          auto* self = static_cast<Operation*>(base);
          ex::set_value(std::move(self->receiver));
        }

        OneShotEvent* event = nullptr;
        Receiver receiver;
        bool started = false;
      };

      template<typename Receiver>
      auto connect(this Sender self, Receiver receiver) -> Operation<Receiver> {
        return Operation<Receiver>{self.event, self.generation, std::move(receiver)};
      }
    };

    auto async_wait() noexcept -> Sender {
      return Sender{this, generation.load(std::memory_order_acquire)};
    }

    mutable std::mutex mutex;
    std::atomic<std::uint64_t> generation{0};
    std::atomic<bool> signaled{false};
    Waiter* waiter = nullptr;
  };
}  // namespace schedlab
