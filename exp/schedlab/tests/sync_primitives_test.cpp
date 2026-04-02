#include "exec/start_detached.hpp"
#include "schedlab/utils/one_shot_event.hpp"
#include "schedlab/utils/pause_gate.hpp"
#include "stdexec/execution.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace {
  namespace ex = stdexec;
  using schedlab::OneShotEvent;
  using schedlab::PauseGate;

  constexpr int k_terminate_exit_code = 99;

  [[noreturn]] void terminate_to_exit() noexcept {
    std::_Exit(k_terminate_exit_code);
  }

  struct NoopReceiver {
    using receiver_concept = ex::receiver_t;
    void set_value() noexcept {}
    void set_error(std::exception_ptr) noexcept { std::terminate(); }
    auto get_env() const noexcept -> ex::__root_env { return {}; }
  };

  struct CountingReceiver {
    using receiver_concept = ex::receiver_t;

    std::atomic<int>* values = nullptr;

    void set_value() noexcept {
      if(values) {
        values->fetch_add(1, std::memory_order_relaxed);
      }
    }

    void set_error(std::exception_ptr) noexcept { std::terminate(); }
    auto get_env() const noexcept -> ex::__root_env { return {}; }
  };

  using OneShotSender = decltype(std::declval<OneShotEvent&>().async_wait());
  using OneShotOperation = decltype(ex::connect(std::declval<OneShotSender>(), NoopReceiver{}));
  static_assert(!std::is_copy_constructible_v<OneShotOperation>);
  static_assert(!std::is_move_constructible_v<OneShotOperation>);

  template<typename Sender>
  bool wait_for_value(Sender&& sender) {
    return ex::sync_wait(std::forward<Sender>(sender)).has_value();
  }

  template<typename Fn>
  bool expect_terminate(Fn&& fn) {
    const pid_t pid = ::fork();
    if(pid < 0) {
      std::perror("fork");
      return false;
    }
    if(pid == 0) {
      std::set_terminate(terminate_to_exit);
      fn();
      std::_Exit(0);
    }

    int status = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while(std::chrono::steady_clock::now() < deadline) {
      const pid_t result = ::waitpid(pid, &status, WNOHANG);
      if(result == pid) {
        return WIFEXITED(status) && WEXITSTATUS(status) == k_terminate_exit_code;
      }
      if(result < 0) {
        std::perror("waitpid");
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }

    (void)::kill(pid, SIGKILL);
    (void)::waitpid(pid, &status, 0);
    return false;
  }

  template<typename Pred>
  bool spin_until(Pred&& pred) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    while(std::chrono::steady_clock::now() < deadline) {
      if(pred()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return pred();
  }

  bool test_one_shot_set_before_wait() {
    OneShotEvent event;
    event.set();
    return wait_for_value(event.async_wait());
  }

  bool test_one_shot_reset_reuse() {
    OneShotEvent event;
    {
      std::jthread signaler([&event] {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        event.set();
      });
      if(!wait_for_value(event.async_wait())) {
        return false;
      }
    }

    event.reset();
    event.set();
    return wait_for_value(event.async_wait());
  }

  bool test_one_shot_duplicate_waiter_terminates() {
    return expect_terminate([] {
      OneShotEvent event;
      auto op = ex::connect(event.async_wait(), NoopReceiver{});
      ex::start(op);
      (void)wait_for_value(event.async_wait());
    });
  }

  bool test_one_shot_reset_with_waiter_terminates() {
    return expect_terminate([] {
      OneShotEvent event;
      auto op = ex::connect(event.async_wait(), NoopReceiver{});
      ex::start(op);
      event.reset();
    });
  }

  bool test_one_shot_old_sender_after_reset_terminates() {
    return expect_terminate([] {
      OneShotEvent event;
      auto sender = event.async_wait();
      event.reset();
      (void)wait_for_value(std::move(sender));
    });
  }

  bool test_pause_gate_initially_open() {
    PauseGate gate;
    return gate.is_open() && wait_for_value(gate.async_wait());
  }

  bool test_pause_gate_sender_lifetime() {
    auto sender = [] {
      PauseGate gate;
      gate.set_open(false);
      return gate.async_wait();
    }();
    return wait_for_value(std::move(sender));
  }

  bool test_pause_gate_broadcast() {
    PauseGate gate;
    gate.set_open(false);
    std::atomic<int> resumed{0};

    exec::start_detached(gate.async_wait() | ex::then([&] noexcept {
                           resumed.fetch_add(1, std::memory_order_relaxed);
                         }));
    exec::start_detached(gate.async_wait() | ex::then([&] noexcept {
                           resumed.fetch_add(1, std::memory_order_relaxed);
                         }));

    gate.set_open(true);
    return spin_until([&] { return resumed.load(std::memory_order_relaxed) == 2; });
  }

  bool test_pause_gate_force_open_sticky() {
    PauseGate gate;
    gate.set_open(false);
    gate.force_open();
    gate.set_open(false);
    return gate.is_open() && wait_for_value(gate.async_wait());
  }

  bool test_pause_gate_destroyed_owner_releases_waiter() {
    std::atomic<int> values{0};
    auto gate = std::make_unique<PauseGate>();
    gate->set_open(false);
    auto op = ex::connect(gate->async_wait(), CountingReceiver{&values});
    ex::start(op);
    gate.reset();
    return values.load(std::memory_order_relaxed) == 1;
  }

  bool test_pause_gate_cancelling_waiter_terminates() {
    return expect_terminate([] {
      PauseGate gate;
      gate.set_open(false);
      auto op = ex::connect(gate.async_wait(), NoopReceiver{});
      ex::start(op);
    });
  }
}  // namespace

int main(int argc, char** argv) {
  if(argc != 2) {
    std::cerr << "usage: sync_primitives_test <mode>\n";
    return 1;
  }

  const std::string_view mode = argv[1];
  if(mode == "one_shot_set_before_wait") {
    return test_one_shot_set_before_wait() ? 0 : 1;
  }
  if(mode == "one_shot_reset_reuse") {
    return test_one_shot_reset_reuse() ? 0 : 1;
  }
  if(mode == "one_shot_duplicate_waiter_terminates") {
    return test_one_shot_duplicate_waiter_terminates() ? 0 : 1;
  }
  if(mode == "one_shot_reset_with_waiter_terminates") {
    return test_one_shot_reset_with_waiter_terminates() ? 0 : 1;
  }
  if(mode == "one_shot_old_sender_after_reset_terminates") {
    return test_one_shot_old_sender_after_reset_terminates() ? 0 : 1;
  }
  if(mode == "pause_gate_initially_open") {
    return test_pause_gate_initially_open() ? 0 : 1;
  }
  if(mode == "pause_gate_sender_lifetime") {
    return test_pause_gate_sender_lifetime() ? 0 : 1;
  }
  if(mode == "pause_gate_broadcast") {
    return test_pause_gate_broadcast() ? 0 : 1;
  }
  if(mode == "pause_gate_force_open_sticky") {
    return test_pause_gate_force_open_sticky() ? 0 : 1;
  }
  if(mode == "pause_gate_destroyed_owner_releases_waiter") {
    return test_pause_gate_destroyed_owner_releases_waiter() ? 0 : 1;
  }
  if(mode == "pause_gate_cancelling_waiter_terminates") {
    return test_pause_gate_cancelling_waiter_terminates() ? 0 : 1;
  }

  std::cerr << "unknown mode: " << mode << '\n';
  return 1;
}
