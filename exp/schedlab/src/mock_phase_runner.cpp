#include "schedlab/utils/mock_phase_runner.hpp"

#include <chrono>
#include <thread>

namespace schedlab {
  namespace {
    void sleep_precisely_for(double duration) {
      if(duration <= 0.0) {
        return;
      }

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double, std::micro>(duration);
      if(duration >= 2000.0) {
        std::this_thread::sleep_until(deadline - std::chrono::duration<double, std::micro>(200.0));
      }
      while(std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
      }
    }
  }  // namespace

  MockPhaseRunner::MockPhaseRunner(MockSearchConfig config)
    : config(config),
      rng(config.random_seed) {}

  void MockPhaseRunner::run_cpu_phase(double duration) {
    sleep_precisely_for(duration);
  }

  auto MockPhaseRunner::playout_descend(std::uint32_t) -> bool {
    run_cpu_phase(config.playout_descend);
    std::scoped_lock lock(rng_mutex);
    std::bernoulli_distribution distribution(config.nn_eval_success_rate);
    return distribution(rng);
  }

  void MockPhaseRunner::preprocess(std::uint32_t) {
    run_cpu_phase(config.playout_preprocess);
  }

  void MockPhaseRunner::postprocess(std::uint32_t) {
    run_cpu_phase(config.playout_postprocess);
  }

  void MockPhaseRunner::playout_ascend(std::uint32_t) {
    run_cpu_phase(config.playout_ascend);
  }
}  // namespace schedlab
