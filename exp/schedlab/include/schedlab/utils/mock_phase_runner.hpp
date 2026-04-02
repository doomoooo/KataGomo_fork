#pragma once

#include <cstdint>
#include <mutex>
#include <random>

namespace schedlab {
  struct MockSearchConfig {
    double playout_descend = 50.0;
    double playout_preprocess = 3.0;
    double playout_postprocess = 3.0;
    double playout_ascend = 50.0;
    double nn_eval_success_rate = 0.5;
    std::uint64_t random_seed = 1;
  };

  // CPU phase 模拟器。
  // 负责按固定配置执行搜索侧 mock phase。
  class MockPhaseRunner {
   public:
    explicit MockPhaseRunner(MockSearchConfig config);

    auto playout_descend(std::uint32_t worker_id) -> bool;
    void preprocess(std::uint32_t worker_id);
   void postprocess(std::uint32_t worker_id);
   void playout_ascend(std::uint32_t worker_id);

   private:
    void run_cpu_phase(double duration);

    MockSearchConfig config{};
    std::mutex rng_mutex;
    std::mt19937_64 rng;
  };
}  // namespace schedlab
