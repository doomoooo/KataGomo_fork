#pragma once

#include "schedlab/utils/mock_phase_runner.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace schedlab {
  // schedlab 全局运行配置。
  // 当前先做成一个进程内单例，统一承载：
  // 1. main 解析出来的运行参数
  // 2. dispatcher / scheduler / trt backend 的默认超参
  struct SchedlabConfig {
    struct RuntimeConfig {
      double run_for_ms = 1200.0;
      std::uint32_t worker_count = 8;
      std::uint32_t host_slot_count = 16;
    };

    struct InferConfig {
      std::uint32_t batch_size = 8;
      std::uint32_t lanes_per_device = 2;
      std::vector<int> cuda_device_ids;
      std::filesystem::path onnx_model_path;
      std::filesystem::path trt_cache_dir;
    };

    struct SchedulerConfig {
      double search_ewma_alpha = 0.15;
      double search_initial_requests_per_us = 0.5 / 100.0;
      double infer_ewma_alpha = 0.2;
      double infer_initial_batch_us = 1000.0;
      double infer_prediction_jitter_weight = 0.5;
      double infer_prediction_last_error_weight = 0.25;
      double max_starvation_probability = 1e-5;
    };

    struct TrtConfig {
      std::size_t tensor_alignment = 256;
    };

    RuntimeConfig runtime{};
    InferConfig infer{};
    MockSearchConfig search{};
    SchedulerConfig scheduler{};
    TrtConfig trt{};
  };

  // 返回全局 schedlab 配置单例。
  auto schedlab_config() noexcept -> SchedlabConfig&;
  // 生成一份带环境相关默认值的配置快照。
  auto make_default_schedlab_config() -> SchedlabConfig;
}  // namespace schedlab
