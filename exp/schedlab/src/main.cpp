#include "schedlab/config.hpp"
#include "schedlab/dispatcher.hpp"
#include "schedlab/scheduler.hpp"
#include "schedlab/search.hpp"
#include "schedlab/trt_backend.hpp"
#include "schedlab/utils/mock_phase_runner.hpp"
#include "schedlab/utils/pause_gate.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace schedlab {
  namespace {
    auto parse_u32(const char* value, const char* name) -> std::uint32_t {
      const auto parsed = std::stoull(value);
      if(parsed > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(std::string(name) + " is out of range");
      }
      return static_cast<std::uint32_t>(parsed);
    }

    auto parse_i64(const char* value, const char* name) -> std::int64_t {
      try {
        return std::stoll(value);
      } catch(const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + name);
      }
    }

    auto parse_double(const char* value, const char* name) -> double {
      try {
        return std::stod(value);
      } catch(const std::exception&) {
        throw std::runtime_error(std::string("invalid value for ") + name);
      }
    }

    auto parse_cuda_device_ids(const std::string& text) -> std::vector<int> {
      std::vector<int> out;
      std::size_t begin = 0;
      while(begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string token = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if(!token.empty()) {
          out.push_back(std::stoi(token));
        }
        if(end == std::string::npos) {
          break;
        }
        begin = end + 1;
      }
      return out;
    }

    auto parse_args(int argc, char** argv) -> SchedlabConfig {
      SchedlabConfig config = make_default_schedlab_config();

      for(int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        auto require_value = [&](const char* name) -> const char* {
          if(i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + name);
          }
          return argv[++i];
        };

        if(flag == "--run-ms") {
          config.runtime.run_for_ms = parse_double(require_value("--run-ms"), "--run-ms");
        } else if(flag == "--workers") {
          config.runtime.worker_count = parse_u32(require_value("--workers"), "--workers");
        } else if(flag == "--batch-size") {
          config.infer.batch_size = parse_u32(require_value("--batch-size"), "--batch-size");
        } else if(flag == "--host-slots") {
          config.runtime.host_slot_count = parse_u32(require_value("--host-slots"), "--host-slots");
        } else if(flag == "--lanes-per-device") {
          config.infer.lanes_per_device = parse_u32(require_value("--lanes-per-device"), "--lanes-per-device");
        } else if(flag == "--cuda-devices") {
          config.infer.cuda_device_ids = parse_cuda_device_ids(require_value("--cuda-devices"));
        } else if(flag == "--onnx-model") {
          config.infer.onnx_model_path = require_value("--onnx-model");
        } else if(flag == "--trt-cache-dir") {
          config.infer.trt_cache_dir = require_value("--trt-cache-dir");
        } else if(flag == "--seed") {
          config.search.random_seed = static_cast<std::uint64_t>(parse_i64(require_value("--seed"), "--seed"));
        } else if(flag == "--descend-us") {
          config.search.playout_descend = parse_double(require_value("--descend-us"), "--descend-us");
        } else if(flag == "--preprocess-us") {
          config.search.playout_preprocess = parse_double(require_value("--preprocess-us"), "--preprocess-us");
        } else if(flag == "--postprocess-us") {
          config.search.playout_postprocess = parse_double(require_value("--postprocess-us"), "--postprocess-us");
        } else if(flag == "--ascend-us") {
          config.search.playout_ascend = parse_double(require_value("--ascend-us"), "--ascend-us");
        } else if(flag == "--success-rate") {
          config.search.nn_eval_success_rate = parse_double(require_value("--success-rate"), "--success-rate");
        } else if(flag == "--help") {
          throw std::runtime_error(
            "usage: schedlab_run [--run-ms N] [--workers N] [--batch-size N] [--host-slots N] "
            "[--lanes-per-device N] [--cuda-devices 0,1,2] [--onnx-model PATH] [--trt-cache-dir PATH] "
            "[--seed N] [--descend-us N] [--preprocess-us N] [--postprocess-us N] [--ascend-us N] "
            "[--success-rate X]");
        } else {
          throw std::runtime_error(std::string("unknown argument: ") + std::string(flag));
        }
      }

      return config;
    }
  }  // namespace
}  // namespace schedlab

int main(int argc, char** argv) {
  try {
    schedlab::schedlab_config() = schedlab::parse_args(argc, argv);
    const schedlab::SchedlabConfig& config = schedlab::schedlab_config();

    schedlab::PauseGate gate;
    schedlab::Scheduler scheduler{config.runtime.worker_count, gate};
    schedlab::MockPhaseRunner mock_runner{config.search};
    schedlab::TrtBackend backend;
    schedlab::Dispatcher dispatcher{scheduler, backend};
    schedlab::SearchRuntime search{dispatcher, scheduler, mock_runner};

    search.start();
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(config.runtime.run_for_ms));
    search.request_stop();
    search.wait();
    dispatcher.wait();
    return 0;
  } catch(const std::exception&) {
    return 1;
  }
}
