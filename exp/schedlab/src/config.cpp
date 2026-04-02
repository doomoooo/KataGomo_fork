#include "schedlab/config.hpp"

#include <cstdlib>
#include <filesystem>

namespace schedlab {
  namespace {
    auto default_onnx_model_path() -> std::filesystem::path {
      if(const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".katago" / "weights" / "b18tf.onnx";
      }
      return std::filesystem::path("~/.katago/weights/b18tf.onnx");
    }

    auto default_trt_cache_dir() -> std::filesystem::path {
      if(const char* env = std::getenv("SCHEDLAB_TRT_CACHE_DIR")) {
        return std::filesystem::path(env);
      }
      if(const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) / ".katago" / "schedlab" / "trtcache";
      }
      return std::filesystem::current_path() / ".schedlab_trtcache";
    }
  }  // namespace

  auto make_default_schedlab_config() -> SchedlabConfig {
    SchedlabConfig config;
    config.infer.onnx_model_path = default_onnx_model_path();
    config.infer.trt_cache_dir = default_trt_cache_dir();
    config.search.playout_descend = 350.0;
    config.search.playout_preprocess = 30.0;
    config.search.playout_postprocess = 30.0;
    config.search.playout_ascend = 320.0;
    config.search.nn_eval_success_rate = 0.5;
    config.search.random_seed = 1;
    return config;
  }

  auto schedlab_config() noexcept -> SchedlabConfig& {
    static SchedlabConfig config = make_default_schedlab_config();
    return config;
  }
}  // namespace schedlab
