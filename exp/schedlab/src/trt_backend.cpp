#include "schedlab/config.hpp"
#include "schedlab/trt_backend.hpp"

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace schedlab {
  namespace {
    using namespace nvinfer1;

    constexpr char k_network_revision[] = "schedlab_b18tf_v1";

    auto make_error(const std::string& where, const std::string& message) -> std::runtime_error {
      return std::runtime_error(where + ": " + message);
    }

    void check_cuda(cudaError_t status, const char* where) {
      if(status != cudaSuccess) {
        throw make_error(where, cudaGetErrorString(status));
      }
    }

    auto align_up(std::size_t value, std::size_t alignment) -> std::size_t {
      return ((value + alignment - 1) / alignment) * alignment;
    }

    auto sanitize_name(std::string value) -> std::string {
      for(char& ch: value) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if(!std::isalnum(uch)) {
          ch = '_';
        }
      }
      return value;
    }

    auto data_type_bytes(DataType type) -> std::size_t {
      switch(type) {
        case DataType::kFLOAT:
        case DataType::kINT32:
          return 4;
        case DataType::kHALF:
          return 2;
        case DataType::kINT8:
        case DataType::kBOOL:
          return 1;
#if NV_TENSORRT_MAJOR >= 10
        case DataType::kUINT8:
          return 1;
        case DataType::kINT64:
          return 8;
        case DataType::kBF16:
          return 2;
#endif
        default:
          throw make_error("data_type_bytes", "unsupported tensor data type");
      }
    }

    auto dims_volume_without_batch(const Dims& dims) -> std::size_t {
      std::size_t volume = 1;
      for(int i = 1; i < dims.nbDims; ++i) {
        volume *= static_cast<std::size_t>(dims.d[i]);
      }
      return volume;
    }

    auto file_signature(const std::filesystem::path& path) -> std::string {
      const auto size = std::filesystem::file_size(path);
      const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(
                           std::filesystem::last_write_time(path).time_since_epoch())
                           .count();
      std::ostringstream out;
      out << sanitize_name(path.filename().string()) << "_sz" << size << "_ts" << stamp;
      return out.str();
    }

    void ensure_directory(const std::filesystem::path& dir) {
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      if(ec) {
        throw make_error("create_directories", ec.message());
      }
    }

    auto read_binary_file(const std::filesystem::path& path) -> std::vector<char> {
      std::ifstream input(path, std::ios::binary);
      if(!input) {
        throw make_error("read_binary_file", path.string() + " open failed");
      }
      input.seekg(0, std::ios::end);
      const auto size = input.tellg();
      input.seekg(0, std::ios::beg);
      std::vector<char> bytes(static_cast<std::size_t>(size));
      input.read(bytes.data(), static_cast<std::streamsize>(size));
      if(!input) {
        throw make_error("read_binary_file", path.string() + " read failed");
      }
      return bytes;
    }

    void write_binary_file(const std::filesystem::path& path, const std::vector<char>& bytes) {
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      if(!output) {
        throw make_error("write_binary_file", path.string() + " open failed");
      }
      output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if(!output) {
        throw make_error("write_binary_file", path.string() + " write failed");
      }
    }

    struct TrtLogger final : ILogger {
      void log(Severity, AsciiChar const*) noexcept override {
      }
    };

    auto make_batch_dims(const Dims& base, int batch_size) -> Dims {
      Dims out = base;
      out.d[0] = batch_size;
      return out;
    }

    auto parser_error_message(nvonnxparser::IParser& parser) -> std::string {
      std::ostringstream out;
      for(int i = 0; i < parser.getNbErrors(); ++i) {
        if(i > 0) {
          out << "; ";
        }
        out << parser.getError(i)->desc();
      }
      return out.str();
    }

    auto build_serialized_plan(
      const std::filesystem::path& onnx_model_path,
      std::uint32_t max_batch_size,
      bool use_fp16) -> std::vector<char> {
      TrtLogger logger;
      auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
      auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0U));
      auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
      if(!parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(ILogger::Severity::kERROR))) {
        throw make_error("parseFromFile", parser_error_message(*parser));
      }

      auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
      auto* profile = builder->createOptimizationProfile();
      const int max_batch = static_cast<int>(max_batch_size);
      for(int i = 0; i < network->getNbInputs(); ++i) {
        ITensor* input = network->getInput(i);
        const char* name = input->getName();
        const Dims dims = input->getDimensions();
        const int opt_batch = std::min(max_batch, 8);
        profile->setDimensions(name, OptProfileSelector::kMIN, make_batch_dims(dims, 1));
        profile->setDimensions(name, OptProfileSelector::kOPT, make_batch_dims(dims, opt_batch));
        profile->setDimensions(name, OptProfileSelector::kMAX, make_batch_dims(dims, max_batch));
      }

      config->addOptimizationProfile(profile);
      config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
      config->setBuilderOptimizationLevel(0);
      if(use_fp16) {
        config->setFlag(BuilderFlag::kFP16);
      }

      auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
      if(!plan) {
        throw make_error("buildSerializedNetwork", "TensorRT returned null plan");
      }
      const auto* begin = static_cast<const char*>(plan->data());
      return std::vector<char>(begin, begin + plan->size());
    }

  }  // namespace

  namespace {
    auto create_event(int cuda_device_index, unsigned int flags = cudaEventDisableTiming) -> cudaEvent_t {
      cudaEvent_t event = nullptr;
      check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
      check_cuda(cudaEventCreateWithFlags(&event, flags), "cudaEventCreateWithFlags");
      return event;
    }

    void destroy_event(int cuda_device_index, cudaEvent_t& event) noexcept {
      if(event) {
        (void)cudaSetDevice(cuda_device_index);
        (void)cudaEventDestroy(event);
        event = nullptr;
      }
    }

    auto is_event_ready(cudaEvent_t event) -> bool {
      const cudaError_t status = cudaEventQuery(event);
      if(status == cudaSuccess) {
        return true;
      }
      if(status != cudaErrorNotReady) {
        check_cuda(status, "cudaEventQuery");
      }
      return false;
    }

  }  // namespace

  TrtBackend::TrtInferState::TrtInferState(
    TrtBackend& backend,
    const HostSlot& host_slot,
    std::uint32_t batch_size,
    std::uint32_t group_id,
    std::uint32_t lane_id,
    std::uint32_t bank_id)
    : backend(backend),
      host_slot(host_slot),
      batch_size(batch_size),
      group_id(group_id),
      lane_id(lane_id),
      bank_id(bank_id),
      input_copy_sizes(backend.batch_layout.input_row_bytes.size()),
      output_copy_sizes(backend.batch_layout.output_row_bytes.size()) {
    for(std::size_t tensor_index = 0; tensor_index < input_copy_sizes.size(); ++tensor_index) {
      input_copy_sizes[tensor_index] = static_cast<std::size_t>(batch_size) * backend.batch_layout.input_row_bytes[tensor_index];
    }
    for(std::size_t tensor_index = 0; tensor_index < output_copy_sizes.size(); ++tensor_index) {
      output_copy_sizes[tensor_index] = static_cast<std::size_t>(batch_size) * backend.batch_layout.output_row_bytes[tensor_index];
    }
  }

  auto TrtBackend::TrtInferState::submit_h2d() -> std::function<bool()> {
    return [this]() {
      if(!bank_acquired) {
        if(bank_is_busy()) {
          return false;
        }
        bank_is_busy() = true;
        bank_acquired = true;

        check_cuda(cudaSetDevice(cuda_device_index()), "cudaSetDevice");
        if(batch_size == schedlab_config().infer.batch_size) {
          check_cuda(
            cudaMemcpyAsync(
              input_slab(),
              host_slot.input_slab,
              backend.batch_layout.input_slab_bytes,
              cudaMemcpyHostToDevice,
              io_stream()),
            "cudaMemcpyAsync(H2D slab)");
        } else {
          cudaMemcpyAttributes attr{};
          attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
          attr.srcLocHint = {cudaMemLocationTypeInvalid, 0};
          attr.dstLocHint = {cudaMemLocationTypeInvalid, 0};
          attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
          std::size_t attr_starts[] = {0};
          std::size_t fail_idx = static_cast<std::size_t>(-1);
          check_cuda(
            cudaMemcpyBatchAsync(
              backend.bank_inputs[group_id][lane_id][bank_id].data(),
              const_cast<void**>(host_slot.inputs.data()),
              input_copy_sizes.data(),
              input_copy_sizes.size(),
              &attr,
              attr_starts,
              1,
              &fail_idx,
              io_stream()),
            "cudaMemcpyBatchAsync(H2D)");
        }
        check_cuda(cudaEventRecord(h2d_done_event(), io_stream()), "cudaEventRecord(h2d_done)");
      }
      return is_event_ready(h2d_done_event());
    };
  }

  auto TrtBackend::TrtInferState::submit_infer() -> std::function<bool()> {
    check_cuda(cudaSetDevice(cuda_device_index()), "cudaSetDevice");

    for(std::size_t tensor_index = 0; tensor_index < backend.input_tensor_names.size(); ++tensor_index) {
      execution_context().setInputShape(
        backend.input_tensor_names[tensor_index].c_str(),
        make_batch_dims(backend.input_tensor_dims[tensor_index], static_cast<int>(batch_size)));
    }

    if(!execution_context().enqueueV3(infer_stream())) {
      throw make_error("enqueueV3", "returned false");
    }
    check_cuda(cudaEventRecord(infer_done_event(), infer_stream()), "cudaEventRecord(infer_done)");
    return [this]() {
      return is_event_ready(infer_done_event());
    };
  }

  auto TrtBackend::TrtInferState::submit_d2h() -> std::function<bool()> {
    check_cuda(cudaSetDevice(cuda_device_index()), "cudaSetDevice");

    if(batch_size == schedlab_config().infer.batch_size) {
      check_cuda(
        cudaMemcpyAsync(
          host_slot.output_slab,
          output_slab(),
          backend.batch_layout.output_slab_bytes,
          cudaMemcpyDeviceToHost,
          io_stream()),
        "cudaMemcpyAsync(D2H slab)");
    } else {
      cudaMemcpyAttributes attr{};
      attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
      attr.srcLocHint = {cudaMemLocationTypeInvalid, 0};
      attr.dstLocHint = {cudaMemLocationTypeInvalid, 0};
      attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
      std::size_t attr_starts[] = {0};
      std::size_t fail_idx = static_cast<std::size_t>(-1);
      check_cuda(
        cudaMemcpyBatchAsync(
          const_cast<void**>(host_slot.outputs.data()),
          backend.bank_outputs[group_id][lane_id][bank_id].data(),
          output_copy_sizes.data(),
          output_copy_sizes.size(),
          &attr,
          attr_starts,
          1,
          &fail_idx,
          io_stream()),
        "cudaMemcpyBatchAsync(D2H)");
    }

    check_cuda(cudaEventRecord(d2h_done_event(), io_stream()), "cudaEventRecord(d2h_done)");
    return [this]() {
      if(!is_event_ready(d2h_done_event())) {
        return false;
      }
      if(bank_acquired) {
        bank_is_busy() = false;
        bank_acquired = false;
      }
      return true;
    };
  }

  auto TrtBackend::TrtInferState::cuda_device_index() const -> int {
    return schedlab_config().infer.cuda_device_ids[group_id];
  }

  auto TrtBackend::TrtInferState::infer_stream() const -> cudaStream_t {
    return backend.infer_streams[group_id][lane_id];
  }

  auto TrtBackend::TrtInferState::input_slab() const -> void* {
    return backend.bank_input_slabs[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::output_slab() const -> void* {
    return backend.bank_output_slabs[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::execution_context() const -> nvinfer1::IExecutionContext& {
    return *backend.execution_contexts[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::io_stream() const -> cudaStream_t {
    return backend.io_streams[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::h2d_done_event() const -> cudaEvent_t {
    return backend.h2d_done_events[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::infer_done_event() const -> cudaEvent_t {
    return backend.infer_done_events[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::d2h_done_event() const -> cudaEvent_t {
    return backend.d2h_done_events[group_id][lane_id][bank_id];
  }

  auto TrtBackend::TrtInferState::bank_is_busy() -> bool& {
    return backend.bank_busy[group_id][lane_id][bank_id];
  }

  TrtBackend::TrtBackend() {
    const auto& infer_config = schedlab_config().infer;
    if(!std::filesystem::exists(infer_config.onnx_model_path)) {
      throw make_error("resolve_onnx_model_path", infer_config.onnx_model_path.string() + " does not exist");
    }
    model_signature = file_signature(infer_config.onnx_model_path);
    ensure_directory(infer_config.trt_cache_dir);
    resolve_cuda_devices();
    initialize_devices();
  }

  TrtBackend::~TrtBackend() {
    const auto& cuda_device_ids = schedlab_config().infer.cuda_device_ids;
    for(std::size_t group_id = 0; group_id < io_streams.size(); ++group_id) {
      (void)cudaSetDevice(cuda_device_ids[group_id]);
      for(std::size_t lane_id = 0; lane_id < io_streams[group_id].size(); ++lane_id) {
        for(std::size_t bank_id = 0; bank_id < 2; ++bank_id) {
          destroy_event(cuda_device_ids[group_id], h2d_done_events[group_id][lane_id][bank_id]);
          destroy_event(cuda_device_ids[group_id], infer_done_events[group_id][lane_id][bank_id]);
          destroy_event(cuda_device_ids[group_id], d2h_done_events[group_id][lane_id][bank_id]);
          if(bank_input_slabs[group_id][lane_id][bank_id]) {
            (void)cudaFree(bank_input_slabs[group_id][lane_id][bank_id]);
          }
          if(bank_output_slabs[group_id][lane_id][bank_id]) {
            (void)cudaFree(bank_output_slabs[group_id][lane_id][bank_id]);
          }
          if(io_streams[group_id][lane_id][bank_id]) {
            (void)cudaStreamDestroy(io_streams[group_id][lane_id][bank_id]);
          }
        }
        if(infer_streams[group_id][lane_id]) {
          (void)cudaStreamDestroy(infer_streams[group_id][lane_id]);
        }
      }
    }
  }

  void TrtBackend::resolve_cuda_devices() {
    auto& infer_config = schedlab_config().infer;
    int available = 0;
    check_cuda(cudaGetDeviceCount(&available), "cudaGetDeviceCount");

    if(infer_config.cuda_device_ids.empty()) {
      const std::uint32_t requested = static_cast<std::uint32_t>(available);
      infer_config.cuda_device_ids.reserve(requested);
      for(std::uint32_t i = 0; i < requested; ++i) {
        infer_config.cuda_device_ids.push_back(static_cast<int>(i));
      }
    }
  }

  auto TrtBackend::allocate_raw_host_storage(std::size_t total_bytes) const -> void* {
    void* raw_storage = nullptr;
    check_cuda(cudaMallocHost(&raw_storage, total_bytes), "cudaMallocHost");
    return raw_storage;
  }

  void TrtBackend::release_raw_host_storage(void* raw_storage) const noexcept {
    if(raw_storage) {
      (void)cudaFreeHost(raw_storage);
    }
  }

  auto TrtBackend::plan_cache_path(const cudaDeviceProp& prop, bool use_fp16, std::uint32_t max_batch_size) const
    -> std::filesystem::path {
    std::ostringstream name;
    name << "trt_onnx_" << k_network_revision << "_trt" << getInferLibVersion() << "_gpu-"
         << sanitize_name(prop.name) << "_sm" << prop.major << prop.minor << "_model-"
         << model_signature << "_batch" << max_batch_size << "_fp" << (use_fp16 ? 16 : 32) << ".plan";
    return schedlab_config().infer.trt_cache_dir / name.str();
  }

  auto TrtBackend::build_or_load_plan(
    const cudaDeviceProp& prop,
    bool use_fp16,
    std::uint32_t max_batch_size,
    bool& loaded)
    -> std::vector<char> {
    const auto cache_path = plan_cache_path(prop, use_fp16, max_batch_size);
    loaded = false;
    if(std::filesystem::exists(cache_path)) {
      loaded = true;
      return read_binary_file(cache_path);
    }

    static std::mutex build_mutex;
    std::lock_guard lock(build_mutex);
    if(std::filesystem::exists(cache_path)) {
      loaded = true;
      return read_binary_file(cache_path);
    }

    std::vector<char> plan = build_serialized_plan(schedlab_config().infer.onnx_model_path, max_batch_size, use_fp16);
    write_binary_file(cache_path, plan);
    return plan;
  }

  void TrtBackend::initialize_layout_from_engine(const nvinfer1::ICudaEngine& engine) {
    const std::size_t tensor_alignment = schedlab_config().trt.tensor_alignment;
    const std::uint32_t max_batch_size = schedlab_config().infer.batch_size;
    if(!input_tensor_names.empty()) {
      return;
    }

    std::size_t input_cursor = 0;
    std::size_t output_cursor = 0;
    for(int i = 0; i < engine.getNbIOTensors(); ++i) {
      const char* name = engine.getIOTensorName(i);
      const Dims dims = engine.getTensorShape(name);
      const std::size_t row_bytes = dims_volume_without_batch(dims) * data_type_bytes(engine.getTensorDataType(name));

      if(engine.getTensorIOMode(name) == TensorIOMode::kINPUT) {
        input_cursor = align_up(input_cursor, tensor_alignment);
        input_tensor_names.push_back(name);
        input_tensor_dims.push_back(dims);
        batch_layout.input_row_bytes.push_back(row_bytes);
        batch_layout.input_tensor_offsets.push_back(input_cursor);
        input_cursor += align_up(row_bytes * static_cast<std::size_t>(max_batch_size), tensor_alignment);
      } else {
        output_cursor = align_up(output_cursor, tensor_alignment);
        output_tensor_names.push_back(name);
        batch_layout.output_row_bytes.push_back(row_bytes);
        batch_layout.output_tensor_offsets.push_back(output_cursor);
        output_cursor += align_up(row_bytes * static_cast<std::size_t>(max_batch_size), tensor_alignment);
      }
    }

    batch_layout.input_slab_bytes = align_up(input_cursor, tensor_alignment);
    batch_layout.output_slab_bytes = align_up(output_cursor, tensor_alignment);
  }

  void TrtBackend::initialize_devices() {
    const auto& infer_config = schedlab_config().infer;
    runtimes.reserve(infer_config.cuda_device_ids.size());
    engines.reserve(infer_config.cuda_device_ids.size());
    infer_streams.reserve(infer_config.cuda_device_ids.size());
    next_bank_ids.reserve(infer_config.cuda_device_ids.size());
    bank_input_slabs.reserve(infer_config.cuda_device_ids.size());
    bank_output_slabs.reserve(infer_config.cuda_device_ids.size());
    bank_inputs.reserve(infer_config.cuda_device_ids.size());
    bank_outputs.reserve(infer_config.cuda_device_ids.size());
    execution_contexts.reserve(infer_config.cuda_device_ids.size());
    io_streams.reserve(infer_config.cuda_device_ids.size());
    h2d_done_events.reserve(infer_config.cuda_device_ids.size());
    infer_done_events.reserve(infer_config.cuda_device_ids.size());
    d2h_done_events.reserve(infer_config.cuda_device_ids.size());
    bank_busy.reserve(infer_config.cuda_device_ids.size());

    for(std::size_t group_id = 0; group_id < infer_config.cuda_device_ids.size(); ++group_id) {
      const int cuda_device_index = infer_config.cuda_device_ids[group_id];
      check_cuda(cudaSetDevice(cuda_device_index), "cudaSetDevice");
      check_cuda(cudaFree(0), "cudaFree(0)");

      cudaDeviceProp prop{};
      check_cuda(cudaGetDeviceProperties(&prop, cuda_device_index), "cudaGetDeviceProperties");
      const bool use_fp16 = prop.major >= 7;

      bool loaded_from_cache = false;
      std::vector<char> plan = build_or_load_plan(prop, use_fp16, infer_config.batch_size, loaded_from_cache);

      TrtLogger logger;
      const std::string cache_file = plan_cache_path(prop, use_fp16, infer_config.batch_size).string();
      auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
      auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
      if(!engine && loaded_from_cache) {
        std::filesystem::remove(cache_file);
        loaded_from_cache = false;
        plan = build_or_load_plan(prop, use_fp16, infer_config.batch_size, loaded_from_cache);
        engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
      }

      initialize_layout_from_engine(*engine);
      std::vector<cudaStream_t> group_infer_streams(infer_config.lanes_per_device, nullptr);
      std::vector<std::uint32_t> group_next_bank_ids(infer_config.lanes_per_device, 0);
      std::vector<std::array<void*, 2>> group_bank_input_slabs(
        infer_config.lanes_per_device, std::array<void*, 2>{nullptr, nullptr});
      std::vector<std::array<void*, 2>> group_bank_output_slabs(
        infer_config.lanes_per_device, std::array<void*, 2>{nullptr, nullptr});
      std::vector<std::array<std::vector<void*>, 2>> group_bank_inputs(infer_config.lanes_per_device);
      std::vector<std::array<std::vector<void*>, 2>> group_bank_outputs(infer_config.lanes_per_device);
      std::vector<std::array<std::unique_ptr<nvinfer1::IExecutionContext>, 2>> group_execution_contexts(
        infer_config.lanes_per_device);
      std::vector<std::array<cudaStream_t, 2>> group_io_streams(
        infer_config.lanes_per_device, std::array<cudaStream_t, 2>{nullptr, nullptr});
      std::vector<std::array<cudaEvent_t, 2>> group_h2d_done_events(
        infer_config.lanes_per_device, std::array<cudaEvent_t, 2>{nullptr, nullptr});
      std::vector<std::array<cudaEvent_t, 2>> group_infer_done_events(
        infer_config.lanes_per_device, std::array<cudaEvent_t, 2>{nullptr, nullptr});
      std::vector<std::array<cudaEvent_t, 2>> group_d2h_done_events(
        infer_config.lanes_per_device, std::array<cudaEvent_t, 2>{nullptr, nullptr});
      std::vector<std::array<bool, 2>> group_bank_busy(
        infer_config.lanes_per_device, std::array<bool, 2>{false, false});
      for(std::uint32_t lane_index = 0; lane_index < infer_config.lanes_per_device; ++lane_index) {
        check_cuda(
          cudaStreamCreateWithFlags(&group_infer_streams[lane_index], cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
        for(std::uint32_t bank_id = 0; bank_id < 2; ++bank_id) {
          check_cuda(
            cudaMalloc(&group_bank_input_slabs[lane_index][bank_id], batch_layout.input_slab_bytes),
            "cudaMalloc(input_slab)");
          check_cuda(
            cudaMalloc(&group_bank_output_slabs[lane_index][bank_id], batch_layout.output_slab_bytes),
            "cudaMalloc(output_slab)");
          group_bank_inputs[lane_index][bank_id].resize(input_tensor_names.size());
          group_bank_outputs[lane_index][bank_id].resize(output_tensor_names.size());
          group_execution_contexts[lane_index][bank_id].reset(engine->createExecutionContext());
          for(std::size_t tensor_index = 0; tensor_index < input_tensor_names.size(); ++tensor_index) {
            auto* addr =
              static_cast<char*>(group_bank_input_slabs[lane_index][bank_id]) + batch_layout.input_tensor_offsets[tensor_index];
            group_bank_inputs[lane_index][bank_id][tensor_index] = addr;
            group_execution_contexts[lane_index][bank_id]->setTensorAddress(input_tensor_names[tensor_index].c_str(), addr);
          }
          for(std::size_t tensor_index = 0; tensor_index < output_tensor_names.size(); ++tensor_index) {
            auto* addr =
              static_cast<char*>(group_bank_output_slabs[lane_index][bank_id]) + batch_layout.output_tensor_offsets[tensor_index];
            group_bank_outputs[lane_index][bank_id][tensor_index] = addr;
            group_execution_contexts[lane_index][bank_id]->setTensorAddress(output_tensor_names[tensor_index].c_str(), addr);
          }
          check_cuda(
            cudaStreamCreateWithFlags(&group_io_streams[lane_index][bank_id], cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
          group_h2d_done_events[lane_index][bank_id] = create_event(cuda_device_index);
          group_infer_done_events[lane_index][bank_id] = create_event(cuda_device_index);
          group_d2h_done_events[lane_index][bank_id] = create_event(cuda_device_index);
        }
      }
      runtimes.push_back(std::move(runtime));
      engines.push_back(std::move(engine));
      infer_streams.push_back(std::move(group_infer_streams));
      next_bank_ids.push_back(std::move(group_next_bank_ids));
      bank_input_slabs.push_back(std::move(group_bank_input_slabs));
      bank_output_slabs.push_back(std::move(group_bank_output_slabs));
      bank_inputs.push_back(std::move(group_bank_inputs));
      bank_outputs.push_back(std::move(group_bank_outputs));
      execution_contexts.push_back(std::move(group_execution_contexts));
      io_streams.push_back(std::move(group_io_streams));
      h2d_done_events.push_back(std::move(group_h2d_done_events));
      infer_done_events.push_back(std::move(group_infer_done_events));
      d2h_done_events.push_back(std::move(group_d2h_done_events));
      bank_busy.push_back(std::move(group_bank_busy));
    }
  }

  auto TrtBackend::make_infer_state(const HostSlot& host_slot, std::uint32_t batch_size, InferLane infer_lane)
    -> std::unique_ptr<InferBackend::InferState> {
    const std::uint32_t group_id = infer_lane.group_id;
    const std::uint32_t lane_id = infer_lane.lane_id;
    const std::uint32_t bank_id = next_bank_ids[group_id][lane_id];
    next_bank_ids[group_id][lane_id] = (bank_id + 1) % 2U;
    return std::make_unique<TrtInferState>(
      *this,
      host_slot,
      batch_size,
      group_id,
      lane_id,
      bank_id);
  }
}  // namespace schedlab
