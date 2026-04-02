#pragma once

#include "schedlab/infer_backend.hpp"

#include <NvInfer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace schedlab {
  // 真实 TRT backend。
  // 负责 plan cache、多 GPU/lane/bank 初始化，以及真实的 H2D -> infer -> D2H 提交。
  //
  // 运行时结构分三层：
  // 1. device: 对应一张实际 CUDA 卡，持有 runtime / engine 和若干 lane
  // 2. lane: 调度层看到的一条逻辑执行 lane，持有一条 infer_stream
  // 3. bank: lane 内的双缓冲单元，持有一组 device input/output slab、一条 io_stream、
  //    一个 execution context，以及几枚常驻完成事件
  //
  // 时序上：
  // 1. H2D 和 D2H 都走 bank.io_stream
  // 2. infer 走 lane.infer_stream
  // 3. make_infer_state() 会轮转选择 bank，但 bank_id 只保留在 TRT 自己的 infer state 里
  // 4. submit_h2d() 的谓词负责 bank handoff：等待 bank 空闲、占住它，并在首次轮询时真正提交 H2D
  // 5. submit_d2h() 的谓词在首次观察到 D2H 完成时释放 bank
  class TrtBackend final : public InferBackend {
   public:
    // 构造一个真实 TensorRT backend。
    // 所有输入都直接来自全局 schedlab_config()：
    // 1. infer.batch_size
    // 2. infer.lanes_per_device
    // 3. infer.cuda_device_ids
    // 4. infer.onnx_model_path
    // 5. infer.trt_cache_dir
    TrtBackend();
    // 释放内部 TRT / CUDA 资源，包括 engine、context、stream、event 和 device buffer。
    ~TrtBackend();

    TrtBackend(const TrtBackend&) = delete;
    auto operator=(const TrtBackend&) -> TrtBackend& = delete;

    // 为一次完整 infer 生命周期创建 TRT 自己的 state。
    // state 只借用 backend 常驻的 lane/bank 资源，并把 bank handoff 细节藏起来。
   auto make_infer_state(const HostSlot& host_slot, std::uint32_t batch_size, InferLane lane)
      -> std::unique_ptr<InferBackend::InferState> override;

   private:
    class TrtInferState final : public InferBackend::InferState {
     public:
      TrtInferState(
        TrtBackend& backend,
        const HostSlot& host_slot,
        std::uint32_t batch_size,
        std::uint32_t group_id,
        std::uint32_t lane_id,
        std::uint32_t bank_id);

      auto submit_h2d() -> std::function<bool()> override;
      auto submit_infer() -> std::function<bool()> override;
      auto submit_d2h() -> std::function<bool()> override;

     private:
      auto cuda_device_index() const -> int;
      auto infer_stream() const -> cudaStream_t;
      auto input_slab() const -> void*;
      auto output_slab() const -> void*;
      auto execution_context() const -> nvinfer1::IExecutionContext&;
      auto io_stream() const -> cudaStream_t;
      auto h2d_done_event() const -> cudaEvent_t;
      auto infer_done_event() const -> cudaEvent_t;
      auto d2h_done_event() const -> cudaEvent_t;
      auto bank_is_busy() -> bool&;

      TrtBackend& backend;
      const HostSlot& host_slot;
      std::uint32_t batch_size = 0;
      std::uint32_t group_id = 0;
      std::uint32_t lane_id = 0;
      std::uint32_t bank_id = 0;
      bool bank_acquired = false;
      std::vector<std::size_t> input_copy_sizes;
      std::vector<std::size_t> output_copy_sizes;
    };

    // 解析并确定要使用的 CUDA 设备列表。
    // 如果用户没显式指定，就按 [0, cudaGetDeviceCount()) 全部启用。
    void resolve_cuda_devices();
    // 申请 / 释放原始 pinned host 内存。
    // InferBackend 基类会基于 BatchLayout 把它统一组装成 HostSlot。
    auto allocate_raw_host_storage(std::size_t total_bytes) const -> void* override;
    void release_raw_host_storage(void* raw_storage) const noexcept override;
    // 计算某张卡对应的 plan cache 路径。
    // 命名里会带上网络版本、TRT 版本、GPU 型号、SM、模型签名、batch 和精度。
    auto plan_cache_path(const cudaDeviceProp& prop, bool use_fp16, std::uint32_t max_batch_size) const
      -> std::filesystem::path;
    // 读取或构建 plan。
    // 优先命中 cache；miss 时串行建图并把 plan 写回磁盘。
    auto build_or_load_plan(const cudaDeviceProp& prop, bool use_fp16, std::uint32_t max_batch_size, bool& loaded_from_cache)
      -> std::vector<char>;
    // 从 engine 读取 batch 版式，并初始化 layout / binding 信息。
    // 这一步只需要做一次，后续所有 host slot / bank 都复用同一套版式。
    void initialize_layout_from_engine(const nvinfer1::ICudaEngine& engine);
    // 初始化所有设备、lane、bank。
    // 同时负责 plan 反序列化、layout 解析，以及各卡上 lane 运行时对象的建立。
    void initialize_devices();

    // 模型签名，用于 cache 命名和失效隔离。
    std::string model_signature;
    // 每张卡一份 TRT runtime。
    std::vector<std::unique_ptr<nvinfer1::IRuntime>> runtimes;
    // 每张卡一份 TRT engine。
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> engines;
    // 每个 lane 的 infer stream。
    std::vector<std::vector<cudaStream_t>> infer_streams;
    // 每个 lane 下一次要轮到的 bank。
    std::vector<std::vector<std::uint32_t>> next_bank_ids;
    // 每个 lane 的两个 bank 输入 slab。
    std::vector<std::vector<std::array<void*, 2>>> bank_input_slabs;
    // 每个 lane 的两个 bank 输出 slab。
    std::vector<std::vector<std::array<void*, 2>>> bank_output_slabs;
    // 每个 lane 的两个 bank 输入 tensor 基地址。
    std::vector<std::vector<std::array<std::vector<void*>, 2>>> bank_inputs;
    // 每个 lane 的两个 bank 输出 tensor 基地址。
    std::vector<std::vector<std::array<std::vector<void*>, 2>>> bank_outputs;
    // 每个 lane 的两个 bank execution context。
    std::vector<std::vector<std::array<std::unique_ptr<nvinfer1::IExecutionContext>, 2>>> execution_contexts;
    // 每个 lane 的两个 bank io stream。
    std::vector<std::vector<std::array<cudaStream_t, 2>>> io_streams;
    // 每个 lane 的两个 bank H2D 完成事件。
    std::vector<std::vector<std::array<cudaEvent_t, 2>>> h2d_done_events;
    // 每个 lane 的两个 bank infer 完成事件。
    std::vector<std::vector<std::array<cudaEvent_t, 2>>> infer_done_events;
    // 每个 lane 的两个 bank D2H 完成事件。
    std::vector<std::vector<std::array<cudaEvent_t, 2>>> d2h_done_events;
    // 每个 lane 的两个 bank busy 标志。
    std::vector<std::vector<std::array<bool, 2>>> bank_busy;
    // 输入 tensor 的名字；enqueueV3 风格接口需要按名字设 shape 和地址。
    std::vector<std::string> input_tensor_names;
    // 输入 tensor 的原始维度；d[0] 会在运行时替换成实际 batch。
    std::vector<nvinfer1::Dims> input_tensor_dims;
    // 输出 tensor 的名字；enqueueV3 风格接口需要按名字设地址。
    std::vector<std::string> output_tensor_names;
  };
}  // namespace schedlab
