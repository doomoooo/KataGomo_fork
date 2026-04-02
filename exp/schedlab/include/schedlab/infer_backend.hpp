#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace schedlab {
  // 单个 batch 在内存中的通用排布描述。
  // dispatcher 用它把 HostSlot 里的 batch tensor 基地址拆成 per-request 地址视图。
  struct BatchLayout {
    // 输入侧每个 tensor 的单行样本字节数，不含 batch 维。
    std::vector<std::size_t> input_row_bytes;
    // 输出侧每个 tensor 的单行样本字节数，不含 batch 维。
    std::vector<std::size_t> output_row_bytes;
    // 每个输入 tensor 在 packed input slab 中的起始偏移。
    std::vector<std::size_t> input_tensor_offsets;
    // 每个输出 tensor 在 packed output slab 中的起始偏移。
    std::vector<std::size_t> output_tensor_offsets;
    // 输入 slab 总字节数。
    std::size_t input_slab_bytes = 0;
    // 输出 slab 总字节数。
    std::size_t output_slab_bytes = 0;
  };

  struct InferLane {
    // 调度层可见的逻辑 group 编号；通常对应一张设备，但不要求等于真实 CUDA device id。
    std::uint32_t group_id = 0;
    // group 内的局部 lane 编号。
    std::uint32_t lane_id = 0;
  };

  // 一个可复用的 host 侧 batch 容器。
  // backend 只负责这些“batch 级”对象：
  // 1. raw_storage 表示一次性申请到的整块原始 host 内存
  // 2. input_slab / output_slab 是其中按对齐规则切出的两段连续区域
  // 3. inputs / outputs 是每个 batch tensor 的基地址
  // per-request 地址视图由 dispatcher 基于 BatchLayout 在外部切出来。
  struct HostSlot {
    // 原始 host 内存首地址；release_host_slots() 需要用它归还资源。
    void* raw_storage = nullptr;
    // 输入 tensor 打包后的 host slab 起点。
    void* input_slab = nullptr;
    // 输出 tensor 打包后的 host slab 起点。
    void* output_slab = nullptr;
    // 每个输入 tensor 在 host 侧的 batch 基地址。
    std::vector<void*> inputs;
    // 每个输出 tensor 在 host 侧的 batch 基地址。
    std::vector<void*> outputs;
  };

  // 推理 backend 的最小接口面。
  // dispatcher / scheduler 只依赖这里暴露的抽象，不直接知道 CUDA stream、TRT context 等细节。
  //
  // 这套接口的核心约束是：
  // 1. dispatcher 先为“一次完整 infer 生命周期”创建一个 InferState
  // 2. HostSlot / batch_size / lane 等本次 infer 固定下来的上下文，全部收进这个 state
  // 3. dispatcher 仍然显式地分三步调用 submit_h2d() / submit_infer() / submit_d2h()
  // 4. 每一步只返回一个 bool 谓词；dispatcher 自己负责 spin wait / awaitable 封装
  class InferBackend {
   public:
    // 一次完整 infer 生命周期的不透明状态对象。
    // backend 可以在里面保存任何只需活到“本次 infer 结束”为止的资源：
    // 例如 TRT backend 的 bank 选择结果、事件句柄、临时运行时引用等。
    //
    // dispatcher 只按固定时序驱动这几个接口：
    // 1. 先调用 submit_h2d() 并等待其返回的完成谓词
    // 2. 再调用 submit_infer() 并等待其返回的完成谓词
    // 3. 最后调用 submit_d2h() 并等待其返回的完成谓词
    struct InferState {
      virtual ~InferState() = default;

      InferState(const InferState&) = delete;
      auto operator=(const InferState&) -> InferState& = delete;

      // 提交本次 infer 的 H2D，并返回“H2D 是否完成”的轮询谓词。
      // backend 可以在这个谓词里顺手完成资源 handoff，
      // 例如等待某个 ping-pong bank 空闲、占住它，并在首次轮询时真正发起 H2D。
      virtual auto submit_h2d() -> std::function<bool()> = 0;
      // 提交本次 infer 的计算，并返回“infer 是否完成”的轮询谓词。
      virtual auto submit_infer() -> std::function<bool()> = 0;
      // 提交本次 infer 的 D2H，并返回“D2H 是否完成”的轮询谓词。
      virtual auto submit_d2h() -> std::function<bool()> = 0;

     protected:
      InferState() = default;
    };

    virtual ~InferBackend();

    InferBackend(const InferBackend&) = delete;
    auto operator=(const InferBackend&) -> InferBackend& = delete;

    // 一次性分配若干个 host slot。
    // backend 自己持有这组 batch 级 HostSlot；dispatcher 只在外层维护对应的 ring 状态。
    void allocate_host_slots(std::uint32_t host_slot_count);
    // 释放当前持有的全部 host slot 资源。
    void release_host_slots() noexcept;
    // 为一次完整 infer 生命周期创建 state。
    // 不同 backend 可以在 state 里收进各自需要维护的临时资源，但这些细节都不应泄漏给 dispatcher。
    virtual auto make_infer_state(const HostSlot& host_slot, std::uint32_t batch_size, InferLane lane)
      -> std::unique_ptr<InferState> = 0;

   protected:
    InferBackend() = default;
    // backend 自己负责提供原始 host 内存的分配与归还方式；
    // 基类会基于 BatchLayout 统一把它组装成 HostSlot。
    virtual auto allocate_raw_host_storage(std::size_t total_bytes) const -> void* = 0;
    virtual void release_raw_host_storage(void* raw_storage) const noexcept = 0;

   public:
    // backend 对单个 batch 的稳定内存排布要求。
    // dispatcher 只关心 row_bytes；InferBackend 会用其余字段统一构造 HostSlot。
    BatchLayout batch_layout{};
    // backend 持有的整组 host slot。
    std::vector<HostSlot> host_slots{};
  };
}  // namespace schedlab
