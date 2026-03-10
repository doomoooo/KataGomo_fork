# Last One Percent Plan

目标：持续收集为了把 ONNX/TensorRT 通信与推理 overlap 压到最后一点性能所需的硬约束、风险点和最小破坏性改造路径。

原则：
- 先记录“已验证事实”，再写“推论”。
- 推论必须偏保守，优先选择文档明确支持的 CUDA 语义。
- 设计优先级是“最小破坏性改造”，不是“最漂亮的新架构”。

当前基线：
- 分支：`onnx-trt-overlapping`
- 首个 commit：`c62c508a` `Add realtime profiling dashboard`

## Research Log

### 2026-03-10: CUDA 多 GPU stream/event 等待语义与 `cudaSetDevice` 开销

已验证事实：

1. `cudaSetDevice()` 设定的是 host thread 当前操作的 device。设备内存分配、kernel launch 都落在 current device；stream 和 event 也在创建时绑定到 current device。
2. 一个 host thread 可以在任意时刻切换自己操作的 device，但把 kernel 发到“不属于 current device 的 stream”会失败。
3. `cudaEventSynchronize()` 和 `cudaEventQuery()` 在 event 绑定的 device 与 current device 不同的情况下仍然可以成功。
4. `cudaStreamWaitEvent()` 在 stream 和 event 属于不同 device 的情况下也可以成功，因此它可以直接用于多 device 之间同步。
5. 每个 device 都有自己的 default stream。多 GPU overlap 路径不应依赖 `NULL` / default stream 的隐式行为。
6. `cudaSetDevice()` 不会对前一个 device 或新 device 做同步。
7. `cudaSetDevice()` 在热路径上通常应当是低开销调用；只有在它需要初始化 runtime/context state 时才可能明显变慢。
8. 从 CUDA 12.0 开始，`cudaSetDevice()` 会在切换 host thread current device 后显式初始化 runtime 和 primary context。也就是说，某张卡第一次被 `cudaSetDevice()` 触达时，context 创建、代码装载、可能的 JIT 等成本都可能落在这一步。
9. 对 host thread 的等待策略，`cudaSetDeviceFlags()` 可以在任何其他 CUDA 调用之前设置。如果确实要主动自旋，可以用 `cudaDeviceScheduleSpin`；如果更关心 CPU 占用，可以考虑 `cudaDeviceScheduleBlockingSync`。

保守结论：

- 如果一个 CPU 线程需要观察多张卡上多个 stream 的完成状态，优先方案应是“每个 stream 上记录 completion event，然后由控制线程 `cudaEventQuery()`/`cudaEventSynchronize()` 观察 event”。
- 我目前没有在 NVIDIA 官方文档里找到与 event 同等级别、明确保证“`cudaStreamQuery()`/`cudaStreamSynchronize()` 在 stream 不属于 current device 时也可靠可用”的表述。因此现阶段不应把“跨 device 直接 query/sync stream”当成设计前提。
- 即使 `cudaSetDevice()` 热路径很便宜，也不应把“在一个中央线程里高频来回切 device”作为核心调度机制。这样做在语义上更脆，也会把首次 context 初始化成本留在容易污染 latency 的路径上。
- 所有参与 overlap 的 GPU、context、stream、event 都应该在启动阶段预热，而不是等到真正的推理热路径第一次触发时再懒初始化。

对当前代码库的低侵入方向：

- 尽量保持“哪个线程拥有某张卡上的 launch 权”，不要让外部协调线程把 kernel 直接发到别的 device 的 stream。
- 更适合暴露给外部协调层的是“completion event / completion signal / queue 状态”，而不是“远程操作另一个线程手里的 CUDA stream”。
- overlap 的最小切入点更像是：在 H2D、enqueue、D2H 等阶段边界记录 event，再由控制逻辑做 query/wait/统计，而不是先重写一整套新的跨设备线程模型。
- 新路径中应显式使用命名 stream，避免 default stream 语义混入。

后续需要验证的问题：

- 当前仓库里，哪些线程真正拥有 TensorRT / ONNX 的 stream 与 launch 权？
- 现在的 H2D、enqueue、D2H、后处理边界分别落在哪些函数和线程上？
- completion event 能否仅通过扩展现有 backend 接口接出来，还是必须引入新的跨 backend 调度抽象？
- 目标机器的多卡拓扑下，真正需要 overlap 的是 host<->device copy、device<->device copy，还是纯粹的多 backend 并发？
- 启动阶段最安全的多 GPU 预热挂点在哪里，才能不污染现有行为和错误处理？

Primary sources:

- CUDA C++ Programming Guide 13.0.1, Multi-Device System, Device Selection / Stream and Event Behavior:
  https://docs.nvidia.com/cuda/archive/13.0.1/pdf/CUDA_C_Programming_Guide.pdf
- CUDA C++ Programming Guide 12.1, Multi-Device System:
  https://docs.nvidia.com/cuda/archive/12.1.0/pdf/CUDA_C_Programming_Guide.pdf
- CUDA Runtime API 13.0.1, `cudaSetDevice`:
  https://docs.nvidia.com/cuda/archive/13.0.1/pdf/CUDA_Runtime_API.pdf
- CUDA Programming Guide 13.1, Runtime Initialization:
  https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf
