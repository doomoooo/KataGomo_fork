# 计划

本文档补充 [v1](./v1.md)，专门展开 TensorRT / GPU 侧的第一版架构设计。

## plan_v1_gpu

状态：首版（冻结 GPU 侧控制面/数据面分层）

### 1. 文档目标

`v1` 已经冻结了搜索侧和 `stdexec` 的总边界，但 GPU 侧仍需要一份单独文档，回答以下问题：

1. 当前 TensorRT 路径里，到底哪些工作被错误地塞进了单线程调度器。
2. GPU 侧哪些状态必须串行化，哪些工作必须并行化。
3. 这套 GPU 运行时应如何和 `NNRequestLayer::submit() -> sender<NNEvalResult>` 对接。

本文件只讨论 GPU / TensorRT 侧，不重新讨论搜索协程协议。

### 2. 当前实现的真实问题

当前基线里，TensorRT fast path 的主控制点仍然是 [nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp) 里的 `NNEvaluator::serveTrtScheduler()`。

从代码上看，这个单线程同时承担了以下工作：

- 维护全局 `SchedulerState`
- 从 `queryQueue` 取 request
- 选择 open batch 的目标 slot / buffer
- 为每一行 request 选择 symmetry 并执行 `trtPackInputRow(...)`
- 为每一行提交 `trtEnqueueInputRowCopy(...)`
- 查询 H2D / infer / D2H CUDA event
- 决定何时 `trtLaunchInferenceAsync(...)`
- 在 batch 完成后逐行执行 `trtUnpackOutputRow(...)`
- 逐行 publish 结果，唤醒等待方

对应当前代码位置主要是：

- [nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L490)
- [nneval.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.h#L277)
- [trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L3143)

这带来的问题不是“只有一个地方做决策”，而是“同一个地方既做控制面，又做 CPU 数据面”。

具体瓶颈有四类：

1. 同一线程串行执行逐行 pack / unpack，无法利用多核 CPU。
2. 多 GPU / 多 slot 的状态都耦合在一个循环里，容易出现跨设备 head-of-line blocking。
3. 调度线程一边忙于数据面 CPU 工作，一边还要负责 CUDA event 轮询和 batch 决策，控制面延迟被数据面污染。
4. GPU completion 的“更新时间估计 -> 发布新目标 -> 唤醒等待协程”时序不够独立，后续很难自然衔接搜索协程架构。

### 3. 从 `v0 / v1` 继承的硬约束

GPU 侧设计必须同时满足以下约束：

- TensorRT-only fast path 允许存在，而且可以 backend-specific。
- 搜索协程仍然只有一个 yield 点，GPU 侧不能再引入第二个搜索侧挂起点。
- GPU completion 对搜索侧的时序必须是：
  1. 更新 GPU 侧时间估计
  2. 重算当前窗口的绝对目标水位线
  3. 再完成等待中的 sender
- 共享区采用全局累计计数模型：
  - `issuedNNEvalCount`
  - `completedNNEvalCount`
  - `targetIssuedNNEvalCount`
- GPU 侧不应再维护“局部 outstanding target”这类容易和超发打架的语义。

### 4. 总体架构结论

本版冻结一个核心结论：

- GPU 侧必须拆成“串行控制面 + 并行数据面”

也就是说：

- 控制面只负责状态机与时序
- 数据面负责真正吃 CPU 的逐行工作

不再允许出现“一个调度线程顺手把 pack/unpack/publish 全都做了”的结构。

### 5. 顶层分层

本版把 GPU 侧拆成五层：

1. `TensorRTRuntime`
2. `GpuDeviceRuntime`
3. `InferSlotRuntime`
4. `BatchArena`
5. `PackExecutor / PostExecutor`

对应关系如下：

```text
NNRequestLayer::submit(request)
  -> TensorRTRuntime
       -> GpuDeviceRuntime[gpu]
            -> GpuControlScheduler   (串行控制面)
            -> InferSlotRuntime[*]   (每个 TRT execution context / stream 组)
            -> BatchArena            (共享 buffer 池)
            -> PackExecutor          (并行逐行 pack/H2D)
            -> PostExecutor          (并行逐行 unpack/result finalize)
```

### 6. 各模块职责

#### 6.1 `TensorRTRuntime`

这是 GPU 侧顶层拥有者。

职责：

- 读取配置，创建所有 canonical GPU 的 runtime
- 暴露给 `NNRequestLayer` 的 backend submit 接口
- 负责关闭、drain、错误传播

它不负责：

- 逐行 pack
- 逐行 unpack
- slot 级 batch 决策细节

#### 6.2 `GpuDeviceRuntime`

每个 canonical GPU 一个 `GpuDeviceRuntime`。

它是 GPU 侧真正的控制面中心，但必须是“轻控制面”，不是“单线程大管家”。

职责：

- 维护该 GPU 的 `TimingModel`
- 维护该 GPU 的 open batch / launched batch / pending publish 状态
- 选择 buffer、选择 slot、决定何时 launch
- 在 batch 完成时更新：
  - `completedNNEvalCount`
  - `targetIssuedNNEvalCount`
  - `epoch`
- 保证 sender completion 的最终顺序

它必须串行化的内容只有：

- buffer 生命周期状态迁移
- slot 状态迁移
- 时间估计更新
- 当前窗口绝对目标水位线发布
- request sender completion

它不应该再直接做逐行 CPU 重活。

#### 6.3 `InferSlotRuntime`

每个 TensorRT execution context / infer stream 组三个关键资源：

- H2D stream
- infer stream
- D2H stream

外加该 slot 自己的 execution context / registered shared buffer state。

职责：

- 承载单个 slot 上的 GPU 执行资源
- 为 `BatchArena` 提供可以 launch 的实际执行位
- 只暴露 slot-ready / infer-done / d2h-done 这类事件给控制面

它不是 scheduler，也不是全局 runtime。

#### 6.4 `BatchArena`

`BatchArena` 统一管理共享输入输出 buffer 池。

每个 `BatchBuffer` 至少要经历以下状态：

- `Free`
- `Filling`
- `PackPending`
- `H2DPending`
- `ReadyToLaunch`
- `InferRunning`
- `D2HPending`
- `PostPending`
- `ReadyToPublish`

每个 buffer 需要明确拥有：

- 目标 GPU / slot
- 当前 row 数
- 每个 row 对应的 request handle
- row 级 pack 完成计数
- row 级 post 完成计数
- 当前 batch 的时间样本元数据

这样控制面只看状态，不必自己做每一行的 CPU 工作。

#### 6.5 `PackExecutor`

这是 GPU 侧第一类必须并行化的数据面。

职责：

- 接受“某个 request 已被分配到某个 buffer row”这一任务
- 执行逐行 pack / layout transform
- 提交对应 row 的 H2D copy
- 把 row-ready 事件回传给 `GpuDeviceRuntime`

本版明确：

- `PackExecutor` 可以用固定大小 CPU 线程池
- 它不是 search worker pool
- 它也不是全局匿名 work-stealing pool 的热路径核心

第一阶段可以接受 `exec::static_thread_pool`，因为这里不在搜索热路径里。

#### 6.6 `PostExecutor`

这是 GPU 侧第二类必须并行化的数据面。

职责：

- 在 D2H 完成后，对 batch 内每一行执行：
  - `trtUnpackOutputRow(...)`
  - 结果对象物化
  - 若需要，也包括 GPU 侧希望承担的轻量 finalize
- 将“row 已可 publish”的事件回传给 `GpuDeviceRuntime`

它的存在意义很直接：

- 不能再让单个控制线程串行 unpack 整个 batch

### 7. 哪些工作必须离开控制面

本版明确，以下工作必须离开 `GpuControlScheduler`：

- 逐行 `trtPackInputRow(...)`
- 逐行 `trtEnqueueInputRowCopy(...)` 的 CPU 提交准备
- 逐行 `trtUnpackOutputRow(...)`
- 逐行结果对象 finalize / 物化

换句话说，控制面最多只做：

- 分配任务
- 收集完成
- 迁移状态
- 做最终发布

### 8. 哪些工作必须继续串行

以下内容不能被随意并行化，必须在 `GpuDeviceRuntime` 的控制 lane 上串行：

- open batch 选择
- buffer 与 slot 的绑定
- “GPU 空闲时立即发射，GPU 忙时可等待更合适 batch”这一规则的执行
- batch 生命周期状态迁移
- 时间模型更新
- 绝对目标水位线发布
- sender completion 的最终顺序

否则会出现：

- 同一个 buffer 被多个 slot 抢占
- 同一个 slot 被并发 launch
- 时间估计和目标发布顺序错乱

### 9. 与 `stdexec` 的映射

#### 9.1 公共异步边界

GPU 侧对外只暴露一类主要边界：

- `NNRequestLayer::submit(...) -> sender<NNEvalResult>`

这条 sender 的完成语义是：

- 请求已经拥有可直接被搜索协程消费的结果

搜索协程不应感知：

- batch buffer
- slot
- CUDA event
- row pack / row unpack

#### 9.2 内部 scheduler

GPU 侧内部至少需要三类 scheduler：

1. `GpuControlScheduler`
2. `PackScheduler`
3. `PostScheduler`

其中：

- `GpuControlScheduler`
  - 每个 canonical GPU 一个
  - 语义必须是单 lane 串行
- `PackScheduler`
  - 固定大小 CPU 池
  - 只跑逐行 pack/H2D 准备
- `PostScheduler`
  - 固定大小 CPU 池
  - 只跑逐行 unpack/finalize

`receiver / operation_state` 仍然只允许留在 sender 实现内部，不上升成公共模块接口。

### 10. 关键设计决定

#### 10.1 symmetry 不再属于 GPU 控制面

当前基线中，scheduler 线程仍然会在接收 request 后决定 symmetry。

这在新架构里应被移走。

本版建议：

- symmetry 在进入 GPU runtime 之前就冻结
- 即由 `NNRequestLayer` 或更上游的 request 物化路径完成

原因是：

- symmetry 是请求语义，不是 GPU 调度语义
- 把它留在 GPU 控制面，只会扩大控制线程负担

#### 10.2 GPU 侧时间模型应对齐“真正可唤醒搜索”的完成点

如果只记录 raw infer kernel 时间，而忽略：

- H2D
- D2H
- row unpack / finalize

那么 GPU 侧发布给搜索侧的目标水位线会偏乐观。

因此本版建议：

- `TimingModel` 至少维护一份 `batchServiceMs`
- 它表示“从 batch 发射到该 batch 全部 row 可 publish”的端到端时间

同时可以保留诊断项：

- `h2dMs`
- `inferMs`
- `d2hMs`
- `rowPostMs`

但真正驱动需求模型的，优先应是 `batchServiceMs`。

### 11. 统一时序

单个 batch 的理想时序如下：

1. `GpuDeviceRuntime` 选择 open batch 的目标 slot / buffer
2. request 被分配 row
3. `PackExecutor` 并行完成逐行 pack，并提交 H2D
4. 控制面在规则允许时 launch batch
5. infer 完成
6. D2H 完成
7. `PostExecutor` 并行完成逐行 unpack/finalize
8. 当 batch 的所有 row 都 ready-to-publish 时，控制面执行：
   - 更新 `TimingModel`
   - 递增 `completedNNEvalCount`
   - 计算新的 `targetIssuedNNEvalCount`
   - 发布新的 `epoch`
9. 最后才 complete 等待中的 request sender

这条时序满足 `v0 / v1` 的核心要求：

- 先更新估计
- 再发布新目标
- 再唤醒搜索协程

### 12. 多 GPU 扩展结论

本版明确反对继续保留“一个全局 scheduler 线程照看所有 GPU”的结构。

正确扩展方向应是：

- 一个 canonical GPU 一个 `GpuDeviceRuntime`
- 每个 GPU 自己维护：
  - 控制 lane
  - slot 集合
  - buffer 池
  - 时间模型
  - pack/post 数据面

这样：

- CPU 数据面不会跨 GPU 串行化
- 控制面也不会因为别的 GPU 的 pack/unpack 压力而延迟

### 13. 迁移顺序

第一阶段建议按以下顺序落地：

1. 从 `NNEvaluator::serveTrtScheduler()` 中抽出 `TensorRTRuntime / GpuDeviceRuntime` 的显式状态对象。
2. 保持控制面仍然串行，但先把逐行 pack 从控制线程迁走。
3. 再把逐行 unpack / finalize 从控制线程迁走。
4. 把“batch ready-to-publish 之后才更新目标并 complete sender”的时序固定下来。
5. 最后再把当前全局 scheduler thread 拆成 per-GPU control lane。

这样做的原因是：

- 先切掉最重的 CPU 数据面，最容易获得确定性的收益
- 再拆控制面拓扑，风险更低

### 14. 本版冻结的结论

本版正式冻结以下结论：

1. 当前单线程 TensorRT scheduler 的根本问题，不是“它是单线程”，而是“它混合了承担控制面和 CPU 数据面”。
2. GPU 侧必须拆成 per-GPU 串行控制面 + 并行 pack/post 数据面。
3. 搜索侧看到的仍然只是一条 `submit() -> sender<NNEvalResult>` 边界。
4. GPU completion 的模型更新点，应对齐“batch 全部 row 可 publish”的时刻，而不是只看 raw infer kernel 结束。
5. 目标发布继续采用全局累计计数模型，而不是回退到局部 outstanding target。

### 15. `v2` 需要继续细化的问题

下一版文档需要继续回答：

- `GpuControlScheduler` 最终是 dedicated thread、serial executor，还是更轻的 custom scheduler
- `PackExecutor` / `PostExecutor` 应该按 GPU 分池，还是做全局共享池
- `BatchBuffer` 的最小状态集合是什么，哪些状态可以合并
- `NNEvalResult` 的 finalize 有多少应该留在 GPU 侧，多少应回到搜索协程恢复后执行
- 当前 `trtPackInputRow / trtUnpackOutputRow` 的 CPU 代价是否需要进一步拆成更细颗粒
