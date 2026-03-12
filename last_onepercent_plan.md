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

### 2026-03-10: 本机同步基线（用于判断“多一个线程 handoff 值不值”）

测试环境：

- OS: Ubuntu 24.04, Linux 6.14
- CPU: AMD Ryzen 9 9900X
- GPU: GeForce RTX 5080
- CUDA toolchain: nvcc 13.1
- Driver/runtime: 580.126.09 / CUDA 13.0

#### A. `pthread_cond` + mutex 的线程 handoff 开销

方法：

- 两个线程固定到指定 CPU。
- 使用单个 `pthread_mutex_t` + `pthread_cond_t` 做 ping-pong。
- 主线程 signal 对端并等待对端 signal 回来。
- 记录单次 round-trip 时间，one-way handoff 近似取 round-trip / 2。
- 这是空闲机器上的“低负载下限量级”，不是生产最坏值。

结果：

- CPU0 <-> CPU1:
  - round-trip mean `2.71 us`
  - one-way mean `1.36 us`
  - one-way p50 `1.23 us`
  - one-way p95 `1.84 us`
  - one-way p99 `2.22 us`
- CPU0 <-> CPU2:
  - one-way mean `1.85 us`
  - one-way p50 `1.66 us`
  - one-way p95 `2.90 us`
  - one-way p99 `3.09 us`
- CPU0 <-> CPU12 (SMT sibling):
  - one-way mean `1.81 us`
  - one-way p50 `1.83 us`
  - one-way p95 `2.47 us`
  - one-way p99 `2.94 us`

保守结论：

- 在这台机器上，“一次条件变量唤醒 + 锁交接”的典型代价是 `~1.3-1.9 us`，p95 常见在 `~1.8-2.9 us`。
- 极端尾延迟会跳到几十微秒甚至毫秒级，这是调度器/中断/系统噪声，不适合作为 steady-state 设计点，但必须在 tail-latency 预算里留余地。
- 如果一个热路径为了 overlap 需要额外引入 2 次线程 handoff，那么光同步本身通常就会吃掉 `~2.5-4 us`，这已经不是可以忽略的量级。

#### B. `cudaGraphLaunch` 的 CPU 侧重复 launch 开销

方法：

- 图先 instantiate，再显式 `cudaGraphUpload()`，避免把首次上传成本混进重复 launch。
- 每次测量只统计 host 线程待在 `cudaGraphLaunch()` 调用里的时间。
- launch 完后再 `cudaStreamSynchronize()`，仅用于保证下一次 launch 看到空 stream；同步时间不计入结果。
- 额外测了普通空 kernel launch，作为对照。

结果：

- 普通空 kernel launch:
  - mean `1.68 us`
  - p50 `1.60 us`
  - p95 `2.08 us`
- `cudaGraphLaunch`, 1-node 直线图:
  - mean `0.98 us`
  - p50 `0.97 us`
  - p95 `1.04 us`
- `cudaGraphLaunch`, 8-node 直线图:
  - mean `1.05 us`
  - p50 `1.01 us`
  - p95 `1.25 us`
- `cudaGraphLaunch`, 32-node 直线图:
  - mean `1.08 us`
  - p50 `1.02 us`
  - p95 `1.29 us`
- `cudaGraphLaunch`, 128-node 直线图:
  - mean `1.06 us`
  - p50 `1.02 us`
  - p95 `1.29 us`

与 NVIDIA 官方资料的关系：

- NVIDIA 2024 年关于 CUDA Graphs 的文章给出的 Ampere + CUDA 12.6 repeat launch CPU overhead 量级是“约 `2.5 us + ~1 ns/node`（直线图，10 node 以上）”。
- 这台 Blackwell + 更新栈上的本机结果更快，且在 1 到 128 node 内几乎常数，可视作“当前开发机上重复 graph launch 大约 `1 us`”。
- 这里测的是 repeat launch，不包括 instantiate、首次 launch/upload、graph update 等一次性或非稳态成本。

保守结论：

- 在这台机器上，一个“已经 upload 的 graph 的重复 launch”大约只要 `~1 us` CPU 时间，和一次 `pthread_cond` one-way handoff 是同一个量级，通常还更便宜。
- 如果 overlap 方案需要“多一个协调线程 + 条件变量唤醒”，那它节省下来的 CPU launch 预算必须至少覆盖 `~1-3 us` 级别的额外同步，才有讨论价值。
- 这进一步支持一个偏保守的方向：优先在“同一拥有线程内减少 launch 开销、拉直提交路径、用 event 做可观测性”，谨慎引入跨线程 handoff。

可复现性说明：

- 这组 benchmark 代码当前只临时放在 `/tmp/condvar_bench.cpp` 和 `/tmp/cudagraph_launch_bench.cu`，尚未入仓库。
- 如果这些数字会反复用来做决策，后续应把 benchmark 整理成仓库内的可复现工具，并把“空闲 / 轻载 / 压测中”三种状态都测一遍。

Primary sources:

- NVIDIA Technical Blog, Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements:
  https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/

### 2026-03-10: 同一 host thread 向不同 device 提交 `cudaGraph` 的 overhead 估算

问题定义：

- 场景是“一个 host thread 轮流向 device 0、device 1、... 的各自 stream 提交已经 instantiate 好的 `cudaGraphExec_t`”。
- 这里讨论的是 steady-state repeat launch，不包括每张卡第一次触达时的 runtime/context 初始化。
- 这里讨论的是“每张卡各有自己的 graph exec + stream”，不是“一个大 graph 跨多个 device 一次 launch”。

已知项：

1. 官方明确说明 `cudaSetDevice()` 不会同步前一个或新 device。
2. 官方明确说明 kernel launch 必须发到当前 device 关联的 stream；因此保守做法仍然是先 `cudaSetDevice(d)`，再 `cudaGraphLaunch(exec_d, stream_d)`。
3. 本机实测“已 upload 的 graph 的 repeat `cudaGraphLaunch()` CPU 侧 API 时间”大约是 `~1.0-1.1 us`。
4. 本机实测“hot `cudaSetDevice(0)` 重复调用”大约是 `~31 ns`/call；第一次触达 device 的冷启动约 `98 ms`。这个 `31 ns` 只是“同 device no-op set”的下界，不等于“在多个已初始化 device 之间切换”的真实开销。

缺失项：

- 我目前没有找到 NVIDIA 官方给出的“多个已初始化 device 之间来回 `cudaSetDevice()` 的 steady-state 定量 overhead”。
- 我也没有多卡本机实测来直接测“thread 在 device 0 / 1 间轮流 graph launch”的真实值。

因此可用的估算公式是：

`T_submit(N devices) ~= N * T_graph_launch + (N - 1) * T_device_switch_hot`

其中：

- `T_graph_launch`：
  - 本机 Blackwell + CUDA 13 栈：约 `1.0 us`
  - NVIDIA 2024 博文中的 Ampere + CUDA 12.6 量级：约 `2.5 us + ~1 ns/node`
- `T_device_switch_hot`：
  - 已知它不包含设备同步。
  - 已知它在“同 device 热路径 no-op”时低到 `~31 ns`。
  - 真实“device A <-> B 切换”会比 `31 ns` 高，但从语义上看它更像“host thread current context/current device 的切换”，不应接近毫秒级冷启动。

保守规划数值：

- 如果只是做架构决策，在没有多卡实测前，可以先把 `T_device_switch_hot` 预算成 `0.2-1.0 us`。
- 则单线程轮流向不同卡提交 graph 的 steady-state host 开销可以先估成：
  - 2 张卡：`~2.2-3.0 us`
  - 4 张卡：`~4.6-7.0 us`
  - 8 张卡：`~9.4-15.0 us`
- 如果你想更保守，直接按“每张卡一次提交约 `2-4 us` host 时间”做上界预算也可以。

对决策的含义：

- 只要所有 device 都已经完成预热，这件事大概率不是“会不会因为 `cudaSetDevice()` 爆炸”的问题，而是“一个 thread 顺序提交这么多 graph，host 提交带宽够不够”的问题。
- 如果目标只是轮流给 2 张或 4 张卡喂已经 upload 的 graph，那么单线程提交在 steady-state 下看起来仍然是可行的，host 侧预算大致是个位数微秒。
- 真正不能忽视的是每张卡第一次触达时的冷启动，必须提前做在非热路径。
- 如果 later pipeline 为了这件事引入额外线程 handoff，那么线程 handoff 自身的 `~1-3 us` 量级，很可能并不比“多 device 顺序 graph launch”更便宜。

推荐的工程预算：

- 2 卡：先按 `3 us` 总 host submit 开销估。
- 4 卡：先按 `7 us` 总 host submit 开销估。
- 8 卡：先按 `15 us` 总 host submit 开销估。
- 这些数值适合拿来做“值不值得再拆线程”的第一轮决策，不适合替代最终微基准。

后续最需要补的实测：

- 双卡机器上直接测：
  - `set(0) -> launch(g0) -> set(1) -> launch(g1)` 的循环成本
  - 对比“两线程各守一张卡”的成本
  - 对比“graph 已 upload”和“未 upload 首次 launch”

### 2026-03-10: 远端双卡服务器实测，同一线程交替向两张卡提交 `cudaGraph`

测试环境：

- Host: `10.101.3.169` (`BulletTime-6GPU`)
- OS: Ubuntu 24.04, Linux 6.17
- CPU: AMD Ryzen Threadripper PRO 7985WX
- GPU: 6 x GeForce RTX 4090 D
- 本次仅使用物理卡 2 和 3：通过 `CUDA_VISIBLE_DEVICES=2,3` 限定，程序内把它们视为 visible device 0 和 1
- CUDA toolkit: `/usr/local/cuda-12.6`

方法：

- benchmark 源文件与二进制都只放在远端 `/tmp`。
- 单个 host thread 固定到一个 CPU core。
- 每张卡各自创建 2 个 non-blocking stream，并各自 instantiate + upload 两个 graph exec。
- graph 是直线 kernel-node graph；分别测了 `1 node` 和 `32 nodes`。
- 每次计时只覆盖 submit 路径：
  - 单卡基线：`cudaGraphLaunch(g0)`
  - 同卡双提交：`set(0) -> launch(g0a) -> set(0) -> launch(g0b)`
  - 双卡交替提交：`set(0) -> launch(g0) -> set(1) -> launch(g1)`
  - 纯切卡：`set(0) -> set(1)`
- 每轮计时后才做 `cudaStreamSynchronize()` 清空 stream，因此结果代表 CPU 侧 submit overhead，而不是 GPU 执行时间。

结果：1-node graph

- 单卡 `cudaGraphLaunch`:
  - card 2: `1.104 us`
  - card 3: `1.097 us`
- 同卡双提交:
  - `2.287 us` total
  - `1.144 us` per launch
- 双卡交替提交:
  - `2.287 us` total
  - `1.144 us` per launch
- 纯热路径 `cudaSetDevice(0) -> cudaSetDevice(1)`:
  - `0.087 us` per pair
  - `0.044 us` per set
- 交替双卡相对同卡双提交的增量：
  - `~0.000 us` per pair
  - 在测量噪声内可视为 0

结果：32-node graph

- 单卡 `cudaGraphLaunch`:
  - card 2: `1.261 us`
  - card 3: `1.272 us`
- 同卡双提交:
  - `2.600 us` total
  - `1.300 us` per launch
- 双卡交替提交:
  - `2.619 us` total
  - `1.310 us` per launch
- 纯热路径 `cudaSetDevice(0) -> cudaSetDevice(1)`:
  - 仍为 `0.087 us` per pair
  - `0.044 us` per set
- 交替双卡相对同卡双提交的增量：
  - `0.019 us` per pair
  - `0.010 us` per launch

直接结论：

- 在这台 4090 + CUDA 12.6 服务器上，如果 graph 已经 instantiate + upload，单线程在两张卡之间交替提交 graph 的 CPU 开销，与“在同一张卡上连续提交两个 graph”几乎没有可测差别。
- 这里真正占主导的是 `cudaGraphLaunch()` 自身的 `~1.1-1.3 us`；热路径 `cudaSetDevice(0<->1)` 只有 `~44 ns`/call，低到基本可以忽略。
- 对 2 卡场景，steady-state host submit 可以直接按“每次 graph launch 约 `1.1-1.3 us`，跨卡切换税约 `0.0-0.05 us`”做预算。

对架构决策的含义：

- 如果只是“同一线程轮流喂两张卡”，在这台机器上没有证据表明 `cudaSetDevice()` 会成为瓶颈。
- 这意味着“为了避免切卡而额外拆线程”的收益门槛更高，因为线程 handoff 本身通常比这里测到的切卡税贵一个数量级。
- 至少对 2 卡而言，更应优先关注：
  - graph 是否能稳定复用
  - H2D / D2H / 后处理是否挡住了 overlap
  - 现有 owning thread 是否已经足够顺滑地持续提交

边界与注意：

- 这些结果只覆盖 steady-state repeat launch，不含首次 context 初始化、首次 graph upload、graph update。
- 这些结果只证明了 2 张 4090 D、当前驱动和内核组合下的行为，不能直接外推到所有多卡机器。
- benchmark 只测 CPU submit 路径；如果真实系统的瓶颈在 pinned-memory copy、allocator、queue contention、postprocess，那么这里的结果不会替代那些测量。

### 2026-03-11: 真实 TRT plan 的单-context `cudaGraphLaunch()` 开销

目标：

- 只测一个 TensorRT execution context 上，steady-state `cudaGraphLaunch()` 的 CPU submit overhead。
- 不测多 context。
- 不测 capture / instantiate / upload。

测试对象：

- plan cache:
  `~/.katago/trtcache/trt-onnx-101501_olv-def_gpu-426b8a57_net-4179b6b29c90_9_exact19x19_batch7_fp16`

方法：

- 写了一个最小测试 `/tmp/trt_graph_launch_min.cpp`。
- 实现上尽量贴近 [trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp)：
  - 从 cache 文件中读取 plan，并按当前 cache 格式剥掉尾部附加的 64-byte model hash。
  - 使用 `cudaStreamPerThread`。
  - `setOptimizationProfileAsync(0, cudaStreamPerThread)`。
  - 对动态输入先取 profile 0 的 `kOPT` shape 做 `setInputShape()`。
  - 为所有 IO tensor 绑定 device buffer。
  - 先 `enqueueV3()` 一次并同步，再 capture。
  - `cudaStreamBeginCapture` -> `enqueueV3` -> `cudaStreamEndCapture` -> `cudaGraphInstantiate` -> `cudaGraphUpload`。
  - 之后只测 steady-state `cudaGraphLaunch()` 调用本身的 host 时间；每轮后 `cudaStreamSynchronize()`，仅用于清空 stream。
- 参数：
  - warmup `20`
  - iterations `100`

结果：

- `graph_launch_repeat_mean_us = 1.219`
- `graph_launch_repeat_p50_us = 1.152`
- `graph_launch_repeat_p95_us = 1.473`
- `graph_launch_repeat_p99_us = 1.793`

直接结论：

- 对这份真实 TRT plan，在当前本机环境上，单个 context 的 steady-state `cudaGraphLaunch()` CPU submit overhead 可以先按 `~1.2 us` 估。
- p95 大约 `~1.5 us`，p99 大约 `~1.8 us`。
- 这个量级和前面空 graph / 小 graph benchmark 的 `~1 us` 是一致的，说明“换成真实 TRT plan 后 launch 本身没有突然膨胀到几微秒以上”。

边界与注意：

- 样本只有 `100` 次，这足够回答“量级是不是 `~1 us`”，但不足以做严肃尾延迟分析。
- 这组数不包含 graph capture、instantiate、upload，也不包含首次 launch 的任何冷路径成本。
- 这组数只回答“launch 会不会阻塞关键路径到很夸张的程度”；不回答真实端到端推理 latency。

### 2026-03-11: `trtbackend.cpp` 结构备查

目的：

- 把当前 TensorRT backend 的职责边界和关键路径落成文字，避免后面做 overlap 设计时反复翻整文件。
- 文件：
  [cpp/neuralnet/trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp)
- 当前长度：
  `2602` 行。

按顺序的结构拆分：

- [trtbackend.cpp:1](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1) 到 [trtbackend.cpp:89](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L89)
  - TensorRT/CUDA 头文件、基础工具函数、`CudaSyncMode` 和 `TrtTilingOptimizationLevel` 映射。
  - 还定义了按 GPU 记录 launch interval 的全局状态，服务于性能 profiling，而不是推理功能本身。

- [trtbackend.cpp:98](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L98) 到 [trtbackend.cpp:198](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L198)
  - `globalInitialize/globalCleanup` 对 TRT backend 基本是空实现。
  - `ComputeContext` 只保存配置，不持有 GPU runtime 对象。
  - `LoadedModel` 负责读模型描述：
    - ONNX 走 `ModelDesc::loadFromONNX`
    - 非 ONNX 走老的模型描述加载，并做 `applyScale8ToReduceActivations()`

- [trtbackend.cpp:206](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L206) 到 [trtbackend.cpp:1074](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1074)
  - 这是 `TRTModel` + `ModelParser`，也是整文件最重的一段。
  - 它不是 runtime，而是在“手工把 KataGo 模型翻译成 TensorRT network definition”。
  - 关键子段：
    - [trtbackend.cpp:250](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L250)
      `ModelParser::build()` 总入口，依次调用 `initInputs()`、`initMaskProcLayers()`、`buildTrunk()`、`buildPolicyHead()`、`buildValueHead()`。
    - [trtbackend.cpp:338](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L338)
      `initInputs()` 定义输入 tensor 和 optimization profile。
    - [trtbackend.cpp:415](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L415)
      `initMaskProcLayers()` 把棋盘 mask 处理成 `maskSum`、`maskScale`、`maskQuad` 等后续 gpool/value head 依赖的特征。
    - [trtbackend.cpp:490](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L490)
      `buildTrunk()` 构建初始 conv/global/meta 分支、block stack、trunk tip。
    - [trtbackend.cpp:548](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L548)
      `buildResidualBlockStack()` 只是 dispatcher，按 block 类型分发到不同 block builder。
    - [trtbackend.cpp:575](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L575)
      `buildPolicyHead()` 产出 `OutputPolicyPass` 和 `OutputPolicy`。
    - [trtbackend.cpp:640](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L640)
      `buildValueHead()` 产出 `OutputValue`、`OutputScoreValue`、`OutputOwnership`。
    - [trtbackend.cpp:697](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L697) 到 [trtbackend.cpp:1073](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1073)
      是各种 layer translator/helper：
      - metadata encoder
      - 普通残差块
      - global-pooling 残差块
      - nested bottleneck 残差块
      - matmul/bias/conv/batchnorm/activation
      - `applyGPoolLayer()`
      - `applyMaskLayer()`
      - `applyCastLayer()`

- [trtbackend.cpp:1076](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1076) 到 [trtbackend.cpp:1174](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1174)
  - `TRTLogger` 和 `TRTErrorRecorder`。
  - 作用不只是打印日志，也负责把部分已知致命错误直接升级为 `fatalError`。

- [trtbackend.cpp:1177](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1177) 到 [trtbackend.cpp:1883](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1883)
  - 这是 `ComputeHandle`，也就是每个 GPU/线程实际持有的 TensorRT runtime 对象。
  - 成员里有：
    - `runtime`
    - `engine`
    - `exec`
    - 所有 device buffer
    - `batchGraphStates`
    - 性能 profiling event
  - 构造函数内部大致分几段：
    - [trtbackend.cpp:1248](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1248) 到 [trtbackend.cpp:1280](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1280)
      builder/config 创建，FP16/INT8 flag 选择。
    - [trtbackend.cpp:1292](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1292) 到 [trtbackend.cpp:1337](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1337)
      ONNX 路径直接 parse；非 ONNX 路径走 `ModelParser`。
    - [trtbackend.cpp:1339](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1339) 到 [trtbackend.cpp:1369](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1369)
      tactic source、builder optimization level、aux streams、timing iterations、tiling level、workspace、profile stream。
    - [trtbackend.cpp:1372](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1372) 到 [trtbackend.cpp:1606](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1606)
      plan cache / timing cache 读写。
      这里很重要的一点是：
      当前 cache 文件不是纯 plan，保存时会把 `modelHashStr` 和 `paramStr` 追加到 plan 尾部，读取时再剥掉。
    - [trtbackend.cpp:1609](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1609) 到 [trtbackend.cpp:1637](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1637)
      runtime/engine/context 创建，给每个 IO tensor 分配 device buffer，并 `setTensorAddress()`。
    - [trtbackend.cpp:1639](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1639) 到 [trtbackend.cpp:1641](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1641)
      如果启用 `trtUseCudaGraph`，在初始化阶段直接预捕获所有 batch size 的 graph。
  - `ComputeHandle` 里和 overlap 直接相关的 helper：
    - [trtbackend.cpp:1663](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1663)
      `maybeRecordLaunchInterval()`，纯 profiling。
    - [trtbackend.cpp:1769](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1769)
      `setInputShapesForBatch()`，按 batch 设置动态输入 shape。
    - [trtbackend.cpp:1800](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1800)
      `captureBatchGraph()`，对某个 batch size 做 `enqueueV3` capture + instantiate。
    - [trtbackend.cpp:1845](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1845)
      `preCaptureAllBatchGraphs()`，把 `1..maxBatchSize` 的 graph 都预先做好。
    - [trtbackend.cpp:1864](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1864)
      `enqueueWithOptionalCudaGraph()`，热路径最终在这里二选一：
      - `exec->enqueueV3(cudaStreamPerThread)`
      - `cudaGraphLaunch(state.graphExec, cudaStreamPerThread)`

- [trtbackend.cpp:1885](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1885) 到 [trtbackend.cpp:1962](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1962)
  - 对外的 handle 工厂和辅助函数。
  - `createComputeHandle()` 里先：
    - `cudaSetDevice(gpuIdxForThisThread)`
    - `cudaSetDeviceFlags(...)`
    - `cudaGetDeviceProperties(...)`
  - 然后才真正构造 `ComputeHandle`。
  - 也就是说，当前设计里“哪个线程拥有哪个 GPU 的 TRT context”是在这里固定下来的。

- [trtbackend.cpp:1964](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1964) 到 [trtbackend.cpp:2163](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2163)
  - `InputBuffers`，负责 host 侧 pinned buffer。
  - 同时兼容两套路径：
    - 非 ONNX 的原生 TRT 输出布局
    - ONNX 的 `out_policy/out_value/out_miscvalue/out_moremiscvalue/out_ownership` 输出布局
  - 这里一次性算好每种张量的元素数和字节数，并分配所有 pinned host buffer。

- [trtbackend.cpp:2173](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2173) 到 [trtbackend.cpp:2513](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2513)
  - `getOutput()`，这是当前最接近端到端关键路径的一段。
  - 可以再拆成 5 段：
    - [trtbackend.cpp:2219](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2219) 到 [trtbackend.cpp:2242](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2242)
      CPU 侧预处理：把 `NNResultBuf` 拷到 pinned host buffer，做 symmetry 展开，并从 spatial 输入里复制 mask。
    - [trtbackend.cpp:2255](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2255) 到 [trtbackend.cpp:2293](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2293)
      H2D + `setInputShape()`。
    - [trtbackend.cpp:2297](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2297) 到 [trtbackend.cpp:2300](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2300)
      launch 点：先记 launch interval，再调 `enqueueWithOptionalCudaGraph(batchSize)`。
    - [trtbackend.cpp:2302](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2302) 到 [trtbackend.cpp:2318](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2318)
      D2H，然后立刻 `cudaStreamSynchronize(cudaStreamPerThread)`。
    - [trtbackend.cpp:2337](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2337) 到 [trtbackend.cpp:2488](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2488)
      host 侧后处理：把 policy/value/ownership/score 等输出解包进 `NNOutput`。
  - 对 overlap 最关键的现状结论：
    - 整个推理提交路径当前固定使用 `cudaStreamPerThread`。
    - H2D、launch、D2H 都在同一个函数里串起来。
    - `getOutput()` 末尾直接 `cudaStreamSynchronize()`，所以当前 API 边界是同步边界，不向上层暴露“已发出但未完成”的状态。

- [trtbackend.cpp:2515](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2515) 到文件末尾
  - 几个 `testEvaluate*` stub，当前基本都是空实现返回 `false`。
  - 不是主推理路径。

对 overlap 设计最值得盯住的切口：

- [trtbackend.cpp:1905](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1905)
  当前 GPU owning thread 的确定点。
- [trtbackend.cpp:1639](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1639) 和 [trtbackend.cpp:1845](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1845)
  当前 graph 生命周期入口，说明 graph 是“每个 batch size 预捕获一份”。
- [trtbackend.cpp:2194](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2194)
  当前 stream 选择被硬编码为 `cudaStreamPerThread`。
- [trtbackend.cpp:2298](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2298)
  当前真正的 TRT submit 点。
- [trtbackend.cpp:2318](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2318)
  当前最强的同步边界，也是以后如果要做 H2D/infer/D2H overlap，最可能需要松动的点。

### 2026-03-11: 用 event 把 IO stream 上的 H2D 串到 compute stream 上的 `cudaGraphLaunch`

目标场景：

- H2D 拷贝发在一个 IO 专用 stream `ioStream` 上。
- 推理 graph replay 发在另一个 compute stream `computeStream` 上。
- 希望 graph 在 H2D 完成后立刻开始。
- 不希望 H2D 完成时还要唤醒 CPU 再显式做一次“交还控制权”。

官方语义结论：

- 可以，推荐做法就是：
  - 在 `ioStream` 上 `cudaMemcpyAsync(...)`
  - 随后在 `ioStream` 上 `cudaEventRecord(doneEvent, ioStream)`
  - 在 `computeStream` 上 `cudaStreamWaitEvent(computeStream, doneEvent, 0)`
  - 然后直接把 `cudaGraphLaunch(graphExec, computeStream)` 排到 `computeStream`
- 这样依赖是由 GPU/driver 按 stream 顺序和 event 依赖来维护的，不需要等 event 完成后再由 CPU 补一个提交动作。
- `cudaGraphLaunch()` 本身就是提交到 `computeStream` 的一项 CUDA work，因此它会排在这个 stream 里前面的 `cudaStreamWaitEvent()` 之后执行。

推荐模式：

```cpp
cudaEvent_t h2dDone;
cudaEventCreateWithFlags(&h2dDone, cudaEventDisableTiming);

cudaMemcpyAsync(dInput, hInput, bytes, cudaMemcpyHostToDevice, ioStream);
cudaEventRecord(h2dDone, ioStream);

cudaStreamWaitEvent(computeStream, h2dDone, 0);
cudaGraphLaunch(graphExec, computeStream);
```

关键边界：

- `cudaStreamWaitEvent()` 等的是“该 event 最近一次被 `cudaEventRecord()` 记录到的那批先前工作”。
- 因此顺序必须是：
  - 先在 `ioStream` 上把 H2D 和 `cudaEventRecord()` 排进去
  - 再在 `computeStream` 上调用 `cudaStreamWaitEvent()`
  - 最后再 `cudaGraphLaunch()`
- 不能指望“先 wait，后 record”去自动等未来的一次 record。

官方意义上的最佳实践：

- 如果只是想表达 stream 之间的先后依赖，优先用 `cudaStreamWaitEvent()`，不要在热路径里用：
  - `cudaEventSynchronize()`
  - `cudaStreamSynchronize()`
  - host callback
- 如果 event 不用于测时，只用于排序，创建时加 `cudaEventDisableTiming`。
- H2D 要和 compute 真正 overlap，host buffer 需要是 pinned memory。
- 尽量使用非 default stream；不要把这种依赖链建在 legacy default stream 语义上。
- 如果 memcpy 的地址、大小、拓扑都稳定，后续可以考虑把 memcpy 也直接并入同一个 CUDA graph；否则“外部 H2D + event + `cudaStreamWaitEvent` + `cudaGraphLaunch`”通常更灵活。

对当前项目的直接含义：

- 如果后续要把 H2D 从 [trtbackend.cpp:2173](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2173) 的主提交流程里拆出去，最自然的低侵入方案不是引入 CPU 线程 handoff，而是：
  - IO stream 做 `cudaMemcpyAsync`
  - record 一个 completion event
  - compute stream 先 `cudaStreamWaitEvent`
  - 再 `cudaGraphLaunch`
- 这样仍然只需要一次 host 侧顺序提交，不需要“等拷贝完成以后再回到 CPU 再发 graph”。
- 真正需要改的地方更像是：
  - 把当前固定的 `cudaStreamPerThread` 拆成至少两个 stream
  - 把 `getOutput()` 里串死的 H2D / launch / D2H 边界拆开
  - 把 `cudaStreamSynchronize()` 从中间热路径里挪出

Primary sources:

- CUDA Runtime API, `cudaStreamWaitEvent`:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- CUDA Runtime API, `cudaEventRecord`:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- CUDA Runtime API, `cudaGraphLaunch` and graph event wait/record nodes:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html
- CUDA C Best Practices Guide, overlap transfer/compute requires pinned memory and non-default streams:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CUDA C Programming Guide, stream/event synchronization model:
  https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html

### 2026-03-11: 单调度线程 + `global open batch` 工作流草案 v0

这节记录当前已经对齐的设计约束，避免后续讨论时再次把“global open batch”和“每请求 H2D”说乱。

背景约束：

- 当前实测历史见 [优化历史.md](/home/wangyize/.katago/KataGomo_fork/优化历史.md)：
  - 空 GPU 时允许 partial batch 发射是有收益的。
  - 非空 GPU 时，强行只发满 batch，能够把稳态 `gpu_batch_time_share` 收敛到 `b7=100%`。
- 因此这版设计明确保留：
  - 空卡立刻发 partial batch
  - 非空卡只发满 batch
  - 不引入 timeout

核心模型：

- 整个推理侧由单个 spin-wait 调度线程管理。
- 配置文件里仍保留 `trtDeviceToUseThread0=0` 这类表达，但只作为前向兼容输入，不再意味着真的有对应的推理线程。
- 假设共有：
  - `n` 张卡
  - 每张卡历史上等价于 `s` 个原推理线程
  - `maxBatchSize = b`
- 每张卡初始化：
  - `s+1` 个推理 stream
  - `s+1` 个 H2D stream
  - `s+1` 个 D2H stream
  - 每个推理 slot 一套自己的 execution context / device buffers / pinned host buffers
  - 每个推理 slot 为 `1..b` 预初始化一套 `cudaGraphExec`

这版最关键的调度约束：

- 全局只有一个 open batch，不是每卡一个 open batch。
- open batch 一旦创建，就必须立刻绑定：
  - `targetDevice`
  - `targetSlot`
  - 必要时还要绑定一个 `markerSlot`
- 原因是本设计坚持“以请求为单位做预处理和 H2D”，所以每个请求一进 open batch，就必须立刻知道预处理写到哪块 pinned input buffer、H2D 拷到哪个 device input buffer。
- 这等价于接受一个 tradeoff：
  - open batch 形成后基本不可迁移
  - 但换来预处理和 H2D 在 batch 形成期间被平滑摊开，而不是在 batch seal 时集中爆发

为什么坚持按请求而不是按 batch 做 H2D：

- 如果按 batch 做 H2D，那么 batch seal 时会出现一坨连续的预处理 + copy，形成明显的 CPU 突刺。
- 当前更想避免的是“猝发的密集预处理请求”，而不是那几个微秒的 H2D 本身。
- 因此更合理的目标是把：
  - 预处理
  - H2D
  都均匀摊到 open batch 的形成过程里。

slot 复用规则：

- slot 只有在 `postDone = true` 时才能复用。
- 这是 host 侧资源复用约束，不是 CUDA event 约束。
- 单调度线程下这里不需要锁竞争，只需要普通状态位维护。

open batch 的最小状态：

- `targetDevice`
- `targetSlot`
- `markerSlot`
- `size`
- `sealed`
- `launched`
- `lastH2DEvent`

关于 `lastH2DEvent` 的关键简化：

- 如果同一个 open batch 的所有请求 H2D 都发到同一条 H2D stream 上，那么 launch 时不需要等待“一组 H2D event”。
- 只需要等待最后一个 `lastH2DEvent` 即可。
- 原因是同一条 stream 上，最后一次 `cudaEventRecord()` 已经天然覆盖了之前排进去的所有 H2D。
- 这能把依赖管理从“event 列表”压缩成“每个 open batch 一个尾事件”。

卡空/卡不空决策：

- 这版明确保留现有策略，不改成 timeout，也不改成“每卡一个 filling batch”。
- 规则是：
  - 如果目标卡当前整体空闲，则允许立刻发 partial batch。
  - 如果目标卡当前非空，则只允许发满 batch。
  - 但一旦那张卡后来整体变空，当前全局 open batch 也应立刻以 partial batch 发出。
- 这本质上就是当前已有“空 GPU 允许 partial，否则等 exact batch”策略的延伸，而不是新的调度哲学。

对“标记流 / marker slot”的理解：

- 当选择的目标卡当前非空时，open batch 创建时还要记录“预计最早完成”的那条推理流，记为 `markerSlot`。
- 这个 slot 的 `graphDoneEvent` 是 launch 依赖之一。
- 由于新的 graph 一定在 `markerSlot` 结束之后才开始执行，因此 `markerSlot` 会在逻辑上把“最早空出来的窗口”让给这个 future launch。

一次请求进入系统时的工作流：

1. 调度线程非阻塞轮询：
   - 输入队列
   - 所有 `graphDoneEvent`
   - 所有 `d2hDoneEvent`
2. 如果输入队列非空：
   - 如果当前不存在 open batch，则立即做 slot 选择，并创建新的 global open batch。
   - 如果当前已经有 open batch，则直接把请求追加到当前 open batch。
3. 预处理立刻执行，写入 open batch 对应 slot 的 pinned input buffer。
4. 同时在 open batch 对应的 H2D stream 上发起这一个请求对应的异步 H2D。
5. 记录新的 `lastH2DEvent`。
6. 如果满足 launch 条件：
   - 目标卡当前为空；或者
   - open batch 已经满 batch
   则排 graph launch 和 D2H。

一次 launch 的依赖：

- graph launch 必须同时等待：
  - 当前 slot 的 `postDone = true`
  - open batch 的 `lastH2DEvent`
  - 如果有 `markerSlot`，则还要等 `markerSlot.graphDoneEvent`
- 其中 CUDA 侧依赖应尽量用：
  - `cudaStreamWaitEvent(inferStream, lastH2DEvent, 0)`
  - `cudaStreamWaitEvent(inferStream, markerGraphDoneEvent, 0)`
- 而 `postDone` 是 host 侧 slot 可复用条件，不能偷换成 CUDA event。

一次 launch 之后立即安排的工作：

- 在 infer stream 上发 `cudaGraphLaunch(graphExec[batchSize], inferStream)`。
- 在对应 D2H stream 上等待 `graphDoneEvent`，然后发异步 D2H。
- 更新该 slot 的：
  - `estimatedDoneTime`
  - `postDone = false`
- 清空当前 global open batch。

完成事件处理：

- 若某个 `graphDoneEvent` 已完成：
  - 更新该卡最近 `10` 次推理时长滚动平均。
  - 这主要用于在线修正 ETA。
  - 只有在“当前卡整体变空”且恰好存在一个已经选中当前卡/slot 的 open batch 时，才需要立即补一次 launch 决策。
- 若某个 `d2hDoneEvent` 已完成：
  - 立刻做后处理。
  - 标记 `postDone = true`。
  - 唤醒外部等待请求结果的线程。

调度线程防碰撞规则：

- 第一版即使不上预处理/后处理线程池，也不能让调度线程一次性吞很多个新请求。
- 建议规则：
  - 每次主循环最多只处理 `1` 个新请求
  - 然后立刻重新轮询 `graphDoneEvent` 和 `d2hDoneEvent`
- 这样可以把“调度线程因为连续 preprocess/postprocess 而错过 launch 时机”的风险压低到单个请求量级，而不是一口气积成一个 batch 的 CPU 突刺。

当前已接受的风险与暂不做的优化：

- 暂不引入 pre/post 线程池。
- 暂不改成 batch 级 H2D。
- 暂不引入 timeout flush。
- 暂不改变“空卡 partial、非空卡 exact batch”的核心策略。

后续真正需要验证的，不是这套语义是否自洽，而是两件事：

- 单调度线程在 steady-state 下，预处理/后处理是否会明显推迟关键 launch 节点。
- `s+1` 个 slot 是否真的能把稳态维持在“始终约有 `s` 个 infer stream 在跑 graph，而额外一个 slot 吃掉 IO 和 host 开销”。

### 2026-03-11: ETA / work accounting 模型 v0

需要解决的问题：

- 一张卡上多个 infer stream 并发时，不能把每个 graph 当成“固定结束时刻”的独立任务。
- 新 graph 插入后，已有 graph 的预期结束时间也会变化。
- 因此 ETA 不能只按“发射时刻 + 固定时长”维护。

采用的近似模型：

- 用“单流等效工作量”来记账，而不是直接记“预计结束时间”。
- 先在初始化阶段测出：
  - `base_ms[device][batchSize]`
- 这里的 `base_ms` 定义为：
  - 在该卡上、单个活跃 infer stream 条件下，对该 `batchSize` 的典型 graph 执行时间。

每个已发射 graph 维护：

- `remaining_work_ms`
- `batchSize`
- `launchTimestamp`
- `slot`
- `device`

初始值：

- graph 发射时：
  - `remaining_work_ms = base_ms[device][batchSize]`

推进规则：

- 对每张卡维护：
  - `activeInferCount`
  - `lastWorkUpdateTime`
- 调度线程每次主循环开始时，先做一次 work accounting 推进：
  - `dt = now - lastWorkUpdateTime`
  - 若该卡当前 `activeInferCount = k > 0`
  - 则该卡上所有 running graph 都执行：
    - `remaining_work_ms -= dt / k`
- 然后把 `lastWorkUpdateTime = now`

这个模型的直观含义：

- 当一张卡上只有 1 个 infer graph 在跑时，它按全速消耗自己的 `remaining_work_ms`。
- 当第 2 个 infer graph 插入后，之后这段时间里两者都按 `1/2` 速度消耗剩余工作量。
- 当并发数变成 `k` 时，每个 graph 之后都按 `1/k` 速度往前推进。
- 这正对应“stream 平分工作量；插入新工作后，已有工作也会变慢”的近似理解。

为什么先用这个模型：

- 它不要求在每次并发数变化时重算所有 slot 的绝对结束时刻。
- 只需要：
  - 记 `remaining_work_ms`
  - 记当前 `activeInferCount`
  - 在每次调度循环前推进一次
- 对单调度线程来说实现非常直接，而且足够表达当前决策里真正关心的“谁最早结束”。

marker slot 选择：

- `markerSlot` 直接选当前卡上 `remaining_work_ms` 最小的 running infer slot。
- 不再做更复杂的“空闲流定义”和二次筛选。
- slot 选择本身按 round robin。
- ETA 只负责：
  - 判断卡是否整体空闲
  - 如果非空，选哪个 `markerSlot`

在线修正：

- 实际 `graphDoneEvent` 触发后，记录真实 graph 耗时。
- 对每张卡、每个 `batchSize` 维护最近 `10` 次滚动平均：
  - `base_ms[device][batchSize]`
- 因此 ETA 表是按：
  - `device + batchSize`
  维护的，而不是只按卡维护。

边界与注意：

- 这不是精确物理模型，只是可维护的调度近似。
- 默认假设同卡多个 infer graph 对算力的竞争是“近似平均分配”的。
- 如果后续 profiling 发现 `remaining_work_ms` 的误差会显著影响决策，再考虑更细的修正项。
- 但第一版先不要把 ETA 模型做得比调度本身还复杂。

### 2026-03-11: 映射到当前代码的最小改造顺序

目标：

- 先把现有“每 server thread 自己 `getOutput()` 一把梭”改成“单 scheduler 持有多个 TRT slot，并按阶段提交”。
- 尽量不扩大改动边界到搜索线程。

需要先动的文件：

- [cpp/neuralnet/nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp)
- [cpp/neuralnet/nneval.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.h)
- [cpp/neuralnet/nninterface.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h)
- [cpp/neuralnet/trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp)
- [cpp/program/setup.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/program/setup.cpp)

第 1 步：先改 `setup.cpp` 的解释层，不改配置格式

- 当前 [setup.cpp:251](/home/wangyize/.katago/KataGomo_fork/cpp/program/setup.cpp#L251) 会把 `trtDeviceToUseThread*` 读成真正的 server-thread 到 GPU 映射。
- 新逻辑里仍然继续读取这批配置，但把它解释为：
  - “逻辑 slot 列表”
  - 同一张卡出现几次，就代表这张卡历史上等价于几个推理 slot
- 也就是说：
  - 保留旧配置写法
  - 改写其 runtime 语义

第 2 步：把 `NNEvaluator` 从“多 server thread”改成“单 scheduler thread”

- 当前 [nneval.cpp:406](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L406) 的 `spawnServerThreads()` 会真地起多个 server thread。
- 第一版应改成：
  - 只起 1 个 scheduler thread
  - 由这个 thread 内部管理所有 device/slot state
- 但外部接口尽量不动：
  - [nneval.cpp:833](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L833) 的 `evaluate()` 仍然：
    - 填 `NNResultBuf`
    - `queryQueue.forcePush(&buf)`
    - 阻塞等 `buf.hasResult`
- 这样搜索线程侧改动面最小。

第 3 步：把 `NNServerBuf` / `InputBuffers` 从“每线程一套”改成“每 slot 一套”

- 当前 [nneval.h:71](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.h#L71) 的 `NNServerBuf` 只有一套 `InputBuffers*`，语义是“每 server thread 一个”。
- 新逻辑里需要把这层提升成：
  - 每个 TRT slot 自己持有一套 host pinned input/output buffers
- 因为按请求 H2D 时，open batch 从一开始就要写进目标 slot 的 buffer。

第 4 步：把 TRT backend 从同步 `getOutput()` 改成分阶段 API

- 当前 [nninterface.h:146](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h#L146) 只有一个同步的 `getOutput()`。
- 当前 [trtbackend.cpp:2173](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2173) 里把：
  - preprocess
  - H2D
  - graph launch
  - D2H
  - `cudaStreamSynchronize`
  - postprocess
  全部串在一起。
- 要支持新调度，最终至少要拆成这些能力：
  - 把一条请求追加写入某个 slot 的 host input buffer
  - 在某个 slot 的 H2D stream 上发单请求 H2D
  - 对某个 slot / batchSize 排 graph launch
  - 对某个 slot 排 D2H
  - 查询 `graphDoneEvent` / `d2hDoneEvent`
  - 做 host 侧 postprocess，把结果写回 `NNResultBuf`

第 5 步：先只服务 TRT backend，不强迫所有 backend 一起升级

- 当前 `nninterface.h` 的统一接口让所有 backend 都共用一套签名。
- 但这次调度模型是 TRT 特化的。
- 第一版更现实的做法应是：
  - 保持旧的 `getOutput()` 接口不删，避免其他 backend 立即爆炸
  - 同时给 TRT backend 新增一组更细的 slot-oriented helper
  - `NNEvaluator` 在 TRT 路径下走新 helper，在其他 backend 下继续走旧路径

第 6 步：先把“阶段边界”跑通，再追求更漂亮的接口

- 第一版最重要的是验证：
  - `global open batch`
  - 请求级 H2D
  - event-gated graph launch
  - `postDone` 约束
  能否跑通并稳定提升 overlap
- 因此接口层允许先有一点 TRT 特化味道。
- 等工作流被验证正确后，再看是否值得把抽象收敛回统一 backend interface。

当前认为最值得先实现的最小闭环：

1. 单 scheduler thread 能替代现有多个 server thread。
2. TRT slot 资源能在初始化时一次性建好。
3. 能把单请求 append + H2D 提前发到目标 slot。
4. 能按 `lastH2DEvent + markerSlot.graphDoneEvent` 触发 `cudaGraphLaunch`。
5. D2H 和 postprocess 能异步完成，并正确唤醒原有 `NNResultBuf` 等待方。

如果以上 5 点能先跑通，后续再做：

- ETA 精度修正
- pre/post 线程池
- 搜索线程接手部分 pre/post
- 更统一的 backend 抽象

### 2026-03-11: 状态机草案 v0

为了避免第一版实现时状态散落在各处，先把 3 个核心对象的状态统一列出来：

- `DeviceState`
- `SlotState`
- `OpenBatchState`

#### A. `DeviceState`

每张卡维护：

- `deviceIdx`
- `slotIndices`
- `nextRrSlot`
- `activeInferCount`
- `lastWorkUpdateTime`
- `baseMsByBatchSize[1..b]`
- `recentGraphMsByBatchSize[1..b]`

派生量：

- `isIdle := (activeInferCount == 0)`

职责：

- 提供 round-robin slot 选择起点。
- 提供 ETA/work accounting 的卡级推进。
- 提供该卡上当前的 `markerSlot` 查询。

#### B. `SlotState`

每个 slot 维护：

- `slotId`
- `deviceIdx`
- `rrIndexWithinDevice`
- `computeHandle`
- `inferStream`
- `h2dStream`
- `d2hStream`
- `host/device input/output buffers`
- `graphExecByBatchSize[1..b]`
- `postDone`
- `hasLaunchQueued`
- `hasRunningGraph`
- `hasD2HQueued`
- `currentBatchSize`
- `remainingWorkMs`
- `graphLaunchTime`
- `graphDoneEvent`
- `d2hDoneEvent`
- `inflightRequests`

第一版建议把 slot 状态压成下面这几个互斥阶段：

- `Idle`
  - 条件：`postDone=true`，无已排队 graph，无运行中 graph，无待处理 D2H
- `LaunchQueued`
  - graph 已经排到 infer stream，但 `graphDoneEvent` 还没完成
- `D2HQueued`
  - graph 已完成，D2H 已排到 d2h stream，但 `d2hDoneEvent` 还没完成
- `PostPending`
  - D2H 已完成，但 host 侧 postprocess 还没做完

映射关系：

- `Idle`
  - `postDone=true`
  - `hasLaunchQueued=false`
  - `hasRunningGraph=false`
  - `hasD2HQueued=false`
- `LaunchQueued`
  - `postDone=false`
  - `hasLaunchQueued=true`
  - `hasRunningGraph=true`
- `D2HQueued`
  - `postDone=false`
  - `hasRunningGraph=false`
  - `hasD2HQueued=true`
- `PostPending`
  - `postDone=false`
  - `hasRunningGraph=false`
  - `hasD2HQueued=false`

实际代码里未必要真的做成 enum，但文档层面按这个互斥模型思考会清晰很多。

#### C. `OpenBatchState`

全局最多一个：

- `exists`
- `targetDevice`
- `targetSlot`
- `markerSlot`
- `size`
- `lastH2DEvent`
- `requestRefs`

第一版可以只有两种逻辑状态：

- `Absent`
- `Open`

其中 `Open` 时再用下面条件区分行为：

- `launchReadyBecauseFull`
- `launchReadyBecauseDeviceIdle`

注意：

- 第一版不需要再额外引入 `sealed` 状态。
- 因为一旦满足 launch 条件并真正排完 graph / D2H，这个 open batch 就会立刻被消费并清空。
- 从实现角度讲，“ready but not yet submitted”最多只应停留在当前调度循环的一个很短窗口里。

### 2026-03-11: 调度主循环伪代码 v0

下面是单 scheduler thread 的第一版目标逻辑，不追求接口漂亮，只追求行为正确。

```cpp
while (!isKilled) {
  now = clock.now();
  update_all_device_work_accounting(now);

  bool madeProgress = false;

  // 1. 先处理 graphDone
  for (slot in all_slots_round_robin) {
    if (slot.state == LaunchQueued && cudaEventQuery(slot.graphDoneEvent) == cudaSuccess) {
      handle_graph_done(slot, now);
      madeProgress = true;
    }
  }

  // 2. 再处理 d2hDone
  for (slot in all_slots_round_robin) {
    if (slot.state == D2HQueued && cudaEventQuery(slot.d2hDoneEvent) == cudaSuccess) {
      handle_d2h_done_and_postprocess(slot);
      madeProgress = true;
    }
  }

  // 3. 如果当前 open batch 因“卡已空闲”而已经满足 partial launch 条件，优先发它
  if (open_batch.exists && should_launch_open_batch_now(open_batch)) {
    launch_open_batch(open_batch, now);
    madeProgress = true;
  }

  // 4. 最多只吸收 1 个新请求，避免 preprocess/postprocess 把关键路径挡太久
  NNResultBuf* req = nullptr;
  if (queryQueue.tryPop(req)) {
    if (!open_batch.exists) {
      create_open_batch_and_bind_slot(req, now);
    }
    append_request_to_open_batch(req, now);

    if (should_launch_open_batch_now(open_batch)) {
      launch_open_batch(open_batch, now);
    }
    madeProgress = true;
  }

  if (!madeProgress) {
    cpu_spin_pause();
  }
}
```

关键 helper 的语义：

- `update_all_device_work_accounting(now)`
  - 用 `dt / activeInferCount` 推进每张卡上所有 running graph 的 `remainingWorkMs`
- `handle_graph_done(slot, now)`
  - 记录真实 graph 耗时
  - 更新 `baseMsByBatchSize`
  - `activeInferCount -= 1`
  - 在 `slot.d2hStream` 上排 D2H
  - record `d2hDoneEvent`
  - slot 转入 `D2HQueued`
- `handle_d2h_done_and_postprocess(slot)`
  - 做 host 侧 postprocess
  - 填回每个 `NNResultBuf`
  - `notify_all()`
  - `postDone = true`
  - slot 转回 `Idle`
- `create_open_batch_and_bind_slot(req, now)`
  - 选目标卡
  - 选目标 slot
  - 若卡非空，则确定 `markerSlot`
  - 建立全局唯一 `OpenBatchState`
- `append_request_to_open_batch(req, now)`
  - 在目标 slot 的 pinned input buffer 中追加一条请求
  - 在目标 slot 的 H2D stream 上发这条请求对应的 H2D
  - 更新 `lastH2DEvent`
  - `size += 1`
- `should_launch_open_batch_now(open_batch)`
  - true 当且仅当：
    - `size == maxBatchSize`
    - 或 `targetDevice.isIdle`
- `launch_open_batch(open_batch, now)`
  - host 侧先确认 `targetSlot.postDone == true`
  - infer stream wait:
    - `lastH2DEvent`
    - 若存在 `markerSlot`，再 wait `markerSlot.graphDoneEvent`
  - 发 `cudaGraphLaunch(graphExec[size])`
  - record `graphDoneEvent`
  - 设置：
    - `remainingWorkMs = baseMsByBatchSize[size]`
    - `postDone = false`
    - `activeInferCount += 1`
  - 清空 global open batch

### 2026-03-11: 第一版实现时故意简化的地方

- `round robin` 只负责 slot 选择，不负责 ETA 最优化。
- `markerSlot` 才使用 ETA/work accounting。
- 不追求同时吞多个新请求。
- 不追求把 postprocess 再拆给线程池。
- 不追求把搜索线程卷进来一起做 pre/post。

也就是说，第一版的重点不是“把 CPU 侧一切工作都并行掉”，而是：

- 先验证 graph / H2D / D2H 的跨 stream 调度语义
- 先验证 `s+1` slot 能否真的把 `s` 条 infer stream 稳住
- 先验证调度线程单核是否已经足够轻

### 2026-03-11: 现有 profiling / 统计结构的兼容性提醒

当前 realtime profiling 仍然带着“每个 inference slot 就是一条推理线程”的假设。

证据：

- [globalperf.h:31](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.h#L31)
  `configureInferenceSlots(const std::vector<int>& gpuIdxByServerThread)`
- [globalperf.h:49](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.h#L49)
  `changeInferenceThreadActiveCount(int inferenceThreadIdx, ...)`
- [globalperf.h:58](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.h#L58)
  `recordRealtimeInferenceBatch(int inferenceThreadIdx, ...)`
- [nneval.cpp:411](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L411)
  当前直接用 `gpuIdxByServerThread` 去配置 inference slots
- [nneval.cpp:415](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L415)
  `serverThreadsIsUsingFP16` 也按真实 server thread 数开

这意味着：

- 一旦改成“单 scheduler thread 管多个 TRT slot”，现有这些指标名义上还叫 thread，但语义已经不再是 thread。
- 如果不提前调整，realtime 页面里：
  - `inference_thread_time_share`
  - `recordRealtimeInferenceBatch`
  - `changeGpuStreamActiveCount`
  这些统计会发生概念漂移。

第一版建议：

- 先保留现有接口形状，但把它们重新解释为“inference slot index”，而不是“真实 OS thread index”。
- 也就是说：
  - `inferenceThreadIdx` 在 single-scheduler 方案下，实际上变成 `slotId`
  - `configureInferenceSlots()` 也应改成基于逻辑 slot 列表，而不是基于真实 server thread 列表
- 这样现有 realtime profile 不至于彻底失效，同时改动面相对可控。

需要一起迁移的还有：

- [nneval.h:126](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.h#L126)
  `getNumServerThreads()`
- [nneval.cpp:290](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L290)
  当前直接返回 `gpuIdxByServerThread.size()`
- [nneval.cpp:373](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L373)
  统计 FP16 使用情况也还是按 server thread 聚合

第一版不要试图在这一步把命名全部改优雅。

更现实的做法是：

- 文档和注释里明确说明：
  - single-scheduler 之后，这些“thread”指标实际上表示逻辑 slot
- 等工作流跑通后，再考虑是否做更系统的重命名：
  - `server thread` -> `inference slot`
  - `gpuIdxByServerThread` -> `gpuIdxByLogicalSlot`

### 2026-03-11: 已确认的两个硬 blocker

这两点如果不先处理，前面的单 scheduler 设计无法真正落地。

#### blocker 1: 当前 TRT backend 完全绑定 `cudaStreamPerThread`

证据：

- [trtbackend.cpp:1635](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1635)
  `setOptimizationProfileAsync(0, cudaStreamPerThread)`
- [trtbackend.cpp:1804](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1804)
  `cudaStreamBeginCapture(cudaStreamPerThread, ...)`
- [trtbackend.cpp:1811](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1811)
  `enqueueV3(cudaStreamPerThread)`
- [trtbackend.cpp:1859](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1859)
  `cudaStreamSynchronize(cudaStreamPerThread)`
- [trtbackend.cpp:1881](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1881)
  `cudaGraphLaunch(state.graphExec, cudaStreamPerThread)`
- [trtbackend.cpp:2194](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2194)
  `cudaStream_t stream = cudaStreamPerThread`

结论：

- 现有实现默认“一个 host 线程只拥有一条 TRT 工作流”。
- 一旦改成“一个 scheduler thread 管理同卡多个 slot”，继续用 `cudaStreamPerThread` 就会让所有 slot 意外共享同一条 stream。
- 这会直接破坏：
  - 多 slot 并发
  - H2D / infer / D2H 分流
  - slot 级 event 依赖

因此第一步必须把 TRT backend 改成显式 stream：

- 每个 slot 自己持有：
  - `inferStream`
  - `h2dStream`
  - `d2hStream`
- 所有 `cudaStreamPerThread` 使用点都要改成显式传入 slot 对应 stream。

#### blocker 2: 单 scheduler 管多卡后，device 选择不能再靠“线程初始化时 set 一次”

证据：

- [trtbackend.cpp:1905](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1905)
  `createComputeHandle()` 创建时只做一次 `cudaSetDevice(gpuIdxForThisThread)`
- 后续大量 runtime 操作都默认当前线程已经处在正确 device 上：
  - stream/event 操作
  - `cudaMemcpyAsync`
  - `cudaGraphLaunch`
  - `cudaEventQuery`

结论：

- 现有模型里“一个 server thread 对应一张固定 GPU”，所以这没问题。
- 但单 scheduler thread 轮流管理多卡时，这个前提失效。
- 因此以后凡是 touching 某个 slot/device 的 CUDA runtime 调用之前，都必须显式确保：
  - `cudaSetDevice(slot.deviceIdx)`

实践上应当统一封装：

- 不要把 `cudaSetDevice()` 零散撒在高层调度代码里。
- 更稳妥的做法是给所有 slot-level TRT helper 统一加一层：
  - `withDevice(slot.deviceIdx) { ... }`
  - 或者每个 helper 开头显式 `cudaSetDevice(deviceIdx)`

这里的工程含义：

- 单 scheduler 方案并不是只改 `nneval.cpp` 就够了。
- `trtbackend.cpp` 内部也必须从“线程拥有 device + per-thread default stream”的模型，迁移到“slot 显式拥有 device + stream”的模型。

### 2026-03-11: `trtbackend.cpp` 准备性重构进展

已完成一层低风险准备性重构，目标不是改调度语义，而是先把 TRT runtime 从
`cudaStreamPerThread` + “创建线程时选一次 device” 这种隐式模型里拉出来。

本次改动点：

- `ComputeHandle` 显式持有 `inferStream`
- `ComputeHandle` 内部新增 `setDevice(opName)` helper，slot-level TRT helper 在 touching CUDA 前先切到所属 device
- `setOptimizationProfileAsync`
- `cudaGraph` capture / destroy / launch
- `enqueueV3`
- `getOutput()` 中的 H2D / infer / D2H 主路径

都已经改成使用 `gpuHandle->inferStream`，不再依赖 `cudaStreamPerThread`。

这层改动当前的边界：

- 只引入了显式 `inferStream`
- 还没有把 H2D / D2H 从 infer stream 上拆开
- `getOutput()` 仍然是同步 API，末尾仍然 `cudaStreamSynchronize(stream)`
- 因此它还不能支撑最终的 single-scheduler overlap 工作流，但已经清掉了最危险的隐式前提

验证结果：

- 2026-03-11 本地增量编译通过：
  - `cmake --build cpp/build --parallel 8 --target katago`
- 无新增编译错误；只有仓库原有 warning

当前判断：

- blocker 1 还没有彻底完成，因为显式 `h2dStream/d2hStream` 还没引入
- 但最核心的一步已经完成：
  - TRT 不再硬绑 `cudaStreamPerThread`
- blocker 2 也部分落地：
  - `ComputeHandle` 自身的 slot-level TRT helper 已经开始显式 `cudaSetDevice(gpuIdxForHandle)`
  - 后续继续拆异步接口时，应沿用这个 ownership 方向，而不是把 `cudaSetDevice()` 重新散落到高层调度逻辑里

### 2026-03-11: `getOutput()` 已拆成内部三段流

继续做了一层仍然保持外部同步语义的准备性重构。

当前 `ComputeHandle` 内部资源已经变成：

- `h2dStream`
- `inferStream`
- `d2hStream`
- `h2dDoneEvent`
- `inferDoneEvent`

现在 `getOutput()` 的实际数据流已经是：

1. CPU 预处理把 row 填进 pinned host buffers
2. H2D 在 `h2dStream`
3. `cudaEventRecord(h2dDoneEvent, h2dStream)`
4. `cudaStreamWaitEvent(inferStream, h2dDoneEvent, 0)`
5. `enqueueV3` 或 `cudaGraphLaunch` 在 `inferStream`
6. `cudaEventRecord(inferDoneEvent, inferStream)`
7. `cudaStreamWaitEvent(d2hStream, inferDoneEvent, 0)`
8. D2H 在 `d2hStream`
9. 末尾仍然 `cudaStreamSynchronize(d2hStream)`，所以对外仍是同步 `getOutput()`

这层改动的意义：

- 已经把最终 overlap 设计最核心的 GPU-side 依赖表达方式写进实际代码路径
- 以后 scheduler 只需要决定“什么时候在某个 slot 上发 H2D / infer / D2H”，而不是再先推翻单 stream 结构
- 这也验证了“event gate + graph launch”这条官方建议路径在现有 TRT backend 里是能自然落地的

当前仍然没做的事：

- 还没有把 `getOutput()` 拆成 submit/query/finish 这种异步接口
- 还没有引入 open batch / slot state machine
- `InputBuffers` 仍然按当前 server-thread 模型分配和复用

验证结果：

- 2026-03-11 本地再次增量编译通过：
  - `cmake --build cpp/build --parallel 8 --target katago`

### 2026-03-11: 当前 CPU 阶段归属需要认清

重新过了一遍 [nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp) 和
[trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp)，需要明确一点：

当前实现里，并不是所有 “preprocess / postprocess” 都在 inference thread 上。

当前真实分工是：

- 调用 `NNEvaluator::evaluate()` 的 search thread 上：
  - `NNInputs::fillRowV*()` 填 feature row
  - SGF metadata row 填充
  - 等待结果后，把 logits 做 softmax / legality filtering / value 后处理
- inference thread / TRT server thread 上：
  - 把每个请求的 row 拷进 batch pinned buffers
  - H2D / infer / D2H
  - 把输出 tensor 解包回 `NNOutput` 的 logits/value fields

证据：

- [nneval.cpp:889](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L889) 附近：
  `evaluate()` 在 push 到 `queryQueue` 之前就调用 `NNInputs::fillRowV*()`
- [nneval.cpp:945](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L945) 之后：
  client 线程 wait result 完成后才做 policy softmax / legality filtering / value postprocess
- [trtbackend.cpp:2230](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2230) 附近：
  `getOutput()` 里做的是 row -> batch pinned buffer 的 copy / symmetry pack
- [trtbackend.cpp:2379](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2379) 附近：
  `getOutput()` 里做的是 output tensor -> `NNOutput` 字段解包，不是最终 softmax

这对 overlap 方案的含义：

- 如果坚持“尽量少的破坏性修改”，第一版 single-scheduler 不必把 feature extraction 和最终后处理搬到调度线程
- 更现实的第一版范围应该是：
  - search thread 继续准备 `NNResultBuf.row*`
  - scheduler 负责 batch packing、H2D、graph launch、D2H、结果回填/唤醒
- 用户提到的“让搜索线程自己接手更多工作”其实当前已经部分成立；后续若还要继续外推，应该先明确是要外推哪一层：
  - row feature extraction
  - batch packing
  - output logits 后处理

这也意味着，前面粗略口头上说的 “preprocess 10us / postprocess 4us 撞上调度关键路径” 需要重新分层看：

- feature extraction / final probability postprocess 目前已经不在 inference server thread 上
- 真正会和 single scheduler 冲突的，是 batch packing / output unpack / event polling / submit 这一层 CPU 工作

### 2026-03-11: `nneval.cpp` 当前握手模型备查

当前 evaluator 侧真实模型仍然是“每个逻辑 inference slot 对应一个 OS thread”。

关键路径：

1. search thread 在 `evaluate()` 里准备好自己的 `NNResultBuf`
2. `NNResultBuf*` 被 push 到全局 `queryQueue`
3. 某个 inference worker thread 在 `serve()` 里从 `queryQueue` 抽 batch
4. 该 worker thread 用自己长期持有的：
   - `NNServerBuf`
   - `ComputeHandle`
   同步调用 `NeuralNet::getOutput()`
5. worker thread 把结果写回每个 `NNResultBuf`
6. 对应 search thread 被各自的 condition variable 唤醒

当前 batching 规则：

- `numGpuBusyClaims` 是现有“空 GPU 可 partial，否则等 exact batch”的核心状态
- 同一 GPU 上第一个拿到 idle claim 的 worker 用 `waitPopUpToN()`
- 其他已经有兄弟 worker 在跑的 worker 用 `waitPopExactN()`

这意味着：

- 当前公开给调用方的 `evaluate()` / `NNResultBuf` 握手，本质上已经是按请求指针做完成通知
- 它并不依赖“哪个线程完成”这一点
- 因此 single-scheduler 改造里，最小 evaluator 改动面主要集中在：
  - `serve()` 主循环
  - `serverThreads`
  - `numGpuBusyClaims`
  - 启动/退出/统计相关状态
- 而不是 `evaluate()` 的外部调用协议

这条观察非常重要，因为它说明：

- 第一版没必要推翻 `NNResultBuf` 这套 per-request handshake
- 更合理的是把“谁消费 `queryQueue`、谁决定 batch 和 slot、谁在完成后 notify”改成单 scheduler 模型
- 但保留外部请求提交和等待接口不变

### 2026-03-11: realtime perf 的最小兼容路径

从 `globalperf.cpp` 看，realtime 监控里“thread”和“slot”混在命名上，但并不是所有指标都真的依赖真实 OS thread 身份。

当前结论：

- 大部分 inference batch / rows / avg batch size / stage timing 统计，本质上只要求样本落在某个 inference slot 上
- 真正 thread 语义最强的是：
  - `changeInferenceThreadActiveCount()`
  - 以及所有直接写到某个 slot-local ring buffer 的事件流

因此第一版最小兼容策略可以是：

- 文档和注释里先承认：
  - `inferenceThreadIdx` 在新设计下其实表示 logical slot id
- 仍然保留现有 JSON key 和前端结构，先不重命名外部字段
- 保证一个 logical slot 仍然只有一个 writer
  - 这样就不需要立刻给 perf ring buffer 加锁
- 如果未来出现“多个真实线程共同向同一个 logical slot 写 perf 事件”，那时再补：
  - per-slot active refcount
  - 或 per-slot ring write lock

工程上更稳妥的做法是：

- 在设计和文档层先把 `serverThreadIdx` 和 `perfSlotIdx` 逻辑区分开
- 但第一版代码可以暂时继续复用同一个整数 id
- 只要保持“一 slot 一 writer”，现有 realtime 聚合基本还能工作

### 2026-03-11: backend 适用范围

这轮单 scheduler 设计目前应明确视为 TRT-first 改造。

原因：

- TRT backend 已经开始改成“slot 显式拥有 device + stream”
- 但其他 backend，尤其 `cudabackend.cpp`，仍然更接近“线程初始化时绑死 device”的模型

因此第一版边界应该写清楚：

- 先只让 single-scheduler 跑在 TensorRT 路径
- 其他 backend 暂时保留原有 one-thread-per-handle 模型
- 不要为了抽象统一，过早把整个 `nninterface` 逼成所有 backend 一次性同步改造

### 2026-03-11: `getOutput()` 已开始拆单体函数

又做了一层纯整理性质的准备性重构，没有改变外部行为。

当前已经从 `getOutput()` 里拆出了三块 helper：

- `packInputRow(...)`
  - 单个请求的 row -> batch pinned input buffer
- `enqueueInputCopies(...)`
  - 整批 H2D + dynamic shape 设置
- `enqueueOutputCopies(...)`
  - 整批 D2H

这层改动的价值不是性能，而是把“CPU batch packing”和“GPU 提交”从超长同步函数里拉开。

这对后续 single-scheduler 的直接帮助：

- request-level H2D 真要落地时，最先复用/扩展的就是 `packInputRow(...)`
- 如果后面要把 submit API 拆成：
  - prepare
  - H2D submit
  - graph launch
  - D2H submit
  - finalize
  这三块 helper 已经给出了自然切口

当前仍未拆开的部分：

- `getOutput()` 末尾同步等待和 perf timing 汇总
- 对外仍然只有同步 `NeuralNet::getOutput(...)`

验证结果：

- 2026-03-11 本地再次增量编译通过：
  - `cmake --build cpp/build --parallel 8 --target katago`

更新：

- 现在 output tensor -> `NNOutput` 的 row unpack 也已经抽成 `unpackOutputRow(...)`

因此当前 `getOutput()` 已经基本只剩“阶段编排器”角色：

- pack rows into batch host buffers
- enqueue H2D
- infer stream 等待并 launch
- enqueue D2H
- 同步等待
- 遍历 rows 做 unpack
- 记录 perf timings

这一步完成后，下一层真正有意义的改造就不再是“继续拆函数”，而是二选一：

- 开始设计/实现异步 submit-query-finish 接口
- 或开始在 `nneval.cpp` 里引入单 scheduler 所需的 slot/open-batch 状态对象

在我看来，更合理的是先做前者，因为 `nneval.cpp` 的状态机必须建立在 backend 已经能暴露异步阶段边界的前提上。

### 2026-03-11: TRT backend 已补齐 slot 级异步接口

这一层已经落地到代码，不再只是设计稿。

新增的 TRT-only async helper 现在由 `nninterface.h` 暴露，`trtbackend.cpp` 实现：

- `trtPackInputRow(...)`
- `trtEnqueueInputRowCopy(...)`
- `trtLaunchInferenceAsync(...)`
- `trtQueryInferenceDone(...)`
- `trtEnqueueOutputCopiesAsync(...)`
- `trtQueryOutputCopiesDone(...)`
- `trtUnpackOutputRow(...)`

对应语义：

- scheduler 可以按请求把单 row 预处理结果写进 slot 自己的 pinned host buffer
- scheduler 可以立刻发 row-level H2D
- infer 和 D2H 已拆成两个独立异步阶段
- 完成后再按 row unpack 回 `NNOutput`

`ComputeHandle` 这一层也已经补了状态：

- 独立 `h2dDoneEvent / inferDoneEvent / d2hDoneEvent`
- `inferPending / d2hPending`
- 显式 `h2dStream / inferStream / d2hStream`

这一步的意义是：

- backend 终于能暴露 single-scheduler 需要的 submit / poll / finish 边界
- `nneval.cpp` 不必再强行围绕同步 `getOutput()` 设计状态机

### 2026-03-11: `nneval.cpp` 已切到单 scheduler + logical slot 模型

当前已实现的运行模型：

- TRT 路径下只启动 1 个真实 scheduler thread
- 旧配置 `trtDeviceToUseThread*` 仍然保留输入兼容
- 逻辑 slot 数改为：
  - 原有每卡 `s`
  - 再额外追加每卡 `+1`
- 也就是 `gpuIdxByLogicalSlot = old slots + one extra slot per unique gpu`

新增的核心状态对象：

- `SchedulerState`
- `DeviceState`
- `SlotState`
- `OpenBatchState`

当前已落地的关键调度规则：

- 全局唯一 `open batch`
- open batch 创建时立即绑定 `target slot`
- 按请求做：
  - row pack
  - row-level H2D
- scheduler 主循环是纯自旋热轮询
  - 不再包含任何 `yield()` / sleep / backoff
- 空卡立刻发 partial batch
- 非空卡等 exact batch
- slot 在 `D2H + output unpack + notify` 完成前不会复用
- device 内部 slot 选择暂时按 round robin

ETA 维护也已经接上代码：

- 每张卡维护 `activeInferCount`
- 每个 running slot 维护：
  - `plannedWorkMs`
  - `remainingWorkMs`
  - `accumulatedEquivalentWorkMs`
- 调度循环每轮按 `dt / activeInferCount` 推进 running slot 的剩余工作量

### 2026-03-11: 启动期 batch 基线测量已实现

之前 `baseWorkMsByBatch` 只是占位默认值。
现在启动 scheduler 时，会真实为每张卡建立初始 batch 耗时表。

当前实现：

- 每张卡只拿该卡第一个 logical slot 做启动期基线测量
- 使用一个合法的 19x19 空棋盘输入生成 row buffer
- 对每个 `batchSize in [1, maxBatchSize]`：
  - 先做一次 warmup launch
  - 再做 10 次 measured launch
- 测的是 infer completion 墙钟时间
- 不把 H2D 计入这张初始表

这样做的目标是：

- 让 scheduler 从第一个真实请求开始就有可用 ETA
- 不必等线上跑出几批后才摆脱 “全 1ms 默认值”

### 2026-03-11: 启动/退出/perf 接线已完成

`spawnServerThreads()` / `killServerThreads()` 现在已经区分两条路径：

- TRT + 非 `debugSkipNeuralNet`
  - 走 single scheduler
- 其他情况
  - 继续走旧的 one-thread-per-handle 路径

同时已补上的兼容点：

- realtime perf 的 inference slot 配置在 TRT 路径下改用 `gpuIdxByLogicalSlot`
- `serverThreadsIsUsingFP16` 在 TRT 路径下按 logical slot 填充
- scheduler 启动失败会把错误带回 `spawnServerThreads()`
- `killServerThreads()` 会释放 `schedulerState`

### 2026-03-11: 当前验证结果

已完成验证：

- 增量编译通过
  - `cmake --build cpp/build --parallel 8 --target katago`
- 真实 raw-NN smoke 通过
  - `source ./env.sh && ./cpp/build/katago evalsgf -config ~/.katago/configs/gtp.cfg -model ~/.katago/weights/b24tf.onnx -m 2 -raw-nn /tmp/katago_trt_scheduler_smoke.sgf`

smoke 观察到的关键现象：

- TRT 路径现在会创建 2 个 logical slot
  - 原配置只有 GPU0 的 1 个旧 slot
  - scheduler 自动补出每卡额外 `+1` slot
- 第一个 slot 建 plan
- 第二个 slot 直接复用现有 plan cache
- 整个 raw NN 请求能正常完成并返回 policy/value

已知仍未做的事：

- pre/post 线程池隔离
- “让搜索线程自己接手 pre/post” 的宽边界方案
- 更精细的 perf 分阶段计时
- 对 queued infer 的 ETA 仍是近似模型，不是精确重建 GPU 内部调度

### 2026-03-11: `globalPerfProfile` / `monitor_page.py` 语义清算

这次 single-scheduler + logical-slot 改造之后，`globalPerfProfile` 里有几类指标已经不再符合原来的名字或前端文案。

#### 1. 已经失效最严重的：`inference_thread_*`

当前生产端：

- scheduler 在 open batch 绑定到 slot 时调用
  - [nneval.cpp:606](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L606)
  - [nneval.cpp:638](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L638)
- batch 全部完成并释放 slot 时调用
  - [nneval.cpp:930](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L930)

聚合端仍然把它叫：

- `currentInferenceThreadActiveCount`
  - [globalperf.cpp:661](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L661)
- `inference_thread_active_time_share`
  - [globalperf.cpp:823](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L823)
- benchmark 文本里也叫 `inference_thread_time_share`
  - [globalperf.cpp:1406](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L1406)

前端也还在写：

- “活跃推理线程数”
  - [monitor_page.py:545](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L545)
  - [monitor_page.py:1001](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L1001)

但现在真实语义已经是：

- “过去 1 秒内处于非 idle 状态的 logical slot 数时间占比”

因此：

- 名字里的 `thread` 已经过时
- 它不再表示真实 OS thread 数
- 也不等于“正在 GPU 上跑 infer 的并发数”
  - 因为 slot 从 `Filling` 到 `D2HPending` 整段都会计活跃

建议改名方向：

- C++ 内部字段：`inference_slot_active_*`
- 前端标题：`活跃推理 slot 数`

#### 2. 前端“推理阶段耗时”文案已经过头，scheduler 路径下只有 `infer_ms` 还接近有意义

realtime 聚合端仍然按 batch 样本收：

- `wait_task_submit_ms / preprocess_ms / h2d_ms / infer_ms / d2h_ms / postprocess_ms`
  - [globalperf.cpp:735](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L735)
  - [globalperf.cpp:835](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L835)

但 scheduler 路径下实际写入是：

- `wait_task_submit_ms = 0`
- `preprocess_ms = 0`
- `h2d_ms = 0`
- `infer_ms = completedInferMs`
- `d2h_ms = 0`
- `postprocess_ms = 0`
  - [nneval.cpp:901](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L901)

前端当前文案却写成：

- “等待提交 / 预处理 / H2D / 推理 / D2H / 后处理，全部按分位数重建为 PDF 轮廓”
  - [monitor_page.py:540](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L540)
  - [monitor_page.py:1114](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L1114)

这在 scheduler 路径下已经不成立：

- 除 `infer_ms` 外，其余 5 个 phase 目前只是占位零值
- `infer_ms` 也只是 ETA / equivalent work 近似，不是严格的 GPU event 实测

结论：

- 这个面板对 scheduler 路径已经属于“spec 失效”
- 如果不立刻补真实 phase timing，前端至少要标注：
  - “当前 TRT overlapping 路径下仅 `推理` 为近似值，其余 phase 暂未采样”

#### 3. “每 GPU 的 cudaStream 活跃数”这个标题现在也过度承诺

当前聚合端的 `cuda_stream_active_time_share_by_gpu` 来源于：

- `changeGpuStreamActiveCount(...)`
  - [globalperf.cpp:1333](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L1333)

scheduler 路径下只在 infer graph 真正 running 时加减：

- launch 进入 `RunningInfer`
  - [nneval.cpp:697](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L697)
- infer 完成离开 `RunningInfer`
  - [nneval.cpp:867](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L867)
- 依赖 slot 转入 `RunningInfer`
  - [nneval.cpp:886](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L886)

但每个 logical slot 实际上显式拥有：

- `h2dStream`
- `inferStream`
- `d2hStream`
  - [trtbackend.cpp:1201](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L1201)

因此当前这个指标真实表示的是：

- “每 GPU 上 active infer stream 数”

而不是：

- “每 GPU 上所有 cuda stream 的活跃数”

前端标题和提示现在写的是：

- “每 GPU 的 cudaStream 活跃数”
  - [monitor_page.py:555](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L555)
- “过去 1 秒不同活跃 stream 数的时间占比”
  - [monitor_page.py:556](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L556)

这在 overlap 改造后已经偏误更大，因为：

- H2D / D2H stream 故意被拆出来了
- 但图上完全没算进去

更准确的标题应改成：

- `每 GPU 的 infer stream 并发数`

#### 4. “GPU Batch 分布”仍可用，但现在更像 ETA 加权的近似分布，不是严格 measured GPU time share

当前聚合方式是按 `sample.inferMs` 给 batchSize 加权：

- [globalperf.cpp:741](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L741)
- [globalperf.cpp:824](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L824)

而 scheduler 路径下写入的 `inferMs` 是：

- `accumulatedEquivalentWorkMs` 或回退到 `plannedWorkMs`
  - [nneval.cpp:894](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L894)

所以这张图现在的真实语义更接近：

- “按 ETA / equivalent work 加权的 batchSize 占比”

前端标题只写：

- “GPU Batch 分布”
  - [monitor_page.py:550](/home/wangyize/.katago/KataGomo_fork/python/monitor_page.py#L550)

这不算完全错误，但已经不再是“严格 measured GPU 时间占比”。

#### 5. 仍然基本成立的项

下面这些目前仍然可以认为语义基本成立：

- `totals.nn_eval / nn_batches / avg_batch_size`
  - batch 完成时仍然逐 batch 正确累加
  - [globalperf.cpp:808](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L808)
- `window1s.nn_eval_per_s / nn_batches_per_s / avg_batch_size`
  - [globalperf.cpp:817](/home/wangyize/.katago/KataGomo_fork/cpp/core/globalperf.cpp#L817)
- 搜索侧：
  - `search_threads`
  - `search_loop`
  - `search_depth_histogram`
  - `queue_length_time_share`
  它们主要来自 search 线程本身，没有被 single-scheduler 改坏

#### 6. 实际修正优先级建议

第一优先级：

- 把 `thread` 全部改名成 `slot`
- 前端面板标题同步改

第二优先级：

- 给前端 “推理阶段耗时” 面板加醒目的 backend note
- 或在 scheduler 路径下直接隐藏那 5 个零值 phase

第三优先级：

- 决定 `cuda_stream_active_time_share_by_gpu` 究竟要：
  - 继续表示 infer stream 并发
  - 还是补齐 H2D / D2H 之后，真的表示所有 stream 活跃数

### 2026-03-11: single-scheduler 监控重建 v1

这轮把 single-scheduler 路径的监控重新接起来了，目标不是“把旧线程时代的图勉强继续用”，而是明确切到新的 spec。

#### 1. realtime 快照现在显式暴露推理模式

- `globalPerfProfile` 的 realtime `status` 新增：
  - `inference_mode = "single_scheduler_slots"` 或 `"legacy_worker_threads"`
- `spawnServerThreads()` 在决定是否启用 TRT scheduler 后，会立刻调用：
  - `GlobalPerfProfile::configureInferenceMode(useTrtScheduler)`

这样前端不需要再猜“当前图表应该按 thread 语义解释，还是按 logical slot 语义解释”。

#### 2. `monitor_page.py` 已按新语义切换标题和卡片

single-scheduler 下：

- “活跃推理线程数” 改成：
  - `占用推理槽位数`
- “每 GPU 的 cudaStream 活跃数” 改成：
  - `每 GPU 的运行中推理图数`
- “GPU Batch 分布” 改成：
  - `按推理工作量加权的 GPU Batch 分布`
- “推理阶段耗时” 面板不再假装 6 个 phase 都有效：
  - 只显示 `infer_ms`
  - 并明确标注它是 scheduler 的 `equivalent work / ETA` 近似，不是 CUDA event 实测

legacy worker-thread 路径仍然保留旧面板。

#### 3. 新增一个 nsight 风格的短时 timeline 面板

前端新增了一整行 timeline 面板，用来观察：

- `Scheduler Thread`
- 选中 slot 的 `H2D`
- 选中 slot 的 `Infer`
- 选中 slot 的 `D2H`

交互：

- 默认跟随最近 `80ms`
- 支持 slot 选择
- 支持鼠标拖拽左右平移
- 支持滚轮水平缩放
- `回到最新` 可以重新跟随尾部

显示风格：

- 阶段用色块表示
- 依赖用箭头表示
- scheduler lane 会显示所有 slot 的 preprocess / postprocess
  - 当前选中 slot 高亮
  - 其他 slot 降低透明度
- H2D / Infer / D2H lane 只显示当前选中 slot

#### 4. backend 为 timeline 发布每秒 1 个约 50ms 的 sampled 窗口

`globalPerfProfile` 的 realtime 快照新增 `timeline`：

- `range_start_ns`
- `range_end_ns`
- `max_spans`
- `dropped_spans`
- `slots`
- `spans`

当前最终编码是 `compact_v1`：

- 每条 span 不再发带重复键名的 object
- 改成 `[id, slot, lane, stage, batchUid, row, startOffsetNs, endOffsetNs, dep0, dep1]`
- `start/end` 都相对 `timeline.range_start_ns`
- 前端在 `monitor_page.py` 里解码回可视化对象

当前采样点如下：

- `preprocess`
  - scheduler thread 上围绕 `symmetry` 决策 + `trtPackInputRow()` 的真实 CPU span
- `h2d`
  - row-level `trtEnqueueInputRowCopy()` 的 enqueue proxy
  - 目前没有为每个 row 单独创建 completion event
  - 所以这条 lane 的 block 主要用于表达顺序和大致占用，不是严格 copy-engine 实测
- `infer`
  - 从 slot 真正进入 `RunningInfer` 到 `trtQueryInferenceDone()` 发现完成
- `d2h`
  - 从 infer 完成时刻到 `trtQueryOutputCopiesDone()` 发现完成
- `postprocess`
  - scheduler thread 上围绕 `trtUnpackOutputRow()` + notify 的真实 CPU span

#### 5. 依赖箭头的当前语义

目前会画这些依赖：

- `preprocess -> h2d`
- `last h2d -> infer`
- `marker infer -> dependent infer`
  - 只有当 source / target span 都在当前视窗内时才会画出来
- `infer -> d2h`
- `d2h -> postprocess`

这已经足够看清：

- scheduler thread 是否被 preprocess / postprocess 撞击关键路径
- H2D 是否在 infer 之前被平滑铺开
- 某个 slot 的 infer 是否在等 marker slot
- 单 slot 的 `H2D -> Infer -> D2H -> Postprocess` 顺序是否符合设计

#### 6. 2026-03-11 晚间修正：timeline realtime payload 不能继续按 3 秒发送

在实际用 `run.sh --gtp` + `kata-analyze 100` + Playwright 看页时，前端本身工作正常，但页面停在搜索开始前的全 0 旧快照。

这更像 realtime sender 的 payload 失效，而不是前端渲染或搜索线程没有记账：

- `globalPerfProfile` 的 realtime 传输是单个 Unix datagram JSON snapshot
- 当 single-scheduler timeline 打开后，`preprocess/h2d` 是按 row 记 span，`infer/d2h/postprocess` 是按 batch 记 span
- 旧实现发“最近 3 秒”窗口，`TIMELINE_SPAN_RING_CAPACITY` 还是 `8192`
- 一个 timeline span 的 JSON 大约在 `~156 bytes`
- 只算 timeline 部分，`8192 * 156 ~= 1.28 MB`

这已经远大于 Unix datagram 在工程上可承受的单包体积，结果大概率会变成：

- `sendto()` 因 payload 过大直接失败，监控页停留在旧快照
- 或者接收端拿到截断包，JSON 解析失败

这和实际现象一致：

- 页面能打开
- `/api/state` 里保留的是搜索前的 startup 快照
- 新的活跃搜索快照没有成功替换它

因此 realtime timeline 的最终策略改成：

- 1 秒常规统计照常发
- timeline 只在每个 1 秒周期的末尾打开一个约 `50ms` 的 capture window
- 只有落在这个 window 内、或与它相交的 span 才会进入 realtime ring
- `nneval.cpp` 在生成 span id 前会先问 `GlobalPerfProfile::wantsRealtimeTimelineSpan(startNs, endNs)`
  - window 外不再持续写 timeline ring
  - window 外也不再无条件生成 row-level preprocess / H2D span id
- timeline 继续使用 `compact_v1` 编码
- 每次 snapshot 最多保留 `1024` 条 span
- 快照里带 `timeline.encoding` / `timeline.max_spans` / `timeline.dropped_spans`

这样就把 timeline 从“持续 tracing”收成了“1Hz 指标 + 每秒一个短时样本窗口”。

#### 7. 当前已知限制

- row-level H2D 仍然是 enqueue proxy，不是严格 GPU completion span
- timeline 只对 TRT single-scheduler 路径接线
- 单页上显示的是“一个 scheduler thread + 一个选中 slot 的 3 条 stream”
  - 不是全卡全 slot 同时展开
- 如果依赖源 span 不在当前时间窗里，箭头不会跨窗显示

#### 8. 2026-03-11 晚间复测结果

第一次把窗口收紧到 `100ms` 并切到 `compact_v1` 之后，realtime 已经恢复，但这仍然是持续 timeline 记录，不符合“每秒附带一个短时快照”的目标。

最终版再加上 1 秒周期内的 `50ms` capture window 之后，用实际链路复跑：

- `python3 python/monitor_page.py --host 127.0.0.1 --port 8765`
- `./run.sh --gtp --katago-bin ./cpp/build/katago`
- 向 GTP stdin 发送：
  - `boardsize 19`
  - `clear_board`
  - `kata-analyze 100`
- 用 Playwright 抓页：
  - `npx playwright screenshot --device="Desktop Chrome" http://127.0.0.1:8765 /tmp/monitor_sampled.png`

这轮端到端结果已经正常：

- `/api/state` 连续刷新
  - `receiver.received_count = 33`
  - `latest.sequence = 33`
- 实时搜索指标正常
  - `visits/s ~= 3982`
  - `nnBatches/s ~= 503`
- timeline 现在是单独的 `50.0 ms` sampled window
  - `timeline.encoding = compact_v1`
  - `len(timeline.spans) = 437`
  - `timeline.dropped_spans = 0`
  - `timeline.max_spans = 1024`
- HTTP 实际 `Content-Length = 41027`
  - 说明这份 sampled snapshot 已经稳定压回约 `41 KB`
- Playwright 截图确认页面能正确解码 compact spans，并把时间线渲染成完整的 `50ms` 默认视窗

#### 9. 本轮验证

- `cmake --build cpp/build --parallel 8 --target katago`
  - 通过
- `python3 -m py_compile python/monitor_page.py`
  - 通过
- `run.sh --gtp` + `kata-analyze 100` + Playwright screenshot
  - 已完成，页面和 `/api/state` 均正常更新

### 2026-03-11: 修正 single-scheduler 的 marker handoff 计数/状态 bug

这次又回头对照了本文前面的 scheduler 设计，发现真正的 bug 不是“缺一个并发阈值”，而是当前实现把文档里的 `markerSlot` handoff 语义压扁错了。

错误点：

- `SlotState` 之前拆成了：
  - `QueuedInfer`
  - `RunningInfer`
- 且 `handleInferCompletion()` 在一个 marker 完成后，只会把 **1 个** 依赖它的 queued slot 提升成 `RunningInfer`。
- 但按本文设计，只要多个 future launch 都在各自 infer stream 上 wait 同一个 `markerSlot.graphDoneEvent`，那么 marker 完成时，它们都应当同时获得启动资格。
- 尤其在单卡冷启动阶段，这决定了能不能从：
  - `1 running + 2 queued-behind-same-marker`
  正确爬升到：
  - `2 running`

修正后的实现：

- `SlotStage` 改回更接近文档的：
  - `Idle`
  - `Filling`
  - `LaunchQueued`
  - `D2HPending`
- `LaunchQueued` 里再用 `inferStarted` 区分：
  - graph 已排队但还在等 marker
  - graph 已真正开始 running
- `OpenBatchState` 也显式保留：
  - `targetSlotIdx`
  - `targetGpuIdx`
  - `markerSlotIdx`
- work accounting 和 marker 选择现在都只看：
  - `stage == LaunchQueued && inferStarted == true`
- 一个 marker 完成后，会把 **所有** `markerSlotIdx == marker` 且 `inferStarted == false` 的 queued launch 一起转成 active infer，并一起增加 `activeInferCount`。

这次修正后的直接结果：

- `activeInferCount` 重新表示“当前真正 running 的 infer graph 数”，不再被错误压成单链。
- realtime monitor 里的 `cuda_stream_active_time_share_by_gpu` 也重新有了可读性。

本轮实时复测：

- 重新启动：
  - `python3 python/monitor_page.py --host 0.0.0.0 --port 8765`
  - `./run.sh --gtp --katago-bin ./cpp/build/katago`
  - GTP stdin:
    - `boardsize 19`
    - `clear_board`
    - `kata-analyze 100`
- 直接抓 `/api/state` 的 `latest.window1s.cuda_stream_active_time_share_by_gpu`：
  - GPU0:
    - `bucket=1 -> 3.9753e-05`
    - `bucket=2 -> 0.999960247`

这说明当前单卡 `TRT_CUDA_STREAMS=2` 的 steady-state 已经重新回到“绝大多数时间 2 条 infer stream 同时在跑”的目标行为。

### 2026-03-11: timeline 改成 cudaEvent 时长 + 依赖回填，不能再把 CPU launch 当成 infer 起点

这轮继续修 timeline，核心结论是：

- `cudaEventElapsedTime()` 只能给出 **两个 CUDA event 之间的设备侧相对时长**
- 它**不能**直接给出和 host `steady_clock` 对齐的绝对 GPU 时间戳
- 所以时间线上的 GPU span 不能简单画成：
  - `start = CPU 调用 cudaGraphLaunch 的时刻`
  - `end = start + cudaEvent duration`

这样会把 infer 明显画早，尤其在 infer stream 前面还挂着：

- H2D 完成依赖
- marker infer 完成依赖

时，图上就会出现明显错误：

- infer span 起点贴着 CPU launch，而不是贴着真正可运行的依赖完成点
- `D2H -> Postprocess` 依赖箭头会被拉成长距离假箭头，看起来像横跨十几毫秒、甚至跨过后续几个 infer

这轮的修正方式：

- H2D / Infer / D2H 的**时长**全部改用 `cudaEventElapsedTime()`
- 但 **位置** 不再用“CPU launch 直接当起点”，而是按依赖关系回填：
  - row-level H2D:
    - enqueue 时只记 `rowIdx + preprocessSpanId + enqueueStartNs`
    - 等 infer 完成时再统一回填 H2D spans
    - 同一 slot 的多次 H2D 在时间线上强制串接，后一条 H2D 不得早于前一条 H2D 的结束
  - infer:
    - 结束时刻仍以 scheduler 发现 `inferDoneEvent` 完成的时刻为上界
    - 起点用 `now - cudaEventDuration` 倒推
    - 再 clamp 到：
      - `slot.inferStartNs`
      - `slot.lastH2DEndNs`
      的较晚者
    - 这样 infer span 至少不会早于：
      - CPU 真正把 launch 排进 stream 的时刻
      - 最后一个 H2D 完成的时刻
  - D2H:
    - 时长也用 `cudaEventElapsedTime()`
    - 起点继续锚在 infer 结束之后

当前这版不是“绝对 GPU 时间戳真值”，但已经满足：

- infer 不会再错误地贴着 CPU `cudaGraphLaunch()` 时刻起跑
- H2D / infer / D2H 的相对顺序与依赖关系一致
- 同 batch 内的依赖箭头不再被错误拉成长跨越

本轮直接复查 `/api/state` 的结果：

- `Postprocess` 依赖的 `dep0`
  - 全部指向同 batch 的 `D2H`
  - 没有跨 batch 依赖
  - 没有 `> 1000 us` 的异常大 gap
- `Infer`
  - 没有任何一条开始在其依赖的最后一个 H2D 结束之前

前端同步更新：

- `/timeline` 页面已经支持把横向视窗缩到 `1 us`
- 时间轴标签按 `ms / us / ns` 自适应显示
- 页面说明也改成：
  - `Scheduler/Pre/Post` 来自 CPU 时钟
  - `H2D/Infer/D2H` 时长来自 `cudaEvent timing`
  - 位置按依赖关系回填

后续又补了一刀显示语义：

- 横轴不再显示“相对最新的负时间”
- 改成“当前窗口内从 `0` 开始向右增长”的正时间轴
- summary 同时显示：
  - `窗口内 0 .. span`
  - `样本内 start .. end`
- 显示单位按当前缩放跨度统一选取：
  - 大窗用 `ms`
  - 微秒级窗口用 `us`
  - 更细时切到 `ns`

### 2026-03-11: 直接读取一段 50ms timeline 后，对“2us overhead” 的实际检查

直接从 monitor 的当前 `50ms` sampled snapshot 读取了一帧真实数据，得到：

- snapshot:
  - `sequence = 31`
  - `range_ms = 50.0`
  - `span_count = 500`

按依赖边界拆出来的 gap：

- `preprocess -> H2D start`
  - `count = 197`
  - `p50 = 0.030 us`
  - `p95 = 1.339 us`
  - `max = 5.630 us`
- `row H2D -> next row H2D`
  - `count = 169`
  - `p50 = 11.679 us`
  - `p95 = 86.485 us`
  - `max = 283.763 us`
- `last H2D -> infer start`
  - `count = 29`
  - `p50 = 1197.349 us`
  - `p95 = 1705.902 us`
  - `max = 1730.819 us`
- `marker infer -> dependent infer`
  - `count = 29`
  - `p50 = 1.777 us`
  - `p95 = 3.513 us`
  - `max = 4.087 us`
- `infer -> D2H start`
  - `count = 30`
  - 全部 `0.000 us`
- `D2H -> postprocess start`
  - `count = 30`
  - `p50 = 5.008 us`
  - `p95 = 15.680 us`
  - `max = 16.375 us`

这些数要这样解释，不能混读：

- `preprocess -> H2D start`
  - 很小，说明 scheduler 线程在单请求路径上从 row pack 转到发 H2D 基本没有明显额外空洞。
- `row H2D -> next row H2D`
  - 这不是 stream/event overhead。
  - 这里主要混进了：
    - 下一条请求的 preprocess
    - 调度线程继续轮询 / 取请求
  - 所以这项更像“请求级 H2D 的铺开节奏”，不是 `cudaStreamWaitEvent` 成本。
- `last H2D -> infer start`
  - 这个 `~1.2 ms` 也**绝不能**读成 wait-event overhead。
  - 这段时间主要是：
    - batch 还没到 launch 条件
    - 或者该 batch 已经做完 H2D，但还在等 marker infer 让路
  - 当前这帧里所有 infer 都带 marker 依赖，因此没有采到“空卡立即发 partial”那类无 marker 样本。
- `marker infer -> dependent infer`
  - 这是当前实现里最接近“GPU 完成被 scheduler 观察到，再把下一个 infer 接上去”的 proxy。
  - 这一项确实落在 `~2 us`：
    - `p50 = 1.777 us`
    - `p95 = 3.513 us`
  - 这和之前对“poll + 状态机 handoff”大约几微秒的预期是一致的。
- `infer -> D2H start = 0`
  - 这是当前 timeline 建模刻意造成的。
  - D2H 在 launch 时就已经排进 stream，并用 `cudaStreamWaitEvent(inferDone)` 接在 infer 后面。
  - 时间线会把 D2H 起点锚在 infer 结束，所以这份图**不能**直接拿来量 `cudaStreamWaitEvent` 本身的设备侧税。
- `D2H -> postprocess start`
  - 这是当前单 scheduler 线程真正看得到的 host-side 消费延迟。
  - `p50 = 5.008 us`，比“2us”更大。
  - 它包含的不是纯 GPU->CPU 通知成本，而是：
    - poll 发现完成
    - 当轮循环中可能先处理别的 slot / 新请求
    - 然后才进入 postprocess

当前最稳的结论：

- “跨线程 handoff ~2us”
  - 继续沿用之前 condvar benchmark 的判断。
- “单 scheduler 里，marker infer 完成后把下一条 infer 接上去”
  - 从真实 snapshot 看，确实也是 `~2 us` 量级。
- “D2H 完成后 CPU 真正开始 postprocess”
  - 当前实现里更接近 `~5 us` 中位数，而不是 `2 us`。
- “`cudaStreamWaitEvent` 本身是不是 2us”
  - 这份 timeline **看不出来**。
  - 如果要判断它本身，仍然需要最小 microbenchmark，而不是读当前 scheduler trace。

### 2026-03-11: 监控页语义收紧

这轮把 realtime 页面再收了一次，避免 dashboard 同时摆两个容易混淆的指标：

- dashboard:
  - 删除了“占用推理槽位数”
  - 保留“每 GPU 的运行中推理图数”
  - 底部布局重排后，把空出来的面积让给：
    - GPU Batch 分布
    - GPU infer 并发分布
- `/timeline`:
  - 单独新增“每 GPU 的运行中推理图数”区块
  - timeline 本体继续只展示 sampled `50ms` 窗口

额外加了 hover 细节：

- 鼠标悬浮在 block 上：
  - tooltip 里显示该 block 的 `duration`
- 鼠标悬浮在 dependency arrow 上：
  - tooltip 里显示这段依赖的 `gap`
- hover duration 一律统一成：
  - `us`
  - 保留 `2` 位小数

这样页面语义更接近当前实现的真实关心点：

- dashboard 看卡级 infer 并发有没有喂满
- timeline 看调度线程和三条 CUDA stream 的相对顺序与依赖细节

### 2026-03-11: 调度模型改版 v2，放弃跨 stream event 依赖，改成 shared-buffer + CPU 热轮询推进

这节记录最新确认的设计变更。它会覆盖本文前面所有基于：

- `s+1` 个 slot
- `markerSlot`
- `lastH2DEvent`
- `cudaStreamWaitEvent(infer, h2dDone)`
- `cudaStreamWaitEvent(infer, markerGraphDone)`
- `cudaStreamWaitEvent(d2h, inferDone)`

的 overlap 方案。

触发原因：

- 重新观察实际 trace 后，当前判断是：**跨 `cudaStream` 做 event 同步本身太贵**。
- 因此接下来的方案不再试图用 “多个 stream + waitEvent handoff” 去拼出 GPU 侧自动接力。
- 阶段推进重新交回单个 scheduler thread，通过 `cudaEventQuery()` 热轮询来做。

最新核心模型：

- 每张卡不再有 `s+1` 个 slot，而是只有 `s` 个 slot。
- 每个 slot 仍然保留：
  - 1 条 H2D stream
  - 1 条 infer stream
  - 1 条 D2H stream
  - 1 个 execution context
- 但每张卡额外维护 **`s+1` 个共享 buffer**。
- buffer 不再和 slot / stream 强绑定。
- 每个 buffer 持有：
  - host pinned input/output buffer
  - device input/output buffer
  - `h2dDoneEvent`
  - `d2hDoneEvent`
  - `postDone`
- 全局仍然只有一个 `open batch`。
- `open batch` 在创建时立即绑定：
  - `targetDevice`
  - `targetSlot`
  - `bufferIdx`
- 这个绑定在 launch 前不再迁移，行为上与目前的 slot 绑定保持一致。

明确删掉的旧设计点：

- 不再有 `n+1` 个流的 round robin。
- 不再有 `markerSlot`。
- 不再有“future launch 在 GPU 侧 wait 另一个 slot 的 inferDoneEvent”。
- 不再有“infer stream wait H2D stream event”。
- 不再有“D2H stream wait infer stream event”。
- `targetSlot` 选择仍然使用 ETA，但 ETA 只用于：
  - 选空卡
  - 或在非空卡中选“预计最早完成”的 slot
- ETA 不再承担“构造 marker 依赖链”的职责。

新的阶段推进语义：

1. scheduler 选择 `targetDevice + targetSlot + bufferIdx`
2. CPU 先检查该 buffer 的上一次后处理已完成：
   - `postDone == true`
3. 立即把该 buffer 标记为：
   - `postDone = false`
4. 预处理写入这个 buffer 的 host input
5. 在 `targetSlot.h2dStream` 上发请求级 H2D
6. scheduler 热轮询该 buffer 的 `h2dDoneEvent`
7. 若满足发射条件：
   - 空卡，允许 partial batch
   - 非空卡，只允许 exact batch
   - 且 `targetSlot` 的 infer 已空出
8. 则 scheduler 在 `targetSlot.inferStream` 上发 infer
9. scheduler 热轮询 `targetSlot.inferDoneEvent`
10. infer 一完成，就由 scheduler 在 `targetSlot.d2hStream` 上发 D2H
11. scheduler 热轮询该 buffer 的 `d2hDoneEvent`
12. D2H 完成后做 postprocess，并将该 buffer 重新标记为可复用

这里要特别注意：

- “slot 什么时候可继续接下一次 infer”和“buffer 什么时候可复用”现在是两件不同的事。
- slot 在 infer 完成、D2H 已经排出之后就可以重新进入下一轮 infer 调度。
- buffer 只有在 postprocess 完成之后才可复用。
- 这正是 `s` 个 slot 共享 `s+1` 个 buffer 的意义：
  - infer 尽量持续喂满
  - postprocess 慢一点时，不必把 infer slot 也一起卡死

对代码和文档的直接含义：

- [cpp/neuralnet/nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp) 里当前 `SchedulerState` 的这些字段后续都应删除或改写：
  - `markerSlotIdx`
  - `pendingH2DSpans` 里“等待 infer 完成后再回填并串接 marker 依赖”的假设
  - 所有“queued launch 等 marker 完成后自动转 running infer”的逻辑
- [cpp/neuralnet/trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp) 里当前这些 helper 的语义后续都要改：
  - `trtLaunchInferenceAsync(..., dependencyHandle)`
  - `trtEnqueueOutputCopiesAsync()` 内部对 `inferDoneEvent` 的 `cudaStreamWaitEvent`
  - 任何以“跨 stream wait event”表达阶段顺序的实现
- [cpp/neuralnet/nninterface.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h) 里的 TRT slot helper 后续要向：
  - slot-owned stream/context
  - buffer-owned input/output storage
  - CPU 显式推进阶段
 这套模型收口。

旧章节状态说明：

- 本文前面的这些章节现在都应视为 **历史方案**，不是后续实现目标：
  - `单调度线程 + global open batch 工作流草案 v0`
  - `ETA / work accounting 模型 v0`
  - `映射到当前代码的最小改造顺序`
  - `状态机草案 v0`
  - `调度主循环伪代码 v0`
  - `修正 single-scheduler 的 marker handoff 计数/状态 bug`
- 这些内容暂时保留，只作为“为什么会演化到当前版本”的背景记录。
- 后续真正实现时，应以本节定义的 v2 shared-buffer 模型为准重写对应段落，而不是继续在旧 `markerSlot` 方案上打补丁。

当前开发状态：

- 代码当前仍然主要站在旧 `markerSlot / waitEvent` 设计上。
- 本节只是先把新的方向正式写进文档，防止后续实现仍沿着旧语义前进。
- 下一步文档工作应是：
  - 把状态机、主循环伪代码、TRT helper 接口，全部改写成 v2 shared-buffer 版本。

### 2026-03-11: v2 shared-buffer scheduler 已开始落地

这节记录目前代码已经完成的部分，供后续跨 context 续写和 reviewer 对照。

#### 已落地的结构变化

- [cpp/neuralnet/nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp)
  - `deriveLogicalSlotGpuIdxs()` 已改为直接保留旧配置映射，不再自动额外补一个 slot。
  - `SchedulerState` 已从“slot 自带整条 batch 生命周期”改成：
    - `SlotState` 只跟踪 infer 占用和 ETA/work accounting
    - `BufferState` 跟踪 `Free/Filling/ReadyToLaunch/InferRunning/D2HPending`
  - 每个 device 现在维护：
    - `slotIndices`
    - `bufferIndices`
    - `rrBufferCursor`
  - 启动时每张卡按：
    - `s` 个 slot
    - `s+1` 个 shared buffer
    初始化资源。
  - `global open batch` 现在只保存：
    - `targetSlotIdx`
    - `targetGpuIdx`
    - `bufferIdx`
    - `batchUid`
  - 热路径现在是：
    - 请求级 preprocess
    - 请求级 H2D
    - CPU 轮询 `h2dDone`
    - CPU 显式发 infer
    - CPU 轮询 `inferDone`
    - CPU 显式发 D2H
    - CPU 轮询 `d2hDone`
    - postprocess
  - `slot` 在 infer 完成、D2H 已提交后立即清空重用。
  - `buffer` 只有 postprocess 完成后才会 `Free`。

- [cpp/neuralnet/nninterface.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h)
  - TRT async helper 接口已经改成 shared-buffer 语义：
    - `trtInitializeSharedBuffer()`
    - `trtRegisterSharedBuffer()`
    - `trtQueryInputCopiesDone(InputBuffers*)`
    - `trtLaunchInferenceAsync(ComputeHandle*, InputBuffers*, int batchSize)`
    - `trtQueryOutputCopiesDone(InputBuffers*)`
  - `dependencyHandle` 已从 infer launch 接口删除。

- [cpp/neuralnet/trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp)
  - `ComputeHandle` 现在为每个 registered shared buffer 创建独立 `IExecutionContext` 和对应的 `batchGraphStates`。
  - `InputBuffers` 现在持有：
    - shared device IO buffers
    - `trtH2DDoneEvent`
    - `trtD2HDoneEvent`
    - timeline timing events
  - 新增 shared-buffer helper：
    - `trtInitializeSharedBuffer()`
    - `trtRegisterSharedBuffer()`
  - 异步 TRT 路径已经改成：
    - row H2D 直接写 shared buffer device storage
    - infer launch 直接使用 `(slot, buffer)` 绑定过的 execution context
    - D2H 由 CPU 在观察到 infer 完成后再显式提交
  - **scheduler 驱动的异步路径已经不再使用 `cudaStreamWaitEvent()`**。

#### 目前仍保留旧语义的部分

- [cpp/neuralnet/trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp) 里的同步 `getOutput()` 路径仍然保留原实现：
  - `cudaStreamWaitEvent(inferStream, h2dDoneEvent, 0)`
  - `cudaStreamWaitEvent(d2hStream, inferDoneEvent, 0)`
- 这是非 scheduler 的旧同步接口，当前没有一起重写。
- 因此“代码里完全没有 `cudaStreamWaitEvent`”这句话现在还不成立；准确说法应是：
  - **TRT shared-buffer scheduler 热路径没有跨 stream wait-event 依赖**
  - **旧同步 fallback 仍保留 wait-event**

#### 本轮实际修掉的一个运行时 bug

- 初版 shared-buffer scheduler 在 `open batch` 已满、但 H2D 尚未全部完成、infer 还没发出时，仍然继续吞新请求。
- 这会让 `rowIdx == maxBatchSize`，触发：
  - `trtPackInputRow: rowIdx out of bounds`
- 现已修正为：
  - 当 `open batch` 的 buffer 已满时，新的请求先挂到 `deferredRequest`
  - 等待 `maybeLaunchOpenBatch()` 把当前 batch 发出去后再继续 intake

#### 当前实测状态

- `./build.sh` 已通过。
- `./run.sh --benchmark` 已通过。
- 当前一次实测结果：
  - `visits/s = 3023.44`
  - `nnEvals/s = 1966.80`
  - `nnBatches/s = 281.38`
  - `avgBatchSize = 6.99`

#### 2026-03-11: benchmark 复核，先清空 GPU 占用

- 上面这组 `3023.44 visits/s` 的 benchmark 后来确认是 **脏结果**。
- 原因不是 shared-buffer scheduler 本身，而是验证时 GPU 上还残留着先前开启的 GTP 进程：
  - `/home/wangyize/.katago/build/katago`
- 在杀掉残留进程并确认 `nvidia-smi --query-compute-apps` 为空后，重新运行：
  - `./run.sh --benchmark`
- 干净环境下的新结果是：
  - `visits/s = 6351.72`
  - `nnEvals/s = 4242.28`
  - `nnBatches/s = 607.06`
  - `avgBatchSize = 6.99`
- 这基本回到了本分支前面几次 benchmark 的量级，因此目前 **不能** 再把 shared-buffer 改造视作“已确认存在明显吞吐回退”。
- 后续所有性能验证前，都应该先做：
  - 检查 `nvidia-smi --query-compute-apps`
  - 杀掉残留的 `run.sh/katago` 会话
  - 再跑 benchmark 或 timeline 观察

#### 当前 perf/spec 对齐状态

- 这次 v2 shared-buffer 改造之后，旧的若干 `globalPerfProfile` 口径已经不再可信，至少包括：
  - `inference_preprocess_ms`
  - `inference_h2d_ms`
  - `inference_wait_gpu_ms`
  - `inference_d2h_ms`
  - `inference_postprocess_ms`
- 在当前 benchmark 输出里，这些项已经全部变成 `n/a`。
- 这符合预期：
  - 热路径已经不再走旧的“单个同步 `getOutput()` 一次性统计分阶段时间”模型。
- `inference_thread_time_share: active0=100%` 这项现在也不能直接解读成 scheduler 已退化为单 slot：
  - 它仍然是旧语义下的指标名和记账口径
  - 后续需要结合新的 slot/buffer 语义重建，或直接以新的 timeline / GPU running infer graph 指标替代

#### 下一步最需要 reviewer 核对的点

- 实现是否确实满足：
  - `global open batch`
  - `s` 个 slot + 每卡 `s+1` 个 shared buffer
  - slot 在 inferDone + D2H submitted 后即可复用
  - buffer 仅在 postprocess 结束后复用
  - scheduler 热路径不依赖 `cudaStreamWaitEvent`
- 以及哪些旧文档段落还没有完全重写，仍然和当前实现表述不一致。

### 2026-03-11: 修正 v2 launch 语义，exact batch 不必等 target slot 空

这条是今天从 raw timeline 数据里重新确认出来的设计修正。它覆盖当前实现里“必须等 `targetSlot` 变空才能 launch infer”的做法。

#### 观察到的问题

- 同一条 infer lane 上，相邻两次 infer 之间出现了稳定的 `~15-23 us` gap。
- 读取 raw timeline 后，能看到：
  - 下一批 exact batch 的最后一条 H2D 往往早在前一批 infer 结束前就已经完成
  - 但下一次 `cudaGraphLaunch()` 仍然要等到 CPU 看见上一批 infer 完成之后才发
- 这说明当前 gap 不是 `cudaStream` 自己的排队税，而是 scheduler 没有提前把下一批 infer 排到同一条 infer stream 上。

#### 正确语义

- 一旦 `open batch` 满足 launch 条件：
  - 空卡：允许 partial
  - 非空卡：必须 exact batch
- 并且这批的 H2D 已经完成
- 那么 **应当立刻把 `cudaGraphLaunch()` enqueue 到它绑定的 `targetSlot.inferStream` 上**
- **不需要等待 `targetSlot` 当前那批 infer 先结束**
- 同一条 CUDA stream 的顺序语义会保证：
  - 先前 infer 先执行
  - 后续 infer 自动接在后面

也就是说：

- 当前错误实现：
  - `inferDone -> CPU bookkeeping -> launch next infer`
- 目标实现：
  - `batch exact + H2D done -> 立即 enqueue 到 target infer stream`
  - `inferDone` 只用于之后 CPU 侧触发 D2H / postprocess

#### 对实现的直接含义

- `slot.inferBusy` 不能再作为 `maybeLaunchOpenBatch()` 的 launch gate。
- slot 不再只允许“至多 1 个已 launch infer batch”。
- 每个 slot 需要显式维护：
  - 已经 enqueue 到这条 infer stream 上的 buffer 队列
- 因为同一条 infer stream 上会连续排多个 batch，所以 infer 完成事件不能再是 slot 级：
  - 必须改成 **buffer 级 inferDoneEvent**
  - infer timing event 也必须改成 **buffer 级**
- CPU 在轮询 infer 完成时，应当查看：
  - 某个 slot 的 infer queue 头部 buffer 是否完成
- 完成后：
  - 为该 buffer 提交 D2H
  - 再从这条 slot queue 头部弹出
  - queue 后面的 infer 不需要 CPU 再次 launch，它早就已经排进 infer stream 里了

#### 时间线建模也要同步修正

- 相邻 infer span 之间应该允许出现：
  - `dep0 = 本 batch 最后一个 H2D span`
  - `dep1 = 同 slot 上一个 infer span`
- 这样时间线才能正确表达：
  - “同一 infer stream 的串行关系”
  - 而不是把它误解为 CPU 晚发导致的空洞

#### 2026-03-11 实现进展

这节对应的代码修复已经完成，核心变化如下：

- `slot.inferBusy` launch gate 已经删除。
- `maybeLaunchOpenBatch()` 现在会在 `open batch` 满足 launch 条件且 H2D 完成后，立刻把 infer enqueue 到 `targetSlot.inferStream`。
- 每个 slot 现在维护 `launchedBufferIndices`，表达“已经排进这条 infer stream 的 buffer FIFO”。
- infer 完成查询已经改成 **buffer 级**：
  - `trtQueryInferenceDone(InputBuffers*)`
  - `trtGetLastInferenceElapsedMs(InputBuffers*)`
- `InputBuffers` 现在拥有自己的：
  - `trtInferDoneEvent`
  - `trtTimelineInferStartEvent`
  - `trtTimelineInferEndEvent`
- scheduler 轮询逻辑现在是：
  - 看每个 slot 的 infer FIFO 头部 buffer 是否完成
  - 完成后提交该 buffer 的 D2H
  - 再把它从 slot FIFO 头部弹出
  - FIFO 后面的 infer 不需要 CPU 再次 launch

#### 2026-03-11 验证结果

- 干净 GPU 环境下，`./run.sh --benchmark` 结果：
  - `visits/s = 6518.64`
  - `nnEvals/s = 4296.19`
  - `nnBatches/s = 615.04`
  - `avgBatchSize = 6.99`
- `window1s.cuda_stream_active_time_share_by_gpu` 显示：
  - GPU0 的 `bucket=2` 占比为 `1.0`
  - 说明当前稳定维持 2 条 infer stream 同时活跃
- 读取一帧 `50ms` raw timeline 样本后：
  - slot 0 上连续 infer 的 gap 为 `0.00 us`
  - slot 1 上连续 infer 的 gap 为 `0.00 us`
  - 说明 exact batch 已经成功提前排进同一条 infer stream
  - 之前那种 `~15-23 us` 的 CPU handoff gap 已经消失

#### 剩余注意事项

- 这次修复只改变了 shared-buffer TRT scheduler 路径。
- `trtbackend.cpp` 里旧的同步 `getOutput()` fallback 仍然保留 `cudaStreamWaitEvent(...)`，因为那条路径不属于当前 scheduler 热路径。
- benchmark 的 `inference_preprocess_ms / inference_h2d_ms / inference_wait_gpu_ms / inference_d2h_ms / inference_postprocess_ms` 目前仍是 `n/a`。
  - 这说明单调度 / shared-buffer 新模型下，`GlobalPerfProfile::recordInferencePhases(...)` 这组老阶段统计还没有重新接线。

#### 2026-03-11: D2H -> postprocess 大 gap 的进一步分析

修正 D2H 时间线起点之后，raw timeline 里仍然能看到：

- `D2H -> postprocess` gap 大约 `0.72-0.89 ms`
- 但 postprocess block 自己只有 `~1.6 us`

当前更合理的解释是：

- 这不是 postprocess 自己慢
- 也不是 D2H span 被画早了
- 而是 **D2H 已经完成之后，scheduler 线程要过 `~0.8 ms` 才重新拿到 CPU 并开始 finalize**

支持这个判断的证据：

- `kata-analyze 100` 下的 raw timeline 里，在 `D2H end -> postprocess start` 之间几乎没有别的可见 scheduler block
- 机器配置是 `24` 个逻辑 CPU，而当前运行时有：
  - `21` 个 search threads
  - `1` 个 scheduler thread
  - 以及 GTP / monitor / 其他辅助线程
- 因此这段 gap 更像 **单 scheduler 在线程资源接近吃满时被 OS 调度推迟**

这条现象需要单独记账，因为它直接关系到后续是否要做：

- 给 scheduler thread 预留 CPU 资源
- 降低 search threads
- 提升 scheduler thread 优先级 / affinity
- 或者把 postprocess 从 scheduler 主循环里拆出去

#### 2026-03-11: reviewer 额外发现并已修复的时间线问题

reviewer 额外指出了一条中等严重度的问题：

- 同一 slot 上如果多个 `D2HPending` buffer 在 scheduler 下一次轮询前都已经完成
- 原实现会按全局 `state->buffers` 顺序 finalize
- 这会让同一条 D2H stream 的 span / dependency 链在时间线上偶发乱序

当前修复是：

- 每个 slot 额外维护自己的 `d2hPendingBufferIndices` FIFO
- 只有 queue 头部 buffer 允许被 `trtQueryOutputCopiesDone(...)` 和 `finalizeCompletedBatch(...)` 处理
- 因此 D2H / postprocess 的时间线顺序现在和同一 slot 的 D2H stream 提交顺序一致

#### 2026-03-11: scheduler 纯自旋约束

在继续追 `D2H -> postprocess` gap 之前，先把一个实现约束固定下来：

- TRT scheduler 线程现在不再包含任何 `std::this_thread::yield()`。
- 启动期的基线测量轮询也不再 `yield()`。
- 当前语义就是：
  - 宁可持续占用一个 CPU 核
  - 也不主动把调度时机让给 OS

这意味着后续如果仍然观察到大的 `D2H -> postprocess` 空窗，不能再归因于 scheduler 代码里显式的 `yield/backoff` 逻辑，需要继续查别的原因。

#### 2026-03-11: `infer -> D2H` 大 gap 的代码级原因与修复

新一轮时间线检查里，又观察到一段 `infer end -> D2H start` 的大 gap。当前已经定位到两个具体代码问题：

- `handleInferCompletion()` 原实现会在真正 `trtEnqueueOutputCopiesAsync()` 之前，先做一整批 row 的 H2D timing 回填：
  - 对每个 row 调一次 `trtGetLastInputRowCopyElapsedMs(...)`
  - 再调一次 `trtGetLastInferenceElapsedMs(...)`
  - 再写一串 realtime timeline spans
- 主循环原实现只在每轮顶部取一次 `nowNs`，然后把这个时间传给同一轮里所有 slot 的 `handleInferCompletion(...)`
  - 如果前一个 slot 的 completion 处理本身已经花掉了几十到几百微秒
  - 后一个 slot 的 infer end 会被错误地 clamp 到更早的旧时间
  - 时间线上就会把 `infer -> D2H` gap 额外夸大

当前修复已经落地：

- `handleInferCompletion()` 不再接收循环顶部的 `nowNs`
- 改为在函数内部用 fresh `completionObservedNs`
- 先立即：
  - 记录 `buffer.d2hEnqueueNs`
  - 调 `trtEnqueueOutputCopiesAsync(...)`
  - 把 buffer 挂到 `d2hPendingBufferIndices`
  - 释放 slot 的 infer occupancy
- 然后才做 H2D / infer timing 回填和 timeline span 记录

这次修复的目标非常明确：

- `infer done` 一被 scheduler 观察到，就先把 D2H 发出去
- timeline instrumentation 不能再挡在 `infer -> D2H` 关键路径中间
- infer end 的时间建模也不能再使用同一轮主循环里过期的 `nowNs`

#### 2026-03-11: 确认真正的热点开销来自 timeline 本地采样，不是 `cudaEventQuery()`

进一步检查后，当前结论已经很明确：

- `cudaEventQuery()` 本身不是问题。
- 在这台机子上的最小 ready-event benchmark 里：
  - `cudaEventQuery()` 对已完成 event 的平均 API 成本只有 `0.0593 us`
- 所以之前 timeline 里看到的：
  - `infer -> D2H ~100 us`
  - `D2H -> postprocess ~10+ us`
  不能归因于 `cudaEventQuery()` 本身。

真正重的是 hot scheduler thread 里为了 sampled timeline 做的本地处理：

- `trtGetLastInputRowCopyElapsedMs(...)`
- `trtGetLastInferenceElapsedMs(...)`
- `trtGetLastOutputCopiesElapsedMs(...)`
- 以及逐 row H2D timeline 回填

当前已经按更保守的低开销方向收了一刀：

- hot path 不再调用任何 `cudaEventElapsedTime()` 系列接口
- H2D timeline 从逐 row 回填，收缩成 **batch-level H2D span**
- infer / D2H timeline 也改成基于 hot thread 观察到的原始时间戳
- 仍然保留 sampled timeline 的相对顺序、依赖链和关键 gap 观测能力

在这一版上重新执行：

- 清 GPU
- `./build.sh`
- 启动 monitor
- `./run.sh --gtp`
- stdin 输入 `boardsize 19 / clear_board / kata-analyze 100`

然后读取新的一帧 raw timeline，得到：

- `infer -> D2H`
  - `min/mean/p50/p95/max = 0.03 / 0.03 / 0.03 / 0.04 / 0.04 us`
- `D2H -> postprocess`
  - `min/mean/p50/p95/max = 0.03 / 0.04 / 0.03 / 0.05 / 0.11 us`

这说明：

- 之前的百微秒级 gap 的确主要是 sampled timeline 本地采样/回填本身制造的
- 在去掉 hot path 的 elapsed 查询和逐 row 回填后，关键 handoff gap 已经基本塌到可忽略量级

后续如果还要继续逼近“热线程只写原始事件时间戳，收集线程再做处理”的最终形态，可以继续把：

- `GlobalPerfProfile::recordRealtimeTimelineSpan(...)`
- 以及 sampled timeline 的依赖整理

也从 hot scheduler thread 再往外挪一层。  
但从这次数据看，真正的大头已经被拿掉了。

#### 2026-03-11: 修正上一条错误结论，timeline 中所有 CUDA-event 阶段恢复到 CUDA 时间基准

继续对照真实 timeline 后，确认上一条收得过头了，具体有两处错误：

- H2D 被压成了单个 batch-level block，而不是按请求 / row 展开成多个 block
- H2D / Infer / D2H 的时间轴被退化成了 scheduler 观察时间，不再主要由 CUDA event 的 duration 决定

这和当前调试目标不符。当前时间线的约束应恢复为：

- 只要阶段边界本来就是由 CUDA event 标记的，就仍然使用 CUDA event 的 duration 做时间建模
- H2D 必须恢复成逐 row / 逐请求 block，不能再压成单个 batch H2D 块
- Scheduler CPU 时间只用于：
  - preprocess / postprocess 的 CPU block
  - 以及给 CUDA timeline 提供一个 host 侧锚点

这也澄清了一点：

- `cudaEventElapsedTime()` 本身并不贵
- 在这台机子上的最小基准里，对已完成 event 做一次 `cudaEventElapsedTime()` 的平均 API 成本大约是 `0.0341 us`

因此前面那条“百微秒级 gap 主要是 `cudaEventElapsedTime()` 本身”的解释是不成立的。  
更可能的真实问题，是我当时把 sampled timeline 的 H2D / Infer / D2H 建模顺手改错了。

当前代码已经按这个修正方向收回：

- 恢复 `trtGetLastInputRowCopyElapsedMs(...)`
- 恢复 `trtGetLastInferenceElapsedMs(...)`
- 恢复 `trtGetLastOutputCopiesElapsedMs(...)`
- H2D timeline 恢复为逐 row span，而不是 batch-level 合并 span

下一步继续观察：

- timeline 是否重新出现逐 row H2D blocks
- H2D / Infer / D2H 的长度和相对顺序是否重新与 CUDA event 观测一致
- 如果恢复精度后再次出现明显性能影响，再单独把“采样数据写入预分配缓冲区，由汇总线程处理”的结构落地

#### 2026-03-11: 修正 shared-buffer timeline 的 CUDA span 锚点

当前进一步确认的问题不是 scheduler 真的在 `infer -> D2H` 间稳定空了几百微秒，而是 timeline 里还残留了这一类错误建模：

- 用 host 的 `enqueue/launch` 时间戳做 span 起点
- 再叠加 `cudaEventElapsedTime()` 给出的 duration

这会把 “真实 GPU 事件时间” 和 “host 提交时间” 混到一条 span 里，尤其在 shared-buffer + 同 slot 提前排队的模型下，会把 infer 结束点系统性画早，进而制造假的 `infer -> D2H` gap。

修正方向：

- 对于 `H2D / Infer / D2H` 三类 CUDA span：
  - 同一批次内统一使用 CUDA event offset 重建
  - 以该批次第一条 `H2D start event` 为 batch-local base
  - 其余 row 的 `H2D start/end`、整批 `infer start/end`、整批 `D2H start/end` 都改成
    - `batchCudaBaseNs + cudaEventElapsedTime(baseEvent, targetEvent)`
- host 时钟只保留在：
  - `preprocess`
  - `postprocess`
  - 以及 scheduler busy span 这种纯 CPU 指标

这不能凭空制造“真正的 GPU 绝对时间戳”，但能保证：

- 所有 CUDA span 的相对位置由同一套 CUDA event 时钟决定
- `infer -> D2H` 之类的 gap 不再由 `host launch time + CUDA duration` 的混合建模伪造出来

修正后的实际观测：

- 重新 build / run / `kata-analyze 100` 后，raw `50ms` timeline 中
  - `infer -> D2H` 不再出现几百微秒量级的假 gap
  - 5 帧聚合后约为
    - `p50 ~= 5.5 us`
    - `p95 ~= 14.1 us`
  - `D2H -> postprocess`
    - `p50 ~= 4.6 us`
    - `p95 ~= 5.0 us`

这说明：

- 之前那种几百微秒的 `infer -> D2H` 主要是时间线建模错误
- 修完后剩下的主要是 scheduler 真实的 CPU 观察 / enqueue / finalize handoff，而不是虚假的 span 锚点漂移

#### 2026-03-11: 新增调度线程空闲时间占比实时统计

为后续继续压 `infer -> D2H -> postprocess` handoff gap，dashboard 侧新增一个更直接的 realtime 指标：

- `scheduler_idle_time_share`

定义方式：

- 调度线程每次真正进入一段具体工作时，记录一段 `busy span`
- `busy span` 使用 wall clock / steady clock 的 `startNs` 和 `endNs`
- realtime publisher 在每秒快照时，把最近 1 秒窗口内所有 `busy span` 做裁窗并求并集长度
- `scheduler_busy_time_share = busy_ns / 1s`
- `scheduler_idle_time_share = 1 - scheduler_busy_time_share`

当前接线范围：

- `infer done -> handleInferCompletion`
- `maybeLaunchOpenBatch`
- `d2h done -> finalizeCompletedBatch`
- 请求处理分支里的 preprocess / H2D enqueue / 可选 launch

展示方式：

- dashboard 左侧“近 60 秒趋势”增加第三张 sparkline 卡片
- 标题为“调度线程空闲”
- 显示当前 1 秒窗口的空闲占比，以及最近 60 秒历史

#### 2026-03-11: 新增调度线程空轮询耗时实时统计

为回答“如果没有其它动作，一次 scheduler poll 本身要多少微秒”，不做单独离线 benchmark，而是直接在 realtime 里新增：

- `scheduler_idle_poll_avg_us`

定义方式：

- 仅统计 scheduler 主循环里“纯轮询、没有具体动作”的 iteration
- 也就是：
  - 没有 `infer done` completion
  - 没有 `launch`
  - 没有 `d2h done` finalize
  - 没有请求处理
- 每次这样的 iteration 记录一次 wall-clock `startNs/endNs`
- publisher 每秒取累计 `count / total_ns` 的 delta，输出
  - `scheduler_idle_poll_avg_us = delta_total_ns / delta_count / 1e3`

展示方式：

- dashboard 左侧“近 60 秒趋势”新增一张“空轮询耗时”卡片
- 显示当前 1 秒窗口的平均空轮询耗时，以及最近 60 秒历史

#### 2026-03-11: timeline 新增 CPU 侧 H2D/D2H cuda call span

为解释 `infer -> d2h` 与 `d2h -> postprocess` 之间的残余微秒级空隙，timeline 新增两类 CPU span，统一画在 scheduler lane 上：

- `H2D CPU`
  - 颜色浅蓝
  - 表示 scheduler 线程调用 `trtEnqueueInputRowCopy()` 的 wall-clock 持续时间
  - 依赖关系为 `Preprocess -> H2D CPU -> H2D(GPU)`
- `D2H CPU`
  - 颜色浅绿
  - 表示 scheduler 线程调用 `trtEnqueueOutputCopiesAsync()` 的 wall-clock 持续时间
  - 依赖关系为 `Infer(GPU) -> D2H CPU -> D2H(GPU) -> Postprocess`

这样 timeline 上不再把 CPU 发起 CUDA call 的那一小段压成“空白”。

补充：

- `H2D/D2H GPU` 的绝对位置仍然是“CPU wall clock 基准 + CUDA event offset”重建出来的
- 因此会有亚微秒级 mixed-clock 偏差，可能出现视觉上的 `GPU start < CPU call start`
- 当前已对 display 做保守钳位：
  - `H2D GPU start >= H2D CPU start`
  - `D2H GPU start >= D2H CPU start`
- 这不改变时长，只是去掉不符合因果关系的显示伪像

#### 2026-03-12: timeline 改成“CPU 观察完成时刻 + CUDA duration 倒推”

继续梳理后确认，旧 timeline 的核心问题不是 `cudaEventQuery()` 慢，而是绝对位置仍混用了：

- `steady_clock` 的 CPU 时间
- 以第一条 `H2D start event` 为 base 的 CUDA offset

这会让 `infer -> d2h`、`d2h -> postprocess` 看起来出现不真实的大 gap。

当前新口径：

- 所有 GPU 操作都保持 `cudaEvent(start) + gpu work + cudaEvent(end)` 的结构
- 这套 pattern 在 TRT backend 里统一封成一个 helper
- timeline 不再使用“base event + offset”来拼绝对位置
- GPU span 的绝对结束时刻改成由 CPU 观察完成时决定
- span 开始时刻由 `cudaEventElapsedTime()` 倒推

具体规则：

- `H2D`
  - 逐 row 仍保留 `start/end` event
  - 绝对结束锚点使用“最后一条 row copy 完成被 CPU 观察到”的时刻
  - 每一 row 的结束时刻通过 `row_end -> last_row_end` 的 CUDA 相对时间倒推
  - 每一 row 的开始时刻再用 `row_start -> row_end` 的 CUDA 时长倒推
- `infer`
  - 绝对结束时刻直接使用 CPU 观察到 `infer done` 的时刻
  - 开始时刻使用 `infer duration` 倒推
- `D2H`
  - 绝对结束时刻直接使用 CPU 观察到 `d2h done` 的时刻
  - 开始时刻使用 `d2h duration` 倒推

这意味着：

- 不再依赖 `cudaEvent` 的“绝对时间戳”能力
- 也不需要 `cudaLaunchHostFunc()` 之类的 CPU/GPU epoch 对齐技巧
- 旧的 `timeline base event` 重建路径应视为废弃

实现切口：

- `cpp/neuralnet/trtbackend.cpp`
  - 统一新增 `enqueueCudaTimedSection(...)`
  - `trtEnqueueInputRowCopy / trtLaunchInferenceAsync / trtEnqueueOutputCopiesAsync`
    全部走这套 helper
- `cpp/neuralnet/nninterface.h`
  - 删除旧的 `trtGetTimeline*OffsetMs(...)`
  - 新增 `trtGetLastInputRowCopyEndToLastCopyEndMs(...)`
- `cpp/neuralnet/nneval.cpp`
  - 删除 `cudaTimelineBaseNs`
  - 改为记录 `h2dDoneObservedNs`
  - `infer` 与 `d2h` 的 GPU span 均直接用 CPU 观察完成时刻作为 `endNs`

当前 live 验证结果：

- `infer -> d2h_cpu`
  - `p50 ~= 0.03 us`
  - `p95 ~= 0.05 us`
  - `max ~= 0.13 us`
- `d2h -> postprocess`
  - `p50 ~= 0.15 us`
  - `p95 ~= 0.22 us`
  - `max ~= 0.26 us`

样本：

- `infer` `28.85 us`
- `d2h_cpu` `7.53 us`
- `d2h` `3.74 us`
- `postprocess` `2.17 us`
- `infer -> d2h_cpu gap` `0.03 us`
- `d2h -> postprocess gap` `0.17 us`

这说明当前几百微秒的假 gap 已经不再来自 timeline 锚点错误。

#### 2026-03-12: timeline 新增浅橙色 `launch_cpu` span

为了把 `cudaGraphLaunch()` 的 CPU 侧提交成本从“空白”里单独抠出来，timeline 新增一类 scheduler lane span：

- `launch_cpu`
  - 颜色浅橙
  - 表示 scheduler 线程调用 `trtLaunchInferenceAsync()` 的 wall-clock 持续时间
  - 依赖关系改成 `H2D -> Launch CPU -> Infer`

实现口径：

- `maybeLaunchOpenBatch()` 在调用 `trtLaunchInferenceAsync()` 前后各取一次 `steady_clock` 时间戳
- 如果当前样本窗口需要捕获 timeline，就记录一段 `TimelineStage::LaunchCPU`
- `infer` 的主依赖优先改成这段 `launch_cpu` span
- `infer` 的显示起点继续由“CPU 观察到 infer 完成的时刻 - CUDA duration”倒推
- 同时做因果钳位：
  - `infer start >= launch_cpu end`

当前 live 样本：

- `launch_cpu_count = 31`
- `launch_cpu`
  - `p50 ~= 7.68 us`
  - `p95 ~= 10.28 us`
  - `max ~= 15.40 us`
- 当前帧里 `infer` 的主依赖多数已经切到 `launch_cpu`

注意：

- `launch_cpu -> infer` 之间如果还看到很大的 gap，那不表示 `cudaGraphLaunch()` 这个 CPU call 自己很慢
- 它表示 infer 在 GPU stream 上还要等前序 infer 排空
