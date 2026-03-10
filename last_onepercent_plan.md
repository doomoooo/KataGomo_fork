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
