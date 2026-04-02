# CUDA Graphs 备忘

整理目标：把 NVIDIA 官方 `CUDA Programming Guide` 中 `4.2 CUDA Graphs` 这一章压缩成一份工程向备忘，方便后续查概念、API 和限制条件。

## 文档来源

- 官方页面：https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- 文档版本：CUDA Programming Guide `v13.2`
- 整理时间：`2026-03-27`
- 说明：以下内容是归纳整理，不是原文逐字翻译。

## 一页总结

CUDA Graph 的核心思想是：先把一批 GPU 工作和依赖关系定义成图，再把这张图实例化成可执行对象，之后重复 launch。这样做的收益主要有两点：

- 降低 CPU 侧反复发射 kernel / memcpy / memset 的提交开销。
- CUDA 可以看到更完整的工作流，从而做更激进的全局优化。

最适合 CUDA Graph 的场景：

- 同一套 GPU 工作流会被重复执行很多次。
- 图的拓扑基本稳定，变化主要体现在 kernel 参数、指针、copy 大小等节点参数上。
- 单个 kernel 很短，launch overhead 在总耗时中占比明显。

## 核心对象和模型

### 1. 两个核心句柄

- `cudaGraph_t`：图模板，描述有哪些节点、节点之间如何依赖。
- `cudaGraphExec_t`：可执行图，由 `cudaGraph_t` 实例化得到，可重复 launch。

### 2. 节点和边

图中的一个操作就是一个节点，依赖关系就是边。节点完成后，后继节点何时被调度由 CUDA 系统决定，不要求按“书写顺序”线性执行。

文档列出的常见节点类型：

- kernel
- CPU function call / host callback
- memcpy
- memset
- empty node
- event wait / event record
- external semaphore wait / signal
- conditional node
- memory node
- child graph

### 3. Edge Data

CUDA 12.3 起，Graph 支持 edge data。可以把它理解成“比默认依赖更细的依赖语义”。

- 默认零值 edge data 表示完整依赖，并带内存同步语义。
- edge data 由 `outgoing port`、`incoming port`、`type` 三部分组成。
- 当前官方文档强调的主要用途是启用 `cudaGraphDependencyTypeProgrammatic`，即 Programmatic Dependent Launch。
- 如果查询 API 省略了非零 edge data，可能返回 `cudaErrorLossyQuery`。

这部分偏高级特性，日常使用 CUDA Graph 时多数场景仍然使用默认依赖即可。

## 生命周期

### 1. 三阶段

CUDA Graph 的工作流可以拆成三步：

1. 定义：创建图模板并加入节点、依赖。
2. 实例化：校验图、做 launch 前准备，得到 `cudaGraphExec_t`。
3. 执行：把可执行图发到某个 stream 上运行。

### 2. 显式 Graph API

如果想完全控制图结构，可以直接使用 Graph API：

```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;

cudaGraphCreate(&graph, 0);

// 省略：cudaGraphAddKernelNode / cudaGraphAddMemcpyNode / ...

cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(graphExec, stream);
```

适合场景：

- 你需要显式控制节点和依赖。
- 你要做节点级更新、插入 memory node、conditional node 等高级操作。
- 你不想把已有 stream 代码整段 capture 进来。

### 3. Stream Capture

如果已有一段基于 stream 的代码，通常更容易从 capture 起步：

```cpp
cudaGraph_t graph;
cudaGraphExec_t graphExec;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

kernelA<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(..., stream);
kernelB<<<grid, block, 0, stream>>>(...);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(graphExec, stream);
```

关键点：

- `cudaStreamBeginCapture()` 之后，工作不会立刻排进执行队列，而是被追加到内部 capture graph。
- `cudaStreamEndCapture()` 返回最终的 `cudaGraph_t`。
- `cudaStreamLegacy`（NULL stream）不能用于 capture。
- `cudaStreamPerThread` 可以 capture。
- 也可以用 `cudaStreamBeginCaptureToGraph()` 直接 capture 到一个已有图里。

### 4. Cross-stream capture 注意事项

文档明确支持用 event 表达跨 stream 依赖：

- `cudaEventRecord()` 记录到 capture 中时，会变成 captured event。
- 其他 stream 对这个 event 做 `cudaStreamWaitEvent()` 时，也会并入同一个 capture graph。
- 调用 `cudaStreamEndCapture()` 的必须是 origin stream，也就是最初 begin capture 的那个 stream。
- 其他因为 event 关系加入 capture 的 stream，最终必须 rejoin 回 origin stream，否则 capture 失败。

### 5. Capture 期间的常见禁区

官方文档明确提到以下行为是无效或危险的：

- 对正在 capture 的 stream 或 captured event 做同步或执行状态查询。
- 当同一 context 中存在 capture，且相关 stream 不是 `cudaStreamNonBlocking` 创建时，再去使用 legacy stream。
- 在上述情况下调用会隐式触碰 legacy stream 的同步 API，例如同步版 `cudaMemcpy()`。
- 试图把两个独立的 capture graph 通过 event wait 合并。
- 在 capture 期间调用部分尚未支持 graph 的异步 stream API，例如 `cudaStreamAttachMemAsync()`。

如果在 capture 期间发生非法操作：

- 相关 capture graph 会被 invalidated。
- 直到 `cudaStreamEndCapture()` 结束前，继续使用这些 captured stream / event 都会报错。
- `cudaStreamEndCapture()` 会返回错误，并给出 `NULL` graph。

### 6. 实例化与执行

- `cudaGraphInstantiate()` 会对图做校验并生成 `cudaGraphExec_t`。
- 一个 `cudaGraphExec_t` 可以 launch 很多次，不需要每次重新实例化。
- 图执行仍然“挂”在 stream 上，但这个 stream 只负责和其他异步工作建立顺序关系，不限制图内部并行度。

官方还特别强调两点：

- `cudaGraph_t` 不是 thread-safe，对同一个图对象的并发访问需要用户自己保证安全。
- 同一个 `cudaGraphExec_t` 不能和自己并发运行；后一次 launch 会排在前一次之后。

## 更新 instantiated graph

当工作流变了，并不一定要销毁重建整张图。只要拓扑没变，通常优先考虑 Graph Update。

### 1. 什么时候适合 update

适合 update 的典型情况：

- kernel 参数变化
- kernel 使用的指针变化
- memcpy / memset 参数变化
- host callback 参数变化
- 外部 semaphore 参数变化

如果变的是拓扑，就仍然需要重新实例化：

- 节点数量变化
- 节点类型变化
- 依赖边变化

### 2. Whole Graph Update

API：`cudaGraphExecUpdate()`

思路是：拿一张新的 `cudaGraph_t` 去更新旧的 `cudaGraphExec_t`，前提是两张图拓扑完全一致。

官方要求比较严格：

- 新旧图必须拓扑一致。
- 依赖边的指定顺序必须一致。
- sink node 的顺序必须一致。
- 对 capture 场景来说，相关 API 调用顺序也要一致。

适合场景：

- 要更新的节点很多。
- 图来自 stream capture，调用方未必直接持有每个节点句柄。

### 3. Individual Node Update

如果只改少量节点，直接改节点参数更轻量。

文档中列出的常用 API 包括：

- `cudaGraphExecKernelNodeSetParams()`
- `cudaGraphExecMemcpyNodeSetParams()`
- `cudaGraphExecMemsetNodeSetParams()`
- `cudaGraphExecHostNodeSetParams()`
- `cudaGraphExecEventRecordNodeSetEvent()`
- `cudaGraphExecEventWaitNodeSetEvent()`
- `cudaGraphExecExternalSemaphoresSignalNodeSetParams()`
- `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`
- `cudaGraphExecChildGraphNodeSetParams()`

经验上：

- 少量节点更新，优先 individual update。
- 大量节点一起改，或者图来自 library capture，优先 whole graph update。

### 4. Node Enable / Disable

文档提供了“节点启停”机制：

- `cudaGraphNodeSetEnabled()`
- `cudaGraphNodeGetEnabled()`

适用节点类型：

- kernel
- memcpy
- memset

语义上：

- disabled node 可以看作 empty node。
- 关闭节点不会丢掉它的参数。
- 关闭期间改参数，重新 enable 后仍然生效。
- enable 状态不会因为 whole graph update 或 individual node update 自动重置。

这个机制很适合做“超集图”：

- 先建一张包含多种可选路径的大图。
- 每次 launch 前按需要打开或关闭部分节点。

### 5. Update 限制

文档明确提到的几条限制：

- kernel 节点不能改到不同 owning context 的函数。
- 原来不使用 CUDA Dynamic Parallelism 的 kernel，不能更新成使用了 CUDA Dynamic Parallelism 的 kernel。
- memcpy / memset 节点的源和目标所处设备不能变。
- 相关内存必须仍然来自与原来一致的 context。

## Conditional Graph Nodes

Conditional node 让图内的条件分支和循环在 device 端完成，不必每轮都回 host 做判断。

### 1. 三种条件节点

- IF：条件非零则执行 body graph；也可以额外提供 else body。
- WHILE：条件非零时反复执行 body graph，直到条件为零。
- SWITCH：根据条件值选择第 `n` 个 body graph 执行一次。

### 2. Conditional Handle

条件值通过 `cudaGraphConditionalHandle` 表示，使用：

- `cudaGraphConditionalHandleCreate()`

来创建。

重要语义：

- handle 必须和单个 conditional node 关联。
- 条件值可由 device 代码通过 `cudaGraphSetConditional()` 设置。
- 如果创建时带 `cudaGraphCondAssignDefault`，则每次 graph launch 开始时都会把条件值重置为默认值。
- 如果不带这个 flag，则每次 launch 开始时条件值是未定义的，不能假设它会沿用上一次执行结果。
- whole graph update 会更新 handle 的默认值和 flag。

### 3. Body Graph 的限制

官方列出的 body graph 约束：

- 所有节点必须位于同一 device。
- 只允许 kernel、empty、memcpy、memset、child graph、conditional node。
- body graph 中的 kernel 不能使用 CUDA Dynamic Parallelism。
- body graph 中的 kernel 不能做 Device Graph Launch。
- cooperative launch 可以用，但 MPS 不能同时启用。
- conditional node 可以嵌套。

## Graph Memory Nodes

这是 CUDA Graph 里非常值得关注的一部分，因为它把内存生命周期也纳入图模型。

### 1. 基本价值

memory node 允许图自己创建和管理 allocation，使用的是 GPU ordered lifetime 语义，和 `cudaMallocAsync` / `cudaFreeAsync` 的流排序内存模型一致。

文档强调了两个非常重要的性质：

- graph allocation 的虚拟地址在图的生命周期内是固定的，包括重复实例化和重复 launch。
- 底层物理内存可以变，且 CUDA 会复用物理内存，只要生命周期不重叠。

这意味着：

- 图里的其他节点可以直接持有 allocation 得到的地址，不必每次 update 指针。
- 但 allocation 内容不是永久有效的，超出生命周期后继续使用会有严重错误风险。

### 2. 两种创建方式

#### 2.1 显式加 memory node

文档说明可以用 `cudaGraphAddNode()` 创建：

- `cudaGraphNodeTypeMemAlloc`
- `cudaGraphNodeTypeMemFree`

注意依赖关系：

- 使用这段内存的节点必须排在 alloc node 之后。
- free node 必须排在所有使用者之后。

#### 2.2 通过 stream capture

也可以直接 capture：

- `cudaMallocAsync()`
- `cudaFreeAsync()`

如果原始 stream 代码的顺序写对了，那么 capture 进图后，memory node 的依赖关系通常也会是正确的。

### 3. 在分配图之外访问或释放

graph allocation 不要求一定由创建它的那张图来 free。

只要顺序关系建立正确，它可以：

- 被后续普通 stream 操作访问
- 被另一张 graph 访问
- 被 `cudaFree()` / `cudaFreeAsync()` 释放
- 被另一张带 free node 的 graph 释放

但文档明确提醒：

- free 必须严格排在所有 device 访问之后。
- 不能指望“内核内部自己做了一点内存同步”来代替 allocation/free 生命周期排序。
- 如果访问越过生命周期边界，可能会静默读写到别的活跃 allocation 的物理页。

### 4. `cudaGraphInstantiateFlagAutoFreeOnLaunch`

默认情况下，如果一张图上一次执行留下了未释放 allocation，再次 launch 同一图会被 CUDA 阻止，因为同一地址重复分配会泄漏内存。

加上：

- `cudaGraphInstantiateFlagAutoFreeOnLaunch`

后，重新 launch 时会自动插入异步 free，允许继续执行。

适合场景：

- 单生产者、多消费者
- 消费路径随运行时条件变化，生产者很难提前知道谁最终会负责 free

但这不是“自动兜底防泄漏”：

- 图销毁时，行为不会因为这个 flag 改变。
- 应用仍然需要保证最终正确释放，避免真实泄漏。

### 5. 地址复用与物理内存复用

官方把复用分成两层：

- 图内的虚拟地址复用
- 图间的物理页复用

具体理解：

- 生命周期不重叠的 allocation，可能被分配到同一个虚拟地址。
- 不会并发运行的不同 graph，可能映射到同一批物理页。
- 如果程序越界使用“已失效的旧指针”，有机会直接踩到另一个还活着的 allocation。

所以经验上要把 graph allocation 当作“严格受生命周期约束的资源”，而不是普通稳定裸指针。

### 6. 性能和 `cudaGraphUpload`

对含 memory node 的 graph，物理内存映射发生在 launch 阶段，而不是 instantiate 阶段，因为 instantiate 时还不知道最终在哪个 stream 运行。

官方建议关注：

- 如果同一张 graph 一直在同一个 stream 上 launch，CUDA 更容易复用已有映射，减少 remap 成本。
- 如果频繁切换 launch stream，或者做 trim，后续 launch 可能要 remap，成本会更高。

可用：

- `cudaGraphUpload()`

把第一次 launch 的一部分映射成本前移。如果 upload 和后续 launch 用的是同一个 stream，收益最好。

### 7. 物理内存占用和 trim

文档明确说：

- 销毁带 memory node 的 graph，不等于立刻把物理内存还给 OS。

如果确实要回收未使用的 graph memory pool，可用：

- `cudaDeviceGraphMemTrim`

它会释放当前没被活动 graph 使用的物理内存，但代价是后续再 launch 这些图时可能要重新分配和 remap。

### 8. Peer Access

graph allocation 可以声明多 GPU 可访问性，CUDA 会按需把它映射到 peer GPU。

两点值得记：

- Node API 创建 allocation 时，可以在 alloc 参数里指定 `accessDescs`。
- 如果是通过 stream capture 创建 allocation，peer accessibility 记录的是 capture 当时 pool 的可访问性，后面再改 pool 配置不会 retroactively 改变这张图。

### 9. Child Graph 与 Memory Node

文档提到 CUDA 12.9 引入一项能力：把 child graph 所有权转移给 parent graph，此后 child graph 中可以包含 allocation/free node。

但转移后的 child graph 会有额外限制，例如：

- 不能再独立 instantiate 或 destroy
- 不能再作为别的 parent 的 child graph
- 不能作为 `cuGraphExecUpdate` 的参数
- 不能继续新增 memory allocation / free node

## Device Graph Launch

如果你的控制流决策本身就想在 GPU 上完成，CUDA 允许“从 device 发射 graph”。

### 1. 适用场景

官方给出的定位是：

- 在运行时根据数据做决策
- 避免 host-device 往返
- 在 GPU 上实现循环、状态机、工作调度器

### 2. 如何创建可 device launch 的 graph

实例化时需要显式带上：

- `cudaGraphInstantiateFlagDeviceLaunch`

前提约束：

- 图中所有节点必须在同一 device。
- 只允许 kernel、memcpy、memset、child graph。
- kernel 不能使用 CUDA Dynamic Parallelism。
- cooperative launch 可以用，但不能和 MPS 混用。

另外要注意：

- 设备端 launch 之前，graph 必须先 upload 到 device。
- 可以显式 `cudaGraphUpload()`。
- 也可以在实例化时通过 `cudaGraphInstantiateWithParams()` 请求 upload。
- 或者先从 host launch 一次，让 upload 隐式发生。

### 3. Device-side launch 的基本规则

- 设备端和主机端都使用 `cudaGraphLaunch()`。
- 设备端 launch 必须发生在“另一张 graph 的执行过程中”。
- 设备端 launch 是 per-thread 的，多线程都可以发，但同一图通常需要你自己选定一个 thread 负责 launch。

官方还特别写了并发限制：

- 同一张 device graph 不能同时被 device 端 launch 两次，否则返回 `cudaErrorInvalidValue`。
- 同一张图如果被 host 和 device 同时 launch，行为未定义。

### 4. 三种 launch mode

设备端 launch 不能用普通 stream，只能用命名 stream 常量表示模式：

- `cudaStreamGraphFireAndForget`
- `cudaStreamGraphTailLaunch`
- `cudaStreamGraphFireAndForgetAsSibling`

#### 4.1 Fire-and-forget

特点：

- 子图会立刻提交，独立运行。
- 启动图是 parent，被启动图是 child。
- 单次 parent graph 执行期间，总 fire-and-forget graph 数量上限为 `120`。

#### 4.2 Tail launch

这是 GPU 侧串行依赖的关键机制。

因为在 device 侧不能像 host 那样随便 `cudaStreamSynchronize()`，所以 tail launch 提供了“等当前执行环境完整结束后再跑下一个图”的语义。

特点：

- 当前 graph 及其 child 工作全部完成后，tail launch 才会开始。
- 同一图排进去的多个 tail launch 按入队顺序串行执行。

#### 4.3 Tail self-launch

设备端可以把“当前正在执行的 graph 自己”再次以 tail launch 的方式排队，适合写纯 GPU 侧 loop。

相关接口：

- `cudaGetCurrentGraphExec()`

如果当前 kernel 不在 device graph 里执行，这个接口会返回 `NULL`。

#### 4.4 Sibling launch

sibling launch 可以理解为“不是作为当前图的 child，而是作为当前图 parent environment 的 child”。

实际效果：

- 它不会被算进当前 launching graph 的执行环境里。
- 因此不会阻塞当前图已经排好的 tail launch。

### 5. Execution Environment 直觉理解

官方用 execution environment 来解释 device-side launch 的同步模型：

- 一个 graph 在 device 端 launch 后，会拥有自己的 execution environment。
- 这个环境包含图本身的工作，以及它通过 fire-and-forget 生成的 child 工作。
- 只有当整个环境都完成后，相关 tail launch 才能继续推进。

这个概念理解透了，tail / sibling / fire-and-forget 的区别就会清晰很多。

## CUDA User Objects

这部分不是 Graph 的“结构”本身，但和异步资源生命周期管理关系很强。

### 1. 为什么需要它

有些库内部资源管理策略和 graph / stream capture 不兼容，例如：

- 事件池驱动的临时资源复用
- 同步创建、异步销毁
- 资源对象句柄每次提交都变

这会导致：

- capture 时隐藏的同步或禁用 API 被触发
- graph 中很难稳定持有资源句柄

### 2. User Object 的语义

CUDA User Object 本质上是：

- 一个用户提供析构回调的对象
- 外加一个由 CUDA 管理的引用计数

常见 API：

- `cudaUserObjectCreate()`
- `cudaGraphRetainUserObject()`

文档强调：

- graph clone 会复制引用。
- `cudaGraphExec_t` 实例化时会保留源图中的引用。
- 如果 exec 被销毁时还有未同步的异步执行，引用会一直保留到执行真正结束。

适合场景：

- 你要把“某个异步资源活到 graph 真正跑完”为止这件事，交给 CUDA 统一管理。

## 常用 API 速查

### 1. 创建 / capture / launch

- `cudaGraphCreate()`
- `cudaGraphAddNode()`
- `cudaGraphAddKernelNode()`
- `cudaGraphAddMemcpyNode()`
- `cudaGraphAddMemsetNode()`
- `cudaGraphAddHostNode()`
- `cudaStreamBeginCapture()`
- `cudaStreamBeginCaptureToGraph()`
- `cudaStreamEndCapture()`
- `cudaGraphInstantiate()`
- `cudaGraphInstantiateWithParams()`
- `cudaGraphUpload()`
- `cudaGraphLaunch()`

### 2. 更新

- `cudaGraphExecUpdate()`
- `cudaGraphExecKernelNodeSetParams()`
- `cudaGraphExecMemcpyNodeSetParams()`
- `cudaGraphExecMemsetNodeSetParams()`
- `cudaGraphExecHostNodeSetParams()`
- `cudaGraphExecChildGraphNodeSetParams()`
- `cudaGraphNodeSetEnabled()`
- `cudaGraphNodeGetEnabled()`

### 3. 条件节点

- `cudaGraphConditionalHandleCreate()`
- `cudaGraphSetConditional()`

### 4. 内存节点

- `cudaMallocAsync()`
- `cudaFreeAsync()`
- `cudaDeviceGraphMemTrim`
- `cudaGraphInstantiateFlagAutoFreeOnLaunch`

### 5. 设备端 launch

- `cudaGraphInstantiateFlagDeviceLaunch`
- `cudaStreamGraphFireAndForget`
- `cudaStreamGraphTailLaunch`
- `cudaStreamGraphFireAndForgetAsSibling`
- `cudaGetCurrentGraphExec()`

## 工程上怎么选

### 优先用 CUDA Graph 的情况

- 一轮计算由很多短 kernel 组成。
- 同一轮工作会重复很多次。
- 现有 pipeline 已经很稳定，变化主要是参数变化而不是结构变化。

### 优先从 stream capture 开始的情况

- 你已经有一套工作正常的 stream 代码。
- 图结构来自多层 library 调用，手写 node 依赖太费劲。
- 你想先低成本验证 Graph 是否能带来 launch overhead 收益。

### 优先用显式 Graph API 的情况

- 你要精细控制图结构。
- 你要做节点级更新、条件节点、memory node。
- 你希望图结构由程序显式构造，而不是隐式从 stream 代码“录制”出来。

## 容易踩坑的点

- capture 期间不要碰 legacy stream。
- capture 期间不要对 captured stream / event 做同步或 query。
- 多 stream capture 最后一定要 rejoin 回 origin stream。
- 不要把 graph allocation 当成永久稳定指针，超出生命周期继续访问非常危险。
- 含 memory node 的 graph 最好稳定地在同一个 stream 上 launch，减少 remap。
- device graph 不能一边 host launch 一边 device launch。
- `cudaGraph_t` 不是线程安全对象。
- 如果图拓扑变了，不要硬上 update，直接重新 instantiate 更稳。

## 备注

如果后续要把这份备忘继续扩展，建议下一步补两类内容：

- 一份最小可运行示例：手写 Graph API 版本 + stream capture 版本。
- 一份性能对比 checklist：普通 stream 提交 vs graph instantiate 后重复 launch。
