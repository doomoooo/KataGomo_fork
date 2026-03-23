# 计划

本文档补充 [v1](./v1.md)，专门处理在动手实现前必须先定清楚的边界情况。

## plan_v1_edge_case

状态：首版

### 1. 文档目标

`v1` 已经冻结了主架构：

- 固定 `SearchWorkerPool`
- 单-yield 搜索协程
- GPU completion 先重估目标、后唤醒协程

但在真正实现前，还有两类情况必须先定清楚：

1. 窗口之间发生重叠时，目标数量应如何更新。
2. 冷启动阶段看起来像“无法打满 batch”的情况，是否真的需要单独策略。

这份文档只回答这两件事。

### 2. 边界情况 A：窗口重叠与少量超发

#### 2.1 问题

`v1` 里已经定义：

- 每次 GPU completion 之后，都会发布一个新的 `epoch`
- 并基于新的时间估计计算下一窗口的需求目标

真正实现时会遇到两类并发事实：

1. 多个 infer stream 的完成时间天然交错，导致窗口可能在上一窗口尚未“自然耗尽”时就被覆盖。
2. 搜索协程在单次 playout 内不可打断，因此达到目标之后，仍可能有少量已经在途的 playout 再额外产出一两个 NN request。

如果处理不清楚，就会出现三种错误：

- 把旧窗口和新窗口的目标叠加，导致重复放量
- 让旧窗口继续驱动搜索，导致新窗口发布后仍按过期目标生成 request
- 把少量额外 NN request 视为异常，试图回滚或从局部 target 中“扣掉”

#### 2.2 决策

本版明确：

- 窗口不是累加对象，而是“最新 completion 发布的权威快照”
- 目标不再用局部 `outstanding target` 表示
- 正确性的权威状态改为“全局累计计数 + 当前窗口发布的绝对目标水位线”

也就是说，共享区里与 NN 需求相关的主状态应是：

- `issuedNNEvalCount`
- `completedNNEvalCount`
- `targetIssuedNNEvalCount`

其中：

- `issuedNNEvalCount` 表示全局累计已经逻辑提交的 NN request 数量
- `completedNNEvalCount` 表示全局累计已经 publish completion 的 NN request 数量
- `targetIssuedNNEvalCount` 表示当前窗口发布的绝对目标水位线

搜索侧真正的放量条件是：

- `issuedNNEvalCount < targetIssuedNNEvalCount`

而不是比较一个局部的 `outstandingNN` 是否达到局部目标。

#### 2.3 更新规则

GPU completion 事件必须经过单一串行点处理。第一阶段就直接要求：

- `TensorRTRuntime` 内部必须有一个串行化的 completion/update 路径
- 不允许多个 stream completion 并发直接改写 `SearchSharedState`

每处理一次 completion，按以下顺序执行：

1. 读取当前 GPU 时钟、各 stream 新的可用时间、以及共享区统计值。
2. 更新 GPU 时间估计，并计算新的 `desiredOutstandingNN`。
3. 递增 `completedNNEvalCount`。
4. 计算新的：
   - `targetIssuedNNEvalCount = completedNNEvalCount + desiredOutstandingNN`
   - `generationOpen = (issuedNNEvalCount < targetIssuedNNEvalCount)`，若保留该提示位
5. 发布新的 `epoch`。
6. 再发布等待中的 completion sender。

因此，窗口重叠时的规则非常简单：

- 旧窗口失效
- 新窗口取代它
- 搜索侧永远只对当前 `epoch` 对应的绝对目标水位线负责

#### 2.4 搜索侧如何响应窗口覆盖

搜索协程需要遵守以下规则：

- 任何“是否继续生成新 request”的决策，只能在 playout 边界读取共享区
- 任何“是否还能再放一个 replacement”的权威判断，都必须基于：
  - 当前 `issuedNNEvalCount`
  - 当前 `targetIssuedNNEvalCount`
- 如果协程在等待 NN completion 期间窗口被覆盖，它不需要立即被取消
- 它只需要在恢复后：
  - `apply_eval(...)`
  - 再读取最新 `epoch / issuedNNEvalCount / targetIssuedNNEvalCount`
  - 然后按新窗口规则继续或退出

因此，窗口覆盖只影响未来决策，不追溯撤销已经提交的 request。

#### 2.5 少量额外 NNEval 如何处理

达到目标后出现少量额外 NN request，不是错误，而是单-yield 协程模型的自然结果。

具体语义应当是：

- 某个 playout 在边界产出 request 时，先递增 `issuedNNEvalCount`
- 如果这次递增已经达到或超过 `targetIssuedNNEvalCount`，它可以关闭 `generationOpen` 这一提示位
- 但那些已经在本地 CPU 相位中运行、尚未走到边界的其它 playout，仍可能再产出少量 request

于是可能出现：

- `issuedNNEvalCount > targetIssuedNNEvalCount`

这不是异常，也不需要回滚。它表示：

- 当前系统已经短暂超前于本窗口的目标水位线

接下来的处理也不需要特殊分支：

- 搜索侧因为 `issued >= targetIssued`，不会再主动生成新的 request
- 下一次 GPU completion 会基于最新全局累计状态重新发布新的绝对目标水位线
- 如果新的 `targetIssuedNNEvalCount` 仍然低于当前 `issuedNNEvalCount`，那就继续保持关闭，直到完成计数追上或新窗口提高目标

因此，“窗口重叠”和“少量超发”是可以统一处理的：

- 两者都不做回滚
- 两者都通过“最新窗口覆盖 + 全局累计计数”自然吸收

#### 2.6 为什么不做增量叠加

不采用“新窗口对旧窗口做 +delta / -delta”的原因是：

- 旧窗口本身就可能已经过时
- 多 stream completion 的先后顺序会让 delta 语义非常脆弱
- 少量超发也会使“旧窗口剩余额度”这种局部概念变得不可靠
- 真正可靠的输入是“当前时刻的完整系统状态”，不是上一个窗口的残量

所以更稳妥的做法是：

- completion 到来
- 基于最新全量状态重算
- 用新结果整体覆盖旧窗口

#### 2.7 对共享区的要求

为支持这一点，共享区至少要有：

```cpp
struct SearchSharedState {
  std::atomic<uint64_t> epoch;
  std::atomic<uint64_t> issuedNNEvalCount;
  std::atomic<uint64_t> completedNNEvalCount;
  std::atomic<uint64_t> targetIssuedNNEvalCount;
  std::atomic<bool> generationOpen;
  ...
};
```

如果 `v2` 需要更强的诊断能力，可以再加：

- `windowPublishedAt`
- `windowReason`
- `windowTargetSource`

但这些都不是主路径语义所必需的。

### 3. 边界情况 B：冷启动时无法打满 batch

本节在复核 `v0` 之后，结论已经修正为：

- 这不是一个独立边界条件
- 不需要为它引入专门的 warmup 调度策略
- 真正需要额外处理的，只有估计器初值

#### 3.1 重新审视问题

上一版把冷启动当成一个特殊难题，隐含前提是：

- GPU 调度器在 batch 不够满时，可能会继续等待

但这和 `v0` 想要的语义并不一致。

按照当前冻结的 GPU 侧规则：

- 如果 GPU 上仍有正在运行的 infer，那么继续等待新的 request，以便形成更合适的 batch
- 如果 GPU 当前空闲，那么只要有 request ready，就立即接受当前 batch，不要求满 batch

在这条规则下，冷启动并不会卡住。

#### 3.2 正确结论

冷启动阶段的 partial batch 不是例外，而是正常规则的自然结果。

只要满足：

- 至少有一个 GPU stream 当前空闲
- ready queue 中至少有 1 个 request

那么第一批 infer 就应该立即发出，即使 batch size 只有 1。

因此：

- 不需要单独的 `bootstrap / warmup` 调度模式
- 不需要 `effectiveBatchGoal`
- 不需要 `coldStartQueueDelayMs`
- 也不需要一套“冷启动时先宽松、稳态时再收紧”的 batch 发射判定

#### 3.3 冷启动真正需要的只有一件事：估计器初值

虽然不需要单独调度模式，但冷启动仍然缺少历史样本。

因此 `GammaEstimate` 仍然要有先验值。

至少需要给出：

- `initialPlayoutCpuMs`
- `initialRequestSuccessProb`
- `initialInferMs`

这些值可以来自：

- 配置默认值
- 上次运行持久化的 profile
- 极保守的硬编码先验

但无论如何：

- `initialRequestSuccessProb` 不能是 `0`
- `initialInferMs` 不能是未定义

否则第一轮目标估计会退化成未定义。

这里尤其要注意：

- `initialInferMs` 的作用不是驱动“等满 batch”
- 而是在第一批真实 completion 到来之前，给 GPU 时间模型一个保守的服务时间先验

也就是说，冷启动的第一轮因果链应当是：

1. GPU 侧看到空闲 stream，因此可用时间表里天然有一个 `t = 0` 的槽位。
2. 搜索侧根据当前共享估计，开始生成新的 NN request。
3. 只要 ready queue 出现第一个 request，空闲 GPU 就立即发出第一批 infer，即使 batch 不满。
4. 第一批 completion 返回后，`initialInferMs` 开始被真实样本替换。

#### 3.4 为什么这和 v0 更一致

这版修正后反而更贴近 `v0`：

- 搜索协程的行为规则完全不变
- GPU completion 仍然是唯一的重估时钟
- GPU 调度没有额外 warmup 分支，只有一条统一规则：
  - 忙时等待更好的 batch
  - 闲时立即接受当前 batch

因此，冷启动不是单独模式，而是正常调度规则在“样本仍很少”这一状态下的自然表现。

### 4. 对 v1 的直接补充

基于上述边界规则，`v1` 需要被理解为还隐含以下约束：

1. `SearchSharedState` 的 `epoch` 代表“当前权威窗口版本”，不是累加窗口队列。
2. 正确性的权威状态应是：
   - `issuedNNEvalCount`
   - `completedNNEvalCount`
   - `targetIssuedNNEvalCount`
   而不是局部 `outstanding target`。
3. `targetIssuedNNEvalCount` 每次都按全量状态重算并覆盖，不做 delta 累加。
4. 少量额外 NN request 是允许的，不能回滚；它们必须被计入全局累计计数。
5. `GammaEstimate` 需要支持初始化先验值。
6. `TensorRTRuntime::TimingModel` 不需要额外的 warmup batch 目标状态。
7. `TensorRTRuntime` 的 batch 发射规则保持单一语义：
   - 有运行中 infer 时，可以等待更多 request
   - GPU 空闲时，只要有 ready request 就立即发射

### 5. `v2` 需要细化的问题

下一版文档需要继续细化：

- completion 串行化路径放在 `TensorRTRuntime` 的哪个具体对象里
- `epoch` 覆盖时搜索协程需要哪些最小同步保证
- `GammaEstimate` 是否要显式暴露 sample count
- `initialInferMs` 应该来自静态配置、机型配置，还是历史 profile
- `generationOpen` 是否应保留为独立提示位，还是完全由计数比较即时导出
- GPU “忙时可等待，闲时立即发射”在多 stream 下的精确定义应放在哪个对象里
