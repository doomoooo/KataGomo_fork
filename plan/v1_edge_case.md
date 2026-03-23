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

### 2. 边界情况 A：窗口重叠

#### 2.1 问题

`v1` 里已经定义：

- 每次 GPU completion 之后，都会发布一个新的 `epoch`
- 并基于新的时间估计计算下一窗口的 `targetOutstandingNN`

真正实现时会遇到两个重叠来源：

1. 多个 infer stream 的完成时间天然交错。
2. 上一个窗口尚未“自然结束”时，新的 completion 已经到来并要求重估。

如果处理不清楚，就会出现两种错误：

- 把旧窗口和新窗口的目标叠加，导致重复放量
- 让旧窗口继续驱动搜索，导致新窗口发布后仍按过期目标生成 request

#### 2.2 决策

本版明确：

- 窗口不是累加对象，而是“最新 completion 发布的权威快照”
- 新窗口永远覆盖旧窗口，不对旧窗口做 delta 累加
- 搜索侧看到的始终只有“当前权威窗口”

也就是说：

- `targetOutstandingNN` 是一个 stock target
- 不是“本轮新增还要补多少”的增量 target

#### 2.3 更新规则

GPU completion 事件必须经过单一串行点处理。第一阶段就直接要求：

- `TensorRTRuntime` 内部必须有一个串行化的 completion/update 路径
- 不允许多个 stream completion 并发直接改写 `SearchSharedState`

每处理一次 completion，按以下顺序执行：

1. 读取当前 GPU 时钟、各 stream 新的可用时间、当前共享区统计值。
2. 用最新快照从头计算新的 `targetOutstandingNN`。
3. `epoch += 1`
4. 覆盖写入：
   - `targetOutstandingNN`
   - `generationOpen`
   - 任何本窗口附带的 metadata
5. 再发布等待中的 completion sender

因此，窗口重叠时的规则非常简单：

- 旧窗口失效
- 新窗口取代它
- 搜索侧永远只对当前 `epoch` 负责

#### 2.4 搜索侧如何响应窗口覆盖

搜索协程需要遵守以下规则：

- 任何“是否继续生成新 request”的决策，只能在 playout 边界读取共享区
- 任何“是否还能再放一个 replacement”的决策，都必须以当前 `epoch` 的值为准
- 如果协程在等待 NN completion 期间窗口被覆盖，它不需要立即被取消
- 它只需要在恢复后：
  - `apply_eval(...)`
  - 再读取最新 `epoch / generationOpen / targetOutstandingNN`
  - 然后按新窗口规则继续或退出

因此，窗口覆盖只影响未来决策，不追溯撤销已经提交的 request。

#### 2.5 为什么不做增量叠加

不采用“新窗口对旧窗口做 +delta / -delta”的原因是：

- 旧窗口本身就可能已经过时
- 多 stream completion 的先后顺序会让 delta 语义非常脆弱
- 真正可靠的输入是“当前时刻的完整系统状态”，不是上一个窗口的残量

所以更稳妥的做法是：

- completion 到来
- 基于最新全量状态重算
- 用新结果整体覆盖旧窗口

#### 2.6 对共享区的要求

为支持这一点，共享区至少要有：

```cpp
struct SearchSharedState {
  std::atomic<uint64_t> epoch;
  std::atomic<int> targetOutstandingNN;
  std::atomic<bool> generationOpen;
  std::atomic<int> outstandingNN;
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
2. `targetOutstandingNN` 每次都按全量状态重算并覆盖，不做 delta 累加。
3. `GammaEstimate` 需要支持初始化先验值。
4. `TensorRTRuntime::TimingModel` 不需要额外的 warmup batch 目标状态。
5. `TensorRTRuntime` 的 batch 发射规则保持单一语义：
   - 有运行中 infer 时，可以等待更多 request
   - GPU 空闲时，只要有 ready request 就立即发射

### 5. `v2` 需要细化的问题

下一版文档需要继续细化：

- completion 串行化路径放在 `TensorRTRuntime` 的哪个具体对象里
- `epoch` 覆盖时搜索协程需要哪些最小同步保证
- `GammaEstimate` 是否要显式暴露 sample count
- `initialInferMs` 应该来自静态配置、机型配置，还是历史 profile
- GPU “忙时可等待，闲时立即发射”在多 stream 下的精确定义应放在哪个对象里
