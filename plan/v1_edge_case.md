# 计划

本文档补充 [v1](./v1.md)，专门处理在动手实现前必须先定清楚的边界情况。

## plan_v1_edge_case

状态：首版

### 1. 文档目标

`v1` 已经冻结了主架构：

- 固定 `SearchWorkerPool`
- 单-yield 搜索协程
- GPU completion 先重估目标、后唤醒协程

但在真正实现前，还有两类边界情况必须先定规则：

1. 窗口之间发生重叠时，目标数量应如何更新。
2. 冷启动阶段无法打满 batch 时，GPU 和搜索侧应如何操作。

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

#### 3.1 问题

冷启动阶段天然缺少两类信息：

- GPU 侧还没有稳定的 infer 时间估计
- 搜索侧还没有稳定的 `playoutCpuMs / requestSuccessProb`

这会带来一个非常直接的问题：

- 如果坚持“必须等满 batch 才 launch”，系统可能根本拿不到第一批样本
- 如果完全放任 partial batch，又可能把初始估计带偏

因此冷启动不能沿用稳态策略。

#### 3.2 决策

本版明确采用“两阶段策略”：

1. `bootstrap / warmup` 阶段
2. `steady-state` 阶段

冷启动阶段允许 partial batch，而且必须允许；否则无法破局。

#### 3.3 冷启动阶段的三个规则

##### 规则 1：所有估计器都要有先验值

`GammaEstimate` 不能从“未知”开始。

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

否则 target 公式会在启动时退化。

##### 规则 2：冷启动使用 `effectiveBatchGoal`，不是名义满 batch

冷启动阶段不直接把 `preferredBatchSize` 当成硬目标，而是引入：

```cpp
effectiveBatchGoal = f(warmupProgress)
```

其中：

- 刚启动时 `effectiveBatchGoal = 1`
- 随着 GPU 样本和搜索样本增多，逐步向 `preferredBatchSize` 拉升

第一阶段推荐最简单的实现：

```text
warmupProgress = clamp(min(gpuSamples, searchSamples) / warmupSampleCount, 0, 1)
effectiveBatchGoal = 1 + floor((preferredBatchSize - 1) * warmupProgress)
```

这意味着：

- 第一个 batch 可以合法是 size 1
- 第二阶段可能是 size 2/3/4...
- 样本足够后再回到满 batch 目标

##### 规则 3：冷启动不能无限等待“更满一点”

即使 `effectiveBatchGoal > readyRequests`，也不能无限等待。

因此需要一个冷启动下的 partial launch 条件：

- 若某个 infer stream 空闲
- 且已有至少 1 个 ready request
- 且满足以下之一：
  - `readyRequests >= effectiveBatchGoal`
  - `oldestReadyAgeMs >= coldStartQueueDelayMs`

则允许 launch 当前 partial batch。

这条规则的本质是：

- 冷启动时优先拿样本
- 稳态时优先逼近满 batch

#### 3.4 何时退出冷启动

本版先定义一个明确但简单的退出条件：

- `gpuTimingSamples >= warmupSampleCount`
- `searchPlayoutSamples >= warmupSampleCount`

二者都满足之后，切到稳态策略：

- `effectiveBatchGoal = preferredBatchSize`
- 按正常窗口模型重估 `targetOutstandingNN`
- partial launch 只保留正常的延迟/超时兜底，不再走 bootstrap 宽松规则

#### 3.5 为什么这和 v0 相容

这套策略仍然满足 `v0`：

- 搜索协程的行为规则没变
- GPU completion 仍然是唯一的重估时钟
- 唯一变化是：冷启动时 GPU 侧使用更保守的 batch 目标和更宽松的 launch 条件

也就是说：

- `v0` 的主架构不变
- 只是 target 模型在 warmup 阶段采用特化版本

### 4. 对 v1 的直接补充

基于上述边界规则，`v1` 需要被理解为还隐含以下约束：

1. `SearchSharedState` 的 `epoch` 代表“当前权威窗口版本”，不是累加窗口队列。
2. `targetOutstandingNN` 每次都按全量状态重算并覆盖，不做 delta 累加。
3. `GammaEstimate` 需要支持初始化先验值。
4. `TensorRTRuntime::TimingModel` 除了正式估计外，还需要维护：
   - `gpuTimingSamples`
   - `warmupProgress`
   - `effectiveBatchGoal`
5. `TensorRTRuntime` 的 launch 条件要区分：
   - warmup 阶段
   - steady-state 阶段

### 5. `v2` 需要细化的问题

下一版文档需要继续细化：

- completion 串行化路径放在 `TensorRTRuntime` 的哪个具体对象里
- `epoch` 覆盖时搜索协程需要哪些最小同步保证
- `GammaEstimate` 是否要显式暴露 sample count
- `effectiveBatchGoal` 的爬坡函数是否需要非线性版本
- `coldStartQueueDelayMs` 应该是固定值、相对 `inferMs` 的倍数，还是自适应值

