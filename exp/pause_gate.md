# Pause Gate Design

## 目标

`PauseGate` 控制的不是“当前 ready queue 里有多少东西”，而是“搜索侧还允许再向 dispatcher 提交多少个 request”。

当前实现里：

- `submitted_requests` 是累计已经交给 dispatcher 的 request 序号
- `target_requests` 是当前允许提交到哪里的绝对 frontier
- 当 `submitted_requests >= target_requests` 时，搜索入口会被 `PauseGate` 关闭

所以 `PauseGate` 的本质是：

- 用一个全局提交 frontier 控制搜索侧供给
- 在每次 infer 完成后重算下一轮 frontier

## 控制时机

当前控制时机有且只有两个：

1. 搜索线程提交 request 时
2. dispatcher 观察到一次 `infer_done` 时

更具体地说：

- 搜索线程在 `ticket->input_ready.set()` 之后调用：
  - `scheduler.infer.on_request_ready()`
  - `scheduler.maybe_close_gate()`
- dispatcher 在每次 `infer_done` 下降沿调用：
  - `scheduler.infer.on_request_done()`
  - `scheduler.maybe_open_gate()`

因此，两个 frontier 刷新点之间，策略不会改变。

## 为什么决策放在 infer done edge

当前实现故意不等到 `d2h_done`、`output_consumed` 或 “所有 host slot 都空了” 才重算 gate。

原因是：

- gate 的职责是给下一轮 infer 提前备货，而不是等旧结果完全回写后再统一放行
- 如果等到整轮 host ring drain 完才做决策，中间已经空出来的 infer 容量会白白闲置
- 每次 `infer_done` 都意味着时间线上少了一段已知 workload，此时刷新 frontier 最及时

因此，PauseGate frontier 现在在每次 `infer_done` 后立刻更新。`output_consumed` 仍然保留，但它只负责告诉 dispatcher“这一行输出已经被搜索线程真正消费完，可以 reset / 复用 host slot 了”。

## 搜索侧供给估计

当前实现不再单独维护 `p` 和 `q`，也不再用累计 `us_per_request`。

`SearchScheduler` 维护的是每个 worker 的 `requests_per_us` EWMA：

- 成功样本：`1 / accumulated_cpu_us`
- 失败样本：`0`
- 全局速率：所有 worker EWMA 的和

初始化时给一个经验初值：

- 全局 `0.5 / 100us`
- 按 worker 数均摊到各个 EWMA 分片

这意味着 `PauseGate` 决策直接用的是“搜索侧当前平均每微秒能补多少 request”。

## 推理侧未来时间线

`InferScheduler::timeline()` 返回一个惰性生成器。

每个 `TimelinePoint` 包含：

- `demand`
  含义：从当前时刻开始，累计到这个未来点为止，GPU 一共会消耗多少个 request
- `tau_us`
  含义：这个未来点距离当前时刻还有多远

时间线的构造方式是：

1. 先把每条 lane 已经在飞的 `pending_work` 按预测完成时间并到一起
2. 如果某条 lane 的队列已经空了，就假设它之后持续跑满 batch，并按 `infer_batch_us[max_batch_size]` 周期继续外推

所以时间线不是单次批次列表，而是一条“当前在飞 + 之后 steady full-batch”拼起来的未来需求曲线。

## 当前 frontier 更新算法

`InferScheduler::on_request_done()` 的当前算法完全按未来 batch 点扫描。

记：

- `first_point = timeline.next()`
- `gpu_requests = first_point.demand`

这里 `first_point` 是硬约束：

- 它对应“下一次 infer 之前最后一次还能抬 target 的机会”
- 所以第一批需求必须先放行

之后继续向后扫描未来 batch 点。对每个点：

1. 计算该点相对 `first_point` 的剩余需求

$$
\text{remaining\_demand} = \text{point.demand} - \text{first\_point.demand}
$$

2. 用搜索侧当前速率估计，从 `first_point` 到这个点之间 CPU 平均还能补出的 request 数

$$
\text{cpu\_mean\_requests} =
(\text{point.tau\_us} - \text{first\_point.tau\_us}) \cdot \text{cpu\_requests\_per\_us}
$$

3. 计算 starvation 概率

$$
\Pr(\mathrm{Poisson}(\text{cpu\_mean\_requests}) < \text{remaining\_demand})
$$

4. 如果这个概率大于阈值 `1e-5`，则把当前 frontier 推到这个点：

$$
\text{gpu\_requests} \leftarrow \text{point.demand}
$$

这一步的语义是：

- 找到最后一个“如果这次不提前放行，等到 `first_point` 再补就会太危险”的未来点
- 当前轮只需要把 frontier 推到这个“最后危险点”

## Burst Gate

当前实现不会无限向后扫描。

停止条件不是旧文档里的 slack / steady-rate 证书，而是一个更直接的 gate：

- 看当前扫描点相对“当前已选 frontier”又额外多了多少 demand
- 如果这个增量已经超过一轮 `max_batch_burst`，就停止继续向后扫

这里的 `max_batch_burst` 指的是：

- 所有 lane 各自再跑一轮满 batch 时，GPU 侧累计会再消耗多少个 request
- 也就是“本轮 decision 最多再替未来多背一轮”的上界

代码里的判断是：

```cpp
point.demand - gpu_requests > max_batch_burst
```

这里的含义是：

- 如果更远的未来风险已经需要再多放一整轮以上的 request
- 那么它不属于当前这一轮 decision 的责任
- 把它留给下一次 `on_request_done()` 处理

因此，当前算法不是“求一个全局最优 horizon”，而是：

- 这轮先保证最近的一段未来时间线
- 更远的风险分批交给后续 decision

## target_requests 的当前语义

当前实现里：

```cpp
target_requests.fetch_add(gpu_requests);
```

也就是说：

- `target_requests` 是一个绝对 frontier
- `on_request_done()` 每次不是重设它，而是在当前 frontier 基础上再向前推 `gpu_requests`

所以：

- `submitted_requests` 单调增加
- `target_requests` 也单调向前推进
- `target_requests - submitted_requests` 表示当前 gate 还允许再放进来多少 request

## 当前实现明确没有做的事

为了避免和旧文档混淆，这里明确列出当前实现**没有**做的事：

- 不再找“最紧 slack 点”
- 不再反解“为了刚好压到 `1e-5` 该给多少 target request”
- 不再做 `steady_gpu_requests_per_us` 的长期供需比较
- 不再做 batch 对齐或 batch-step 向上取整
- 不再用 `remaining_spare_time_us` 当早停证书

当前 `PauseGate` 设计就是：

- 搜索侧用 `requests_per_us` EWMA 估计供给
- 推理侧用惰性 timeline 给出未来 batch 点
- 在这些未来点上直接算 Poisson starvation probability
- 找到最后一个危险点
- 再用 `burst` gate 截断扫描范围

这就是当前代码真实执行的策略。
