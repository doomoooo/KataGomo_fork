# Schedlab Design

## 目标

`exp/schedlab` 是一个最小化的异步调度实验工程。

它只保留三件事：

1. 搜索线程如何用 `stdexec` 维持自复制 playout。
2. dispatcher 如何把 host ring、batch seal、lane 选择和 `H2D -> infer -> D2H` 串起来。
3. scheduler 如何把搜索侧供给估计和推理侧未来时间线拼起来，并据此控制 `PauseGate`。

当前版本故意不做：

- 场景系统
- trace / 指标系统
- 参数漂移与突变
- 自动观测与回归框架

一次进程启动就是一次测试，所有参数都从命令行进入。

## 文件结构

主模块：

- `exp/schedlab/include/schedlab/search.hpp`
- `exp/schedlab/src/search.cpp`
- `exp/schedlab/include/schedlab/dispatcher.hpp`
- `exp/schedlab/src/dispatcher.cpp`
- `exp/schedlab/include/schedlab/trt_backend.hpp`
- `exp/schedlab/src/trt_backend.cpp`
- `exp/schedlab/include/schedlab/scheduler.hpp`
- `exp/schedlab/src/scheduler.cpp`
- `exp/schedlab/src/main.cpp`

辅助设施：

- `exp/schedlab/include/schedlab/utils/pause_gate.hpp`
- `exp/schedlab/include/schedlab/utils/one_shot_event.hpp`
- `exp/schedlab/include/schedlab/utils/time_types.hpp`
- `exp/schedlab/include/schedlab/utils/mock_phase_runner.hpp`
- `exp/schedlab/src/mock_phase_runner.cpp`

`PauseGate` 的控制细节单独写在 [pause_gate.md](./pause_gate.md)。

## 运行入口

唯一入口是 `schedlab_run`。

示例：

```bash
./build/schedlab-trt-redesign/schedlab_run \
  --run-ms 5000 \
  --workers 48 \
  --batch-size 8 \
  --host-slots 36 \
  --lanes-per-device 1 \
  --cuda-devices 0,1,2 \
  --descend-us 50 \
  --preprocess-us 3 \
  --postprocess-us 3 \
  --ascend-us 50 \
  --success-rate 0.95
```

## 搜索侧

### SearchRuntime

`SearchRuntime` 为每个搜索 worker 持有：

- 一个 `exec::single_thread_context`
- 一个 `exec::async_scope`
- 一个 `worker_id`

长期运行的任务是 `root_playout()`。

### Root Playout 流程

当前实现中的单条 playout 流程是：

1. `co_await scheduler.pause_gate.async_wait()`
2. 新建局部 `SearchPlayoutState`
3. 执行 `playout_descend()`
4. 若本轮不需要 NN：
   - 直接 `playout_ascend()`
   - `finish(false)` 并把样本提交给 `SearchScheduler`
   - 立即补 spawn 下一条 root playout
5. 若需要 NN：
   - 暂停 CPU 计时
   - `co_await dispatcher.acquire_ticket()`
   - 恢复 CPU 计时并执行 `preprocess()`
   - `ticket->input_ready.set()`
   - `scheduler.infer.on_request_ready()`
   - `scheduler.maybe_close_gate()`
   - 立刻补 spawn 下一条 root playout
   - 暂停 CPU 计时并等待 `ticket->output_ready`
   - 恢复 CPU 计时并执行 `postprocess()` 与 `playout_ascend()`
   - `finish(true)` 并把样本提交给 `SearchScheduler`
   - 最后 `ticket->output_consumed.set()`

这里的关键点是：

- 搜索 CPU 统计只覆盖 `descend / preprocess / postprocess / ascend`
- 等 ticket、等 GPU 输出都不计入搜索 CPU 样本
- `output_consumed` 故意放在 `playout_ascend()` 之后，确保旧结果真正并入搜索状态后，dispatcher 才会触发下一次 gate 决策

### SearchScheduler

`SearchScheduler` 只做一件事：维护搜索侧 `requests_per_us` 估计。

当前实现不是累计均值，而是“每个 worker 一个 EWMA 分片”：

- 每个 worker 有一个 `requests_per_us_ewma`
- 初值是全局 `0.5 / 100us`，按 worker 数均摊
- 一条 playout 的样本值是：
  - 成功产出 request：`1 / accumulated_cpu_us`
  - 未产出 request：`0`
- `requests_per_us()` 返回所有 worker EWMA 的和

因此，scheduler 当前直接使用“搜索侧 request 生成速率”，而不是旧版本里的 `p/q` 双统计或 `us_per_request` 累计均值。

## 推理侧

### Dispatcher

`Dispatcher` 运行在自己的 `exec::single_thread_context` 上，负责：

- host ring 管理
- 当前 open host slot 的 row 分配
- 批量 seal
- lane 选择与提交
- completion 轮询
- 向 `Scheduler` 上报 `request_ready / infer_submitted / infer_done / request_done`

### Host Ring

每个 `HostSlotState` 包含：

- `BackendHostSlot`
- `assigned_rows`
- `sealed`
- `input_ready_count`
- `output_consumed_count`

`RequestState::input_ready` 和 `RequestState::output_consumed` 会绑定到 host-slot 级计数器上，因此 dispatcher 只需要等待计数达到 `assigned_rows`。

### acquire_ticket()

`acquire_ticket()` 的当前逻辑是：

1. 切到 dispatcher lane
2. 等待 `current_host_slot` 未被 seal
3. 给当前 host slot 分配一行
4. 如果这一行让 batch 满了，立即发车
5. 否则如果存在完全空闲的 group，也立即把当前部分 batch 发车

它返回的是一行固定好的 `RequestState*`，其输入输出地址和同步句柄都已经落在对应 host slot 上。

### infer_coro()

每个被 seal 的 host slot 会启动一条 `infer_coro()`，顺序流程是：

1. 等本批所有 `input_ready`
2. 如果调用方指定了 lane，就用指定 lane；否则通过 `InferScheduler::select_lane()` 选预测最早完成的 lane
3. `backend.make_launch(lane)`
4. 等 `backend.submit_ready(launch)`
5. `submit_h2d`
6. 等 `h2d_done`
7. `submit_infer`
8. `scheduler.infer.on_infer_submitted(...)`
9. 等 `infer_done`
10. `scheduler.infer.on_infer_done(...)`
11. 若刚刚完成的 group 已完全空闲，且当前 open host slot 已经积累了至少 1 行，则立刻把当前 open slot 发车
12. `submit_d2h`
13. 等 `d2h_done`
14. 逐行发布 `output_ready`
15. 等整批 `output_consumed`
16. reset host slot
17. 在每次 `infer_done` 下降沿，调用：
    - `scheduler.infer.on_request_done()`
    - `scheduler.maybe_open_gate()`

这里的决策触发点就是每次 `infer_done` 的下降沿，而不是“所有 host slot 都重新空闲”的时刻。也就是说，PauseGate frontier 会随着每次 infer 完成持续前推，而不是等整轮 host ring drain 完才统一刷新。

### TrtBackend

`TrtBackend` 只负责具体的 TensorRT / CUDA 资源与提交，不做调度。

它负责：

- TensorRT plan cache
- 多 GPU / 多 lane / 多 bank 初始化
- host slot 分配与释放
- `submit_ready / h2d_done / infer_done / d2h_done`
- `submit_h2d / submit_infer / submit_d2h`
- 启动期 benchmark，给每条 lane 的每个 `batch_size` 提供初始 infer workload

`Scheduler` 只看到 opaque `BackendLaunch` 与完成 token，不直接碰 CUDA/TRT 细节。

## Scheduler

### Search 侧状态

`SearchScheduler` 提供：

- `make_new_state()`
- `submit_state(worker_id, playout_state)`
- `requests_per_us()`

它不再暴露 timer 工厂、开始/暂停接口或独立的 `SearchEstimatorSnapshot`。

### Infer 侧状态

`InferScheduler` 当前维护：

- 每条 lane 的 `InferEstimator`
- 每条 lane 的 `pending_work`
- 每条 lane / group 的 `inflight_count`
- `submitted_requests`
- `target_requests`
- `max_batch_burst`

其中 `max_batch_burst` 的含义是：

- 所有 lane 各自跑一轮满 batch 时，总共会再吞掉多少个 request
- 也是单次 `on_request_done()` 允许继续向远处扫描的最大增量 gate

`InferEstimator` 当前建模的是：

- `infer_batch_us[batch_size]`
- `last_infer_done_us`
- `ewma_jitter_us`
- `last_prediction_error_us`

初始化时，`infer_batch_us[batch_size]` 直接用 backend benchmark 值写入；运行中每次 `infer_done` 再用真实样本更新。

### TimelineIterator

`InferScheduler::timeline()` 返回一个惰性时间线生成器。

它的语义是：

- 每次 `next()` 给出一个 `TimelinePoint{demand, tau_us}`
- `demand` 是从“现在”开始累计到这个未来点的 GPU request 需求
- `tau_us` 是这个未来点距离当前时刻的时间偏移

时间线先消费各 lane 已经在 `pending_work` 队列里的预测完成点；队列耗尽后，再按“该 lane 以后持续跑满 batch”继续外推。

### PauseGate 控制回路

`Scheduler` 本身只暴露三个和 gate 有关的接口：

- `request_stop()`
- `maybe_close_gate()`
- `maybe_open_gate()`

当前控制回路是：

1. 搜索线程每提交一个 request：
   - `submitted_requests.fetch_add(1)`
   - 若 `submitted_requests >= target_requests`，则关闭 `pause_gate`
2. dispatcher 在每次 `infer_done` 之后：
   - 调 `InferScheduler::on_request_done()`
   - 再调 `maybe_open_gate()`

`InferScheduler::on_request_done()` 的当前算法不是“反解最小库存阈值”，而是：

1. 取时间线第一个点 `first_point`
2. 先把 `gpu_requests` 设为 `first_point.demand`
3. 继续向后扫描未来 batch 点
4. 对每个未来点，计算：
   - `remaining_demand = point.demand - first_point.demand`
   - `cpu_mean_requests = (point.tau_us - first_point.tau_us) * cpu_requests_per_us`
   - `starvation_probability = P(Poisson(cpu_mean_requests) < remaining_demand)`
5. 若该点 `starvation_probability > 1e-5`，则把 `gpu_requests` 更新到这个点的 `demand`
6. 若当前扫描点相对“当前已选 target”的额外增量已经超过 `max_batch_burst`，就停止继续向后扫，把更远的风险留给下一次 `on_request_done()`
7. 扫描结束后执行 `target_requests.fetch_add(gpu_requests)`

因此，当前实现的 `target_requests` 语义是：

- 它不是 ready queue 长度
- 它是“累计已提交 request 序号”上的一个绝对 frontier
- `target_requests - submitted_requests` 表示在下一次关 gate 前，搜索侧还允许再往 dispatcher 送多少个 request

## PauseGate 与基础设施

### PauseGate

`PauseGate` 是一个多等待者 `manual-reset gate`：

- 打开时，新 waiter 直接通过
- 关闭时，新 waiter 会挂到 waiter 链表
- `force_open()` 是 sticky 的，后续不会再重新关闭
- 析构时会强制打开并放行所有等待者

当前 `schedlab` 里它只用在搜索 root playout 的入口，不再承担 host slot gate 的角色。

### OneShotEvent / RequestCounter

`RequestState` 里保留的是 request 级接口：

- `input_ready`
- `output_ready`
- `output_consumed`

但其中 `input_ready` 和 `output_consumed` 底层都绑定到 host-slot 级计数器上，所以 dispatcher 等的是整批计数，而不是逐个 row 的 condition variable。

## 当前非目标

当前版本仍然明确不做：

- trace / metric / log 框架
- 复杂场景系统
- 自动回归 harness
- 旧版 scheduler 里那些更重的 estimator 抽象

文档里的算法与接口描述都以 `exp/schedlab/src/scheduler.cpp`、`exp/schedlab/src/search.cpp`、`exp/schedlab/src/dispatcher.cpp` 当前实现为准。
