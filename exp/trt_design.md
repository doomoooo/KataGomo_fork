# TensorRT Dispatcher 最终设计

## 1. 目标

把 TensorRT 推理路径收成一个模块，对外只暴露单请求接口：

```cpp
class SingleRequestHandle {
public:
  void* inputs_mem_addr;
  void* outputs_mem_addr;
  Event input_mem_ready;
  Event output_mem_ready;
  Event output_mem_consumed;
};

SingleRequestHandle getOutput();
```

caller 的使用方式固定为：

```cpp
auto h = getOutput();
preprocess(data, h.inputs_mem_addr);
h.input_mem_ready.set();
h.output_mem_ready.wait();
postprocess(h.outputs_mem_addr, data);
h.output_mem_consumed.set();
```

模块内部负责：

1. 把单请求聚合成 batch
2. 在多 GPU / 多 slot 之间做调度
3. 管理 host pinned ring
4. 管理 device bank、TensorRT context、stream、event
5. 提交 H2D / infer / D2H
6. 把 bank 级完成信号转换成 caller 可见的 `output_mem_ready`

## 2. 设计结论

这版最终设计固定为：

- 使用单线程 dispatcher
- host 侧使用全局 pinned ring
- device 侧按 `slot -> 2 banks` 组织
- 每个 slot 共享一条 `infer_stream`，每个 bank 各自维护 `io_stream`
- I/O 主路径使用两次 `cudaMemcpyBatchAsync`
- 满 batch 且 host/device 都已经是 packed slab 时，优化到单次 `cudaMemcpyAsync` slab copy
- TensorRT binding 地址初始化后固定，不在运行期修改
- 如果 infer 侧启用 CUDA Graph，则每个 bank 都有自己独立的 `IExecutionContext + cudaGraphExec_t`
- dispatcher 只靠事件和内部状态机推进，不使用 CUDA host callback

一句话概括：

`单线程 dispatcher + 全局 host ring + 多 GPU 多 slot + 每 slot 双 bank + 固定地址 TRT context + H2D/infer/D2H 显式调度`

## 3. 非目标

已删除。文档中不需要提醒哪些“不是”目标。

## 4. TensorRT 对象层级

### 4.1 `IRuntime`

进程级对象。

职责：

- 反序列化 TensorRT plan
- 作为执行期入口

建议：

- 进程内通常维护一个全局 `IRuntime`

### 4.2 `ICudaEngine`

模型级对象。

职责：

- 表示优化后的模型
- 持有 tactic / kernel 选择
- 持有 tensor 元信息和 profile 信息

建议：

- 每个 device / model variant 对应一份 engine
- 同一个 engine 可以被多个 bank 共享

### 4.3 `IExecutionContext`

执行实例级对象。

职责：

- 持有当前运行时状态
- 持有当前 shape / profile
- 持有当前 tensor address 绑定
- 执行 `enqueueV3()`

关键约束：

- context 是有状态的，不适合在多个 bank 之间复用
- 如果 infer 侧启用 CUDA Graph capture，一张 captured TRT graph 应绑定到它自己的 context

因此最终映射是：

- `Runtime`：进程级
- `Engine`：模型级
- `Context`：bank 级

## 5. 总体对象模型

```text
Dispatcher
  ├── HostRing
  │     ├── HostSlot[0]
  │     ├── HostSlot[1]
  │     └── ...
  └── DeviceSlot[]
        ├── Slot(device=0, slot=0)
        │     ├── Bank ping
        │     └── Bank pong
        ├── Slot(device=0, slot=1)
        └── ...
```

### 5.1 `HostSlot`

一个 `HostSlot` 表示“一次 batch dispatch 的 host 侧承载体”。

它包含：

- packed host input slab
- packed host output slab
- 该 batch 内每个 request row 的元信息
- generation
- `assigned_rows`
- `input_ready_rows`
- `output_ready_rows`
- `output_consumed_rows`

### 5.2 `DeviceSlot`

一个 `DeviceSlot` 是 dispatcher 视角下的调度单位。

它对应：

- 一张卡上的一个逻辑执行 lane
- 一个 slot 里最多同时只有一个 infer 处于活跃阶段
- 但前一批的 D2H 可以与下一批的 H2D / infer overlap

它包含：

- `device_id`
- `slot_index`
- 共享 `ICudaEngine*`
- 一条 slot 级 `infer_stream`
- `Bank ping`
- `Bank pong`
- `next_launch_bank`
- `active_infer_bank`
- 调度 telemetry

### 5.3 `Bank`

一个 bank 是最小的可执行资源单元。

它固定绑定：

- 一个 device input slab
- 一个 device output slab
- 一个 `IExecutionContext`
- 一条 bank 级 `io_stream`
- 一个 `h2d_done_event`
- 一个 `infer_done_event`
- 一个 `d2h_done_event`
- 可选：
  - 一个 `cudaGraphExec_t`，仅当 infer 侧启用 graph

一个 bank 在任意时刻最多承载一批请求。

## 6. Host 内存布局

### 6.1 全局 ring

host 侧维护一个全局 pinned ring。

默认大小：

```text
3 * sum(gpu.batch * gpu.slots) over all gpus
```

这个大小的目的不是数学最优，而是给：

- 当前正在 fill 的 host slot
- 已经 dispatch 但尚未 `output_consumed` 的 host slot
- 下一轮可继续接收请求的 host slot

留出足够的流水空间。

### 6.2 单个 `HostSlot` 的布局

每个 `HostSlot` 是一段连续 pinned host memory：

```text
  +-----------------------------------------------------------------------------+
  | packed input slab | packed output slab | per-row metadata[] | slot metadata |
  +-----------------------------------------------------------------------------+
```

规则：

- input slab 和 output slab 在物理上连续放置
- 每个 tensor 起始地址按 `256B` 对齐
- 布局在所有 host slot 中完全一致
- row view 只是映射到 packed slab 内部的对应 offset，不单独复制

### 6.3 Row view

`SingleRequestHandle.inputs_mem_addr` / `outputs_mem_addr` 在语义上不是“独立缓冲区”，而是：

- 指向该 request row 在 host slot 内的视图起点
- preprocess / postprocess 通过这个视图写入或读取本 row 数据
- 同时这些 row view 又落在整 batch 的 packed slab 上

这样才能同时满足：

1. caller 以“单请求”方式访问
2. dispatcher 在满 batch 时做“整 slab 连续 copy”

### 6.4 Packed slab 约束

对于当前 `b18tf, batch=8` 的测量：

- 输入 raw `254752 B`，packed `254816 B`
- 输出 raw `81728 B`，packed `82208 B`

padding 税很小，因此 packed slab 是可接受的长期方案。

## 7. Device 内存布局

### 7.1 Bank 内布局

每个 bank 持有固定地址的 device slab：

```text
input slab  -> [tensor0][pad][tensor1][pad]...
output slab -> [tensor0][pad][tensor1][pad]...
```

TensorRT 的各个 tensor address 在初始化时一次性绑定到这些 offset：

```cpp
exec->setTensorAddress("input_spatial", input_base + offset_input_spatial);
exec->setTensorAddress("input_global",  input_base + offset_input_global);
exec->setTensorAddress("out_policy",    output_base + offset_out_policy);
...
```

运行期不再改地址。

### 7.2 为什么按 bank 固定地址

这是为了满足两件事：

1. host 侧逻辑尽量简单，不在热路径改 TensorRT binding
2. infer graph 如果启用，capture 后地址保持稳定

因此：

- 切换 host slot：靠 memcpy 的源/宿地址变化
- 切换 device bank：靠选择不同 bank，对应不同 context / stream / graphExec

## 8. 事件系统

## 8.1 对外事件

每个 request handle 暴露三个事件：

- `input_mem_ready`
  - caller 写完输入后置位
- `output_mem_ready`
  - dispatcher 确认本 row 的输出已 host-visible 后置位
- `output_mem_consumed`
  - caller 读完输出后置位

这三个事件都是：

- one-shot
- generation-tagged

不能跨 generation 复用状态。

## 8.2 对内事件

每个 bank 维护三个 CUDA event：

- `h2d_done_event`
  - 记录在 H2D 之后
  - 作用：dispatcher 在 host 侧观察到它完成后，才把该 bank 的 infer 提交到本 slot 的 `infer_stream`
- `infer_done_event`
  - 记录在 infer 之后
  - 作用：dispatcher 在 host 侧观察到它完成后，启动该 bank 的 D2H
- `d2h_done_event`
  - 记录在 D2H 之后
  - 作用：表示输出已经回到 host，随后可以置位 caller 可见的 `output_mem_ready`

### 8.3 为什么要拆成三个事件

如果只保留一个 `done_event`：

- dispatcher 无法把：
  - H2D 完成
  - infer 完成
  - D2H 完成
 这三个阶段分开推进
- 这样会损失 slot 级 infer stream 的排队能力，也会让 H2D / D2H overlap 模型说不清楚

拆成三层以后：

- `h2d_done_event` 用于把请求接到 slot 的 infer stream 上
- `infer_done_event` 用于推进关键路径
- `d2h_done_event` 用于 bank 资源回收

### 8.4 Host slot 与 bank 的配合

注意三层完成语义是不同的：

1. `h2d_done`
   - 只表示 device input slab 已经准备好
   - dispatcher 可以把本批 infer 挂到 slot 的 infer stream 上
2. `infer_done`
   - 表示本批 infer 关键段结束
   - dispatcher 可以启动本批 D2H
3. `d2h_done`
   - 表示输出已经回到 host
   - dispatcher 可以置位 `output_mem_ready`
   - bank 可复用
4. `output_consumed`
   - 表示 caller 真的把 host 输出消费完
   - host slot 可复用

所以：

- bank 复用不需要等 `output_consumed`
- host slot 复用必须等所有 row 的 `output_mem_consumed`

## 9. Dispatcher wait policy

dispatcher 自己只有一个线程，但等待策略可配置：

- `spin`
  - busy-poll
  - 用于极限延迟实验
- `park`
  - 基于事件/条件变量阻塞等待
  - 用于评估 event-driven overhead

无论哪种 policy：

- 逻辑状态机完全相同
- 不能维护两套不同实现

## 10. 初始化流程

初始化顺序固定为：

1. 创建全局 `IRuntime`
2. 为每张卡预热 context：

```cpp
cudaSetDevice(device);
cudaFree(0);
```

3. 反序列化 engine
4. 创建 host pinned ring
5. 为每个 `DeviceSlot` 创建两个 bank
6. 为每个 `DeviceSlot` 创建一条共享 `infer_stream`
7. 对每个 bank：
   - `cudaSetDevice(device)`
   - `cudaMalloc(input_slab)`
   - `cudaMalloc(output_slab)`
   - 创建 `IExecutionContext`
   - 创建 bank 级 `io_stream`
   - 创建 `h2d_done_event`
   - 创建 `infer_done_event`
   - 创建 `d2h_done_event`
   - `setTensorAddress(...)`
   - 如果启用 infer graph：
     - 以该 bank 的 context 和地址 capture 一张独立 graph
     - instantiate 为该 bank 自己的 `graphExec`
     - 运行时始终把它 launch 到本 slot 的 `infer_stream`

## 11. 调度模型

调度对象不是“裸 GPU”，而是：

```text
slot = (device, logical slot)
bank = slot.ping | slot.pong
```

dispatcher 先选 slot，再在 slot 内选当前可用 bank。

每个 slot 维护：

- `predReadyAtUs`
- `ewmaH2dUs`
- `ewmaInferUs`
- `ewmaD2hUs`
- `ewmaJitterUs`
- `lastPredictionErrorUs`

评分公式：

```text
pred_finish(slot) =
  max(now_us + submit_budget_us, predReadyAtUs)
  + pred_h2d_us
  + pred_infer_us
  + pred_d2h_us
  + uncertainty_penalty_us
```

其中：

- `pred_h2d_us / pred_d2h_us`
  - 满 batch slab path 时用 single-memcpy 历史值
  - 其他情况用 batch-memcpy 历史值
- `pred_infer_us`
  - 用 slot 的 infer EWMA
- `uncertainty_penalty_us`
  - 基于 jitter 和最近误差

dispatcher 只需要扫 slot，不需要扫更复杂的全局状态。

## 12. H2D / infer / D2H 的最终提交方式

## 12.1 主路径：batch memcpy

### H2D

默认路径使用一次 `cudaMemcpyBatchAsync` 提交整个 batch 的输入 tensor copy：

```cpp
cudaMemcpyBatchAsync(
  dsts,
  srcs,
  sizes,
  count,
  &attrs,
  attrStarts,
  attrCount,
  &failIdx,
  bank.io_stream
);
```

属性固定为：

- `srcAccessOrder = cudaMemcpySrcAccessOrderAny`
- `flags = cudaMemcpyFlagPreferOverlapWithCompute`

H2D 提交后，立即记录：

```cpp
cudaEventRecord(bank.h2d_done_event, bank.io_stream);
```

### infer

dispatcher 在 host 侧等待 `bank.h2d_done_event` 完成后，才把 infer 提交到本 slot 的 `infer_stream`。

infer 只有两种实现方式：

1. 非 graph 模式

```cpp
exec.setInputShape(...);
exec.enqueueV3(slot.infer_stream);
```

2. infer graph 模式

```cpp
cudaGraphLaunch(bank.graphExec, slot.infer_stream);
```

但 graph 模式下，`graphExec` 必须是 bank 自己的那一份。

### infer 完成标记

infer 提交后，立即在同一条 stream 上记录：

```cpp
cudaEventRecord(bank.infer_done_event, slot.infer_stream);
```

这里有个关键语义：

- `slot.infer_stream` 是 slot 级共享资源，不是 bank 私有资源
- dispatcher 不需要等这个 stream 空闲再提交
- 如果前一 bank 的 infer 还在运行，新的 infer 会依靠 CUDA stream 的串行语义自然排队

### D2H

dispatcher 在 host 侧等待 `bank.infer_done_event` 完成后，再启动本 bank 的 D2H。

默认路径再使用一次 `cudaMemcpyBatchAsync` 提交所有输出 tensor：

```cpp
cudaMemcpyBatchAsync(..., bank.io_stream);
```

最后记录：

```cpp
cudaEventRecord(bank.d2h_done_event, bank.io_stream);
```

## 12.2 满 batch fallback：single slab memcpy

当下面条件同时成立时：

- 请求是满 batch
- preprocess 直接写 packed host input slab
- postprocess 直接读 packed host output slab
- host / device slab 排布完全一致
- tensor 起始地址都已做 `256B` 对齐

则切换到：

```cpp
cudaMemcpyAsync(bank.input_slab, host_slot.input_slab, input_packed_bytes, cudaMemcpyHostToDevice, bank.io_stream);
cudaEventRecord(bank.h2d_done_event, bank.io_stream);
wait(h2d_done_event);
infer on slot.infer_stream;
cudaEventRecord(bank.infer_done_event, slot.infer_stream);
wait(infer_done_event);
cudaMemcpyAsync(host_slot.output_slab, bank.output_slab, output_packed_bytes, cudaMemcpyDeviceToHost, bank.io_stream);
cudaEventRecord(bank.d2h_done_event, bank.io_stream);
```

这样 host 侧提交开销最低。

## 13. `getOutput()` 路径

dispatcher 始终维护一个当前正在 fill 的 `HostSlot`。

当 caller 调用 `getOutput()`：

1. 在当前 host slot 中分配一个 row
2. 返回绑定该 row view 的 `SingleRequestHandle`
3. 如果这次分配使 host slot 满 batch：
   - 立刻挂起一个异步 dispatch 分支
4. ring head 移动到下一个 host slot
5. 在把新 host slot 暴露给 caller 前，确保该位置上一 generation 已全部 `output_consumed`

这条路径本质串行，而且只做：

- row reservation
- generation 检查
- 必要的状态推进

不做重操作。

## 14. 单个 slot 内 2 个 bank 的管理

这是最终设计里最关键的补充。

## 14.1 基本约束

每个 slot 内：

- 只有两个 bank：`ping` 和 `pong`
- 每个 bank 有自己的：
  - input/output slab
  - context
  - io stream
  - h2d/infer/d2h event
  - optional graphExec
- 整个 slot 只有一条共享的 `infer_stream`
- 同一个 slot 在任一时刻最多只有一个 infer 处于关键执行阶段
- 后一 bank 的 H2D 可以和前一 bank 的 infer overlap
- 前一 bank 的 D2H 可以和后一 bank 的 infer overlap
- 后一 bank 的 infer 可以在 `infer_stream` 忙碌时提前挂上去，依靠 stream 串行语义自然排队

## 14.2 典型时间线

假设当前 slot 先在 `ping` 上跑 batch A：

```text
ping.io_stream:
  H2D(A) -> record h2d_done(A)

slot.infer_stream:
  wait host-side h2d_done(A) -> infer(A) -> record infer_done(A)
```

dispatcher 观察到 `h2d_done(A)` 后：

1. 在 `slot.infer_stream` 上提交 A 的 TRT graph

如果这时 batch B 已经 ready，dispatcher 可以立刻在 `pong.io_stream` 上提交 `H2D(B)`。

于是 A 的 infer 期间，时间线可以变成：

```text
ping.io_stream:
  H2D(A) -> record h2d_done(A)

pong.io_stream:
  H2D(B) -> record h2d_done(B)

slot.infer_stream:
  infer(A) -> ...
```

dispatcher 观察到 `h2d_done(B)` 后，不要求 `infer_stream` 当前空闲，而是直接把 B 的 infer 挂到同一条 stream 上：

```text
slot.infer_stream:
  infer(A) -> record infer_done(A) -> infer(B) -> record infer_done(B)
```

随后 dispatcher 观察到 `infer_done(A)` 后：

1. 在 `ping.io_stream` 上启动 `D2H(A)`
2. 说明 slot 的关键 infer 段已经结束
3. 如果当前 ring head 上已经有请求：
   - 满 batch就正常发
   - 否则如果该 slot 当前没有其它活跃 infer，则触发 partial flush

于是：

```text
ping.io_stream:
  ... -> D2H(A) -> record d2h_done(A)

pong.io_stream:
  H2D(B) -> record h2d_done(B)

slot.infer_stream:
  infer(A) -> record infer_done(A) -> infer(B) -> ...
```

这样就实现了：

- `H2D(B)` 与 `infer(A)` overlap
- `D2H(A)` 与 `infer(B)` overlap
- `infer(B)` 提前进入 slot 的串行队列，不要求 infer stream 当场空闲

## 14.3 Bank 状态机

每个 bank 的状态：

```text
Idle
  -> H2DInFlight
  -> H2DDoneInferQueued
  -> InferDoneD2HInFlight
  -> HostVisible
  -> Idle
```

解释：

- `Idle`
  - bank 没有绑定任何 host slot
- `H2DInFlight`
  - H2D 已提交到 bank 的 `io_stream`
  - `h2d_done_event` 尚未就绪
- `H2DDoneInferQueued`
  - `h2d_done_event` 已就绪
  - infer 已提交到 slot 的 `infer_stream`
  - 可能正在执行，也可能正在排队
- `InferDoneD2HInFlight`
  - `infer_done_event` 已就绪
  - D2H 已提交到 bank 的 `io_stream`
  - `d2h_done_event` 尚未就绪
- `HostVisible`
  - `d2h_done_event` 已就绪
  - dispatcher 已对对应 handle 置 `output_mem_ready`
  - bank 即将回到 `Idle`

注意：

- `HostVisible` 不意味着 host slot 可复用
- 它只意味着 bank 自己可以解绑并回收

## 14.4 Slot 级状态

slot 自己维护的不是“谁做完了全部工作”，而是：

- `active_infer_bank`
- `last_completed_infer_bank`
- `next_launch_bank`

具体规则：

1. H2D 永远先发到 `next_launch_bank`
2. `next_launch_bank` 在 `ping/pong` 间切换
3. 只有 `h2d_done_event` 就绪后，该 bank 才允许把 infer 提交到 slot 的 `infer_stream`
4. 只有 `infer_done_event` 就绪后，slot 才认为关键路径上的 infer 已完成
5. 只有 `d2h_done_event` 就绪后，对应 bank 才允许再次被分配

## 14.5 为什么不是一个 bank 一个 doneEvent 就够

因为三个问题需要分开：

1. 什么时候可以把请求挂到 slot 的 infer stream
2. 什么时候 infer 关键路径完成
3. 什么时候这个 bank 的资源真的全部释放

这三个时刻分别由：

- `h2d_done_event`
- `infer_done_event`
- `d2h_done_event`

表示。

## 15. 外部事件系统如何和 bank 配合

## 15.1 从 caller 到 host slot

caller 只接触 handle 事件：

- `input_mem_ready`
- `output_mem_ready`
- `output_mem_consumed`

dispatcher 在 host slot 级别做 fan-in / fan-out。

### fan-in

当 host slot 的所有 row 都 `input_mem_ready`：

- host slot 进入可 dispatch 状态

### fan-out

当 bank 的 `d2h_done_event` 完成：

- dispatcher 遍历这批对应的所有 row
- 对每个 row 的 `output_mem_ready` 置位

## 15.2 从 host slot 到 ring 复用

host slot 的复用条件不是 bank done，而是：

- 本 generation 的所有 row 都 `output_mem_consumed`

因此 host slot 的状态机是：

```text
Empty
  -> Filling
  -> WaitingAllInputsReady
  -> Dispatched
  -> OutputsReady
  -> AllOutputsConsumed
  -> Empty(next generation)
```

## 15.3 dispatcher 的职责边界

dispatcher 只负责：

- 观察 caller 事件
- 观察 CUDA event
- 推进 slot / bank / host slot 状态机

dispatcher 不负责：

- 在 caller 线程里做复杂工作
- 在 CUDA callback 里改状态

## 16. 为什么这版设计成立

这版设计把三个层面的复用边界分清了：

1. `bank`
   - 受 `d2h_done_event` 约束
2. `slot` 的关键路径推进
   - 受 `infer_done_event` 约束
3. `host slot`
   - 受 `output_mem_consumed` 约束

同时又保住了三个目标：

1. caller 接口仍然是单请求模型
2. host 侧可以在满 batch 时走 packed slab 快路径
3. slot 内可以用双 bank 把 `H2D(B)` 与 `infer(A)`、`D2H(A)` 与 `infer(B)` 都重叠起来

## 17. 落地顺序

建议按下面顺序实现：

1. 先把最终接口和 host ring 搭起来
2. 实现 host slot generation 和对外事件系统
3. 为每个 slot 加入 `ping/pong` bank
4. 固定 device slab 地址并一次性 `setTensorAddress()`
5. 先用 `enqueueV3()` 跑通 H2D / infer / D2H
6. 加入：
   - `h2d_done_event`
   - `infer_done_event`
   - `d2h_done_event`
   - slot 级共享 `infer_stream`
   - infer-end partial flush
7. 最后才决定是否启用 bank 级 infer graph

最终不建议再沿着“capture 后修改 TensorRT 地址”或者“统一 graph 表达完整 I/O pipeline”的方向继续投入。
