# Modern Stdexec Design

## 目标

这份文档描述的是一版“彻底按现代 `stdexec` 架构重写”的 `schedlab`。

这里的“现代”不是“在现有代码上多包几层 sender 外壳”，而是从根上重新立约束：

1. runtime 多态不是默认选项，compile-time 组合才是默认选项。
2. completion 必须首先表现为 sender，而不是 `bool()`、event、或手工轮询。
3. slot 是异步生命周期的拥有者，request 只是它的行级视图。
4. 结构化并发、stop token、shared sender、typed sender graph 都要成为第一等公民。

这意味着当前代码里的许多东西在目标架构里都会消失：

- `InferBackend` 虚基类
- `InferState` 虚函数对象
- `function<bool()> + wait_pred(...)`
- `RequestState::output_ready`
- `OneShotEvent`
- `spawn_root_playout()` 的递归自复制

## 总体判断

如果真的按“最现代化”的方向来做，`schedlab` 应该变成：

- 一个 compile-time parameterized system
- 一个 sender-first orchestration layer
- 一个 slot-centric completion model

而不是：

- 一个传统 OO backend 接口
- 外面再包一点 sender/coroutine
- 中间继续靠轮询和手工事件把生命周期粘起来

所以这份文档不再假设“保留当前接口、逐步小修”。它直接描述终态。

## 一句话架构

最现代的一版 `schedlab` 大概会是：

1. `Dispatcher` 变成 `template<class Backend> class Dispatcher`
2. `Backend` 满足 concept，而不是继承虚基类
3. backend 的每一步 completion 都直接返回 sender
4. `DispatcherSlotState` 持有 slot 级 shared completion sender
5. `SearchRuntime` 的 worker loop 用 `repeat_effect_until` 或等价的循环 sender 表达
6. 停机走 stop token，而不是额外的原子标志

## 顶层原则

### 1. 不默认使用虚函数继承

在现代 sender-first 架构里，虚函数继承不是首选。

原因很简单：

- sender 的真实类型本身就携带了大量语义
- 一旦走虚基类，就会被迫在边界上做类型擦除
- 然后又为了拿回 sender 组合能力，再做第二层 sender 擦除

这会形成很重的“双重抽象税”：

- 一层 runtime 多态
- 一层 sender type erasure

所以更现代的做法是：

```cpp
template<class Backend>
concept InferBackend = requires(Backend& backend, typename Backend::HostSlot& slot) {
  typename Backend::HostSlot;
  typename Backend::Lane;
  typename Backend::InferOp;
  typename Backend::SubmitSender;

  { backend.batch_layout } -> std::same_as<const BatchLayout&>;
  { backend.acquire_host_slot() } -> std::same_as<typename Backend::HostSlot>;
  { backend.make_infer_op(slot, std::uint32_t{}, typename Backend::Lane{}) }
    -> std::same_as<typename Backend::InferOp>;
};
```

也就是说：

- “有没有某个 backend 能接进来”由 concept 决定
- “dispatcher 怎么调它”由泛型代码决定
- 不再由虚表决定

如果以后真的需要运行时选择 backend，最外层可以再用：

- `std::variant<TrtBackend, MockBackend>`
- 或更小范围的专门擦除层

但那应该是组合根的责任，不应该成为系统内部的默认抽象方式。

### 2. sender 是完成语义，不是装饰品

在目标架构里，sender 不是“最终等待前顺手套一下”的装饰层，而是 completion 的唯一正式表达。

因此下面这些接口都不该存在：

- `bool is_ready()`
- `function<bool()>`
- `wait_pred(...)`
- `OneShotEvent::set()/reset()`

取而代之的应该是：

- `submit_ready() -> sender`
- `submit_h2d() -> sender`
- `submit_infer() -> sender`
- `submit_d2h() -> sender`

如果某件事最终会完成，那它就应该首先表现成 sender。

### 3. slot 是异步生命周期单位

当前系统真正的生命周期单位不是 request，而是 slot。

因为真正被 seal、被提交、被 D2H、被 reset 的，是 slot。

所以目标架构里：

- slot 拥有 completion
- request 只共享 slot completion
- request 不拥有自己的异步事件对象

换句话说，request 是：

- 地址视图
- 行索引
- arrival token

而不是：

- 独立 completion source

## 核心数据结构

## BatchLayout

`BatchLayout` 仍然是稳定的 batch 级版式描述：

```cpp
struct BatchLayout {
  std::vector<std::size_t> input_row_bytes;
  std::vector<std::size_t> output_row_bytes;
};
```

这层没有必要更激进。

它表达的是 backend 对 batch buffer 的稳定要求，这一点在现代架构下同样成立。

## HostSlot

`HostSlot` 仍然是 batch 级对象，但应该完全 RAII 化：

```cpp
struct HostSlot {
  void* raw_storage = nullptr;
  std::size_t raw_storage_bytes = 0;

  void* input_slab = nullptr;
  std::size_t input_slab_bytes = 0;

  void* output_slab = nullptr;
  std::size_t output_slab_bytes = 0;

  std::vector<void*> inputs;
  std::vector<void*> outputs;
};
```

这里依然是 batch 级，不看 request。

但和当前不同的是：

- `HostSlot` 不应该通过 `allocate_host_slot()/release_host_slot()` 这样的手工协议管理
- 它应该是 move-only RAII value
- backend 负责构造它，析构自动回收资源

也就是说，现代版本更像：

```cpp
auto acquire_host_slot() -> HostSlot;
```

而不是：

```cpp
auto allocate_host_slot() -> HostSlot;
void release_host_slot(HostSlot&);
```

## DispatcherSlotState

现代架构下，slot 元数据应该成为真正的中心对象：

```cpp
template<class Backend>
struct DispatcherSlotState {
  using host_slot_t = typename Backend::HostSlot;
  using shared_void_sender = /* slot 级共享 completion sender */;
  using arrival_latch = /* slot 级 sender-native latch */;

  host_slot_t host_slot;
  std::unique_ptr<RequestState[]> request_states;

  std::uint32_t assigned_rows = 0;
  std::uint64_t generation = 0;
  bool sealed = false;

  arrival_latch input_ready;
  arrival_latch output_consumed;

  std::optional<shared_void_sender> d2h_done;
};
```

这里最大的变化有两个：

1. `output_ready` 消失，改成 slot 级 `d2h_done`
2. `input_ready_count` / `output_consumed_count` 也不再直接暴露为原子计数器，而是被 sender-native latch 封装

## RequestState

最现代的一版里，`RequestState` 应该变得非常薄：

```cpp
struct RequestState {
  std::vector<void*> inputs_mem_addr;
  std::vector<void*> outputs_mem_addr;

  DispatcherSlotState<Backend>* owner_slot = nullptr;
  std::uint32_t row_index = 0;

  arrival_token input_ready;
  arrival_token output_consumed;
};
```

这里故意不再有：

- `output_ready`
- slot 计数器裸指针

因为这些都不该由 request 直接拥有。

现代版本里，`RequestState` 只做三件事：

1. 暴露地址视图
2. 暴露 arrival token
3. 在需要等待 D2H 完成时，回到 owner slot 取 shared sender

例如：

```cpp
auto output_ready_sender() const -> shared_void_sender {
  return owner_slot->d2h_done.value();
}
```

## 需要新增的两个本地基础设施

vendored `stdexec` 里有：

- `exec::split`
- `exec::ensure_started`
- `exec::repeat_effect_until`
- `exec::any_sender_of.hpp`

但它没有两个刚好匹配我们场景的直接成品：

1. slot 级 countdown latch
2. 可存放在 slot 里的 shared void sender 包装

因此，最现代的一版 `schedlab` 里，应该新增两个很小的本地 primitives。

## 1. countdown_latch

用途：

- 等这个 slot 的 `assigned_rows` 行都调用了 `notify_input_ready()`
- 等这个 slot 的 `assigned_rows` 行都调用了 `notify_output_consumed()`

要求：

- sender-native
- 单次完成
- 每代 slot 可重建
- request 侧拿到的是 `arrival_token`
- dispatcher 侧拿到的是 `async_wait()` sender

概念上：

```cpp
struct countdown_latch {
  auto make_token() -> arrival_token;
  auto async_wait() -> sender_of<set_value_t()>;
};
```

这会替代现在裸露在 `DispatcherSlotState` 上的：

- `input_ready_count`
- `output_consumed_count`

## 2. shared_void_sender

用途：

- 在 slot 上保存“这一代 D2H 完成”的共享 sender
- request 行可多次拷贝/提取这个 sender 并等待

语义上它本质就是：

```cpp
exec::split(exec::ensure_started(...))
```

但为了避免把极其复杂的具体 sender 类型传播到整个数据结构层，应该包成一个小的稳定值类型。

这个包装不是倒退。

关键区别在于：

- 它擦除的是“共享 completion sender 的存储类型”
- 不是整个 backend 行为边界

也就是说，真正的系统主边界仍然是 compile-time 的，只有 slot 内部持久化某个 sender 时才做最小必要擦除。

## Backend 形态

最现代的一版里，backend 应该是具体类型，不是虚基类。

以 TRT 为例，大概会变成：

```cpp
class TrtBackend {
public:
  using HostSlot = TrtHostSlot;
  using Lane = InferLane;
  using InferOp = TrtInferOp;

  BatchLayout batch_layout;

  auto acquire_host_slot() -> HostSlot;
  auto make_infer_op(HostSlot&, std::uint32_t batch_size, Lane lane) -> InferOp;
};
```

其中 `TrtInferOp` 是值类型 operation object：

```cpp
class TrtInferOp {
public:
  auto submit_ready() -> sender_of<set_value_t()>;
  auto submit_h2d() -> sender_of<set_value_t()>;
  auto submit_infer() -> sender_of<set_value_t()>;
  auto submit_d2h() -> sender_of<set_value_t()>;
};
```

这和当前的核心语义仍然一致：

- dispatcher 继续感知三次分开的调用
- bank / cudaEvent / stream 全都仍然藏在 TRT 内部

但接口不再是：

- 虚函数
- `std::unique_ptr<InferState>`
- `function<bool()>`

## Dispatcher 形态

现代版本里，`Dispatcher` 应该是一个模板：

```cpp
template<InferBackend Backend>
class Dispatcher;
```

它的职责仍然和现在差不多：

- host slot ring
- ticket 分配
- slot seal
- lane 选择
- scheduler side-effect

但它不再自己轮询 completion，也不再手工事件广播。

## Dispatcher 的核心变化

### 1. slot seal 后立即生成 pipeline sender

当前实现是：

- seal slot
- 起 `infer_coro`
- 协程里一步一步轮询推进

现代实现应该是：

- seal slot
- 立刻构造 `slot_pipeline(slot)` sender
- 立刻 `ensure_started`
- 立刻 `split`
- 把共享 completion sender 存进 `slot.d2h_done`

也就是说，slot 一旦 seal，它的异步命运就已经完整固定下来。

### 2. `infer_coro` 变成 sender pipeline 装配器

最现代的版本里，`dispatcher` 甚至不一定还需要 `infer_coro` 这个 coroutine 名字。

更自然的形态会是：

```cpp
auto slot_pipeline(DispatcherSlotState<Backend>& slot) {
  auto lane = choose_lane(slot);
  auto infer_op = backend.make_infer_op(slot.host_slot, slot.assigned_rows, lane);

  return slot.input_ready.async_wait()
    | stdexec::then([&] {
        scheduler.infer.on_infer_submit(lane, slot.assigned_rows);
      })
    | stdexec::let_value([op = std::move(infer_op)]() mutable {
        return op.submit_ready()
          | stdexec::let_value([&]() mutable { return op.submit_h2d(); })
          | stdexec::let_value([&]() mutable { return op.submit_infer(); })
          | stdexec::then([&] {
              scheduler.infer.on_infer_done(lane);
              maybe_launch_open_slot_if_group_idle(lane.group_id);
            })
          | stdexec::let_value([&]() mutable { return op.submit_d2h(); });
      })
    | stdexec::let_value([&] {
        return slot.output_consumed.async_wait();
      })
    | stdexec::then([&] {
        reset_slot(slot);
        scheduler.infer.on_request_done();
        scheduler.maybe_open_gate();
      });
}
```

上面这段不是要求逐字符照抄，它只是表达方向：

- 数据依赖和时序依赖直接进 sender graph
- scheduler side-effect 插在 `then(...)`
- latch 和 backend completion 都是 sender
- slot reset 是 pipeline 的尾部 side-effect

### 3. output ready 不再手工逐行广播

现代版不会再有：

```cpp
for(each row) {
  request.output_ready.set();
}
```

而是：

```cpp
slot.d2h_done = make_shared_completion(slot_pipeline_until_d2h(...));
```

然后每个 request 都共享等待这个 sender。

这正是 `split` 的天然使用场景。

## SearchRuntime 形态

当前 `search.cpp` 里最老派的一块，其实是：

- `spawn_root_playout()`
- 在 `root_playout()` 里手工递归补 spawn

现代版不应该继续这么写。

## worker loop 应改为重复 sender

更现代的方式是：

- 每个 worker 启动一个长期存在的 loop sender
- loop 用 `repeat_effect_until` 或同类算法表达
- 停机通过 stop token 或停止谓词收束

概念上：

```cpp
auto worker_loop(WorkerLane& worker) {
  return exec::repeat_effect_until([&]() {
    return one_playout(worker);
  });
}
```

然后：

```cpp
worker.scope.spawn(ex::starts_on(worker.scheduler(), worker_loop(worker)));
```

这样搜索侧并发就不再依赖“在某个分支里手工再 spawn 下一条”。

## one_playout 形态

`one_playout(worker)` 仍然可以用 coroutine，也可以进一步写成 sender graph。

如果按最现代的版本，我会更倾向于 sender graph：

1. `pause_gate.async_wait()`
2. `descend`
3. `if need_nn_eval`
4. `acquire_ticket`
5. `preprocess`
6. `ticket.input_ready.arrive()`
7. `scheduler.infer.on_request_ready()`
8. 等 `ticket.output_ready_sender()`
9. `postprocess`
10. `ascend`
11. `ticket.output_consumed.arrive()`

这里 `need_nn_eval` 这种条件分支，正适合用：

- `let_value`
- `just`
- `just_stopped`
- 或专门的小分支 sender

也就是说，现代版搜索侧的“复制自己”不会再由递归 `spawn_root_playout()` 完成，而是由 loop sender 自然表达。

## 停机模型

最现代的一版不应该再主要依赖额外的 `stopping` 原子。

它应该改成：

- 顶层拥有 stop source
- worker loop、slot pipeline 都运行在带 stop token 的环境里
- `request_stop()` 触发 stop source
- `PauseGate` 和等待中的 sender 通过 `upon_stopped` / `unless_stop_requested` 自然收束

这会让“停机”也变成 sender graph 的一部分，而不是 graph 外的特殊控制流。

## Scheduler 在现代版中的位置

`Scheduler` 的职责并不会因为现代化而消失。

它仍然要做：

- `PauseGate` frontier 控制
- search 供给估计
- infer timeline 预测
- lane 选择

但它在异步图里的位置会更清晰：

- 它不负责等待
- 它不负责 completion 存储
- 它只是 sender graph 里若干 `then(...)` 的 side-effect owner

也就是说，现代化后的 `Scheduler` 更像纯控制状态机，而不是异步推进器。

## 最终文件形态

如果彻底重写，我会建议文件结构更像：

- `exp/schedlab/include/schedlab/backend_concepts.hpp`
- `exp/schedlab/include/schedlab/shared_completion.hpp`
- `exp/schedlab/include/schedlab/countdown_latch.hpp`
- `exp/schedlab/include/schedlab/dispatcher.hpp`
- `exp/schedlab/include/schedlab/search.hpp`
- `exp/schedlab/include/schedlab/trt_backend.hpp`
- `exp/schedlab/include/schedlab/scheduler.hpp`

其中：

- `backend_concepts.hpp` 只放 concept 和通用 batch 类型
- `shared_completion.hpp` 封装 slot 级 shared sender
- `countdown_latch.hpp` 封装 sender-native latch
- `dispatcher.hpp` 直接是模板，不再藏虚接口

## 代价

这版架构更现代，但代价也明确：

- 模板量会更多
- 编译时间会上升
- backend 接口不会像虚基类那样“一眼就像 plugin”
- 某些 sender 类型若不做局部包装会非常难读

但这些代价换来的收益也很直接：

- 异步语义终于直接出现在类型层面
- completion 不再被轮询/事件二次编码
- slot 成为真正的一等生命周期单位
- request 和 backend 的边界会更纯
- `stdexec` 算法不再是点缀，而是核心结构材料

## 总结

如果真的按“最现代化”的架构重来一版，`schedlab` 的核心结论应该是：

1. backend 用 concept，不用虚基类
2. completion 用 sender，不用 `bool()` 或 event
3. slot 用 shared sender 持有 D2H 完成，不再逐 request 持有 `output_ready`
4. request 用 arrival token 向 slot barrier 汇报，不再直接摸计数器
5. search worker 用重复 sender / 结构化并发，不再递归 spawn
6. stop 用 stop token，不再主要依赖额外原子标志

这才是真正从“旧式异步框架 + 一点 stdexec”跨到“sender-first 系统设计”的版本。
