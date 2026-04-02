# Search Design With `stdexec`

## 1. 目标

本文只整理 `exp/search_pseudo.py` 想表达的那套搜索调度结构，并把它改写成一份可落地的 `stdexec` 设计说明。

目标不是重写整个搜索模块，而是说明：

1. 如何保留 `playout()` 这种递归式、可读的源码结构。
2. 如何把“遇到 GPU 叶子就让出当前搜索线程，先让后继 playout 顶上去”写成 `stdexec` 风格。
3. 如何避免显式状态机、手写 continuation 链，并把底层调度泵隐藏在执行上下文里。
4. 如何保证单搜索线程上的协程在 C++20 编译后自然 lower 成 coroutine state machine。
5. 如何把 `pause_gate`、`SingleRequestHandle`、以及终局等待都改成 sender-first 接口。

本文是搜索侧文档，因此只引用 `pause_gate` / TRT dispatcher 的外部接口，不重复它们内部实现。

## 2. 硬性约束

这一版设计有几个不可退让的约束。

### 2.1 明确禁止

以下做法在本文中一律视为不接受：

1. 在搜索逻辑里显式写 `run_loop`、`event loop`、`while(pop_ready())` 一类底层调度泵。
2. 手写 `enum State { ... } + switch(state)` 形式的状态机。
3. 把 `output_mem_ready` 的 continuation 丢去一个“随便什么线程池”收尾。
4. 在热路径里使用 `sync_wait()`、`condition_variable.wait()`、`future.get()` 之类阻塞等待。
5. 让多个 in-flight playout 共享一份可变 `SearchThread` 遍历状态。

### 2.2 允许使用的抽象层次

允许使用的抽象只到下面这一层：

1. `exec::task<T>` 表达递归异步控制流。
2. `stdexec` sender algorithms 表达边界上的等待、转换和调度。
3. `exec::async_scope` 管理“动态生成、但必须在作用域结束前 join”的任务。
4. 一个串行 scheduler 抽象，负责保证“一个搜索 worker 同一时刻只在一条 OS 线程上恢复 continuation”。

换句话说：

- 搜索算法层只能看见 coroutine、sender、scope 和 scheduler。
- scheduler 背后的队列、唤醒、线程停放，都属于执行上下文实现细节，不属于搜索逻辑。

## 3. `search_pseudo.py` 真正要表达什么

`exp/search_pseudo.py` 的关键意图可以概括成四句话。

### 3.1 正常情况仍然是递归 playout

伪代码第 9-13 行的意思很简单：

- 如果还能继续往下选子节点，就继续递归。
- 子递归回来以后，正常回溯更新父节点。

这部分本来就非常适合 coroutine，因为它本质上就是“递归函数里夹了几个异步暂停点”。

### 3.2 GPU 叶子会把当前 worker 让出来

第 16-28 行表达的是这份设计最重要的地方：

1. 当前 playout 到了需要 NN eval 的叶子。
2. 它拿一个 `SingleRequestHandle`，做 preprocess，然后发出 `input_mem_ready`。
3. 在等待 `output_mem_ready` 之前，先在“同一个搜索线程”上排一个新的 playout。
4. 自己随后挂起，等 GPU 结果回来再继续 postprocess 和回溯。

这不是普通“异步 I/O”而已，而是一个 very specific scheduling shape：

- 当前 playout 不结束，只是挂起。
- 搜索 worker 继续执行后继 playout。
- 早先挂起的 playout 以后会回来收尾。

### 3.3 根结点决定是否继续自复制

第 33-44 行的意思是：

- 每个 root playout 在结束时决定要不要再启动一个新的 root playout。
- 如果这个 playout 在叶子处已经提前排过 successor，就不要再排第二次。
- 如果这次没碰到 GPU，就在 root 结束时自己补一个 successor。

所以它本质上是一个 self-replicating root task。

### 3.4 更正后的 `pause_gate` 语义

更新后的伪代码已经把 `pause_gate` 放进了 `if node is root:` 分支里。

这正是本文采用的语义：

- `pause_gate` 只控制 root playout 的起步；
- 已经进入递归中的同一条 playout，不会在每一层递归入口重复等待；
- 因而它控制的是“是否允许新 playout 继续推进并制造新的 GPU request”。

所以这一点上，最新伪代码与本文设计是一致的。

## 4. 现有代码对新设计的约束

### 4.1 现在的 `SearchThread` 不能直接复用为 in-flight coroutine state

当前 `SearchThread` 里放着整条 playout 的可变遍历状态：

- `pla`
- `board`
- `history`
- `graphHash`
- `graphPath`
- `shouldCountPlayout`
- `rand`
- `nnResultBuf`
- `statsBuf`
- `upperBoundVisitsLeft`
- `waitNNEvalTimeThisPlayoutMs`
- `oldNNOutputsToCleanUp`
- `illegalMoveHashes`

这些字段都在 [`cpp/search/search.h`](../cpp/search/search.h) 第 43-79 行。

当前同步实现一次只跑一个 blocking playout，所以它们可以安全地挂在 worker 上。

但新调度不是这样。

一旦 root playout 在 GPU 叶子处挂起，同一个搜索 worker 会继续跑新的 root playout。于是同一时刻会有多个 in-flight playout。此时：

- 遍历状态必须是 per-playout 的；
- 不能再放在 worker 唯一一份的 `SearchThread` 上；
- 否则第二个 playout 会把第一个挂起 playout 的 `board/history/graphPath` 覆盖掉。

### 4.2 当前 `playoutDescend()` 的递归形状应该尽量保留

当前核心逻辑在 [`cpp/search/search.cpp`](../cpp/search/search.cpp) 第 1172-1452 行。

这段代码的结构其实非常适合直接搬成 coroutine：

1. 先处理终局叶子。
2. 再处理 `STATE_UNEVALUATED / STATE_EVALUATING / EXPANDED`。
3. 选子、落子、检查环。
4. 递归 descend。
5. 回溯更新父节点。

我们要保留的是这个结构。

我们不该做的是：

- 把它拆成很多小 callback；
- 或者改成显式状态机；
- 或者再发明一套 scheduler-side DSL。

### 4.3 本文不抽象业务侧附加节流

本文只讨论通用搜索调度骨架：playout coroutine、pause gate、request ticket 和 completion 等待。

任何为了特定业务策略而引入的“额外等待一步再继续搜索”逻辑，都不属于这里的通用设计，也不进入本文接口。

## 5. 从 `stdexec` 源码得到的本地事实

### 5.1 `exec::task` 默认就是 sticky coroutine

本地 `stdexec` 的 `exec::task` 在 `await_transform` 里会对非 affine sender 自动做 `continues_on(current_scheduler)`。

见 [`third_party/stdexec/include/exec/task.hpp`](../third_party/stdexec/include/exec/task.hpp) 第 543-558 行。

这意味着：

1. coroutine 当前关联了某个 scheduler。
2. `co_await some_sender` 时，如果 sender 自己不保证 completion affine，
3. `task` 会把 continuation 自动拉回当前 scheduler。

这正是搜索 worker 需要的语义。

### 5.2 `co_await schedule(other_sched)` 不会迁移 coroutine

本地测试已经明确写死了这个语义。

在 [`third_party/stdexec/test/exec/test_task.cpp`](../third_party/stdexec/test/exec/test_task.cpp) 第 57-61 行和第 69-76 行：

- `co_await schedule(other_sched)` 不会迁移当前 `exec::task` 的归属。
- 真正迁移 coroutine 的唯一显式做法是 `co_await exec::reschedule_coroutine_on(other_sched)`。

所以搜索侧不能把

```python
await sched_on(global_thread_pool, h.output_mem_ready.get())
```

翻译成“先把等待放到别处，然后在别处收尾”。

正确翻译应该是：

```cpp
co_await h.output_ready();
```

然后依赖 sticky `task` 自动把 continuation 恢复回当前搜索 worker 的 scheduler。

### 5.3 `async_scope` 是正确的动态任务所有者

本地 `async_scope` 的实现会：

1. 在 `spawn/nest` 时增加 active 计数。
2. 在子任务结束时递减计数。
3. `on_empty()` 在 active 归零时完成。

见 [`third_party/stdexec/include/exec/async_scope.hpp`](../third_party/stdexec/include/exec/async_scope.hpp) 第 72-108 行和第 158-235 行。

这正适合管理“root task 自复制”这种 weakly-structured 并发。

### 5.4 `single_thread_context` 是可接受的串行执行上下文

本地 `single_thread_context` 直接包了一个 `run_loop`：

- 成员是 `STDEXEC::run_loop loop_`
- 构造里 `thread_([this] { loop_.run(); })`
- 析构里 `loop_.finish()`

见 [`third_party/stdexec/include/exec/single_thread_context.hpp`](../third_party/stdexec/include/exec/single_thread_context.hpp) 第 25-45 行。

这说明两件事：

1. `single_thread_context` 确实自带一层 `run_loop` 封装。
2. 但只要搜索逻辑本身不直接碰这层 `run_loop`，它就是可接受的执行上下文实现细节。

## 6. 外部 best practices 提炼

下面这些原则是本文采用的外部依据。

### 6.1 用 sender 统一包装异步边界

Eric Niebler 在 2024-02-04 的文章《What are Senders Good For, Anyway?》里强调，senders 的价值之一就是把旧式 callback/event 异步接口统一包装成可组合的标准形状，并把资源、operation state、start、completion 统一起来。

尤其是文中第 179-183 行明确解释了：

- sender 只是描述工作；
- `connect` 得到 operation state；
- 真正启动发生在 `start()`；
- 把启动与描述分离，才能把复杂异步图聚合成低额外成本的 operation state。

对搜索来说，这条原则的直接结论就是：

- `pause_gate`
- `output_mem_ready`

都应该先变成 sender，再进入 coroutine。

搜索侧不应该直接等待原始 `Event`。

### 6.2 coroutine 负责非线性控制流，sender 负责异步边界和组合

Lucian Teodorescu 在 2025-02 的《Using Senders/Receivers》里反复强调两件事：

1. senders 用来描述工作和执行图；
2. `async_scope` 很适合管理递归过程中动态生成的任务；
3. 当控制流非线性、具有递归结构时，保留核心逻辑原貌非常重要。

这和搜索是高度匹配的：

- `playoutDescend()` 的核心是递归、分支、回溯。
- 它不是一个适合只靠 `then/let_value/when_all` 平铺出来的线性 pipeline。

因此最佳实践不是“把整个搜索写成 sender pipeline”，而是：

1. 用 `exec::task` 写递归控制流；
2. 用 sender 封装 pause / eval completion / throttling / scheduler transitions。

### 6.3 `async_scope` 用于 weakly-structured 自复制任务

Teodorescu 在同文第 187-239 行把 `async_scope` 解释得很清楚：

- 动态 spawn 的工作可以超出调用它的那个栈帧；
- 但必须在 scope 销毁前全部完成；
- 这是一种 weakly-structured concurrency。

搜索侧的 self-replicating root playout 正是这个模型。

### 6.4 structured concurrency 的最大价值是词法作用域和生命周期对齐

Eric Niebler 在 2021-08-29 的《Asynchronous Stacks and Scopes》里给出的核心观点是：

1. coroutine 的最大价值不是“语法像同步代码”；
2. 而是 async scope 可以重新对齐到 lexical scope；
3. 局部变量天然跟着异步操作活到正确的时间点。

因此搜索侧最自然的写法不是共享 `SearchThread` 临时状态，而是：

- 每个 in-flight playout 拥有自己的 `PlayoutFrame`；
- 它的寿命和 root coroutine 的寿命一致。

### 6.5 hot path 不要依赖 `sync_wait`

Teodorescu 在 2024-12 的《Senders/Receivers: An Introduction》里把 `sync_wait` 描述为“启动 sender 并阻塞当前线程直到完成”的桥接工具。

这意味着：

- `sync_wait` 适合程序边界、测试边界、或者 shutdown；
- 不适合搜索热路径里的每次 NN eval、pause gate、或 completion throttle。

## 7. 设计结论

这份文档最终给出的结论是：

1. 搜索侧核心控制流使用 `exec::task` coroutine。
2. 递归下降仍然写成递归函数，不显式编码状态机。
3. 每个搜索 worker 使用一个“串行 scheduler 抽象”。
4. 当前本地 `stdexec` 验证推荐用 `exec::single_thread_context` 实现这个串行 scheduler。
5. root playout 使用 `exec::async_scope` 做 self-replication。
6. 所有等待点都暴露成 sender-first 接口。
7. `SearchThread` 会拆成：
   - `SearchWorker`：持久 shard
   - `PlayoutFrame`：per in-flight playout state
8. v1 不引入专门的 postprocess pool、helper pool 或 ascend work-stealing。
9. GPU 输出 ready 之后，postprocess + commit/backprop 都先留在 originating worker 上完成。
10. 不允许在等待 `output_mem_ready` 时把 continuation 迁移到“任意别的线程池”。
11. `output_mem_consumed` 仍然必须延后到 CPU postprocess 和统计更新完成之后再发。

## 8. 推荐的对象模型

### 8.1 `SearchWorker`

`SearchWorker` 是持久对象，一一对应逻辑搜索线程。

它负责：

1. 串行 scheduler。
2. `async_scope`。
3. 持久统计 shard。
4. frame 池。
5. playout sequence 分配。
6. 停止标志观察。

建议形状如下：

```cpp
struct SearchWorker {
  exec::single_thread_context context;
  exec::async_scope scope;

  WorkerStats stats;
  FramePool frame_pool;
  std::atomic<uint64_t> next_playout_seq{0};

  auto scheduler() noexcept {
    return context.get_scheduler();
  }
};
```

### 8.2 `PlayoutFrame`

`PlayoutFrame` 是 per in-flight root playout 的状态。

它应该接管当前 `SearchThread` 上所有“在 playout 过程中会被改写”的字段。

建议至少包含：

```cpp
struct PlayoutFrame {
  int worker_idx;
  uint64_t playout_seq;

  Player pla;
  Board board;
  BoardHistory history;
  Hash128 graphHash;
  std::unordered_set<SearchNode*> graphPath;

  bool shouldCountPlayout = true;
  Rand rand;

  NNResultBuf nnResultBuf;
  std::vector<MoreNodeStats> statsBuf;
  double upperBoundVisitsLeft = 0.0;
  double waitNNEvalTimeThisPlayoutMs = 0.0;

  std::vector<std::shared_ptr<NNOutput>*> oldNNOutputsToCleanUp;
  std::set<Hash128> illegalMoveHashes;

  bool spawned_successor = false;
  bool touched_gpu = false;
  std::optional<SearchEvalTicket> pending_eval;
  RootTimer timer;
};
```

### 8.3 为什么是“一条 playout 一份 frame”

这是本文最关键的结构决定之一。

一旦 root playout A 在 GPU 等待点挂起，同 worker 上的 root playout B 就会开始运行。因此：

- A 和 B 不可能共享同一份 `board/history/graphPath/rand`。

但在同一条 playout 内部：

- 递归 descend 仍然是顺序控制流；
- 不存在同一条 playout 的 sibling 并行；
- 所以不需要“每层递归一份 frame”。

因此最合适的粒度就是：

- 每个 in-flight root playout 一份 `PlayoutFrame`。

## 9. 搜索侧必须拿到的 sender-first 接口

### 9.1 `PauseGate`

搜索侧不应该直接拿到一个阻塞式 gate。

推荐接口：

```cpp
struct PauseGate {
  auto async_wait() noexcept;   // sender<set_value()>
  bool is_open() const noexcept;
};
```

语义：

1. gate 已开时，`async_wait()` 立即完成。
2. gate 已关时，返回一个 sender，在 reopen 时完成。
3. sender 本身不负责线程迁移。
4. 由外层 sticky `exec::task` 保证 continuation 回到当前搜索 worker。

### 9.2 `SearchEvalTicket`

搜索侧也不应该碰原始阻塞 event。

推荐接口：

```cpp
struct SearchEvalTicket {
  void** inputs_mem_addr = nullptr;
  void** outputs_mem_addr = nullptr;

  void publish_input_ready() noexcept;
  auto output_ready() noexcept;     // sender<set_value()>
  void publish_output_consumed() noexcept;
};

SearchEvalTicket NNEvaluator::getSearchEvalTicket();
```

这里的设计重点有四个：

1. 搜索侧拿到的是 sender-friendly ticket，而不是三个裸 `Event`。
2. 搜索侧等待的是“host-visible 输出 ready”这个边界。
3. v1 不在 ticket 内部偷偷再起一层 hot future 或 CPU postprocess pool。
4. 允许 legacy blocking caller 在更外层再包一层 blocking facade，但搜索热路径只看 sender-first 版本。

也就是说，当前推荐语义是：

1. `co_await ticket.output_ready();`
2. 然后在当前搜索 worker 上直接做 `postprocess + commit/backprop`。

这样设计不是因为做不到更复杂的拆段，而是因为当前已知量级下：

1. `postprocess` 只有约 `3us`；
2. 真正重的是 tree-sensitive 的 `ascend`，约 `50us`；
3. 因此 v1 先保留最简单、最容易验证正确性的本地 completion 结构。

### 9.3 不引入业务侧额外等待接口

本文只保留真正属于通用异步骨架的等待点：

1. `pause_gate.async_wait()`
2. `ticket.output_ready()`

除此之外不再抽象任何搜索业务层的附加节流接口。

## 10. 串行 scheduler 的选择

### 10.1 本文推荐：`exec::single_thread_context`

在当前本地 `stdexec` 条件下，推荐的 worker 执行上下文是：

```cpp
exec::single_thread_context context;
auto sched = context.get_scheduler();
```

原因：

1. 它提供标准 scheduler 接口。
2. 每个 worker 天然对应一条单独线程。
3. sticky `exec::task` 的恢复语义与搜索 worker 非常贴合。
4. 业务代码本身不需要直接操作底层 `run_loop`。

### 10.2 为什么现在接受它

这里的让步是明确的：

1. 允许 `single_thread_context` 自带的那层 `run_loop` 封装存在。
2. 但搜索逻辑仍然只看 scheduler，不直接写调度泵。
3. 因此“避免手写 event loop”的可读性目标仍然成立。

### 10.3 未来若替换 scheduler，实现不影响搜索算法层

搜索算法层只依赖：

```cpp
auto SearchWorker::scheduler() noexcept -> scheduler;
```

因此以后如果要换成别的串行 scheduler，只要保留这个语义即可。

本文不讨论更底层的自定义 scheduler 实现。

## 11. root task 的标准形状

root task 建议写成：

```cpp
auto root_playout(Search& search, SearchWorker& worker) -> exec::task<void>;
```

它的职责非常固定。

### 11.1 root preamble

1. 如果 `shouldStopNow` 已经成立，直接返回，不再自复制。
2. 从 frame 池拿一个 `PlayoutFrame`。
3. 用 root 状态初始化：
   - `pla = rootPla`
   - `board = rootBoard`
   - `history = rootHistory`
   - `graphHash = rootGraphHash`
   - `graphPath.clear()`
4. 生成独立 RNG 种子。
5. `co_await pause_gate.async_wait()`。
6. 启动 root timer。
7. `co_await descend(search, worker, frame, *rootNode, true)`。

### 11.2 root epilogue

1. 停止 timer。
2. 更新 worker-local 统计。
3. 如果本 playout 有 `pending_eval`：
   - 在 CPU postprocess / 回溯 / 统计全部完成之后，
   - 再调用 `publish_output_consumed()`。
4. 如果本 playout 没有提前排 successor，并且未收到 stop：
   - 再 `scope.spawn(starts_on(worker.scheduler(), root_playout(...)))`。
5. 回收 frame。

## 12. 递归下降 coroutine 的标准形状

建议保持和现有 `playoutDescend()` 尽可能接近：

```cpp
auto descend(
  Search& search,
  SearchWorker& worker,
  PlayoutFrame& frame,
  SearchNode& node,
  bool isRoot
) -> exec::task<bool>;
```

返回值仍然沿用当前语义：

- `true` 表示父路径需要继续更新祖先；
- `false` 表示这条路径不应该继续计入祖先更新。

### 12.1 终局分支

终局分支在本文里直接作为本地完成路径处理：

```cpp
if (frame.history.isGameFinished && !node.forceNonTerminal) {
  add_terminal_leaf_value(...);
  co_return true;
}
```

### 12.2 `STATE_UNEVALUATED`

这里和现有逻辑一样，v1 继续保留“first wins, losers abort current playout”语义：

1. 先尝试抢到“由我来负责本次 eval”的资格。
2. 抢失败则 `frame.shouldCountPlayout = false; co_return false;`
3. 抢成功则走 NN eval 叶子逻辑。

本文不在这里引入更复杂的 node 状态机。

### 12.3 已展开节点

对于已展开节点，流程继续保持现状：

1. 选 best child。
2. 处理 illegal move / regenerate。
3. 处理新 child / 已有 child。
4. 处理 graph cycle。
5. 递归 `co_await descend(...)`。
6. 回溯更新 edge visits 和父节点统计。

这部分不是并发调度问题，尽量不改它的形状。

## 13. GPU 叶子的标准写法

这是整个设计的核心。

### 13.1 不接受的翻译

伪代码这一段：

```python
launch_task(this_thread, playout)
await sched_on(global_thread_pool, h.output_mem_ready.get())
```

不能翻译成“把整条搜索 continuation 原封不动丢到 global pool”：

1. 在某个全局线程池里等待 `output_mem_ready`；
2. completion 后直接在那个线程池里继续整条 playout；
3. 树回溯、统计、ticket 生命周期也都跟着那个线程池跑；
4. 最后再想办法回到“搜索线程语义”。

原因：

1. `timer`、逻辑 worker 统计、`pending_eval` 生命周期会被迫跟着 arbitrary pool thread 漂移。
2. 树敏感的 `commit/backprop` 会和线程迁移耦合到一起，正确性边界更难讲清楚。
3. 这会把本来可以简单验证的搜索设计，过早升级成复杂的跨核均衡系统。

### 13.2 当前推荐：全部 completion 先留在 originating worker

基于当前已知量级，v1 的推荐方案是更简单的：

1. `preprocess` 仍在当前 worker 上做。
2. `output_ready()` 到来后，当前 coroutine 在 originating worker 上恢复。
3. 然后由这个 worker 顺序做：
   - `postprocess`
   - `commit/backprop`
   - 本 playout 统计更新

选择这个方案的理由很明确：

1. 当前估算里 `postprocess` 只有约 `3us`，并不值得单独起复杂的 pool/hot-future 机制。
2. 真正重的是 `ascend`，约 `50us`，而它又天然更接近 tree-sensitive 的 worker-affine 段。
3. 从本地 microbenchmark 看，同线程 inline sender / sticky coroutine 的推进是函数调用量级；真正贵的是跨线程 hop。
4. 因此 v1 更该优先追求简单、可验证和局部性，而不是一开始就做复杂均衡。

### 13.3 推荐写法

推荐写法如下：

```cpp
if (need_nn_eval) {
  auto ticket = search.nnEvaluator->getSearchEvalTicket();

  preprocess(node, frame, ticket.inputs_mem_addr);
  ticket.publish_input_ready();

  if (!frame.spawned_successor && !search.shouldStopNow()) {
    frame.spawned_successor = true;
    worker.scope.spawn(
      stdexec::starts_on(worker.scheduler(), root_playout(search, worker)));
  }

  frame.touched_gpu = true;
  frame.pending_eval = std::move(ticket);
  frame.timer.pause_cpu_budget();
  co_await frame.pending_eval->output_ready();
  frame.timer.resume_cpu_budget();

  postprocess_eval(search, frame, node, *frame.pending_eval);
  commit_eval(search, frame, node, *frame.pending_eval);

  co_return true;
}
```

这里真正利用到的 `stdexec` 语义是：

1. `descend()` 依然是 sticky `exec::task`。
2. `co_await frame.pending_eval->output_ready()` 之后，continuation 仍然恢复在当前搜索 worker。
3. 所以 `postprocess + commit/backprop + 统计` 可以直接写成正常顺序代码，不需要额外 helper pool。

### 13.4 为什么这版先不做复杂均衡

按照当前估算：

1. `postprocess` 约 `3us`
2. `ascend` 约 `50us`
3. 单次 infer 周期约 `1000us`

这意味着即使短时相位重合、同一 worker 上连续来了几个 completion：

1. 真正堆起来的是 `ascend`；
2. `postprocess` 并不是主要矛盾；
3. 与其一开始就为 `3us` 的段引入复杂拆段，不如先把本地 completion 路径跑通。

如果后续 profiling 证明某个 worker 的 completion backlog 确实经常失控，那么优先级应当是：

1. 先测 `output_ready -> ascend_start` 的 p95/p99；
2. 先看 `PauseGate` 和 request admission 是否已经足够抑制 backlog；
3. 只有在数据证明简单策略不够时，再引入 helper/steal 或二段 completion。

### 13.5 为什么 successor 要先排

因为这正是 `search_pseudo.py` 的精髓。

如果先等 GPU，再排 successor，那么整个搜索 worker 在这段时间就空了。

而正确顺序是：

1. 先排 successor。
2. 再挂起当前 playout。

这样才能形成“等待中的老 playout + 正在跑的新 playout”。

### 13.6 为什么 `output_mem_consumed` 要延后

这一点和 `pause_gate.md` 保持一致：

`publish_output_consumed()` 必须在以下动作之后才发生：

1. CPU 侧 postprocess 已完成。
2. 节点回溯更新已完成。
3. root 统计已完成。

否则 pause gate 可能会在“CPU 还没把老结果写回全局状态”时就重新放开，导致新 request 看到过老的全局状态。

因此：

- `ticket` 在叶子处只保存到 `frame.pending_eval`；
- 真正 `publish_output_consumed()` 在 root epilogue。

## 14. 自复制策略

### 14.1 只在 root 级自复制

递归 descend 本身不 spawn 新任务。

只有 root playout 会在两种时机排 successor：

1. GPU 叶子处，当前 playout 即将挂起时。
2. root 结束处，如果本 playout 从未提前排过 successor。

### 14.2 `spawned_successor` 是必须的

没有这个布尔位，root 结束时就可能重复排任务。

最简单的规则就是：

1. 默认 `false`
2. 任一提前自复制动作成功后设成 `true`
3. root epilogue 只在它仍为 `false` 时补排 successor

### 14.3 新 playout 仍然要先过 `pause_gate`

这点非常重要。

当前 playout 在 GPU 叶子处提前排 successor，不意味着 successor 可以无条件继续制造新 request。

正确语义是：

1. successor 被排到同一 worker 上；
2. 它启动后首先执行 root preamble；
3. root preamble 里再 `co_await pause_gate.async_wait()`。

这样：

- 已经开始的老 playout 可以回来收尾；
- 新 playout 是否继续推进到 GPU 叶子，由 pause gate 控制。

## 15. 可读性与“零成本抽象”如何理解

### 15.1 这里的“零成本”不等于“完全没有 coroutine frame”

必须说实话。

只要用了 coroutine，就会有 coroutine frame 这个概念。

Clang 官方文档《Debugging C++ Coroutines》直接把 coroutine frame、resume function、destroy function 讲得很清楚。

因此这里的“零成本抽象”只能合理地理解成：

1. 不手写状态机。
2. 状态机由编译器生成。
3. 不引入额外线程迁移层。
4. 不引入热路径类型擦除和阻塞等待。
5. 不把业务逻辑重写成难以阅读的 continuation spaghetti。

### 15.2 本地 clang 验证

我用本地工具链：

```text
.local/toolchains/llvm-22.1.1/bin/clang++
```

做了三件最小验证。

#### 验证 A

编译并运行了一个最小骨架：

1. `exec::task`
2. `exec::async_scope`
3. `exec::single_thread_context`
4. 一个 sender 化的手工事件 `async_wait()`
5. root task 在等待前先 `scope.spawn()` successor

结果：

- 编译通过；
- 运行通过；
- 挂起点之后的 continuation 仍然恢复在原 worker 线程。

#### 验证 B

编译并运行了一个更窄口径的 microbenchmark，专门拆开测：

1. 函数调用
2. 同线程 coroutine resume
3. `exec::task` await inline sender
4. 显式 cross-thread `reschedule_coroutine_on`

结果见：

- `exp/stdexec_switch_bench/tax_micro.cpp`
- `exp/stdexec_switch_bench/TAX_MICRO_RESULTS.md`

结论：

1. 同线程 inline await 的成本确实是函数调用量级，约 `2-3ns`。
2. `stdexec_task_await_task` 这种“await child task”路径已经上升到约 `0.12us`。
3. 真正显著的税来自 cross-thread hop；而在本地测量里，`single_thread_context` 这一项是可接受的。

这正支持本文的 v1 选择：

1. 继续依赖 sticky worker 上的本地 completion；
2. 暂时不为了 `3us` 的 postprocess 去引入跨线程均衡结构；
3. 把复杂拆段留到 profiling 明确证明需要时再做。

#### 验证 C

用同一把 clang 对一个最小 `exec::task<int>` 协程发出 LLVM IR，生成物里能直接看到：

- `tiny.resume`
- `tiny.destroy`

这说明这套写法确实 lower 成了编译器生成的 coroutine state machine，而不是源码层显式状态机。

### 15.3 热路径额外开销控制原则

为避免把 coroutine 写成“看着高级、跑着很贵”，热路径还要遵守三条规则：

1. 等待 sender 必须是 concrete type，不要 `any_sender`。
2. 等待节点要做 intrusive / opstate 内嵌，不要每次 `new waiter`。
3. `PlayoutFrame` 要从 worker-local 池分配，避免每条 playout 都走通用堆。

本文允许 coroutine frame 存在，但不允许热路径每个暂停点都掉进多层堆分配和类型擦除。

## 16. RNG 与顺序语义

异步化以后，一个 worker 上的 playout 完成顺序会变化。

以前是：

- A 完成后才开始 B。

现在会变成：

- A 在 GPU 叶子挂起；
- B、C 可能先完成；
- A 之后才回来。

因此不能继续依赖“worker 上一条线性 RNG 流”来隐式定义行为。

推荐做法：

1. `SearchWorker` 维护 `next_playout_seq`。
2. 每次 root playout 开始时拿一个 `playout_seq`。
3. 用 `(search_seed, worker_idx, playout_seq)` 构造 `frame.rand`。

这样：

- RNG 状态属于 playout；
- 不受 completion 先后顺序影响；
- 更符合 per-playout frame 的设计。

## 17. 一份建议代码骨架

下面这份骨架不是完整代码，只是把推荐结构钉死。

```cpp
auto root_playout(Search& search, SearchWorker& worker) -> exec::task<void> {
  if (search.shouldStopNow()) {
    co_return;
  }

  auto frame = worker.frame_pool.acquire();
  init_root_frame(search, worker, *frame);

  co_await search.pause_gate.async_wait();

  frame->timer.start();
  bool counted = co_await descend(search, worker, *frame, *search.rootNode, true);
  frame->timer.stop();

  if (counted) {
    update_worker_stats(worker, *frame);
  }

  if (frame->pending_eval.has_value()) {
    frame->pending_eval->publish_output_consumed();
  }

  if (!frame->spawned_successor && !search.shouldStopNow()) {
    worker.scope.spawn(
      stdexec::starts_on(worker.scheduler(), root_playout(search, worker)));
  }

  worker.frame_pool.release(std::move(frame));
}

auto descend(
  Search& search,
  SearchWorker& worker,
  PlayoutFrame& frame,
  SearchNode& node,
  bool isRoot
) -> exec::task<bool> {
  if (frame.history.isGameFinished && !node.forceNonTerminal) {
    add_terminal_leaf_value(search, frame, node);
    co_return true;
  }

  if (node_needs_nn_eval(node)) {
    auto ticket = search.nnEvaluator->getSearchEvalTicket();
    preprocess(search, frame, node, ticket.inputs_mem_addr);
    ticket.publish_input_ready();

    if (!frame.spawned_successor && !search.shouldStopNow()) {
      frame.spawned_successor = true;
      worker.scope.spawn(
        stdexec::starts_on(worker.scheduler(), root_playout(search, worker)));
    }

    frame.pending_eval = std::move(ticket);
    frame.timer.pause_cpu_budget();
    co_await frame.pending_eval->output_ready();
    frame.timer.resume_cpu_budget();

    postprocess_eval(search, frame, node, *frame.pending_eval);
    commit_eval(search, frame, node, *frame.pending_eval);
    co_return true;
  }

  auto [child, count_edge_visit] = select_or_expand_child(search, frame, node, isRoot);
  if (child == nullptr) {
    add_leaf_value_without_expansion(search, frame, node);
    co_return true;
  }

  bool should_update = co_await descend(search, worker, frame, *child, false);
  if (should_update && count_edge_visit) {
    update_parent_after_child(search, frame, node, *child, isRoot);
  }
  co_return should_update && count_edge_visit;
}
```

这份骨架的关键特征是：

1. 递归结构保留了。
2. 异步等待只出现在 sender-first 边界。
3. self-replication 只发生在 root 层。
4. 没有业务层显式 event loop。
5. 没有显式状态机。

## 18. 最终建议

如果把 `search_pseudo.py` 落成生产实现，我建议按下面这条路线走。

1. 先引入 `SearchWorker` / `PlayoutFrame` 二分。
2. 把 `pause_gate` 和 `SingleRequestHandle` 先 sender 化。
3. 先把 root self-replication 跑通。
4. 然后再把 `playoutDescend()` 整体搬成 `exec::task<bool>`。
5. v1 先坚持“本地 worker completion”，不要一开始就引入 helper/steal。
6. 整个过程中禁止把等待点偷偷改回阻塞式。

一句话总结本文的设计立场：

`search_pseudo.py` 想要的结构，完全可以先用 `stdexec` 的 best practice 写成“递归 coroutine + sender 化等待 + async_scope 自复制 + 本地 worker completion”，允许底层执行上下文用 `single_thread_context` 承载，而不必退回显式状态机或过早引入复杂的跨核均衡逻辑。

## 19. 参考资料

### 本地代码

1. `exp/search_pseudo.py`
2. `cpp/search/search.h`
3. `cpp/search/search.cpp`
4. `cpp/neuralnet/nneval.cpp`
5. `third_party/stdexec/include/exec/task.hpp`
6. `third_party/stdexec/include/exec/async_scope.hpp`
7. `third_party/stdexec/include/exec/single_thread_context.hpp`
8. `third_party/stdexec/test/exec/test_task.cpp`
9. `third_party/stdexec/README.md`
10. `exp/stdexec_switch_bench/tax_micro.cpp`
11. `exp/stdexec_switch_bench/TAX_MICRO_RESULTS.md`

### 外部资料

1. Eric Niebler, “What are Senders Good For, Anyway?”, 2024-02-04
   https://ericniebler.com/2024/02/04/what-are-senders-good-for-anyway/
2. Eric Niebler, “Asynchronous Stacks and Scopes”, 2021-08-29
   https://ericniebler.com/2021/08/29/asynchronous-stacks-and-scopes/
3. Lucian Radu Teodorescu, “Senders/Receivers: An Introduction”, Overload 32(184), 2024-12
   https://accu.org/journals/overload/32/184/teodorescu/
4. Lucian Radu Teodorescu, “Using Senders/Receivers”, Overload 33(185), 2025-02
   https://accu.org/journals/overload/33/185/teodorescu/
5. Lucian Radu Teodorescu, “Structured Concurrency in C++”, Overload 30(168), 2022
   https://accu.org/journals/overload/30/168/teodorescu/
6. Clang documentation, “Debugging C++ Coroutines”
   https://clang.llvm.org/docs/DebuggingCoroutines.html
