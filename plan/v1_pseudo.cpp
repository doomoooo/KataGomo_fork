// v1_pseudo.cpp
//
// 这不是可直接编译的代码，而是“实际动工时应当尽量贴近的 C++ 摹本”。
//
// 本文件的目标不是展示某个最小 demo，而是把最终重构时最关键的调度逻辑、
// ownership 边界、线程/协程/slot 的职责划分，一次性铺平。
//
// ---------------------------------------------------------------------------
// 设计取舍（与 v1.md 对齐，同时考虑 KataGo 现有工程）
// ---------------------------------------------------------------------------
//
// 1. 保留 MCTS 的递归形状。
//    Search::playoutDescend() 当前的递归逻辑很深，直接改成显式状态机会把 Search
//    子系统整体重写一遍，这明显超出本次重构预期。
//    因此这里选择：
//      - 让 playout 继续保持“递归函数”的形状；
//      - 但把它提升为 exec::task<bool>，只在 NN 边界真正 suspend。
//
// 2. Search 侧自己参与 batch 构建。
//    这意味着：
//      - Search 线程负责准备输入（pre-process / fillRow / packRow）；
//      - Search 线程负责在 batch 中 claim 一行；
//      - Search 线程负责在结果返回后做 post-process / storeNNOutput；
//      - 单线程 GPU scheduler 不再集中帮 Search 做 preprocess/postprocess。
//
// 3. GPU 资源管理尽量借现有代码。
//    这里直接沿用现有 TensorRT 路径已经成熟的资源壳子与 helper：
//      - ComputeHandle
//      - InputBuffers / NNServerBuf
//      - trtPackInputRow / trtEnqueueInputRowCopy / trtLaunchInferenceAsync /
//        trtEnqueueOutputCopiesAsync / trtUnpackOutputRow / query helpers
//    但把“谁在什么时机调用它们”改写为 v1 调度模型。
//
// 4. 关注扩展性。
//    本文件显式为以下能力预留接口与状态位：
//      - root 切换
//      - 手动 pause / resume
//      - 多个 Search 实例共存
//      - 多个 evaluator 共存
//    这些功能本次不必 fully implement，但状态边界要从一开始就放对位置。
//
// 5. stdexec 的使用策略。
//    不是把所有步骤都 sender 化，而是：
//      - 常驻 worker loop / scheduler loop -> exec::task<void>
//      - 真正的异步边界 -> native sender/event/scheduler
//      - 热路径递归搜索逻辑 -> 普通 C++ + task 递归
//
// ---------------------------------------------------------------------------
// 本文件刻意使用现有 KataGo 类型名，减少迁移阻力：
//   Search / SearchThread / SearchNode / NNEvaluator / NNResultBuf / NNServerBuf /
//   ComputeHandle / InputBuffers / SchedulerState ...
// ---------------------------------------------------------------------------

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <stdexec/execution.hpp>
#include <exec/async_scope.hpp>
#include <exec/single_thread_context.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/task.hpp>

namespace ex = stdexec;

// ============================================================================
// 0. native async primitives
// ============================================================================

// 与 v1.md 一致，这里只定义“原生等待点”，而不去包装旧 callback。
// 下面两个 primitives 是整个系统里最重要的 async 基元：
//
// 1. Notice:
//    - 多生产者、多消费者的“有工作了”通知。
//    - 支持 wait until predicate。
//    - Search worker loop / GPU scheduler loop 都依赖它。
//
// 2. OneShotEvent:
//    - 单次完成事件。
//    - 一个 playout task 在等待某个 NN 结果返回时使用。
//    - GPU scheduler 在 publish 某一行结果时将其 complete。
//
// 注意：
// - 这里不展开 operation_state 的细节实现；
// - 实际编码时，它们都应该做成 zero-allocation / intrusive waiter list；
// - 必须正确传播 stop token。

struct Notice {
  template <class Pred>
  auto async_wait(Pred pred) -> ex::sender auto;

  void notify_one() noexcept;
  void notify_all() noexcept;
};

struct OneShotEvent {
  auto async_wait() -> ex::sender auto;
  void notify() noexcept;
  void reset() noexcept;
};

struct SchedulerState;

// ============================================================================
// 1. Search continuation scheduler
// ============================================================================

// 这是本版最关键的“桥”。
//
// 为什么需要它：
// - exec::task 默认 sticky；
// - 但 v1 要求 NN 返回后的 continuation 是高优先级，并可被任意搜索线程恢复；
// - 所以不能简单让 task 在原 worker 上等待后自己恢复。
//
// 解决办法：
// - 定义一个 custom scheduler，名字就叫 SearchContinuationScheduler；
// - schedule() 的语义不是“切到某个固定线程”，而是：
//     把当前 continuation 入队到高优先级 continuation queue，然后唤醒任一搜索 worker。
//
// 然后 playout task 在 NN 边界前：
//   co_await exec::reschedule_coroutine_on(searchRuntime.continuationScheduler());
//   co_await pendingEval.completion.async_wait();
//
// 这样一来：
// - 递归 task 形状保住了；
// - continuation 不再绑定原搜索线程；
// - Search worker loop 仍然能统一做“continuation 优先于新 root”的调度。

struct SearchContinuationScheduler {
  struct Runtime;
  Runtime* runtime = nullptr;

  struct sender;

  auto schedule() const noexcept -> sender;
  bool operator==(const SearchContinuationScheduler&) const noexcept = default;
};

struct SearchContinuationScheduler::Runtime {
  // 队列里放的不是业务 frame，而是 coroutine continuation opstate。
  // 这样既能保住递归协程形状，又能满足任意 worker 恢复。
  IntrusiveContinuationQueue ready;

  // continuation 和“允许生成新 root”共用同一个 work notice。
  // 这样 Search worker loop 只需要等待一个统一的“有活可干”信号。
  Notice* wakeSearchWorkers = nullptr;
};

struct SearchContinuationScheduler::sender {
  using sender_concept = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

  Runtime* runtime = nullptr;

  template <class Receiver>
  struct operation {
    Runtime* runtime;
    Receiver receiver;

    void start() & noexcept {
      // 实际实现里，这里应把当前 continuation 的恢复动作挂到 intrusive queue，
      // 然后唤醒一个搜索 worker。
      runtime->ready.push(this);
      runtime->wakeSearchWorkers->notify_one();
    }

    static void resume(operation* self) noexcept {
      ex::set_value(std::move(self->receiver));
    }
  };

  template <class Receiver>
  auto connect(Receiver rcvr) -> operation<Receiver> {
    return operation<Receiver>{runtime, std::move(rcvr)};
  }
};

inline auto SearchContinuationScheduler::schedule() const noexcept -> sender {
  return sender{runtime};
}

// ============================================================================
// 2. Search-side runtime
// ============================================================================

struct SearchControlPlane {
  // stop 整个 search session
  std::atomic<bool> stopRequested{false};

  // 只阻止“生成新的 root playout”，不阻止 continuation 恢复
  std::atomic<bool> manualPause{false};

  // root 切换时递增，所有 in-flight continuation 恢复时都要校验
  std::atomic<std::uint64_t> rootEpoch{1};

  // v1 credit / gate
  std::atomic<std::uint64_t> nnIssueCurrent{0};
  std::atomic<std::uint64_t> nnIssueTarget{0};
};

struct SearchStdexecRuntime {
  Search* owner = nullptr;

  exec::async_scope        scope;
  exec::static_thread_pool workerPool;
  Notice                   workerNotice;
  SearchContinuationScheduler::Runtime continuationRuntime;
  SearchContinuationScheduler          continuationScheduler;

  explicit SearchStdexecRuntime(Search* search, int numThreads)
    : owner(search),
      workerPool(numThreads),
      workerNotice(),
      continuationRuntime(),
      continuationScheduler{&continuationRuntime}
  {
    continuationRuntime.wakeSearchWorkers = &workerNotice;
  }
};

// 迁移时，SearchThread 不再等同于“OS thread 上的常驻 scratch state”，
// 而是“一个 playout task 的递归上下文”。
//
// 这能最大程度保住现有 Search::playoutDescend(...) 对 SearchThread 的使用方式。
//
// 也就是说，本次重构建议做的最小语义迁移是：
//   SearchThread => SearchPlayoutContext
// 但为了减少 diff，可以先保留名字 SearchThread。
struct SearchTaskState {
  Search*       ownerSearch = nullptr;
  SearchThread* threadState = nullptr;   // 仍沿用现有字段布局
  std::uint64_t rootEpochAtSpawn = 0;
  int           workerHint = -1;         // 仅做 locality hint，不强绑定

  // 一个 playout 只会在一个 NN 边界上 suspend 多次，所以复用同一个 pending buf 足够。
  struct PendingEval {
    NNResultBuf*   resultBuf = nullptr;
    SchedulerState* schedulerState = nullptr;
    int            handleIdx = -1;
    int            row = -1;
    OneShotEvent   completion;
    bool           isActive = false;
    bool           isHuman = false;
  };

  PendingEval mainEval;
  PendingEval humanEval;
};

// ============================================================================
// 3. GPU-side scheduler state
// ============================================================================

// 为了减小迁移阻力，这里保留现有 SchedulerState 的整体命名。
// 但内部语义改成更接近 v1：
// - Search 线程自己 claim row + pack row
// - 单线程 scheduler 只负责 batch 生命周期 / H2D / infer / D2H / publish

struct SchedulerState {
  enum class HandleStage {
    Free,
    Open,
    Sealed,
    H2DSubmittedPartial,
    InferRunning,
    D2HPending,
    Published
  };

  struct SlotState {
    int slotIdx = -1;
    int gpuIdx = -1;
    int deviceStateIdx = -1;
    ComputeHandle* gpuHandle = nullptr;

    // 当前对 Search 开放 claim 的 handle。
    std::atomic<int> openHandleIdx{-1};

    // 已经 launch、等待 infer query 的 handle 队列。
    std::deque<int> launchedHandleIndices;

    // infer 已完成、等待 D2H query 的 handle 队列。
    std::deque<int> d2hPendingHandleIndices;

    // 用于 idle seal / timeline 估计。
    double remainingWorkMs = 0.0;
    bool isUsingFP16 = false;
  };

  struct HandleState {
    int handleIdx = -1;
    int slotIdx = -1;
    NNServerBuf* serverBuf = nullptr;

    std::atomic<HandleStage> stage{HandleStage::Free};

    // v1 的 open batch 核心状态：
    // - claimedRows: 已 claim 的总数
    // - sealedRows: seal 后固定大小
    // - h2dSubmittedRows: scheduler 已经发起 H2D 的行数
    // - readyRows: Search 线程已经完成 pack/publish 的行数
    // - remainingConsumers: post_process 尚未完成的消费者数
    std::atomic<int> claimedRows{0};
    std::atomic<int> sealedRows{0};
    std::atomic<int> h2dSubmittedRows{0};
    std::atomic<int> readyRows{0};
    std::atomic<int> remainingConsumers{0};

    // 只允许一次 seal/reservation。
    std::atomic<bool> sealRequested{false};
    std::atomic<bool> reservationCommitted{false};

    // Search 侧仍然通过 NNResultBuf 与 scheduler 交互，减少类型改动。
    std::vector<NNResultBuf*> requests;
    std::vector<OneShotEvent*> asyncCompletions;
    std::vector<NNOutput*>    outputs;

    // 每一行一个 ready flag，比 packed bitset 更直观，也更接近 v1.md 的发布语义。
    std::vector<std::atomic<uint8_t>> rowReady;

    double plannedInferMs = 0.0;
    double accumulatedEquivalentWorkMs = 0.0;

    void resetForReuse(int maxBatchSize) {
      stage.store(HandleStage::Free, std::memory_order_relaxed);
      slotIdx = -1;
      claimedRows.store(0, std::memory_order_relaxed);
      sealedRows.store(0, std::memory_order_relaxed);
      h2dSubmittedRows.store(0, std::memory_order_relaxed);
      readyRows.store(0, std::memory_order_relaxed);
      remainingConsumers.store(0, std::memory_order_relaxed);
      sealRequested.store(false, std::memory_order_relaxed);
      reservationCommitted.store(false, std::memory_order_relaxed);
      plannedInferMs = 0.0;
      accumulatedEquivalentWorkMs = 0.0;
      if((int)requests.size() != maxBatchSize)
        requests.resize(maxBatchSize);
      if((int)asyncCompletions.size() != maxBatchSize)
        asyncCompletions.resize(maxBatchSize);
      if((int)outputs.size() != maxBatchSize)
        outputs.resize(maxBatchSize);
      if((int)rowReady.size() != maxBatchSize)
        rowReady.resize(maxBatchSize);
      for(int i = 0; i < maxBatchSize; i++) {
        requests[i] = nullptr;
        asyncCompletions[i] = nullptr;
        outputs[i] = nullptr;
        rowReady[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  struct DeviceState {
    int gpuIdx = -1;
    std::vector<int> slotIndices;
    std::vector<int> handleIndices;
    int rrCursor = 0;
    int rrHandleCursor = 0;
    int activeInferCount = 0;
    int64_t lastProgressNs = 0;
    std::vector<double> baseWorkMsByBatch;
    std::vector<std::vector<double>> workSamplesByBatch;
  };

  // 每个 Search 实例单独一份 runtime；不允许用全局单例。
  // 这样未来多个 evaluator 可以直接并存。
  NNEvaluator* owner = nullptr;

  std::mutex  stateMutex;
  Notice      schedulerNotice;
  std::vector<SlotState>   slots;
  std::vector<HandleState> handles;
  std::vector<DeviceState> devices;

  std::atomic<bool> stopRequested{false};

  // 这份 rand 只给 scheduler 自己用，不与 SearchThread 混用。
  Rand rand;

  explicit SchedulerState(NNEvaluator* nneval, const std::string& seed)
    : owner(nneval), stateMutex(), schedulerNotice(), slots(), handles(), devices(), rand(seed) {}
};

// ============================================================================
// 4. Search-facing async NN ticket
// ============================================================================

// 为了减少改动，Search 仍然沿用 NNResultBuf 当“请求+结果”的壳子。
// 只是对 Search path，新增 async completion 路径：
//
// - Blocking caller:
//     evaluate() -> submitPackedRequest() -> wait on condition_variable
//
// - Search coroutine:
//     fillAndSubmitPackedRequestAsync() -> submitPackedRequestFromSearch(..., &completion)
//                                      -> co_await completion.async_wait()
//
// - Blocking caller:
//     submitPackedRequestFromSearch(..., nullptr) -> wait on condition_variable
//
// 两条路复用同一个 scheduler 与同一个 result publication 协议。
//
// 注意：
// - row 级别的 scheduler 元数据（async completion / handle ownership）不塞进
//   NNResultBuf，而是放在 HandleState 与这个 ticket 里；
// - 这样更贴近当前 KataGo 的 NNResultBuf 角色，也更利于渐进迁移。

struct AsyncEvalTicket {
  NNResultBuf* resultBuf = nullptr;
  SchedulerState* schedulerState = nullptr;
  int handleIdx = -1;
  int row = -1;
};

// ============================================================================
// 5. Search worker loops
// ============================================================================

// Search worker loop 只做两件事：
// 1. 优先恢复 continuationScheduler 送回来的递归 playout continuation。
// 2. continuation 空时，若 gate 允许，则创建新的 root playout task。
//
// 这里不直接写“线程永远持有一个 SearchThread scratch state”。
// 因为 continuation 可能迁移到别的 worker，上下文必须跟 task 走。

static bool canSpawnNewRoot(const SearchControlPlane& ctl) {
  if(ctl.stopRequested.load(std::memory_order_acquire))
    return false;
  if(ctl.manualPause.load(std::memory_order_acquire))
    return false;
  return ctl.nnIssueCurrent.load(std::memory_order_acquire) <
         ctl.nnIssueTarget.load(std::memory_order_acquire);
}

auto searchWorkerLoop(Search& search, SearchStdexecRuntime& rt, int workerIdx) -> exec::task<void> {
  while(true) {
    co_await rt.workerNotice.async_wait([&]() {
      return search.controlPlane.stopRequested.load(std::memory_order_acquire) ||
             !rt.continuationRuntime.ready.empty() ||
             canSpawnNewRoot(search.controlPlane);
    });

    if(search.controlPlane.stopRequested.load(std::memory_order_acquire))
      co_return;

    // 先恢复高优先级 continuation。
    if(rt.continuationRuntime.ready.try_resume_one()) {
      continue;
    }

    // 再考虑新 root。
    if(canSpawnNewRoot(search.controlPlane)) {
      SearchTaskState* task = search.acquireTaskState();
      task->ownerSearch = &search;
      task->threadState = search.acquirePlayoutThreadState();
      task->rootEpochAtSpawn = search.controlPlane.rootEpoch.load(std::memory_order_acquire);
      task->workerHint = workerIdx;

      search.prepareFreshRootTask(*task);

      auto workerSched = rt.workerPool.get_scheduler_on_thread(workerIdx);

      // starts_on:
      // - 让新 root playout 从当前 worker 启动；
      // - 后续若遇到 NN 边界，再显式迁移到 continuationScheduler。
      rt.scope.spawn(ex::starts_on(workerSched, search.runSinglePlayoutTask(*task)));
      continue;
    }
  }
}

// ============================================================================
// 6. Search task entrypoint
// ============================================================================

auto Search::runSinglePlayoutTask(SearchTaskState& task) -> exec::task<void> {
  SearchThread& thread = *task.threadState;

  // 与现有 runSinglePlayout 基本同形，只是变成 task。
  thread.upperBoundVisitsLeft = computeUpperBoundVisitsLeftForFreshRootTask();
  thread.waitNNEvalTimeThisPlayoutMs = 0.0;
  thread.shouldCountPlayout = true;

  bool finishedPlayout = co_await playoutDescendTask(task, *rootNode, true);
  (void)finishedPlayout;

  // 恢复根状态，与现有 runSinglePlayout 保持一致。
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
  thread.graphHash = rootGraphHash;
  thread.graphPath.clear();

  if(thread.shouldCountPlayout)
    onPlayoutFinished(task, /*countPlayout=*/true);
  else
    onPlayoutFinished(task, /*countPlayout=*/false);

  releaseTaskState(task);
  co_return;
}

// ============================================================================
// 7. 保留递归形状的 playoutDescendTask
// ============================================================================

// 这段是整份 pseudo 最重要的迁移点：
// - 代码形状尽量贴当前 Search::playoutDescend；
// - 只把“叶子触发 NN eval”的部分改成 co_await async helper。

auto Search::playoutDescendTask(
  SearchTaskState& task,
  SearchNode& node,
  bool isRoot
) -> exec::task<bool> {
  SearchThread& thread = *task.threadState;

  if(thread.history.isGameFinished && !node.forceNonTerminal) {
    co_await waitForInFlightEvalIfAnyAsync(task);
    addTerminalLeafValue(node, thread);
    co_return true;
  }

  SearchNodeState nodeState = node.state.load(std::memory_order_acquire);
  if(nodeState == SearchNode::STATE_UNEVALUATED) {
    bool suc = co_await initNodeNNOutputAsync(task, node, isRoot, /*skipCache=*/false, /*isReInit=*/false);
    if(!suc) {
      thread.shouldCountPlayout = false;
      co_return false;
    }

    bool wonRace = node.state.compare_exchange_strong(
      nodeState, SearchNode::STATE_EVALUATING, std::memory_order_seq_cst);
    if(!wonRace) {
      thread.shouldCountPlayout = false;
      co_return false;
    }

    node.initializeChildren();
    node.state.store(SearchNode::STATE_EXPANDED0, std::memory_order_seq_cst);
    co_return true;
  }
  else if(nodeState == SearchNode::STATE_EVALUATING) {
    thread.shouldCountPlayout = false;
    co_return false;
  }

  maybeRecomputeExistingNNOutputFastPath(task, node, isRoot);

  // 下面的 child 选择 / allocate / virtual loss / graph cycle 逻辑，
  // 尽量复用现有 playoutDescend 的代码结构，不在本 pseudo 里重新设计算法。
  //
  // 重点仅在“递归进入 child”仍然是递归 await，而不是改写成显式栈。
  ChildSelection sel = selectBestChildToDescendPseudo(thread, node, nodeState, isRoot);

  if(sel.kind == ChildSelectionKind::TreatAsLeaf) {
    addCurrentNNOutputAsLeafValue(node, false);
    co_return true;
  }

  SearchNode* child = co_await materializeOrFindChildPseudo(task, node, nodeState, sel, isRoot);
  if(child == nullptr) {
    thread.shouldCountPlayout = false;
    co_return false;
  }

  if(isCyclePseudo(thread, child, sel.countEdgeVisit)) {
    co_return sel.countEdgeVisit;
  }

  bool shouldUpdateChildAncestors = co_await playoutDescendTask(task, *child, false);
  shouldUpdateChildAncestors = shouldUpdateChildAncestors && sel.countEdgeVisit;

  if(shouldUpdateChildAncestors) {
    SearchNodeState afterChildState = node.state.load(std::memory_order_acquire);
    SearchNodeChildrenReference children = node.getChildren(afterChildState);
    children[sel.bestChildIdx].addEdgeVisits(1);
    updateStatsAfterPlayout(node, thread, isRoot);
  }
  child->virtualLosses.fetch_add(-1, std::memory_order_release);

  co_return shouldUpdateChildAncestors;
}

// ============================================================================
// 8. Async NN helper on Search side
// ============================================================================

// 这一层的职责：
// 1. 基于当前 SearchThread 状态生成 nnInputParams
// 2. 在 Search 线程上 fillRow / packRow
// 3. claim 一个 batch row
// 4. 在真正 await 之前，把当前递归 task 迁移到 continuationScheduler
// 5. 等结果返回后，保持现有 node.storeNNOutput / addLeafValue 语义

auto Search::initNodeNNOutputAsync(
  SearchTaskState& task,
  SearchNode& node,
  bool isRoot,
  bool skipCache,
  bool isReInit
) -> exec::task<bool> {
  SearchThread& thread = *task.threadState;

  bool includeOwnerMap = isRoot || alwaysIncludeOwnerMap;

  MiscNNInputParams nnInputParams = buildNNInputParamsForNode(thread, isRoot);

  // 这里仍然保留当前 SearchNN helper 的“算 hash / 查 cache / 可能做 root noise”职责。
  if(tryFastPathFromCachePseudo(thread, node, isRoot, nnInputParams, includeOwnerMap, skipCache, isReInit))
    co_return true;

  // --- main evaluator request ------------------------------------------------
  preparePackedNNRequestOnSearchThread(
    *nnEvaluator,
    thread,
    nnInputParams,
    &searchParams.humanSLProfile,
    includeOwnerMap,
    /*out*/ thread.nnResultBuf
  );

  task.mainEval.resultBuf = &thread.nnResultBuf;
  task.mainEval.schedulerState = nullptr;
  task.mainEval.handleIdx = -1;
  task.mainEval.row = -1;
  task.mainEval.completion.reset();
  task.mainEval.isActive = true;
  task.mainEval.isHuman = false;
  {
    AsyncEvalTicket ticket = submitPackedRequestFromSearch(
      *nnEvaluator,
      thread.nnResultBuf,
      &task.mainEval.completion
    );
    task.mainEval.schedulerState = ticket.schedulerState;
    task.mainEval.handleIdx = ticket.handleIdx;
    task.mainEval.row = ticket.row;
  }

  // --- optional human evaluator request --------------------------------------
  std::optional<NNResultBuf> humanBuf;
  task.humanEval.resultBuf = nullptr;
  task.humanEval.schedulerState = nullptr;
  task.humanEval.handleIdx = -1;
  task.humanEval.row = -1;
  task.humanEval.isActive = false;
  if(needsHumanOutputInTree() || (isRoot && needsHumanOutputAtRoot())) {
    humanBuf.emplace();
    preparePackedNNRequestOnSearchThread(
      *humanEvaluator,
      thread,
      nnInputParams,
      &searchParams.humanSLProfile,
      includeOwnerMap,
      /*out*/ *humanBuf
    );

    task.humanEval.resultBuf = &*humanBuf;
    task.humanEval.completion.reset();
    task.humanEval.isActive = true;
    task.humanEval.isHuman = true;
    {
      AsyncEvalTicket ticket = submitPackedRequestFromSearch(
        *humanEvaluator,
        *humanBuf,
        &task.humanEval.completion
      );
      task.humanEval.schedulerState = ticket.schedulerState;
      task.humanEval.handleIdx = ticket.handleIdx;
      task.humanEval.row = ticket.row;
    }
  }

  // 关键点：
  // 在等待 NN 前，先显式迁移到 continuation scheduler。
  // 这样结果返回后，恢复发生在 continuation 高优先级队列上，而不是绑回原 worker。
  co_await exec::reschedule_coroutine_on(searchStdexecRuntime->continuationScheduler);

  if(humanBuf.has_value()) {
    co_await ex::when_all(
      task.mainEval.completion.async_wait(),
      task.humanEval.completion.async_wait()
    );
  }
  else {
    co_await task.mainEval.completion.async_wait();
  }

  // completion 只代表“scheduler 已经 publish 到 NNResultBuf”。
  // handle 真正可复用，要等 Search 侧明确声明自己已经消费完结果。
  if(task.mainEval.isActive) {
    releasePublishedHandleConsumerPseudo(*task.mainEval.schedulerState, task.mainEval.handleIdx);
    task.mainEval.resultBuf = nullptr;
    task.mainEval.schedulerState = nullptr;
    task.mainEval.handleIdx = -1;
    task.mainEval.row = -1;
    task.mainEval.isActive = false;
  }
  if(task.humanEval.isActive) {
    releasePublishedHandleConsumerPseudo(*task.humanEval.schedulerState, task.humanEval.handleIdx);
    task.humanEval.resultBuf = nullptr;
    task.humanEval.schedulerState = nullptr;
    task.humanEval.handleIdx = -1;
    task.humanEval.row = -1;
    task.humanEval.isActive = false;
  }

  // root 切换后，旧 continuation 可以平滑自杀，不再污染新 root 搜索。
  if(task.rootEpochAtSpawn != controlPlane.rootEpoch.load(std::memory_order_acquire)) {
    thread.shouldCountPlayout = false;
    co_return false;
  }

  // 把结果接回当前 node，逻辑尽量复用现有 initNodeNNOutput。
  std::shared_ptr<NNOutput>* result = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
  std::shared_ptr<NNOutput>* humanResult =
    humanBuf.has_value() ? new std::shared_ptr<NNOutput>(std::move(humanBuf->result)) : nullptr;

  if(isRoot) {
    std::shared_ptr<NNOutput>* noised = maybeAddPolicyNoiseAndTemp(thread, true, result->get());
    if(noised != nullptr) {
      delete result;
      result = noised;
    }
  }

  node.nodeAge.store(searchNodeAge, std::memory_order_release);
  if(isReInit) {
    if(humanResult != nullptr)
      node.storeHumanOutput(humanResult, thread);
    bool wasNullBefore = node.storeNNOutput(result, thread);
    co_return wasNullBefore;
  }
  else {
    if(humanResult != nullptr) {
      bool humanSuc = node.storeHumanOutputIfNull(humanResult);
      if(!humanSuc)
        delete humanResult;
    }
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc) {
      delete result;
    }
    else {
      addCurrentNNOutputAsLeafValue(node, true);
    }
    co_return suc;
  }
}

// ============================================================================
// 9. Search-side request preparation
// ============================================================================

// 这里是迁移阻力最小的一步：
// - 现有 evaluate() 内部做的 NNInputs::fillRowV* 逻辑搬出来；
// - 仍然复用 NNResultBuf.{rowSpatialBuf,rowGlobalBuf,rowMetaBuf}；
// - 只是 ownership 从 “NNEvaluator 线程预处理” 转成 “Search 线程预处理”。

static void preparePackedNNRequestOnSearchThread(
  NNEvaluator& evaluator,
  SearchThread& thread,
  const MiscNNInputParams& nnInputParams,
  const SGFMetadata* sgfMeta,
  bool includeOwnerMap,
  NNResultBuf& out
) {
  out.hasResult = false;
  out.includeOwnerMap = includeOwnerMap;
  out.boardXSizeForServer = thread.board.x_size;
  out.boardYSizeForServer = thread.board.y_size;
  out.errorLogLockout = false;
  out.policyOptimism = nnInputParams.policyOptimism;

  // 复用现有 NNInputs::fillRowV* 路径。
  fillRowBuffersPseudo(
    evaluator,
    thread.board,
    thread.history,
    thread.pla,
    sgfMeta,
    nnInputParams,
    out.rowSpatialBuf,
    out.rowGlobalBuf,
    out.rowMetaBuf,
    out.hasRowMeta
  );
}

// ============================================================================
// 10. Search -> evaluator 提交路径
// ============================================================================

// 这条路径是 v1 的核心：
// - Search 线程自己 claim row；
// - Search 线程自己把 row pack 到目标 handle 的 host staging；
// - scheduler 线程只负责后续 H2D / infer / D2H / publish。

struct ClaimResult {
  SchedulerState::HandleState* handle = nullptr;
  SchedulerState::SlotState*   slot = nullptr;
  int row = -1;
  bool sealedByCaller = false;
};

static ClaimResult claimRowForSearchRequest(
  NNEvaluator& nneval,
  SchedulerState& state,
  NNResultBuf& request,
  OneShotEvent* asyncCompletion
) {
  // 实际实现中这里需要严格的原子协议。
  // 为了讲清调度，这里伪代码化为：
  //
  // 1. 按 timeline / remainingWork 选一个 slot。
  // 2. 拿到该 slot 当前 open handle；没有则分配一个。
  // 3. 原子 claim row。
  // 4. 满 batch 时原子 seal，并负责 reservation。

  std::lock_guard<std::mutex> lock(state.stateMutex);

  SchedulerState::SlotState* slot = pickSlotForNewClaimPseudo(state);

  // getOrCreateOpenHandlePseudo 必须保证：
  // - 返回的 open handle 已经完成 resetForReuse(maxBatchSize)；
  // - 因而 requests/asyncCompletions/rowReady 都已具备 row 索引能力。
  SchedulerState::HandleState* handle = getOrCreateOpenHandlePseudo(state, *slot);

  int row = handle->claimedRows.fetch_add(1, std::memory_order_acq_rel);
  bool sealedByCaller = false;

  if(row == 0) {
    handle->stage.store(SchedulerState::HandleStage::Open, std::memory_order_release);
  }

  if(row + 1 == nneval.getCurrentBatchSize()) {
    sealedByCaller = trySealHandleAndReservePseudo(nneval, state, *slot, *handle, row + 1);
  }

  // 用 row 直接索引元数据，避免依赖 push_back 顺序来隐式绑定 row。
  handle->requests[row] = &request;
  handle->asyncCompletions[row] = asyncCompletion;
  return ClaimResult{handle, slot, row, sealedByCaller};
}

static AsyncEvalTicket submitPackedRequestFromSearch(
  NNEvaluator& nneval,
  NNResultBuf& request,
  OneShotEvent* asyncCompletion
) {
  SchedulerState& state = *nneval.schedulerState;

  ClaimResult claim = claimRowForSearchRequest(nneval, state, request, asyncCompletion);

  SchedulerState::HandleState& handle = *claim.handle;
  SchedulerState::SlotState& slot = *claim.slot;

  // Search 线程把 row 内容 pack 到目标 batch 的 host staging。
  // 这一步直接复用现有 trtPackInputRow helper。
  NeuralNet::trtPackInputRow(
    handle.serverBuf->inputBuffers,
    &request,
    claim.row,
    slot.gpuHandle
  );

  request.hasResult = false;

  // release store: 发布 row 数据已经可由 scheduler 线程消费。
  handle.rowReady[claim.row].store(1, std::memory_order_release);
  handle.readyRows.fetch_add(1, std::memory_order_release);

  // 通知 scheduler：
  // - 可能有新的 ready row 可以做 H2D；
  // - 也可能 caller 刚刚 seal 了一个 batch。
  state.schedulerNotice.notify_one();

  return AsyncEvalTicket{
    .resultBuf = &request,
    .schedulerState = &state,
    .handleIdx = handle.handleIdx,
    .row = claim.row,
  };
}

// ============================================================================
// 11. Legacy blocking API 兼容层
// ============================================================================

// 这层的存在是为了减小迁移阻力。
// 也就是说：
// - 外部仍然可以调用现有 NNEvaluator::evaluate(...)；
// - 它内部不必再走老 queryQueue batching；
// - 而是走同一个“fillRow -> claim row -> scheduler publish”的新路径；
// - 最后只是继续用条件变量阻塞等待结果。

void NNEvaluator::evaluate(
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  assert(!isKilled);

  if(tryCacheFastPathPseudo(board, history, nextPlayer, sgfMeta, nnInputParams, buf, skipCache, includeOwnerMap))
    return;

  preparePackedNNRequestForBlockingCallerPseudo(
    *this,
    board,
    history,
    nextPlayer,
    sgfMeta,
    nnInputParams,
    includeOwnerMap,
    buf
  );

  AsyncEvalTicket ticket = submitPackedRequestFromSearch(*this, buf, /*asyncCompletion=*/nullptr);

  // 兼容现有 blocking evaluate 语义。
  std::unique_lock<std::mutex> lock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(lock);
  lock.unlock();

  // blocking caller 与 Search coroutine 一样，必须在消费完结果后显式归还 handle。
  releasePublishedHandleConsumerPseudo(*ticket.schedulerState, ticket.handleIdx);
}

// ============================================================================
// 12. 单线程 GPU scheduler loop
// ============================================================================

// 这部分是大规模重写的核心，但应尽量借用当前 TRT backend 的资源管理 helper。

auto NNEvaluator::serveTrtSchedulerTask() -> exec::task<void> {
  SchedulerState& state = *schedulerState;

  initializeSlotsAndHandlesPseudo(state);
  initializeBaseWorkEstimatesPseudo(state);

  while(true) {
    co_await state.schedulerNotice.async_wait([&]() {
      return state.stopRequested.load(std::memory_order_acquire) ||
             hasAnyOpenOrInflightWorkPseudo(state);
    });

    if(state.stopRequested.load(std::memory_order_acquire) && allWorkDrainedPseudo(state))
      break;

    int64_t nowNs = steadyClockNowNsPseudo();
    advanceAllDeviceProgressPseudo(state, nowNs);

    // -----------------------------------------------------------------------
    // 1. 对每个 slot 的 open handle：
    //    - 如果 GPU idle 且有部分 batch，则 idle seal
    //    - 对新 ready 的 row 发起 H2D
    //    - 若已 seal 且所有 row 均已 H2D，就 launch inference
    // -----------------------------------------------------------------------
    for(SchedulerState::SlotState& slot: state.slots) {
      SchedulerState::HandleState* open = getOpenHandlePseudo(state, slot);
      if(open != nullptr) {
        maybeIdleSealHandlePseudo(*this, state, slot, *open);
        submitReadyRowsH2DPseudo(*this, state, slot, *open);
        maybeLaunchSealedHandlePseudo(*this, state, slot, *open);
      }
    }

    // -----------------------------------------------------------------------
    // 2. 查询 launched handles 的 infer 完成
    // -----------------------------------------------------------------------
    for(SchedulerState::SlotState& slot: state.slots) {
      if(slot.launchedHandleIndices.empty())
        continue;
      SchedulerState::HandleState& handle = state.handles[slot.launchedHandleIndices.front()];
      if(queryInferenceDonePseudo(*this, slot, handle)) {
        moveToD2HPendingPseudo(*this, state, slot, handle);
      }
    }

    // -----------------------------------------------------------------------
    // 3. 查询 D2H 完成并 publish
    // -----------------------------------------------------------------------
    for(SchedulerState::SlotState& slot: state.slots) {
      if(slot.d2hPendingHandleIndices.empty())
        continue;
      SchedulerState::HandleState& handle = state.handles[slot.d2hPendingHandleIndices.front()];
      if(queryD2HDonePseudo(*this, slot, handle)) {
        finalizePublishedHandlePseudo(*this, state, slot, handle);
      }
    }
  }

  destroySlotsAndHandlesPseudo(state);
  co_return;
}

// ============================================================================
// 13. scheduler helpers
// ============================================================================

static void submitReadyRowsH2DPseudo(
  NNEvaluator& nneval,
  SchedulerState& state,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  if(handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::Open &&
     handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::Sealed &&
     handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::H2DSubmittedPartial) {
    return;
  }

  int ready = handle.readyRows.load(std::memory_order_acquire);
  int submitted = handle.h2dSubmittedRows.load(std::memory_order_acquire);

  while(submitted < ready) {
    // acquire 对应 Search 线程的 rowReady release publish。
    if(handle.rowReady[submitted].load(std::memory_order_acquire) == 0)
      break;

    NeuralNet::trtEnqueueInputRowCopy(slot.gpuHandle, handle.serverBuf->inputBuffers, submitted);
    handle.h2dSubmittedRows.fetch_add(1, std::memory_order_acq_rel);
    submitted += 1;
  }

  if(submitted > 0)
    handle.stage.store(SchedulerState::HandleStage::H2DSubmittedPartial, std::memory_order_release);
}

static bool trySealHandleAndReservePseudo(
  NNEvaluator& nneval,
  SchedulerState& state,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle,
  int sealedRows
) {
  bool expected = false;
  if(!handle.sealRequested.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    return false;

  handle.sealedRows.store(sealedRows, std::memory_order_release);
  handle.stage.store(SchedulerState::HandleStage::Sealed, std::memory_order_release);

  bool expectedReservation = false;
  if(handle.reservationCommitted.compare_exchange_strong(
         expectedReservation, true, std::memory_order_acq_rel)) {
    reserveGpuTimelinePseudo(nneval, state, slot, sealedRows);
  }

  return true;
}

static void maybeIdleSealHandlePseudo(
  NNEvaluator& nneval,
  SchedulerState& state,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  if(handle.sealRequested.load(std::memory_order_acquire))
    return;

  int claimed = handle.claimedRows.load(std::memory_order_acquire);
  if(claimed <= 0)
    return;

  if(slot.remainingWorkMs <= 0.0) {
    (void)trySealHandleAndReservePseudo(nneval, state, slot, handle, claimed);
  }
}

static void maybeLaunchSealedHandlePseudo(
  NNEvaluator& nneval,
  SchedulerState& state,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  if(handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::Sealed &&
     handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::H2DSubmittedPartial) {
    return;
  }

  int sealed = handle.sealedRows.load(std::memory_order_acquire);
  int submitted = handle.h2dSubmittedRows.load(std::memory_order_acquire);
  if(sealed <= 0 || submitted < sealed)
    return;

  handle.plannedInferMs = estimateBatchWorkMsPseudo(state, slot, sealed);
  handle.accumulatedEquivalentWorkMs = 0.0;
  allocateOutputsForHandlePseudo(handle, sealed);

  NeuralNet::trtLaunchInferenceAsync(slot.gpuHandle, handle.serverBuf->inputBuffers, sealed);
  slot.launchedHandleIndices.push_back(handle.handleIdx);
  slot.remainingWorkMs += handle.plannedInferMs;
  handle.stage.store(SchedulerState::HandleStage::InferRunning, std::memory_order_release);

  clearOpenHandleIfMatchesPseudo(slot, handle.handleIdx);
}

static bool queryInferenceDonePseudo(
  NNEvaluator& nneval,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  if(handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::InferRunning)
    return false;
  return NeuralNet::trtQueryInferenceDone(handle.serverBuf->inputBuffers);
}

static void moveToD2HPendingPseudo(
  NNEvaluator& nneval,
  SchedulerState& state,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  int batchSize = handle.sealedRows.load(std::memory_order_acquire);
  updateGpuEstimatePseudo(state, slot, handle, batchSize);
  reconcileGpuTimelinePseudo(nneval, state, slot);

  NeuralNet::trtEnqueueOutputCopiesAsync(slot.gpuHandle, handle.serverBuf->inputBuffers, batchSize);
  slot.d2hPendingHandleIndices.push_back(handle.handleIdx);
  if(!slot.launchedHandleIndices.empty() && slot.launchedHandleIndices.front() == handle.handleIdx)
    slot.launchedHandleIndices.pop_front();
  handle.stage.store(SchedulerState::HandleStage::D2HPending, std::memory_order_release);

  updateSearchNNTargetNumPseudo(nneval);
  notifySearchIfGateReopenedPseudo(nneval);
}

static bool queryD2HDonePseudo(
  NNEvaluator& nneval,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  if(handle.stage.load(std::memory_order_acquire) != SchedulerState::HandleStage::D2HPending)
    return false;
  return NeuralNet::trtQueryOutputCopiesDone(handle.serverBuf->inputBuffers);
}

static void finalizePublishedHandlePseudo(
  NNEvaluator& nneval,
  SchedulerState& state,
  SchedulerState::SlotState& slot,
  SchedulerState::HandleState& handle
) {
  int batchSize = handle.sealedRows.load(std::memory_order_acquire);
  handle.remainingConsumers.store(batchSize, std::memory_order_release);

  for(int row = 0; row < batchSize; row++) {
    NNResultBuf* request = handle.requests[row];
    NNOutput* output = handle.outputs[row];
    OneShotEvent* asyncCompletion = handle.asyncCompletions[row];

    // 结果 unpack 仍由 scheduler 线程完成；
    // 但 Search 侧的 post-process / node install 仍在 playout coroutine 恢复后做。
    NeuralNet::trtUnpackOutputRow(handle.serverBuf->inputBuffers, request, output, row, slot.gpuHandle);

    {
      std::unique_lock<std::mutex> resultLock(request->resultMutex);
      request->result = std::shared_ptr<NNOutput>(output);
      request->hasResult = true;
      request->clientWaitingForResult.notify_all();
    }

    if(asyncCompletion != nullptr) {
      asyncCompletion->notify();
    }
  }

  handle.outputs.clear(); // shared_ptr ownership 已转交给各 request
  if(!slot.d2hPendingHandleIndices.empty() && slot.d2hPendingHandleIndices.front() == handle.handleIdx)
    slot.d2hPendingHandleIndices.pop_front();

  handle.stage.store(SchedulerState::HandleStage::Published, std::memory_order_release);
}

// ============================================================================
// 14. Search-side result consumption
// ============================================================================

// Search task在 OneShotEvent 唤醒后，结果已经写进 NNResultBuf.result。
// 此处不再需要额外跨线程 postprocess，仅继续沿用当前 initNodeNNOutput 的 node 赋值逻辑即可。
//
// 但有一件事 scheduler 线程不能替 Search 做：
// - handle 何时真正可以复用，必须等 Search 侧“消费完结果”。

static void releasePublishedHandleConsumerPseudo(SchedulerState& state, int handleIdx) {
  SchedulerState::HandleState& handle = state.handles[handleIdx];

  if(handle.remainingConsumers.fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;

  // 最后一个消费者负责归还 handle。
  std::lock_guard<std::mutex> lock(state.stateMutex);
  handle.resetForReuse(/*maxBatchSize=*/(int)handle.rowReady.size());
}

// ============================================================================
// 15. Search root switch / pause / resume hooks
// ============================================================================

// 这部分本次不要求 fully implement，但一定要在 runtime shape 上预留。

void Search::pauseGeneratingNewRootTasks() {
  controlPlane.manualPause.store(true, std::memory_order_release);
}

void Search::resumeGeneratingNewRootTasks() {
  controlPlane.manualPause.store(false, std::memory_order_release);
  searchStdexecRuntime->workerNotice.notify_all();
}

void Search::notifyRootChanged() {
  // root 切换只需要 bump epoch。
  // in-flight continuation 在恢复时看到 epoch mismatch 会自行放弃。
  controlPlane.rootEpoch.fetch_add(1, std::memory_order_acq_rel);
  searchStdexecRuntime->workerNotice.notify_all();
}

// ============================================================================
// 16. Search / evaluator boot and teardown
// ============================================================================

void Search::spawnStdexecSearchRuntimeIfNeeded() {
  if(searchStdexecRuntime != nullptr)
    return;

  searchStdexecRuntime = std::make_unique<SearchStdexecRuntime>(this, searchParams.numThreads);

  for(int workerIdx = 0; workerIdx < searchParams.numThreads; workerIdx++) {
    auto sched = searchStdexecRuntime->workerPool.get_scheduler_on_thread(workerIdx);
    searchStdexecRuntime->scope.spawn(
      ex::starts_on(sched, searchWorkerLoop(*this, *searchStdexecRuntime, workerIdx))
    );
  }
}

void Search::shutdownStdexecSearchRuntime() {
  if(searchStdexecRuntime == nullptr)
    return;

  controlPlane.stopRequested.store(true, std::memory_order_release);
  searchStdexecRuntime->workerNotice.notify_all();
  ex::sync_wait(searchStdexecRuntime->scope.on_empty());
  searchStdexecRuntime.reset();
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  bool useTrtScheduler = false;
#ifdef USE_TENSORRT_BACKEND
  useTrtScheduler = !debugSkipNeuralNet;
#endif

  if(useTrtScheduler) {
    delete schedulerState;
    schedulerState = new SchedulerState(this, randSeed + ":NNEvalScheduler");

    // 这里建议改成 stdexec 常驻单线程 context，而不是手搓 thread loop。
    schedulerStdexecScope = std::make_unique<exec::async_scope>();
    schedulerStdexecCtx = std::make_unique<exec::single_thread_context>();

    auto sched = schedulerStdexecCtx->get_scheduler();
    schedulerStdexecScope->spawn(ex::starts_on(sched, serveTrtSchedulerTask()));
    return;
  }

  // 非 TRT backend 仍可继续走现有 serve(...) 线程池路径。
  spawnLegacyBackendThreadsPseudo();
}

void NNEvaluator::killServerThreads() {
  if(schedulerState != nullptr && schedulerStdexecScope != nullptr) {
    schedulerState->stopRequested.store(true, std::memory_order_release);
    schedulerState->schedulerNotice.notify_all();
    ex::sync_wait(schedulerStdexecScope->on_empty());
    schedulerStdexecScope.reset();
    schedulerStdexecCtx.reset();
  }

  killLegacyBackendThreadsPseudo();
  delete schedulerState;
  schedulerState = nullptr;
}

// ============================================================================
// 17. Notes for actual implementation
// ============================================================================

/*

真正开始写正式代码时，建议按以下顺序落地，迁移风险最低：

第一阶段：只落 evaluator scheduler
--------------------------------
1. 保留 Search 完全不动。
2. 先把现有 serveTrtScheduler() 内部状态改成更接近这里的 SlotState/HandleState 形状。
3. 先让 blocking evaluate() 也走“fill -> claim -> scheduler publish”的统一路径。
4. 确认 TensorRT shared buffer / ComputeHandle / InputBuffers 生命周期稳定。

第二阶段：把 Search 的 NN 边界改成 async
---------------------------------------
1. 只改 searchnnhelpers.cpp，把 initNodeNNOutput() 拆成 blocking/async 两版。
2. SearchThread 的字段先原样搬进 SearchTaskState，不要急着重命名和清理。
3. 保留 playoutDescend 的递归结构，把函数提升成 exec::task<bool>。
4. 引入 SearchContinuationScheduler，让 continuation 优先级和跨 worker 恢复成立。

第三阶段：再谈优化
-------------------
1. 决定是否要把 submitReadyRowsH2D 从“逐行提交”进一步改成“区间合并提交”。
2. 决定是否要把 main evaluator / human evaluator 的 await 从顺序改成 when_all。
3. 决定是否要把 Notice / OneShotEvent 实现成更激进的 lock-free intrusive form。

一定不要一上来同时做下面三件事：
- 改 MCTS 递归形状
- 改 GPU scheduler 状态机
- 改 Search 线程池/生命周期

这样风险会非常高，排错也会失去锚点。

*/
