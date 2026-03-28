# NEW_TRT_BACKEND + stdexec 落地计划

## 1. 目标与约束

本计划对应 [exp/trt_design.md](/home/wangyize/.katago/KataGomo_fork/exp/trt_design.md) 的最终设计，且不修改设计本身，只解决“如何在现有 KataGo 代码结构里落地”。

新增约束如下：

- `NEW_TRT_BACKEND` 不是编译期开关，而是运行时配置项。
- 打开 `NEW_TRT_BACKEND` 后：
  - 只支持 ONNX 模型。
  - 不支持 humanSL。
  - TensorRT 路径切到 `stdexec` 驱动的单线程 dispatcher 实现。
- 关闭 `NEW_TRT_BACKEND` 后：
  - 维持当前 TensorRT 行为不变。
  - 其他 backend 行为不变。

本计划默认：

- `stdexec` 作为控制面抽象，目标是让“等待输入 ready / 等待 CUDA event / 释放 host slot”这些阶段以线性协程表达，而不是继续堆回调或大 while 状态机。
- 不把 `stdexec` 扩散到搜索层或其他 backend，只包住 TensorRT 新实现。
- 为了使用 `exec::task` 和 coroutine，TensorRT 构建路径需要升级到 C++20。

## 2. 当前代码的关键断点

当前 TensorRT 路径是下面这条链：

1. [cpp/program/setup.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/program/setup.cpp) 解析 `TRTConfigs`，创建 [NNEvaluator](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.h)。
2. [NNEvaluator::spawnServerThreads()](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L402) 为每个 `gpuIdxByServerThread` 起一个 server thread。
3. [NNEvaluator::serve()](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L452) 从 `queryQueue` 拉批。
4. [NeuralNet::getOutput()](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp#L2161) 在一个线程里同步做 preprocess/H2D/infer/D2H/postprocess。

而新设计要求：

- 不再由“每个 slot 一个 OS thread”拉批。
- 改成“单 dispatcher 线程 + host ring + `slot -> ping/pong bank`”。
- 对外不是批接口，而是单请求 handle + 三个事件。

因此，`NEW_TRT_BACKEND=true` 本质上不是重写一两个函数，而是给 TensorRT backend 增加一条新运行时实现路径。

## 3. 实施总原则

1. 只新增，不污染旧路径。
2. 通用 NN 接口尽量少改；新 TensorRT 内部接口可以单独加。
3. 所有 ONNX 特化都写成显式类型，不再保留 `.bin.gz` / `InputMask` / `InputMeta` 双轨兼容分支。
4. `stdexec` 只负责控制流可读性，不引入额外线程池，也不违反“单线程 dispatcher”设计。
5. humanSL 在“配置阶段”和“运行阶段”都双重拒绝，确保不是 warning，而是直接失败。

## 4. 构建与配置改动

### 4.1 C++20 与 stdexec

文件：

- [cpp/CMakeLists.txt](/home/wangyize/.katago/KataGomo_fork/cpp/CMakeLists.txt)

需要做的改动：

1. 将 `katago` 目标的 C++ 标准从 17 升到 20。
2. 把 `third_party/stdexec/include` 加入 `katago` 的 include path。
3. 初版只使用 header-only 组件：
   - `exec/single_thread_context.hpp`
   - `exec/async_scope.hpp`
   - `exec/task.hpp`
   - `stdexec/execution.hpp`
4. 不依赖 `STDEXEC::stdexec` target 也能先跑通，因为本次只用 header-only 组件；如果后续需要 `system_context` / `nvexec`，再追加 `add_subdirectory(../third_party/stdexec ...)`。

### 4.2 新配置项

文件：

- [cpp/neuralnet/nninterface.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h)
- [cpp/program/setup.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/program/setup.cpp)

在 `TRTConfigs` 中新增：

```cpp
enum class TrtDispatcherWaitPolicy {
  Spin,
  Park,
};

struct TRTConfigs {
  bool trtUseCudaGraph = false;
  CudaSyncMode trtCudaSyncMode = CudaSyncMode::Blocking;
  int trtBuilderOptimizationLevel = -1;
  int trtMaxAuxStreams = -1;
  int trtAvgTimingIterations = -1;
  TrtTilingOptimizationLevel trtTilingOptimizationLevel = TrtTilingOptimizationLevel::None;

  bool newTrtBackend = false;  // 对应配置项 NEW_TRT_BACKEND
  TrtDispatcherWaitPolicy trtDispatcherWaitPolicy = TrtDispatcherWaitPolicy::Park;
};
```

配置解析规则：

- `NEW_TRT_BACKEND[-idx]`: `bool`
- `trtDispatcherWaitPolicy[-idx]`: `spin | park`

说明：

- `NEW_TRT_BACKEND` 用大写做配置 key，内部字段名仍用 `newTrtBackend`。
- 这是启动时配置，不支持热切换；`NNEvaluator` 构造完成后不再改变。

### 4.3 配置期拒绝规则

文件：

- [cpp/program/setup.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/program/setup.cpp)

新增校验：

1. 若 `NEW_TRT_BACKEND=true` 且主模型文件不是 `.onnx`，`ConfigParsingError` 直接失败。
2. 若 `NEW_TRT_BACKEND=true` 且提供了 `-human-model`，直接失败。
3. 若 `NEW_TRT_BACKEND=true` 且任何 `humanSL*` 参数被显式设置，直接失败。
4. 若 `NEW_TRT_BACKEND=true` 且 `humanSLProfile` 被设置，直接失败。

这样可以在 setup 阶段就把“不支持 humanSL”的语义锁死，而不是等跑到 `evaluate()` 再晚报错。

## 5. 新的内部模块与文件划分

### 5.1 新增文件

新增：

- [cpp/neuralnet/trtbackend.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.h)

职责：

- 只声明 `NEW_TRT_BACKEND` 所需的 TensorRT 内部类型和接口。
- 不暴露给其他 backend。
- 保持 [cpp/neuralnet/nninterface.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h) 的 generic API 尽量稳定。

### 5.2 trtbackend.h 中要定义的类型

#### 5.2.1 事件与 handle

```cpp
namespace TrtBackendV2 {

struct EventState {
  std::mutex mutex;
  std::condition_variable cv;
  uint64_t generation = 0;
  bool signaled = false;
};

class OneShotGenerationEvent {
public:
  OneShotGenerationEvent();
  explicit OneShotGenerationEvent(EventState* state, uint64_t generation);

  void set() const;
  void wait() const;
  bool isSet() const;

private:
  EventState* state_;
  uint64_t generation_;
};

struct OnnxInputRowView {
  float* spatial = nullptr;
  float* global = nullptr;
  size_t spatialElts = 0;
  size_t globalElts = 0;
};

struct OnnxOutputRowView {
  float* policy = nullptr;
  float* value = nullptr;
  float* miscvalue = nullptr;
  float* moremiscvalue = nullptr;
  float* ownership = nullptr;
  size_t policyElts = 0;
  size_t valueElts = 0;
  size_t miscvalueElts = 0;
  size_t moremiscvalueElts = 0;
  size_t ownershipElts = 0;
};

struct SingleRequestHandle {
  void* inputs_mem_addr = nullptr;   // 实际指向 OnnxInputRowView
  void* outputs_mem_addr = nullptr;  // 实际指向 OnnxOutputRowView
  OneShotGenerationEvent input_mem_ready;
  OneShotGenerationEvent output_mem_ready;
  OneShotGenerationEvent output_mem_consumed;
  uint32_t hostSlotIndex = 0;
  uint16_t rowIndex = 0;
  uint64_t generation = 0;
};

}
```

设计决策：

- `inputs_mem_addr` / `outputs_mem_addr` 保持文档要求的 `void*` 语义。
- 但新实现内部直接把它们解释为 `OnnxInputRowView*` / `OnnxOutputRowView*`，避免 generic tensor map 带来的心智负担。

#### 5.2.2 布局描述

```cpp
namespace TrtBackendV2 {

struct TensorLayoutEntry {
  std::string name;
  size_t rowBytes = 0;
  size_t batchBytes = 0;
  size_t slabOffset = 0;
};

struct PackedLayout {
  TensorLayoutEntry inputSpatial;
  TensorLayoutEntry inputGlobal;
  TensorLayoutEntry outPolicy;
  TensorLayoutEntry outValue;
  TensorLayoutEntry outMiscvalue;
  TensorLayoutEntry outMoremiscvalue;
  TensorLayoutEntry outOwnership;

  size_t inputSlabBytes = 0;
  size_t outputSlabBytes = 0;
  size_t perHostSlotBytes = 0;
};

}
```

说明：

- 因为 `NEW_TRT_BACKEND` 只支持 ONNX，所以 layout 直接固定成 2 个输入 + 5 个输出。
- 这样 `cudaMemcpyBatchAsync` 的参数表、row view 指针、slab fast path 判断都能写成固定字段，不需要字符串查表。

#### 5.2.3 Host ring

```cpp
namespace TrtBackendV2 {

enum class HostSlotState {
  Empty,
  Filling,
  WaitingAllInputsReady,
  Dispatched,
  OutputsReady,
};

struct HostRow {
  OnnxInputRowView inputView;
  OnnxOutputRowView outputView;
  EventState inputReady;
  EventState outputReady;
  EventState outputConsumed;
  bool reserved = false;
};

struct HostSlot {
  void* allocation = nullptr;  // cudaMallocHost 出来的整块 pinned memory
  char* inputSlabBase = nullptr;
  char* outputSlabBase = nullptr;
  std::vector<HostRow> rows;

  uint64_t generation = 0;
  HostSlotState state = HostSlotState::Empty;
  int assignedRows = 0;
  int inputReadyRows = 0;
  int outputReadyRows = 0;
  int outputConsumedRows = 0;
  bool partialFlushQueued = false;
};

}
```

#### 5.2.4 Device 侧对象

```cpp
namespace TrtBackendV2 {

enum class BankState {
  Idle,
  H2DInFlight,
  H2DDoneInferQueued,
  InferDoneD2HInFlight,
  HostVisible,
};

struct EngineBundle {
  int deviceId = -1;
  bool usingFP16 = false;
  int modelVersion = -1;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  PackedLayout layout;
};

struct DeviceBank {
  int bankIndex = 0;
  BankState state = BankState::Idle;
  int assignedHostSlotIndex = -1;
  uint64_t assignedGeneration = 0;
  int assignedRows = 0;
  bool useSlabPath = false;

  void* inputSlab = nullptr;
  void* outputSlab = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> exec;
  cudaStream_t ioStream = nullptr;
  cudaEvent_t h2dDoneEvent = nullptr;
  cudaEvent_t inferDoneEvent = nullptr;
  cudaEvent_t d2hDoneEvent = nullptr;

  std::unordered_map<int, cudaGraphExec_t> graphExecByBatchSize;
};

struct DeviceSlot {
  int deviceId = -1;
  int slotIndex = -1;
  std::shared_ptr<EngineBundle> engine;
  cudaStream_t inferStream = nullptr;
  std::array<DeviceBank,2> banks;

  int nextLaunchBank = 0;
  int activeInferBank = -1;
  int lastCompletedInferBank = -1;

  double predReadyAtUs = 0.0;
  double ewmaH2dBatchUs = 0.0;
  double ewmaH2dSlabUs = 0.0;
  double ewmaInferUs = 0.0;
  double ewmaD2hBatchUs = 0.0;
  double ewmaD2hSlabUs = 0.0;
  double ewmaJitterUs = 0.0;
  double lastPredictionErrorUs = 0.0;
};

}
```

### 5.3 stdexec 驱动层

```cpp
namespace TrtBackendV2 {

class DispatcherWakeSource {
public:
  void notify();
  void requestStop();
  bool stopRequested() const;
  exec::task<void> waitForWake(uint64_t lastSeenEpoch);

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  uint64_t wakeEpoch_ = 0;
  bool stopRequested_ = false;
};

class TrtDispatcherBackend {
public:
  TrtDispatcherBackend(...);
  ~TrtDispatcherBackend();

  void start();
  void stop();
  SingleRequestHandle getOutput();
  void waitForOneCompletionIfAny();
  bool isUsingFP16() const;

private:
  exec::single_thread_context controlContext_;
  exec::async_scope scope_;
  DispatcherWakeSource wakeSource_;

  std::mutex apiMutex_;
  std::condition_variable slotReusableCv_;
  std::condition_variable completionCv_;

  std::vector<HostSlot> hostRing_;
  std::vector<DeviceSlot> slots_;
  std::vector<int> gpuIdxBySlot_;

  size_t fillHostSlotIndex_ = 0;
  uint64_t completionEpoch_ = 0;
  bool started_ = false;
  bool stopping_ = false;

  exec::task<void> dispatcherMain();
  exec::task<void> dispatchHostSlot(size_t hostSlotIndex, uint64_t generation);
  exec::task<void> reclaimHostSlot(size_t hostSlotIndex, uint64_t generation);
  template<class Pred>
  exec::task<void> waitUntil(Pred pred);
};

}
```

这里的关键点：

- `single_thread_context` 精确提供“单 dispatcher 线程”。
- `async_scope` 管理所有 in-flight coroutine，`stop()` 时可以 `request_stop + sync_wait(scope_.on_empty())`，收尾很干净。
- `dispatchHostSlot()` 用线性 coroutine 表达完整生命周期，比手写 500 行 pump/state machine 更清楚。

## 6. 各老接口的具体改动

### 6.1 nninterface.h

改动：

1. `TRTConfigs` 新增 `newTrtBackend` 与 `trtDispatcherWaitPolicy`。
2. 不修改 `NeuralNet::getOutput()` / `createComputeHandle()` / `createInputBuffers()` 签名。
3. 这些 generic API 在 `NEW_TRT_BACKEND=false` 时继续使用，在 `true` 时变成 legacy path only。

这样可以避免所有 backend 一起改签名。

### 6.2 nneval.h

新增字段：

```cpp
#if defined(USE_TENSORRT_BACKEND)
  bool useNewTrtBackend;
  std::unique_ptr<TrtBackendV2::TrtDispatcherBackend> trtDispatcher;
#endif
```

保留但转成 legacy-only 的字段：

- `serverThreads`
- `queryQueue`
- `NNServerBuf`
- `serve(...)`

接口层改动：

- `spawnServerThreads()` 名字不改，但当 `useNewTrtBackend=true` 时，不再启动多个 server thread，而是启动 `trtDispatcher->start()`。
- `killServerThreads()` 名字不改，但新路径里调用 `trtDispatcher->stop()`。
- `waitForNextNNEvalIfAny()` 新路径里转到 `trtDispatcher->waitForOneCompletionIfAny()`。
- `requiresSGFMetadata()` 在新路径下一律返回 `false`，因为 humanSL 被禁止。

### 6.3 nneval.cpp

#### 构造函数

在 [NNEvaluator::NNEvaluator](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp#L47) 中新增：

1. `useNewTrtBackend = trtConfigs.newTrtBackend;`
2. `loadedModel = NeuralNet::loadModelFile(...)` 后做双重校验：
   - 若 `useNewTrtBackend && !desc.onnxHeader.isOnnx`，直接 `throw StringError(...)`。
   - 若 `useNewTrtBackend && desc.numInputMetaChannels > 0`，直接 `throw StringError(...)`。
3. 新路径下不分配 `NNServerBuf::inputBuffers` 风格的 thread-local batch buffer。

#### spawnServerThreads / killServerThreads

新路径下：

1. 不调用 `queryQueue.unsetReadOnly()`。
2. 不起 `numThreads` 个 OS thread。
3. `gpuIdxByServerThread` 的每个 entry 解释为一个 `DeviceSlot`，不是一个 server thread。
4. `serverThreadsIsUsingFP16` 改为从 `trtDispatcher->isUsingFP16()` 填充，长度仍按 `numThreads` 维持，保证旧统计接口不炸。

#### evaluate

`evaluate()` 在 `useNewTrtBackend=true` 时改成：

1. 继续保留现在的 cache 命中逻辑。
2. 调整 symmetry / policyOptimism。
3. 调用 `SingleRequestHandle handle = trtDispatcher->getOutput();`
4. 将 `handle.inputs_mem_addr` 转成 `OnnxInputRowView*`，把这一 row 的输入写进去。
5. `handle.input_mem_ready.set();`
6. `handle.output_mem_ready.wait();`
7. 将 `handle.outputs_mem_addr` 转成 `OnnxOutputRowView*`，做 ONNX 输出解码并填充 `NNOutput`。
8. `handle.output_mem_consumed.set();`

运行时 humanSL 双重拒绝：

- 如果 `sgfMeta != nullptr && sgfMeta->initialized`，直接 `Global::fatalError("NEW_TRT_BACKEND does not support humanSL")`。
- 不再保留当前“`numInputMetaChannels > 0` 时把 `policyOptimism` 清零”的兼容逻辑。

### 6.4 trtbackend.cpp

保留旧实现，但结构上分两段：

```cpp
#ifdef USE_TENSORRT_BACKEND

// 旧路径：if (!context->newTrtBackend) { legacy ComputeHandle/InputBuffers/getOutput }
// 新路径：TrtBackendV2 namespace + stdexec dispatcher

#endif
```

具体原则：

- 旧 `ComputeHandle`、`InputBuffers`、`NeuralNet::getOutput()` 原样保留在 legacy 分支，供 `newTrtBackend=false` 使用。
- 新 `TrtDispatcherBackend` 完全不调用旧 `NeuralNet::getOutput()`。
- 当前 `ComputeHandle` 里的 ONNX plan build / cache / engine deserialize / exec 创建逻辑要抽出给 `EngineBundle` 和 `DeviceBank` 复用。

## 7. 新路径的 stdexec 执行模型

### 7.1 核心思想

不是写一个“无限 while + if + switch + poll 一切”的 dispatcher，而是拆成：

- 一个长期存在的 `dispatcherMain()` coroutine，负责：
  - 唤醒
  - 检查当前 fill host slot 是否需要 full-batch dispatch
  - 检查是否需要 partial flush
  - 为每个新 dispatch 启动 `dispatchHostSlot()`
- 一个 `dispatchHostSlot()` coroutine 表达单个 host slot 的完整 GPU 生命周期：
  - 等所有 `input_mem_ready`
  - 选 slot/bank
  - 提交 H2D
  - 等 `h2d_done_event`
  - 提交 infer
  - 等 `infer_done_event`
  - 提交 D2H
  - 等 `d2h_done_event`
  - fan-out `output_mem_ready`
  - bank 回收
- 一个 `reclaimHostSlot()` coroutine 表达 host slot 的复用生命周期：
  - 等所有 `output_mem_consumed`
  - generation++
  - 状态回到 `Empty`

### 7.2 waitUntil 抽象

为了避免把 `condition_variable + predicate` 写散，统一做：

```cpp
template<class Pred>
exec::task<void> TrtDispatcherBackend::waitUntil(Pred pred) {
  uint64_t epoch = 0;
  while(!pred()) {
    co_await wakeSource_.waitForWake(epoch);
    epoch = ...; // 读新的 wake epoch
  }
}
```

用途：

- 等某个 host slot `inputReadyRows == assignedRows`
- 等 `cudaEventQuery(bank.h2dDoneEvent) == cudaSuccess`
- 等 `cudaEventQuery(bank.inferDoneEvent) == cudaSuccess`
- 等 `cudaEventQuery(bank.d2hDoneEvent) == cudaSuccess`
- 等 `hostSlot.outputConsumedRows == hostSlot.assignedRows`

这样所有“等待某条件成立再继续”的代码都写成线性 coroutine，不再手搓多层状态推进。

### 7.3 wait policy

`DispatcherWakeSource::waitForWake()` 内部根据 `trtDispatcherWaitPolicy` 分两种：

- `Spin`: busy spin + 轻量 `pause/yield`
- `Park`: `condition_variable::wait`

无论哪种模式，`dispatchHostSlot()` / `reclaimHostSlot()` 的逻辑完全不变，只是 `waitForWake()` 的实现不同，符合设计要求。

## 8. Host / Device 内存布局与接口

### 8.1 Host ring 大小

按文档字面实现：

- `ringSlots = 3 * sum(maxBatchSize * slots_per_gpu)`。

这里的 `slots_per_gpu` 直接从 `gpuIdxByServerThread` 统计。

### 8.2 HostSlot 初始化流程

每个 `HostSlot` 初始化时：

1. `cudaMallocHost(perHostSlotBytes)`。
2. 切出 `inputSlabBase`。
3. 切出 `outputSlabBase`。
4. 根据 `PackedLayout` 为每个 row 预先算好：
   - `inputView.spatial/global`
   - `outputView.policy/value/miscvalue/moremiscvalue/ownership`
5. 每个 row 的三个 `EventState.generation` 初始等于 `hostSlot.generation`。

### 8.3 DeviceBank 初始化流程

每个 `DeviceBank` 初始化时：

1. `cudaSetDevice(slot.deviceId)`
2. `cudaMalloc(inputSlab)`
3. `cudaMalloc(outputSlab)`
4. `engine->createExecutionContext()`
5. `cudaStreamCreate(ioStream)`
6. `cudaEventCreate(h2dDoneEvent / inferDoneEvent / d2hDoneEvent)`
7. `exec->setTensorAddress(...)`
8. 如果 `trtUseCudaGraph=true`，预捕获本 bank 的 `graphExecByBatchSize[1..maxBatchSize]`

注意：

- graph 缓存按 `batchSize` 做 map，是内部实现细节。
- 对外语义仍然是“graph 是 bank 私有的，不在运行期改地址”。

## 9. 调度与提交逻辑

### 9.1 slot 选择

新增函数：

```cpp
struct SlotPrediction {
  int slotIndex = -1;
  int bankIndex = -1;
  double predictedFinishUs = 0.0;
  bool useSlabPath = false;
};

SlotPrediction chooseBestSlot(int rows, bool packedSlabEligible);
```

内部使用字段：

- `DeviceSlot::predReadyAtUs`
- `DeviceSlot::ewmaH2dBatchUs`
- `DeviceSlot::ewmaH2dSlabUs`
- `DeviceSlot::ewmaInferUs`
- `DeviceSlot::ewmaD2hBatchUs`
- `DeviceSlot::ewmaD2hSlabUs`
- `DeviceSlot::ewmaJitterUs`
- `DeviceSlot::lastPredictionErrorUs`

### 9.2 dispatchHostSlot() 的线性流程

`dispatchHostSlot(hostSlotIndex, generation)` 的实际步骤固定为：

1. `co_await waitUntil(all_input_ready)`
2. 选中一个 `(slot, bank)`
3. 提交 H2D
4. 记录 `bank.h2dDoneEvent`
5. `co_await waitUntil(h2d_done)`
6. 在 `slot.inferStream` 上提交 infer
7. 记录 `bank.inferDoneEvent`
8. `co_await waitUntil(infer_done)`
9. 在 `bank.ioStream` 上提交 D2H
10. 记录 `bank.d2hDoneEvent`
11. `co_await waitUntil(d2h_done)`
12. 对 host slot 内每个 row 调 `output_mem_ready.set()`
13. bank 状态切回 `Idle`
14. 启动 `reclaimHostSlot(hostSlotIndex, generation)`

### 9.3 H2D / D2H 两条路径

提交函数明确拆成两个 helper：

```cpp
void submitBatchMemcpyH2D(DeviceBank&, HostSlot&, int rows);
void submitBatchMemcpyD2H(DeviceBank&, HostSlot&, int rows);
void submitSlabMemcpyH2D(DeviceBank&, HostSlot&);
void submitSlabMemcpyD2H(DeviceBank&, HostSlot&);
```

选择条件：

- `rows == maxBatchSize`
- `hostSlot` 是 packed 直写
- `DeviceBank` slab 布局与 `HostSlot` 完全同构

否则永远走 `cudaMemcpyBatchAsync`。

## 10. ONNX 专属 pre/postprocess 位置

### 10.1 输入写入

建议保留在 [cpp/neuralnet/nneval.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nneval.cpp)。

原因：

- 它依赖 `Board/BoardHistory/NNInputs`，属于 evaluator 域逻辑，不属于 TensorRT 运行时。

新增 helper：

```cpp
void fillNewTrtOnnxInputRow(
  const NNEvaluator& nnEval,
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  TrtBackendV2::OnnxInputRowView& rowView
);
```

### 10.2 输出解码

建议放到 [cpp/neuralnet/trtbackend.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.cpp)，因为它是 ONNX tensor 语义的一部分。

新增 helper：

```cpp
void decodeNewTrtOnnxOutputRow(
  const TrtBackendV2::OnnxOutputRowView& rowView,
  int modelVersion,
  int nnXLen,
  int nnYLen,
  int symmetry,
  float policyOptimism,
  NNOutput& out
);
```

它直接复用当前 ONNX 后处理逻辑：

- `out_policy`
- `out_value`
- `out_miscvalue`
- `out_moremiscvalue`
- `out_ownership`

但输入源从旧 `InputBuffers` 改为 row view。

## 11. 精确的实施步骤

### Step 0: 构建基础

- 改 [cpp/CMakeLists.txt](/home/wangyize/.katago/KataGomo_fork/cpp/CMakeLists.txt) 到 C++20。
- 把 `third_party/stdexec/include` 接进 TensorRT 构建。
- 编译不改逻辑，先验证空接入不破坏现有 build。

### Step 1: 配置链路

- 改 [cpp/neuralnet/nninterface.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/nninterface.h) 增加新 enum/field。
- 改 [cpp/program/setup.cpp](/home/wangyize/.katago/KataGomo_fork/cpp/program/setup.cpp) 解析 `NEW_TRT_BACKEND` 和 wait policy。
- 把配置打印日志补上。

### Step 2: setup 阶段拒绝规则

- 在 `setup.cpp` 里对 `.onnx` / `humanSL*` / `-human-model` 做直接失败。
- `maybeWarnHumanSLParams()` 对新路径不再 warning，直接不允许进入。

### Step 3: 新头文件与纯数据结构

- 新建 [cpp/neuralnet/trtbackend.h](/home/wangyize/.katago/KataGomo_fork/cpp/neuralnet/trtbackend.h)。
- 只先声明 `SingleRequestHandle`、row view、layout、host/device structs、dispatcher 类，不接 TensorRT 实现。

### Step 4: 抽取 EngineBundle

- 从旧 `ComputeHandle` 抽出 ONNX plan/cache/build/deser 逻辑。
- 实现“每 device 一份 engine，每 bank 一份 context”的初始化函数。
- 此步先不接 dispatcher。

### Step 5: Host ring 与固定地址 binding

- 实现 `PackedLayout` builder。
- 实现 `HostSlot` pinned 分配与 row view 计算。
- 实现 `DeviceBank` 固定 slab 分配与 `setTensorAddress()`。

### Step 6: stdexec dispatcher 骨架

- 实现 `DispatcherWakeSource`。
- 实现 `TrtDispatcherBackend::start/stop/getOutput()`。
- `dispatcherMain()` 先只支持 full-batch dispatch，不做 partial flush。

### Step 7: 单 host slot 生命周期跑通

- 实现 `dispatchHostSlot()`：
  - 等 `input_mem_ready`
  - H2D
  - infer
  - D2H
  - `output_mem_ready`
- 先只用 `enqueueV3()`，不启 graph。

### Step 8: 接入 NNEvaluator

- `NNEvaluator::spawnServerThreads/killServerThreads/evaluate/waitForNextNNEvalIfAny` 分支到新 dispatcher。
- cache / symmetry / stats 逻辑保留在 `NNEvaluator`。

### Step 9: 双 bank + 共享 infer stream

- DeviceSlot 改成 `ping/pong`。
- 加入 `h2d_done_event / infer_done_event / d2h_done_event`。
- infer 改成在 `slot.inferStream` 排队，不在 bank 自己 stream 上跑。

### Step 10: partial flush + earliest-finish scheduler

- 加入 `chooseBestSlot()`。
- `dispatcherMain()` 补 partial flush。
- `reclaimHostSlot()` 正式接管 host slot 复用。

### Step 11: graph

- 按 bank + batchSize 预捕获 `graphExecByBatchSize`。
- `trtUseCudaGraph` 开时：
  - 满足 graph 条件的 batch 用 `cudaGraphLaunch`
  - 否则 fallback `enqueueV3()`

### Step 12: 清理 legacy 交叉引用

- `NNServerBuf`、`queryQueue`、`serve()` 保留，但明确标为 legacy-only。
- `trtbackend.cpp` 中旧 `InputBuffers` / `getOutput()` 不再被新路径引用。

## 12. 测试与验收

### 12.1 配置级测试

- `NEW_TRT_BACKEND=false`：现有 TRT 行为完全不变。
- `NEW_TRT_BACKEND=true` + `.bin.gz`：启动即失败。
- `NEW_TRT_BACKEND=true` + `humanSLProfile`：启动即失败。
- `NEW_TRT_BACKEND=true` + `-human-model`：启动即失败。

### 12.2 功能测试

- 单 GPU / 单 slot / batch=1
- 单 GPU / 多 slot / partial flush
- 多 GPU / 多 slot / full-batch slab path
- 开/关 `trtUseCudaGraph`

### 12.3 正确性测试

- 同一 ONNX 模型，对比旧 TensorRT 路径和新路径输出 logits。
- 验证 symmetry、policy optimism、ownership 解码完全一致。

### 12.4 资源回收测试

- 重复 `spawnServerThreads()` / `killServerThreads()`
- 验证：
  - host pinned ring 释放
  - device slab 释放
  - stream/event 释放
  - graphExec 释放
  - `scope_.on_empty()` 正常返回

## 13. 最终建议

真正让 `stdexec` 发挥作用的地方，不是把每个 CUDA API 都包装成 sender，而是：

- 用 `single_thread_context` 精确表达“单 dispatcher 线程”。
- 用 `async_scope` 管 in-flight host slot 生命周期。
- 用 `task<void>` 把 `input_ready -> h2d -> infer -> d2h -> outputs_ready -> outputs_consumed` 写成线性流程。

这样既严格符合 `trt_design.md`，也能把现在最难读的多阶段异步状态推进，压缩成少数几个职责清楚的 coroutine。
