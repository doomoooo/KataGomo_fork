# TensorRT I/O 与调度设计

## 最终结论

当前阶段把设计固定为下面这一版：

- 使用单线程 dispatcher，统一管理所有 GPU 和所有 infer stream。
- 不再尝试把 `H2D + infer + D2H` 合成一个统一的 `CudaGraph`。
- I/O 主路径使用两次 `cudaMemcpyBatchAsync`：
  - 一次 H2D batch
  - 一次 D2H batch
- 当请求是满 batch，且 preprocess / postprocess 可以直接面向 packed slab 写入 / 读取时，切换到“单次 `cudaMemcpyAsync` fallback”：
  - 一次 H2D slab copy
  - 一次 D2H slab copy
- TensorRT `setTensorAddress()` 在初始化时一次性绑定到固定 device slab，运行期不再改。
- stream 选择使用简单的 earliest-finish-time scheduler，按“基于当前状态和最近历史，哪个 slot 预期最早返回”来选。

这版设计替代之前的“统一 graph 表达异步 I/O pipeline”的方向。

## 为什么定成这版

已经做完的实验给出的趋势比较一致：

- `cudaMemcpyBatchAsync` 作为整批 I/O 路径是稳定的，没有出现灾难性退化。
- 如果 host / device 两侧都能预先按对齐要求排成连续 slab，单次 `cudaMemcpyAsync` 的 host 提交开销最低。
- `CudaGraph` 适合只包 infer 本体，但不适合表达你要的“外部长期存在 stream 上的异步 I/O 提前排队”语义。
- 单线程调度的 CPU 成本很低，足够在请求路径上顺手做一次“最快 slot 估计”。

## 当前测量依据

测试对象固定为 `b18tf`、`batch=8`、`RTX 4090 / device 1`。

### 1. I/O host 提交开销

来自 `test_copy_launch`：

- 输入 H2D：
  - `single_memcpy_async` median `1.211 us`
  - `memcpy_batch_async` median `2.394 us`
- 输出 D2H：
  - `single_memcpy_async` median `1.182 us`
  - `memcpy_batch_async` median `2.104 us`

### 2. packed slab 的对齐税

- 输入 raw `254752 B`，packed `254816 B`，padding `64 B`，税 `0.025%`
- 输出 raw `81728 B`，packed `82208 B`，padding `480 B`，税 `0.587%`

### 3. 纯 infer graph 的 host launch 开销

- `cudaGraphLaunch` median `1.392 us`

如果以后 infer 侧保留 graph，这个数字可以作为调度估算里的固定项。

### 4. dispatcher 额外 CPU 成本

本地 toy benchmark：

- 稳态 `cudaSetDevice()` 切卡 median 约 `0.07 us`
- 扫 `3 GPU x 8 stream = 24` 个候选并做一次 earliest-finish-time 选优，约 `0.035 us / pick`

所以“单线程 dispatcher + 多卡多 stream + 估算最快返回 slot”在 CPU 成本上是完全可接受的。

## 设计约束

### 1. H2D 和 D2H 仍然分两段提交

`cudaMemcpyBatchAsync` 不保证 batch 内部 copy 的相对顺序，所以每个请求必须拆成：

1. H2D
2. infer
3. D2H

顺序由 stream 自己保证。

### 2. `cudaMemcpyBatchAsync` 的属性固定

当前统一使用：

- `srcAccessOrder = cudaMemcpySrcAccessOrderAny`
- `flags = cudaMemcpyFlagPreferOverlapWithCompute`

因此 host 侧 pinned slot 在 `doneEvent` 完成前不能复用。

### 3. 满 batch slab path 的前提

只有在下面条件都成立时，才走单次 `cudaMemcpyAsync` fallback：

- 请求是满 batch
- preprocess 可以直接把输入写进 packed host input slab
- postprocess 可以直接消费 packed host output slab，或者允许按 slab offset 读取
- host slab 和 device slab 的 tensor 排布完全一致
- 所有 tensor 起始地址都做了 `256B` 对齐

### 4. device binding 固定

运行期不改 TensorRT binding 地址。每个 infer slot 持有一套固定 device slab：

- input slab
- output slab
- `IExecutionContext`
- infer stream
- `doneEvent`

### 5. 预热所有 device context

进程启动时先对每张卡做：

```cpp
cudaSetDevice(device);
cudaFree(0);
```

避免首次切卡把 primary context 初始化开销带进请求路径。

## 调度模型

调度对象不是“裸 GPU”，而是 `slot = (device, inferStream, context, fixed slabs)`。

每个 slot 维护：

- `predReadyAtUs`
- `ewmaH2dUs`
- `ewmaInferUs`
- `ewmaD2hUs`
- `ewmaJitterUs`
- `lastPredictionErrorUs`
- `inFlight`
- `doneEvent`

每次来了一个满 batch，就估计每个候选 slot 的预期完成时间：

```text
pred_finish(slot) =
  max(now_us + submit_budget_us, predReadyAtUs)
  + pred_h2d_us
  + pred_infer_us
  + pred_d2h_us
  + uncertainty_penalty_us
```

其中：

- `pred_h2d_us / pred_infer_us / pred_d2h_us` 默认取该 slot 的 EWMA
- `uncertainty_penalty_us = 0.5 * ewmaJitterUs + 0.25 * abs(lastPredictionErrorUs)`
- 满 batch 且 slab 可用时，`pred_h2d_us` / `pred_d2h_us` 采用 single-memcpy 路径的历史值
- 其他情况采用 batch-memcpy 路径的历史值

选择 `pred_finish` 最小的 slot 即可。

这是一个刻意保持简单的 earliest-finish-time scheduler，不引入复杂队列论或全局优化。

## 完整参考代码

下面是一份完整参考代码片段，表达的是当前已经固定下来的结构：

- 单线程 dispatcher
- 多卡多 slot
- 默认两次 `cudaMemcpyBatchAsync`
- 满 batch slab fallback 到单次 `cudaMemcpyAsync`
- 简单 earliest-finish-time 选优

```cpp
#include <cuda_runtime.h>

#include <NvInfer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

namespace trt_dispatch {

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error(msg);
}

void cudaCheck(cudaError_t status, const char* what) {
  if(status != cudaSuccess) {
    fail(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void trtCheck(bool ok, const char* what) {
  if(!ok) {
    fail(std::string(what) + " failed");
  }
}

double nowUs() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

size_t alignUp(size_t x, size_t align) {
  return ((x + align - 1) / align) * align;
}

struct DeviceBuffer {
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) : bytes(bytes) {
    cudaCheck(cudaMalloc(&ptr, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if(ptr != nullptr) {
      (void)cudaFree(ptr);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if(this == &other) {
      return *this;
    }
    if(ptr != nullptr) {
      (void)cudaFree(ptr);
    }
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
    return *this;
  }

  void* ptr = nullptr;
  size_t bytes = 0;
};

struct PinnedBytes {
  PinnedBytes() = default;
  explicit PinnedBytes(size_t bytes) : bytes(bytes) {
    cudaCheck(cudaMallocHost(&ptr, bytes), "cudaMallocHost");
    std::fill_n(static_cast<unsigned char*>(ptr), bytes, 0);
  }
  ~PinnedBytes() {
    if(ptr != nullptr) {
      (void)cudaFreeHost(ptr);
    }
  }

  PinnedBytes(const PinnedBytes&) = delete;
  PinnedBytes& operator=(const PinnedBytes&) = delete;

  PinnedBytes(PinnedBytes&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
    other.ptr = nullptr;
    other.bytes = 0;
  }

  PinnedBytes& operator=(PinnedBytes&& other) noexcept {
    if(this == &other) {
      return *this;
    }
    if(ptr != nullptr) {
      (void)cudaFreeHost(ptr);
    }
    ptr = other.ptr;
    bytes = other.bytes;
    other.ptr = nullptr;
    other.bytes = 0;
    return *this;
  }

  void* ptr = nullptr;
  size_t bytes = 0;
};

struct CudaStream {
  CudaStream() = default;
  explicit CudaStream(const char* what) {
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), what);
  }
  ~CudaStream() {
    if(stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = delete;
  CudaStream& operator=(CudaStream&&) = delete;

  cudaStream_t stream = nullptr;
};

struct CudaEvent {
  CudaEvent() = default;
  explicit CudaEvent(const char* what) {
    cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), what);
  }
  ~CudaEvent() {
    if(event != nullptr) {
      (void)cudaEventDestroy(event);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&&) = delete;
  CudaEvent& operator=(CudaEvent&&) = delete;

  cudaEvent_t event = nullptr;
};

struct TensorSpec {
  std::string name;
  bool isInput = false;
  Dims dims{};
  size_t rowBytes = 0;
  size_t maxBytes = 0;
};

struct SlabLayoutEntry {
  std::string name;
  size_t offset = 0;
  size_t bytes = 0;
};

struct SlabLayout {
  std::vector<SlabLayoutEntry> entries;
  size_t packedBytes = 0;

  size_t offsetOf(const std::string& name) const {
    for(const SlabLayoutEntry& e : entries) {
      if(e.name == name) {
        return e.offset;
      }
    }
    fail("slab entry not found: " + name);
  }
};

SlabLayout buildLayout(const std::vector<TensorSpec>& specs) {
  SlabLayout layout;
  size_t cursor = 0;
  for(const TensorSpec& spec : specs) {
    cursor = alignUp(cursor, 256);
    layout.entries.push_back(SlabLayoutEntry{spec.name, cursor, spec.maxBytes});
    cursor += spec.maxBytes;
  }
  layout.packedBytes = alignUp(cursor, 256);
  return layout;
}

struct BatchSubmitter {
  cudaMemcpyAttributes attr{};
  std::array<size_t, 1> attrStarts{0};
  std::vector<void*> dsts;
  std::vector<void*> srcs;
  std::vector<size_t> sizes;

  BatchSubmitter() {
    attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderAny;
    attr.srcLocHint = {cudaMemLocationTypeInvalid, 0};
    attr.dstLocHint = {cudaMemLocationTypeInvalid, 0};
    attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
  }

  void submit(cudaStream_t stream) {
    size_t failIdx = SIZE_MAX;
    cudaCheck(
      cudaMemcpyBatchAsync(
        dsts.data(),
        srcs.data(),
        sizes.data(),
        sizes.size(),
        &attr,
        attrStarts.data(),
        attrStarts.size(),
        &failIdx,
        stream
      ),
      "cudaMemcpyBatchAsync"
    );
  }
};

enum class IoPath {
  BatchMemcpy,
  SingleSlabMemcpy,
};

struct RequestBuffers {
  void* inputSpatial = nullptr;
  void* inputGlobal = nullptr;
  void* outPolicy = nullptr;
  void* outValue = nullptr;
  void* outMiscValue = nullptr;
  void* outMoreMiscValue = nullptr;
  void* outOwnership = nullptr;
  int batchSize = 0;
  bool fullBatchPackedSlabAvailable = false;
};

struct SlotTelemetry {
  double predReadyAtUs = 0.0;
  double ewmaH2dBatchUs = 2.4;
  double ewmaD2hBatchUs = 2.1;
  double ewmaH2dSlabUs = 1.2;
  double ewmaD2hSlabUs = 1.2;
  double ewmaInferUs = 1500.0;
  double ewmaJitterUs = 0.0;
  double lastPredictionErrorUs = 0.0;
};

struct Slot {
  int device = -1;
  int slotIndex = -1;
  std::unique_ptr<CudaStream> inferStream;
  std::unique_ptr<CudaEvent> doneEvent;
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> exec;

  DeviceBuffer inputSlab;
  DeviceBuffer outputSlab;
  SlabLayout inputLayout;
  SlabLayout outputLayout;

  PinnedBytes hostInputSlab;
  PinnedBytes hostOutputSlab;

  bool inFlight = false;
  IoPath lastIoPath = IoPath::BatchMemcpy;
  double lastPredFinishUs = 0.0;
  double lastLaunchUs = 0.0;
  SlotTelemetry telemetry;
};

struct Candidate {
  Slot* slot = nullptr;
  IoPath ioPath = IoPath::BatchMemcpy;
  double predFinishUs = std::numeric_limits<double>::infinity();
};

double ewmaUpdate(double oldValue, double sample, double alpha = 0.2) {
  return oldValue * (1.0 - alpha) + sample * alpha;
}

double estimateFinishUs(const Slot& slot, IoPath ioPath, double nowUsValue, double submitBudgetUs) {
  const SlotTelemetry& t = slot.telemetry;
  const double h2dUs = ioPath == IoPath::SingleSlabMemcpy ? t.ewmaH2dSlabUs : t.ewmaH2dBatchUs;
  const double d2hUs = ioPath == IoPath::SingleSlabMemcpy ? t.ewmaD2hSlabUs : t.ewmaD2hBatchUs;
  const double uncertaintyUs = 0.5 * t.ewmaJitterUs + 0.25 * std::abs(t.lastPredictionErrorUs);
  const double startUs = std::max(nowUsValue + submitBudgetUs, t.predReadyAtUs);
  return startUs + h2dUs + t.ewmaInferUs + d2hUs + uncertaintyUs;
}

IoPath chooseIoPath(const RequestBuffers& req, int maxBatchSize) {
  if(req.batchSize == maxBatchSize && req.fullBatchPackedSlabAvailable) {
    return IoPath::SingleSlabMemcpy;
  }
  return IoPath::BatchMemcpy;
}

Candidate pickBestSlot(std::vector<Slot>& slots, const RequestBuffers& req, int maxBatchSize) {
  const double tNow = nowUs();
  constexpr double kSubmitBudgetUs = 0.5;
  Candidate best;
  for(Slot& slot : slots) {
    const IoPath ioPath = chooseIoPath(req, maxBatchSize);
    const double predFinish = estimateFinishUs(slot, ioPath, tNow, kSubmitBudgetUs);
    if(predFinish < best.predFinishUs) {
      best.slot = &slot;
      best.ioPath = ioPath;
      best.predFinishUs = predFinish;
    }
  }
  if(best.slot == nullptr) {
    fail("no slot available");
  }
  return best;
}

void bindTensorAddressesOnce(Slot& slot) {
  auto inputBase = static_cast<char*>(slot.inputSlab.ptr);
  auto outputBase = static_cast<char*>(slot.outputSlab.ptr);

  trtCheck(slot.exec->setTensorAddress("input_spatial", inputBase + slot.inputLayout.offsetOf("input_spatial")), "setTensorAddress(input_spatial)");
  trtCheck(slot.exec->setTensorAddress("input_global", inputBase + slot.inputLayout.offsetOf("input_global")), "setTensorAddress(input_global)");
  trtCheck(slot.exec->setTensorAddress("out_policy", outputBase + slot.outputLayout.offsetOf("out_policy")), "setTensorAddress(out_policy)");
  trtCheck(slot.exec->setTensorAddress("out_value", outputBase + slot.outputLayout.offsetOf("out_value")), "setTensorAddress(out_value)");
  trtCheck(slot.exec->setTensorAddress("out_miscvalue", outputBase + slot.outputLayout.offsetOf("out_miscvalue")), "setTensorAddress(out_miscvalue)");
  trtCheck(slot.exec->setTensorAddress("out_moremiscvalue", outputBase + slot.outputLayout.offsetOf("out_moremiscvalue")), "setTensorAddress(out_moremiscvalue)");
  trtCheck(slot.exec->setTensorAddress("out_ownership", outputBase + slot.outputLayout.offsetOf("out_ownership")), "setTensorAddress(out_ownership)");
}

void setShapes(IExecutionContext& exec, int batchSize) {
  Dims inputSpatial{};
  inputSpatial.nbDims = 4;
  inputSpatial.d[0] = batchSize;
  inputSpatial.d[1] = 22;
  inputSpatial.d[2] = 19;
  inputSpatial.d[3] = 19;

  Dims inputGlobal{};
  inputGlobal.nbDims = 2;
  inputGlobal.d[0] = batchSize;
  inputGlobal.d[1] = 19;

  trtCheck(exec.setInputShape("input_spatial", inputSpatial), "setInputShape(input_spatial)");
  trtCheck(exec.setInputShape("input_global", inputGlobal), "setInputShape(input_global)");
}

void submitBatchH2D(const RequestBuffers& req, Slot& slot, int batchSize) {
  BatchSubmitter submitter;
  auto inputBase = static_cast<char*>(slot.inputSlab.ptr);

  submitter.dsts = {
    inputBase + slot.inputLayout.offsetOf("input_spatial"),
    inputBase + slot.inputLayout.offsetOf("input_global"),
  };
  submitter.srcs = {req.inputSpatial, req.inputGlobal};
  submitter.sizes = {
    static_cast<size_t>(batchSize) * 22 * 19 * 19 * sizeof(float),
    static_cast<size_t>(batchSize) * 19 * sizeof(float),
  };
  submitter.submit(slot.inferStream->stream);
}

void submitBatchD2H(const RequestBuffers& req, Slot& slot, int batchSize) {
  BatchSubmitter submitter;
  auto outputBase = static_cast<char*>(slot.outputSlab.ptr);

  submitter.dsts = {
    req.outPolicy,
    req.outValue,
    req.outMiscValue,
    req.outMoreMiscValue,
    req.outOwnership,
  };
  submitter.srcs = {
    outputBase + slot.outputLayout.offsetOf("out_policy"),
    outputBase + slot.outputLayout.offsetOf("out_value"),
    outputBase + slot.outputLayout.offsetOf("out_miscvalue"),
    outputBase + slot.outputLayout.offsetOf("out_moremiscvalue"),
    outputBase + slot.outputLayout.offsetOf("out_ownership"),
  };
  submitter.sizes = {
    static_cast<size_t>(batchSize) * 6 * 362 * sizeof(float),
    static_cast<size_t>(batchSize) * 3 * sizeof(float),
    static_cast<size_t>(batchSize) * 10 * sizeof(float),
    static_cast<size_t>(batchSize) * 8 * sizeof(float),
    static_cast<size_t>(batchSize) * 19 * 19 * sizeof(float),
  };
  submitter.submit(slot.inferStream->stream);
}

void submitSlabH2D(Slot& slot) {
  cudaCheck(
    cudaMemcpyAsync(
      slot.inputSlab.ptr,
      slot.hostInputSlab.ptr,
      slot.inputLayout.packedBytes,
      cudaMemcpyHostToDevice,
      slot.inferStream->stream
    ),
    "cudaMemcpyAsync(input slab)"
  );
}

void submitSlabD2H(Slot& slot) {
  cudaCheck(
    cudaMemcpyAsync(
      slot.hostOutputSlab.ptr,
      slot.outputSlab.ptr,
      slot.outputLayout.packedBytes,
      cudaMemcpyDeviceToHost,
      slot.inferStream->stream
    ),
    "cudaMemcpyAsync(output slab)"
  );
}

void maybeReapSlot(Slot& slot) {
  if(!slot.inFlight) {
    return;
  }
  const cudaError_t q = cudaEventQuery(slot.doneEvent->event);
  if(q == cudaSuccess) {
    const double finishedUs = nowUs();
    const double observedTotalUs = finishedUs - slot.lastLaunchUs;
    const double predictedTotalUs = slot.lastPredFinishUs - slot.lastLaunchUs;
    const double errorUs = observedTotalUs - predictedTotalUs;
    slot.telemetry.lastPredictionErrorUs = errorUs;
    slot.telemetry.ewmaJitterUs = ewmaUpdate(slot.telemetry.ewmaJitterUs, std::abs(errorUs));
    slot.telemetry.predReadyAtUs = finishedUs;
    slot.inFlight = false;
    return;
  }
  if(q != cudaErrorNotReady) {
    cudaCheck(q, "cudaEventQuery(doneEvent)");
  }
}

void launchRequest(const RequestBuffers& req, std::vector<Slot>& slots, int maxBatchSize) {
  for(Slot& slot : slots) {
    maybeReapSlot(slot);
  }

  Candidate chosen = pickBestSlot(slots, req, maxBatchSize);
  Slot& slot = *chosen.slot;

  cudaCheck(cudaSetDevice(slot.device), "cudaSetDevice");

  setShapes(*slot.exec, req.batchSize);

  const double launchUs = nowUs();
  if(chosen.ioPath == IoPath::SingleSlabMemcpy) {
    submitSlabH2D(slot);
  }
  else {
    submitBatchH2D(req, slot, req.batchSize);
  }

  if(!slot.exec->enqueueV3(slot.inferStream->stream)) {
    fail("enqueueV3 failed");
  }

  if(chosen.ioPath == IoPath::SingleSlabMemcpy) {
    submitSlabD2H(slot);
  }
  else {
    submitBatchD2H(req, slot, req.batchSize);
  }

  cudaCheck(cudaEventRecord(slot.doneEvent->event, slot.inferStream->stream), "cudaEventRecord(doneEvent)");

  slot.inFlight = true;
  slot.lastIoPath = chosen.ioPath;
  slot.lastLaunchUs = launchUs;
  slot.lastPredFinishUs = chosen.predFinishUs;
  slot.telemetry.predReadyAtUs = chosen.predFinishUs;
}

void prewarmDevices(const std::vector<int>& devices) {
  for(int device : devices) {
    cudaCheck(cudaSetDevice(device), "cudaSetDevice(prewarm)");
    cudaCheck(cudaFree(0), "cudaFree(0)");
  }
}

}  // namespace trt_dispatch
```

## 运行时语义

这版设计下，每次请求的时间线是：

1. dispatcher 扫所有 slot，估计哪个 slot 最早返回
2. `cudaSetDevice(slot.device)`
3. 在选中的 slot 上提交 H2D
4. `enqueueV3(slot.inferStream)`
5. 提交 D2H
6. `cudaEventRecord(slot.doneEvent)`
7. 更新该 slot 的 `predReadyAtUs`

如果 slot 上还有旧请求未完成，不会同步阻塞当前线程；只会通过：

- `predReadyAtUs`
- `cudaEventQuery(doneEvent)`

来进行非阻塞地状态更新和预测。

## 这版相对旧方案的变化

相对之前的文档，这里有四个关键变化：

1. 删掉“统一 graph 表达 H2D + infer + D2H”的设计。
2. 明确把 dispatcher 固定为单线程。
3. I/O 主路径固定为 `cudaMemcpyBatchAsync`，满 batch slab path 走单次 `cudaMemcpyAsync` fallback。
4. 增加一个简单 earliest-finish-time scheduler，而不是只做静态 GPU 轮转或 busy-claim。

## 落地建议

落地顺序建议如下：

1. 先把当前 TensorRT backend 的 H2D / D2H 改成两次 `cudaMemcpyBatchAsync`
2. 把 device buffer 改成固定 slab，并在初始化时一次性 `setTensorAddress()`
3. 让 preprocess 支持直接写 packed host input slab
4. 让 postprocess 支持直接读 packed host output slab
5. 在 dispatcher 里加入这个 earliest-finish-time 打分
6. 最后再决定 infer 是否继续保留纯 infer graph

当前阶段不再建议沿着“graph 统一表达外部异步 I/O pipeline”的方向继续投入。
