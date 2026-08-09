# KataGo Plan 驱动 CUDA Fork

中文 | [English](README.md)

本仓库从官方 [`lightvector/KataGo`](https://github.com/lightvector/KataGo)
的 `master` 提交
`6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19`（`v1.17.2`，获取于
2026-08-07）重新分叉，在保留 KataGo 的 GTP、分析、搜索、模型和围棋逻辑
的基础上，增加了面向 NVIDIA SM89 和 SM120 的 shape-specialized、Plan
驱动 CUDA 推理系统。

本项目只优化 CUDA backend。优化路径不需要也不使用 TensorRT。本分支的增量
不是官方 KataGo 的组成部分，也不代表受到上游官方支持。

## 相对于官方 fork 点的主要增量

| 方面 | 官方 fork 点 | 本分支 |
| --- | --- | --- |
| Kernel 选择 | backend 默认值和库内部启发式 | 离线整图扫描器生成、显式且可版本化的 tactic plan |
| Batch shape | 当前请求数可能直接进入推理 | 固定物理 batch；满 batch 立即启动，GPU 完全空闲时才允许不足量启动并 padding |
| Host 提交 | evaluator 线程可能串行经历提交和完成 | 每条推理 lane 一个常驻 host worker，并发、非阻塞提交 |
| 数据搬运 | preprocess、H2D、计算、D2H、postprocess 更多处于同一关键路径 | pinned staging、独立上传/下载流和 CUDA event |
| Buffer 复用 | 通常等待完成后再复用 | 单个 device slot，由 input-consumed/output-consumed event 防止覆盖；不需要 ping-pong buffer |
| CUDA Stream | 部分历史自定义路径使用隐式或 per-thread stream | 所有优化 kernel 都显式使用 NNServer 所拥有的 stream |
| 多卡 | evaluator 线程映射到设备 | stream、event、cuBLAS handle、buffer 和 idle 状态全部按接收设备隔离并 fail-closed |
| CUDA Graph | 没有 plan 层的精确 shape 契约 | SM89 可选精确 shape replay；外部 ready/consumed event 保持在 capture 之外 |
| 正确性 | KataGo 常规 backend 测试 | 不可变的 8192-row 全 FP32 门、输入同一性检查、精确 batch 尾部检查、GTP 形态长测 |
| 分发 | 常规源码/编译流程 | 源码完备的离线 autotune tar，以及独立的非侵入式预编译 runtime tar |

目标不是为某个 batch 堆专用特例。扫描器在每一个 B4-B32 精确 batch 上都搜索
同一套完整 tactic family。维护集合是 SM89 与 SM120 所有历史上产生过真实正
收益且数值有效实现的并集。

## 当前认证状态

| 架构 | 已实现搜索空间 | Production plan | 硬件认证 |
| --- | --- | --- | --- |
| SM89 | 20 个 family、62 条正收益历史记录、精确 B4-B32 共 3738 个候选 | 已提交 RTX 4090 D B12/S2 plan | 已完成 GTP 加载、单/双卡和 8192-row 全输出 FP32 replay |
| SM120 | 23 个 family、64 条正收益历史记录、精确 B4-B32 共 4234 个候选 | 等待新的统一扫描 | backend/静态闭环已完成；production 性能和 GTP 认证尚未完成 |

此前 RTX 5080 的 B18 `2586.579` physical nnEval/s 来自尚未合并完整历史
优化集合的旧扫描。该旧 plan 会被 production loader 拒绝，也不会作为可发布
plan 打包。

当前纳入 Git 的 SM89 plan 为：

```text
final-migration/plans/sm89/rtx4090d-b12-s2/best-tactic-plan.json
```

文件 SHA-256 为
`57aba0d9f5ff009f0103fe792766bd3fe065d156c13396cb99bc40b5488f9edb`。
整图 long gate 为 `3026.196859` physical nnEval/s。随后通过普通 evaluator
调度路径验证：单张 RTX 4090 D 上 8192/8192 正确且达到 3035.87 physical
nnEval/s；两张 RTX 4090 D 上 8192/8192 正确且达到 6072.97 physical
nnEval/s。这些数字只描述被测主机、频率、模型、batch 和拓扑，不是对所有机器
的普适性能承诺。

## Plan 驱动 backend

autotuner 输出 schema 1 的 `cuda-tactic-plan` JSON。一个 production plan
完整绑定：

- 架构以及接收端硬件能力约束；
- 精确 19x19、FP16/NHWC、模型 SHA-256、batch 和每设备 stream 数；
- 每个选中 family 的自包含 override；
- 源码、生成物、配置和二进制 hash；
- discovery 与整图 long gate 结果；
- 正收益历史闭环；
- 最优 plan 对应的全 FP32 正确性证书。

`cpp/neuralnet/cudatacticplan.cpp` 在 evaluator 创建前加载 plan。schema、
完成状态、历史闭环、模型、棋盘、精度、架构、batch、stream 拓扑或接收端能力
任意一项不匹配都会在启动时失败。某个 tactic 不受支持时不会偷偷 fallback 到
官方 kernel 后仍声称 plan 已经激活。

backend 只有在具体实现实际 launch 成功之后才发出 activation marker。因此
每个扫描候选必须同时具有四条链路：

1. backend 实现；
2. 扫描候选；
3. 实际 launch 之后的激活证据；
4. 精确的 plan apply 映射。

`python/cuda_tactic_history.py` 保存历史正收益合同。所有支持 batch 上的所有
记录没有闭合这四条链路时，plan 生成会直接失败。

### 维护的 tactic 类型

两种架构只有在硬件确实不同时才保留不同 family，外部工作流使用同一套实现。
覆盖内容包括但不限于：

- exact-mask preprocess 及下游 mask elision；
- initial convolution/global 和 pointwise BN/activation；
- wide QKV/FFN/head projection 及多个 projection bundle；
- fused QKV + RoPE、packed QKV + RoPE；
- FlashAttention 不同 accumulation、tile 和 warp 路径；
- CUTLASS、CuTe、TileLang 的 fused dual FFN + SwiGLU；
- residual GEMM、linear2、outproj、preconv、postconv；
- RMSNorm、head BN、postconv BN + SiLU、value terminal；
- persisting L2，以及只有在真实 cache hit 后才视为激活的权重共享。

运行时不存在 B13 特权，也没有为旧实验选项名称保留兼容别名。旧 B13-only
生成物和重复的 SM120 扫描脚本已经删除。

## 统一 autotuner

离线 SDK 通过 CUDA Runtime 自动识别指定设备：

- compute capability 8.9 进入 SM89 工作流；
- compute capability 12.0 进入 SM120 工作流。

外部 orchestration、plan schema、历史合同、编译、测量、精度和打包逻辑均为
同一套代码。只有硬件确实要求不同实现时，才保留架构相关的候选生成和 backend
源码。

默认域为精确 B4-B32，每张卡两条推理 stream。正确顺序是“对一个 batch 完成
一整套决策流程”，而不是“一个决策横扫全部 batch 后再做下一个决策”。这样，
某个 shape 上获利的 tactic 会成为该 batch 的正常搜索坐标，而不是隐藏的固定
batch 专用实现。

对每个 batch，工作流会物化完整 family 空间，从显式的官方等价 baseline
开始，进行带激活门的 discovery，累积出自包含配置，再测 final joint 整图状态
并记录结果。29 个 batch 都完成之后，long gate 按稳定整图吞吐排序。只有
nnEval/s 最高的稳定 plan 会进行一次 8192-row FP32 replay，最终输出单 batch
的 `best-tactic-plan.json`。

discovery 短测数字不能作为发布性能。只有整图 long gate 和正确性门都通过时，
plan 才能标记为 production-ready。

### GPU 干扰策略

分发工程中已经完全移除 `gpu-lock`。它只用于多个开发 session 在同一机器上
协作，不属于用户运行时需求。

每个 benchmark 子进程在运行期间由 `nvidia-smi pmon` 监控。只占显存但 SM
占用为零的进程不算干扰；外部 PID 一旦在测量期间产生非零 SM 活动，benchmark
进程组会被停止，该样本作废。如果监控器无法确认状态，则测量 fail-closed。

## Batch-aware 前端与异步 pipeline

前端增加了两个彼此独立的开关：

- `nnBatchAwareDispatch`：等待完整 batch；只有目标 GPU 完全空闲时，才接受
  不足量的请求组，但物理 launch 仍 padding 到 plan batch。
- `cudaAsyncInferPipeline`：使用 pinned host memory、独立 DMA stream、常驻
  submission worker 和 CUDA event，将 preprocess/staging、H2D、D2H、输出完成
  移出计算关键路径。

plan loader 会强制第一个开关，因为精确 shape tactic 不能在不满足 batch 契约
时安全运行。异步 pipeline 仍是单独的开关，因为它改变 host 调度和内存生命
周期，而不是 tactic 选择。

单 slot event 协议如下：

```text
CPU 填充 pinned input
        |
        v
upload stream --inputReady--> compute stream --applyComplete--> download stream
       ^                            |                                  |
       |                            v                                  v
 inputConsumed 允许           执行 plan kernel                 outputReady 唤醒 CPU
 host slot 继续填充                                               |
       ^                                                           v
       +---------------- outputConsumed 允许原地覆盖 device slot
```

output-consumed event 从外部传给 backend。compute stream 在脏写单个 device
output slot 之前等待它，从而无需 ping-pong buffer，也不会覆盖尚未消费的输出。
H2D/D2H 在硬件支持时使用 copy engine，并不主动占用 SM。

每条 inference lane 都由不同的常驻 host worker 管理。中央 scheduler 可以让
每条 lane 各等待一个 batch，同时搜索侧继续生产另一个 batch，避免 scheduler
等待某条 stream 完成整段 host submission 后才开始喂下一条 stream。

## 多 GPU 隔离

plan 中的 stream 数是“每张卡”的数量，不是全局总数。双 stream plan 在一张
卡上使用两个 NNServer thread，在两张卡上使用四个，并在 GTP config 中将每一
对 thread 映射到对应设备。

所有 stream/event/copy/graph/cuBLAS 操作前都会选择 handle 所属的设备。GPU
idle 状态也按物理设备维护；不能因为另一张卡空闲而允许当前卡发出 partial
batch。双卡 8192-row 证书确认四条 lane 都实际处理了请求。

## 使用已提交 plan 运行 GTP

当前 plan 绑定模型 SHA-256：
`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`。
在正常 GTP 配置中使用 plan 的绝对路径。

单卡双流：

```cfg
cudaTacticPlanFile = /path/to/final-migration/plans/sm89/rtx4090d-b12-s2/best-tactic-plan.json
cudaTacticPlanBatch = 12

numNNServerThreadsPerModel = 2
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0

cudaAsyncInferPipeline = true
cudaEventPipelineUseGraph = false
```

双卡、每卡双流：

```cfg
numNNServerThreadsPerModel = 4
cudaDeviceToUseThread0 = 0
cudaDeviceToUseThread1 = 0
cudaDeviceToUseThread2 = 1
cudaDeviceToUseThread3 = 1
```

loader 会提供并验证精确 B12、精确 19x19、FP16/NHWC、
`nnBatchAwareDispatch=true`、只对最大 batch warmup，以及全部 CUDA tactic
override。用户配置与 plan 冲突时会报错，不会静默覆盖。

搜索线程数量可从以下公式开始调节：

```text
numSearchThreads = batch * (总 inference lane 数 + 1) + C
```

`C` 只是较小的 CPU/搜索长尾余量，通常可从 12-32 起步；仍需根据 CPU 和棋力
目标实测。`visits/s` 与神经网络求值数不是同一个指标：visits 必须严格大于真实
nnEval。固定 shape 的性能比较使用
`物理 launch batch 数 * 精确 batch / wall time`，包括 padding 行。

精简版运行契约见 [RUNTIME.md](RUNTIME.md)。

## CUDA Graph 边界

`cudaEventPipelineUseGraph=true` 同时要求异步 pipeline 和固定 batch
dispatcher。input-ready/output-consumed 这两个外部 event 保持在 capture 之外，
在所属 stream 上控制 replay；不会尝试把动态的外部 event 依赖捕获进 graph。

SM89 CUDA Graph 的功能和正确性已经验证。但在认证用 RTX 4090 D 上，eager
提交约快 0.8%，所以当前推荐 eager。SM120 graph replay 尚未进入 production
认证范围。

## 正确性护栏

发布精度使用离线不可变的全 FP32 输出，生成时显式关闭 SM89/SM120 两种自定义
backend。reference metadata 绑定二进制、模型、输入 corpus、行数、精确 batch
行为和全部 hash。

比较器要求：

- 逻辑行数严格等于 8192；
- reference 与 candidate 的 targets 和所有输入 section 逐字节一致；
- 模型和 corpus SHA-256 与当前输入相符；
- candidate metadata 证明最大 batch 和固定尾部 padding 正确；
- policy top-1/probability、value、score、ownership 和 weighted loss 均通过
  阈值。

物理尾 batch 通过重复真实请求补满，但只序列化原始 8192 个逻辑行。这样能让
exact-batch AOT kernel 在尾部仍保持激活，同时不会掩盖请求错位、重排或计数
错误。

`katago runnngtpstresstest` 将准备好的 search 形态请求经过普通 evaluator
scheduler，CPU 对所有输出 head 与离线 FP32 结果比较，第一次错误立即停止。
通过重复 8192 个局面进行稳定性和多卡长测，但不会拆掉或替换待测主体。

## 可复现环境与离线交付物

工程生成两类不同的 tar：

1. autotune SDK：包含完整 KataGo 源码、固定 CPython 3.12.13、CUDA 13.2
   编译工具链、cuDNN 9.25、CUTLASS/CuTe、FlashAttention、TileLang、Triton、
   Quack、TVM-FFI、zlib 源码、固定 wheels、模型、8192-row corpus、plans、
   patches 和 SHA-256 manifest。目标机不需要 GitHub clone、PyPI、APT 或自行
   寻找第三方库。
2. 预编译 runtime tar：包含编译后的 KataGo CUDA backend、私有的
   CUDA/cuDNN/C++/glibc runtime、plans、installer 和 hash。它安装到一个
   隔离目录，目标机只需要兼容的 NVIDIA 驱动。

制作 release 时会解析当时最新的上游源码版本，随后把精确 revision 冻结进
tar。关键优化器依赖都从携带源码编译进私有 Python 环境；PyPI 上的琐碎依赖
固定版本和 hash。release 没有写死 Ubuntu 24.04：源码环境脚本识别受支持的
Ubuntu 版本，离线 SDK 的基线是 Linux x86-64、glibc 2.28 或更高。

编译并行度根据 CPU affinity 和可用/cgroup 内存保守计算，不会盲目使用
`-j$(nproc)`，也不固定写死 `-j4`/`-j8`；用户仍可显式指定 jobs。

### 配置开发环境

在仓库根目录执行：

```bash
./final-migration/environment/setup.sh all
```

### 制作源码完备的 autotune tar

```bash
AUTOTUNE_CORPUS=/path/to/8192-full19.npz \
AUTOTUNE_CORPUS_MANIFEST=/path/to/8192-full19.manifest.json \
./final-migration/autotune/package-autotune.sh
```

将 tar 解压到可写的持久化目录后：

```bash
./setup.sh
./run-autotune.sh --device 0
```

### 制作预编译推理 tar

```bash
./final-migration/environment/setup.sh package
```

每个外层 tar 都会生成相邻 `.sha256`；预编译 runtime 还会生成经过 hash 校验的
非侵入式安装脚本。

## 目录说明

- `cpp/neuralnet/cudatacticplan.*`：production plan loader 和接收端校验。
- `cpp/neuralnet/cudabackend_sm89*`：维护中的 SM89 backend。
- `cpp/neuralnet/cudabackend_sm120*`、`sm120_aot/`：维护中的 SM120 backend。
- `cpp/neuralnet/nneval.*`：batch-aware dispatcher 和异步 event scheduler。
- `cpp/tests/testnnbatchingdispatcher.cpp`：scheduler 状态机测试。
- `cpp/tests/testnngtpharness.cpp`：全输出 GTP 形态正确性 harness。
- `python/cuda_tactic_workflow.py`：统一的架构感知扫描器。
- `python/cuda_tactic_history.py`：正收益历史四链路合同。
- `final-migration/autotune/`：离线 SDK 制作与运行入口。
- `final-migration/environment/`：开发环境和预编译 runtime 打包。
- `final-migration/plans/`：纳入 Git 的 production plans。
- `final-migration/records/`：精简的认证记录。

完整优化历史审计见
[OPTIMIZATION_HISTORY_AUDIT_20260808.md](OPTIMIZATION_HISTORY_AUDIT_20260808.md)，
SM89 runtime 认证见
[records/plan-runtime-sm89-20260809.md](records/plan-runtime-sm89-20260809.md)。

## 已知边界

- 本项目只优化 CUDA backend，不需要 TensorRT。
- production plan 当前要求精确 19x19 和匹配的模型 hash。
- 已提交的 production plan 是 SM89/RTX 4090 D B12/S2；设备或模型不兼容时
  会被有意拒绝。
- SM120 backend 和扫描器已实现，但仍需要新的统一硬件扫描和最优 plan 认证。
- CUDA Graph 是可选项，当前不是 SM89 已认证的最快模式。
- 在短搜索或固定 visits 场景中，过高搜索线程数即使提高 GPU 利用率，也可能
  降低棋力。

KataGo 上游的通用功能、GTP 命令和使用方式仍以仓库根目录未改动的官方文档为
准。
