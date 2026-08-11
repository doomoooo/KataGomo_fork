# KataGo Plan 驱动 CUDA Fork

中文 | [English](README.md)

本仓库从官方 [`lightvector/KataGo`](https://github.com/lightvector/KataGo)
提交 `6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19`（`v1.17.2`，获取于
2026-08-07）重新分叉。KataGo 的 GTP、分析、搜索、模型和围棋逻辑保持不变，
新增面向 NVIDIA SM89 和 SM120 的 shape-specialized、Plan 驱动 CUDA 推理路径。

本项目只优化 CUDA backend。production 路径不需要也不使用 TensorRT。本分支
的改动不属于上游官方 KataGo。

## 本分支增加的方法

| 方面 | 方法 |
| --- | --- |
| Kernel 选择 | 离线整图扫描生成显式、可版本化的 tactic plan |
| Shape 合同 | 严格 19x19、FP16/NHWC 和已认证的物理 batch |
| Batch 调度 | 满 batch 立即启动；GPU 空闲时允许逻辑不足量请求 padding 到 plan batch |
| Host 提交 | 每条 inference lane 一个常驻 worker，不在 lane 之间串行等待 host completion |
| 数据搬运 | pinned staging、独立 upload/download stream 和 CUDA event 依赖 |
| Buffer 复用 | 单个 device slot，由 input-consumed/output-consumed event 保护 |
| CUDA Stream | 每个优化 kernel 显式使用所属 NN-server stream |
| 多 GPU | stream、event、handle、buffer 和 idle 状态均按设备隔离 |
| 正确性 | 不可变 8192-row 全 FP32 reference、请求同一性检查和 GTP 形态长测 |
| 分发 | 源码完备 autotune tar 和独立的非侵入式预编译 runtime tar |

## 性能参考

这是 README 中唯一的性能汇总。所有结果使用相同的 70M 参数模型、严格 19x19、
FP16、双推理 stream，并按
`物理 launch batch 数 * batch / wall time` 计算。前两行是已接受的历史 RTX
4090 baseline；plan 行是当前 RTX 4090 D 硬件证书。GPU 型号和宿主机不同，
因此这里只作参考，不声称是归一化 speedup。

| Backend | GPU | Batch | Physical nnEval/s | 证据 |
| --- | --- | ---: | ---: | --- |
| 官方 CUDA baseline | RTX 4090 | 13 | 1876.270 | [baseline 记录](../docs/baseline-2026-08-05.md) |
| TensorRT baseline | RTX 4090 | 13 | 2542.940 | [baseline 记录](../docs/baseline-2026-08-05.md) |
| 当前最优已提交 CUDA plan | RTX 4090 D | 12 | 3110.690824 | [plan 证书](plans/sm89/rtx4090d-b12-s2/README.md) |

TensorRT 仅作为历史对比，不进入本工程的环境、编译、运行时或 release 包。

## Plan 驱动 backend

autotuner 输出 schema 1 的 `cuda-tactic-plan` JSON。production plan 完整绑定：

- 架构与接收端硬件能力；
- 严格 19x19、FP16/NHWC、模型 SHA-256、batch 和每设备 stream 数；
- 每个实现目录的自包含 override；
- 源码、生成 artifact、配置和被测 binary 的 hash；
- discovery 与稳定整图 long gate 证据；
- 所有保留正收益实现的闭环；
- 最优 plan 唯一一次不可变全 FP32 正确性证书。

`cpp/neuralnet/cudatacticplan.cpp` 在 evaluator 创建前加载 plan。schema、模型、
棋盘、精度、架构、batch、stream 拓扑、设备能力或 tactic 任意一项不兼容都会
拒绝启动。计划中的实现不能静默 fallback 到官方 kernel。

backend 只在选中实现真正 launch 成功之后记录激活。每个可搜索实现必须闭合
四条链路：

1. backend 实现；
2. 已物化扫描候选；
3. launch 后 activation marker；
4. 精确 plan apply 映射。

`python/cuda_tactic_history.py` 是保留正收益历史合同。任意支持 batch 上的任意
记录缺少一条链路，plan 生成都会失败。

### 实现目录与决策组

目录名称只是实现清单，不表示一个 transformer block 中有同样数量的算子，也
不表示所有调优轴彼此正交。两种架构都暴露十个有序决策组。静态闭环门要求每个
共享 runtime key 和声明式依赖只属于一个决策组。

bundle 在所属组内测量；后面的组不能重写前面组的状态。SM120 中 packed QKV
只是 input layout 选择，不会强制 FlashAttention 的 tile 或 accumulator 模式。

维护集合覆盖 initial path、pointwise activation、wide projection、QKV/RoPE、
FlashAttention、fused FFN/SwiGLU、residual/projection GEMM、normalization/head、
persisting L2，以及只有真实 cache hit 才算激活的模型权重共享。

优化 backend 只支持严格 19x19、FP16、NHWC。不存在 mask tactic、动态棋盘兼容
路径、B13 特权或旧实验选项兼容层。

## Autotune 流程

SM89 与 SM120 共用外部 orchestration、plan schema、历史合同、测量、正确性和
打包代码。只有硬件实现确实不同时才保留架构相关候选生成。

默认流程如下：

1. 检测目标 CUDA 设备。compute capability 8.9 选择 SM89，12.0 选择 SM120。
2. 使用自包含且不依赖 artifact 的稳定优化图扫描 B4-B32。
3. 选择吞吐最高的三个 batch。
4. 每个入选 batch 独立编译精确 shape artifact，并物化完整实现目录。
5. 按决策组顺序执行带激活门的第一轮，同时累积自包含计算图。
6. 在改进后的整图上重扫每个目录第一轮的 top 10。用更长 ABBA 测量确认临时
   变化，并在有界次数内重复 refinement 直到不再变化。
7. 执行稳定整图 long gate，按 physical nnEval/s 排序 batch。
8. 只对最快 plan 执行一次 8192-row FP32 replay。
9. 历史、激活、稳定性和精度门全部通过后才生成
   `best-tactic-plan.json`。

正确顺序是一个 batch 完成整个决策流程后再优化另一个 batch。
`--full-batch-scan` 会对 B4-B32 每个 shape 执行完整流程，默认关闭。

discovery 短测只用于剪枝，不是发布性能。

### GPU 干扰策略

每个 benchmark 子进程都由 `nvidia-smi pmon` 监控。只占显存且 SM 使用为零的
进程允许存在。外部 PID 在测量期间出现非零 SM 活动时，benchmark 进程组会被
停止，该样本作废。无法确认监控状态时测量 fail-closed。

分发工作流不包含 GPU lock，也不修改功耗上限或频率。

## 运行时 pipeline

前端有两个独立开关：

- `nnBatchAwareDispatch` 控制请求聚合和精确物理 batch padding。
- `cudaAsyncInferPipeline` 使用 pinned memory、独立 DMA stream、常驻 worker
  和 CUDA event，将 staging、H2D、D2H 与完成处理移出计算关键路径。

plan loader 强制 batch-aware dispatch，因为精确 shape tactic 在其他条件下不
安全。异步 pipeline 仍可独立配置，因为它改变调度与内存生命周期，不改变 tactic
选择。

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
       +---------------- outputConsumed 允许原地复用 device slot
```

外部传入的 output-consumed event 保证前一份结果被消费前，backend 不会脏写单个
device output slot，因此无需 ping-pong output allocation。H2D/D2H 在硬件支持
时使用 copy engine，不主动占用 SM。

每条 inference lane 都有自己的常驻 host worker。scheduler 可以让每条 lane 各
等待一个 batch，同时搜索侧继续生产另一个 batch，不会等待一条 stream 的完整
host submission 后才喂下一条。

### 多 GPU 映射

plan 的 stream 数是每设备数量。双 stream plan 在一张卡上使用两个 NN-server
thread，在两张卡上使用四个。每一对 thread 必须映射到对应接收设备。stream、
event、copy、CUDA Graph、cuBLAS、buffer 和 idle-state 操作都会先选择该设备。

GPU 空闲只允许它自己的不足量逻辑请求组启动。另一张卡的 idle 状态不能授权
当前忙卡发出 partial batch。

## 运行 GTP

完成 setup 和 autotune 后，在 CUDA Runtime device 0 上启动 GTP：

```bash
./run.sh
```

launcher 会检测接收设备、选择兼容的已认证 plan，校验模型、plan 文件和被测
binary 的 hash，并自动设置精确 batch、双 lane 设备映射、batch-aware dispatch、
异步 event pipeline 和搜索线程预算。它优先使用被测 binary hash；只有保留的
result 能证明 target、batch 和完整 apply 映射相同，才会自动接受重编译 binary，
backend activation 仍然 fail-closed。需要时可以选择其他 CUDA Runtime ordinal
或显式覆盖输入：

```bash
./run.sh --device 1
./run.sh --model /data/model.bin.gz --config /data/gtp.cfg
```

`./run.sh --help` 列出显式 plan/binary 覆盖和 GTP 参数透传方法。loader 仍然
校验精确 batch、full-board shape、FP16/NHWC、只 warmup 最大 batch 和全部
tactic override。

搜索线程数量可从以下公式开始：

```text
numSearchThreads = batch * (总 inference lane 数 + 1) + C
```

`C` 是较小的 host/search 长尾余量，必须根据目标 CPU 和棋力实测。
`visits/s` 必须严格大于真实 logical `nnEval/s`；固定 shape backend 比较使用
前文定义的物理 padding 指标。

精简接收端合同见 [RUNTIME.md](RUNTIME.md)。

### CUDA Graph 边界

`cudaEventPipelineUseGraph=true` 同时要求异步 pipeline 和固定 batch dispatcher。
input-ready 与 output-consumed event 保持在 capture 之外，在所属 stream 上控制
replay。CUDA Graph 是可选搜索/运行模式，不是 plan 格式的默认假设。

## 正确性门

不可变 reference 通过官方全 FP32 路径生成，SM89/SM120 优化 backend 均显式
关闭。metadata 绑定 binary、模型、corpus、行数、精确 batch 行为和 hash。

认证要求：

- 逻辑行数严格等于 8192；
- reference 与 candidate 的 targets 和所有 input section 逐字节一致；
- 模型与 corpus SHA-256 匹配；
- 具有精确最大 batch 和固定尾部 padding 证据；
- policy、value、score、ownership 与 weighted loss 聚合阈值通过；
- 逐请求最坏 maximum-absolute 和每个 head RMSE 阈值通过。

`katago runnngtpstresstest` 将重复的 search 形态请求发送给普通 evaluator
scheduler，CPU 检查每个输出 head，第一次错误立即停止。它不会替换待测的
scheduler 或 backend。

## 可复现分发

工程生成两类非侵入式产物：

1. 源码完备 autotune SDK：包含 KataGo 源码、固定 CPython、CUDA 编译工具链、
   cuDNN、优化器/第三方库源码、锁定 wheels、模型、corpus、plans、patches、
   licenses 和 SHA-256 manifests。目标机无需 GitHub clone 或自行寻找依赖。
2. 预编译 runtime tar：包含 CUDA backend、所需用户态 runtime 库、plans、
   installer、licenses 和 hash。接收端只需要兼容的 NVIDIA 驱动。

关键优化依赖从 tar 携带源码编译；PyPI 小依赖固定精确版本与 hash。setup 检测
受支持 Ubuntu 版本，不写死 Ubuntu 24.04。编译并行度根据内存保守计算。

开发环境：

```bash
./final-migration/environment/setup.sh all
```

源码完备 autotune tar：

```bash
AUTOTUNE_CORPUS=/path/to/8192-full19.npz \
AUTOTUNE_CORPUS_MANIFEST=/path/to/8192-full19.manifest.json \
./final-migration/autotune/package-autotune.sh
```

解压到可写持久化目录后：

```bash
./setup.sh
./run-autotune.sh --device 0
```

预编译 runtime tar：

```bash
./final-migration/environment/setup.sh package
```

每个外层 tar 都有相邻 `.sha256`。runtime installer 安装到隔离 prefix 前还会
验证内部 manifest。

## 目录说明

- `cpp/neuralnet/cudatacticplan.*`：production plan loader 和接收端校验。
- `cpp/neuralnet/cudabackend_sm89*`：维护中的 SM89 backend。
- `cpp/neuralnet/cudabackend_sm120*`、`sm120_aot/`：维护中的 SM120 backend。
- `cpp/neuralnet/nneval.*`：batch-aware dispatcher 和异步 scheduler。
- `python/cuda_tactic_workflow.py`：统一架构感知扫描器。
- `python/cuda_tactic_history.py`：保留正收益历史合同。
- `final-migration/autotune/`：离线 SDK 入口和规范。
- `final-migration/environment/`：环境与 runtime 打包。
- `final-migration/plans/`：每种已认证 GPU 型号一个当前 production plan。
- `final-migration/records/`：详细编译、实验和认证日志。
- `docs/cuda-tactic-workflow.md`：详细搜索合同。

本工程只覆盖 CUDA backend 和严格 19x19 FP16/NHWC 路径。通用 KataGo 与 GTP
行为继续以未修改的上游文档为准。
