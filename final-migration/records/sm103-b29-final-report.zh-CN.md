# B300 B29/S2 最终优化报告

日期：2026-08-18

最终保留的 CUDA 图在 B29/S2 下达到 **9201.745296 combined nnEval/s**，
相对极差为 **0.842%**。相对于固定的 TensorRT 10.16.1.11 中位数
**6733.719141 nnEval/s**，提升为 **+36.652%**。这是固定工作负载下的优化
资格结果；本报告不作部署认证声明。

## 固定契约

| 项目 | 固定值 |
| --- | --- |
| 设备与目标 | NVIDIA B300 SXM6 AC，148 个 SM，compute capability 10.3，加速目标 `sm_103a` |
| 工作负载 | 固定 batch 29、两个独立推理流（S2）、19x19 棋盘 |
| 模型 | `b11c768h12nbt3tflrs-fson-silu`；SHA-256 `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`；展平后 10469 行 |
| CUDA 图格式 | FP16/NHWC |
| 优化目标 | 在通过固定的 8192 行聚合精度门和逐请求精度门的前提下，最大化 B29/S2 combined nnEval/s |
| 对比锚点 | TensorRT 10.16.1.11；二进制 SHA-256 `883024dc8bbc02e7f6b05b0431034652931acc760b76e7fd455dc996af278612`；5 个 1000 iteration 样本；中位数 `6733.719141`；极差 `0.4415%` |

不可变身份和基准契约见[基线锚点](../plans/sm103/b300-b29-s2/baseline-anchor.json)
与[请求精度门控制](../plans/sm103/b300-b29-s2/request-gate-control.json)。下文所有
吞吐率都是两个推理流的合计值。

## 保留谱系与实测阶梯

最终累积谱系为：

`portable tactics -> fused FFN -> no-AB12 -> native FA4 -> pinned QKV cuBLASLt id70 -> QKV aux2`

| 阶段 | 累积变更 | nnEval/s 中位数 | 极差 | 相对父阶段 | 相对 TRT 10.16 | 保留提交 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 固定对照 | TensorRT 10.16.1.11 基线 | 6733.719141 | 0.442% | -- | -- | 仅控制 |
| 测量锚点 | CUDA portable-empty，未保留 tactic | 5000.394925 | 0.743% | -- | -25.741% | -- |
| 02a | beta=1 residual GEMM | 5600.299778 | 0.607% | +11.997% | -16.832% | `6723ea0c` |
| 02b | C384 warp4-vec8 RMSNorm | 5974.164849 | 0.155% | +6.676% | -11.280% | `abdeb8e7` |
| 02c | planar fused Q/K RoPE | 6495.313934 | 0.079% | +8.723% | -3.540% | `f88e423a` |
| 02d | half2 affine-SiLU | 6737.964266 | 0.768% | +3.736% | +0.063% | `23f42fb9` |
| 02e | half8 C1152 SwiGLU | 7150.091831 | 0.012% | +6.116% | +6.183% | `1c2aad4c` |
| 03a | 保留所需 FP16 投影 round-trip 的 fused FFN | 8077.636942 | 0.219% | +12.972% | +19.958% | `5d7adaa2` |
| 03b | 删除 fused FFN 未使用的 AB12 输出与存储路径 | 8587.808904 | 0.831% | +6.316% | +27.534% | `38fe3334` |
| 06 | native FA4 D32 attention，FP32 QK/PV accumulation | 8949.114828 | 0.333% | +4.207% | +32.900% | `230dd195` |
| 07a | 精确固定的 QKV cuBLASLt id70 tuple | 9058.808862 | 0.414% | +1.226% | +34.529% | `d40282a2` |
| 08 | Q 在主流，K/V 分别在两个辅助流 | **9201.745296** | **0.842%** | **+1.578%** | **+36.652%** | `b5e17d76` |

portable 子阶段阶梯的每步使用两个 1000 iteration 样本，并由累积后的 8192 行
replay 完成精度闭环；Stage 03a 起使用 200 次 warmup 后的 5 个 1000 iteration
长测样本。增量百分比以每个阶段声明的父阶段为基准，而不是以 TensorRT 行为基准。
规范的保留选择记录在 [B29/S2 tactic plan](../plans/sm103/b300-b29-s2/portable-baseline.json)。

### 保留谱系如何形成

1. 调优前先冻结基线和数值控制。第一个 fused FFN 原型虽快，却省略了官方 FP16
   投影边界；其 policy 与 ownership 请求误差约为 TensorRT 控制的 10--20x，
   因而被拒绝。恢复投影 round-trip 后，才得到可保留的 Stage 03a 实现。
2. 继续 native 调优前，先累积并消融五项已验证的 portable tactics。residual fusion、
   C384 RMSNorm、planar fused Q/K RoPE、affine-SiLU 与 C1152 SwiGLU 得以
   保留；persisting-L2 trunk/inner 与 policy/head bundle 分别回退 `1.886%`
   和额外 `1.642%`，因此排除。
3. fused FFN 删除了 8052 次 projection/SwiGLU launch，并在整图 profile 中使该
   边界快 `1.987x`。随后删除其死 AB12 输出，又取得独立的 `+6.316%` 长测提升。
4. 多种 QKV/RoPE 和 FFN 资源整形方案未通过整图或隔离门。随后保留的 attention
   替换减少了暴露的关键路径时间，并增加 `+4.207%`。低精度 accumulator 控制
   没有迁移成功，FP32 模式保持不变。
5. 通用 QKV cuBLASLt autotuning 偶尔选择 id71 并丢失收益，其中一个进程仅有
   `8780.674` nnEval/s。因此显式构造并回读精确的 id70/tile23/stages35/
   cluster5/zero-workspace tuple。最后一步保持此算法完全不变，只调整 Q/K/V
   依赖拓扑。

## 本次新发现且值得推广的架构无关优化技巧

本次工作中新发现、可广泛复用的优化结论严格只有以下两项。portable tactic bundle
是本仓库已有经验；FA4、tcgen05 与 SM103 专用指令选择只是最终实现事实，不作为
这里声称的通用经验。

### 1. 修改 ready DAG，扩大 ready frontier

**机制。** 深的 host 提交队列不等于宽的 dependency-ready GPU grid 集合。控制
trace 中每个外层流已有约 1016 个 pending descriptor（p99/max 为 1024），host
领先约 17.45 ms；但同流 FIFO 依赖通常仍只暴露两个流头。把
`CUDA_SCALE_LAUNCH_QUEUES` 增大到 2x 或 4x，或把 Hyper-Q connection 从默认
8 增至 16，均无收益。

事后检查固定 TensorRT engine，得到 `num_aux_streams=0`，其严格 B29 结构探针
的 439 个 kernel 也全部位于单一 kernel stream。它把 Q/K/V 投影做成一个 grouped
grid，而不是辅助流 ready DAG；这证明保留的 frontier 改动并非照搬该 plan。

Stage 08 在 pre-RMS 后记录一个 event，让 Q 留在主流，并让 K、V 分别在两个
持久 nonblocking 辅助流上运行；两个分支各自拥有独立 cuBLASLt handle/plan 和
私有 workspace。主流在既有 RoPE 与 attention 前等待两个 done event。这样 Q、
K、V 成为兄弟节点：S2 下的 ready frontier 从最多两个 Q grid 扩为最多六个
Q/K/V grid，同时没有改变 GEMM 算法、tensor、launch 数或下游顺序。

**证据。** 稳态 100-forward 对比中，两边均精确 launch 40,596 个 kernel，其中
9,900 个为 QKV。QKV 总服务时间反而从 `100.438 -> 109.451 ms` 增加
`8.974%`，但 QKV interval union 从 `90.124 -> 66.377 ms` 降低 `26.349%`。
整图 union 从 `331.831 -> 322.454 ms` 降低 `2.826%`，overlap 增加
`3.787%`，trace 中 concurrency depth 至少为 3 的时间从
`9.470 -> 30.456 ms`。五样本长测提升 `1.578%`。单辅助流诊断重新引入 K 到 V
的 FIFO 边，只比控制快 `0.346%`，并比 aux2 慢 `2.000%`；因此原因是 frontier
宽度，而不是减少 event 数。

**迁移条件。** 仅在兄弟算子共享只读输入、写入互不相交的输出，并具有清晰下游
join 时使用。每个分支需要所有权正确的 stream-local library state 与 workspace；
event、allocator lifetime、teardown 与 graph capture 都必须维持 join。kernel 的
资源形状也必须为整图调度留出合理机会。应验证真实 ready DAG 与端到端 union；
仅给串行或资源饱和的依赖链增加 stream 并不足够。

证据：[submission-window analysis](../plans/sm103/b300-b29-s2/evidence/scheduler-submission-window-qkv-aux-plan.json)、
[queue environment probe](../plans/sm103/b300-b29-s2/evidence/scheduler-queue-environment-probe.json)、
[Stage 08 result](../plans/sm103/b300-b29-s2/evidence/stage-08-qkv-aux-streams.json)
、[aux1 diagnostic](../plans/sm103/b300-b29-s2/evidence/stage-08b-qkv-aux1-diagnostic.json)
与 [TensorRT engine inventory](../plans/sm103/b300-b29-s2/evidence/tensorrt-engine-plan-audit.json)。

### 2. 删除下游未消费的输出及其存储流量

**机制。** fused FFN provider 在生成最终 C 的同时，还把两个 FP16 投影物化为
AB12；KataGo 实际只消费 C。每个流的 AB12 输出为 `48,241,152 B`。Stage 03b
保留外部可见 ABI 参数与所有必要数值边界，但从 device 路径中删除 AB12 device
pointer/descriptor、register/shared fragment、四阶段 shared ring、copy 和全部
TMA store。tcgen05 mainloop、FP16 投影 round-trip、fast SwiGLU 算术与 C store
均保持不变。这不只是减少算术：算术被刻意固定，删除的是未使用值及其完整存储
流水线。

**证据。** 在相同的 148-block/192-thread launch 下，dynamic shared memory 从
`214016 -> 181248 B`，每线程 register 从 `74 -> 66`，L2 sector 从
`2262595 -> 754848`（`-66.64%`）。NCU replay duration 从
`44.58 -> 40.10 us`；整图 fused-FFN 总时间从 `110.524 -> 95.245 ms`；累积
长测增加 `6.316%`。AB12 sentinel 保持未写，紧致输出检查与完整 replay 的所有
精度门均通过。

**迁移条件。** 必须从 consumer graph 证明该输出为死值，且不存在 alias、callback、
synchronization 或外部可观察副作用。即使不再物化 tensor，也要保留旧输出路径可能
施加的语义 rounding boundary。应删除完整 descriptor/copy/store/ring 路径，而不
只是抑制最后一次 store；随后重新测量资源、整图调度与数值输出，因为资源变化可能
在源代码算术不变时仍扰动累积浮点 rounding。

证据：[Stage 03b no-AB12 profile](../plans/sm103/b300-b29-s2/evidence/stage-03-no-ab12-profile.json)。

## 精度与稳定性结果

最终精度门使用固定的 8192 行、19x19 corpus 和同一 CUDA FP32 replay。对于
policy probability、value probability、raw score 与 ownership probability，
每个 head 的最坏请求 maximum-absolute error 和 RMSE 均不得超过相应 TensorRT
10.16 控制的 `2.25x`；除逐请求指标全部通过外，还必须通过聚合门。

| 最终 Stage 08 指标 | 结果 |
| --- | ---: |
| Policy probability RMSE | `9.53026e-5` |
| 相对 FP32 的 policy top-1 一致率 | `99.8291%` |
| Value outcome RMSE | `0.00213206` |
| Score mean RMSE | `0.00184150` |
| Ownership sigmoid RMSE | `0.000233715` |
| 相对 TensorRT 控制的最大请求比率 | `1.94636x`（通过，上限 `2.25x`） |

输入与 target 相对 reference byte-identical。由于调度变化扰动了累积 FP16
rounding，Stage 08 与 Stage 07 并非 byte-identical；但两者 policy top-1 一致率为
100%，probability RMSE 仅 `2.866e-6`，固定 FP32/TRT 精度门均通过。

Stage 03a bad-case 调查还排除了一个潜在实现错误。fast-math 最坏样本 row 4124/
move 358 是合法的真实 B29 行，而非 padding；input 与 target byte-identical，raw-logit
correlation 为 `0.9999865`，且没有 permutation、固定 index 或 spatial-layout
signature。strict exp/div 修复该行，却把整图吞吐降至约 `5554` nnEval/s。一次
Newton refinement 保持吞吐，但把最坏样本移到 row 7111/move 111。保留的 fast
variant 最大请求比率为 `2.148x`，而损坏的 no-roundtrip variant 仍约为 10--20x
并失败。该现象被归类为小算术路径差异的非线性放大，而非实现 bug。详见
[bad-case study](../plans/sm103/b300-b29-s2/evidence/stage-03a-bad-case-study.json)。

## 附录 A：负结果与教训

以下均为实验闭环，不是推荐的优化技巧。

| 路线 | 实测结果与处置 |
| --- | --- |
| 未恢复投影 rounding 的原始 fused FFN | 整图速度有潜力，但 policy max-abs/RMSE 达 TensorRT 控制的 `15.132x/19.756x`，ownership 达 `15.711x/10.029x`。拒绝；恢复缺失的 FP16 投影边界。 |
| Fused planar QKV+RoPE | 隔离 S1 改善到 `22.756 us`，但整图从 `8587.809 -> 8048.267` nnEval/s。kernel sum 虽下降，union 却上升 `5.02%`，overlap 下降 `17.19%`，属于异构资源生命周期互扰。G148 retry 在 A-B-B-A 中回退 `4.692%`，整图 union 增加 `2.82%`；ordered split retry 回退 `9.319%`。路线关闭。 |
| Attention outproj + next-FFN RMSNorm | 精确输出与更多 overlap 无法弥补服务时间膨胀：current-best A-B-B-A center 回退 `3.868%`，整图 union 增加 `3.041%`（`466.821 -> 481.016 ms`）。丢弃。 |
| FA4 FP16 accumulator controls | 相对 FP32 S2 `37.682 us`，QK16/PV16/both16 为 `41.486/39.792/42.779 us`，慢 `10.10%/5.60%/13.53%`。所有模式保持相同物理 TMEM footprint，故保留 FP32。 |
| DeepGEMM BF16 | 包含 cast 的 coordinated S2 为 `38.2016 us`，id70 FP16 为 `18.4898 us`，即慢 `2.066x`（`+106.61%`）；raw DeepGEMM BF16 已慢 `17.41%`。整图接入前关闭。 |
| FlashInfer standalone D32 attention | 精度正确，但 S1/S2 比 cuDNN 慢 `11.16%/14.01%`。仅保留作源码参考；未进入整图。 |
| Triton controls | Persistent-TMA plain wide QKV 在 S1/S2 慢 `2.099x/2.712x`，dual FFN 慢 `2.200x/2.258x`。Triton linear2+residual 约慢 `51%/65%`。plain-kernel 路线均在整图前停止。 |
| FFN 资源整形 | AB3 达到预测的 two-CTA residency 边界，却在累积图中失败；修正后的 M128xN64/AB4 在隔离 S2 中慢 `18.67%`。仅有 residency 并不构成收益。 |
| 其他已审计替换 | cuDNN SDPA engine/knob surface 仍由既有配置胜出；MSLK 1.3 没有可直接替换的 SM103a FP16 路径；其他调查的 backend 需要大规模移植或精度变更，却没有可取代保留图的实测依据。它们未被包装为优化结论。 |
| TensorRT engine inventory | 固定 plan 暴露 grouped QKV、两次复合 RoPE、`_gemm_mha_v2` 与三次 launch 的 FFN，但没有尚未尝试且可直接替换的新算子。正常 GTP 会消费 ownership；benchmark 专用的空输出指针不是 production 死输出，该假设已明确关闭。 |

完整负结果由[仅追加阶段日志](sm103-b29-optimization-stages.md)索引，包括
[QKV+RoPE profile](../plans/sm103/b300-b29-s2/evidence/stage-04-qkv-rope-profile.json)、
[persistent G148 result](../plans/sm103/b300-b29-s2/evidence/stage-04b-qkv-rope-persistent-g148-result.json)、
[split-QKV result](../plans/sm103/b300-b29-s2/evidence/stage-04c-qkv-rope-split3-result.json)、
[outproj/RMSNorm result](../plans/sm103/b300-b29-s2/evidence/outproj-rmsnorm-fusion-profile.json)、
[accumulator controls](../plans/sm103/b300-b29-s2/evidence/stage-06b-fa4-accumulator-modes.json)、
[DeepGEMM kill test](../plans/sm103/b300-b29-s2/evidence/deepgemm-bf16-kill-test.json)
与 [MSLK audit](../plans/sm103/b300-b29-s2/evidence/mslk-1.3-sm103-low-hanging-audit.json)。

## 复现与证据入口

1. 使用 `./setup.sh install` 安装严格 managed 环境，通过
   `source final-migration/environment/activate-sm103.sh` 进入，再运行
   `./setup.sh verify` 与 `./setup.sh build`。已提交脚本会校验 Python 3.14.7
   archive、80 个精确 package pin、CUDA/cuDNN 身份、源码 commit 与 managed
   `ptxas` 路径。
2. 从[基线锚点](../plans/sm103/b300-b29-s2/baseline-anchor.json)、
   [请求控制](../plans/sm103/b300-b29-s2/request-gate-control.json)和
   [保留 tactic 选择](../plans/sm103/b300-b29-s2/portable-baseline.json)开始。
3. 按[阶段日志](sm103-b29-optimization-stages.md)及其 tracked evidence JSON 复现。
   保留 native 阶段记录依次为
   [fused FFN](../plans/sm103/b300-b29-s2/evidence/stage-03a-fused-ffn.json)、
   [no-AB12](../plans/sm103/b300-b29-s2/evidence/stage-03-no-ab12-profile.json)、
   [native attention](../plans/sm103/b300-b29-s2/evidence/stage-06-fa4-native-profile.json)、
   [pinned id70](../plans/sm103/b300-b29-s2/evidence/stage-07-cublaslt-qkv.json)和
   [aux2](../plans/sm103/b300-b29-s2/evidence/stage-08-qkv-aux-streams.json)。
4. 重新构建并运行 tracked source-contract tests，包括
   [FFN](../../python/tests/test_sm103_cudnn_oss_ffn_hook.py)、
   [FA4](../../python/tests/test_sm103_fa4_native_hook.py)、
   [id70](../../python/tests/test_sm103_projection_gemm_lt_contract.py)和
   [aux2/lifetime](../../python/tests/test_sm103_qkv_aux_contract.py)。所有 GPU 工作通过
   [with-gpu-lock.sh](../environment/with-gpu-lock.sh)串行化。
5. 重复精确 B29/S2 长测协议（最终阶段为 200 次 warmup、5 个独立的
   1000 iteration 样本），再运行 8192 行 replay 与固定请求对比。evidence JSON
   包含 binary、report、replay、comparison 和 source hash；被忽略的本地 raw
   artifact 不能替代这些 tracked identity。

保留提交链为：

```text
be6b4107  portable adapter infrastructure
10afb5ca  fixed B29/TRT control
6723ea0c  beta=1 residual
abdeb8e7  C384 RMSNorm
f88e423a  planar Q/K RoPE
23f42fb9  affine-SiLU
1c2aad4c  C1152 SwiGLU
5d7adaa2  fused FFN with projection round-trip
38fe3334  no-AB12
230dd195  native FA4
d40282a2  pinned QKV id70
b5e17d76  QKV aux2 ready-DAG
```

same-binary A/B/B/A、NSYS interval-union/overlap accounting、targeted NCU
resource measurement、fail-closed selection check、bad-case analysis 与 GPU
locking 是判断该谱系所用的实验和验证方法；它们不是额外的优化技巧。
