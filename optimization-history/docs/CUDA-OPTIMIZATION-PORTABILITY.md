# 固定 19x19 CUDA 优化的可移植性与小范围搜索总览

更新时间：2026-08-06 UTC

本文合并以下两份专项审计：

- `BATCH-GPU-PORTABILITY.md`：RTX 5090 D / SM120、B13、S2；
- `4090-optimization-portability.md`：RTX 4090 / SM89、B13、S2。

目标不是把两个当前配置机械拼在一起，而是建立一份统一的优化目录，明确：

1. 哪些收益机制能跨 batch、同架构 GPU 或跨架构保留；
2. 哪些实现因 exact-B13 guard 会立即回退；
3. 哪些具体 tactic 在换 batch、换卡或换流拓扑后需要重选；
4. 如何用较小搜索空间恢复这些收益。

本文只讨论固定 `19x19`，即空间大小 `S=361`。所有 mask 相关结论都以完整
19x19 棋盘为前提，不考虑可变棋盘和任意 mask。

## 1. 当前参考点

| 项目 | SM89 参考点 | SM120 参考点 |
|---|---|---|
| GPU | RTX 4090 | RTX 5090 D |
| compute capability | 8.9 | 12.0 |
| batch | B13 | B13 |
| token/GEMM M | `13*361=4693` | `13*361=4693` |
| precision/layout | FP16 / NHWC | FP16 / NHWC |
| topology | 两个 NN server、两条 CUDA stream，S2 | 同左 |
| 参考配置 | `bench-cuda-gpu0-4090-s2.cfg` | `bench-cuda-gpu2-5090d-s2.cfg` |

4090 文档记录的当前最佳为 3288.9709 nnEval/s，TensorRT B13/S2 基线为
2432.1979 nnEval/s，即相对 TensorRT `+35.226%`。这些数字只是参考点，不应
外推成其他 GPU 或 batch 的预期收益。

## 2. 统一术语：机制、kernel family、tactic 与 policy

迁移时需要把四个层级分开：

1. **优化机制**：删除工作、融合边界、复用数据或减少 launch。例如 residual
   `beta=1`、Q/K RoPE 融合、FFN A-fragment reuse。
2. **kernel family**：实现该机制的算法族。例如 fused dual-GEMM FFN、wide QKV、
   FlashAttention、warp-per-row RMSNorm。
3. **tactic**：某个 kernel family 的具体 tile、stage、warp、swizzle、copy mapping、
   register/shared-memory 配置。
4. **runtime policy**：L2 window、hit ratio、选择哪个 tactic、使用 S1 还是 S2。

通常可移植的是机制；经少量适配后可移植的是 kernel family；最容易失效的是 tactic
和 runtime policy。

“失效”也有两种含义：

- **路径失效**：shape/architecture guard 不满足，安全回退到 cuBLAS、cuDNN 或通用
  CUDA 路径；正确性不受影响，但专用收益消失。
- **性能失效**：kernel 仍能运行，但 tile wave、occupancy、L2 或 S2 相位改变，原来
  的赢家不再是赢家。

## 3. Batch 改动必须区分 actual batch 与 max batch

### 3.1 只改变一次 forward 的 actual batch

例如模型仍以 `nnMaxBatchSize=13` 创建，但某次 forward 使用 B12：

- exact-B13 AOT kernel 的入口 guard 失败并回退；
- 以 max batch 为构造 gate 的 L2 window 仍可能被安装；
- 动态 batch 的 RoPE、RMSNorm、pointwise 和 SM120 FA4 B1-B13 路径继续工作。

### 3.2 同时改变 `nnMaxBatchSize`

模型直接以 B12 创建时：

- exact-B13 AOT 同样回退；
- SM120 当前 trunk/inner persisting-L2 因 `maxBatchSize==13` gate 而关闭；
- SM89 initial-conv B13 frontend plan 也会回退；
- 其他动态 batch kernel 不会仅因 max batch 改变而退出。

因此 tactic key 必须同时记录 actual batch 与 max batch，不能只记录一个“B”。

## 4. 全部可移植优化目录

下表覆盖两份源文档中提到的全部已启用、可移植或可通过小搜索恢复的优化。

| 优化机制 / family | SM89 当前形态 | SM120 当前形态 | 换 batch | 同架构换卡 | 恢复方式 |
|---|---|---|---|---|---|
| exact-19 mask 消除 | mask、attention bias、preprocessing 消除 | 同一语义可直接采用 | 稳定 | 稳定 | 无需搜索 |
| residual `beta=1` | outProj/linear2 通用融合 | 通用 cuBLAS fallback 也保留 | 稳定 | 稳定 | 通常无需搜索 |
| FlashAttention | 固定 B13 FA，含 4090 调度假设 | FA4 both16，当前覆盖 B1-B13 | SM89 回退；SM120 B1-B13 可用 | tactic 敏感 | 少量 attention tile/warp/stage |
| wide QKV / batched QKV | QKV+RoPE AOT epilogue融合 | 固定 B13 wide-QKV AOT | exact AOT 回退 | tactic 敏感 | 2-4 个候选 |
| standalone Q/K RoPE | 动态 batch fused Q/K | fused Q/K + half2 I/O | 稳定 | 较稳定 | 通常不搜，必要时 block/grouping |
| fused FFN | dual GEMM + SwiGLU | fused FFN + SwiGLU AOT | exact AOT 回退 | tactic 敏感 | 3-24 个架构相关候选 |
| FFN A-fragment reuse | 可移植机制，SM89 当前文档未单列部署 | 当前 AOT 已启用，降低重复 `ldmatrix(A)` | 随 AOT 回退 | 机制稳定、tactic 敏感 | 新 FFN 默认保留后验证 |
| standalone SwiGLU | 通用 fallback | C1152 half8 | 动态 | 较稳定 | vector width/block 小搜 |
| linear2 + residual AOT | 固定 B13 | 固定 B13 | 回退到通用 `beta=1` | tactic 高敏感 | 3-24 个架构相关候选 |
| nested preConv AOT | 固定 B13 1x1 AOT | 尚无对应已接受专用项 | SM89 回退 | tactic 敏感 | 有限 GEMM tile 搜索，可移植到 SM120 |
| postConv + 下一层 BN/SiLU | 固定 B13 跨边界融合 | 尚无对应已接受专用项 | SM89 回退 | 融合稳定、tactic 敏感 | 重建 GEMM并小搜 tile/stage |
| QKV epilogue直接 RoPE | SM89 固定 AOT | SM120 当前采用独立 fused RoPE | exact AOT 回退 | 融合稳定、资源敏感 | 可作为 SM120/新 B 的融合候选 |
| RMSNorm | C384 warp-per-row | C384 vec8 | 动态 | 较稳定 | reduction/block 极小搜索 |
| affine + SiLU | C768 vec8 exact-B13 | C384/C768 half2 动态 | SM89 当前回退；SM120 可用 | 向量化稳定 | width `{4,8}`、block `{128,256,512}` |
| initial 3x3 convolution | cuDNN frontend fixed B13 plan | 可采用同类 frontend tactic 搜索 | plan 回退 | engine/knob 高敏感 | heuristic top-K + 少量 knobs |
| fused policy P1 | conversion+bias+BN/SiLU 融合 | 可移植候选/前置机制 | 当前 exact-B13 回退 | 融合稳定、geometry 敏感 | block geometry 小搜 |
| persisting-L2 trunk | trunk window | trunk window | 机制可用，参数改变 | 容量/竞争敏感 | on/off、window、hitRatio |
| persisting-L2 inner | inner window | inner window | 同上 | 同上 | 同上 |
| S1/S2 topology | B13/4090 选择 | B13/5090D 选择 | 相位改变 | 相位改变 | kernel settle 后整图复测 |

`cudaWarmupOnlyMaxBatchSize` 只影响启动时 plan/AOT 暖机范围，不是稳态 forward
优化，不应放入吞吐 tactic 搜索。

## 5. 高稳定性：优先无条件保留的机制

### 5.1 exact-19 mask、bias 与 preprocessing 消除

固定完整 19x19 时，mask 是常量语义，可以直接消除相关 mask、attention bias 和
预处理工作。它不依赖 batch、SM 数、tile 或流数量，是两份文档中最稳定的优化。

### 5.2 residual `beta=1`

让 GEMM epilogue直接完成 residual add，可以删除独立 add kernel、一次中间读写和一次
launch。即使专用 linear2 AOT 因 batch 改变而退出，通用 cuBLAS `beta=1` 仍可保留。

### 5.3 standalone fused Q/K RoPE

合并 Q/K 两条相同变换，并用 half2 进行成对 I/O。grid 可以随 batch 动态扩展，同代
GPU 上通常无需重新选择大规模参数。

### 5.4 RMSNorm 与 pointwise 向量化

- C384 warp-per-row / vec8 RMSNorm；
- C384/C768 affine-SiLU half2 或 vec8；
- C1152 half8 SwiGLU。

这些 family 的通道维固定，batch 主要改变行数，机制可稳定继承。若新 GPU 或小 batch
出现尾波和 utilization 问题，只需搜索 vector width 与 block size。

### 5.5 跨边界工作量消除

以下机制都应保留为可移植 family，即使当前只在 SM89 上完成部署：

- QKV epilogue直接生成旋转后的 Q/K；
- dual GEMM epilogue直接完成 SwiGLU；
- postConv residual 与下一层 BN/SiLU 融合；
- policy P1 的 conversion、global bias 与 BN/SiLU 融合。

它们删除 launch、中间张量或重复访存。迁移时应重选调度参数，而不是退回到永久拆分。

## 6. 机制稳定但 tactic 敏感的项目

### 6.1 Fused FFN

稳定部分：

- 两个输入投影共享 A/input；
- GEMM 与 SwiGLU 融合；
- A-fragment reuse 减少重复 `ldmatrix(A)` 和 register 压力。

敏感部分：

- `M=B*361` 改变后的 CTA 数和尾波；
- M/N/K tile、pipeline stage、swizzle；
- register/shared-memory 对两个 stream 共驻和相位的影响。

新 batch 的 kernel 应默认保留 A-fragment reuse，再对具体 tactic 做整图验证。

### 6.2 Wide QKV / QKV+RoPE

稳定部分是把三次 Q/K/V 投影合并，减少 launch，并尽量直接产出所需布局或旋转结果。
敏感部分是 K tile、stage、shared memory、copy mapping 和 planar epilogue资源。

SM89 的 QKV+RoPE epilogue和 SM120 的 wide-QKV 是同一优化方向的两种架构实现，不能
直接互换二进制，但可共享搜索框架和融合边界。

### 6.3 Linear2、preConv 与其他固定 GEMM AOT

residual fusion 或固定边界是稳定方向，CUTLASS tile/stage 则高度依赖：

- GPU 的 SM 数；
- CTA wave 是否完整；
- resident CTA 限制；
- 与 QKV、out-projection 或 FFN 的并发资源关系。

应保留 AOT family，但不能把当前 B13 tactic 当作跨 batch 或跨卡默认最优解。

### 6.4 FlashAttention / FA4

- SM89 当前实现固定 B13，并含 `num_sm=128` 等 4090 假设；
- SM120 FA4 both16 的接口已覆盖 B1-B13，但超过 B13 仍需重建和验证；
- accumulator 精度、tile、warps、stage、SM 数与尾波都可能影响结果。

FlashAttention family 可移植；SM89 与 SM120 的具体 kernel/tactic 不可直接移植。

### 6.5 Persisting-L2

trunk/inner residual 生命周期和复用机制可移植，硬编码 B13 gate、窗口大小、set-aside 与
hit ratio 不可移植。它尤其依赖每卡 L2 容量和两条 stream 的合计工作集。

### 6.6 cuDNN initial convolution

使用 frontend plan 的方向可移植，固定 engine 和 knobs 不可移植。engine 45 或其他历史
赢家只应作为候选，不应按 engine index 或旧卡结论强制部署。

## 7. GPU 迁移边界

### 7.1 RTX 4090 -> RTX 4080：同为 SM89

现有 SM89 kernel 通常仍能执行，不会仅因设备名称改变而功能性退出。但必须重选或验证：

- 固定 GEMM/attention tactic；
- 以 128 SM 为前提的 FlashAttention 调度；
- L2 set-aside/window/hitRatio；
- cuDNN engine/knobs；
- S1/S2 topology。

RTX 4080 的 SM 数较少，固定 CTA 数会形成不同波数，原来的寄存器和 shared-memory
平衡点也可能改变。

### 7.2 RTX 5090 D -> RTX 5080：同为 SM120

SM120 backend 检查 compute capability 12.0，而不是显卡名称，因此现有 kernel 不会
立即退出。但 FFN、QKV、linear2、L2 和最终 S1/S2 topology 都需要整图重裁决。

### 7.3 SM89 <-> SM120：跨架构

这不是小范围参数搜索就能完成的迁移。需要：

1. 为目标架构提供/编译对应 backend 与 kernel；
2. 适配 Tensor Core 指令、AOT ABI、shared-memory 和 cluster/CTA 能力；
3. 将稳定的融合边界移植到新的 kernel family；
4. 再执行本文的小范围 tactic 搜索；
5. 重建 Nsys/NCU、整图 S2 和精度证据链。

因此应迁移“机制与边界”，不能直接复用跨架构 cubin 或具体 tile 结论。

## 8. 统一的小范围恢复搜索

搜索目标不是穷举，而是为每个 `(architecture class, GPU, batch, streams)` 找到资源
平衡点。历史成功/失败结果用于限定邻域。

### 8.1 SM120 fused FFN

固定：

- C384 -> C1152；
- 19x19；
- FP16 MMA、现有 SwiGLU arithmetic 与输出布局；
- A-fragment reuse。

第一轮搜索：

- M tile：`{64,128}`；
- K tile：`{32,64}`；
- stage：`{2,3}`；
- N tile 先固定64，小 batch 才增加一个邻近值。

每个 batch 约 3-6 个裁剪后候选。

### 8.2 SM89 fused FFN 与固定 GEMM AOT

适用于 dual GEMM+SwiGLU、linear2、preConv、QKV+RoPE、postConv+BN/SiLU：

- threadblock M：`{64,128}`；
- threadblock N：`{64,128}`；
- pipeline stage：`{3,4,5}`；
- swizzle：`{1,2}`。

按 shape、shared-memory 和已知失败组合预裁剪后，每个算子通常保留 12-24 个候选。
FFN 在移植 A-fragment reuse 后，可以再缩小到 SM120 风格的聚焦邻域。

### 8.3 SM120 wide QKV

优先只保留：

- K64 / stage 2：当前 B13 方向；
- K32 / stage 3：较低 shared-memory 方向；
- 小 batch 时增加一个 M64 版本。

保持 planar Q/K/V 输出和无额外 reformat/materialize。每个组合 2-4 个候选。

### 8.4 SM89 QKV+RoPE

保留宽投影与 RoPE epilogue融合，只搜索有限 M/N tile、stage 和 swizzle。若融合版本
资源压力过高，standalone fused Q/K RoPE 是可靠 fallback，而不是退回两条独立 RoPE。

### 8.5 Linear2 + residual

SM120 首轮候选：

- 当前固定版本；
- 128x128x32、stage 3、约49 KiB shared-memory 的历史方向；
- 一个更低 shared-memory 的 N96 邻域版本。

SM89 使用统一固定 GEMM 搜索空间。两种架构都重点观察 CTA waves、resident CTA、
registers/shared memory 和与大 peer kernel 的共驻关系。

### 8.6 Nested preConv 与 postConv+BN/SiLU

这些当前主要来自 SM89。迁移或换 batch 时：

1. 先保留跨边界融合；
2. 为新的 `M=B*361` 生成 GEMM；
3. 使用 linear2 相同的有限 tile/stage 邻域；
4. 对 top 候选直接跑自然全图 S2。

### 8.7 FlashAttention

SM89 第一轮：

- `num_sm` 改为读取设备属性；
- 保持 M64、4 warps；
- 只比较 N64/N96；
- 两者都不理想才增加 stage/warp。

SM120 B1-B13 默认复用当前 FA4 both16。只有 profiler 显示 wave、occupancy 或资源利用
显著恶化时，才增加当前 tile 与一个邻近 tile；B>13 需要重新构建并验证。

### 8.8 Pointwise、RMSNorm 与小型融合

候选限制为：

- vector width：`{4,8}`，half2/half8 作为已有实现点；
- CUDA block：`{128,256,512}`；
- policy P1 只增加少量二维 `block.y`；
- RMSNorm 只比较当前 warp-per-row/vec8 与一个邻近 reduction/block 方案。

这类收益主要来自工作量与 launch 减少，不应做完整 tile 网格。

### 8.9 Persisting-L2

按真实 shape 和活跃 stream 数计算：

```text
trunk_per_stream = B * 361 * 768 * sizeof(half)
inner_per_stream = B * 361 * 384 * sizeof(half)
requested_window = active_streams * sum(enabled per-stream windows)
```

第一轮只测试：

- off；
- trunk-only；
- inner-only；
- trunk+inner。

只对正收益组合搜索 `hitRatio={0.5,0.75,1.0}`。必须同时检查设备实际 grant、L2 hit、
DRAM bytes 和 evict-last sectors。

### 8.10 cuDNN frontend initial convolution

- 获取目标 `(GPU,batch)` 的 heuristic top-K；
- 限制 workspace；
- 对 top-K 和少量 TILE_SIZE/STAGES 等 knobs 做短 profile；
- 按完整 engine tag 选择，不依赖旧的 index；
- plan 不可用或不正确时回到 legacy cuDNN。

### 8.11 S1/S2 topology

流数量是全图 policy，不是单个 kernel 参数。只有主要 kernel/tactic settle 后才扫描 S1/S2。
任何换 batch、换 GPU 或接受新热点 kernel 的操作都可能改变 S2 相位。

## 9. 推荐的低成本裁决流程

对每个新 batch 或同架构新 GPU：

1. 建立通用 fallback 的真实整图基线；
2. 确认 exact-19 和所有稳定机制已启用；
3. 只为固定 AOT、attention、L2 和 cuDNN plan 生成上述小候选集；
4. 先跑 correctness、S1 和 NCU，淘汰明显慢、spill 或资源恶化的版本；
5. 将剩余 top 候选直接放入**自然完整计算图 S2**做 forward/reverse；
6. 不使用 homogeneous/mixed 局部双流 micro 作为门控指标；
7. 对短测正收益版本运行长 ABBA/BAAB；
8. 运行 8192-row 全 head FP32-reference 或等价完整精度回归；
9. 每接受一项后重新跑全图 Nsys 和 ordinal NCU，更新下一热点；
10. kernel 全部 settle 后再扫描最终 S1/S2 topology。

NCU 的严格资源改善和 S1 正收益是保留候选进入整图的理由，但不是跳过整图 S2 的理由。
最终裁决对象始终是自然完整计算图。

## 10. 推荐的 tactic registry

生产部署不应在每次 forward 动态搜索。建议离线生成并缓存：

```text
(compute_capability,
 gpu_class,
 actual_batch,
 max_batch,
 board=19x19,
 streams,
 model_shape)
  -> attention family/tactic
  -> FFN family/tactic
  -> QKV family/tactic
  -> linear2 tactic
  -> preConv/postConv fusion tactic
  -> pointwise geometry
  -> initial-conv cuDNN plan tag
  -> L2 window policy
  -> topology
```

部署规则：

- 热 batch 使用离线验证的固定 AOT 变体；
- 每个 family 只保留 profiler 预筛后的少量候选或最终赢家；
- 启动时查询真实 SM 数、L2、shared-memory 上限和 L2 grant；
- 生产 forward 不做在线大搜索；
- 冷门 batch、未知卡或计划构建失败时使用安全的 cuBLAS/cuDNN/通用 CUDA fallback。

## 11. 最终分类

### 11.1 可跨 batch、同架构 GPU 稳定继承

- exact-19 mask、attention bias 和 preprocessing 消除；
- 通用 GEMM `beta=1` residual 融合；
- standalone fused Q/K RoPE；
- 动态 RMSNorm；
- 动态 affine-SiLU / SwiGLU 向量化；
- 融合边界与工作量消除的设计原则。

### 11.2 换 batch 后专用路径会立即回退，但可小搜索恢复

- SM89 fixed-B13 FlashAttention；
- SM89/SM120 fused FFN；
- wide QKV / QKV+RoPE epilogue；
- linear2 residual AOT；
- SM89 nested preConv；
- SM89 postConv+BN/SiLU；
- SM89 C768 vec8 affine-SiLU 当前 exact-B13 实现；
- SM89 initial-conv frontend plan；
- SM89 fused policy P1；
- 随固定 FFN 一起退出的 A-fragment reuse 实现。

### 11.3 同架构换卡仍可运行，但必须重选或重验

- 所有固定 GEMM tile/stage/swizzle；
- FlashAttention/FA4 tactic；
- cuDNN frontend engine/knobs；
- persisting-L2 policy；
- S1/S2 topology。

### 11.4 跨 SM89/SM120 需要先移植实现，再做小搜索

- fused FFN 与 A-fragment reuse；
- wide QKV / QKV+RoPE；
- linear2/preConv/postConv AOT；
- FlashAttention family；
- policy/head 和 initial-conv 专用路径；
- L2 runtime hooks。

最终应稳定继承的是“删除了什么工作、融合了什么边界、复用了什么数据”；具体 tile、stage、
engine、L2 参数和双流相位都属于目标 `(GPU,batch,streams)` 的离线 tactic，不应被当成
跨平台常量。
