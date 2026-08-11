# Stage 39：5090D B13/19x19 下一轮整图优先级交叉审计

日期：2026-08-06 UTC

## 范围和结论

本审计只讨论 RTX 5090 D、固定 B13、固定 19x19、FP16、真实双流整图 S2。
没有修改 KataGo 共享代码，也没有运行 GPU。证据只来自：

- `/workspace/cuda-optimization-history.md` 的 5080 B19/S2 成败历史；
- `/workspace/results/rebuild/HISTORY.md` 的 5090D B13/S2 当前时间线；
- `/workspace/results/rebuild/stage38-post-rope-half2-profile/accepted-fullgraph-ranking.md`
  的最新整图 S2 Nsys 与 344-ordinal S1 NCU 联合归因；
- 上述历史直接引用的已有 5090D 报告，用于核实候选是否已经整图裁决。

明确排除以下已被真实整图否决、或已被更强当前路径取代的实现族：

- linear2 `M128N96/S4`：当前主线真实 S2 整图 `3865.088 -> 3827.442 nn/s`
  （`-0.974%`），4/4 相邻 pair 为负；
- 常规 attention out-projection TileLang AOT：Stage 22 整图 smoke 约 `-2.55%`，
  且 Stage 27 候选 S1 已慢于 cuBLAS；
- 常规 outer contract/expand TileLang/CUTLASS2 AOT：contract 的 S1/S2 本体持续更慢，
  expand 只有噪声级信号；
- FA4 N48/N96、register-Q、M128/256-thread，QKV M64/低 stage/persistent，
  FFN stage1/MB4/persistent 等已有机制级反证的原样重试；
- fixed-S361/tail mask 等 mask 路线。目标固定为完整 19x19，而且该路线已经直接变慢。

排除后，建议的前三项是：

| 排名 | 方向 | 可信的整图收益预算上限 | 物理上限/参考上限 | 成本与风险 |
|---:|---|---:|---:|---|
| 1 | 保留 A-fragment reuse，同时恢复 FFN 线性权重提前预取 | **约 `0.5%--1.5%`** | 仅消除当前 FFN S2 excess 的绝对上限约 `3.1%` | 中高成本；流水依赖、barrier 和寄存器风险高 |
| 2 | wide-QKV 的 async-copy 映射/producer-consumer 流水小搜索 | **约 `0.1%--0.5%`** | 5080 copy-atom 路线 `+0.373%`；按其本体幅度外推约 `0.48%` | 中成本；高调度风险，当前 TileLang 与旧 CuTe atom 不可机械对应 |
| 3 | no-split C384 wide head，fused P1 只作为 stride-aware 前置 | **约 `0.1%--0.35%`** | 三个窄投影全部删除的绝对上限约 `0.54%` | 高接入成本；中高 S2 相位和 scratch 生命周期风险 |

这里的“收益预算上限”是给搜索投入设止损线，不是承诺。按 Stage 37 长跑
`3855.728 nn/s`，一次双流 forward 的墙钟时间约为
`26 / 3855.728 = 6.743 ms`。所有微秒到整图百分比的换算都使用这个量级，且不把
Nsys 的相互重叠 duration 简单相加。

## 1. FFN：A-reuse 之后恢复提前预取

### 为什么仍是第一优先级

Stage 38 中 fused FFN 仍是整图最大工作桶：

- `1479.4 us / stream-forward`，占整图 S2 work `25.94%`；
- S2 excess `203.5 us / stream-forward`，`S2/S1 = 1.158x`；
- 136 registers/thread、32.8 KiB shared memory、1.31 waves/SM；
- achieved occupancy `19.7%`、eligible cycles `23.1%`、tensor throughput
  `43.4%`、wait/issue `3.19`。

当前 H33b 已证明这一 kernel 仍能从指令/调度层取得整图收益：A fragment 复用把
registers/thread 从 `146` 降到 `136`、S1 kernel 提速 `7.321%`，最终真实整图
`3824.934 -> 3859.725 nn/s`（`+0.910%`）。这不是基于 proxy S2 的推断，而是 4/4
相邻 pair 为正的长整图结果。

同时，Stage 33 的源码审计已经指出当前变换的残余代价：gate MMA 被移到下一轮
linear-weight prefetch 之前，延迟了线性权重的提前搬运。当前接受版因此解决了重复
`ldmatrix(A)`，但没有同时保持原先更早的 producer 进度。这是一个尚未经过
5090D B13/19x19 真实整图 S2 裁决的、与 stage1/MB4/persistent 不同的单一机制。

### 收益上限

- 若只把当前 `203.5 us` FFN S2 excess 全部消除，按 6.743 ms forward 换算，整图
  绝对物理上限约为 `203.5 / (6743.2 - 203.5) = 3.11%`。实际单次流水重排不可能
  吃满这个上限。
- 一个能回收约四分之一到二分之一 excess、且不损伤 S1 本体的候选，对应约
  `0.8%--1.6%`；因此本轮搜索预算上限定为 `1.5%`，低于 `0.5%` 则不应无限扩展
  变体。
- H33b 的 `+0.910%` 证明 `0.5%--1.5%` 是合理投入区间，但不得把它再次计入预期。

### 实现成本和风险

- 需要在已生成 AOT kernel 上做精确 schedule 变换，而不是重新扫 tile/stage。
- 可行形态是：复用寄存器中的 A fragment；在 gate MMA 覆盖窗口内尽早发出下一轮
  linear-weight async copy。为了共享缓冲安全，可能需要额外 block barrier；barrier
  成本可能抵消预取收益。
- 编译器可能因同时保持 A/B fragment 或 async-copy 状态而重新增寄存器，破坏 H33b
  的 136-register 优势。
- 即使 S1 更快，也可能像旧 homogeneous proxy 所示出现不同的并发行为；只能由真实
  整图 S2 决定。

### 准确的启动/重新开启条件

只在以下静态条件全部满足时生成候选：

1. 源码/SASS 能证明下一轮 linear-weight copy 比当前接受版更早发出；不是只改源码顺序
   而被编译器重新排回原位。
2. 仍只有一次 non-transposed A `ldmatrix`/fragment/`ki`，不能退回 control 的重复 A load。
3. grid `(18,37,1)`、128 threads、32,768 B dynamic smem 与 FP16 MMA/epilogue 算术不变。
4. ptxas 无 local spill；registers/thread 不高于当前 136。若为了验证调度必须允许更高
   register，需作为单独的资源 trade-off 假设，不能冒充严格改进。
5. deterministic boundary 输出对当前接受 kernel bit-exact。

满足后直接做 S1+NCU 机制确认，再进入当前主线真实 S2 短整图和长 ABBA/BAAB；不得插入
homogeneous/mixed S2 proxy。若 wait/issue、eligible cycles、tensor throughput 均无机制方向，
或真实整图不稳定为正，则关闭“提前预取”这一条，不重开 stage1/MB4/persistent 老路线。

## 2. wide-QKV async-copy/流水轴

### 证据

Stage 38 中 wide QKV 仍是第三大重复 GEMM 工作桶：

- `785.1 us / stream-forward`，work share `13.77%`；
- S2 excess `149.6 us`，`S2/S1 = 1.235x`；
- 136 registers/thread、65.5 KiB smem、1.96 waves/SM；
- achieved occupancy 只有 `8.3%`，eligible cycles `6.9%`，wait/issue `4.89`，
  long-scoreboard/issue `2.09`。

5090D Stage 21 已经搜索并接受了 `M128-N128-K64-S2-T128-MB3` 的固定 wide-QKV，
并取得 `+3.806%` 整图收益；因此不能把“实现 wide QKV”再次列为机会。Stage 21 搜索覆盖
了 tile、K tile、stage、threads/min-blocks，但没有单独裁决旧 5080 路线中的 global-to-shared
copy mapping/copy atom 轴。

5080 在既有 QKV AOT 上把 copy atom `2x2 -> 4x2`，本体
`39.325 -> 37.717 us`（约 `-4.09%`），真实 B19/S2 整图
`2828.845 -> 2839.390 nn/s`（`+0.373%`）。当前 TileLang kernel 已使用 16-byte
`cp.async`，所以不能机械复制“4x2”常量；可迁移的是“改变 warp/thread 到连续 input/weight
transaction 的映射，并检查 producer-consumer 距离”这一机制。

### 收益上限

- 将 5080 的 `4.09%` kernel 降幅施加到当前 `785.1 us` work，约省 `32.1 us/fwd`，
  对应整图约 `0.48%`；因此合理搜索预算上限为 `0.5%`。
- 当前 QKV 全部 `149.6 us` S2 excess 的物理上限约 `2.27%`，但 copy mapping 不可能独自
  消除全部并发 excess，不能用这个数字为大范围 tile 重搜辩护。

### 实现成本和风险

- 需要从当前生成 CUDA/SASS 识别 input/weight 的 16-byte cp.async 合并、sector/request、
  shared bank mapping 和 copy 到 MMA 的等待距离，再只改变一个 copy 映射或 producer 距离。
- 65.5 KiB smem 已使该 kernel 无法与同级大 CTA 随意共驻；任何增加 smem 或 register 的
  “更快 S1”版本都可能放大 S2 干扰。
- 当前 planar Q/K/V epilogue 是 RoPE/FA4 的零重排前提，不能改变输出布局。

### 准确的启动/重新开启条件

1. 先从 Stage 38 代表 ordinal 的 NCU/SASS 证明至少一个具体问题：不完整合并的 global
   sectors、async-copy wait 暴露、shared bank 冲突，或可延长的 producer-consumer 距离。
   只有 `eligible=6.9%` 本身不足以授权盲扫。
2. 只生成 1--3 个保持 `M128-N128-K64-S2-T128-MB3`、128 threads、65,536 B smem、
   planar output 不变的候选；一次只改变 copy mapping 或 prefetch distance。
3. 结果必须 bit-exact、无 spill，registers/thread 不高于 136；NCU 中被假设的 sector/wait
   指标必须按预测改善。
4. 通过后直接进入真实整图 S2；不设置旧式最小收益门槛。若本体变化没有超过其重复测量
   离散度且 NCU 机制指标不动，或长整图在预声明分辨率下没有稳定正信号，则停止该 copy
   轴。不得据此重开已否决的 M64、低 stage 或 persistent QKV。

## 3. no-split wide head；P1 是前置，不是收益主体

### 证据

Stage 38 的三个窄 head 投影分别为：

- policy p1 `8.5 us/fwd`；
- policy g1 `12.2 us/fwd`；
- value v1 `15.8 us/fwd`。

合计约 `36.5 us/fwd`。它们都从同一个 `[4693,768]` trunk-tip tensor 读取，输出通道为
`96 + 96 + 192 = 384`。5080 的最终 no-split wide-head 路线经过长 ABBA 得到
`2855.676 -> 2862.953 nn/s`（`+0.255%`），并通过全 FP32 精度回归。5090D 的
`cudaUseWideHeadProjection` 仍未形成当前 B13/S2 真实整图裁决。

现有 Stage 30 审计已经厘清接入依赖：wide `[4693,768] x [768,384]` 输出不能再切成三个
临时 tensor；p1/g1/v1 的首个消费者必须读取 stride 384 的 slice。fused P1 是最简单的
stride-aware p1 consumer，因此实现顺序是 `fused P1 -> no-split wide head`。这不意味着
P1 本身比 wide head 收益高；P1 单独边界只有约 6 us 量级。

### 收益上限

- 三个当前投影全部消失的不可实现绝对上限为
  `36.5 / (6743.2 - 36.5) = 0.54%`。
- 若 no-split wide GEMM 像已有 B13 实现证据那样把三个投影边界缩短约一半，约省
  `18--20 us`，整图上限约 `0.27%--0.30%`；5080 的 `+0.255%` 与此一致。
- 因此把整个 staged program 的搜索预算上限定为 `0.35%`。P1 单独只预期
  `0.03%--0.15%`，不能因其为前置而无限深入。

### 实现成本和风险

- 需要 Model/head 级协调：合并权重、跨 policy/value 保持一块 C384 scratch 存活，并给
  三类首消费者传递 stride/offset；不是一个通用 ConvLayer hook 能安全完成的局部修改。
- 三个原投影与一个 wide GEMM 的归约/tactic 不同，必须全量 accuracy；不能假定 bit-exact。
- S2 下三次小 GEMM 变一次大 GEMM 会改变 peer 和相位。4090 已有“局部边界明显变快但 S2
  退化”的 BN 反例，因此不接受 S1 外推。
- value-v1 half 输出仍被 ownership 使用；不能只保留 float consumer。

### 准确的启动/重新开启条件

1. 先实现且只测试 exact fused P1：读取普通 stride 96，且接口同时支持未来 stride 384/
   offset 0；一个 kernel 必须完整替代 half-to-float、global-bias add、FP32 affine-SiLU，
   不允许 partial fallback。
2. P1 本地 one-kernel boundary 应低于约 `3.5 us`，正确性保持当前操作次序。P1 即使只有
   小的真实整图正信号也可保留，因为它是 wide-head 必需 plumbing；若 S2 明确退化则停止
   整个当前设计，不能把负收益前置硬塞进 bundle。
3. wide head 必须一次直接产生 `[4693,384]`，首消费者直接读 slice；若需要 materialize/
   split 三块输出，则路线失去机制基础，应直接关闭。
4. 完整 wide projection + first-consumer 本地边界必须在正反 profile 顺序中都比三个窄路径
   明显更短，无 spill，再直接做当前主线 S2 ABBA/BAAB 和 8192-row replay。
5. standalone BN-to-float 不重开。只有 wide head 已被 S2 接受后，才允许一次
   wide+BN-to-float 的独立候选；任何顺序敏感或负向结果立即关闭。

## 未进入前三的候补和暂缓项

### Initial convolution engine 47：最确定的低上限候补

Stage 30 已在准确 B13 shape 上枚举 36 个 unique cuDNN frontend plan；engine 47 将 event
boundary `29.150 -> 16.172 us`，Nsys kernel boundary 约降低 `6 us`，smem
`81.92 -> 4.10 KiB`，occupancy `8.32% -> 26.07%`。它尚未接入真实 S2 整图。

但 Stage 38 当前 frontend initial-conv 只有 `24.2 us/fwd` work、`4.7 us` excess；按已观测
kernel saving 换算约 `0.09%`，连 event-boundary 极限也约 `0.19%`。因此它是 QKV copy
轴没有 NCU 支持时最合适的替补，而不是高于前三的主路线。重新开启条件已经很精确：按
完整 tag 查询 engine 47、找不到即 fail closed，零 workspace、先整图 S2 再 full replay。

### Initial-global fusion：上限约 `0.05%--0.15%`

Stage 38 的 initial-global matmul 和 broadcast-add 只约 `5.0 + 10.1 us/fwd`；5080 保 FP32
dot 的版本整图约 `+0.139%`，但 4090 B13/S2 曾 `-1.187%`。只有在 engine 47 或别的
frontend 路线形成可融合 epilogue 时才值得重新打开更大边界；原样独立 kernel 的 S2
整图若失败，不再扫更多 rows-per-CTA。

### 显式双流 trunk 分相：理论空间大，但目前不满足启动条件

5080 历史只留下未实现架构方向。5090D Stage 25 的 phase-offset sweep 能改变 collision
分布，但 forward/reverse 顺序没有共同稳定最优 offset，不能作为生产收益证据。只有在当前
mainline 上新的正反 phase sweep 找到同一稳定窗口、相对自然相位两种顺序都超过约 `0.5%`，
且 Nsys 证明它确实避开 linear2/outproj 等大资源 collision 后，才值得设计 trunk-entry gate。
在此之前，它是诊断工具，不是前三实现任务。

### 普通权重共享：不值得当前投入

5080 仅 `+0.126%`，4090 整图 `-0.716%`，5090D 还有 96 MiB L2。当前没有 5090D 真实整图
裁决，但生命周期/所有权成本高，可信收益上限低于 `0.15%`；放在上述候补之后。

## 建议的下一步决策

1. **先做 FFN prefetch-preserving A-reuse 的静态/SASS 可行性检查。**若无法同时满足
   “更早 copy、136 regs、不增加 barrier 关键路径”，不要为了最大热点盲写大量变体。
2. 若 FFN 静态条件不成立，检查 QKV 代表 ordinal 的 NCU/SASS 是否给出明确 copy 问题；
   有证据则做 1--3 个小候选，没有证据则立即跳到 initial-conv engine 47。
3. no-split wide head 作为独立 staged program 排在其后；P1 只做到足以支撑 wide stride，
   不把小前置误当作主要收益目标。
4. 每个被接受的变化都按当前工作流：真实整图 S2 长 A/B、8192-row full replay、随后重新跑
   全图 S2/S1 Nsys 与 344-ordinal S1 NCU，再重排下一项。

## 证据索引

- 5080 完整历史：`/workspace/cuda-optimization-history.md`
- 5090D 当前历史：`/workspace/results/rebuild/HISTORY.md`
- Stage 38 最新排名：
  `/workspace/results/rebuild/stage38-post-rope-half2-profile/accepted-fullgraph-ranking.md`
- A-reuse 接受报告：
  `/workspace/results/rebuild/stage33-fused-ffn-a-reuse/report-h33b-accepted.md`
- Stage 21 wide-QKV：`/workspace/results/rebuild/stage21/REPORT.md`
- head 审计：
  `/workspace/results/rebuild/stage30-head-audit/fixed-b13-head-operator-audit.md`
- initial-conv plan 搜索：`/workspace/results/rebuild/stage30-initial-conv/report.md`
- initial-global 审计：
  `/workspace/results/rebuild/stage28/initial-global-matmul-add-audit.md`
- 旧 proxy gate 重审：
  `/workspace/results/rebuild/stage34-resource-positive-reaudit/initial-audit.md`
