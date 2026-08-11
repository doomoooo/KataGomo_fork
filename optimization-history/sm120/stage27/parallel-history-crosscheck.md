# B13/S2 5090D 与 5080 优化历史交叉审计

日期：2026-08-06

## 范围与结论

本审计只讨论 5090D、固定 `B13`、固定 `19x19`、`S2`。固定空间行数为
`M = 13 * 361 = 4693`。按要求排除所有 mask 优化；当前 trace 中 mask 相关操作的
S2 excess 也只有约 `0.057 ms / 60 forwards`，不影响以下排序。

审计依据是：

- `/workspace/SKILL.md` 的 baseline -> accuracy -> Nsys -> NCU -> 单假设 -> micro + 整网
  -> full accuracy 证据链；
- `/workspace/cuda-optimization-history.md` 的 5080 已接受/否决历史；
- `/workspace/results/rebuild/5080-CROSSCHECK.md`；
- 当前实现和 git diff；
- Stage 20--26 的 5090D 实验；
- `/workspace/results/rebuild/stage27/current-s2-ordinal-attribution.md` 的当前 344 个固定
  forward ordinal 归因。

冻结快照为 `/workspace/katago` HEAD
`090caa6115c2ae86a75839d1b4fddeacd23d7444` 加当前 dirty rebuild 改动，模型 SHA256
`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`。同源
TensorRT B13/S2 基线为 `3260.834 nn/s`；当前已接受到 Stage 22
linear2 AOT，长 ABBA control/candidate 均值分别为 `3809.428 / 3830.328 nn/s`
（该项 `+0.549%`）。本审计的机会排序以这个已接受状态和 Stage 27 trace 为起点，
而不是以 5080 的 `2862.953 nn/s` 终点作为可直接缩放的 baseline。

结论不是“核心算子已穷尽”。当前最大的两个未闭环区域仍是通用 `library_gemm` 中的：

1. 33 次 attention out-projection + residual；
2. 11 对 outer `C768 -> C384 -> C768` projection。

它们合计贡献当前 `library_gemm` 桶 `29.420 ms / 60 forwards` S2 excess 中的
`28.503 ms`，即约 `96.9%`。因此先拆 ordinal 再定向优化是可解释的；继续把整个
`library_gemm` 当一个桶搜索则不是。

下表中的“预期整网机会”是决定搜索预算的区间，不是承诺，也不能相加。5080 结果是
`B19/S2` 且硬件不同，只能证明路线曾经成立；区间同时受当前 B13/S2 hotspot 上界和
5090D 已有反证约束。

## 优先级

| 排名 | 优化项 | 5090D 状态 | 可解释的预期整网机会 | 实施成本 | 风险 |
|---:|---|---|---:|---|---|
| 1 | Attention out-projection + residual 的 S2 资源平衡 AOT | 只浅试过一个错误资源点 | `0.3%--1.2%` | 中 | 高：micro 与整网方向可相反 |
| 2 | Outer `768->384` contract 与 `384->768+residual` 固定 AOT | 未实施 | `0.4%--1.2%` | 中高 | 中高：两种 shape、双流配对 |
| 3 | RMSNorm exact-tree vec8/warp 调度 | 只复现了 half2/一 warp版本 | `0.2%--0.8%` | 低中 | 中：必须保持归约树与精度 |
| 4 | trunk/inner persisting-L2 window | 配置存在但运行时未实施 | `0.2%--0.8%` | 中 | 中：容量和 S2 驱逐行为 |
| 5 | C384/C768 affine-SiLU flat vec8 | 只复现 half2 | `0.1%--0.3%` | 低 | 低中 |
| 6 | policy/value 小 head 三项 | 配置存在但运行时未实施 | 合计 `0.2%--0.6%` | 中 | 中：FP16/FP32 边界 |
| 7 | B13 初始卷积固定 cuDNN frontend tactic | 配置存在但运行时未实施 | `0.05%--0.25%` | 中 | 低；tactic 不可跨卡照搬 |
| 8 | Wide-QKV copy atom / pipeline 几何再搜索 | 已有 AOT，但未复现同一调度维度 | `0%--0.4%` | 中 | 高：可能破坏 S2 平衡 |
| 9 | initial global matmul + broadcast-add FP32-dot 融合 | 配置存在但运行时未实施 | `0.05%--0.15%` | 中 | 中：数值次序 |
| 10 | 双 server 共享普通 GEMM 权重 | 配置存在但运行时未实施 | `0%--0.15%` | 高 | 中：所有权和生命周期 |

## 1. Attention out-projection：最高优先级，但不能重放 Stage 22 tile

**当前 B13/S2 证据**

- 逻辑 ordinal 为 `15, 23, 31, ... , 311`，每 forward 33 次。
- 固定 shape：`M4693, N384, K384, beta=1 residual`。
- 当前走通用 cuBLAS `Kernel2`：`grid=148`、`block=128`、`164 regs/thread`、
  `81920 B smem/CTA`。
- 当前 60 forwards：isolated reference `17.074 ms`，S2 total `35.735 ms`，
  `2.093x`，excess `18.661 ms`。这是尚未专用化的最大逻辑桶。
- 它最常与另一条流的 `library_gemm` 重叠；遇到 fused-FFN 时单 ordinal 可达约
  `2.5x` slowdown。瓶颈因此是并发资源形状，不只是 isolated latency。

**5080 证据**

- 固定 out-projection AOT 在旧锁频 S2 路径中从 `2584.370` 到 `2615.048 nn/s`，
  `+1.187%`，8192 行精度通过。来源：
  `/data/wangyize/katago/results/cuda-both16-linear2-outproj-aot-b19-s2-paired-reverse.json`，
  汇总见 `/workspace/cuda-optimization-history.md`。

**5090D 已有反证及下一步**

- Stage 22 TileLang `M128-N128-K32-S4-T128` 在 isolated/dual micro 都比 control 快，
  但整网 smoke 从 `3815.642` 降到 `3718.246 nn/s`，`-2.55%`。文件：
  `stage22/smoke-control-n128s4-build.json` 与 `stage22/smoke-outproj.json`。
- 这不是“outproj 无空间”，而是证明该大 tile 在真实相位下改变了邻居资源竞争。
- 下一轮先对当前 cuBLAS ordinal 和候选做 NCU：Tensor/DRAM SOL、active warps、
  eligible warps、寄存器/SMEM 限制、CTA residency；然后只扫一个资源轴，例如
  `N64/N128` 或 stage 数。验收必须是 B13/S2 ABBA/BAAB 整网，不能以 dual micro
  代替。

## 2. Outer projection pair：5080 的完整收益尚未在 5090D 复现

**当前 B13/S2 证据**

- contract：11 次/forward，ordinals `10, 38, 66, ... , 290`，shape
  `M4693,N384,K768`。当前 `Kernel2` 为 `grid=148`、`block=128`、`164 regs`、
  `81920 B smem`。S2 total `11.614 ms / 60 forwards`，excess `3.678 ms`。
- expand + residual：11 次/forward，ordinals `36, 64, ... , 316`，shape
  `M4693,N768,K384,beta=1`。当前 `Kernel2` 为 `grid=148`、`block=256`、
  `154 regs`、`73728 B smem`。S2 total `15.768 ms`，excess `6.164 ms`。
- 两者合计 `9.842 ms` excess，且 expand 的末段 ordinals 是当前 top hotspot。

**5080 证据**

- 固定 CUTLASS contract/expand AOT：`2783.711 -> 2817.070 nn/s`，`+1.198%`，
  all-head+p0loss 一致。
- expand warp shape `64x32` 又有经长 ABBA 验证的 `+0.078%`。
- 来源：`cuda-current-outer-projection-aot-b19-s2-locked2430-pl380.json` 和
  `cuda-outer-warp64x32-long-abba-summary.json`，汇总见优化历史。

**实现缺口与建议**

- `Sm120Options::useOuterProjectionAot` 虽由
  `cpp/neuralnet/cudabackend_sm120.cpp:273` 解析且默认 true，但没有任何消费点；这是
  dead option，不是已实施功能。
- 先分别建立两个固定 B13 micro，保持一次只改一个 shape。优先 expand，因为当前
  excess 更大；通过整网后再叠加 contract。不要同时融合 SiLU：5080 已测
  `-2.057%`。

## 3. RMSNorm：旧 vec8 调度只被部分复现

**当前实现与热点**

- `cpp/neuralnet/cudabackend_sm120_kernels.cu:48` 的
  `rmsNorm384Half2Kernel` 使用 128-thread CTA、每 warp 一行、每 lane 六次 half2 load，
  四行/CTA；资源签名 `38 regs`、无动态 smem、`grid=1174`。
- 66 次/forward。当前 S2 total `19.305 ms / 60 forwards`，median `4.608 us`，
  `1.719x`，excess `8.615 ms`，并且 peer overlap 为 100%。
- Stage 13 的这条一-warp exact reduction 已给 5090D 整网 `+2.172%`；所以不能把
  5080 的全部 RMS 收益再次计入。

**5080 尚未复现的部分**

- 旧 accepted kernel 是每 CTA 四个 warp/四行，但每 lane 通过
  `uint4 + uint2` 完成 vec8/12-value 加载；锁频 checkpoint
  `2619.734 -> 2725.906 nn/s`，`+4.053%`。
- 迁移价值在 vector memory schedule，不在“每 warp 一行”这一点，后者当前已有。

**建议**

- 先用 NCU 比较 global load 指令数、sector/request、L1/L2 throughput、issue stalls；
  若 memory instruction pressure 确认存在，再只替换 load/store vectorization，保持
  当前六组加法和 shuffle 归约树不变。
- 不能重开 480-thread/激进 row schedule：5080 vec8 和 row480 整网分别降至
  `2731.307` 和 `2728.839 nn/s`。

## 4. Persisting L2：配置是空壳，且 B13/S2 容量条件成立

- `cudaUsePersistingL2Trunk` / `cudaUsePersistingL2Inner` 在
  `cpp/neuralnet/cudabackend_sm120.h:230` 附近定义并由 cpp 解析，但没有运行时消费点。
  `cudabackend_sm120.h:595` 仍明确 TODO per-GPU cache/persisting-L2 window。
- 当前 5090D trace 报告 L2 为 `100663296 B`（96 MiB）。固定 B13 的单流 C768
  residual 约 `4693*768*2 = 7.21 MB`，C384 inner 约 `3.60 MB`；两个流的这些窗口
  在名义容量上可容纳，但还需考虑权重和其他 working set。
- 5080 旧结果：trunk 24M window `2731.790 -> 2740.127`，`+0.305%`；inner window
  `2757.380 -> 2779.610`，`+0.806%`。它们只改变 cache policy，算术不变。
- 建议先单独实施 trunk window，记录 L2 hit rate/sector 和 S2 ABBA；inner 是第二个
  独立变量。不要一次同时打开两个 dead option，否则无法归因。

## 5. Affine-SiLU：half2 已有，flat vec8 尚无

- 当前 `affineSiluHalf2Kernel` 位于
  `cpp/neuralnet/cudabackend_sm120_kernels.cu:492`：每 thread 一个 half2；C384 为
  `block=192`，C768 为 `block=384`，均每行一个 CTA。
- 23 次/forward（11 个 outer pre C768、11 个 outer post C384、1 个 trunk tip）。
  S2 total `8.672 ms / 60 forwards`，`1.330x`，excess `2.757 ms`。
- 5080 accepted 版本是 flat `block=256`、每 thread vec8/half8 group；NCU
  `15.840 -> 14.208 us`，在已经接受 RMS 后整网 `2725.906 -> 2731.790`，
  `+0.216%`。
- 因为当前 Stage 13 half2 已在 5090D 带来 `+0.931%`，只应搜索 vec8 的增量。
  分 C768 和 C384 两次提交，优先 excess 更大的 outer-pre C768。

## 6. 小型 heads：三条曾验证路线当前都是 dead options

以下选项只有定义和配置解析，没有运行时消费：`useFusedPolicyP1`、
`useHeadBNHalfToFloat`、`useWideHeadProjection`。当前 fixed-forward ordinals 318--343
仍显示拆分的 copy、norm、bias、pool 和通用 GEMM。

5080 的独立证据为：

| 路线 | 5080 S2 整网证据 | 当前应保持的语义 |
|---|---:|---|
| fused policy P1 小图 | 正序 `+0.191%`，逆序 `+0.169%` | 不改变归一化/激活次序 |
| head BN 直接 FP16 输入、FP32 输出 | 正序 `+0.233%`，逆序 `+0.162%` | 保持最终 FP32 边界 |
| wide head no-split projection | `2855.676 -> 2862.953`，`+0.255%` | 不增加中间 split/reformat |

三项必须分别做 B13/S2 ABBA 和 full accuracy，再逐项累加。不要重开 fused head
pooling：其正序 `+0.211%` 在逆序只剩 `+0.003%`，已经判定为顺序漂移。

## 7. Initial convolution frontend tactic：路线有效，常量不可迁移

- 当前 ordinal 5 为通用 cuDNN `Kernel`：`grid=296x3`、`block=128`、`94 regs`、
  `81920 B smem`；isolated `19.569 us`，S2 `21.697 us`，excess 只有
  `0.131 ms / 60 forwards`。
- `useInitialConvFrontend` 仅在 `cudabackend_sm120.cpp:277` 解析，没有消费点。
- 5080 的固定 B19 frontend engine 45/tile0/stage2 为 `36.417 us`，整网
  `2835.976 -> 2849.036`，`+0.461%`，bit-exact。
- engine 45 是 B19/5080 的 tactic 标识，不能直接写死到 B13/5090D。应在当前固定
  shape 上枚举 frontend plans，再以 S2 整网选择；当前 hotspot 上界也说明预期收益
  应低于旧百分比。

## 8. Wide-QKV：是调度维度缺口，不是功能缺口

- 当前已有 Stage 21 fixed B13 wide-QKV TileLang AOT，不能把 5080 的 QKV 收益当作
  “尚未实施”。当前 33 次/forward，S2 excess `12.955 ms / 60 forwards`，但主体已
  经较深 tile/stage 搜索。
- 5080 在既有 CuTe kernel 上把 copy atom `2x2 -> 4x2`，micro
  `39.325 -> 37.717 us`，整网 `2828.845 -> 2839.390`，`+0.373%`。
- 当前 AOT 的 copy/pipeline 实现不同，不能机械移植 atom 常量；只可把“全局加载
  transaction/copy atom 几何”作为一个尚未单独扫描的轴。先 NCU，只有在 load
  efficiency 或 issue stalls 支持时才开小范围搜索。
- 不要重开 M64、低 stage、persistent QKV：5080 和当前 Stage 21/26 都已有反证。

## 9. Initial global matmul + broadcast-add：小而完整的边界子图

- 当前 `cpp/neuralnet/cudabackend.cpp:2738` 先调用通用 `initialMatMul->apply`，随后
  broadcast-add。ordinals 6--8 分别是 matmul、split-K reduce、broadcast-add；
  isolated 约 `2.624 + 1.280 + 7.729 us`。
- `useInitialGlobalMatMulAdd` 只有定义/解析，没有消费点。
- 5080 保留 FP32 dot 的融合版本 `2857.765 -> 2861.736`，约 `+0.139%`，并通过
  p0loss/all-head 门槛。第一版更快但改变数值，不能采用。
- 当前实现应针对逻辑 shape `[B13, global_features] -> [B13,C768]` 后广播到
  `[4693,C768]`，保持 FP32 dot；不要把 global feature K 值未经模型描述核实就写死。

## 10. 普通 GEMM 权重共享：有证据，但当前性价比最低

- `shareModelWeights`、`shareWideQKVWeights`、`shareOuterProjectionWeights` 均只有
  option/parser，没有 per-GPU shared cache；代码 TODO 也确认 device weight ownership
  尚未实现。
- 5080 仅共享普通 matmul 权重的 ABBA：`2818.446 -> 2822.008`，`+0.126%`。
  再共享 wide-QKV/outer 特殊权重只有 `+0.028%`，无稳定信号并被拒绝。
- 5090D 有更大的 96 MiB L2，重复权重造成的相对压力可能更小；同时跨 server cache
  需要严格的 device、model hash、allocation lifetime 和 teardown 所有权。故放在
  kernel/L2 window 之后，只考虑普通 GEMM 权重。

## 明确不应重复搜索的路线

以下不是遗漏，而是已有当前或 5080 S2 反证：

- CUDA Graph；
- QKV + RoPE 单 AOT 融合；
- RMS scale 折叠进 QKV/FFN；
- DSM/cluster launch；
- 激进 RMS row480；
- outer fused-SiLU 和 projection swizzle8；
- share-all weights；
- two-way RoPE；
- QKV M64、低 stage、persistent；
- fused head pooling；
- FA4 N48/N96、register-Q、M128/256-thread；
- FFN stage1/MB4/persistent；
- Stage 26 balanced linear2（整网 `-2.305%`）；
- Stage 24 通用 cuBLASLt（只做 S1 且 `-0.103%`），尤其不能把它误认为 beta=1
  out-projection 的实验。

RoPE half2 在 5080 曾有 `+0.630%`，但当前 B13/5090D 已直接测试：短 S2 只有
`+0.443%`、低于接受阈值，direct micro 反而约 `-7.6%`。因此它是“当前暂停、可在
邻居布局改变后复测”，不是未实施首选；B19 的静态 unroll19 与 B13 目标无关。

## 建议执行顺序与停止条件

1. 先做 outproj 当前/候选 NCU，建立“资源签名 -> peer -> 整网变化”的解释链；然后
   只扫一个 tile/resource 轴。
2. 独立实现 outer expand AOT，通过 B13/S2 ABBA + accuracy 后再做 contract。
3. 同时可用低成本 CPU/AOT 分支准备 RMS exact-tree vec8 与 affine C768 vec8，但每次
   benchmark 只接入一个变量。
4. 再做 trunk L2 window、inner L2 window；两者分开验证。
5. 最后处理 heads、initial conv/global 和普通权重共享。

每个候选都应保存：固定配置、binary SHA、micro isolated + dual、自然 S2 Nsys、必要的
NCU、长 ABBA/BAAB 和完整 8192 行 all-head+p0loss 精度。只有当上述高优先级项均完成
资源导向的小范围搜索，并且剩余 ordinal 的 S2 union/exclusive 上界低于测量噪声，才可
声称“找不到优化空间”。当前显然还没有达到这个停止条件。

## 证据索引

- 当前 ordinal/S2 interference：
  `/workspace/results/rebuild/stage27/current-s2-ordinal-attribution.md`
- 5080 迁移状态旧审计：`/workspace/results/rebuild/5080-CROSSCHECK.md`
- 5080 全历史：`/workspace/cuda-optimization-history.md`
- 5090D 重建时间线：`/workspace/results/rebuild/HISTORY.md`
- outproj/linear2 假设与反证：
  `/workspace/results/rebuild/stage22/hypothesis-h22-residual-projection-aot.md`、
  `/workspace/results/rebuild/stage22/`
- 当前 S2 family interference：
  `/workspace/results/rebuild/stage24/current-s2-interference.md`
- balanced linear2 反证：`/workspace/results/rebuild/stage26/`
- 当前实现：`/workspace/katago/cpp/neuralnet/cudabackend_sm120.{h,cpp}`、
  `/workspace/katago/cpp/neuralnet/cudabackend_sm120_kernels.cu`、
  `/workspace/katago/cpp/neuralnet/cudabackend.cpp`
