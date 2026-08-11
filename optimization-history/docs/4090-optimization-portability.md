# RTX 4090 优化的 Batch 与 GPU 可移植性

更新时间：2026-08-06 UTC

## 适用范围

本文审计当前 RTX 4090 部署配置中的已启用优化，回答以下问题：

1. 哪些优化在 batch size 从 B13 改为其他值时会直接失效？
2. 哪些优化从 RTX 4090 切换到 RTX 4080 等同代显卡后不再可靠？
3. 哪些收益来自稳定的工作量消除，具有较强可移植性？
4. 对于失效的优化，哪些可以通过小范围参数搜索恢复？

当前部署目标固定为：

- GPU：RTX 4090，SM89
- 棋盘：严格 19x19
- batch：B13
- precision/layout：FP16 NHWC
- 拓扑：两个 NN server、两条独立 CUDA stream，即 S2
- 当前最佳吞吐：3288.9709 nnEval/s
- TensorRT B13/S2 基线：2432.1979 nnEval/s
- 当前 CUDA 相对 TensorRT：+35.226%

部署配置见 `bench-cuda-gpu0-4090-s2.cfg`，完整实验记录见
`results/4090/HISTORY.md`。

本文只统计部署配置中已启用的优化，不包含默认关闭的 strict-local
候选和正在进行的实验。

## 核心结论

- 修改 B13 后，大部分固定 AOT kernel 会因运行时 shape guard 直接回退到
  cuBLAS、cuDNN 或通用 CUDA 实现，主要性能收益随即消失。
- RTX 4090 和 RTX 4080 都是 SM89，因此 4090 到 4080 通常不会造成指令集或
  kernel 功能性失效；真正失效的是以 128 个 SM、4090 L2 和 B13/S2 波形为
  前提选出的调度参数。
- 边界消除、launch 消除、中间张量流量消除和简单向量化是最可迁移的机制。
- 固定 GEMM tile、FlashAttention tile、cuDNN tactic 和 persisting-L2 参数
  需要按新 batch 或新显卡重新验证。
- 4090 到 5090/SM120 不是同代迁移。SM89 AOT kernel 不应直接作为 SM120
  部署路径，需要单独的 SM120 backend 和调度参数。
- 即使局部 NCU 严格更优，也不能无条件保证另一张卡上的 S2 整网吞吐；双流
  相位变化仍需要短正反 S2 测试确认。

## 当前已启用优化的可移植性

| 优化 | 改变 B13 | RTX 4090 -> RTX 4080 | 小范围搜索恢复能力 |
|---|---|---|---|
| 固定 B13 FlashAttention | B13/S361/H12/D32 guard 失败，回退到 cuDNN 或通用 attention | SM89 指令可运行，但当前实现含 `num_sm=128` 和 4090 选择出的 tile，收益不可信 | 高。参数化 B、查询真实 SM 数，搜索 N64/N96、warp/stage |
| dual GEMM + SwiGLU | 固定 `M=B*361`，非 B13 回退到 cuBLAS 加独立 SwiGLU | 可运行，但 CTA 波数和双流重叠改变 | 高。为新 B 生成 AOT，搜索 M/N tile、stage、swizzle |
| linear2 residual AOT | 非 B13 回退到 cuBLAS | 可运行，但 111 CTA 在较少 SM 上形成完全不同的波形 | 高。搜索 M64/M128、N64/N128、stage 3-5 |
| nested preConv AOT | 非 B13 回退到 cuBLAS 1x1 路径 | 固定 stage5 未必仍是赢家 | 高。使用与 linear2 相同的有限 tile 搜索 |
| QKV+RoPE epilogue | 固定 AOT 回退；standalone fused Q/K RoPE 仍可继续生效 | 240 registers 和约 49 KiB shared 的调度可能改变 | 高。保留融合机制，重新搜索 tile/stage |
| standalone fused Q/K RoPE | 通常支持动态 batch | 同代 SM89 上机制较稳定 | 通常无需搜索；必要时只调整 block 或 batch grouping |
| C768 vec8 affine+SiLU | 当前 exact-B13 guard 失败后回退到通用 pointwise kernel | 向量化机制稳定，block geometry 可能需要调整 | 很高。参数化元素总数，搜索 block 128/256/512 |
| postConv + 下一层 BN/SiLU | 固定 B13 GEMM 失败，11 个 BN launch/forward 恢复 | 完整局部边界仍应减少工作，但双流收益需重测 | 高。为新 B 生成 GEMM，仅搜索少量 tile/stage |
| initial 3x3 cuDNN frontend plan | `maxBatchSize==13` guard 失败，回退到 legacy cuDNN | engine 45 可能不再最优或不受支持；代码会尝试 heuristic fallback | 高。搜索 heuristic top-K 和少量 engine knobs |
| fused policy P1 | 非 B13 回退到原来的 conversion、bias、BN/SiLU 三段路径 | 局部融合机制稳定 | 很高。参数化 B，搜索二维 block geometry |
| trunk/inner persisting L2 | 不会直接回退，但工作集大小随 B 线性变化 | 不会功能性失效，但 L2 容量、竞争和最佳 hit ratio 变化 | 中高。搜索 on/off 组合、window 和 hit ratio |
| exact-19 mask/attention-bias 消除 | 不受 batch 影响 | 不受 RTX 4080 影响 | 无需搜索；唯一硬条件是严格 19x19 |
| exact-19 mask preprocessing 消除 | 初始化时按 max batch 创建常量 mask sum，可随 batch 调整 | 与显卡型号基本无关 | 无需搜索；只需保持全棋盘 mask 恒等语义 |

`cudaWarmupOnlyMaxBatchSize` 只缩短 plan/AOT 暖机过程，不是稳态 forward
优化。改变它会影响启动时间和编译范围，但不会直接改变已选 kernel 的稳态性能。

## 改 batch 后会直接失效的路径

以下路径把 B13 或 `Tokens=13*361=4693` 编译进实现，并在入口检查 exact
shape。只修改 `nnMaxBatchSize` 或运行时 batch，不生成新 kernel 时，它们会
立即回退：

- FlashAttention B13/D32
- dual GEMM + SwiGLU
- linear2 residual GEMM
- nested preConv GEMM
- QKV+RoPE epilogue GEMM
- C768 vec8 affine+SiLU
- postConv+下一层 BN/SiLU
- initial-conv frontend B13 plan
- fused policy P1

这类失效通常是“优化路径没有被调用”，不是输出错误。代码保留了通用 fallback，
所以正确性一般仍在，但性能会退回到旧路径。

### 改 batch 后仍可工作的路径

- exact-19 mask 和 attention-bias 消除
- exact-board mask preprocessing 消除
- standalone fused Q/K RoPE
- 通用 wide QKV/FFN batched 调用
- 通用 residual `beta=1` 融合
- C384 warp-per-row RMSNorm
- persisting-L2 策略本身

其中 persisting-L2 虽然不会被 guard 关闭，但收益随工作集大小变化，不能视为
跨 batch 已验证。

## RTX 4090 到 RTX 4080 的影响

RTX 4090 和 RTX 4080 都是 compute capability 8.9。现有 SM89 kernel 在
指令集层面通常仍可执行，因此没有一项仅因设备名称从 4090 变成 4080 就必然
功能性失效。

真正需要重新验证的是调度假设：

- SM 数量减少会改变 CTA 首波填充、尾波浪费和双流并发波形。
- 固定 111 CTA 的 M128xN128 projection 在 4090 上不足一整波，在 4080 上
  则会跨越更多波；原先选择它的理由可能消失。
- register/shared-memory 较高的 QKV、linear2、preConv 和 dual-FFN 会形成
  不同的 resident CTA 数和竞争关系。
- FlashAttention 当前将 `num_sm` 固定为 128，迁移时必须改为运行时设备属性。
- persisting-L2 的有效 set-aside、竞争和最佳 hit ratio 会改变。
- cuDNN engine/tactic 的可用性和最优性必须在目标设备上重新选择。
- S2 相位会随 kernel duration 和波形改变，因此单流或单 kernel 胜出不能代替
  4080 上的完整 S2 结果。

## 最稳定、最可迁移的收益

以下机制严格减少工作量，因而比具体 tile 更有可迁移性：

1. exact-19 mask、attention bias 和 mask preprocessing 消除。
2. wide QKV/FFN batched 调用，减少独立 GEMM launch。
3. outProj/linear2 使用 `beta=1` 直接完成 residual，删除独立 residual-add。
4. C384 warp-per-row RMSNorm。
5. standalone Q/K RoPE 合并。
6. QKV epilogue直接生成旋转后的 Q/K，删除独立 RoPE launch 和中间读写。
7. dual GEMM epilogue直接完成 SwiGLU，删除 standalone SwiGLU 和中间流量。
8. C768 vec8 pointwise 向量化。
9. policy P1 conversion、global bias、BN/SiLU 边界融合。
10. postConv residual 与下一层 BN/SiLU 边界融合。

对这些优化，应把“融合或工作量消除机制”与“当前 4090/B13 tile”分开保存。
迁移时优先保留机制，只重选调度参数。

## 对硬件和拓扑较敏感的优化

以下优化即使仍能运行，也不能假设继续提升整网吞吐：

1. linear2、preConv、QKV、dual-FFN 的固定 CUTLASS tile/stage/swizzle。
2. FlashAttention 的固定 Q/N tile、warps、stage 和 `num_sm`。
3. initial-conv 的固定 cuDNN engine 45 和 knobs。
4. trunk/inner persisting-L2 window、set-aside 和 hit ratio。
5. 所有主要依靠 B13/S2 ABBA，而非严格工作量消除接受的调度选择。

特别是 S2：本项目已经多次观察到局部 NCU 和 S1 严格更优、S2 却因相位变化
而回归。因此跨卡验收仍需要短正反 S2，不宜只跑单 kernel microbenchmark。

## 小范围恢复搜索方案

### 固定 GEMM AOT

先为目标 batch 生成 `M=B*361` 对应的 AOT 变体，再搜索：

- threadblock M：64、128
- threadblock N：64、128
- pipeline stages：3、4、5
- swizzle：1、2

根据算子约束提前排除不合法组合后，每个算子通常只剩 12-24 个候选。使用
NCU/短 S2 预筛，不进行无限 tile 搜索。

适用算子：

- dual GEMM + SwiGLU
- linear2 residual
- preConv
- QKV+RoPE
- postConv+BN/SiLU

### FlashAttention

- batch 改为运行时参数或预编译小批量 B 集合。
- `num_sm` 必须读取目标设备属性。
- 第一轮只搜索 N64/N96，保持当前 M64 和四 warps。
- 只有两个候选都不理想时才增加 stage 或 warp 变量。

### Pointwise/fusion kernels

- vector width：4、8
- CUDA block：128、256、512
- 对二维 policy kernel 搜索少量 `block.y` 候选

这些 kernel 的收益主要来自减少线程、launch 和中间流量，通常不需要复杂搜索。

### Persisting L2

第一轮只测试：

- trunk off/inner off
- trunk on/inner off
- trunk off/inner on
- trunk on/inner on

对正收益组合再搜索 `hitRatio={0.5,0.75,1.0}`。需要用 NCU 确认 L2 hit、
DRAM bytes 和 evict-last sectors，而不是只看整网噪声。

### cuDNN frontend

- 获取目标 batch/GPU 的 heuristic top-K plan。
- 限制 workspace 上限。
- 对 top-K 和少数 TILE_SIZE/STAGES knob 做短 profile。
- 不假设 4090 上的 engine 45 在 4080 或其他 batch 上仍是最优。

## 推荐的多 batch 部署结构

如果只需支持少数热 batch，例如 B8、B10、B13、B16，推荐：

1. 按 `(compute capability, batch, board size, model shape)` 生成固定 AOT 变体。
2. 每个 batch 只保留 profiler 预筛后的 2-3 个候选 tile。
3. 启动时查询真实 SM 数、L2 和 shared-memory 限制。
4. 在目标 S2 拓扑上做很短的 autotune，而不是只测 S1。
5. 缓存最终选择；生产 forward 不进行动态搜索。
6. 保留 cuBLAS/cuDNN/通用 CUDA fallback，覆盖冷门 batch 和未知设备。

仅修改配置而不生成新 AOT kernel，无法恢复硬 B13 路径，因为 shape guard 会在
参数搜索之前直接拒绝候选。

## 迁移验收顺序

每个新 batch 或新 GPU 建议沿用以下短流程：

1. smoke、fallback 和全输出正确性。
2. Nsys 确认候选路径实际命中，并统计 launch 和完整边界。
3. broad NCU 检查 registers、shared memory、waves、spill、L2/DRAM。
4. 使用局部结果裁剪候选参数。
5. 在真实 S2 上跑短 forward/reverse。
6. 仅对稳定正收益候选运行一次短 ABBA。
7. 最终候选运行 8192 行全 head FP32-reference 回归。

这个流程的目标是恢复可迁移的机制收益，同时避免把 4090/B13 的偶然双流相位
当成跨 batch、跨 GPU 的稳定规律。
