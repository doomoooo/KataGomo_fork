# KataGo 1.17.1 CUDA/SM89 4090 优化历史

整理时间：2026-08-06（UTC）

范围：RTX 4090（SM89）、KataGo 1.17.1、模型
`b11c768h12nbt3tflrs-fson-silu.bin.gz`、CUDA 13.2.86、cuDNN 9.25.0、
独立 worktree `/workspace/katago-4090`（分支 `4090-opt`）。

基线制度（复用 `/workspace/results/baseline-fixed/HISTORY.md`）：
S2-CUDA-FINAL：2 个 NN server 各自私有 CUDA stream，B13/19x19/FP16，
300 次 timed forward / 10 次 warmup，未锁频，gpu-lock。
基线 CUDA 4090 中位数 `1876.270 nnEval/s`；本 worktree 的官方复测
（100 次迭代）`1885.957 nnEval/s`。

当前优化目标（stage-31 起重新冻结）：仅 exact 19x19/B13/FP16/S2。
S1 历史结果只保留为旧实验记录，不再指导候选选择或接受决定。

## 测量制度

| 标记 | 条件 |
|---|---|
| L0 | 未锁频；S2/B13；纯设备 `benchmarknn`；CUDA 官方路径（aac3a3d） |
| LT1 | 未锁频；S2/B13；`cudaUseMatmulLt=true`；其余同 L0 |
| L1-EXACT19-B13-S2 | 未锁频；固定 19x19；每个 server 仅 B13；S2 独立 stream；FP16；10 warmup/300 timed；gpu-lock；`requireMaxBoardSize=true`；只预热 B13 graph |
| L1K-EXACT19-B13-S2 | 同 L1，但锁定 SM 2400MHz、候选/控制分别预热；500 timed；三轮独立正反序 ABBA |
| L1S-EXACT19-B13-S2 | 同 L1，但锁定 SM 2400MHz；NCU 最多 4 个目标样本、正反 Nsys 各 20 timed、单轮 300 timed 正反 ABBA；只作快速迭代筛选 |
| L1S1-EXACT19-B13 | 锁定 SM 2400MHz；S1/B13/19x19/FP16；NCU 每个几何最多 2 个目标样本、正反 Nsys 各 20 timed、单轮 100 timed 正反 ABBA；S1 主迭代制度 |
| L1Q-EXACT19-B13-S2 | 锁定 SM 2400MHz；S2/B13/19x19/FP16；NCU 2-3 个目标样本、正反 Nsys 各 20 timed；仅短门均通过才跑一次 100-timed ABBA，性能通过后才跑 8192 accuracy |
| GTP-EAGER-2205 | 真实 event-gated GTP；exact-19/B13/S2/t96；请求 SM/显存 2205/10501MHz，遥测实际 2205/10251MHz；common-wall physical launch rows；graph/eager ABBA |

## 主时间线

| 结果保存时间（UTC） | 优化或阶段 | 结论 | 制度 | 单 kernel / 子图耗时 | 端到端吞吐（nnEval/s） | 精度证据 |
|---|---|---|---|---:|---:|---|
| 2026-08-05 13:19:00 | worktree 官方 baseline 复测 | 起点 | L0 | NCU dominant GEMM 10.4us | 1885.957 | 复用 baseline-fixed ACC |
| 2026-08-05 13:35:00 | cuBLASLt FP16 GEMM（fallback 内原型） | 否决：64x64 kernel 仍 8.3% occupancy、grid 36/108，整网反而变慢 | LT1 | 待测 | 1487.267（官方 1885.957） | 未做 ACC（性能已否决） |
| 2026-08-05 13:36:00 | wide QKV（fallback 内原型，布局未定稿） | 暂存：ABBA 控制 1870 vs 候选 1935（+3.5%），但需移入 sm89 后重验 | L0 | 待测 | 1935.00 | 初版布局错误产生垃圾输出；已回滚，不作为接受证据 |
| 2026-08-05 13:45:00 | SM89 隔离架构 stage-0 | 保留：cudabackend_sm89.h/cpp + 薄 dispatch，官方 fallback 数值不变 | L0 | 未记录 | 1930.993（stage-0 复测） | 与官方路径相同 |
| 2026-08-05 14:00:00 | SM89 独立 forward + cuDNN SDPA（stage-1） | 保留：cudabackend_sm89_forward 自包含 forward，数值正确，吞吐回到官方水平 | L0 | 未记录 | 1909.245（100 iter） | policy top-1 99.7559%，p0loss 1.591556 vs ref 1.591528 |
| 2026-08-05 14:15:00 | wide QKV + wide FFN（cublasHgemmStridedBatched） | 保留：一次 strided-batched 调出三块连续 QKV / 两块连续 FFN，ABBA 控制 1882 vs 候选 1972（+4.8%） | L0 | 未记录 | A1 1879.673 / B1 1942.107 / B2 2001.981 / A2 1884.207 | 8192 行全部 head + p0loss 通过；p0loss 1.591556 |
| 2026-08-05 14:30:00 | 融合 outProj/linear2 residual（beta=1 + mask zero） | 保留：ABBA 控制 1948 vs 候选 1987（+2.0%）；新 sm89 kernel 单独文件 | L0 | 未记录 | A1 1941.105 / B1 2000.325 / B2 1973.545 / A2 1955.118 | 8192 行全部 head + p0loss 通过；p0loss 1.591556 |
| 2026-08-05 14:50:00 | SM89 warp-per-row RMSNorm（C384） | 保留：ABBA 控制 2005 vs 候选 2042（+1.9%）；数值通过全 head 回归 | L0 | 未记录 | A1 2005.697 / B1 2055.214 / B2 2029.302 / A2 2004.510 | p0loss 1.591540，policy top-1 99.7559% |
| 2026-08-05 15:14:33 | exact 19x19 恒等 mask / 零 attention-bias 消除（stage-6） | 保留：同一 binary 的 ABBA+反向顺序控制中位数 2049.535、候选 2440.351（+19.07%）；仅适用于严格 19x19 | L1-EXACT19-B13-S2 | mask-zero 与 attention-bias kernel 均 0 次；SDPA 仍为下一热点 | 2049.535 -> 2440.351 | 8192 行全部 head + p0loss 对 FP32 reference 通过；policy top-1 99.7559%，p0loss 1.591540 |
| 2026-08-05 15:49:35 | 固定 B13 FlashAttention D32/M64xN96/W4/no-pack（stage-7） | 保留：同一 binary 的 ABBA+反向顺序均为正；合并控制中位数 2440.297、候选 2682.562（+9.93%） | L1-EXACT19-B13-S2 | cuDNN 双流 Nsys 平均 60.13us -> Flash 26.83us；attention exclusive critical path 60.613ms -> 34.231ms；最终 NCU 28.48us、22.44% occupancy、0 spill | 2440.297 -> 2682.562 | 8192 行全部 head + p0loss 对 FP32 reference 通过；policy top-1 99.7803%，p0loss 1.591574 |
| 2026-08-05 16:23:27 | 固定 B13 CUTLASS dual GEMM + SwiGLU，M128xN64xK32/W64x32/stage3/swizzle2（stage-8） | 保留：300 次正反 ABBA 均为正；合并控制中位数 2647.477、候选 2951.138（+11.47%） | L1-EXACT19-B13-S2 | 隔离完整边界 53.35us -> 37.51us；双流 Nsys FFN summed 127.643ms -> 82.599ms，union 283.620ms -> 264.711ms | 2647.477 -> 2951.138 | 8192 行全部 head 对 FP32 reference 通过；policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-05 16:58:03 | 固定 B13 CUTLASS linear2 residual GEMM，M128xN128xK32/W64x64/stage4/swizzle1（stage-9） | 保留：真实 `beta=1` 边界；300 次正反 ABBA 四个配对均为正；合并中位数 +2.38% | L1-EXACT19-B13-S2 | 双流 micro pair 36.70us -> 32.73us；最后 30 forward 双流 Nsys union 258.932ms -> 252.386ms | 2948.330 -> 3018.561 | 8192 行 replay 与 stage-8 接受输出逐 byte 一致；对 FP32 policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-05 17:41:30 | 固定 B13 nested preConv GEMM，M128xN128xK32/W64x64/stage5/swizzle1/beta0（stage-11） | 保留：三轮锁频正反 ABBA 均为正；12+12 样本合并中位数 +1.34%，实际墙钟 +1.29% | L1K-EXACT19-B13-S2 | 双流 micro pair 25.62us -> 21.40us；NCU 17.82us -> 15.68us；最后 30 forward 双流 Nsys union 253.156ms -> 251.528ms | 2890.965 -> 2929.611 | 8192 行 replay 与 stage-9 逐 byte 一致；对 FP32 policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-05 18:18:42 | 融合 Q/K learnable RoPE 终栈复核（stage-13） | 保留：三轮锁频均为正；12+12 样本中位数 +2.05%，正序/反序均为正，实际墙钟 +2.07% | L1K-EXACT19-B13-S2 | 每 forward 66 次 RoPE launch -> 33 次；最后 30 个双流完整 forward 的 Nsys union 272.153ms -> 266.441ms（-2.10%） | 2904.690 -> 2964.290 | 8192 行 replay 与 stage-11 逐 byte 一致；policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-05 19:24:59 | QKV GEMM epilogue 融合 learnable RoPE（stage-16） | 保留：三轮锁频分别 +2.58%/+4.45%/+3.47%，12/12 相邻配对均为正，实际墙钟 +3.55% | L1K-EXACT19-B13-S2 | 每流每 forward 344 -> 311 个 kernel；最后 30 个双流完整 forward 的 Nsys union 266.861ms -> 255.423ms（-4.29%）；NCU 28.67us、零 spill | 3003.995 -> 3106.163（+3.40%） | 8192 行 replay 与 stage-13 逐 byte 一致；policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-06 01:33:19 | C768 trunk scratch 持久化 L2（stage-20） | 保留：三轮锁频正反 ABBA 的 pooled 中位数 +1.01%，正序/反序均为正，10/12 相邻配对为正 | L1K-EXACT19-B13-S2 | NCU C768 13.728us -> 12.704us、L2 hit 51.95% -> 82.64%、DRAM 7.218MB -> 2.687MB；最后 30 个双流完整 forward 的 Nsys union 255.150ms -> 251.586ms（-1.40%） | 3082.698 -> 3113.886（+1.01%） | 8192 行 replay 与 stage-16 逐 byte 一致；policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-06 01:52:43 | C384 inner residual 持久化 L2（stage-21） | 保留：三轮锁频 pooled +1.29%，正序/反序均为正，三轮均为正，8/12 相邻配对为正 | L1K-EXACT19-B13-S2 | 干净 NCU C384 6.62us -> 6.59us、L2 hit 52.51% -> 97.99%；两次 Nsys 等权 union 248.018ms -> 247.749ms（-0.11%），summed kernel -1.74% | 3117.614 -> 3157.842（+1.29%） | 8192 行 replay 与 stage-20 逐 byte 一致；policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-06 02:07:08 | TensorRT 10.16 exact-19x19/B13/S2 竞争基线 | 基线：同一 GPU、时钟、shape、batch、双 stream 和纯 forward 计时；交叉后端正反 ABBA 4/4 配对一致 | L1K-EXACT19-B13-S2 | 未记录 | TensorRT 2432.198；当前 CUDA 3145.511（同轮 +29.33%） | TensorRT 8192 行全部 head+p0loss 对 FP32 通过；policy top-1 99.7803%，p0loss 1.591563 |
| 2026-08-06 02:30:27 | C768 affine+SiLU flat vec8（stage-22） | 保留：三轮有效锁频 pooled +0.95%，正序/反序均为正，三轮均非负，10/12 相邻配对为正 | L1K-EXACT19-B13-S2 | NCU 11.680us -> 6.896us（-40.96%），无 spill；Nsys 目标 14.777us -> 8.632us，双流 union 251.930ms -> 245.016ms（-2.74%） | 3125.481 -> 3155.317（+0.95%） | 8192 行 replay 与 stage-21 逐 byte 一致；policy top-1 99.6948%，p0loss 1.591545（FP32 1.591528） |
| 2026-08-06 03:12:00 | 初始 3x3 卷积固定 cuDNN frontend plan（stage-24） | 保留：实际执行 `eng45_k14=2_k2=0`；短 ABBA 4/4 配对为正，收益符合该算子的关键路径占比 | L1S-EXACT19-B13-S2 | NCU 31.200us -> 22.048us（-29.33%）；正反 Nsys 目标约 -26.5%；最后 20 forward 双流 union 等权 -0.224% | 3251.925 -> 3257.140（+0.160%） | 8192 行 replay 与 stage-22 逐 byte 一致；完整 FP32 误差包络不变 |
| 2026-08-06 03:27:27 | 融合 policy P1 conversion + global bias + BN/SiLU（stage-25） | 微小保留：局部与正反 Nsys 均改善；短 ABBA pooled 为正但反向近乎持平，低置信 | L1S-EXACT19-B13-S2 | Nsys 三核 8.616/9.073us -> 单核 3.115/3.344us（约 -63%）；NCU v2 3.20-3.23us、零 spill；双流 union 正反 -1.77%/-0.55% | 3220.066 -> 3222.436（+0.074%） | 8192 行 replay 与 stage-24 逐 byte 一致 |
| 2026-08-06 03:58:14 | 初始 global K19->C768 FP32 dot + FP16 spatial add 融合（stage-27） | S1 保留：8 rows/CTA 为赢家；S2 配置保持关闭，后续改用 S1 作主迭代制度 | L1S1-EXACT19-B13 | NCU 三核合计 16.70us -> 9.52us（-42.99%）；S1 Nsys 15.872/15.875us -> 8.733/8.747us（约 -45%），零 spill | S1 2460.772 -> 2463.670（+0.118%，4/4 配对为正）；单顺序 S2 probe 3271.145 -> 3232.313（-1.187%） | 8192 行全部 head 对 FP32 通过；policy top-1 99.707%，prob RMSE 1.273e-4，value outcome RMSE 0.00265 |
| 2026-08-06 04:17:12 | policy/value 无拆分宽 head projection（stage-28） | S1 保留：C96+C96+C192 合并为一次 C384 AOT GEMM；不做 S2 相位调参 | L1S1-EXACT19-B13 | NCU projection 35.164us -> 15.940us（-54.67%），零 spill；正反 Nsys projection+首 BN 40.103/40.267us -> 21.778/21.781us（约 -45.8%），少 2 次 launch | S1 2459.496 -> 2474.237（+0.599%，4/4 配对为正） | 8192 行全部 raw output 与 stage-27 逐 byte 一致；FP32 误差包络不变 |
| 2026-08-06 04:27:53 | wide-head BN 直接输出 FP32（stage-29） | S1 低幅度保留：删除 g1/v1 两次 half-to-float copy；stage-26 的 S2 否决不变 | L1S1-EXACT19-B13 | NCU BN+copy 13.99us -> 8.06us（-42.39%），零 spill；正反 Nsys 12.089/12.064us -> 7.133/7.143us（约 -40.9%），少 2 次 launch | S1 2473.996 -> 2475.926（+0.078%，4/4 配对为正） | 8192 行全部 raw output 与 stage-28 逐 byte 一致；FP32 误差包络不变 |
| 2026-08-06 09:23:34 | exact-19 mask 预处理边界消除（stage-50） | 保留：精确 19x19 直接复用初始化期上传的 B13 `maskSum=361`，每 forward 删除 mask extract/half-to-float/sum 三次 launch；非 exact fallback 不变 | L1Q-EXACT19-B13-S2 | NCU 边界 6.848us -> 0；正反 Nsys busy union -3.65%/-1.98%，目标三核均归零 | 正反 Nsys +2.76%/+2.85%；锁 2400MHz 100-timed ABBA 均值 3218.860 -> 3253.014（+1.061%）；相对 TensorRT 2432.198 为 +33.75% | 8192 行全部输出与 stage-25 接受 S2 逐 byte 一致；FP32 policy top-1 99.6948%，prob RMSE 1.251e-4，value outcome RMSE 0.002610 |
| 2026-08-06 09:40:15 | value/score 终端 projection 合并（stage-51） | 机制保留、吞吐中性、S2 不启用：两次 projection+bias 合为一次 384->9 SGEMM + split/bias；默认关闭 | L1Q-EXACT19-B13-S2 + S1 attribution | S2 边界正反 10.903/11.013us -> 5.665/5.660us（约 -48%），4 -> 2 launches；S1 约 -49% | S2 +0.134%/-1.043%；S1 Nsys +0.147%/+0.135%，但 100-timed ABBA 2460.029 -> 2459.222（-0.033%，配对异号） | 8192 行全部输出与 stage-50 逐 byte 一致；FP32 包络不变 |
| 2026-08-06 19:06:04 | GTP pipeline metric/graph/scheduling 审计（stage-71） | 保留 eager event pipeline；禁用 graph。common-wall 修正旧 median-sum 指标；`MAX_CONNECTIONS=1` 与非对称 priority 均否决 | GTP-EAGER-2205 | Nsys graph/eager trace `3058.27 -> 3220.24` physical rows/s；priority 使 union busy `99.48% -> 93.03%` | GTP graph `3180.840 ->` eager `3236.025` physical rows/s（`+1.735%`）；direct-eager residual `0.196%` | graph/eager 26 行全部 raw output byte-identical；copy-control NCU 与完整边界均无回归 |

## 被否决或暂停的路线

| 结果保存时间（UTC） | 路线 | 单 kernel / 子图耗时 | 端到端吞吐（nnEval/s） | 决定与原因 |
|---|---|---:|---:|---|
| 2026-08-05 15:10:00 | fused QK RoPE（单 kernel 同时旋转 Q/K） | 未记录 | ABBA 候选 2014.5 vs 控制 2035.0（-1.0%）；BAAB 候选 2059.1 vs 控制 2008.5（+2.5%） | 暂停：两轮顺序结果冲突，未锁频漂移无法归因；开关默认 false，保留代码待锁频复核 |
| 2026-08-05 15:05:24 | `cudaDisableWarmup=true` 用于 B13-only benchmark | 未记录 | 无有效结果（plan 在 timed path 构建，运行被中止） | 否决：关闭预热破坏计时边界；改为保留预热但仅构建 B13 graph |
| 2026-08-05 15:33:00 | FlashAttention 上游 rounded-D64/M128xN112 | 53.12us | 未接整网 | 否决：实际 D32 被路由到 D64 specialization；native D32 立即降至 29.46us |
| 2026-08-05 15:36:00 | FlashAttention M128xN96 / `PackGQA=true` | 稳态 24.19-24.21us / M64 pack 23.51-23.56us | 未单独接整网 | 否决：M64 更适合 S361 的波次；Hq=Hkv 时关闭 pack 再快约 1% |
| 2026-08-05 15:39:00 | FlashAttention `MinBlocksPerMultiprocessor=4` | 23.69-23.74us（默认 23.28-23.34us） | 未接整网 | 否决：压到 128 registers 后产生 12B spill，理论 occupancy 提升未兑现为时延收益 |
| 2026-08-05 16:30:21 | DualGemm 动态 shared-memory attribute 只在初始化设置一次 | kernel 数学与 tile 不变 | repeat 中位 2980.778，preset 中位 2922.371（-1.96%） | 否决：300 次正反 ABBA 均变慢；重复 API 调用可能改善两个 host 提交线程的节奏与 S2 公平性，恢复上游 `run()` 行为 |
| 2026-08-05 16:38:01 | DualGemm swizzle 4 | 单流 micro 约比 swizzle2 快 0.46%，但 666 个逻辑 CTA 填充为 740 个 launch CTA | 锁 2400MHz/预热/500 次：sw2 中位 2841.114，sw4 2832.306（-0.31%） | 否决：反向组和稳态剔除首样本仍为负；B13/S2 额外尾块抵消单流 micro 收益 |
| 2026-08-05 17:06:47 | Linear2 更深 stage / 更多 warp 消融 | 同一 `beta=1` 双流 pair：接受的 stage4/4-warp 32.58us；stage4/8-warp 33.08us；stage5 33.76us；stage6 34.44us | 未接整网 | 否决：三条替代路线均慢于 stage4/4-warp；保留完整 10000-iteration ABBA 原始日志，若改变 epilogue 或数据驻留再重开 |
| 2026-08-05 17:22:00 | Attention out-projection 固定 CUTLASS GEMM（K384/beta=1） | 双流 micro 15.24us -> 14.23us；NCU 单流 13.47us cuBLAS / 14.05us CUTLASS，均零 spill | 未锁频 300：2983.149 -> 2984.718（+0.05%）；三轮锁 2400MHz/500 的合并变化 +0.60%、+0.40%、-0.55% | 否决：12+12 样本合并中位 +0.52%，但聚合正序 -0.53%、反序 +0.59%，符号冲突；开关默认 false，仅在融合边界或能解释双平台波动时重开 |
| 2026-08-05 17:56:38 | Nested postConv 固定 CUTLASS GEMM（K384/N768/beta=1，stage-12） | 25 个 tile 方向；赢家双流 micro 26.78us -> 23.14us；全部可实现候选 bit-exact | 三轮锁 2400MHz/500：-0.13%、-0.16%、-0.09%；合并 2905.854 -> 2900.916（-0.17%） | 否决：正序 -0.28%、反序 -0.16%、实际墙钟 -0.22% 均为负；开关默认 false。M256xN256 已实际编译运行，在 `can_implement` 被 CUTLASS 拒绝，并非因版本跳过 |
| 2026-08-05 18:35:00 | learnable RoPE float2 预计算表（stage-14） | kernel 8.773us -> 8.459us；完整 replay 与 stage-13 逐 byte 一致 | 池化中位数表面 +1.39%，但相邻配对仅 6/12 为正、配对中位 +0.38%；Nsys 双流 union 259.535ms -> 265.928ms（-2.46%） | 否决：约 36.6MB 常驻表的 L2/调度代价抵消单 kernel 小收益；开关默认 false，继续测试无额外表的 batch-grouped RoPE |
| 2026-08-05 18:52:00 | B13 batch-grouped fused RoPE（stage-15） | group 1/2/3/4/7/13 全扫；group2 kernel 8.860us -> 8.457us，8192 行 replay 逐 byte 一致 | 三轮 +2.85%/-1.17%/+0.31%；池化 +0.18%，相邻配对中位 +0.26%；Nsys 双流 union 仅 -0.02% | 否决：一轮为负且正反方向不对称，整网信号小于状态漂移；生产保持 group1，继续向 QKV/Flash 边界融合以删除中间流量 |
| 2026-08-05 19:31:30 | 19x19 可分离 RoPE 小表（stage-17） | direct-recompute S2 32.420-32.494us；可分离表 49.502-49.507us（约慢 52.5%）；两边相对各自 standalone 算术均逐 byte 一致 | 未接整网 | 否决：即使每层热表仅 29KiB，两次 float2 load 与 addition-identity 临时量仍远慢于直接频率 load + `__sincosf`；不是因 CUDA/CUTLASS 版本跳过 |
| 2026-08-05 19:39:11 | dual-GEMM swizzle1 对最终网络复核（stage-18） | 与 swizzle2 同为 666 CTA；当前 swizzle2 NCU 39.81us、168 registers、15.33% occupancy、零 spill | 三轮 +1.02%/-0.65%/-1.17%；池化 3083.024 -> 3066.456（-0.54%）；12 个相邻配对仅 3 个为正 | 否决：正序 -0.74%、反序 -0.11%、实际墙钟 -0.41% 均为负；保留 swizzle2，补齐了此前未测的最终网络 S2 证据 |
| 2026-08-05 19:42:36 | dual-GEMM 最终网络条件下的 S2 tile 复扫（stage-19） | 接受 tile 67.363/68.236us；最近的 warp32x64 68.561us；其余 80.927-104.464us；全部局部精度通过 | 未接整网 | 保留现状：`TB128x64x32/warp64x32x32/stage3/swizzle2` 仍为 S2 赢家；没有候选通过局部性能门槛，不以 S1 排名或工具链版本代替 S2 实测 |
| 2026-08-06 02:52:55 | 双 server 共享普通 `Sm89MatMul` 权重（stage-23） | NCU 11.680us -> 11.725us；476 次分配/H2D 被去除，减少 254,435,328 H2D bytes；正反 Nsys union 符号冲突 | 1500 次锁频正反 ABBA：3185.972 -> 3163.154（-0.72%） | 否决：显存去重机制成立，但 isolated kernel 无收益，长窗口正序 -0.85%、反序 -0.34%；开关默认 false。第二轮长测按用户要求中止并标为受污染 |
| 2026-08-06 03:37:07 | head BN half-to-float 双输出融合（stage-26） | policy g1 边界约 -47%，value v1 边界约 -38%；309 -> 307 kernels/forward/stream；正反 Nsys union +1.01%/-0.64% | 300 次锁频正反 ABBA：3246.156 -> 3205.214（-1.26%） | 否决：4/4 配对、正序和反序全部回归；局部 copy 消除破坏 S2 head 调度收益。开关默认 false，宽 head 边界改变时再重开 |
| 2026-08-06 04:56:05 | 融合 QKV epilogue 使用预计算 RoPE float2 表（stage-30） | NCU 29.728us -> 29.515us（-0.72%），240 -> 255 registers；L2 throughput 55.47% -> 67.32%，零 spill；正反 Nsys QKV 均约 -1.2% | Nsys 20 次：正序 2439.061 -> 2430.633（-0.346%），反序 2429.362 -> 2442.265（+0.531%） | 否决并回滚：局部未过 5% 门槛且整网方向冲突；26 行 byte-exact smoke 通过，不跑 ABBA/8192。融合后 18.3MiB 表把 math stall 换成 L2/scoreboard stall |
| 2026-08-06 05:10:51 | Linear2 CUTLASS Ampere Stream-K（stage-31） | S2 Stream-K 实际仍为 111 CTA，未补齐 128 SM；162 -> 168 registers。正序 linear2 28.927 -> 28.616us，反序 28.322 -> 28.943us | S2 Nsys 20 次：正序 3170.676 -> 3194.142（+0.740%），反序 3158.418 -> 3137.575（-0.660%） | 否决并回滚：预期的 wave-quantization 修复未发生，kernel 与整网方向均随顺序翻转；26 行 byte-exact smoke 通过，不跑 ABBA/8192。候选 NCU 因 `Kernel`/`Kernel2` 过滤错误未包含目标 launch，保留为无效工件 |
| 2026-08-06 05:27:19 | C384 RMSNorm 对齐 `uint4+uint2` vec12（stage-32） | S2 NCU 中位数 4.61 -> 5.38us（+16.7%）；long-scoreboard 约 8.5 -> 14.3-14.6 cycles；40 registers、零 spill。正反 Nsys RMSNorm 5.489 -> 6.419us / 6.084 -> 6.353us | S2 Nsys 20 次：正序 3207.440 -> 3136.197（-2.221%），反序 3234.771 -> 3154.948（-2.468%） | 否决并回滚：粗粒度向量 load-use 依赖降低 latency hiding，局部和整网在两个顺序均退化；26 行 S2 smoke 全部有限且 policy top-1 一致，不跑 ABBA/8192 |
| 2026-08-06 05:48:00 | TileLang 固定 B13 fused FFN `M128-N64-K32-S2-T128`（stage-33） | S2 NCU 共享内存 49.15 -> 32KiB、resident CTA 2 -> 3、理论 occupancy 16.67% -> 25%，但中位 41.34 -> 48.00us（+16.11%），no-eligible 约 78.4% -> 82.3%，零 spill；正反 Nsys FFN 43.284 -> 49.612us / 43.122 -> 49.443us | S2 Nsys 20 次：正序 3207.668 -> 3099.214（-3.381%），反序 3203.612 -> 3108.369（-2.973%） | 否决并回滚：3 CTA 机制成立但生成调度的依赖/指令链更差；26 行 smoke 全部有限、policy top-1 一致；单核与两个整网顺序均失败，不跑 ABBA/8192 |
| 2026-08-06 06:05:00 | C384 affine+SiLU flat vec8（stage-34） | S2 NCU 中位 6.85 -> 4.67us（-31.82%），CTA 4693 -> 880、waves 9.17 -> 1.15、16 -> 31 registers、零 spill；正反 Nsys 目标 8.725 -> 6.027us / 8.694 -> 6.083us | S2 Nsys 20 次：正序 3202.153 -> 3133.346（-2.149%），反序 3144.021 -> 3131.981（-0.383%）；summed kernel time -0.99%/-1.46%，但 GPU busy union +1.39%/+2.91% | 否决并回滚：局部向量化成立且 26 行逐 byte 一致，但双流 overlap/相位在两个顺序均恶化；不跑 ABBA/8192。前置 Flash NCU 27.97-28.22us，已为 3 CTA/SM 且既有 tile sweep 无新变量 |
| 2026-08-06 06:20:53 | C384 affine+SiLU phase-moderated vec4（stage-35） | S2 NCU 中位 6.85 -> 4.32us（-36.93%），CTA 4693 -> 1760、waves 9.17 -> 2.29、23 registers、零 spill；正反 Nsys 目标约 -35.5%/-35.1%，busy union -0.34%/-1.42% | S2 Nsys 20 次正序 +0.865%、反序 +1.623%；锁 2400MHz 的 100-timed ABBA 为 3250.984 -> 3224.363（-0.819%） | 否决并回滚：26 行逐 byte 一致且短门全部为正，但长 ABBA 一正一负、候选中位回归；局部收益小于未控制的双流相位漂移。跳过 8192，待公共 phase controller 落地后重开 |
| 2026-08-06 06:42:05 | QKV operand-B copy map 8x4 -> 16x2（stage-36） | 控制 S2 NCU 中位 29.632us、240 registers、49.15KiB shared、1.30 waves/SM、约 15% achieved occupancy、零 spill；候选未进入性能 profile | 未测：26 行正确性门先失败 | 否决并回滚：候选编译成功但 policy top-1 对控制为 0/26、value raw-logit RMSE 7.024。CUTLASS row-major-B 全局 copy map 与 congruous shared-memory lane layout 隐式耦合，不能单独换 alias；只有以匹配 global/shared/MMA 映射的 CuTe/custom mainloop 才可重开，并非因版本跳过 |
| 2026-08-06 06:51:23 | no-split C384 wide head 的 S2 独立验收（stage-37） | 候选 NCU 16.00/16.10us、162 registers、81.92KiB shared、0.87 waves/SM、零 spill；正反完整 head 边界 -39.39%/-41.32%，每 66 forward 少 132 launches | S2 Nsys 20 次：正序 3201.661 -> 3213.153（+0.359%），反序 3203.615 -> 3134.730（-2.150%）；reverse busy union +2.273% | 否决 S2 启用：26 行逐 byte 一致且局部机制两序均成立，但反序 overlap/吞吐明确回归；不跑 ABBA/8192。实现保持 default-off，S2 配置不启用，待 phase controller 或保持现有相位的 launch shape 后重开 |
| 2026-08-06 07:05:28 | QKV 合法 column-major B iterator（stage-38） | NCU control/candidate 中位 29.89 -> 24.96us，但 global excessive sectors 同为 55%，shared excessive wavefronts 9% -> 11%；正反 Nsys QKV 29.993 -> 30.614us / 29.520 -> 30.597us | S2 Nsys 20 次：正序 3203.028 -> 3167.817（-1.099%），反序 3115.454 -> 3103.961（-0.369%） | 否决并回滚：26 行逐 byte 一致、合法 layout 且无 spill，但访问冲突理论不成立，持续双流中的目标 kernel 两序均变慢；隔离 NCU 收益不可外推。不跑 ABBA/8192，候选源码与开关全部移除 |
| 2026-08-06 07:16:34 | C384 RMSNorm 8 warps/CTA（stage-39） | NCU 1174x128 -> 587x256 CTA，achieved occupancy 约 54.7% -> 58.8-60.1%，但中位仅 4.61 -> 4.58us（-0.65%）；40 registers、0.76 waves/SM、零 spill不变 | 未测：未过预设 3% NCU 门 | 否决并回滚：26 行逐 byte 一致，CTA grouping 机制成立但每 launch 只省约 0.03us，不足以进入相位敏感的全图测试；不跑 Nsys/ABBA/8192，恢复 4 warps/CTA |
| 2026-08-06 11:52:25 | projection residual + 下一层 C384 RMSNorm 全行 CTA 融合（stage-58） | 分离 linear2+RMS 23.072+4.640=27.712us；融合 M64xN512 中位 47.810us（+72.52%），184 registers、73.73KiB shared、0.58 waves/SM、16.67% occupancy | 未测：未过预设局部 NCU 门 | 否决并回滚：为让单 CTA 拥有完整 C384 行必须 pad N 到 512，33% 额外 MMA 与仅 74 CTA 的低 wave 成本远大于删除一次 RMS launch；26 行 smoke 通过，不跑 Nsys/ABBA/8192 |
| 2026-08-06 19:24:18 | both16 FlashAttention M64xN128 复扫（stage-72） | natural-event `20.866 -> 21.454us`（+2.817%）；NCU `21.50 -> 22.18us`，117 -> 168 registers、4 -> 3 CTA/SM、29.32% -> 22.28% achieved occupancy、无 spill | 未测：同 kernel 局部门失败 | 否决且未改源码：N128 虽把 K/V 循环从 4 轮降到 3 轮且 padding 同为 384，但更宽 fragment 重建寄存器瓶颈；不跑 M128、不跑整网/accuracy/新全图，复用 clean eager profile |

## 证据索引

- NCU dominant GEMM（ampere_h1688gemm_128x128，B13/S2）：
  grid 108 blocks / 128 SM，achieved occupancy 8.31%，
  no-eligible 85.79%，CPI stall 54.7% fixed-latency dependency。
- Nsys kernel summary（baseline-fixed gpu0-b23-fixed-s2）：
  FP16 GEMM 合计约 47%（41.1+4.9+1.2+0.7+0.4），
  cuDNN SDPA 25.5%。
- exact-19x19/B13 stage-6 Nsys：两条 timed stream 的 union busy
  `315.904 ms`；cuDNN SDPA exclusive `60.613 ms`，约占 union busy `19.2%`；
  mask-zero 和 attention-bias kernel 均已消失。
- stage-7 FlashAttention：上游固定在 `5835c733`、CUTLASS `7127592`，
  以 SM89 AOT 编译 native D32/M64xN96/W4/no-pack。最终双流 trace 的
  union busy `293.558 ms`，attention exclusive `34.231 ms`；NCU 为 168
  registers/thread、3 CTA/SM、22.44% achieved occupancy、无 local spill。
- stage-7 ABBA 工件：`stage7/final-abba/summary.json`；完整精度：
  `stage7/replay-sm89-flash-d32-m64n96-vs-fp32.json`；NCU/Nsys union：
  `stage7/ncu-flash-final-summary.json` 与 `stage7/flash-integrated-critical-path.json`。
- stage-8 dual GEMM + SwiGLU：CUTLASS `examples/45_dual_gemm` 固定为
  M128xN64xK32、warp M64xN32xK32、stage3、swizzle2。NCU 为 168
  registers/thread、约 49KB dynamic shared memory、无 spill；双流 trace 消除
  1980 个 timed SwiGLU launch，并将最后 30 个 forward/stream 的 union busy
  从 `283.620ms` 降为 `264.711ms`。ABBA：
  `stage8/integrated-abba-300/summary.json`；路径：
  `stage8/dual-gemm-integrated-critical-path.json`；全精度：
  `stage8/replay-sm89-dual-gemm-swiglu-sw2-vs-fp32.json`。融合相对当前
  cuBLAS control 非 bit-exact（累加顺序变化），但全部绝对 FP32 门槛通过；
  isolated random-input micro 的 bit-exact 结果不得外推为整网 bit-exact。
- stage-9 linear2 residual GEMM：CUTLASS 固定为 M128xN128xK32、warp
  M64xN64xK32、stage4、swizzle1，真实边界为 FP16 `beta=1`。完整 Nsys
  capture 精确移除 3498 个 cuBLAS launch 并增加 3498 个 CUTLASS launch；
  最后 30 forward/stream 的双流 union 从 `258.932ms` 降为 `252.386ms`。
  精确 NCU 为 162 registers/thread、64KiB dynamic shared memory、8.29%
  achieved occupancy、零 spill。ABBA：`stage9/integrated-abba-300/summary.json`；
  路径：`stage9/linear2-integrated-critical-path.json`；全精度：
  `stage9/replay-sm89-linear2-gemm-vs-fp32.json`。候选 replay 与 stage-8
  接受输出文件逐 byte 一致，因此整个 stage-8 FP32 误差包络保持不变。
- stage-11 nested preConv GEMM：固定 B13 `M=4693,N=384,K=768,beta=0`，
  CUTLASS tile 为 M128xN128xK32、warp M64xN64xK32、stage5、swizzle1。
  完整 Nsys capture 精确移除 `1166 = 106x11` 个 cuBLAS launch 并增加
  1166 个 CUTLASS launch；最后 30 forward/stream 的双流 union 从
  `253.156ms` 降为 `251.528ms`。精确 NCU 为 162 registers/thread、80KiB
  dynamic shared memory、8.30% achieved occupancy、零 spill。三轮锁频 ABBA：
  `stage11/integrated-decision-summary.json`；路径：
  `stage11/preconv-integrated-critical-path.json`；全精度：
  `stage11/replay-sm89-preconv-gemm-vs-fp32.json`。候选 replay 与 stage-9
  接受输出逐 byte 一致。
- stage-13 fused Q/K learnable RoPE：完整 capture 将
  `6996 = 106x33x2` 次单缓冲 RoPE launch 替换为 `3498 = 106x33` 次融合
  launch；每个 timed stream 的稳态 kernel 数由 377/forward 降到
  344/forward。最后 30 个完整双流 forward 的 union 从 `272.152509ms`
  降到 `266.441370ms`。三轮锁频 ABBA、稳定性复核和完整精度见
  `stage13/integrated-decision-summary.json`；路径见
  `stage13/fused-qk-rope-critical-path.json`；候选 replay 与 stage-11
  接受输出逐 byte 一致。
- stage-16 QKV+RoPE epilogue：固定 CUTLASS batched GEMM 为
  `M=4693,N=384,K=384,batch=3`，tile `M128xN128xK32`、warp
  `M64xN64xK32`、stage3。完整 capture 精确移除 3498 次 batched QKV 和
  3498 次 standalone RoPE，增加 3498 次 custom GEMM；候选 out-projection
  同形状调用仍为 3498 次，证明没有额外 Q/K/V fallback launch。最后 30
  forward/stream 的双流 union 从 `266.861ms` 降至 `255.423ms`。NCU 为
  240 registers/thread、49.15KiB dynamic shared、15.14% achieved occupancy、
  零 local/shared spill。最终决策：`stage16/final-decision-summary.json`；
  路径：`stage16/qkv-rope-epilogue-critical-path.json`；完整精度：
  `stage16/replay-qkv-rope-epilogue-vs-fp32.json`。
- stage-20 C768 trunk persisting L2：每个 stream 的 6.88MiB 长生命周期
  `trunkScratch` 在 trunk 区间内标记为 evict-last；S2 共 13.75MiB，低于
  4090 的 49.50MiB persisting-L2 上限。配对 NCU 显示目标 C768 consumer
  的 L2 hit 51.95% -> 82.64%、DRAM bytes -62.78%；双流 Nsys 没有新增
  memcpy、memset 或同步，最后 30 forward/stream 的 union 255.150ms ->
  251.586ms。三轮锁频 ABBA 与完整精度见 `stage20/abba-pooled-3r-summary.json`
  和 `stage20/replay-persisting-l2-trunk-vs-fp32.json`，机制与决策见
  `stage20/hypothesis-persisting-l2-trunk.md`。
- stage-21 C384 inner persisting L2：每个 nested block 在 preConv 写入前
  将 access-policy window 切到 3.44MiB `mid` residual，在六个 inner block
  和 postBN 后恢复 C768 window；S2 combined set-aside 为 20.63MiB。干净
  NCU 显示 C384 L2 hit 52.51% -> 97.99%，C768 没有回归。两次 Nsys 的
  union 方向一正一负，等权为 -0.11%，但 summed kernel 均下降；三轮长
  ABBA +1.29%。第一次 12x4-pass NCU 在 report 写完后的析构阶段退出码 6，
  明确标为受污染；后续 one-pass/split-metric profiles 均正常退出。完整证据
  和精度见 `stage21/hypothesis-persisting-l2-inner.md`、
  `stage21/abba-pooled-3r-summary.json`、
  `stage21/replay-persisting-l2-inner-vs-fp32.json`。
- TensorRT 10.16.1.11 exact-19x19/B13/S2 基线：GPU0、FP16、S2、锁
  2400MHz，以 TensorRT 为 A、当前 CUDA Stage21 为 B 的 500-timed 正反
  ABBA 得到 2432.198 vs 3145.511 nnEval/s，CUDA 高 29.33%，4/4 相邻配对
  一致。Plan 构建和 tactic cache 均在 event 计时区间外；完整数据和当前
  TensorRT 8192 行回归见 `tensorrt-baseline/summary.json` 与
  `tensorrt-baseline/replay-trt-exact19-b13-s2-vs-fp32.json`。
- stage-22 C768 affine+SiLU vec8：固定 `(N,XY,C)=(13,361,768)`，每线程
  处理连续 8 个 half，保留原 half FMA 与 float `expf` 算术。NCU 显示
  14,079 CTA / 3,604,224 threads 降为 1,760 CTA / 450,560 threads，31
  registers/thread、零 spill，耗时 -40.96%。最后 30 个完整双流 forward 的
  union -2.74%，长 ABBA +0.95%。一次 no-sample NCU、一次报告生成超时
  Nsys 和一次 timed 前启动 core 均单独保留且未用于决策。完整证据见
  `stage22/final-decision-summary.json`、`stage22/ncu-summary.json`、
  `stage22/nsys-summary.json` 和 `stage22/abba-pooled-3r-summary.json`。
- stage-23 普通 matmul 权重共享：启动 Nsys 精确减少 476 次分配和 H2D，
  共 254,435,328 bytes，证明物理去重成立；代表 out-projection NCU 中位数
  11.680us -> 11.725us，186 registers/thread、0.43 waves/SM、约 99.3% L2 hit
  均未改善。正反 Nsys union 符号冲突；完整 1500 次锁频正反 ABBA pooled
  3185.972 -> 3163.154（-0.72%），因此拒绝且默认关闭。详见
  `stage23/final-decision-summary.json` 和
  `stage23/hypothesis-shared-matmul-weights.md`。
- stage-24 初始 3x3 卷积 frontend plan：frontend 日志确认固定执行
  `eng45_k14=2_k2=0`，workspace 557,056 bytes。正确目标的 NCU 中位数
  31.200us -> 22.048us，寄存器 254 -> 244，occupancy 8.32% -> 12.70%，
  无 spill；另两个被通用 kernel 名过滤器误命中的 grid-111 GEMM 已剔除。
  正反 Nsys 对目标卷积均约 -26.5%，最后 20 forward/stream 的 union 方向
  相反但等权 -0.224%。单轮 300-timed 正反 ABBA 4/4 配对为正，pooled
  3251.925 -> 3257.140（+0.160%），与算子约 0.4% 的关键路径占比相符。
  8192 行 replay 与 stage-22 逐 byte 一致。详见
  `stage24/final-decision-summary.json` 与
  `stage24/hypothesis-initial-conv-frontend.md`。cuDNN 全库 debug logger 的一次
  并发构造崩溃仅作受污染诊断记录，不参与性能或稳定性判断。
- stage-25 policy P1 融合：原路径在 `13x361x96` 图上连续执行 half->float、
  per-batch global bias add、FP32 BN+SiLU。基线 NCU 的后两核约 2.9us 和
  3.15us，compute/memory SOL 均低于 35%，支持 launch/中间流量融合。
  v1 的 96-thread CTA 只有约 29% occupancy、4.38-4.48us；按 NCU 证据改成
  与原核一致的 96x5 geometry 后，v2 为 3.20-3.23us、66.5-68.3%
  occupancy、零 spill。Nsys 证实每 forward/stream 311 -> 309 kernels，
  目标边界约 -63%，正反双流 union 均下降。单轮短 ABBA pooled +0.074%，
  但反向 -0.042%，因此只按低置信微小收益保留；8192 行全部输出与 stage-24
  逐 byte 一致。详见 `stage25/final-decision-summary.json` 与
  `stage25/hypothesis-fused-policy-p1.md`。
- stage-26 head BN half-to-float：policy g1 直接输出 FP32，value v1 同时
  输出原 half（供 ownership）与 FP32（供 pooling），算术显式保留 half FMA、
  SiLU 后 half rounding 和 half-to-float 顺序。NCU 无 spill，两个完整局部
  边界分别约快 47% 和 38%，Nsys 也确认每 forward/stream 删除两个 copy；
  但双流 union 正反冲突，短 ABBA 4/4 配对为负，pooled -1.26%。因此拒绝并
  默认关闭，未做完整 accuracy。详见 `stage26/final-decision-summary.json` 与
  `stage26/hypothesis-head-bn-half-to-float.md`。
- stage-27 initial global dot + broadcast-add：固定 B13/K19/C768，以 256
  线程连续覆盖通道并复用 8 个空间行。前置 NCU 证明原 GEMM/reduce 仅
  0.05/0.09 waves/SM；最终 kernel 为 9.44-9.60us、40 registers、2.34
  waves/SM、82.9% occupancy、零 spill。S1 正反 Nsys 的完整边界约 -45%，
  100-timed ABBA 4/4 配对为正，2460.772 -> 2463.670（+0.118%）。单顺序
  topology probe 的 S2 为 -1.187%，因此只在新 S1 配置中启用；8192 行全部
  head 对 FP32 通过。详见 `stage27/final-decision-summary.json`、
  `stage27/hypothesis-initial-global-matmul-add.md` 与
  `/workspace/bench-cuda-gpu0-4090-s1.cfg`。
- stage-28 no-split wide head projection：前置 NCU 证明原 C96/C96/C192
  投影只有 0.58/0.58/0.29 waves/SM、约 8.3% occupancy。将三组权重拼为
  C384 后复用固定 AOT preConv GEMM，p1/g1/v1 首个 consumer 直接使用
  row-stride/offset，未 materialize split。候选 NCU 为 15.94us、162
  registers、0.87 waves/SM、零 spill；原三 GEMM 合计 35.164us。S1 正反
  Nsys 完整边界均约 -45.8%，100-timed ABBA 4/4 配对为正，2459.496 ->
  2474.237（+0.599%）。8192 行 replay 与 stage-27 逐 byte 一致。详见
  `stage28/final-decision-summary.json` 与
  `stage28/hypothesis-wide-head-nosplit.md`。
- stage-29 wide-head BN-to-FP32：重开 stage-26 时只针对已改变的 S1 宽头
  边界。copy NCU 的 compute SOL 仅 11.7%/16.8%，支持删除热 L2 搬运；
  新 kernel 直接读取 C384 slice，g1 写 FP32，v1 同时写原 rounded half 与
  FP32。候选 NCU 合计 8.06us、20 registers、零 spill；正反 Nsys 完整边界
  均约 -41%，并精确少 2 次 copy launch。100-timed ABBA 收益仅 +0.078%，
  但 4/4 配对和正反序均为正；8192 行与 stage-28 逐 byte 一致。详见
  `stage29/final-decision-summary.json` 与
  `stage29/hypothesis-wide-head-bn-to-float.md`。
- stage-33 TileLang fused FFN：用固定 B13 的 32KiB AOT 调度替换 CUTLASS
  49.15KiB dual-GEMM，NCU 验证 resident CTA 确实由 2 增至 3、理论 occupancy
  由 16.67% 增至 25%，但内核中位数反而慢 16.11%，no-eligible 也升到约
  82.3%。S2 正反 Nsys 吞吐分别 -3.381%/-2.973%，因此完整回退且跳过长测。
  详见 `stage33/final-decision-summary.json` 与
  `stage33/hypothesis-tilelang-fused-ffn.md`。
- stage-34 C384 affine+SiLU vec8：单核与 summed work 都明显减少，但 S2
  GPU busy union 正反序分别增加 1.39%/2.91%，吞吐均下降。这是局部优化
  改坏双流 overlap 的直接证据，因此完整回退。详见
  `stage34/final-decision-summary.json` 与
  `stage34/hypothesis-c384-scale-bias-silu-vec8.md`。
- stage-35 C384 affine+SiLU vec4：把 launch 从 vec8 的 1.15 waves 调回
  2.29 waves 后，NCU 中位数仍比 scalar control 快 36.93%，短 Nsys 的两个
  顺序也都转正，说明相位扰动幅度确实受 launch shape 控制；但一次缩短的
  100-timed ABBA 为 -0.819%，相邻配对一正一负，因此未越过稳定性门槛并
  完整回退。详见 `stage35/final-decision-summary.json` 与
  `stage35/hypothesis-c384-scale-bias-silu-vec4.md`。重新开启条件是公共
  dual-stream phase controller 可用，而不是改回 S1。
- stage-36 QKV B copy map：当前 S2 NCU 先确认 fused QKV+RoPE 仍是
  29.632us 中位、240 registers、49.15KiB shared、1.30 waves/SM 的低占用
  热点。直接把 CUTLASS row-major-B warp map 从 8x4 改成 16x2 虽能编译，
  但 26 行 replay 的 policy top-1 对控制为 0/26，静态审计确认其破坏了
  global-copy 与 `RowMajorTensorOpMultiplicandCongruous` shared lane mapping
  的隐式契约。因此在正确性门立即回退，未跑候选 NCU/Nsys/ABBA/8192。
  该方向只能以 global tiled-copy、shared layout、MMA consumer 三者匹配的
  CuTe/custom mainloop 重开；详见 `stage36/final-decision-summary.json` 与
  `stage36/hypothesis-qkv-b-copy-map-16x2.md`。
- stage-37 no-split wide head 的 S2 验收：候选 NCU 为 16.00/16.10us、
  0.87 waves/SM、零 spill；正反 Nsys 的 projection+first-consumer 边界都
  缩短约 40%，并在 66 个 forward 中减少 132 次 launch。正序吞吐 +0.359%、
  busy union -0.675%，但反序吞吐 -2.150%、busy union +2.273%，证明局部
  work reduction 再次改变了不受控的双流 overlap。S2 不启用且不跑
  ABBA/8192；详见 `stage37/final-decision-summary.json` 与
  `stage37/hypothesis-wide-head-nosplit-s2.md`。
- stage-38 QKV column-major B：为避免 stage-36 的非法 lane-map 替换，
  使用 CUTLASS 原生匹配的 column-major global/shared/MMA iterator 和初始化
  期转置权重。26 行逐 byte 一致；NCU 中位表面为 29.89 -> 24.96us，但
  global excessive sectors 仍为 55%，shared excessive wavefronts 反从 9%
  升至 11%。真实 S2 Nsys 中 QKV 正反均慢 2.07%/3.65%，吞吐均下降，故
  回滚且不跑 ABBA/8192。详见 `stage38/final-decision-summary.json` 与
  `stage38/hypothesis-qkv-column-major-b.md`。
- stage-39 RMSNorm 8 warps/CTA：保持每行一个 warp 和全部算术/访存顺序，
  只把 1174x128 改为 587x256。NCU achieved occupancy 从约 54.7% 升到
  58.8-60.1%，但中位时延仅 4.61 -> 4.58us（-0.65%），未过 3% 门槛。
  26 行逐 byte 一致；直接回滚，不跑 Nsys/ABBA/8192。详见
  `stage39/final-decision-summary.json` 与 `stage39/hypothesis-rmsnorm-warps8.md`。
- stage-40 CUTLASS dual-FFN stage 2：当前接受版 S2 Nsys 再确认该 kernel
  2178 次、平均 43.233us、累计 94.161ms，为 summed kernel time 第一热点；
  前置 NCU 为 49.15KiB shared、2 CTA/SM、16.67% 理论 occupancy。补齐固定
  CUTLASS wrapper 的 stage-2 实现缺口后，26 行逐 byte 一致；NCU 也验证
  shared 降至 32.77KiB、理论/实际 occupancy 升至 25%/约 21.9%，但中位
  时延反从 41.34 升至 42.27us（+2.25%），No Eligible 仍约 78.8%。因此
  stage 数和第三方实验补丁均回滚，不跑 Nsys 对比/ABBA/8192。瓶颈不在 CTA
  驻留数，而在 dual mainloop 的指令依赖/发射效率；详见
  `stage40/final-decision-summary.json` 与
  `stage40/hypothesis-cutlass-dual-ffn-stage2.md`。
- stage-41 dual-FFN A `cp.async.ca`：前置 S2 NCU 确认控制版 L1 hit 为 0%、
  L2 hit 99.78-99.79%、L2 throughput 72.72-73.77%。只把 A 从 `.cg`
  改成 `.ca` 后，26 行逐 byte 一致，L1 hit 出现但仅 2.02-2.24%，L2
  throughput 降到 69.47-69.87%；中位时延却从 41.28 升到 42.62us
  （+3.25%），No Eligible 也升至约 79.1%。因此回滚，不跑 Nsys 对比、
  ABBA 或 8192；详见 `stage41/final-decision-summary.json` 与
  `stage41/hypothesis-dual-ffn-cache-a-ca.md`。
- stage-42 dual-FFN B `cp.async.ca`：针对 B0/B1 权重沿 M 方向的理论复用，
  单独把 B 从 `.cg` 改成 `.ca`。26 行逐 byte 一致，但三次 NCU 的 L1 hit
  仍全为 0%，中位时延 41.28 -> 41.50us（+0.53%），L2 throughput 与
  No Eligible 也无实质改善。机制被证伪并回滚，不跑 Nsys 对比/ABBA/8192；
  详见 `stage42/final-decision-summary.json` 与
  `stage42/hypothesis-dual-ffn-cache-b-ca.md`。
- stage-43 dual-FFN 去除 `.L2::128B` hint：SASS 验证目标 TU 的 24 条
  `cp.async.cg` 均从带 128B prefetch 变成普通形式，26 行逐 byte 一致；
  但 NCU 中位 41.28 -> 41.34us（+0.15%），L2 throughput 仍约 73-74%，
  没有请求放大被消除的证据。因此本地 define 与 CUTLASS 实验 guard 均回滚，
  不跑 Nsys 对比/ABBA/8192；详见 `stage43/final-decision-summary.json` 与
  `stage43/hypothesis-dual-ffn-disable-l2-prefetch.md`。
- stage-44 dual-FFN horizontal HMMA visit：针对 NCU 中约 33.2% 的 execution
  pipe stall，只对 dual TU 取消 CUTLASS 的 SM89 vertical 特判。目标 SASS
  哈希变化而 HMMA 数保持 64，26 行逐 byte 一致；但 NCU 中位 41.28 ->
  41.54us（+0.63%），Warp Cycles 和 No Eligible 都无改善。因此确认 upstream
  vertical 顺序更优并完整回滚，不跑 Nsys 对比/ABBA/8192；详见
  `stage44/final-decision-summary.json` 与
  `stage44/hypothesis-dual-ffn-horizontal-mma-visit.md`。
- stage-45 dual-FFN B-only transform：模板源码看似对同一个 shared-A 做两次
  transform，因此增加 `transform_B()` 并复用第一次 A 结果。编译后目标函数
  SASS SHA256 与控制完全相同，HMMA 均为 64、move-like 指令均为 97，证明
  nvcc 已消掉第二次 A 赋值。按预设静态门直接回滚，不跑 smoke/NCU/Nsys/
  ABBA/8192；详见 `stage45/final-decision-summary.json` 与
  `stage45/hypothesis-dual-ffn-transform-b-only.md`。
- stage-46 dual-FFN 两 K-group 延迟 wait：S2 SourceCounters 先把主循环 barrier
  归因为 816 个采样中的 802 个 long-scoreboard；源码将 wait/barrier 从第 1
  组 MMA 后移到第 2 组后，但 `ptxas` 在控制和候选中都把 barrier 排在第 22
  条 HMMA 后。再加 accumulator 自依赖仍被完全化简，两版候选 SASS 相同，
  都是 64 HMMA、24 LDGSTS、19 barriers。因机器码未实现理论调度而在静态门
  回滚，不跑 smoke/候选 NCU/Nsys/ABBA/8192；详见
  `stage46/final-decision-summary.json` 与
  `stage46/hypothesis-dual-ffn-delay-stage-wait.md`。
- stage-47 dual-FFN 提前 next-stage copy：把固定两 K-group 的两组 copy 和
  fence 从 MMA 后移到 MMA 前，但 `ptxas` 仍将第一条主循环 LDGSTS 排在第 9
  条 HMMA 后；最后一条只从第 18 条后变为第 17 条后，barrier 从第 22 条后
  变为第 21 条后。64 HMMA、24 LDGSTS、19 barriers 全部不变，未形成理论
  要求的 copy head start，故在静态门回滚，不跑 smoke/NCU/Nsys/ABBA/8192；
  详见 `stage47/final-decision-summary.json` 与
  `stage47/hypothesis-dual-ffn-early-next-stage-copy.md`。
- stage-48A attention RMSNorm 折叠：当前最佳 S2 全图 checkpoint 先确认
  dual-FFN/Flash/QKV 的 exclusive busy 分别为 20.06%/11.05%/10.36%，因此
  从指令调度转向删除算子边界。将 gamma 折入 QKV 权重、RMS 改为每 token
  一个 FP32 invRMS 并在 QKV epilogue 缩放后，26 行 top-1 全部一致；隔离
  NCU 边界 34.464 -> 29.728us（-13.74%）。但真实 S2 Nsys 中 attention
  边界正反分别 +2.20%/+3.79%，整网吞吐 -0.271%/+0.160% 符号冲突，故
  否决且不跑 ABBA/8192。post-reject 全图 Nsys/NCU 已重采，下一投入点为
  FFN-only RMS folding。详见 `stage48/attention-folded-rms-decision.json`、
  `stage48/attention-folded-rms-nsys-summary.json` 和
  `stage48/post-attention-reject-checkpoint-summary.md`。
- stage-49 FFN RMSNorm 折叠：将 gamma 折入 linear1/gate 两组权重，只写
  每 token 一个 FP32 invRMS，并在 dual-GEMM SwiGLU epilogue 缩放。26 行
  top-1 全部一致，compute-sanitizer 对修正后的尾 tile 报 0 error；但 NCU
  边界中位从 45.952 增至 46.624us（+1.46%），真实 S2 Nsys 的边界正反
  分别 +1.68%/+1.95%，吞吐分别 -1.48%/-0.62%。候选和临时 CUTLASS 接口
  已完整回滚，replay 与冻结控制逐 byte 一致；未跑 ABBA/8192。回退后的
  全图 Nsys/广覆盖 NCU 已重采，后续转向 exact-19 路径上的无效 mask/shape
  边界。详见 `stage49/final-decision-summary.json`、
  `stage49/ffn-folded-rms-nsys-summary.json` 和
  `stage49/post-ffn-reject-checkpoint-summary.md`。
- stage-50 exact-19 mask 边界消除：固定 19x19 全有效 mask 后，跳过输入
  channel-0 提取、half-to-float copy 和 reduce-sum，改为上传持久的 361 元素
  mask 与常量 sum。目标边界从 6.848us 降为 0；S2 正反 Nsys 吞吐分别
  +2.759%/+2.854%，锁频 100-timed ABBA 为 3218.860 -> 3253.014
  （+1.061%）。8192 行 replay 与 stage-25 逐 byte 一致，已启用为当前 S2
  最佳；详见 `stage50/final-decision-summary.json`。
- stage-51 value/score terminal projection 融合：把同一 C384 输入的 384->3
  与 384->6 SGEMM 合并为 384->9，再用一个 split+bias kernel，局部边界约
  -49%，每个 value head 从 4 次 launch 降为 2 次。S2 正反吞吐
  +0.134%/-1.043%，S1 100-timed ABBA -0.033% 且两对异号，因此归为
  mechanism-accepted / throughput-neutral；简单实现和 8192 行逐 byte 精度
  证据保留，但 S2 默认关闭。详见 `stage51/final-decision-summary.json`。
- stage-52 intrinsic fusion bundle：将 stage-27 initial-global、stage-28/37
  wide-head projection 和 stage-29 head-BN-to-FP32 三个已有严格 NCU 证据的
  小融合累计启用。每 forward 少 6 次 launch；S1 正反短 Nsys 为
  +0.720%/+0.530%，100-timed ABBA 两对均为正，均值 +0.623%。但 S2
  正反分别 -3.630%/+2.279%，且 control 自身落入两个相位档，故整个 bundle
  归为 intrinsic-accepted / S2 phase-sensitive，源码保留而部署默认关闭。
  从此增加 cumulative intrinsic bundle 通道：后续每个 NCU+S1 严格更优项
  都进入累计包并整体复测，不再由单项 S2 相位门控删除。详见
  `stage52/final-decision-summary.json` 与 `ITERATION-PROTOCOL.md`。
- stage-53 strict-local fusion bundle：把 S1 中性的 stage-51 value-terminal
  融合加入前三项累计包。完整候选在 64 个双流 forward 中稳定少 512 次
  launch，即每 forward 少 8 次；但 S2 正反吞吐分别 -3.357%/-2.332%，
  busy union 分别 +3.975%/+2.035%，未进入 ABBA。四项均保留为默认关闭的
  strict-local bundle，当前部署仍为 stage-50；详见
  `stage53/final-decision-summary.json`。
- stage-54 C384 affine+SiLU vec4 重新归因：恢复旧 stage-35 被回滚的 exact
  B13/C384 vec4。26 行与 8192 行均对 stage-50 逐 byte 一致；S2 来源的
  NCU 中位 6.816 -> 4.224us（-38.03%），CTA 4693 -> 1760，waves/SM
  9.17 -> 2.29，23 registers/thread、零 spill。S1 正反短 Nsys
  +0.730%/+0.673%，100-timed ABBA 两对均为正，均值 +0.543%，因此改判为
  intrinsic-accepted 并保留默认关闭。加入后的五项累计包 S2 正反仍为
  -1.561%/-1.275%，故部署仍维持 stage-50；详见
  `stage54/final-decision-summary.json`。
- stage-55 head BN+SiLU+pooling 融合：尝试删除 policy/value 两张 FP32
  spatial 中间量和两次 pool launch。26 行逐 byte 一致，但 S2 来源 NCU
  直接证伪：policy 完整边界 8.768 -> 10.368us（+18.25%），value
  9.280 -> 10.400us（+12.07%）。pool 只有 0.07/0.10 waves/SM，把原本高并行
  的 half-FMA+expf 串入每线程 46 次循环后，节省的流量不足以抵消并行度损失。
  因局部机制失败而完整回滚，未跑 S1/S2 Nsys 或长测；详见
  `stage55/final-decision-summary.json`。
- stage-56 跨 block postConv+下一层 BN/SiLU 融合：每个 postConv epilogue
  同时保留原 FP16 residual，并从该舍入后的 fragment 计算下一 nested block
  的 C768 affine+SiLU；最后一块直接生成 trunk-tip 输出。每 forward 删除
  11 次 C768 BN launch 和 11 次 7.2MiB residual 读取。S2 来源 NCU 的完整
  边界 26.496 -> 21.120us（-20.29%），register 186 -> 164，waves/SM 保持
  0.87；S1 正反 +1.169%/+1.296%，S2 短测 +3.383%/+2.319%。锁频
  100-timed S2 ABBA 为 3277.003 -> 3288.971（+0.365%，两对均正），8192
  行通过全 head FP32 envelope，已启用为新 current best；相对 TensorRT
  2432.198 为 +35.226%。把旧五项 strict-local 一并开启后 S2 正反仍为
  -2.158%/-1.016%，因此旧五项继续默认关闭。详见
  `stage56/final-decision-summary.json` 与
  `stage56/post-stage56-current-best-checkpoint.md`。
- stage-57 final inner FFN linear2+下一层 C384 BN/SiLU 融合：每个 nested
  block 的最后一个 inner FFN 直接在 linear2 residual epilogue 中生成 postBN
  activation，每 forward 删除 11 次 C384 BN launch。S2 调用来源的 NCU
  完整边界 30.112 -> 24.640us（-18.17%）；融合 kernel 单独比 linear2 慢
  6.65%，但删除独立 BN 后总边界严格缩短。S1 正反短测 +0.941%/+0.968%，
  锁频 100-timed ABBA 为 2492.209 -> 2512.120（+0.799%，两对均正），
  8192 行通过全 head FP32 envelope，因此源码以 commit `91f6aae` 保留。
  单项有效 S2 正序为 -2.991%，反序 control 两次落入约半吞吐污染相位；加入
  旧五项后的完整七路线 bundle 正反为 +0.628%/-2.360%，仍然相位敏感，故
  开关默认关闭，current best 仍为 stage-56 的 3288.971 nnEval/s（相对
  TensorRT +35.226%）。详见 `stage57/final-decision-summary.json`。
- stage-58 projection residual+下一层 RMSNorm 全行 CTA 融合：全图首先确认
  每 forward 有 22 个 linear2->RMS、33 个 outProj->RMS 和 11 个
  preConv->RMS 边界。用最严格的 linear2 做 feasibility prototype；两个
  literal N384 warp-grid 形状不能正确映射，correctness-valid 的常规 CUTLASS
  形状必须 pad 为 M64xN512。26 行 smoke 通过，但 S2 来源 NCU 的完整边界
  27.712 -> 47.810us（+72.52%）；候选只有 74 CTA/0.58 waves，184
  registers/thread、73.73KiB shared、16.67% occupancy，还做 33% 额外输出
  MMA。局部机制被严格证伪，未扩展到 outProj/preConv，源码完整回滚，不跑
  Nsys/ABBA/8192；详见 `stage58/final-decision-summary.json`。
- stage-59 SM89 FlashAttention both16：仅在 exact B13/S361/H12/D32 路径
  把 QK/PV Tensor Core accumulator 改为 FP16，online-softmax 的 row max、
  row sum 与 LSE 仍保持 FP32。S2 来源 NCU/SASS 证实 HMMA 从 F32 变为 F16，
  registers/thread 168 -> 117、驻留上限 3 -> 4 CTA/SM、eligible warps/cycle
  0.348 -> 0.870、零 spill，kernel 中位 28.224 -> 20.864us（-26.08%）。
  锁频 100-timed ABBA+BAAB 均值 3274.600 -> 3417.873 nnEval/s（+4.375%），
  中位 +4.700%；8192 行全部输出通过认可的 both16 精度 envelope。接受后整图
  Nsys 中 Flash raw time -19.74%、exclusive -44.92%，已用 commit `7d299d0`
  启用为新 current best；相对 TensorRT 2432.198 为 +40.526%。下一宏观热点
  转为 dual-FFN（18.30% exclusive），其次 QKV+RoPE（11.47%）。详见
  `stage59/final-decision-summary.json` 与
  `stage59/post-stage59-current-best-checkpoint.md`。
- stage-60 attention RMSNorm->QKV 折叠复审：先恢复旧 FP32 invRMS
  epilogue 缩放；当前 both16 图的 S2 来源 NCU 边界 34.592 -> 35.200us
  （+1.76%），单流 intrinsic NCU 34.944 -> 35.264us（+0.92%），与旧
  stage-48 的三 launch 结论相反。继续把逐元素 FP32 缩放改成每行一次
  FP16 scale + half2 multiply 后，26 行 top-1 保持 100%、fallback 与 control
  逐 byte 一致；但完整 198 次 QKV block 序列的短 S1 Nsys 最终确认相关
  边界 7.265 -> 7.525ms（+3.57%），整网 2562.598 -> 2538.452 nnEval/s
  （-0.942%）。根因是 QKV epilogue 的逐输出缩放成本仍大于删除 RMS
  materialization 的收益。两版均完整回滚；按回滚路线规则不跑 S2 ABBA、
  8192 或新全图 checkpoint，继续复用 stage-59。详见
  `stage60/final-decision-summary.json`。
- stage-61 五项 intrinsic fusion bundle 在 both16 图上复验：复用五个组件
  已有的 -38% 到 -55% 局部 NCU/边界证据和每 forward 少 8 launch 的机制，
  直接在 Stage59 S2 做锁频 100-timed ABBA+BAAB。正序为 -1.594%，反序
  +0.614%，pooled mean 3441.447 -> 3424.559（-0.491%），median -0.621%，
  仍是相位敏感且总体为负。因此不部署，五项实现继续默认关闭；不跑 8192
  或新全图 profile，继续复用 stage-59 checkpoint。详见
  `stage61/final-decision-summary.json`。
- stage-62 SM89 dual-FFN half2 tanh SwiGLU：保留已有 CUTLASS dual-GEMM
  mainloop/tile，只把 epilogue 的 FP32 exp/reciprocal sigmoid 改写为 packed
  FP16 `0.5*tanh(0.5*x)+0.5`。SASS 每 launch 保持 64 HMMA，同时把
  64 EX2 + 64 RCP + 96 F2F 替换为 64 TANH + 64 F2F 与 half2 算术；资源
  保持 168 registers/thread、49.152KiB shared、2.60 waves/SM、零 spill。
  NCU 三次均值 41.536 -> 41.024us（-1.23%），短 Nsys 完整 198-launch
  边界 7.998 -> 7.900ms（-1.22%）。锁频 100-timed S2 ABBA+BAAB 为
  3389.124 -> 3424.124 nnEval/s（均值 +1.033%，中位 +1.031%，正反均正）；
  8192 行全部输出通过 both16 精度 envelope，已部署。接受后全图 Nsys 为
  3418.728 nnEval/s，dual-FFN union 仍占 36.69%，其次 QKV+RoPE 25.34%、
  FlashAttention 22.89%；相对 TensorRT 2432.198 的配对候选均值为
  +40.783%。详见 `stage62/final-decision-summary.json` 与
  `stage62/post-stage62-current-best-checkpoint.md`。
- stage-63 fused QKV epilogue half2 RoPE：当前全图 QKV+RoPE 仍占 25.34%
  union / 11.50% exclusive，前置 NCU 为 240 registers/thread、1.30
  waves/SM。候选保留 FP32 `sincos`，只把每对旋转改成 packed half2；SASS
  确认每 launch 的 64 HMMA/64 SIN/64 COS 不变，scalar FFMA 192 -> 64、
  FMUL 256 -> 128，并新增 64 HMUL2+64 HFMA2，但 F2F 128 -> 192 且新增
  64 PRMT。资源完全不变，NCU 三次均值仅 30.005 -> 29.963us（-0.14%），
  短 Nsys 完整 198-launch 边界仅 5.734 -> 5.728ms（-0.11%），S1 吞吐
  2570.487 -> 2560.208（-0.40%）。机制成立但无严格局部收益，故完整回滚，
  不跑 S2/8192/新全图；恢复后 26 行全 head 与 stage-62 为 0 差异。详见
  `stage63/final-decision-summary.json`。
- stage-64 拆分 AOT QKV+RoPE 边界：plain QKV 保持每 launch 64 HMMA，
  去掉 64 SIN/64 COS/192 FFMA/256 FMUL，registers/thread 240 -> 168，
  无 spill。独立 NCU replay 错误预测 `30.005 -> 23.445+7.093us`
  （+1.78%）；真实 S1 Nsys launch-to-completion 边界却是
  `28.941 -> 25.948us`（-10.34%），整网 `2569.975 -> 2846.742`
  nnEval/s（+10.77%）。这证明独立 NCU 时延不可相加作跨 kernel
  边界门控，新规则已写入 `SKILL.md` 和 `ITERATION-PROTOCOL.md`。
  真实 S2 Nsys 中 plain QKV 约 25.10us，但 standalone RoPE 约
  8.87us 且含 launch gap 的完整边界 `29.634 -> 38.743us`（+30.74%），
  短整网吞吐约 -3.71%；因此实现按 `intrinsic-accepted / S2-regressed`
  保留为默认关闭，不部署。8192 行全部输出与 stage-62 为 0 差异。
  历史审计将 stage-55/stage-58 的 additive-NCU 拒绝重分类为待复验，
  stage-39 则因旧 3% 任意门槛列入低优先级重开。详见
  `stage64/final-decision-summary.json` 与
  `stage64/historical-measurement-gate-audit.md`。
- stage-65 plain QKV native-half epilogue：仅作用于 stage-64 默认关闭的
  split 路径，以 `cudaPlainQKVVariantSm89=0/1` 在同 binary 中对照。
  候选与 control 在 26/8192 行全部输出上都是 0 差异；SASS 保持
  64 HMMA/launch，删除全部 64 F2F/launch，168 registers/thread、
  49.15KiB shared、1.30 waves/SM 和零 spill 不变。排除双方共同的
  首个 80-81us replay 异常后，配对 NCU `23.505 -> 23.425us`
  （-0.34%）；同档自然 S1 Nsys 完整边界 `29.595 -> 29.525us`
  （-0.23%）。50-timed S1 ABBA 均值 `2622.251 -> 2623.135` nnEval/s
  （+0.034%，正反均非负），不构成稳定吞吐收益。因此按
  `mechanism-accepted / throughput-neutral` 保留 variant 1，默认仍为 0，
  不进入 S2 部署门控；继续复用 stage-62 current-best 全图 checkpoint。
  详见 `stage65/final-decision-summary.json`。
- stage-66 plain QKV stage-2：保持 stage-65 的 native-half epilogue 和
  M128xN128xK32/warp64x64，只将 CUTLASS mainloop 由 3 stage 改为
  2 stage。26 行与 stage-3 为 0 差异，shared 如预期从 49.15KiB
  降到 32.77KiB；但 ptxas 将 registers/thread 从 168 提到 206，
  理论 occupancy 仍为 16.67%，没有获得第三个驻留 CTA。NCU
  `23.530 -> 25.260us`（+7.35%），自然 S1 Nsys launch-to-completion
  `29.578 -> 33.289us`（+12.55%），整网 `2565.378 -> 2500.134`
  nnEval/s（-2.54%）。机制明确失败并完整回滚，不跑 S2/ABBA/
  8192/新全图 checkpoint。详见 `stage66/final-decision-summary.json`。
- stage-67 plain QKV N64 tile：保持 stage-3/K32/M128/native-half，将
  threadblock N128/warp64x64 改为 N64/warp64x32。26 行与 control
  为 0 差异，registers/thread 168 -> 128、shared 49.15 -> 36.86KiB，
  但 36.86KiB 仍只能驻留 2 CTA/SM，理论 occupancy 仍为 16.67%。
  grid 333 -> 666 CTA 只使 waves/SM 1.30 -> 2.60；NCU
  `23.400 -> 29.203us`（+24.80%），自然 S1 完整边界
  `29.571 -> 35.125us`（+18.78%），整图吞吐 -3.40%。完整回滚，
  不跑 S2/ABBA/8192/全图；详见 `stage67/final-decision-summary.json`。
- stage-68 CUDA backend 外部 stream 接口：调用方为每个 compute handle 创建并持有
  一条显式 non-blocking stream，CUDA handle、cuBLAS/cuDNN、SM89 独立 forward、
  全部 custom helper kernel、运行期 H2D/D2H、event 和同步统一使用该 stream；移除
  CUDA backend 的 PTDS 编译策略，缺失 stream 直接报错。26 行 B13 全 head 与
  stage-62 bit exact，`runnnlayertests` 28 个配置全部通过。自然 S2 Nsys 捕获的
  5916 个 kernel 全部位于 4 条显式 non-blocking stream，测量 forward stream
  81/82 各自包含对应 H2D/D2H，default stream 上无 forward kernel。三次
  100-iteration S2 为 3462.633/3460.038/3450.634 nnEval/s（均值
  3457.768），只作为无回归证据，不把未配对的 +1.142% 宣称为因果加速。
  接口修复保留；详见 `stage68/final-decision-summary.json`。
- stage-69 dual-FFN N128 tile：把 M128xN64/4-warps 扩为
  M128xN128/8-warps，期望在保持每 SM 8 resident warps 的同时减半 CTA 与
  A tile 重复流量。两种 swizzle 的 26 行全 head 均与 control 0 差异。
  swizzle-2 因奇数 N tile padding 为 370 CTA/2.89 waves，NCU 中位
  `40.96 -> 44.45us`（+8.52%）；针对性改为 swizzle-1 后准确降到
  333 CTA/2.60 waves，但仍为 `44.22us`（+7.96%）。候选 registers
  168 -> 156、L2 throughput 约 74% -> 52%、occupancy 15.3% -> 16.65%，
  说明资源变化成立，但单个 8-warp CTA 的同步/调度代价更大。完整回滚；按同
  kernel NCU 预设门不跑自然 Nsys/S2/长测/新全图，继续复用 stage-68 checkpoint。
  详见 `stage69/final-decision-summary.json`。
- stage-70 fused QKV 预计算 RoPE 表复审：按新工作流重开 stage-30 曾被旧
  S2 门槛误删的 float2 `(cos,sin)` 表方案，并用编译期双变体保证 default
  control 保持原始 240 registers/thread。26 行 B13 全 head 逐项 0 差异；
  但当前来源/工具链上的三次 NCU 中位为 `29.696 -> 30.368us`
  （+2.263%），registers 240 -> 255，L2 throughput 中位约
  55.46% -> 64.58%，long-scoreboard stall/issue 1.589 -> 2.844，issue
  active 20.71% -> 18.60%。说明表读取与寄存器压力超过删除 `sincos` 的收益，
  旧 stage-30 的 +0.72% 未能复现。按同 kernel 门完整回滚，不跑自然
  Nsys/S2/长测/新全图，继续复用 stage-68 checkpoint；无源码提交。详见
  `stage70/final-decision-summary.json`。
- stage-71 GTP pipeline metric/graph/scheduling 审计：在独立 worktree
  `/workspace/katago-gtp-pipeline-gap` 增加 completed physical B13 launches /
  common wall 指标，证实旧 per-lane median-sum 仅高估 0.047%，但独立 S2
  process 的相位跨度达到 2.79%。真实 t96 GTP ABBA 中 graph/eager physical
  rows 为 3180.840/3236.025（eager +1.735%），real nnEval/s 为
  3130.865/3183.790（+1.690%），26 行全部输出 byte-identical。Nsys 显示
  graph 把双流 overlap/union 从 60.07% 提到 63.27%，DualGemm/Flash/
  Ampere-GEMM/RMSNorm 分别慢 4.33%/6.90%/9.67%/16.94%；当前 graph 不是
  launch-gap 优化而是更差的固定资源竞争相位，因此部署改为 eager。
  `CUDA_DEVICE_MAX_CONNECTIONS=1` 将吞吐压到 2471.6，2-32 无稳定赢家；
  任一 lane 设 -1 priority 使 eager 下降 5.5-6.2%，Nsys 证实低优先级 lane
  kernel sum 增长约 11% 且 union busy 跌到 93.03%，实现已回滚。脚本请求
  10501MHz 显存锁但 decision telemetry 实际稳定为 10251MHz；后续制度必须
  记录观测值。诊断工具 commit `7e6ec01`，部署报告 commit `67a4d79`；详见
  `gtp-pipeline-gap-investigation/investigation-report.md`。
- stage-72 both16 FlashAttention retile：Stage 7 的 tile sweep 发生在 FP32
  accumulation 资源制度下，Stage 59 变为 both16 后因此重开最有机制差异的
  M64xN128。锁 2205MHz natural-event ABBA/BAAB 中控制 M64xN96 为
  `20.866us`，N128 为 `21.454us`（+2.817%，4/4 配对均慢）。NCU 解释为
  registers `117 -> 168`、dynamic shared `16.77 -> 20.86KiB`、resident
  CTA `4 -> 3`、achieved occupancy `29.32% -> 22.28%`、eligible warps
  `0.87 -> 0.69`；无 spill。按预设门停止，不做源码变更、整网或全图复采。
  详见 `stage72/final-decision-summary.json`。
- 模型结构（sm89-debug，已移除）：11 个 nested bottleneck block，每个
  preConv 768->384、6 个 inner block（3 attention C384/h12/d32 + 3 FFN
  C384->1152）、postConv 384->768；trunkNorm=standard BN。
