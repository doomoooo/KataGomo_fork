# KataGo 1.17.1 CUDA/SM120 优化历史

整理时间：2026-08-05（Asia/Shanghai）

范围：RTX 5080（SM120）、KataGo 1.17.1、模型
`b11c768h12nbt3tflrs-fson-silu.bin.gz`、CUDA 13.2、cuDNN 9.24.0。
本文记录从官方 CUDA/cuDNN backend 到当前固定 B19 CUDA 实现的完整优化路径，
同时保留已经证伪或暂停的路线。

时间戳规则：表中的时间均为 Asia/Shanghai（UTC+8），精确到秒。可点击的时间直接链接到
对应 benchmark/profile 结果文件；复合 A/B 或 ABBA 实验使用汇总文件或最后一份确认结果的
保存时间。若早期原始 JSON 已没有单独留存，则使用承载该组结果的最终报告保存时间。只有诊断
而没有 benchmark 的条目会明确标为“诊断记录”或“无结果时间”，不会用代码修改时间冒充测试时间。

百分比只在相同测量制度内计算。时钟、batch 或 NN server 拓扑不同的结果只作为绝对里程碑，
不把它们的增幅串联相加。

## 测量制度

| 标记 | 条件 |
|---|---|
| U1 | 前期未锁频；1 个 NN server；B19；纯设备 `benchmarknn` |
| U2 | 前期未锁频；2 个 NN server，各自 B19；总并发 batch 为 38 |
| GRID | 在 GPU 空闲监控下扫描 B1-B32 与 NN server 数；未锁频；平台期停止前完成 S1-S2 |
| L | 核心目标频率 2430 MHz、显存 14801 MHz、PL 380 W；2 条独立 NN server stream；每条 B19；总并发 38；长跑或有序 A/B/ABBA |
| ACC | 固定 8192 行 19x19 语料；全 FP32 CUDA/cuDNN 参考；覆盖全部 head 与 p0loss |

最终 ABBA 中，2430 MHz 应用时钟在接近 380 W 时会因功耗限制在 2415-2430 MHz 间波动。
因此 A-B-B-A 顺序是测量协议的一部分，不是附带的记账形式。

`benchmarknn` 用 CUDA event 测量重复的、数据已驻留设备端的 forward。计时不包含预处理、
后处理、H2D、D2H、模型加载、搜索和 TensorRT plan 构建。

## 数值精度脚手架

精度语料不是从 SGF 重建，而是从真实训练数据中无放回均匀抽取并固定保存。每行已经包含模型
所需的空间 history 特征、全局/规则特征及全部训练目标。

| 属性 | 值 |
|---|---|
| 棋盘尺寸 | 仅 19x19 |
| 合格训练行 / shard | 673,940 / 9,925 |
| 固定样本数 | 8,192 |
| 随机种子 | 20260803 |
| 源 archive SHA256 | `8837a68993d09d2f871d7f6e677fb1483f19080c5a8a5ce25697b078554efc85` |
| p0loss 语料 SHA256 | `262b0fbdf8e43ed9427cccd37cdcaa05e5a2bc79b17c9d03831a82bc9667e957` |
| ground truth | CUDA 13.2 + cuDNN 9.24，全 FP32 |
| 统计输出 | policy/optimistic policy 概率与 logits、outcome、score、ownership、分布指标、top-1 一致率、p0loss |

最终 manifest：
[2026-08-04-19x19-8192-seed20260803-p0loss.manifest.json](/data/wangyize/katago/trainingdata/accuracy/2026-08-04-19x19-8192-seed20260803-p0loss.manifest.json)

## 主时间线

单 kernel / 子图列只填写 profiler 或 microbenchmark 直接记录的数值；端到端列只填写
`benchmarknn` 直接记录的吞吐。两列可能来自不同测量协议，不能互相反推；没有直接证据时写“未记录”。

| 结果保存时间（UTC+8） | 优化或阶段 | 结论 | 制度 | 单 kernel / 子图耗时（us） | 端到端吞吐（nnEval/s） | 精度证据 |
|---|---|---|---|---:|---:|---|
| [2026-08-03 16:25:24](/data/wangyize/katago/results/trt-10.16-cuda13.2-b19-fp16-final.json) | TensorRT 10.16.1.11，S1/B19，FP16 | TensorRT 起始基线 | U1 | 未记录 | `1942.597` | 后续统一对全 FP32 参考比较 |
| [2026-08-03 16:38:02](/data/wangyize/katago/results/README.md) | `benchmarknn` 脚手架 | 保留 | 全部 | 未记录 | 未记录 | 建立 CUDA 与 TensorRT 一致的纯设备计时边界 |
| [2026-08-03 16:50:39](/data/wangyize/katago/results/cudnn-9.24-cuda13.2-b19-fp16-final.json) | 官方 CUDA backend，CUDA 13.2 + cuDNN 9.24，FP16 I/O | 起点 | U1 | cuDNN SDPA `99.61` | `1479.162` | 后续统一对全 FP32 参考比较 |
| [2026-08-03 17:25:07](/data/wangyize/katago/trainingdata/accuracy/2026-07-30-19x19-8192-seed20260803.manifest.json) | 固定 8192 行 19x19 精度语料和全 FP32 参考 | 保留 | ACC | 未记录 | 未记录 | 建立后续候选的统一数值门槛；p0loss 于下方时间点补入 |
| [诊断记录：2026-08-03 19:54:34](/home/wangyize/.katago/src/KataGo-v1.17.1/docs/cudnn-sdpa-compute-type-research.md) | cuDNN attention 精度诊断 | 保留结论 | U1 | cuDNN SDPA `99.61`；TRT `_gemm_mha_v2` `34.78` | 未记录 | cuDNN 公开接口要求 FP32 compute 语义；TRT B19 tactic 的 QK/PV 均为 FP16 HMMA 累加 |
| [2026-08-03 21:05:38](/data/wangyize/katago/results/fa4-sm120-accumulator-scan-b19-s361-h12-d32.json) | FA4 累加模式扫描 | 为最终采用 `both16` 提供依据 | U1，FA4 直接 A/B | FP32 `42.609`；qk16 `34.303`；pv16 `33.059`；both16 `32.120` | 未记录 | 三种 FP16 累加模式均进入 8192 行全 head 比较 |
| [2026-08-03 21:51:29](/data/wangyize/katago/results/flash-attention-sm120-aot-integration-report.md) | 固定 `B19/S361/H12/D32`、noncausal FA4/CuTe AOT；外部 stream；形状不匹配回退 cuDNN | 保留；此阶段使用 qk16，最终改为 both16 | U1，直接 A/B | both16 C++ wrapper `29.548` | cuDNN `1445.534`；qk16 `1663.992`；pv16 `1660.741`；both16 `1674.315` | 8192 行全部 head；qk16 综合绝对误差最好，both16 最快 |
| [2026-08-03 22:19:28](/data/wangyize/katago/profiles/cuda-fa4-qkv-b19-current.nsys-rep) | Q/K/V 三次独立投影改成一次 `cublasHgemmStridedBatched(batchCount=3)`；共享输入，仍输出三块连续 Q/K/V | 保留；这不是后来的一次宽 QKV GEMM | U1，micro + 初轮 A/B | 三次 Hgemm 合计 `44.760 -> 42.260` | `1672.753 -> 1678.140`；差值接近漂移，不单独报稳定增幅 | 开关前后 8192 行统计逐项一致 |
| [2026-08-03 22:32:54](/data/wangyize/katago/profiles/cuda-fa4-qkv-residual-b19-current.nsys-rep) | attention out projection 与 FFN linear2 使用 GEMM `beta=1` 直接写回 residual；删除 66 个独立 residual-add kernel | 保留 | U1，micro + 直接 A/B | linear2 `42.263 -> 36.268`；outProj `22.654 -> 14.941` | `1683.316 -> 1751.194`（`+4.03%`） | 8192 行所有统计字段逐项一致 |
| [2026-08-03 23:21:06](/data/wangyize/katago/results/ffn-dual-gemm-sm120-report.md) | CUTLASS shared-A dual GEMM + SwiGLU；替代 `linear1`、`gate` 两次 GEMM 和独立 SwiGLU | 当时保留，后被 TileLang 版本替代 | U1，直接 A/B | 子图 `103.8 -> 63.75`；整网 Nsys 中位数 `64.45` | `1759.678 -> 2037.232`（`+15.77%`） | 8192 行全部 head 通过 |
| [2026-08-03 23:37:39](/data/wangyize/katago/profiles/cuda-fa4-qkv-residual-dualffn-qkrope-b19.nsys-rep) | Q、K 两次 learnable RoPE 合并成一次 kernel，共享同一组 FP32 `sincos` | 保留 | U1，直接 A/B + Nsys | `2 x 8.128 -> 7.872` | `2043.9 -> 2108.0` | 完整 8192 行 FP32 对照通过 |
| [2026-08-04 01:08:01](/data/wangyize/katago/profiles/rope-batch-shared-b19.nsys-rep) | batch-shared RoPE：每个 position/head/pair 只算一次 FP32 `sincos`，再遍历 B19 | 进入优化分支；后续 U2 配对确认其双流调度收益 | U1，热稳态交错 + Nsys | `8.352 -> 7.328` | `2090.8 -> 2110.3` | 数值公式和 FP32 旋转顺序不变 |
| [2026-08-04 01:48:20](/data/wangyize/katago/profiles/rms-ept3-ordered-b19.nsys-rep) | ordered-EPT3 RMSNorm：减少线程数并保留 EPT1 的 FP32 归约顺序 | 保留 | U1，Nsys + ACC | `10.240 -> 7.424` | 未记录 | [8192 行结果](/data/wangyize/katago/results/cuda-qk16-rms-ept3-ordered-b19-accuracy.json)通过，policy top-1 `99.7925%` |
| [2026-08-04 01:52:32](/data/wangyize/katago/profiles/silu-half2-b19.nsys-rep) | CScaleBias+SiLU 使用 half2 成对向量化 | 保留 | U1，首轮 A/B + Nsys | `12.657 -> 7.553` | `2138.9 -> 2179.6`（首轮；最终由下方 bundle 复核） | [8192 行结果](/data/wangyize/katago/results/cuda-qk16-rms-ordered-silu-half2-b19-accuracy.json)通过 |
| [2026-08-04 01:58:37](/data/wangyize/katago/results/cuda-qk16-rms-ordered-silu-half2x3-b19-accuracy.json) | SiLU 每线程处理三个 half2，使主要 shape 使用 512-thread block | 保留；正反序相对 half2 约 `+0.36%` 至 `+0.64%` | U1 | 未记录 | 未记录 | 与 half2 版本逐字节一致 |
| [2026-08-04 02:20:12](/data/wangyize/katago/results/cuda-qk16-optimized-b19-20260804.json) | qk16 FA4 + strided-batched QKV + 融合 Q/K RoPE + batch-shared RoPE + beta=1 residual + CUTLASS dual-FFN + ordered-EPT3 RMSNorm + half2x3 SiLU | 前期单流完整 checkpoint | U1 | 未记录（整套 bundle） | `2170.146` | policy top-1 `99.7925%`；optimistic top-1 `99.7437%` |
| [2026-08-04 14:35:24](/data/wangyize/katago/results/benchmarknn-grid-b1-32-s1-4-pmon/raw/tensorrt-s1-b19.json) | TensorRT 10.16.1.11，S1/B19，FP16 | 在 B1-B32/S1-S4 grid 中复测单流基线 | GRID | `_gemm_mha_v2` `34.78` | `1940.138` | 使用同一精度脚手架比较 |
| [2026-08-04 15:01:37](/data/wangyize/katago/results/benchmarknn-grid-b1-32-s1-4-pmon/raw/tensorrt-s2-b19.json) | TensorRT 10.16.1.11，S2/B19，总并发 38 | TensorRT 双流基线 | GRID | 未记录 | `2038.553`；相对 S1/B19 高 `5.07%` | 不改变数值 |
| [2026-08-04 15:01:56](/data/wangyize/katago/results/benchmarknn-grid-b1-32-s1-4-pmon/raw/cuda_optimized-s2-b19.json) | 上述 `2170` bundle；每个 NN server 持有独立 non-blocking CUDA stream；Nsys 观察到 stream 13/14 各 7,702 次 launch | 当时 grid 最优拓扑 S2/B19；尚不含 both16、宽 QKV AOT、TileLang FFN及其后的优化 | GRID | 未记录（整套 bundle） | `2425.604` | 不改变数值 |
| [2026-08-04 15:02:15](/data/wangyize/katago/results/benchmarknn-grid-b1-32-s1-4-pmon/raw/cuda_original-s2-b19.json) | 官方 CUDA 配置为 S2/B19 的尝试 | **不是有效的双流基线**；两个 server 仍落在 legacy/default stream | GRID | 未记录 | `1628.601`；不得与独立双流结果计算同条件增幅 | 不改变数值 |
| [2026-08-04 16:59:12](/data/wangyize/katago/results/cuda-qk16-qkv-cute-c384-b19-final.json) | 真正的宽 QKV AOT：一次 `C384 -> QKV1152`；固定 CuTe `128x128x64` | 保留 | U2 | CuTe QKV `40.13` | `2400.850 -> 2450.227`；正序 `+2.06%`，逆序约 `+1.53%` | 聚合精度与前一路径完全一致 |
| [2026-08-04 17:16:54](/data/wangyize/katago/results/cuda-qk16-qkv-cute-c384-b19-rope-batchshared-final.json) | 在宽 QKV/U2 路径重新配对确认 batch-shared RoPE | 保留；收益来自减少 CTA 数与双流 SM 竞争，不是单 kernel 更短 | U2 | per-batch `10.273 ->` batch-shared `13.152` | 长跑 `2437.321 -> 2474.380`（`+1.52%`）；短跑 `+0.96%`；S1 `-0.20%` | 8192 行全部 head 通过 |
| [诊断汇总：2026-08-04 17:32:00](/data/wangyize/katago/results/cuda-nvidia-profile-qkv-c384-report.md) | 重新验证 beta/epilogue residual 融合 | 继续保留 | U2 | scalar residual `13.48`；half2 residual `10.89`；融合路径无独立 add kernel | 禁用融合 `2332.90`；half2 fallback `2405.24`；融合路径约 `2450` | 数学语义不变 |
| [2026-08-04 17:36:10](/data/wangyize/katago/trainingdata/accuracy/2026-08-04-19x19-8192-seed20260803-p0loss.manifest.json) | 在固定 8192 行语料中接入 p0loss | 保留 | ACC | 未记录 | 未记录 | 从此精度门槛覆盖全部 head 和 p0loss |
| [2026-08-04 19:11:40](/data/wangyize/katago/results/cuda-both16-tilelangffn-b19-s2-benchmark-reverse.json) | TileLang FFN：`M128-N64-K32-S2-T128`、两级流水、min-blocks 3、half2 tanh SwiGLU、二维 swizzle 输出 | 保留；替代 CUTLASS dual FFN | U2，正序/逆序 | TileLang `60.843` | CUTLASS `2488.437 ->` TileLang `2537.979`（`+1.991%`） | 独立 FP32 RMSE `2.179e-5`；整网 8192 行门槛通过 |
| [2026-08-04 20:21:18](/data/wangyize/katago/results/cuda-both16-tilelangffn-linear2aot-direct-b19-s2-confirm-reverse.json) | linear2 + residual：CUTLASS `128x128x32`、stage 3、128 threads、49.15 KiB；直接 residual epilogue | 保留 | U2，配对 | `34.05 -> 30.77` | `2534.096 -> 2592.216`（`+2.294%`） | 与 CUDA baseline bit-identical；all-head+p0loss JSON 字节一致 |
| [2026-08-04 20:56:56](/data/wangyize/katago/results/cuda-both16-linear2-outproj-aot-b19-s2-paired-reverse.json) | attention out projection + residual 固定 AOT，直接 residual epilogue | 保留 | U2，配对 | 未记录 | `2584.370 -> 2615.048`（`+1.187%`） | 8192 行通过 |
| [2026-08-04 23:26:54](/data/wangyize/katago/results/cuda-current-baseline-b19-s2-locked2430-pl380.json) | 锁频后的当前基线：both16 FA4、宽 QKV AOT、batch-shared RoPE、融合 residual、TileLang FFN、linear2/out-projection AOT | L 制度起点 | L | 未记录（整套 bundle） | `2619.734` | p0loss 脚手架通过 |
| [2026-08-04 23:57:42](/data/wangyize/katago/results/cuda-current-rmsnorm-warp4-vec8-c384-b19-s2-locked2430-pl380.json) | C384 RMSNorm：每 block 四个 warp、每 warp 一行、vec8 `uint4+uint2` load、warp shuffle reduction | 保留 | L，顺序 checkpoint | 未记录（没有同制度单-kernel 配对值） | `2619.734 -> 2725.906`（`+4.053%`） | p0loss `1.592262594`；policy top-1 `99.7314%`；optimistic top-1 `99.6704%` |
| [2026-08-05 00:11:16](/data/wangyize/katago/results/cuda-current-rmsnorm-warp4-vec8-scalebias-silu-c768-flat256-b19-s2-locked2430-pl380.json) | C768 affine + SiLU：一维 flat launch，block 256，每线程处理一个 vec8/half8 group | 保留 | L | NCU `15.840 -> 14.208` | `2725.906 -> 2731.790`（`+0.216%`） | 聚合精度与前一候选一致 |
| [2026-08-05 00:35:25](/data/wangyize/katago/results/cuda-current-persisting-l2-trunk24m-b19-s2-locked2430-pl380.json) | 长生命周期 C768 trunk residual 的 per-stream persisting-L2 access-policy window | 保留 | L | 未记录 | `2731.790 -> 2740.127`（`+0.305%`） | 只改变 cache policy，不改变算术 |
| [2026-08-05 00:48:41](/data/wangyize/katago/results/cuda-current-persisting-l2-trunk24m-rope-half2-b19-s2-locked2430-pl380.json) | half2 RoPE：成对向量化 load/store/rotation | 保留 | L | 未记录 | `2740.127 -> 2757.380`（`+0.630%`） | 8192 行通过 |
| [2026-08-05 02:25:42](/data/wangyize/katago/results/cuda-current-persisting-l2-inner-c384-b19-s2-locked2430-pl380.json) | 增加 C384 inner residual 的 persisting-L2 生命周期管理 | 保留 | L | 未记录 | `2757.380 -> 2779.610`（`+0.806%`） | 只改变 cache policy |
| [2026-08-05 03:20:51](/data/wangyize/katago/results/cuda-current-outer-projection-aot-b19-s2-locked2430-pl380.json) | outer C768/C384 projection 固定 CUTLASS AOT expand/contract kernel | 保留 | L，直接 control/candidate | 未记录 | `2783.711 -> 2817.070`（`+1.198%`） | all-head+p0loss 与前一路径一致 |
| [2026-08-05 04:05:33](/data/wangyize/katago/results/cuda-shared-matmul-abba4-control-b19-s2-locked2430-pl380.json) | 两个 NN server 共享普通 matmul 权重，缩小 L2 working set | 保留；宽 QKV 与 outer projection 特殊权重仍私有 | L，ABBA | 未记录 | `2818.446 -> 2822.008`（`+0.126%`） | 算术不变 |
| [2026-08-05 05:07:34](/data/wangyize/katago/results/cuda-rope-unroll19-long-abba-control2-b19-s2-locked2430-pl380.json) | 固定 B19 的 RoPE batch loop 静态展开 | 保留 | L，长 ABBA | 未记录 | `2816.927 -> 2824.929`（`+0.284%`） | 8192 行通过 |
| [2026-08-05 06:15:47](/data/wangyize/katago/results/cuda-qkv-atom4x2-long-abba-summary.json) | QKV CuTe copy atom 从 2x2 改为 4x2 | 保留 | L，ABBA | `39.325 -> 37.717` | `2828.845 -> 2839.390`（`+0.373%`） | padded-reference p0loss `1.592262288`；全部 head 通过 |
| [2026-08-05 06:51:39](/data/wangyize/katago/results/cuda-outer-warp64x32-long-abba-summary.json) | C384->C768 outer-expand warp shape 改为 64x32 | 保留 | L，ABBA | 未记录 | `2837.917 -> 2840.140`（`+0.078%`） | 聚合精度与前一候选一致 |
| [2026-08-05 07:22:12](/data/wangyize/katago/results/cuda-initialconv-frontend-long-abba-summary.json) | 初始卷积使用固定 B19 cuDNN frontend engine 45、tile size 0、stages 2 | 保留 | L，ABBA | frontend kernel `36.417` | `2835.976 -> 2849.036`（`+0.461%`） | 与前一 CUDA 输出 bit-exact |
| [2026-08-05 08:48:46](/data/wangyize/katago/results/cuda-fused-policy-p1-reverse-baab-summary.json) | 融合 policy P1 小图 | 保留 | L，ABBA + 逆序 | 融合 kernel `3.777`；原三-kernel 子图未记录 | 正序 `2847.834 -> 2853.264`（`+0.191%`）；逆序 `2841.196 -> 2846.008`（`+0.169%`） | 语义不变 |
| [2026-08-05 09:03:09](/data/wangyize/katago/results/cuda-head-bn-half-to-float-reverse-baab-summary.json) | head batch-normalization 直接消费 FP16 并输出 FP32，删除中间转换 | 保留 | L，ABBA + 逆序 | 融合 kernel `5.408`；原子图未记录 | 正序 `2847.748 -> 2854.396`（`+0.233%`）；逆序 `2842.729 -> 2847.335`（`+0.162%`） | 重新验证最终整网精度 |
| [2026-08-05 09:50:26](/data/wangyize/katago/results/cuda-initial-global-matmul-add-fp32dot-cbbc-4-candidate-b19-s2-locked2430-pl380.json) | 融合 global-feature dot 与 spatial broadcast-add；为精度保留 FP32 dot | 保留 | L | 未记录 | 第一版 `2857.798 -> 2862.815`（`+0.176%`）；FP32-dot `2857.765 -> 2861.736`（约 `+0.139%`） | 最终 p0loss 与全部 head 门槛通过 |
| [2026-08-05 10:51:34](/data/wangyize/katago/results/cuda-wide-head-nosplit-abba-summary.json) | policy/value head 使用一次不拆分中间输出的宽投影 | 最终保留项 | L，ABBA | 未记录 | `2855.676 -> 2862.953`（`+0.255%`） | 最终全 FP32 对比见下文 |

同一锁频制度内的连续区间为 `2619.734 -> 2862.953 nnEval/s`，可复现提升
`+9.284%`，B19x2 拓扑和时钟/功耗策略保持一致。官方 U1 的 `1479.162` 与最终 L 结果不能
合并成一个优化百分比，因为 NN server 拓扑和时钟策略都不同。

## 被否决或暂停的路线

此表沿用主时间线的结果口径。百分比旁有原始绝对值时一并列出；没有直接保存的结果写“未记录”。

| 结果保存时间（UTC+8） | 路线 | 单 kernel / 子图耗时（us） | 端到端吞吐（nnEval/s） | 决定与原因 |
|---|---|---:|---:|---|
| [2026-08-03 16:54:03](/data/wangyize/katago/results/cuda-custom-flash-b19-fp16-check.json) | 第一版手写 custom flash kernel | 未记录 | 官方 CUDA `1479.162 ->` custom flash `980.156` | 删除，改用 FA4/CuTe |
| [2026-08-03 23:16:22](/data/wangyize/katago/results/qflash-sm120-benchmark-report.md) | QFlash D32 | 完整调用 `207.056`；当时 TRT attention 约 `34.8` | 未记录 | 不接入 |
| [2026-08-03 23:21:06](/data/wangyize/katago/results/trt-fa4-plugin-report.md) | TensorRT + 窄边界 FA4 plugin | 未记录 | 原生 TRT `1942.597`；连续布局 plugin `758.165`；stride-368 plugin `892.915` | 否决。33 个 plugin layer 每层引入三次输入 reformat 和一次输出 reformat，并破坏 RMSNorm/QKV/RoPE/MHA/out-projection 周围的 Myelin 融合 |
| [2026-08-03 23:21:06](/data/wangyize/katago/results/trt-fa4-plugin-report.md) | TensorRT profile `{19,19,19}` 或全静态 B19 network | 未记录 | dynamic/fixed/static 长跑中位数 `1853.210 / 1856.426 / 1850.920` | 只常量折叠 shape plumbing，没有改善主要计算 kernel；否决 |
| [2026-08-03 23:31:54](/data/wangyize/katago/results/sageattention-d32-sm120-experiment.md) | SageAttention true-D32 | 完整调用 `100.816` | 未记录 | 不接入 |
| [2026-08-04 17:28:12](/data/wangyize/katago/results/cuda-qk16-qkv-cute-c384-b19-s2-graph-reverse-confirm.json) | CUDA Graph | 未记录 | 普通 `2468.444`；graph `2464.397`（`-0.164%`） | host launch 不是瓶颈；graph 还会限制未来的分相调度；否决 |
| [2026-08-04 21:33:49](/data/wangyize/katago/results/cuda-both16-linear2-outproj-aot-rmsffn-folded-tilelang-fp32scale-b19-s2-paired-reverse.json) | RMSNorm -> FFN 代数折叠 | 未记录 | 绝对配对值未完整保存；保精度版本 `-0.016%` | 早期 S1 看似正收益，但改变 FP16 舍入并移动多个 max；禁用 |
| [2026-08-04 23:22:33](/data/wangyize/katago/results/cuda-both16-qkvgemm-rope-preload2-b19-s2-locked2430-abba4-control.json) | QKV GEMM + RoPE 合成一个 AOT kernel | 未记录 | 早期配对 `2627.394 -> 2554.454`；锁频 ABBA `2613.630 -> 2586.493`（`-1.038%`） | 否决，保留分离的 QKV 与 RoPE kernel |
| 无独立 benchmark；[审计记录：2026-08-05 00:10:52](/data/wangyize/katago/results/trt-engine-plan-optimization-audit.md) | SM120 DSMEM / clustered launch | 未记录 | 未记录 | 当前网络的数据生命周期中没有可由多个 SM 稳定复用的 tile；TRT NCU 的 cluster-launch 指标也为零 |
| 无结果时间；仅有[架构记录：2026-08-05 01:19:03](/home/wangyize/.katago/src/KataGo-v1.17.1/docs/cuda-dual-trunk-phase-scheduling.md) | 双流 trunk 分相调度 | 未记录 | 未记录 | 允许 feature extraction/head 自由重叠，手动控制进入 trunk 的固定最优相位；尚未实现 |
| [2026-08-05 01:20:33](/data/wangyize/katago/results/c384-row480-paired-candidate-row480-b19-s2-locked2430-pl380.json) | 更激进的 C384 vec8/row480 schedule | 未记录 | vec8 `2759.472 -> 2731.307`；row480 `2761.255 -> 2728.839` | 否决 |
| [2026-08-05 03:50:56](/data/wangyize/katago/results/cuda-projection-swizzle-abba4-sw1-b19-s2-locked2430-pl380.json) | outer projection swizzle 8 | 未记录 | swizzle 1 `2814.988 ->` swizzle 8 `2814.556`（`-0.015%`） | 无有效信号，否决 |
| [2026-08-05 04:17:05](/data/wangyize/katago/results/cuda-shared-all-weights-abba4-control-b19-s2-locked2430-pl380.json) | 同时共享宽 QKV 与 outer-projection 特殊权重 | 未记录 | control `2820.717 ->` share-all `2821.495`（`+0.028%`） | component cross-check 无稳定收益；只保留普通 matmul 权重共享 |
| [2026-08-05 04:56:45](/data/wangyize/katago/results/cuda-rmsfold-qkv-abba-control2-b19-s2-locked2430-pl380.json) | 把 RMSNorm scale 折叠进 QKV | 未记录 | control `2824.528 ->` folded `2814.547`（`-0.353%`）；atom4x2 后 `-0.445%` | 否决 |
| [2026-08-05 05:03:26](/data/wangyize/katago/results/cuda-rope-twoway-abba-control2-b19-s2-locked2430-pl380.json) | 双向 RoPE schedule | 未记录 | one-way `2816.287 ->` two-way `2815.727`（`-0.020%`） | 否决 |
| [2026-08-05 07:38:51](/data/wangyize/katago/results/qkv-atom4x2-two-stream-stage-abba.json) | 更低 stage 或 persistent QKV | 双缓冲 `33.54`；最佳 persistent `34.56` | 未记录 | 正确的低 stage 版本更慢；否决 |
| [2026-08-05 08:06:36](/data/wangyize/katago/results/cuda-outer-fused-silu-long-abba-summary.json) | outer projection fused-SiLU EVT | 未记录 | `2846.370 -> 2787.821`（`-2.057%`） | 否决 |
| [2026-08-05 08:28:51](/data/wangyize/katago/results/cuda-wide-head-projection-long-abba-summary.json) | 第一版宽 head | 未记录 | `2847.053 -> 2847.827`（`+0.027%`） | 与噪声同量级；后续 no-split 版本独立验证后以 `+0.255%` 保留 |
| [2026-08-05 08:40:15](/data/wangyize/katago/results/cuda-fused-head-pooling-reverse-baab-summary.json) | 融合 head pooling | 未记录 | 正序 `2845.066 -> 2851.082`（`+0.211%`）；逆序 `2847.551 -> 2847.637`（`+0.003%`） | 属于运行顺序漂移，否决 |
| [2026-08-05 09:18:55](/data/wangyize/katago/results/cuda-initialconv-bias-frontend-explicituid-vs-current-b19-accuracy.json) | 初始卷积 bias frontend | 未记录 | 未记录 | 正确性路径存在，但没有可保留的整网性能证据；禁用 |
| [2026-08-05 10:31:26](/data/wangyize/katago/results/cuda-fa4-n48-long-abba-summary.json) | FA4 N96 / N48 | N64 `33.95`；N96 `33.60`；N48 `35.97`（NCU） | N96 `2851.662 -> 2828.052`（`-0.828%`）；N48 `2850.671 -> 2819.618`（`-1.089%`） | 保留 N64；双流资源重叠比隔离延迟更重要 |
| [2026-08-05 11:00:17](/data/wangyize/katago/results/cuda-ffn-stage1-mb3-vs-stage2-abba-summary.json) | FFN stage1 MB3/MB4、persistent、late-prefetch | 未记录 | stage2/MB3 `2869.089 ->` stage1/MB3 `2853.927`（`-0.528%`） | MB4/persistent 虽提高 occupancy，却降低 tensor SOL；保留 stage2/MB3 |
| [2026-08-05 12:03:29](/data/wangyize/katago/results/qkv-m64-atom4x2-two-stream-abba.json) | QKV M64 tile | 当前 M128 双流 `68.629`；M64 候选 `80.912-84.684` | 未记录 | 结果 bit-identical；额外并行度抵不过效率损失，否决 |
| [2026-08-05 12:07:27](/data/wangyize/katago/results/fa4-both16-packedqkv-q-in-regs-two-stream-abba.json) | FA4 把 Q 从 shared memory 改放寄存器 | 双流 shared-Q `33.661`；register-Q `34.114`（慢 `1.345%`） | 未记录 | 结果 bit-identical；否决 |
| [2026-08-05 12:08:33](/data/wangyize/katago/results/fa4-sm120-both16-scaled-warps-n64-b19-s361-h12-d32.json) | FA4 M128 / 256-thread 扩展 | M64/128-thread `32.952`；M128/256-thread `35.091`（慢约 `6.5%`） | 未记录 | 误差相同；否决 |
| 无结果时间 | BF16 | 未记录 | 未记录 | 未 benchmark；现有精度损失以尾数舍入为主，而 BF16 尾数精度低于 FP16，按决定暂停 |

## 最终保留状态

保留的二进制：

```text
/data/wangyize/katago/build/cuda-candidate-controls/katago-wide-head-nosplit-accepted
SHA256 4cd3769e42542df2c718ee7e7c734b3bfff25fdf79f75cb44cd982f6b783a87a
```

保留的配置：

```text
numNNServerThreadsPerModel=2
useFP16=true
cudaFlashAttentionSm120Accum=both16
cudaUseWideQKV=true
cudaUseQKVGemmAot=true
cudaUseQKVGemmRopeAot=false
cudaUseFusedQKRoPE=true
cudaUseBatchSharedRoPE=true
cudaUseBatchSharedRoPEUnroll19=true
cudaUseBatchSharedRoPETwoWay=false
cudaUseFusedResidual=true
cudaUseProjectionGemmLt=false
cudaUseFusedFFN=true
cudaUseFusedRMSNormFFN=false
cudaUseRMSNormQKVGemmAot=false
cudaUseGraph=false
cudaUsePersistingL2Trunk=true
cudaUsePersistingL2Inner=true
cudaUseOuterProjectionAot=true
cudaShareModelWeights=true
cudaShareWideQKVWeights=false
cudaShareOuterProjectionWeights=false
cudaUseInitialConvFrontend=true
cudaUseInitialConvBiasFrontend=false
cudaUseInitialGlobalMatMulAdd=true
cudaUseFusedPolicyP1=true
cudaUseHeadBNHalfToFloat=true
cudaUseWideHeadProjection=true
```

最终锁频 ABBA 的各 run 中位数均值：

```text
2 个 NN server x B19 = 38 个并发 evaluation
2862.953 nnEval/s
每个并发 batch 13.273 ms
```

最终 CUDA 相对全 FP32 的 8192 行绝对精度：

| 指标 | 最终 CUDA | FP32 参考（适用时） |
|---|---:|---:|
| p0loss | `1.5922355867` | `1.5922696173` |
| policy top-1 一致率 | `99.74365%` | 100% |
| optimistic-policy top-1 一致率 | `99.68262%` | 100% |
| policy probability RMSE | `1.04413e-4` | 0 |
| outcome RMSE | `9.80829e-3` | 0 |
| score RMSE | `7.21478e-3` | 0 |
| ownership RMSE | `3.18547e-3` | 0 |

最终双流 critical-path union 中，剩余独占 GPU busy 时间排序如下：

| 类别 | 独占 busy 占比 |
|---|---:|
| 融合 FFN 输入投影 + SwiGLU | 24.215% |
| 宽 QKV | 12.024% |
| FA4 | 9.323% |
| outer C384->C768 expand | 2.465% |
| FFN linear2 | 0.833% |
| heads 及其他 | 0.589% |
| attention output projection | 0.477% |
| 初始卷积 | 0.413% |
| outer C768->C384 contract | 0.403% |

以上百分比使用两条 stream 的时间区间 union/exclusive 归因，不是逐 kernel duration 求和，
不能直接代入简单的 Amdahl 估算。

## 关键证据索引

- 官方 CUDA baseline：[cudnn-9.24-cuda13.2-b19-fp16-final.json](/data/wangyize/katago/results/cudnn-9.24-cuda13.2-b19-fp16-final.json)
- CUDA/TensorRT 初始 benchmark 方法：[README.md](/data/wangyize/katago/results/README.md)
- 固定精度语料 manifest：[2026-08-04-19x19-8192-seed20260803-p0loss.manifest.json](/data/wangyize/katago/trainingdata/accuracy/2026-08-04-19x19-8192-seed20260803-p0loss.manifest.json)
- cuDNN SDPA compute-type 审计：[cudnn-sdpa-compute-type-research.md](/home/wangyize/.katago/src/KataGo-v1.17.1/docs/cudnn-sdpa-compute-type-research.md)
- FA4 接入、累加模式比较和完整 head 指标：[flash-attention-sm120-aot-integration-report.md](/data/wangyize/katago/results/flash-attention-sm120-aot-integration-report.md)
- 同一扫描的原始 CUDA / 优化 CUDA / TensorRT：[summary.json](/data/wangyize/katago/results/benchmarknn-grid-b1-32-s1-4-pmon/summary.json)
- 独立 NN server stream trace：[benchmarknn-b19-s2-owned-streams-20260804.sqlite](/data/wangyize/katago/profiles/benchmarknn-b19-s2-owned-streams-20260804.sqlite)
- 宽 QKV、batch-shared RoPE、residual 与 CUDA Graph 报告：[cuda-nvidia-profile-qkv-c384-report.md](/data/wangyize/katago/results/cuda-nvidia-profile-qkv-c384-report.md)
- 融合 FFN 历史：[ffn-dual-gemm-sm120-report.md](/data/wangyize/katago/results/ffn-dual-gemm-sm120-report.md)
- TileLang FFN handover 与 NCU 审计：[tilelang-sm120-fused-ffn-handover.md](/home/wangyize/.katago/src/KataGo-v1.17.1/docs/tilelang-sm120-fused-ffn-handover.md)
- Linear2 + residual handover 与配对结果：[tilelang-sm120-linear2-residual-handover.md](/home/wangyize/.katago/src/KataGo-v1.17.1/docs/tilelang-sm120-linear2-residual-handover.md)
- TensorRT FA4 plugin 与 static/fixed-profile 失败记录：[trt-fa4-plugin-report.md](/data/wangyize/katago/results/trt-fa4-plugin-report.md)
- TensorRT plan 审计及 L2/RMSNorm tactic 证据：[trt-engine-plan-optimization-audit.md](/data/wangyize/katago/results/trt-engine-plan-optimization-audit.md)
- 锁频制度起点：[cuda-current-baseline-b19-s2-locked2430-pl380.json](/data/wangyize/katago/results/cuda-current-baseline-b19-s2-locked2430-pl380.json)
- QKV atom4x2 ABBA：[cuda-qkv-atom4x2-long-abba-summary.json](/data/wangyize/katago/results/cuda-qkv-atom4x2-long-abba-summary.json)
- 初始卷积 frontend ABBA：[cuda-initialconv-frontend-long-abba-summary.json](/data/wangyize/katago/results/cuda-initialconv-frontend-long-abba-summary.json)
- 最终 wide-head no-split ABBA：[cuda-wide-head-nosplit-abba-summary.json](/data/wangyize/katago/results/cuda-wide-head-nosplit-abba-summary.json)
- 最终全 FP32 精度对比：[cuda-wide-head-nosplit-b19-vs-fp32-accuracy.json](/data/wangyize/katago/results/cuda-wide-head-nosplit-b19-vs-fp32-accuracy.json)
- 最终双流结构化 critical path：[cuda-wide-head-nosplit-nsys-critical-path.json](/data/wangyize/katago/results/cuda-wide-head-nosplit-nsys-critical-path.json)
- 量化 attention 负结果：[attention-quantization-benchmark-summary.md](/data/wangyize/katago/results/attention-quantization-benchmark-summary.md)
- 未来双 trunk 分相调度方向：[cuda-dual-trunk-phase-scheduling.md](/home/wangyize/.katago/src/KataGo-v1.17.1/docs/cuda-dual-trunk-phase-scheduling.md)
