# KataGo b11 benchmarknn baseline (S2)

> 已废弃：本目录中的数字是在修复 CUDA legacy stream 问题之前测的，
> 不可作为 baseline。请以
> [/workspace/results/baseline-fixed/HISTORY.md](/workspace/results/baseline-fixed/HISTORY.md)
> 为准。

建立时间：2026-08-05（UTC）

## 测量协议

- 模型：[b11c768h12nbt3tflrs-fson-silu.bin.gz](/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz)
  （SHA-256 `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`）
- 二进制：`/workspace/katago/build-cuda/katago`，revision
  `e1a88a4d394ee4f1fd790ea2bc66b8b8266b5799-cuda`
- 子命令：`benchmarknn`（纯 GPU forward，排除预处理/后处理/H2D/D2H/search）
- 拓扑：每 GPU 2 个 NN server 线程（S2），各绑定私有 CUDA stream
- 输入：FP16，B19/B24/B28/B36/B38 见下文，19x19
- 长确认：每轮 10 次 warmup + 300 次 timed forward
- 时钟策略：未锁频；每轮前后记录 `nvidia-smi` 环境 CSV
- CUDA 设备映射：CUDA 0 = RTX 4090（nvidia-smi 0），
  CUDA 2 = RTX 5090 D（nvidia-smi 1）

## Batch 扫描（每档 100 次迭代，S2）

### RTX 4090

| B | nnEval/s |
|---:|---:|
| 1 | 308.8 |
| 2 | 677.0 |
| 4 | 991.7 |
| 8 | 1301.1 |
| 12 | 1520.5 |
| 16 | 1536.2 |
| 19 | 1601.2 |
| 20 | 1599.7 |
| 22 | 1616.4 |
| 24 | 1642.6 |
| 25 | 1656.6 |
| 26 | 1658.0 |
| 27 | 1656.1 |
| 28 | 1658.4 |
| 29 | 1635.9 |
| 30 | 1640.3 |
| 32 | 1607.9 |
| 40 | 1596.3 |
| 48 | 1547.2 |
| 64 | 1527.7 |

峰值平台约在 B24-B28。按“差距不大取更小 batch”，选定 **B24**。

### RTX 5090 D

| B | nnEval/s |
|---:|---:|
| 1 | 297.4 |
| 2 | 774.9 |
| 4 | 1301.5 |
| 8 | 1717.9 |
| 12 | 1976.1 |
| 16 | 2102.4 |
| 19 | 2160.0 |
| 24 | 2080.5 |
| 32 | 2136.3 |
| 36 | 2172.3 |
| 37 | 2183.0 |
| 38 | 2190.5 |
| 39 | 2174.9 |
| 40 | 2169.9 |
| 42 | 2137.9 |
| 44 | 2135.4 |
| 48 | 2144.1 |
| 64 | 2103.0 |

原始峰值在 B38，但 B19/B36/B38 差距不大；按偏好取更小 batch，
且长跑中 B19 与 B36 几乎持平（ABBA 差约 0.1%），选定 **B19**。

## 长确认（300 次迭代，3 轮）

| GPU | B | 3 轮 nnEval/s | 中位数 |
|---|---|---:|---:|
| RTX 4090 | 24 | 1662.70 / 1642.08 / 1640.73 | **1642.08** |
| RTX 4090 | 28 | 1650.95 / 1652.34 / 1648.37 | 1650.95 |
| RTX 5090 D | 19 | 2148.85 / 2142.45 / 2143.65 | **2143.65** |
| RTX 5090 D | 36 | 2156.83 / 2153.06 / 2149.06 | 2153.06 |
| RTX 5090 D | 38 | 2192.40 / 2184.79 / 2180.49 | 2184.79 |

ABBA 交错复核：

| GPU | 对比 | ABBA 中位数 |
|---|---|---:|
| RTX 4090 | B24 vs B28 | 1652.35 vs 1658.50（+0.37%） |
| RTX 5090 D | B19 vs B36 | 2159.23 vs 2162.11（+0.13%） |

## Baseline 结论

- RTX 4090：**S2 / B24**，约 **1642 nnEval/s**
- RTX 5090 D：**S2 / B19**，约 **2144 nnEval/s**

配置：

- `/workspace/bench-cuda-gpu0-4090-s2.cfg`
- `/workspace/bench-cuda-gpu2-5090d-s2.cfg`

## 工件

- 扫描原始结果：`/workspace/results/baseline/scan/`
- 长确认 JSON/raw/env：`/workspace/results/baseline/final/`
- 环境记录见各 `env-before.csv` / `env-after.csv`

## 备注

- 4090 的 B12 在计时完成后清理阶段出现 `double free or corruption`
  （进程退出码 134），但 JSON 已正常输出；该 batch 不在候选范围。
- 5090D 若只看原始峰值，B38 比所选 B19 高约 1.9%；按用户偏好
  （差距不大取更小 batch）最终选 B19。
