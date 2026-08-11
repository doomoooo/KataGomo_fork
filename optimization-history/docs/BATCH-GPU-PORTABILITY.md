# 固定 19x19 优化的 Batch 与 GPU 可迁移性

## 1. 范围

本文审计当前 RTX 5090 D、固定 19x19、B13、双流 S2 配置中的已启用优化，回答：

1. 哪些优化会在 batch 改变后立即退出专用路径；
2. 哪些优化会在 GPU 改变后立即退出；
3. 哪些收益机制可以稳定继承；
4. 对失效或不再最优的项目，哪些可以用小范围参数搜索恢复。

当前目标配置为：

- `bench-cuda-gpu2-5090d-s2.cfg`
- board：固定 `19x19`，即 `S=361`
- batch：`B=13`，token 数 `M=13×361=4693`
- topology：两个独立 NN server / CUDA stream
- precision/layout：FP16、NHWC
- architecture：SM120

本文只讨论固定 19x19。改变棋盘尺寸属于另一套 shape，不在本文的泛化范围内。

## 2. 必须区分的两种 batch 改动

“改变 batch”有两种不同含义。

### 2.1 只改变实际 forward batch

例如仍以 `nnMaxBatchSize=13` 创建模型，但某次 forward 使用 B12。

结果：

- 固定 B13 的 FFN、QKV、linear2 AOT 立即不再使用；
- persisting-L2 仍会安装，因为它的构造 gate 检查的是 max batch；
- FA4、RMSNorm、RoPE、affine-SiLU 等动态 batch kernel 继续使用。

### 2.2 同时改变 `nnMaxBatchSize`

例如模型直接以 `nnMaxBatchSize=12` 创建。

结果：

- 固定 B13 的三个 AOT 路径不再使用；
- trunk/inner persisting-L2 也立即禁用，因为当前实现明确要求
  `maxBatchSize == 13`；
- 其他动态 batch kernel 不会仅因 max batch 改变而退出。

因此，讨论可迁移性时必须同时记录 actual batch 与 max batch，不能只写一个“batch”。

## 3. 当前生效优化的 batch 可迁移性

| 当前优化 | actual B≠13 | max batch≠13 | fallback 后的路径 | 小范围搜索能否恢复 |
|---|---|---|---|---|
| 融合 FFN + SwiGLU AOT | 立即退出固定 AOT | 立即退出固定 AOT | 官方两次 GEMM，随后可使用独立 half8 SwiGLU | 能，优先级高 |
| FFN A-fragment reuse | 随固定 FFN 一起失效 | 同左 | 不适用 | 能；应保留到新 B 的 FFN kernel 中 |
| 宽 QKV AOT | 立即退出固定 AOT | 立即退出固定 AOT | 当前配置回到三次普通 Q/K/V GEMM | 能，优先级高 |
| linear2 + residual AOT | 固定 AOT 失效 | 固定 AOT 失效 | 通用 cuBLAS `beta=1` residual GEMM | 能，优先级高 |
| GEMM `beta=1` residual 融合 | 继续工作 | 继续工作 | 无需 fallback | 通常不需要搜索 |
| FA4 both16 | B1–B13 内继续工作 | B1–B13 内继续工作 | shape 不满足时回官方 attention | 通常不需要；B>13 时重建/验证 |
| C384 RMSNorm vec8 | 继续工作 | 继续工作 | shape 不满足时回官方路径 | 通常稳定 |
| fused Q/K RoPE half2 | 继续工作 | 继续工作 | shape 不满足时回官方路径 | 通常稳定 |
| C384/C768 affine-SiLU half2 | 继续工作 | 继续工作 | shape 不满足时回官方路径 | 通常稳定 |
| C1152 half8 SwiGLU | 继续工作 | 继续工作 | shape 不满足时回官方路径 | 通常稳定；B13 时被融合 FFN 覆盖 |
| persisting-L2 trunk | max 仍是13时继续工作 | 立即禁用 | 无 access-policy window | 很容易恢复 |
| persisting-L2 inner | max 仍是13时继续工作 | 立即禁用 | 无 access-policy window | 很容易恢复 |

固定 B13 gate 的主要实现位置：

- FFN：`katago/cpp/neuralnet/cudabackend_sm120.cpp:810`
- wide QKV：`katago/cpp/neuralnet/cudabackend_sm120.cpp:972`
- linear2 AOT：`katago/cpp/neuralnet/cudabackend_sm120.cpp:1083`
- persisting-L2 构造：`katago/cpp/neuralnet/cudabackend_sm120.cpp:602`

## 4. GPU 可迁移性

### 4.1 RTX 5090 D → RTX 5080

不会有优化因为显卡名称改变而立即退出。

SM120 backend 只检查 compute capability 是否为 `12.0`，不检查设备名称。5090 D 与
5080 都属于 SM120，因此：

- SM120 kernel 可以继续加载；
- 固定 B13 AOT kernel 仍然功能正确；
- 所有 shape gate 保持不变；
- 只需修改正确的 CUDA device 配置。

架构 gate 位于：

- `katago/cpp/neuralnet/cudabackend_sm120.cpp:250`

但“继续运行”不等于“仍是最优”。两张卡的 SM 数、L2 容量、显存带宽、Tensor/非 Tensor
执行比例和可同时驻留 CTA 数不同，因此以下项目需要重新做整图裁决：

- FFN 的 tile、stage、register/shared-memory 平衡；
- QKV 的 K32/S3 与 K64/S2 选择；
- linear2 的 N tile、stage、CTA 数；
- persisting-L2 的 set-aside、hitRatio、trunk/inner 组合；
- 最终 S1/S2 topology。

### 4.2 SM120 → 非 SM120

若换成 RTX 4090 等非 SM120 GPU，整个 SM120 backend 会退出，回到官方 CUDA backend。
这不是参数搜索可以恢复的问题，需要：

- 为目标架构重新编译/移植 kernel；
- 重新验证可用的 Tensor Core 指令、shared-memory 上限和 AOT ABI；
- 重新建立该卡上的完整 Nsys/NCU 与精度证据链。

## 5. 可以稳定继承的收益机制

这里的“稳定”是指机制通常能跨 batch 或同架构 GPU 保持正确方向，不代表整网百分比恒定。

### 5.1 高稳定性

#### GEMM `beta=1` residual 融合

- 删除独立 residual-add kernel；
- 减少一次中间张量读写和一次 launch；
- 不依赖固定 B13 tile；
- actual batch 改变后仍继续使用通用 cuBLAS 路径。

这是当前最稳定的优化之一。

#### fused Q/K RoPE 与 half2 I/O

- 融合 Q/K 两条相同变换；
- half2 成对 load/store/rotation；
- grid 随 batch 动态扩展；
- 5080 与 5090D 历史均观察到正收益。

#### RMSNorm vec8

- 每一空间行的 C384 shape 不随 batch 改变；
- batch 只改变行数；
- vector load、归约树和 register 布局具有较强可迁移性。

#### affine-SiLU half2 与 SwiGLU half8

- 属于连续逐元素向量化；
- 线程块数量随 batch 线性变化；
- 一般不需要为每个 batch 重新生成 kernel。

### 5.2 机制稳定、具体 tactic 不稳定

#### FA4 both16

FA4 替代官方 attention 的大方向稳定；both16 降低 accumulator 资源和耗时的机制也稳定。
当前 AOT 已按 B13 生成，但运行接口支持并已用于 B1–B13 范围。若超过 B13，需要重新构建、
验证 wave 数和完整精度，不能直接假定继续有效。

#### 融合 FFN

共享 A/input、把两次输入投影与 SwiGLU 融合的机制稳定；但精确 tile、stage 和 CTA 资源占用
非常依赖 `M=B×361`、GPU SM 数与 S2 调度。

A-fragment reuse 本身也具有较强可迁移性：它减少重复 `ldmatrix(A)`，并在当前 kernel 中把
registers/thread 从 146 降到 136。新 batch 的 FFN AOT 应默认保留这一变换，再由整图验证。

#### 宽 QKV

把三次 Q/K/V GEMM 合成一次宽投影、直接生成 planar Q/K/V 的机制稳定；但 K tile、stage、
shared-memory 和 copy mapping 需要随 batch/GPU 重选。

#### linear2 AOT

固定 residual epilogue 的方向稳定，但这类 GEMM 对 tile wave、shared memory 和 peer kernel
干扰非常敏感。应把“保留 AOT family”和“保留当前具体 kernel”分开判断。

#### persisting-L2

trunk/inner residual 的生命周期和复用机制稳定，而且 5080、5090D 历史均测得正收益。
不稳定的是当前硬编码的 max-B13 gate、双流总窗口计算和具体 hitRatio。

## 6. 用小范围搜索恢复失效优化

目标不是为每个 batch 进行大规模穷举，而是用已有成功/失败历史限制候选空间。

### 6.1 FFN

固定：

- C384 → C1152；
- A-fragment reuse；
- FP16 MMA、现有 SwiGLU arithmetic 和输出布局；
- 19x19。

建议仅搜索：

- M tile：`{64,128}`；
- K tile：`{32,64}`；
- stage：`{2,3}`；
- N tile 通常先固定为64，只在小 batch 下增加一个替代值。

每个 batch 生成约3–6个候选即可。先用 S1+NCU 排除明显较慢或资源恶化的候选，再让前二
直接进入真实整图 S2；不使用 homogeneous/mixed 局部 S2 代理。

### 6.2 宽 QKV

优先复用现成的两个主要方向：

- K64/S2：当前 B13/5090D 接受版本；
- K32/S3：更低 shared memory，可能适合不同 batch 或较少 SM 的卡。

对小 batch 再加入一个 M64 版本即可。保持：

- planar Q/K/V 输出；
- 128-thread 基础映射；
- 无额外 reformat/materialize。

每个 batch/GPU 通常只需2–4个候选。

### 6.3 linear2 + residual

建议候选：

- 当前固定版本；
- 5080 历史中的 128×128×32、stage 3、约49 KiB shared-memory 方向；
- 一个更低 shared-memory 的 N96 邻域版本。

每个 batch 约3–5个候选。重点观察：

- CTA wave 是否贴合 SM 数；
- 是否能与 QKV/out-projection 等大 CTA 共驻；
- registers、shared memory、eligible cycles；
- 真实整图 S2，而不是孤立或同构双流结果。

### 6.4 persisting-L2

窗口大小应改为按真实 shape 计算：

```text
trunk_window = B × 361 × 768 × sizeof(half)
inner_window = B × 361 × 384 × sizeof(half)
total_window = active_streams × (enabled windows per stream)
```

只需搜索：

- `off`；
- `trunk-only`；
- `trunk+inner`；
- 必要时 hitRatio `{0.5, 0.75, 1.0}`。

这是最容易自动泛化、搜索成本最低的一项。

### 6.5 FA4 与逐元素 kernel

在 B1–B13 内，FA4、RMSNorm、RoPE、affine-SiLU 和 SwiGLU 通常直接沿用。只有 profiler
显示 wave、occupancy 或 vector utilization 明显恶化时，才增加极小候选集：

- FA4：当前 N64 与一个邻近 attention tile；
- 逐元素 kernel：threads `{128,256}` 或当前 vector width 与一个邻近值。

不要为这些项目做全 tile 网格。

## 7. 推荐的 tactic 表

不要强求一套参数覆盖所有 batch。建议维护按 batch 和 GPU class 选择的静态 tactic 表：

```text
(gpu_class, batch, streams)
  -> FFN kernel
  -> QKV kernel
  -> linear2 kernel
  -> L2 window policy
```

其中：

- FA4、RMSNorm、RoPE、affine-SiLU 和通用 beta=1 residual 可作为共享默认项；
- FFN/QKV/linear2 使用少量 shape-specialized kernel；
- L2 policy 在模型构造时按 batch、stream 数和设备实际 grant 计算；
- kernel settle 后，再做一次 S1/S2 topology 扫描。

## 8. 建议的低成本搜索流程

每个新 batch 或新 SM120 GPU：

1. 运行官方/当前泛化路径的真实整图基线；
2. 只对 FFN、QKV、linear2 生成上述小候选集；
3. 用 S1 与 NCU 淘汰明确较慢、spill 或资源恶化版本；
4. 局部前二直接进入真实整图 S2 短 ABBA/BAAB；
5. 正收益候选运行长整图、8192-row full replay；
6. 每接受一项后重新跑全图 Nsys 与完整 ordinal NCU；
7. 最后扫描 S1/S2，而不是在 kernel 仍变化时提前确定 topology。

粗略候选数：

| family | 每个 batch/GPU 的建议候选数 |
|---|---:|
| FFN | 3–6 |
| QKV | 2–4 |
| linear2 | 3–5 |
| L2 policy | 3–6 |
| FA4/逐元素 | 默认不搜，必要时1–2 |

这使“每个 batch 找资源平衡点”的搜索保持在低计算量范围，同时保留整图 S2 作为最终裁决。

## 9. 最终分类

### 改 actual batch 就立即失去的专用收益

- 固定 B13 融合 FFN；
- FFN A-fragment reuse；
- 固定 B13 宽 QKV；
- 固定 B13 linear2 AOT。

### 改 max batch 才额外失去的收益

- persisting-L2 trunk；
- persisting-L2 inner。

### 5090D → 5080 不会立即失效，但应重选 tactic

- 融合 FFN；
- 宽 QKV；
- linear2 AOT；
- L2 policy；
- 最终 S1/S2 topology。

### 可稳定继承

- FA4/both16 的主要机制；
- GEMM beta=1 residual 融合；
- fused RoPE 与 half2 I/O；
- RMSNorm vec8；
- affine-SiLU half2；
- SwiGLU half8；
- A-fragment reuse 的指令与 register 减少机制；
- persisting-L2 的生命周期管理机制。

稳定继承的是“机制”，不是某次测得的百分比。任何新的 `(GPU, batch, streams)` 组合仍应由
真实整图 Nsys/NCU、长 S2 A/B 和完整精度回归共同裁决。
