# Hypothesis LT1: FP16 MatMulLayer GEMM 切换到 cuBLASLt

创建时间：2026-08-05（UTC），先于实现。

## 证据

4090（SM89）B13/S2 官方 CUDA 路径的 Nsys 中，FP16 GEMM 家族合计约占
GPU kernel 时间的 47%，其中
`ampere_h1688gemm_128x128_ldg8_stages_32x1_nn` 占 41.1%。

NCU 对 dominant GEMM 的实测：

| 指标 | 值 |
|---|---:|
| grid | 108 blocks（128 threads/block） |
| #SM | 128 |
| waves per SM | 0.42 |
| achieved occupancy | 8.31% |
| active warps / scheduler | 1.02 |
| no-eligible scheduler cycles | 85.79% |
| CPI stall（fixed-latency dependency） | 54.7% |
| compute (SM) throughput | 41.55% |
| L2 hit rate | 94.05% |

## 机制

legacy `cublasHgemm` 在 4090 上选择的大 tile 128x128 内核
每个 SM 只驻留约 1 个 CTA（受寄存器/shared memory 限制），且 grid 只有
108 个 block，无法铺满 128 个 SM。这不是 DRAM 或 L2 瓶颈，而是
launch geometry + occupancy 瓶颈。

cuBLASLt 的 per-shape heuristic 可以选择更小 tile / 更高 occupancy /
更完整 wave 的 FP16 tensor-core kernel，从而缩短 GEMM 关键路径。

## 预计变化

1. dominant GEMM 的 grid 覆盖所有 SM，achieved occupancy 上升。
2. no-eligible cycles / fixed-latency stall 下降。
3. 整网 B13/S2 吞吐上升；GEMM 子图绝对时间下降。

## 风险

- cuBLASLt heuristic 可能仍选中相同或更差的 kernel：以实测为准，无收益则否决。
- FP32 compute/FP16 I/O 与 `cublasHgemm` 的舍入不一定逐位一致；用完整 8192
  行 FP32 reference 门槛验收，不做 bit-exact 假设。
- 每个 shape 首次调用 heuristic 有一次性开销；已在 `CublasLtPlanCache` 中
  按 (m,n,k,ld) 缓存。

## 验证

1. 编译 + smoke：`cudaUseMatmulLt=true` 下 benchmarknn 正常。
2. Nsys：确认 dominant GEMM kernel 名称/网格变化。
3. NCU：重复同一指标，occupancy 上升、no-eligible 下降。
4. B13/S2 同制度 A/B（正反序，3x300 或长跑），记录绝对 nnEval/s。
5. 完整 8192 行 all-head + p0loss 回归。

## 重新开启条件

若 cuBLASLt 未选到更好 kernel 或吞吐无改善，否决；后续改用 CUTLASS/CuTe
固定 AOT GEMM 或宽 QKV/融合 epilogue。
