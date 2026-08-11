# Stage 16 hypothesis: rotate Q/K in the QKV GEMM epilogue

Date: 2026-08-05 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Model, corpus, config, clocks, warmup, and acceptance gates remain those in
  `/workspace/SKILL.md` and `/workspace/results/4090/HISTORY.md`.

## Evidence and mechanism

- The accepted stage-13 kernel rotates Q and K together, but still launches 33
  kernels per forward and reads then rewrites every Q/K element after the QKV
  projection.
- The QKV projection is one fixed strided-batched GEMM with shape
  `M=4693, N=384, K=384, batch=3`; batch indices 0/1/2 are Q/K/V.
- CUTLASS's row-major tensor-op epilogue writes eight adjacent FP16 channels per
  128-bit access. Head dimension 32 and interleaved RoPE pairs mean every access
  contains exactly four complete pairs and never crosses a head boundary.
- A specialized output iterator can round accumulators to FP16, rotate Q/K from
  the model-lifetime float2 table, and write the final values directly. V remains
  unchanged. Every Q/K element is rotated once and the standalone kernel and its
  Q/K read-write pass disappear.
- Fusing into FlashAttention's load path was inspected and is not the chosen
  mechanism: fixed `M64/N96` creates six Q tiles, so each K row would be rotated
  six times. This is a fixed-shape reuse issue, not a software-version bypass.

## Falsifiable test

1. Build an isolated fixed-shape microbenchmark. Compare its fused epilogue
   byte-for-byte against the same CUTLASS QKV kernel followed by the accepted
   float2-table RoPE arithmetic.
2. Measure S1 and S2 against the current cuBLAS strided-batched QKV plus accepted
   recompute RoPE kernel at locked 2400MHz after thermal priming. Sweep only
   compile-time tile/stage choices with a plausible structural mechanism.
3. Proceed to integration only if S2 has clear headroom beyond run drift and the
   fused result passes the full 8192-row replay gates.
4. Final acceptance still requires three independent locked forward/reverse ABBA
   rounds and Nsys complete-forward critical-path reduction. Kernel-only speed is
   insufficient.

## Risks

- The float2 table adds epilogue loads and stage 14 showed that table residency
  can perturb the whole network. Direct-write fusion must recover more traffic
  and launch cost than those loads add.
- CUTLASS may use a different GEMM reduction order than cuBLAS. Byte identity is
  required only against the unfused form of the same CUTLASS GEMM; production
  correctness is decided by the frozen replay thresholds.
- Extra epilogue registers can reduce occupancy or elongate the QKV kernel enough
  to erase the standalone-kernel saving under S2 contention.

## Result

- Accepted for RTX 4090 exact 19x19/B13/FP16/S2 and enabled in
  `/workspace/bench-cuda-gpu0-4090-s2.cfg`.
- The first float2-table integration was rejected despite a faster isolated
  kernel because its model-lifetime tables perturbed L2. Direct frequency
  recomputation retained only 1.5KiB per attention block and was byte-identical
  to the accepted stage-13 replay.
- The isolated direct-recompute winner is `M128xN128xK32`, warp
  `M64xN64xK32`, stage 3. At 2400MHz its S2 pair changed from `49.307us` for
  cuBLAS QKV plus recompute RoPE to `33.244us` fused (`-32.58%`).
- Three locked 2400MHz forward/reverse ABBA rounds improved by `+2.58%`,
  `+4.45%`, and `+3.47%`. The pooled median changed from `3003.995` to
  `3106.163 nnEval/s` (`+3.40%`); all 12 adjacent pairs were positive and their
  median improvement was `+3.71%`.
- Final Nsys removed 3498 batched-QKV launches and 3498 standalone RoPE
  launches, replacing them with 3498 custom GEMMs. Per-stream kernel count fell
  from 344 to 311 per forward, and the last-30-forward dual-stream union fell
  from `266.861ms` to `255.423ms` (`-4.29%`).
- Exact NCU measured `28.67us`, 240 registers/thread, 49.15KiB dynamic shared
  memory, 15.14% achieved occupancy, and zero local/shared spill.
- Lower-stage and asymmetric-warp variants were tested rather than excluded by
  toolkit version. Stage 2 was `41.74us` under S2; capping it to 168/160
  registers degraded to `110.56us/124.18us`, so stage 3 remains the winner.
