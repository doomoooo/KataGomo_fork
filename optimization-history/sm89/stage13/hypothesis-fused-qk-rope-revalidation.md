# Stage 13 hypothesis: revalidate fused Q/K learnable RoPE on final B13 stack

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- The accepted stage-11 complete Nsys capture contains 6996 learnable-RoPE
  kernels, exactly `106 forwards * 33 attention blocks * 2 Q/K launches`.
  They average `6.134us` and sum to `42.911ms`.
- Q and K have identical B13 MHA shapes and frequency tables. The existing
  `sm89ApplyRoPEQKHalfKernel` loads each frequency and computes `sincos` once,
  then rotates both buffers, replacing two launches with one.
- Stage 5 paused this path because unlocked forward/reverse results conflicted.
  FlashAttention, dual GEMM, fixed linear2, and fixed preConv have since changed
  the S2 schedule, so the old whole-network result is not transferable.

## Falsifiable test

1. Compare the existing disabled and fused paths on the current accepted stack
   with thermally primed, 2400MHz-locked, 500-iteration forward/reverse ABBA.
2. Require all three independent rounds and aggregate forward/reverse groups to
   be positive before profiling.
3. If accepted by throughput, require Nsys to remove exactly one of two RoPE
   launches per attention block, and run the full 8192-row FP32 comparison.

## Result

- Accepted and enabled in the exact RTX 4090 B13/S2 configuration.
- Three independent 2400MHz-locked, thermally primed rounds produced positive
  per-round changes of `+1.9814%`, `+3.4693%`, and `+0.7251%`. Across all
  12+12 samples, median throughput improved from `2904.690480` to
  `2964.289965` eval/s (`+2.051836%`). Forward and reverse aggregate groups
  were both positive (`+2.059989%` and `+3.314240%`); median actual wall time
  improved `+2.069298%`.
- Complete Nsys capture replaced exactly `6996 = 106 * 33 * 2` baseline RoPE
  launches with `3498 = 106 * 33` fused launches. Baseline launches averaged
  `6.636us`; fused launches averaged `8.767us`. In the last 30 complete
  forward groups across the two timed streams, kernel union time fell from
  `272.152509ms` to `266.441370ms` (`-2.098507%`).
- The 8192-row replay is byte-identical to the accepted stage-11 replay, SHA256
  `7dde3f6b36e240eb4e92ffc632ecc578d059052fb2c13816b043d0a7093ba484`.
  FP32 comparison remains policy top-1 `0.996948242188` and weighted p0loss
  `1.591545462608` versus FP32 `1.591527938843`.

## Stability audit

One initial third-round candidate process printed a complete JSON result and
then exited with SIGSEGV under `timeout`; an earlier attempt also stalled during
startup. The fused kernel index bounds are exact for B13/S361/H12/D32. CUDA
memcheck reported no invalid device access, although its run was polluted by 39
cuBLAS/cuDNN fatbinary/ISA-120 API instrumentation errors. A subsequent
alternating process stress test completed 20/20 candidate and 20/20 control
processes, and four additional 500-iteration candidate processes under GDB all
exited normally with no signal. The replacement third ABBA round also completed
normally. The anomaly is retained in the evidence rather than attributed to a
toolkit version or silently discarded.
