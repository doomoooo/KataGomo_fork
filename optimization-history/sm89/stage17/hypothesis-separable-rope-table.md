# Stage 17 hypothesis: separable 19x19 RoPE table in the QKV epilogue

Date: 2026-08-05 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Stage 16 QKV+RoPE epilogue is the control; model, corpus, clocks, warmup,
  precision gates, and complete-forward measurement remain frozen.

## Evidence and mechanism

- Final Stage 16 Nsys attributes `58.995ms` of summed kernel time to QKV+RoPE
  in the last 30 complete forwards on both streams.
- Exact NCU reports only 16.08% DRAM throughput, 96.45% L2 hit rate, zero spill,
  and 80.67% scheduler cycles with no eligible warp. The math-pipe stall is
  consistent with four `__sincosf` evaluations per 128-bit output access.
- Full `(xy, head-pair)` float2 tables were already rejected in Stage 14. They
  occupy 554KiB per attention block and perturb the whole-network L2 working
  set.
- For exact 19x19, the angle is separable:
  `angle(x,y,hp) = x*freqX(hp) + y*freqY(hp)`. Two tables containing
  `sincos(x*freqX)` and `sincos(y*freqY)` require only
  `2*19*192*sizeof(float2) = 29184 bytes` per block. The hot table for the
  current block fits in L1, and angle addition becomes four FP32 operations via
  the sine/cosine addition identities.

## Falsifiable test

1. Extend only the Stage 16 microbenchmark. Compare the fused epilogue
   byte-for-byte with the same CUTLASS GEMM followed by a standalone kernel
   using the identical separable-table arithmetic.
2. At locked 2400MHz, compare S1 and S2 against the accepted direct-recompute
   stage-3 tile. Reject before integration unless S2 improves clearly.
3. If it passes micro, integrate behind a separate runtime switch and require
   full replay against FP32, Nsys complete-forward union reduction, and three
   locked forward/reverse ABBA rounds. Because the trig identity changes
   rounding, byte identity with Stage 16 is not assumed.

## Risks

- Two float2 loads per pair double the frequency-side bytes even though the
  table is small; L1 conflicts or dual-stream cache interference may dominate.
- The addition identity changes floating-point rounding and must pass absolute
  FP32 gates.
- Extra epilogue temporaries may increase register pressure above the already
  high 240 registers/thread.

## Result

- Rejected at the isolated boundary; no production integration was made.
- The first generic three-source iterator invalidated its own control by
  increasing direct-recompute from about 30us to 58us. That result is retained
  as an implementation failure and was not used for the decision.
- Compile-time single-source specializations restored the Stage 16 control.
  Two locked ABBA observations gave direct-recompute S2 `32.420us/32.494us`
  and separable-table S2 `49.502us/49.507us`; the candidate was about 52.5%
  slower. S1 likewise changed from `29.242us/29.196us` to
  `34.948us/34.944us`.
- Both specializations had zero byte mismatches against their corresponding
  standalone arithmetic. Correctness did not rescue the performance failure.
- The 29KiB working set is small enough for cache capacity, but two float2
  loads per pair and the addition-identity temporaries cost much more than
  direct frequency loads plus `__sincosf`. Reopen only if a warp-shared table
  load or a substantially lower-register epilogue changes that byte/temporary
  balance.
