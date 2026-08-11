# Stage 8 hypothesis: fixed-B13 FFN GEMM and SwiGLU

Date: 2026-08-05 UTC

## Frozen target and critical-path evidence

- RTX 4090 SM89, exact 19x19, B13, FP16, two independent server streams.
- Accepted stage-7 FlashAttention path remains enabled.
- Each of 33 FFN blocks calls `cublasHgemmStridedBatched` with column-major
  `M=1152, N=4693, K=384, batchCount=2`, reusing the same input matrix and
  writing the linear and gate matrices separately.
- In the last 30 complete forwards per stream, this grid `(9,37,2)` accounts
  for `91.987 ms` summed duration and `52.014 ms` exclusive critical-path time.
  It is 69% of the exclusive FP16 GEMM critical path and the largest remaining
  fixed-shape target.

The isolated cuBLAS boundary runs in `43.317 us` (`191.7 TFLOP/s`). NCU on the
same call reports `44.16 us`, 186 registers/thread, 15.37% achieved occupancy,
84.14% no-eligible cycles, 57.27% tensor-pipe throughput, 67.96% L2 throughput,
12.43% DRAM throughput, 97.45% L2 hit rate and no spilling. The dominant warp
stalls are fixed-latency wait (3.96 cycles/issue), long scoreboard (1.88), math
pipe (1.82) and MIO throttle (1.20).

## Hypotheses

1. A fixed-shape SM89 CUTLASS AOT kernel with a smaller accumulator tile can
   trade some per-CTA reuse for more active warps and reduce isolated latency.
   The falsifiable threshold is at least 3% below cuBLAS over long alternating
   runs. Variants that merely improve occupancy but not latency are rejected.
2. The two GEMMs use the same B/input matrix. A dual-accumulator mainloop that
   loads B once, computes both weight matrices and emits `SiLU(linear) * gate`
   directly can remove the standalone SwiGLU launch and roughly 43 MB of
   intermediate output traffic per call (two 21.6 MB read/write passes), while
   preserving only the 10.8 MB activated output. This is the higher-upside
   candidate if tile-only tuning cannot exceed cuBLAS.

The fused route risks doubled accumulator pressure, reduced occupancy and a
different FP16 rounding point. It must validate every output against an
unfused cuBLAS + SwiGLU reference in the microbenchmark, then pass the full
8192-row all-head FP32 regression after integration.

## Validation and decision

- Sweep CUTLASS kernels compiled for SM89 at exactly M/N/K, including launch
  geometry, stages and warp tile as independent variables where practical.
- Measure complete call boundaries on the caller stream, not kernel body only.
- Retain the cuBLAS path behind a runtime switch and as fallback for every
  non-B13/non-C384/non-C1152 shape.
- Require at least 1% whole-network B13/S2 gain in forward and reverse ABBA,
  plus Nsys union/exclusive evidence matching the eliminated traffic/kernel.
- Reject accuracy drift beyond the established FP16 envelope against the fixed
  8192-row FP32 reference.

## Result

- Accepted tile: threadblock M128xN64xK32, warp M64xN32xK32, three stages,
  swizzle 2. The isolated complete cuBLAS+SwiGLU boundary fell from about
  53.35us to 37.51us.
- The last 30 complete B13 forwards per timed stream reduced dual-stream union
  busy from 283.620ms to 264.711ms. FFN summed duration fell from 127.643ms to
  82.599ms even though competing 128x128 GEMMs increased by 27.427ms.
- 300-forward ABBA plus reverse order improved the combined median from
  2647.477 to 2951.138 nnEval/s (+11.470%).
- The 8192-row all-head regression passed directly against FP32. The fused
  path is not bit-exact to current cuBLAS: policy top-1 agreement is 99.7437%
  between the FP16 paths. Its p0loss (1.591545) is closer to FP32 (1.591528)
  than the cuBLAS control (1.591574).
- Swizzle 4 and one-time dynamic-shared-memory attribute setup were rejected
  by whole-network B13/S2 ablations despite local reasons to expect a gain.
