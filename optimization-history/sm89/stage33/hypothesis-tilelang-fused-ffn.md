# Stage 33 hypothesis: TileLang fused FFN for exact 19x19 B13 S2

## Frozen target

- GPU: RTX 4090 (SM89), GPU 0, graphics clock locked to 2400 MHz.
- Workload: exact 19x19, batch 13, two saturated evaluator streams (S2).
- Model: `b11c768h12nbt3tflrs-fson-silu.bin.gz`.
- S1 is out of scope. Every kernel and full-network decision in this stage uses S2.
- Candidate is default-off behind `cudaUseTileLangFusedFFNSm89` and retains the accepted CUTLASS fallback.

## Profile evidence before implementation

The accepted S2 path spends 33 calls per forward in the CUTLASS `DualGemm` fused
linear1/linearGate/SwiGLU kernel. A three-sample NCU capture is at
`ncu-cutlass-dual-ffn-s2-control/control.ncu-rep`.

| Metric | S2 control |
|---|---:|
| duration | 41.02, 41.47, 41.34 us (median 41.34 us) |
| registers/thread | 168 |
| dynamic shared memory/CTA | 49.152 KiB |
| resident CTA/SM | 2 |
| theoretical occupancy | 16.67% |
| achieved occupancy | 15.30-15.33% |
| no eligible warp | 78.23-78.51% |
| waves | 2.60 |
| local/shared spills | 0 |

The kernel is latency-hidden poorly: 49.152 KiB shared memory limits it to two CTAs
per SM even though the register file can admit three. The high no-eligible fraction is
consistent with insufficient independent warps, not a spill problem.

## Candidate and mechanism

Port the already-generated TileLang B13 schedule from the local SM120 worktree, while
compiling and measuring it for SM89. It preserves the fixed arithmetic shape
`M=13*361=4693, N=1152, K=384`, fuses the two FP16 tensor-core projections and SwiGLU,
and uses `M128-N64-K32-S2-T128`, `__launch_bounds__(128,3)`, and 32 KiB shared memory.
The expected mechanism is three resident CTAs/SM (25% theoretical occupancy), providing
more eligible warps while processing the same 666 output tiles.

The AOT source derives from local TileLang checkout commit
`8001cc4ccf6149382d2019654a19f59c1d4d0482` (MIT). Its prior SM120 measurement is only
design evidence; no SM120 performance result is used as an SM89 result.

## Falsifiable gates

1. Build and 26-row exact-B13 S2 smoke must complete with finite outputs.
2. Three-sample S2 NCU capture must show 32 KiB shared memory, three CTA/SM, 25%
   theoretical occupancy, no spills, and median kernel time at most 38.0 us (at least
   8% below the 41.34 us control). The no-eligible fraction should fall materially.
3. Short S2 Nsys forward and reverse order runs (20 measured iterations each) must both
   improve the fused-FFN aggregate and end-to-end throughput. An order-dependent or
   end-to-end regression rejects the candidate.
4. Only after those gates pass: locked-clock 100-iteration S2 ABBA, then the 8192-position
   FP32-reference accuracy suite.

Failure of the occupancy mechanism, the single-kernel threshold, or either short full-graph
order gate causes immediate revert. No long test is run for a failed candidate.

## Numerical risk

Both paths use FP16 MMA accumulation and FP16 output, but the TileLang activation expression
and reduction ordering differ from CUTLASS. Exact bit identity is not assumed. Smoke checks
catastrophic errors first; the full FP32-reference gate is mandatory only after performance
passes.
