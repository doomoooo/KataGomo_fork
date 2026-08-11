# Stage 30 hypothesis: precomputed learnable RoPE in the fused QKV epilogue

Date: 2026-08-06 UTC

## Frozen target and baseline

- GPU: RTX 4090, SM89, GPU 0, locked 2400 MHz for decisions.
- Shape: exact 19x19, B13, FP16, one NN server / one CUDA stream.
- Model: `b11c768h12nbt3tflrs-fson-silu`.
- Control binary SHA256:
  `2911970dc5def0257c3d7962e88197a7f110db987ec7005720b0ce8f29b84889`.
- Control config: `/workspace/bench-cuda-gpu0-4090-s1.cfg`; the existing
  `cudaUsePrecomputedQKRoPESm89` option is absent and therefore false.

Current full-graph Nsys measures 5.3115 ms per forward. Summed kernels are
about 5.1040 ms, leaving about 0.2075 ms (3.91%) outside kernels. The top three
kernel families account for 61.94% of summed kernel time:

| Family | us / forward | Kernel share |
|---|---:|---:|
| dual GEMM + SwiGLU | 1324.145 | 25.94% |
| QKV GEMM + RoPE | 953.082 | 18.67% |
| FlashAttention | 883.951 | 17.32% |

Artifact: `stage29/nsys-paired-forward/candidate.sqlite`.

## Pre-implementation NCU and SASS evidence

The short NCU capture profiles three current launches from each hotspot. The
QKV + RoPE kernel averages:

- 29.73 us, 240 registers/thread and 49.15 KiB dynamic shared memory;
- 16.67% theoretical / 14.95% achieved occupancy, 1.30 waves/SM;
- 47.40% compute, 55.47% L2 throughput, only 1.78% DRAM throughput;
- 99.45% L2 hit, zero local/shared spill;
- only 0.25 eligible warps per scheduler-cycle and about 33.3% of the issue
  interval attributed to math-pipe throttle.

SASS from the same report contains repeated `MUFU.SIN` and `MUFU.COS`
instructions in the QKV epilogue. This matches the source, which recomputes
learnable `sincos` for every Q/K output pair even though coordinates and
frequencies are invariant for the model lifetime.

For context, the dual-GEMM kernel is already at 67.96% compute / 73.66% L2
throughput, while FlashAttention's dominant math-pipe stalls come from the
attention softmax and are less isolated. QKV is therefore the first candidate.

Artifact: `stage30/ncu-hotspots/current-s1-top3.ncu-rep`, SHA256
`32d37449bb035e93b83953e072b3414aec00d0f7c51fbb7672e5c79823eb6cb3`.

## Why the rejected stage-14 route is conditionally reopened

Stage 14 put the table in a standalone, memory-bound RoPE kernel. It only
improved that kernel from 8.773 to 8.459 us and hurt S2 Nsys union, where two
server models added about 36.6 MiB of tables. Stage 16 later moved RoPE into
the 29.73 us QKV kernel. The current candidate removes special-function work
and coordinate arithmetic from a low-occupancy hot epilogue; it does not add a
standalone kernel or Q/K intermediate traffic.

The direct epilogue already loads two FP32 frequencies per output pair. The
candidate instead loads one FP32 `float2` from L2, so bytes per pair do not
increase. For S1, the 33 per-layer tables total about 18.3 MiB.

## Single-variable implementation

Reuse the existing model-lifetime `ropeCosSinTable`, generated with the same
`__sincosf` expression. Pass it to `Sm89QKVRoPEGemmB13` only when
`cudaUsePrecomputedQKRoPESm89=true`. In the QKV output iterator:

- Q/K load `(cos,sin)` from the table;
- V remains unchanged;
- the false/null path retains the current direct `__sincosf` implementation;
- GEMM tile, stage, launch geometry, FP16 rounding and output layout remain
  unchanged.

## Falsifiable predictions and gates

1. Candidate replay must be byte-identical to the Stage 29 accepted output.
2. Candidate execution must reduce math-pipe stalls. Because the same binary
   keeps a runtime null-table fallback, static SASS will retain the direct
   `MUFU.SIN/COS` branch even when the candidate does not execute it.
3. Three-launch NCU should reduce QKV duration by at least 5%, without spill;
   L2 hit should remain above 95% and DRAM throughput must not become dominant.
4. Forward and reverse 20-iteration Nsys must both reduce the summed QKV
   boundary and must not add a steady-state kernel launch.
5. Only if the local boundary and both full-graph directions improve, run one
   locked 100-iteration forward/reverse ABBA. Require both order directions
   and at least 3/4 adjacent pairs positive.
6. Run the full 8192-row FP32 comparison only after the performance gate.

This stage decides S1 first. If S1 passes, S2 is measured separately; any
cross-stream table sharing is a later, separately attributed experiment.

## Result

Rejected and reverted before ABBA or full accuracy.

- A 26-row smoke replay was byte-identical between control and candidate:
  both KRNN files have SHA256
  `0ee632cfec280a4da9b7b224e779509d635a02ced97edcf6e0d054d4f3f0b95d`.
- Three-launch NCU improved QKV only `29.728 -> 29.515 us` (`-0.72%`),
  below the 5% gate. Registers increased `240 -> 255`; L2 throughput rose
  `55.47% -> 67.32%`; L2 hit remained `99.45% -> 99.43%`; DRAM remained low
  at `1.56%`; no spill occurred. Math-pipe stall fell about 10%, but long
  scoreboard stall rose about 51%, so table loads replaced rather than removed
  the dominant latency.
- Forward Nsys reduced the QKV family `968.397 -> 957.269 us/forward`, but
  full throughput fell `2439.061 -> 2430.633 nnEval/s` (`-0.346%`).
- Reverse Nsys reduced QKV `969.747 -> 955.235 us/forward`, while full
  throughput rose `2429.362 -> 2442.265 nnEval/s` (`+0.531%`).
- The contradictory whole-graph directions and failed local gate make the
  result no-signal/negative. The 18.3 MiB S1 table working set is the likely
  source of displaced L2 work elsewhere in the graph.
