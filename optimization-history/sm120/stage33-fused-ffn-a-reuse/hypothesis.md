# H33: reuse each fused-FFN A fragment across linear and gate MMA

## Frozen protocol

- GPU: RTX 5090 D, logical GPU 2 under `nvidia-env.sh` and `gpu-lock`.
- Target: fixed B13, 19x19, FP16/NHWC, S2. No mask variants, other batches,
  phase tuning, head work, or graph-wide bundle are in scope.
- Mainline remains the accepted Stage29 trunk+inner L2 build. This experiment
  initially changes only a standalone generated fused-FFN candidate; it must
  not edit shared KataGo source or the target configuration before the local
  gates pass.
- Current full-graph evidence is
  `stage31-current-graph-overview/current-fullgraph-ranking.md` and its
  matching 344-ordinal NCU/Nsys artifacts. Because H32 was rejected without a
  mainline change, those profiles remain the selection basis.

## Evidence and mechanism

The accepted fused FFN is the largest current kernel family by summed S2 work:
`1477.6 us/stream-forward` (25.90%), with `119.1 us` S2 excess and `1.086x`
S2/S1 slowdown. Full-graph NCU reports 146 registers/thread, 32.8 KiB dynamic
shared memory, 1.31 waves/SM, 20.3% achieved occupancy, 23.4% eligible cycles,
and 40.5% tensor utilization.

In `fused_ffn_b13.cu`, each K fragment of `input_shared` is loaded by
`ldmatrix` once for the linear MMA and again for the gate MMA. The two products
share A exactly. A bounded candidate will load each A fragment once and use it
for both accumulator families before advancing the K fragment. This removes
one set of shared-memory `ldmatrix` instructions, but moves gate MMA ahead of
the next linear-weight prefetch. The latter is the explicit failure risk.

## Falsifiable prediction

If repeated A-fragment loading contributes to the current fixed-latency and
eligible-warp limits, the candidate should:

1. preserve the exact 666-CTA grid, 128 threads, and 32 KiB shared footprint;
2. reduce dynamic shared-load instructions and standalone fused-FFN latency by
   more than 0.5%;
3. keep registers at or below 154/thread and not reduce achieved occupancy;
4. improve the homogeneous S2 pair by more than 0.5%; and
5. avoid more than 0.25% regression in pair makespan or peer duration with the
   current linear2 and wide-QKV kernels.

Output must be bit-identical to the current AOT kernel for a fixed deterministic
input. The candidate is rejected immediately if compilation spills local
memory, registers exceed 154, S1 fails to improve by 0.5%, homogeneous S2 fails
to improve by 0.5%, or either mixed peer regresses by more than 0.25% in an
interleaved measurement.

## Validation order

1. Build the one source-level candidate and compare deterministic output.
2. Measure interleaved S1 and homogeneous S2 against the exact current AOT.
3. If both gates pass, measure mixed pairs with current linear2 and wide QKV.
4. If all local gates pass, collect representative NCU and integrate behind an
   exact-shape default-false switch.
5. Only after integrated whole-network accuracy and long symmetric S2 ABBA pass
   may the candidate be accepted. An accepted change must immediately trigger
   a fresh full S2 Nsys and matching 344-ordinal S1 NCU rerank before any next
   optimization target is chosen.

This hypothesis does not authorize another tile/stage/thread sweep. A failure
closes A-fragment reuse unless a later profile exposes a different dependency
mechanism.
