# H32: M128N96 linear2-residual coexistence tile

## Scope

- GPU: RTX 5090 D (locked logical device 2)
- Fixed shape only: `M=13*19*19=4693`, `N=384`, `K=1152`, FP16
- Operation: `output = input @ weights + residual`
- Target: two concurrent inference streams (S2)
- All S2 measurements use distinct weight allocations for the two streams.

## Evidence

The integrated current kernel is `M128N128K32/S4/T128`: 111 CTAs, 162
registers/thread, 65.5 KiB dynamic shared memory, 0.65 waves/SM, 8.3%
achieved occupancy, and 9.2% eligible warps.  The latest whole-network S2
measurement attributes 1077 us/stream-forward and 395 us excess to this
kernel family.  The earlier `M128N64` candidate increased the grid to 222 CTAs
and won an isolated dual-stream microbenchmark, but regressed the integrated
schedule.  Stage26's `M256N64` alternative was also rejected and is outside
this search.

## Falsifiable mechanism

`M128N96` produces `ceil(4693/128) * ceil(384/96) = 37 * 4 = 148` CTAs.  This
is between the current under-filled 111-CTA grid and the rejected 222-CTA grid.
It may improve SM coverage and reduce the N-fragment register footprint while
avoiding the extra tail wave of `N64`.  `K32` stage counts 2, 3, and 4 test the
latency-hiding versus shared-memory/co-residency tradeoff without changing the
arithmetic decomposition.  The primary screen uses T128; an adjacent
`M96N96` or T256 variant is allowed only if M128N96 exposes a clear resource or
tail limitation.

Expected observations if the mechanism is true:

1. M128N96 is no slower than current AOT in S1.
2. Homogeneous, phase-aligned S2 pair makespan improves by more than 0.5% with
   private weights per stream.
3. When paired separately with the accepted fused-FFN and wide-QKV kernels,
   pair makespan improves by more than 0.5% without increasing peer duration
   or reducing overlap materially.
4. Resource inspection shows the intended 148-CTA grid and a plausible
   occupancy/co-residency improvement rather than a timing-only anomaly.

## Controls and metrics

- Exact library control: C++ `cublasHgemm` with FP16 alpha=1 and beta=1,
  configured for the same row-major logical operation via the standard
  transposed column-major mapping.
- Exact incumbent control: TileLang `M128N128K32/S4/T128/min_blocks=3`, the
  same schedule that generated the integrated AOT source.
- Candidate correctness is checked against the cuBLAS result, and incumbent
  and candidate are benchmarked in the same process with identical tensor
  shapes and independently allocated stream weights.
- Report S1 kernel time, homogeneous S2 per-operation pair makespan and each
  stream duration, mixed-pair makespan/stream durations, overlap, grid,
  registers/thread, shared memory, and occupancy evidence.

## Decision rule

Reject the family unless a candidate is S1-neutral (within measurement noise)
and produces a repeatable greater-than-0.5% local improvement over the exact
incumbent in homogeneous S2 and both mixed peers.  Any material peer slowdown,
loss of overlap, correctness failure, or dependence on shared weight pointers
is a rejection.  No KataGo integration is permitted from microbenchmark-only
evidence.
