# Stage 11 hypothesis: fixed-B13 nested preConv 1x1 GEMM

Date: 2026-08-05 UTC

## Evidence and mechanism

- Frozen target: RTX 4090 SM89, exact 19x19, B13, FP16, S2.
- `Sm89Conv` already lowers NHWC FP16 1x1 convolutions to cuBLAS. Each of the
  11 nested blocks has one non-accumulating preConv with exact row-major
  boundary `M=4693,N=384,K=768`, `alpha=1,beta=0`.
- In the accepted stage9 Nsys trace these are the first `grid(3,37,1)` GEMMs
  in each seven-call nested-block pattern and take roughly 28-32us each.
- cuBLAS again launches only 111 M128xN128 CTAs on 128 SMs. A fixed CUTLASS
  tile may improve S2 scheduling, and the beta-zero epilogue can omit the
  source read entirely.

## Falsifiable test

1. Sweep exact beta-zero CUTLASS tiles/stages and compare every output element
   against cuBLAS.
2. Select by 10,000-iteration dual-stream ABBA. Do not extrapolate from the
   beta-one linear2/out-projection results.
3. Integrate only behind B13/S361/768-to-384 and non-accumulate guards if the
   isolated S2 boundary improves by at least 3%.
4. Require positive forward/reverse whole-network ABBA before Nsys and full
   FP32 accuracy acceptance.

## Outcome

- Accepted tile: threadblock `128x128x32`, warp `64x64x32`, five stages,
  identity swizzle, and a beta-zero epilogue with no source read.
- Exact random-input comparison was bit-exact. In 10,000-iteration S2 ABBA,
  the pair median improved from `25.622170us` to `21.397908us` (`+19.741%`
  throughput ratio). Stage 4 and stage 6 were both slower.
- Three independent, thermally primed, 2400MHz-locked 500-iteration
  forward/reverse ABBA rounds produced 12 samples per path. The combined
  process median was `2890.965423 -> 2929.610878 nnEval/s` (`+1.336766%`);
  forward and reverse groups were separately positive at `+1.822181%` and
  `+1.315403%`. Actual-wall throughput improved `+1.289133%`.
- Complete Nsys capture removed exactly `1166 = 106 * 11` cuBLAS calls and
  added exactly 1166 fixed CUTLASS calls. The last 30 forward-sized groups on
  both timed streams reduced kernel-time union from `253.156184ms` to
  `251.528396ms` (`0.642998%` reduction).
- Exact NCU measured `17.82us` for cuBLAS and `15.68us` for CUTLASS. The
  candidate uses 162 registers/thread and 80KiB dynamic shared memory, reaches
  8.30% achieved occupancy, and has zero local/shared spill.
- The 8192-row fixed-19 replay is byte-identical to accepted stage 9, preserving
  the full FP32 error envelope. Decision: accept and enable in the fixed B13
  4090 configuration.
