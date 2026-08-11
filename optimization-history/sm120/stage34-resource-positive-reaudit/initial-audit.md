# Resource-positive candidate re-audit after the Stage33 gate correction

## Scope

This audit applies the revised retention policy to fixed RTX 5090 D, B13,
19x19, FP16/NHWC, S2 only. It cross-checks `results/rebuild/HISTORY.md`, the
Stage 1--33 reports, `cuda-optimization-history.md`, and the current Stage31
whole-graph profile. Historical measurements and original decisions are not
rewritten.

The audit asks a narrow question: was a correct candidate with a clear S1 win
and improved resource signature denied a current-mainline whole-graph S2 test
only because a homogeneous/micro or minimum-gain proxy gate failed?

## Reopen queue

| priority | candidate | evidence under the new policy | action |
|---:|---|---|---|
| 1 | Stage33 fused-FFN A-fragment reuse | bit-exact; S1 `-7.321%`; registers `146 -> 136`; no spills; same grid/threads/smem; old proxy data is non-decisional | reopened as H33b; current-mainline whole graph decides |
| 2 | Stage32 linear2 M128N96/S4 | bit-exact; S1 about `-21.4%`; smem `65.54 -> 57.34 KiB`; old proxy data disagreed and is now non-decisional | retain as a resource-tradeoff candidate; run one current-mainline short whole-graph screen after H33b is decided |
| 3 | Stage15 fused RoPE half2 I/O | direct RoPE `-7.60%`; short S1 `+0.301%`; short S2 `+0.443%`; rejected only for missing the old `0.5%` promotion threshold | re-enable on the then-current mainline and run symmetric long S2; collect NCU if the signal persists |

Stage32 is not strict resource dominance because its grid expands `111 -> 148`
CTAs and the lower shared-memory allocation does not change its one-CTA/SM
residency or permit co-residency with wide QKV. It is nevertheless retained
because the S1 margin is large. The real graph must adjudicate directly.

## Already queued, not an old proxy-gate rejection

| candidate | evidence | status |
|---|---|---|
| Stage30 initial convolution engine47 | event boundary `29.150 -> 16.172 us`; Nsys kernel boundary about `-27.7%`; smem `81.92 -> 4.10 KiB`, occupancy `8.32 -> 26.07%`, but registers `94 -> 128` | nominated for narrow S2 integration; not previously rejected |
| initial-global matmul + broadcast-add | audited boundary and 5080 precedent, no 5090 D implementation result | unexplored, not missed by a gate |
| wide head / fused P1 bundle | operator audit only; P1 is a structural prerequisite, not a current full-graph priority | unexplored/paused, not missed by a gate |

## Do not reopen unchanged

| route | reason under the revised policy |
|---|---|
| Stage6 single-wide FFN and Stage7 strided QKV | their S1 paths were valid, but they are superseded in S2 by the accepted fused-FFN and planar wide-QKV AOT kernels; they are not alternatives to the current operator boundaries |
| Stage20 FP32-accumulation fused FFN | resource direction is worse (`168` registers) and current whole graph regressed `4.74%` |
| Stage29 flat vec8 C768 affine-SiLU | already received a current-mainline long whole-graph decision: `3800.148 -> 3793.430 nn/s`, only one of four adjacent pairs positive |
| Stage27 attention out-projection AOT | candidate is slower in S1 (`9.347` vs cuBLAS `8.137 us`) and much slower in homogeneous S2; lower static resources did not create overlap |
| Stage28 outer contract/expand CUTLASS2 AOT | contract is consistently slower in S1 and S2; expand is noise-level; NCU shows lower resources but inferior individual GEMM work |
| batch-shared RoPE / two-way RoPE | 5090 D B13 candidate made the direct kernel 36.3% slower and did not improve S2; this is not an S1-positive resource-dominant case |
| two-warp RMSNorm, QKV M64, FA4 register-Q/M128, FFN stage1/MB4 | the intended occupancy/resource mechanism was directly falsified by slower kernel work or lower tensor throughput |
| RMS fast-tree | performance is irrelevant after the full accuracy gate failed |

## 5080 history cross-check

The 5080 rejected routes do not add another strict match to the reopen rule:

- FA4 N96 had a slightly shorter isolated NCU duration but regressed the whole
  graph `0.828%`; no resource-dominance evidence was recorded.
- FFN stage1/MB4 raised occupancy but reduced tensor SOL and regressed the
  whole graph `0.528%`.
- fused head pooling collapsed from `+0.211%` in one order to `+0.003%` in the
  reverse order; no NCU resource mechanism supports reopening it unchanged.
- 5080 accepted candidates that remain absent on 5090 D (initial convolution,
  initial-global fusion, head paths, and ordinary-weight sharing) are genuine
  unexplored work, not candidates rejected by the old homogeneous-S2 gate.

## Conclusion

The old policy did create false-negative risk, but the evidence does not support
claiming that every S2 rejection was wrong. Three 5090 D candidates warrant a
current-mainline recheck, led by H33b. Homogeneous and synthetic mixed S2
microbenchmarks are removed from all future work. The queue is deliberately short: after
each accepted candidate, fresh full-graph Nsys/NCU can reorder or invalidate the
remaining two.
