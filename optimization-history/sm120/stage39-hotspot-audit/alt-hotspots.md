# Stage 39 alternate-hotspot audit: FA4 and wide QKV

- Audit time: 2026-08-06 11:18:39 UTC.
- Scope: RTX 5090 D, fixed B13, fixed 19x19/S361, FP16, accepted S2 graph
  after RoPE-half2. Other batches, masks, and topologies are out of scope.
- This is a read-only audit. It changed no KataGo source or configuration and
  ran no performance benchmark.
- Primary evidence:
  - Stage38 full-graph ranking JSON SHA256
    `fa17ec185cd07bd59764e41a3a7ddac1d066f786bf535bd5bc49ebbdbead8712`.
  - Stage38 344-ordinal NCU CSV SHA256
    `fcc3b9b0dcd9e1f2eb9d5ae62c4e97aaf9b7c307d8701404f90504c7b57883f3`.

## Result

Only one current-mainline recheck and one new implementation hypothesis clear
the evidence bar. Both are QKV changes. No FA4-only change is currently better
supported than another FFN iteration.

| priority | single variable | state | realistic whole-graph signal | queue relative to FFN |
|---:|---|---|---:|---|
| 1 | enable the existing QKV `M128N128K32/S3` schedule on the current accepted graph | implemented but never adjudicated by the current-mainline real S2 graph | about `+0.5%` to `+1.5%`; hard ceiling about `+3.1%` | **before** a new FFN implementation, because the candidate and S1 evidence already exist |
| 2 | keep the accepted QKV tile, but launch `clusterDim.x=3` and load each A tile once per cluster through DSM | not implemented | about `+0.5%` to `+1.5%` if L2/load stalls fall; hard ceiling roughly `+3%` | **after** the next evidence-backed FFN change; higher engineering and synchronization risk |

The percentage ranges are hypotheses, not measurements. The real S2 full graph
remains the acceptance gate.

## Current hotspot evidence

| group | S2 work / stream-forward | S2 excess | S2/S1 | registers | dynamic shared | waves/SM | eligible | tensor | wait/issue | long-scoreboard/issue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wide QKV | `785.1 us` | `149.6 us` | `1.235x` | 136 | 65.536 KiB | 1.96 | 6.88% | 42.66% | 4.89 | 2.09 |
| FA4 | `592.0 us` | `200.2 us` | `1.511x` | 247 | 24.576 KiB | 1.38 | 37.56% | 47.74% | 1.37 | 0.51 |

The important difference is that QKV has a directly visible load/issue
bottleneck: median L2-sector activity is 60.34% of peak, L2 hit rate is about
100%, LG-throttle is 3.21 instructions per issue, and only 6.88% of cycles
issue an eligible warp. FA4 has essentially no LG-throttle (`0.003`) and a much
healthier issue rate. Its S2 excess is primarily concurrent Tensor-pipe sharing,
not a simple memory or launch-resource defect.

## H39-Q1: reopen QKV K32/S3 on the real current graph

### Evidence

The accepted QKV tile is `M128N128K64/S2/T128`; the dormant alternative is
`M128N128K32/S3/T128`. This is a schedule-only change. Grid, output layout,
FP16 MMA accumulation, packed weights, and arithmetic result remain unchanged.

Historical fixed-B13 evidence is unusually strong:

- Direct S1 kernel: `18.8397 -> 17.1283 us`, `-9.084%`.
- Real S1 whole graph, symmetric A-B-B-A:
  - control mean: `(3116.704766 + 3113.372812) / 2 = 3115.038789 nn/s`;
  - candidate mean: `(3155.143799 + 3156.786344) / 2 = 3155.965072 nn/s`;
  - delta: `+1.314%`.
- Source and launch boundary reduce dynamic shared memory from 65.536 to
  49.152 KiB. Including the 1 KiB static allocation, the shared-memory resident
  limit changes from one to two CTAs per SM on the 102.4 KiB SM.
- The checked binary's SM120 resource records report 136 registers/thread for
  the incumbent and 142 for K32/S3, with no local memory or stack allocation.
  Therefore this is not register dominance; it is a small register increase in
  exchange for crossing the shared-memory residency threshold.

The old isolated dual-stream result was negative (`30.12 -> 31.74 us` per
stream). Under the corrected workflow this is risk evidence, not a rejection
gate: it is a homogeneous local proxy and was never retested on the accepted
current real graph.

### Falsifiable mechanism and expected NCU change

K32/S3 gives the copy pipeline one additional stage and permits two resident
CTAs instead of one. It should hide the current long-scoreboard/LG-throttle
gaps. A successful current result should show:

- dynamic shared `65.536 -> 49.152 KiB` and no spills;
- achieved occupancy `8.31% -> >= 12%`;
- eligible cycles `6.88% -> >= 8%`;
- wait/issue `4.89 -> <= 4.3` and long-scoreboard below 2.0;
- isolated NCU duration at or below about `17.8 us` on the current binary.

Failure is equally clear: if the doubled K-loop/barrier count leaves duration
at or above the incumbent, or the real S2 full graph is negative outside run
dispersion, the extra residency did not pay for its synchronization cost.

### Whole-graph bound and decision

At the current profile, QKV's isolated reference is about 635.5 us per
stream-forward. Applying the historical 9.084% serial saving gives about
57.7 us. Adding the entire current 149.6 us interference excess produces an
unreachable best-case saving of about 207 us. Against the current approximately
6.743 ms S2 pair time, that is a hard ceiling near `3.1%`. A credible expected
result is `0.5-1.5%`.

This recheck should precede a new FFN implementation because it requires only
one existing configuration switch and already has a real S1 whole-graph win.
If accepted, it must immediately be followed by the normal full S1/S2 Nsys and
all-ordinal S1 NCU profile; it should then be committed as one optimization.

## H39-Q2: three-CTA QKV cluster sharing of the A tile

### Evidence

The accepted grid is `(9,37,1)`. For each M tile, its normalized input A tile
is loaded independently by all nine N tiles. Grouping three adjacent N CTAs
creates three clusters per M tile, and all three CTAs in each cluster consume
the identical A tile while retaining distinct weight tiles.

For one fixed QKV invocation, the current useful half-precision traffic is
approximately:

- repeated A loads: 32,438,016 bytes;
- weight loads: 32,735,232 bytes;
- output stores: 10,812,672 bytes.

One A loader per three-CTA cluster reduces A traffic by 21,625,344 bytes and
reduces A+B load traffic by 33.18%. This attacks the measured L2-sector,
LG-throttle, and long-scoreboard bottleneck directly. The nearly 100% L2 hit
rate means the expected gain is L2/interconnect relief, not DRAM relief.

The local TileLang headers already provide
`rasterization2DRowWithCluster<panel_width,cluster_dim_x>`, cluster barriers,
and remote-shared address mapping. A cluster-aware swizzle is required so the
three physical CTAs retain one logical M coordinate. No tile, stage count,
epilogue, or FP16 arithmetic change is part of this hypothesis.

### Falsifiable mechanism and expected NCU change

Keep `M128N128K64/S2/T128`, grid 333 CTAs, registers, and 65.536 KiB dynamic
shared memory unchanged. Launch 111 clusters of three CTAs. The rank-zero CTA
loads A; all ranks load their own B tile and consume rank zero's A through DSM.

A successful prototype should show:

- L2 sector activity materially below 60.34%, target `<= 45%`;
- LG-throttle `3.21 -> <= 2.5` and long-scoreboard `2.09 -> <= 1.7`;
- eligible cycles above 8% and Tensor-pipe elapsed utilization above 47%;
- direct QKV duration `19.39 -> <= 17.5 us` (`>= 10%`);
- no arithmetic difference; direct output should be byte-identical.

Reject immediately if cluster-barrier stalls exceed about 1.0 per issue,
duration does not fall by at least 5%, cluster launch reduces grid residency,
or DSM latency merely replaces the current L2-hit latency. The largest risks
are remote-shared latency, one cluster-wide synchronization per pipeline
stage, and less flexible CTA scheduling under real S2 overlap.

### Whole-graph bound and priority

A 10% QKV duration reduction is worth roughly 60-80 us per stream-forward,
or around `1%` before second-order peer effects. Eliminating all QKV excess is
not realistic, so a practical target remains `0.5-1.5%`; roughly `3%` is a hard
ceiling. This is meaningful but less certain and more invasive than the next
evidence-backed FFN change. It should follow H39-Q1 and the next FFN result,
not displace them.

## Why no FA4-only candidate is promoted

FA4 remains a large graph component, but the latest NCU does not identify a
cheap resource defect:

- both16 accumulation is already selected;
- fixed-S361 mask specialization was directly slower by 0.236%;
- register-Q and larger-warp variants were slower in prior tests;
- current LG-throttle is effectively zero and long-scoreboard is only 0.51;
- one FA4 CTA and one linear2 CTA can already co-reside statically:
  shared memory is `24.576 + 65.536 = 90.112 KiB`, and their block-register
  demand is `128 * (247 + 162) = 52,352`, both below the SM limits.

Therefore merely lowering FA4 registers/shared memory does not unlock the
important one-FA4-plus-one-linear2 coexistence case. Narrower N tiles add more
online-softmax/barrier iterations without reducing padded Tensor work, and the
existing fixed-tail-mask experiment already falsifies mask bookkeeping as a
useful target. A true FA4 improvement now needs less Tensor work or a wider
fusion boundary; neither is a controlled, higher-confidence single variable.

FA4 should remain behind FFN and the QKV K32/S3 recheck in the investment
queue. Reopen it only if a new AOT/toolchain supplies a demonstrably lower-work
kernel, or if a fresh full-graph profile changes its stall signature.

## Source/result cross-checks

- `stage38-post-rope-half2-profile/accepted-fullgraph-ranking.md`
- `stage38-post-rope-half2-profile/accepted-s2-vs-s1.md`
- `stage21/REPORT.md`
- `stage21/search-m128-n128-k64-s2-t128-mb3.json`
- `stage23/tile-search/qkv-m128n128k32s3t128mb3.json`
- `stage23/qkv-s1-short-abba/`
- `stage34-resource-positive-reaudit/initial-audit.md`
- `katago/cpp/neuralnet/tilelang_aot/wide_qkv_b13.cu`
- `katago/cpp/neuralnet/tilelang_aot/wide_qkv_b13_s1.cu`
- `katago/cpp/neuralnet/fa4_aot/build_aot.py`
