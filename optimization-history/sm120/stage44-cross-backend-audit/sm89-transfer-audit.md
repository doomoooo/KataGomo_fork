# RTX 4090/SM89 -> RTX 5090 D/SM120 transfer audit

Date: 2026-08-06 UTC

## Scope and decision

This is a read-only audit for the fixed production target: RTX 5090 D, exact
19x19, B13, FP16/NHWC, and the natural two-server/two-stream whole graph. No GPU
command was run, so the CUDA ordinal versus `nvidia-smi` index mapping did not
enter this audit.

The main conclusion is that the 4090 history contributes three credible missing
experiments that were not in the previous 5090 D priority list:

1. eliminate the still-live exact-board mask preprocessing;
2. fuse each outer `postConv` residual epilogue with the following C768
   BN/SiLU (including trunk tip on the final block);
3. fuse the final inner FFN `linear2` residual epilogue with the outer C384
   BN/SiLU.

The first is the best immediate experiment because it is exact, small, and
almost mechanical. The two cross-boundary epilogues have much larger local
boundaries and are the best subsequent implementation program. They are
materially different from the already-rejected standalone outer AOT sweep: the
new mechanism deletes a full tensor read and a launch, rather than merely
replacing cuBLAS with a slower low-resource GEMM.

The source of the transfer evidence is the complete SM89 history
(`/workspace/results/4090/HISTORY.md:1-451`), especially Stage 50
(`/workspace/results/4090/HISTORY.md:375-380`), Stage 56
(`/workspace/results/4090/HISTORY.md:417-428`), and Stage 57
(`/workspace/results/4090/HISTORY.md:429-439`). Current costs come from
`/workspace/results/rebuild/stage38-post-rope-half2-profile/accepted-s2-vs-s1.json`.

## Current SM120 cost accounting

The Stage38 ranking implies about `5702.97 us` of summed S2 kernel work per
stream-forward. The following costs are not wall-clock savings and must not be
added as though the two streams were serialized. They are search-budget and
boundary-size evidence; natural whole-graph S2 remains the deployment decision.

| Candidate boundary | Current S2 work (us/stream-forward) | S2 excess (us/stream-forward) | Current work share | What can actually be deleted |
|---|---:|---:|---:|---|
| Exact mask extract + half-to-float + sum | `5.72` | `1.94` | `0.10%` | 3 launches, all target work |
| 11 postConv + following C768 BN/SiLU | `348.33` | `134.68` | `6.11%` | 11 BN launches and 11 full C768 residual reads; GEMM remains and its epilogue grows |
| 11 final-inner linear2 + outer C384 BN/SiLU | `402.74` | `136.60` | `7.06%` | 11 BN launches and 11 full C384 residual reads; linear2 remains and its epilogue grows |
| p1/g1/v1 narrow head projections | `36.49` | `16.46` | `0.64%` | 3 GEMMs become 1, hence only the difference versus the wide GEMM is available |
| initial global matmul/reduce/broadcast-add | `16.77` | `5.19` | `0.29%` | 3 launches become 1 |
| value + score terminal projections/biases | `11.69` | `2.86` | `0.20%` | 4 launches become 2 |

Derivations from the Stage38 JSON:

- mask: `(0.100738 + 0.084835 + 0.157474) ms / 60 * 1000 = 5.717 us`;
- Stage56-shaped boundary:
  `15.251907/60*1000 + (10/11)*(5.680412/60*1000) + 0.483877/60*1000 = 348.330 us`.
  The `10/11` excludes the first outer pre-BN, which has no previous postConv;
  the trunk-tip BN is then added explicitly;
- Stage57-shaped boundary:
  `(11/33)*(61.746201/60*1000) + 3.582147/60*1000 = 402.737 us`;
- the corresponding excess values use the same formulas with
  `0.031618+0.029155+0.055714`, `5.763087`, `2.351102`, `0.180517`,
  `20.472471`, and `1.372107`.

The source JSON identifies the post projection at
`accepted-s2-vs-s1.json:378-401`, the next C768 norms at `523-545` plus trunk
tip at `679-692`, the C384 post norm at `594-617`, and mask preprocessing at
`919-932` and `1039-1067`. The concise current ranking independently reports
postConv `254.2 us/fwd`, linear2 `1029.1 us/fwd`, outer pre/post norms
`94.7/59.7 us/fwd`, and their NCU resource signatures
(`/workspace/results/rebuild/stage38-post-rope-half2-profile/accepted-fullgraph-ranking.md:9-29`).

## Ranked transfer queue

### 1. Exact-19 mask preprocessing elision — implement first

Current SM120 still allocates mask scratch, extracts channel 0, converts the
half mask to float, and reduces it before setting only the mask pointers to null
(`/workspace/katago/cpp/neuralnet/cudabackend.cpp:3358-3390`). Stage38 still
contains all three ordinals: aggregate `5.72 us/stream-forward`, three launches,
and `1.94 us` interference excess.

SM89 Stage50 removed exactly these launches by uploading a persistent B13 float
vector containing `361` once, while retaining the non-exact fallback
(`/workspace/results/4090/stage50/hypothesis-exact-mask-elision.md:3-17`). Its
isolated boundary went `6.848 -> 0 us`; locked S2 ABBA measured
`3218.860 -> 3253.014 nn/s` (`+1.061%`), and the 8192-row output was byte-identical
(`/workspace/results/4090/stage50/final-decision-summary.json:9-18,38-58`).

Portability is high. `requireExactNNLen` already makes downstream mask pointers
null on SM120. For an exact full board, channel 0 is one at all 361 points, and
the FP32 reduction result is exactly representable as `361.0f`. Replacing only
that producer with a fixed B13 `maskSum` changes neither pooling formulas nor any
trunk arithmetic. Guard on exact 19x19 and exact B13; retain the official path
for every other shape. A credible expected whole-graph range is
`+0.05%--0.30%`; the 4090 `+1.06%` should be treated as phase-amplified upside,
not as a portable promise.

### 2. postConv -> next C768 BN/SiLU — highest-confidence large fusion

SM89 Stage56 observed strict adjacency, then made each postConv epilogue write
both the original C768 residual and the following block's activated C768 input.
The final block generated trunk-tip activation. It deleted 11 launches and 11
full C768 residual reads per forward
(`/workspace/results/4090/stage56/hypothesis-postconv-next-bn-fusion.md:9-31`).
Its S2-sourced NCU boundary was `26.496 -> 21.120 us` (`-20.29%`), registers
fell `186 -> 164`, and waves/SM stayed `0.87`. Short S2 improved in both orders;
locked S2 ABBA was `3277.003 -> 3288.971 nn/s` (`+0.365%`, both adjacent pairs
positive), followed by full 8192-row FP32-envelope acceptance
(`/workspace/results/4090/stage56/final-decision-summary.json:13-29,31-62`).

This is not the rejected SM120 standalone outer AOT route. The current standalone
contract/expand sweep found contract consistently slower and expand noise-sized,
and explicitly allowed reopening only for a fusion that removes traffic or
launches (`/workspace/results/rebuild/stage28/report-h28-outer-projection-aot.md:89-96`).
Stage56 is precisely that reopen condition.

The implementation cost is high: current generic SM120 execution independently
runs outer post-BN and postConv (`/workspace/katago/cpp/neuralnet/cudabackend.cpp:1539-1577`),
and no accepted SM120 postConv AOT exists. A fixed SM120-native postConv mainloop
plus dual-output epilogue and block-stack lookahead are required. Nevertheless,
the current `348.33 us/fwd` boundary and `134.68 us/fwd` excess justify the work.
A credible S2 search budget is `+0.20%--0.70%`.

### 3. final inner linear2 -> outer C384 BN/SiLU — strong intrinsic candidate

Every nested block ends its six inner blocks with an FFN. SM89 Stage57 extended
the same output-iterator mechanism to the final linear2, deleting 11 C384
BN/SiLU launches and residual reads
(`/workspace/results/4090/stage57/hypothesis-linear2-outer-bn-fusion.md:7-27`).
The complete NCU boundary was `30.112 -> 24.640 us` (`-18.17%`), although the
fused kernel itself was `6.65%` slower than linear2. S1 ABBA was a clean
`+0.799%`, zero-spill, and full accuracy passed. Its one valid S2 order was
`-2.991%`; the reverse controls fell into a contaminated half-throughput phase,
so SM89 retained the source but left it disabled
(`/workspace/results/4090/stage57/final-decision-summary.json:18-30,32-69,80-86`).

Under the current workflow this is not an implementation rejection: it is a
large, correct intrinsic boundary reduction that must go directly to the
current natural whole graph after S1/NCU proof. Porting is somewhat easier than
Stage56 because SM120 already owns a fixed B13 linear2 AOT. Its current
Stage57-shaped boundary is `402.74 us/fwd` with `136.60 us/fwd` excess. A
credible natural-S2 prior is `0%--0.50%`, with a stronger expected S1 gain of
roughly `0.4%--1.0%`; deploy only if the current whole graph is stable.

### 4. QKV projection + RoPE epilogue — large upside, high resource risk

SM89 Stage16 fused learnable RoPE into the QKV epilogue, removed 33 standalone
RoPE launches, and improved locked S2 by `+3.40%`
(`/workspace/results/4090/HISTORY.md:48,149-158`). Current SM120 still spends
`785.1 + 177.9 = 963.0 us/fwd` on this boundary, or `16.89%` of summed work.

It is not an immediate port recommendation. The SM89 fused kernel used 240
registers/thread and 49.15 KiB shared memory, while the current SM120 QKV uses
136 registers and 65.5 KiB. More importantly, the 5080 fused-QKV/RoPE AOT was
directly rejected at `-1.038%`
(`/workspace/cuda-optimization-history.md:121`), and the prior 5090 D audit
therefore placed unchanged QKV+RoPE AOT in the do-not-repeat list
(`/workspace/results/rebuild/stage27/parallel-history-crosscheck.md:234-245`).

Reopen only if a low-live-range epilogue can preserve the current planar layout
and avoid materially raising registers, or after another QKV mainloop change
creates room. Do not mechanically port the SM89 240-register CUTLASS kernel.

### 5. No-split C384 wide head — still unimplemented, but smaller on current SM120

SM89 Stage28 concatenated p1/g1/v1 weights (`96+96+192=384`), ran one
M4693/N384/K768 projection, and let first consumers read stride-384 slices
without materializing splits. Local projection work fell `35.164 -> 15.940 us`;
S1 ABBA improved `+0.599%`, 4/4 pairs positive, and output was byte-identical
(`/workspace/results/4090/stage28/final-decision-summary.json:11-29,31-70`).

Current SM120 still has the three narrow projections, totaling
`36.49 us/fwd` in S2 with `16.46 us/fwd` excess. However, current isolated
ordinal medians total only about `20.0 us`, already close to the SM89 wide
candidate's `15.94 us`; therefore the old 54.7% local saving is not portable.
The SM120 fused-P1 prerequisite is already present in source but remains
default-off after order-conflicting short current-mainline tests
(`/workspace/katago/cpp/neuralnet/cudabackend.cpp:3010-3044` and
`/workspace/results/rebuild/stage30-head-audit/hypothesis-fused-policy-p1.md:80-95`).

Retain wide head as a staged candidate after the three items above. Require one
C384 output with stride-aware p1/g1/v1 consumers and no split copy. Expected S2
budget: `+0.05%--0.25%`.

### 6. Lower-priority retained SM89 ideas

| Route | Current SM120 assessment | Action |
|---|---|---|
| Initial convolution fixed frontend plan (SM89 Stage24) | Current work `24.2 us/fwd`; SM120 has already enumerated engine 47 at `29.150 -> 16.172 us` event boundary but has not integrated it into the natural graph | Ready low-cost fallback; expected about `0.05%--0.20%` |
| Initial global dot + broadcast-add (SM89 Stage27) | Missing runtime path; current boundary `16.77 us/fwd`; SM89 local `-43%`, S1 `+0.118%`, one-order S2 probe `-1.187%` | Low priority unless frontend fusion widens the boundary |
| Wide-head BN directly to FP32 (SM89 Stage29) | Missing, but depends on an accepted wide head; current two copy launches total only about `6.40 us/fwd` | Test once only after wide head; `0%--0.10%` |
| Value/score terminal 384->9 fusion (SM89 Stage51) | Missing; current boundary `11.69 us/fwd`; SM89 local `-48%` but S1 ABBA `-0.033%` and S2 order conflict | Retain as reusable low-priority code idea, not a current hotspot |
| C384 affine/SiLU vec4 (SM89 Stage54) | Superseded: current SM120 half2 C384 isolated average is about `3.35 us`, already faster than SM89 vec4 `4.224 us`, with 17 rather than 23 registers | Do not port unchanged |

The initial-conv SM120 evidence is documented at
`/workspace/results/rebuild/stage39-priority-audit/report.md:206-215`; the
initial-global caution at `217-222`. Stage27/29/51/54 primary results are at
`/workspace/results/4090/stage27/final-decision-summary.json:10-67`,
`stage29/final-decision-summary.json:12-61`,
`stage51/final-decision-summary.json:10-69`, and
`stage54/final-decision-summary.json:15-64`.

## Rounding and accuracy contract for the two cross-boundary fusions

The important transferable detail is not merely “fuse BN”. It is where the
rounding boundary is preserved.

The SM89 output iterator first stores the post-residual half fragment, copies
that already-rounded fragment, loads half scale and bias, executes half FMA,
converts to float for `expf`/SiLU, and rounds the activated result back to half
(`/workspace/katago-4090/cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu:150-190`).
This matches the current SM120 standalone half2 affine/SiLU order:
`__hfma2`, half-to-float, float `expf`, and float-to-half
(`/workspace/katago/cpp/neuralnet/cudabackend_sm120_kernels.cu:500-506`).

For Stage57, changing only the accepted current linear2 AOT epilogue can preserve
the existing GEMM mainloop and residual half bits, making byte identity a
reasonable target. For Stage56, replacing the current cuBLAS postConv with a
custom mainloop may change accumulation order even though the inter-operator
half rounding is preserved. The SM89 Stage56 replay was not byte-identical to
Stage50, but it passed the established full FP32 envelope. Therefore:

- require boundary bit identity for the Stage57 prototype unless SASS/compiler
  evidence explains a harmless difference;
- expect Stage56 to require full all-head FP32 regression rather than claiming
  byte identity;
- in both cases, the residual buffer must still be written because it is needed
  by the outer skip connection; only its second full read by BN is removed.

## Explicit deduplication: do not repeat unchanged

- Standalone outer contract/expand CUTLASS2 AOT was already tested on 5090 D and
  rejected; Stage56 is a distinct, wider fusion boundary.
- C768 flat vec8 affine/SiLU already received a current-mainline long natural-S2
  rejection (`3800.148 -> 3793.430 nn/s`) and is not reopened by SM89 Stage54,
  which is C384 and slower than the current half2 kernel.
- Fused P1 is already implemented on SM120 and screened; it is plumbing for a
  future wide head, not a new standalone priority.
- Shared ordinary weights failed SM89 long S2 (`-0.72%`) and has low value on the
  96 MiB-L2 5090 D; do not implement now.
- RMS folding, head BN+pooling fusion, projection+full-row RMS fusion, Stream-K,
  QKV B-copy alias substitution, and the SM89 FFN stage/cache/wait variants were
  mechanism- or correctness-rejected. The accepted SM120 FFN/QKV implementations
  also differ enough that these are not portable missing features.
- Internal exact-board masking and attention-bias removal already happen through
  `requireExactNNLen`; only the three input mask preprocessing launches remain.
  Fixed-S361 attention tail-mask specialization is a separate, already-rejected
  route and must not be confused with Stage50.

## Recommended execution order

1. Implement exact-mask preprocessing elision as one exact B13/19x19 switch.
2. Run correctness, S1/NCU mechanism evidence, natural whole-graph S2, and if
   accepted the full long/replay/profile/commit loop.
3. From that fresh whole-graph profile, choose between Stage56 and Stage57. The
   current prior favors Stage56 for deployment confidence and Stage57 for lower
   implementation cost; do not implement both as one bundle.
4. Only after each accepted fusion and fresh whole-graph rerank decide whether
   QKV+RoPE low-register feasibility, wide head, or initial-conv engine 47 is the
   next best investment.

This sequence follows the current rule that every accepted optimization is
committed independently and followed by fresh natural-S2 Nsys plus matching
344-ordinal S1 NCU before selecting the next target.
