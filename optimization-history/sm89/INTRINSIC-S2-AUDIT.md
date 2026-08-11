# 4090 intrinsic optimization versus S2 deployment audit

This audit applies the split retention/deployment policy in
`ITERATION-PROTOCOL.md` to routes previously gated primarily by S2 behavior.
S1 is attribution evidence only; the deployed throughput target remains exact
19x19/B13/S2.

| Stage | Route | Existing intrinsic evidence | Existing S2 evidence | Revised status | Action |
|---|---|---|---|---|---|
| 14 | precomputed RoPE float2 table | local kernel -3.6%, but adds about 36.6 MiB persistent data and more L2 pressure | busy union +2.46% | mechanism tradeoff, not strict resource improvement | no priority retest |
| 15 | batch-grouped fused RoPE | group2 kernel about -4.5%, fewer frequency/trig evaluations, byte-exact | pooled +0.18%, one negative round | obsolete in current graph | no retest: stage16/current `cudaUseQKVRoPEGemmSm89=true` performs RoPE in the QKV epilogue and bypasses this kernel |
| 23 | shared ordinary matmul weights | removes initialization H2D only; steady NCU unchanged/slower | locked S2 -0.72% | mechanism-rejected for forward throughput | no retest |
| 26 | head BN half-to-float dual output | local policy/value boundaries -47%/-38%, two fewer launches | S2 ABBA -1.26% | subsumed by stage 29/52 | no standalone retest: the direct wide-head variant is already in the cumulative bundle and has stable positive S1 attribution |
| 27 | initial global dot plus spatial add | NCU boundary -43%; S1 4/4 positive, +0.118%; full accuracy passed | one S2 probe -1.187% | intrinsic-accepted / S2-regressed | keep default-off in S2; phase-controller retest |
| 28/37 | no-split wide head projection | NCU -54.7%; S1 4/4 positive, +0.599%; full accuracy passed | S2 +0.359%/-2.150% by order | intrinsic-accepted / S2-regressed | keep default-off in S2; phase-controller retest |
| 29 | wide-head BN direct FP32 | NCU boundary -42.4%; S1 4/4 positive, +0.078%; full accuracy passed | prior S2 head-BN route regressed | intrinsic-accepted / S2-regressed | keep with wide-head dependency, default-off in S2 |
| 34 | C384 affine+SiLU vec8 | NCU -31.8%, fewer instructions/launch work, byte-exact | S2 both orders regressed | dominated by stage 54 vec4 | no retest: vec4 is faster locally, has stable positive S1 attribution, and is retained in the cumulative bundle |
| 35 | C384 affine+SiLU vec4 | NCU -36.9%, byte-exact; short S2 Nsys positive | S2 100-iter ABBA -0.819% and order-conflicting | attribution incomplete, high-value audit target | reconstruct and run short S1 attribution before any new micro-tuning |
| 38 | QKV column-major B | isolated NCU appeared faster, but actual continuous-S2 target kernel slowed and conflict metrics worsened | both S2 orders regressed | mechanism-rejected | no retest without a new matched layout design |
| 39 | RMSNorm 8 warps/CTA | only -0.65% NCU, below signal gate | not run | insufficient intrinsic signal | no retest |
| 48/49 | attention/FFN RMS folding | local complete boundaries slowed 1.5% or more | S2 regressed or conflicted | mechanism-rejected | no retest unchanged |
| 51 | fused value terminal projection | four launches to two; boundary about -49%; all-head byte-exact | S2 +0.134%/-1.043%; S1 ABBA -0.033% with split signs | mechanism-accepted / throughput-neutral | implementation retained default-off; reconsider inside a larger value-head fusion |
| 52 | accumulated stages 27+28/37+29 fusion bundle | component NCU boundaries -43%/-54.7%/-42.4%; combined S1 ABBA +0.623% with both pairs positive; six fewer launches per forward | S2 -3.630%/+2.279% by order, with control itself switching phase bands | intrinsic-accepted bundle / S2 phase-sensitive | keep all three default-off in deployed S2; carry the complete bundle into each subsequent intrinsic retest |
| 53 | add stage-51 value-terminal fusion to the cumulative bundle | complete bundle removes eight launches per forward; all four components passed their local and accuracy gates | S2 -3.357%/-2.332%, busy union +3.975%/+2.035% | strict-local bundle retained / S2-regressed | keep all four default-off; continue accumulation rather than deleting locally strict mechanisms |
| 54 | restored C384 affine+SiLU vec4 | NCU 6.816->4.224us (-38.03%), CTA 4693->1760, zero spill; S1 ABBA +0.543% with both pairs positive; 8,192 byte-identical | five-item bundle S2 -1.561%/-1.275% | intrinsic-accepted / S2-disabled | retain default-off as bundle item five; this confirms Stage35 was misclassified by the old S2-only retention gate |
| 55 | fused wide-head BN+SiLU with policy/value pooling | 26-row byte-identical, but NCU policy boundary 8.768->10.368us and value 9.280->10.400us | not run after local mechanism failed | mechanism-rejected | fully reverted; low-wave pooling grids serialize expf and lose the BN path's parallelism |
| 56 | cross-block postConv plus following C768 BN+SiLU | NCU complete boundary 26.496->21.120us (-20.29%), registers 186->164, S1 +1.169%/+1.296%, 8,192-row FP32 envelope passed | S2 short +3.383%/+2.319%; locked ABBA +0.365%, both pairs positive | intrinsic-accepted / S2-enabled | enable as new deployed current best; prior five-item bundle remains default-off after six-item S2 -2.158%/-1.016% |
| 57 | final inner FFN linear2 residual plus following C384 BN+SiLU | NCU complete boundary 30.112->24.640us (-18.17%), removes 11 BN launches/forward; S1 short +0.941%/+0.968%, locked ABBA +0.799% with both pairs positive; 8,192-row FP32 envelope passed | valid forward S2 -2.991%; reverse controls entered a half-throughput contaminated phase; complete seven-route bundle +0.628%/-2.360% by order | intrinsic-accepted / S2-regressed | retain in commit `91f6aae`, default-off; carry it as strict-local item six and retest only through the complete bundle/phase controller |

The clearest previously under-classified wins are stages 27, 28/37, and 29:
they already satisfy both stable S1 gain and strict local work reduction. Stages
26 and 34 are now structurally subsumed by stages 29/52 and 54 respectively;
stage 35 is resolved by stage 54; stage 15 has been structurally subsumed and
must not be benchmarked unchanged.
Stage 52 establishes the cumulative-bundle lane: strict local wins are no longer
discarded or permanently isolated merely because an intermediate S2 bundle has
not yet landed in a stable overlap phase.
