# Stage32 linear2-residual M128N96 rerank

## Decision

Reject all tested `M128N96K32/T128` candidates for the fixed RTX 5090 D,
B13, 19x19, S2 target.  `M128N96/S4` has a large isolated S1 advantage, but
does not clear the predeclared greater-than-0.5% homogeneous-S2 threshold and
regresses the wide-QKV mixed pair.  No KataGo source, CUDA source, or runtime
configuration was changed.

The adjacent `M96N96` check is also rejected at the S1 and homogeneous-S2
screens, so no mixed-peer run was justified for that family.

## Exact controls

The exact C++ control calls `cublasHgemm` with FP16 `alpha=1`, `beta=1`, using
the row-major-to-column-major transpose mapping for
`[4693,1152] * [1152,384] + [4693,384]`.  Each stream owns a distinct weight
allocation.

| Control | S1 median | S2 stream 0 | S2 stream 1 | S2 pair median |
|---|---:|---:|---:|---:|
| C++ `cublasHgemm(beta=1)` | 22.905 us | 33.403 us | 33.544 us | 33.764 us |
| Current TileLang AOT, focused ABBA | 22.864 us | 29.557 us | 29.596 us | 33.246 us |

The C++ control matters: `torch.addmm(out=...)` dispatched to a different
library path and measured 31.905 us S1 in the same Python process, so it was
not used as the acceptance baseline.  A direct C++ launch of the generated
incumbent and N96/S4 kernels against `cublasHgemm(beta=1)` produced bit-exact
FP16 output for all 1,802,112 elements (`max_abs=0`, `relative_l2=0`).  The
separate Python dispatch check was also within tolerance (`max_abs=0.00390625`,
RMSE 0.0003313).

## M128N96 screen

The first private-weight screen used 7 repeats, 300 timed operations, and a
common phase gate for the two streams.  The incumbent was measured both before
and after the candidates; its bracket was 22.928/22.894 us S1 and
33.211/33.069 us S2 pair.

| Tile | S1 median | Homogeneous S2 pair | Result versus incumbent bracket |
|---|---:|---:|---|
| current `M128N128/S4` | 22.928 us | 33.211 us | control |
| `M128N96/S2` | 18.987 us | 35.378 us | S2 +6.5% to +7.0%, reject |
| `M128N96/S3` | 19.152 us | 35.825 us | S2 +7.9% to +8.3%, reject |
| `M128N96/S4` | 19.115 us | 33.699 us | S2 +1.5% to +1.9%, focus again |
| current post bracket | 22.894 us | 33.069 us | control |

Because S4 was the only close S2 result, it was repeated with 6 ABBA cycles,
400 operations per sample, and 12 samples per variant.  Absolute S1 timing for
the N96 kernel drifted with sustained load (17.76--18.58 us in this run), but
the interleaved S2 and mixed comparisons are stable enough for the decision.

| Focused pair | Current pair | N96/S4 pair | Delta | Current/N96 overlap |
|---|---:|---:|---:|---:|
| Homogeneous linear2 + linear2 | 33.246 us | 33.092 us | -0.462% | 87.79% / 87.35% |
| linear2 + fused-FFN | 50.073 us | 48.585 us | -2.973% | 91.37% / 91.25% |
| linear2 + wide-QKV | 33.324 us | 33.673 us | **+1.049%** | 89.10% / 87.88% |

For the wide-QKV pair, candidate linear2 duration improved from 30.072 to
29.840 us, but peer duration regressed from 29.676 to 30.061 us.  The pair
makespan and overlap both worsened, which independently triggers the rejection
rule.  The fused-FFN result alone is not sufficient to retain the candidate.

## Resource evidence

Nsight Compute collected one isolated launch for each schedule.  The `waves`
value includes the schedule's maximum resident block count, so a smaller value
for S2/S3 does not mean the physical grid became smaller.

| Tile | CTAs | Registers/thread | Dynamic shared memory | Shared-memory resident limit | Waves/SM | Achieved occupancy |
|---|---:|---:|---:|---:|---:|---:|
| current `M128N128/S4` | 111 | 162 | 65.54 KiB | 1 CTA | 0.65 | 8.25% |
| `M128N96/S2` | 148 | 138 | 28.67 KiB | 3 CTAs | 0.29 | 8.24% |
| `M128N96/S3` | 148 | 160 | 43.01 KiB | 2 CTAs | 0.44 | 8.24% |
| `M128N96/S4` | 148 | 162 | 57.34 KiB | 1 CTA | 0.87 | 8.25% |

S4 validates only the first half of the hypothesis: 148 CTAs improve isolated
card coverage.  It does not reduce registers, remains limited to one CTA/SM,
and its 58.37 KiB total per-block allocation cannot co-reside with the current
wide-QKV block (about 66.56 KiB) in 102.4 KiB per-SM shared memory.  The larger
148-CTA grid therefore adds scheduling pressure without opening a new
co-residency mode.  S2 and S3 reduce resources enough for more same-kernel CTA
residency, but their homogeneous S2 pair times are substantially worse.

An isolated eligible-warp counter moved only from 0.71% of peak for the
incumbent to 0.76% for N96/S4.  This counter was collected outside the full S2
graph and is used only as supporting evidence, not as a substitute for the
pair timings.

## Adjacent M96N96 boundary check

The M128N96 result exposed a shared-memory/co-residency boundary, so the
predeclared adjacent `M96N96` check was run with the same private-weight and
phase-gated protocol.

| Tile | S1 median | Homogeneous S2 pair | S1 delta |
|---|---:|---:|---:|
| current `M128N128/S4` | 22.900 us | 33.162 us | control |
| `M96N96/S2` | 25.873 us | 39.194 us | +13.0% |
| `M96N96/S3` | 25.899 us | 37.543 us | +13.1% |
| `M96N96/S4` | 26.243 us | 37.748 us | +14.6% |

All three fail the S1-neutral requirement before mixed-peer testing.

## Artifacts

- `hypothesis.md`: pre-code falsifiable hypothesis and decision rule
- `cublas_linear2_control.cu`, `cublas-control.json`: exact C++ library control
- `exact_aot_cublas_check.cu`, `exact-aot-cublas-check.json`: direct AOT versus
  cuBLAS boundary correctness
- `exact_tilelang_micro.py`, `exact-private-micro.json`: all N96 stages and peers
- `focused_interleaved.py`, `focused-interleaved.json`: ABBA S4 confirmation
- `adjacent_m96_screen.py`, `adjacent-m96-screen.json`: allowed boundary check
- `profile_one.py`, `ncu-*.ncu-rep`: launch and occupancy evidence
- `linear2_m128n96k32s{2,3,4}t128mb3.generated.cu`: preserved generated candidates

The tested hypothesis is closed: increasing the grid from 111 to 148 CTAs can
make linear2 much faster in isolation, but the S2 win is below threshold and
is paid for by a measurable wide-QKV peer regression.  Integration is not
recommended.
