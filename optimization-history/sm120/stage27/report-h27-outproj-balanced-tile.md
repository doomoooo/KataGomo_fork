# Stage 27: fixed-B13/S2 out-projection coexistence search

## Decision

Rejected for the current standalone attention out-projection + residual
boundary. No TileLang candidate approached the actual backend's
`cublasHgemm(beta=1)` S2 result, so no repository integration or full accuracy
run was justified.

Target: RTX 5090 D, fixed 19x19, `B13`, two independent streams, FP16,
`M=4693, N=384, K=384`, private but value-identical weights. The candidate
computes GEMM plus the residual epilogue in one launch.

The discovery sweep originally assumed one shared weight pointer. A later
ownership audit found that each server currently owns an independent `Model`;
the weight-sharing configuration is dead scaffolding. Both the exact cuBLAS
control and the M96N128 finalist were rerun with pointer-distinct weights. The
corrected result below is the acceptance result; the original sweep remains as
historical search evidence only.

## Baseline correction

Stage 22 compared against `torch.addmm`, which measures about 14.6 us S1 and
30--32 us/stream S2 in this environment. It does not select the same compute
path as KataGo's direct FP16 `cublasHgemm` and overstated the opportunity.

The exact C++ cuBLAS boundary measured:

| implementation | S1 median (us) | S2 median/stream (us) | repeat range |
|---|---:|---:|---:|
| `cublasHgemm(beta=1)`, private weights | 8.137 | 11.431 | S1 8.134--8.170; S2 11.426--11.458 |
| M96N128 K32 S3, private-weight confirmation | 9.347 | 18.663 | S1 9.070--9.371; S2 18.223--19.152 |
| M96N128 K32 S3, original shared-weight discovery | 9.183 | 18.032 | S1 8.988--9.252; S2 17.800--18.219 |

The corrected finalist is 63.3% slower than cuBLAS under direct S2. The
ownership correction therefore strengthens rather than weakens the rejection.

## Search results

The original discovery candidates passed the standalone tolerance with maximum absolute error
`1.46484375e-3` and RMSE `1.2507045e-4` against `torch.addmm`.

| tile | CTAs | intended residency | S1 median us | S2 median us/stream |
|---|---:|---:|---:|---:|
| M96N128 K32 S3 T128 | 147 | up to 2 CTA/SM | 9.183 | **18.032** |
| M128N96 K32 S3 T128 | 148 | up to 2 CTA/SM | 8.648 | 18.061 |
| M192N64 K32 S3 T128 | 150 | up to 2 CTA/SM | 8.298 | 18.436 |
| M160N64 K32 S3 T128 | 180 | up to 2 CTA/SM | 10.466 | 19.089 |
| M256N64 K32 S2 T256 | 114 | up to 2 CTA/SM | 9.876 | 18.698 |
| M256N64 K64 S2 T256 | 114 | 1 CTA/SM | 9.449 | 18.160 |
| M256N128 K32 S2 T256 | 57 | 1 CTA/SM | 15.396 | 18.271 |
| M96N128 K32 S4 T128 | 147 | 1 CTA/SM | 8.435 | 19.122 |
| M96N128 K64 S2 T128 | 147 | 1 CTA/SM | 8.373 | 20.314 |
| M128N96 K32 S4 T128 | 148 | 1 CTA/SM | 8.676 | 18.824 |
| M128N96 K64 S2 T128 | 148 | 1 CTA/SM | 8.409 | 20.391 |
| M160N64 K32 S4 T128 | 180 | 1 CTA/SM | 12.185 | 18.943 |
| M192N64 K32 S4 T128 | 150 | 1 CTA/SM | **8.127** | 18.426 |

## Scheduler evidence

NCU for M96N128 K32 S3 reports 147 CTAs, 128 threads, 154
registers/thread, 43.01 KiB dynamic shared memory, a two-CTA shared-memory
limit, 0.43 waves/SM, and 8.32% achieved active-warps occupancy. This is
materially lighter than the current cuBLAS kernel's Nsys launch attributes:
148 CTAs, 128 threads, 164 registers/thread, 80 KiB dynamic and 100 KiB
executed shared memory.

The lighter resource footprint did not produce concurrency. In the corrected
private-weight Nsys trace, the two candidate streams have zero overlapping
`residual_gemm_kernel` pairs. Their approximately 9 us launches serialize,
which explains the approximately 18 us/stream result.

By contrast, the corrected private-weight cuBLAS trace has 237 cross-stream
overlap pairs and 1,300.271 us total overlap, averaging 5.486 us per pair.
Despite allowing only
one CTA/SM, its short 148-CTA waves are interleaved across streams. Raising the
candidate shared-memory footprint to force one CTA/SM did not reproduce that
behavior: all six one-CTA schedules remained at 18.4--20.4 us/stream.

This falsifies the original mechanism. Static occupancy headroom alone does
not predict cross-stream scheduling for this boundary; the library kernel's
launch implementation and CTA dispatch behavior are part of its advantage.

## Artifacts and reopening condition

- Hypothesis: `hypothesis-h27-outproj-balanced-tile.md`
- Raw searches: `search/*.json`, `one-cta-search/*.json`
- Exact baseline: `cublas-outproj-dual.json`
- Private-weight finalist: `private-weight-correction-m96n128.json`
- Nsys: `nsys-m96n128-direct-dual.nsys-rep`,
  `nsys-cublas-direct-dual.nsys-rep`; corrected traces:
  `nsys-m96n128-private-direct-dual.nsys-rep`,
  `nsys-cublas-private-direct-dual.nsys-rep`
- NCU: `ncu-m96n128-candidate.ncu-rep`
- Generated source: `outproj_residual_b13_m96n128k32s3t128mb2.generated.cu`

Do not reopen this as another conventional standalone TileLang tile sweep.
Reopen only for a kernel family that can demonstrate cuBLAS-like S2 overlap in
the exact boundary microbenchmark, or for a changed fusion boundary that
removes additional traffic/launches. Any such candidate still requires whole
network forward/reverse ABBA, Nsys interference/phase analysis, and full
8192-row all-head accuracy before acceptance.
