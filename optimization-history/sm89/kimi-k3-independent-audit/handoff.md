# RTX 4090 B13/19x19 CUDA optimization: independent audit handoff

Date: 2026-08-06 UTC

You are performing an independent, read-only kernel and whole-graph optimization audit. Do not
edit source or result files, do not build, and do not run GPU benchmarks. Inspect the existing
source, profiler artifacts, reports, and neighboring 5090D implementation, then identify any
optimization opportunities we missed. Do not merely restate the existing priority list. Challenge
past conclusions where current graph conditions have changed, but distinguish a genuinely new
mechanism from an unchanged failed experiment.

## Exact scope and objective

- Repository: `/workspace/katago-4090`
- Branch/revision: `4090-opt`, clean `bd6b8a6a32c5b7742b0eb8f872753c3e4d66e638`
- GPU: NVIDIA RTX 4090, Ada SM89
- Software: KataGo 1.17.1, CUDA 13.2.86, cuDNN 9.25.0, TensorRT 10.16.1.11
- Model: `/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz`, model v17,
  11 nested bottleneck outer blocks, C768 trunk; each outer block has preConv 768->384,
  six alternating transformer blocks (three attention and three FFN), postConv 384->768.
- Only shape of interest: exact 19x19, batch 13, FP16 NHWC.
- Deployment topology: S2, two NN server threads, two caller-owned non-blocking CUDA streams on
  one 4090. The objective is maximum aggregate pure-forward `nnEval/s`, not S1 latency.
- Precision policy: FP16 is acceptable. FlashAttention already uses FP16 Tensor Core accumulation
  for both QK and PV while online softmax state/LSE remains FP32. Similar controlled precision
  reductions are allowed if full FP32-reference accuracy stays within the accepted envelope.
- Do not reject an idea because of library/toolchain version availability. A custom AOT
  CUTLASS/CuTe/CUDA implementation is allowed when evidence supports its value.

## Methodology and revised acceptance policy

Read these completely before drawing conclusions:

- `/workspace/SKILL.md`
- `/workspace/results/4090/ITERATION-PROTOCOL.md`
- `/workspace/results/4090/HISTORY.md` (authoritative record of stages 0 through 70)
- `/workspace/results/4090/INTRINSIC-S2-AUDIT.md`
- `/workspace/results/4090/stage64/historical-measurement-gate-audit.md`
- `/workspace/cuda-optimization-history.md` (general strategy catalog previously audited)

Important lessons already learned:

1. NCU durations from separate kernels cannot be added to accept/reject a fusion or split.
   Cross-kernel boundaries require a natural Nsys/NVTX interval or CUDA-event subgraph timing.
   Stage 64 proved this dramatically: independent NCU predicted split QKV+RoPE slower, while
   natural S1 launch-to-completion improved 10.34%.
2. S2 is phase-sensitive. A locally faster operator can shift overlap and reduce aggregate S2.
   Therefore a byte-correct candidate with strictly better same-kernel NCU resources/time and a
   strictly better natural S1 local boundary may be retained behind a default-off switch even if
   S2 is neutral or negative. S2 decides deployment, not whether a sound intrinsic primitive is
   kept. Accumulated local wins can later be retested as a bundle.
3. After every accepted/deployed change, a fresh full-graph Nsys/NCU checkpoint chooses the next
   investment. Failed and fully reverted candidates reuse the prior full-graph profile.
4. Short profiles are deliberately used for iteration. Long ABBA and 8192-row accuracy are only
   run after local and short whole-graph gates pass.
5. The control must preserve the current kernel. Stage 70 initially polluted its control by adding
   a runtime branch that raised both variants to 255 registers; it was corrected to compile-time
   variants before measurement.

## Performance baseline and current best

- Original clean CUDA S2 baseline: 1876.270 nnEval/s (local official retest 1885.957).
- TensorRT exact-19/B13/FP16/S2 locked-2400 baseline: 2432.198 nnEval/s.
- Stage 62 deployed CUDA paired mean: 3424.124 nnEval/s; about +40.8% over TensorRT.
- Stage 68 external-stream no-regression runs: 3462.633, 3460.038, 3450.634 nnEval/s,
  mean 3457.768. This is not a paired causal speedup claim.
- Clean post-Stage68 shortened Nsys trace: 3431.046 nnEval/s under trace.
- Deployment config: `/workspace/bench-cuda-gpu0-4090-s2.cfg`.
- TensorRT evidence: `/workspace/results/4090/tensorrt-baseline/summary.json` and its sibling files.
- A serialized ONNX TensorRT plan that may be structurally inspected exists at
  `/workspace/results/luminal/onnx/plan_19x19_fp16_exact.plan`; do not assume it is byte-identical
  to KataGo's in-process baseline without checking provenance.

Current S2 config enables exact-board mask/bias elimination, B13-only warmup, native D32
FlashAttention both16, dual-GEMM+SwiGLU with half2 tanh, fixed linear2 residual GEMM, fixed
preConv GEMM, fused QK RoPE fallback, fused QKV+RoPE epilogue, C768 and C384 persisting-L2
windows, C768 vec8 affine+SiLU, postConv plus next C768 BN/SiLU epilogue fusion, a fixed cuDNN
frontend initial-convolution plan, fused policy-P1 boundary, and exact-mask preprocessing elision.

The CUDA execution API was corrected in Stage 68: caller creates one explicit non-blocking stream
per handle; CUDA/cuBLAS/cuDNN/custom kernels/H2D/D2H/events/synchronization all consume it. Nsys
captured 5916 forward kernels on explicit streams and zero default-stream forward kernels.
Commit: `bd6b8a6`. Design/evidence:
`/workspace/results/4090/stage68/design-external-cuda-stream.md` and
`/workspace/results/4090/stage68/final-decision-summary.json`.

## Latest full-graph evidence (clean Stage 68)

Read:

- `/workspace/results/4090/stage68/post-stream-current-best-checkpoint.md`
- Nsys report: `/workspace/results/4090/stage68/post-stream-full-nsys-s2/current.nsys-rep`
- Nsys SQLite: locate the sibling exported `.sqlite` if needed
- NCU report: `/workspace/results/4090/stage68/post-stream-full-ncu-s2/current.ncu-rep`
- Existing text/CSV exports in the same Stage 68 directories

Kernel span is 132.982 ms and busy union is 131.180 ms (98.645%); launch gaps are not the primary
macro bottleneck. Natural S2 family accounting:

| Family | Launches | Raw ms | Union ms | Exclusive ms | Union share | Exclusive share |
|---|---:|---:|---:|---:|---:|---:|
| dual FFN projection + SwiGLU | 1122 | 48.540 | 48.501 | 24.129 | 36.97% | 18.39% |
| QKV + fused RoPE | 1122 | 34.120 | 34.076 | 14.756 | 25.98% | 11.25% |
| FlashAttention both16 | 1122 | 30.435 | 30.365 | 6.800 | 23.15% | 5.18% |
| FFN linear2 + residual | 1122 | 30.395 | 30.242 | 2.353 | 23.05% | 1.79% |
| attention out-projection | 1122 | 24.413 | 24.328 | 1.697 | 18.55% | 1.29% |
| RMSNorm | 2244 | 13.466 | 13.466 | 1.365 | 10.27% | 1.04% |
| outer postConv | 374 | 10.038 | 10.029 | 0.911 | 7.64% | 0.69% |
| outer preConv | 374 | 7.272 | 7.249 | 0.631 | 5.53% | 0.48% |
| heads/frontend/other | 1032 | 6.018 | 4.621 | 2.994 | 3.52% | 2.28% |

Representative broad NCU:

| Family | us | SM% | DRAM% | L2% | issue% | active warps% | regs | dyn smem | waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dual FFN + SwiGLU | 41.632 | 67.6 | 16.0 | 73.0 | 19.8 | 15.3 | 168 | 49.152 KiB | 2.60 |
| QKV + RoPE | 30.528 | 45.9 | 15.1 | 53.6 | 20.8 | 14.9 | 240 | 49.152 KiB | 1.30 |
| FlashAttention both16 | 23.584 | 42.7 | 46.9 | 49.5 | 48.1 | 29.3 | 117 | 16.768 KiB | 1.83 |

Do not rank only by raw summed duration: S2 exclusive/union and resource complementarity matter.

## High-impact accepted evolution

The full history has exact measurements; this is only a navigation summary:

- Wide batched QKV and FFN replaced many small cuBLAS launches.
- Fused residual epilogues and a custom warp-per-row RMSNorm.
- Exact-board identity mask and attention-bias removal: roughly +19% at Stage 6.
- Native-D32 M64xN96 4-warp FlashAttention: roughly +9.9% at Stage 7.
- CUTLASS B13 dual-GEMM + fused SwiGLU: roughly +11.5% at Stage 8.
- Fixed linear2 residual, nested preConv, fused Q/K RoPE, then QKV-GEMM RoPE epilogue.
- C768/C384 persisting-L2 windows.
- Vectorized affine+SiLU, fixed initial-conv frontend tactic, policy-P1 fusion, exact mask-sum
  elision.
- Cross-block postConv epilogue also emits the following C768 BN+SiLU: local boundary -20.29%,
  locked S2 +0.365%, deployed.
- FlashAttention QK+PV both16: NCU 28.224 -> 20.864 us (-26.08%), S2 +4.375%, deployed,
  commit `7d299d0`.
- Dual-FFN half2 tanh sigmoid/SwiGLU: kernel about -1.23%, S2 +1.033%, deployed,
  commit `6fd19dc`.

Retained intrinsic but default-off primitives include split QKV+standalone RoPE, native-half plain
QKV epilogue, initial-global fusion, wide unsplit head, direct-FP32 wide-head BN, C384 vec4
affine+SiLU, value terminal projection fusion, and final-inner linear2 plus following C384
BN+SiLU. Read `INTRINSIC-S2-AUDIT.md` for dependency/subsumption status.

## Recent failed experiments that must not be proposed unchanged

Use `/workspace/results/4090/HISTORY.md` for all failures and reopen conditions. Particularly:

- Dual FFN M128xN128/8-warp vs current M128xN64/4-warp: despite regs 168->156 and less L2
  pressure, corrected swizzle-1 NCU was 40.96 -> 44.22 us (+7.96%). Stage 69.
- Fused-QKV precomputed RoPE table retest with true compile-time control: bit-exact but NCU median
  29.696 -> 30.368 us (+2.263%), regs 240->255 and long-scoreboard stall 1.589->2.844. Stage 70.
- Plain-QKV stage2 increased regs and slowed; N64 tile doubled CTA but slowed badly.
- QKV alternative column-major B looked faster under isolated NCU but was slower in continuous
  natural execution; a mismatched copy-map variant was numerically wrong.
- RMS folding into QKV/FFN epilogues slowed the complete natural boundaries.
- Full-row projection+next-RMS fusion padded N384 to N512, had too few waves, and was +72.5%.
- Head BN+pool fusion serialized expf work and was +12-18% locally.
- FlashAttention tile/packing/min-block variants were already swept before both16; revisit only
  with a new mechanism or because both16 materially changes the resource optimum.
- Stream-K attempts did not improve the fixed low-wave shapes.
- Several strict local fusions worsened S2 solely by phase/overlap. They remain relevant only as a
  cumulative bundle or under phase control, not as unchanged standalone proposals.

## Neighboring implementations and phase work

- RTX 5090D repository: `/workspace/katago` (SM120 work is mainly
  `cpp/neuralnet/cudabackend_sm120*`; its wrapper currently delegates much of the forward to the
  official CUDA model). Compare algorithms and precision choices, not SM120-only launch features.
- General 4090 vs 5090 phase-controller design:
  `/workspace/cuda-dual-stream-trunk-phase-design.md`. It proposes shared logical trunk checkpoints
  and CUDA-event gating between two lanes. It is not implemented. Its statements about PTDS are
  stale after Stage 68; the explicit external stream architecture now provides the correct base.
- Phase control is a real macro opportunity, but this audit must also look for kernel/fusion/layout
  opportunities independent of phase control.

## Source navigation

Start with:

- `cpp/neuralnet/cudabackend_sm89_forward.cpp` for full forward, trunk/block sequencing, buffers,
  layouts, cuDNN/cuBLAS integration, and operator dispatch.
- `cpp/neuralnet/cudabackend_sm89*.cu` and headers for fixed-shape kernels.
- `cpp/neuralnet/cudabackend.cpp` for the official path and reusable operators.
- `cpp/neuralnet/cudabackend_sm120*` in `/workspace/katago` for 5090D ideas.
- Current accepted commit series from `git log`; default-off mechanisms are present in source.

## Required independent output

Return a written audit in Chinese with these sections:

1. **Executive verdict:** whether any plausible untried route can still deliver >1%, >3%, or >5%
   aggregate S2 throughput, with honest uncertainty.
2. **Missed-opportunity table:** at least five candidates if evidence supports them. For each give:
   exact source boundary/files, profiler evidence, mechanism, why prior stages did not already test
   it, expected local and S2 upside ranges, implementation cost, numerical risk, S2 phase risk, and
   the cheapest falsification experiment (specific Nsys/NCU counters/boundary).
3. **Top three next stages:** ordered by expected value = upside × success probability / cost.
   Each must be concrete enough for another engineer to implement without guessing the boundary.
4. **Historical false-negative audit:** identify past rejected/default-off routes whose assumptions
   changed after both16, half2 tanh, cross-block fusion, or explicit streams. Do not reopen unchanged
   failures without a specific changed mechanism.
5. **TensorRT/5090D lessons:** specific transferable fusion/layout/precision/cache ideas, and what is
   architecture-specific or already subsumed.
6. **Blind spots in our measurement workflow:** any way our profiles or gates could still hide a
   true optimization, including concurrency/fairness, cache warmness, launch-order, or metric
   attribution problems.
7. **No-op list:** attractive ideas that should not consume more time, tied to existing evidence.

Be adversarial and source-grounded. Cite local paths and line numbers or symbols. If an estimate is
an inference rather than directly measured, label it. Do not claim that isolated NCU durations add
up across kernels. Do not edit anything.
