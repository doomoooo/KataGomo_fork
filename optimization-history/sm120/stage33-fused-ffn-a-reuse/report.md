# Stage 33: fused FFN A-fragment reuse

## Decision

**Rejected at the homogeneous-S2 gate.** The candidate is bit exact and is
substantially faster in isolation, but it does not improve the target S2 pair.
Per the preregistered validation order, mixed-peer tests and NCU were not run.
Nothing was integrated into the KataGo source tree or target configuration.

## Frozen protocol

- GPU command: `source /workspace/container-setup/nvidia-env.sh && gpu-lock with --gpu 2 -- ...`
- Observed device: `NVIDIA GeForce RTX 5090 D`
- Shape: fixed B13, 19x19, FP16, `M=4693, N=1152, K=384`
- Target topology: S2 with private input and weight allocations per stream
- Timing: CUDA events, 8 symmetric alternating ABBA cycles, 40 warmups and
  600 iterations per sample
- Control source SHA256:
  `d175acb45f5844de9c6643df08fb40edcd964de318aba65c842772fda10f4348`
- Result timestamp: `2026-08-06 09:29:01.711018204 +0000`

## Single variable

The generated candidate retains each input A fragment after its one
non-transposed `ldmatrix`, then uses that fragment for both the linear and gate
MMA accumulators before advancing `ki`. It changes no tile, stage count, thread
count, grid, shared-memory layout, arithmetic, or output epilogue.

Static generated-source validation reduced the non-transposed A-load sites from
six to three. Both kernels launch `grid=(18,37,1)`, 128 threads, and 32768 bytes
of dynamic shared memory.

## Correctness and resources

| Metric | Control | A-reuse candidate |
|---|---:|---:|
| Output mismatched FP16 words | 0 | 0 |
| Maximum absolute difference | 0 | 0 |
| Registers/thread | 146 | 136 |
| Spill stores / loads | 0 / 0 | 0 / 0 |

The candidate therefore passed deterministic bit-exactness, the no-spill gate,
and the preregistered `<=154` register gate.

## Performance

All values are medians in microseconds.

| Test | Control | A-reuse candidate | Candidate delta | Gate |
|---|---:|---:|---:|---|
| S1 kernel | 37.680 | 34.921 | -7.321% | pass: improvement >0.5% |
| Homogeneous S2 pair makespan | 62.682 | 62.854 | +0.275% | **fail** |
| Homogeneous S2 stream 0 | 59.849 | 60.369 | +0.869% | diagnostic |
| Homogeneous S2 stream 1 | 60.051 | 60.207 | +0.261% | diagnostic |
| Homogeneous S2 overlap | 95.591% | 95.812% | +0.221 pp | diagnostic |

The removed A loads and lower register count explain the strong isolated result,
but the reordered gate MMA delays the next linear-weight prefetch. Under two
concurrent instances that schedule does not reduce pair makespan and slightly
increases each stream's duration. This falsifies H33 for the target topology.

## Stopping rule

The preregistered condition required homogeneous S2 pair improvement greater
than 0.5%. The observed delta is a 0.275% regression. Testing stopped before:

- mixed current-linear2 coexistence;
- mixed wide-QKV coexistence;
- NCU collection;
- integration, whole-network accuracy, or whole-network benchmark.

The route remains closed unless a later full-graph profile identifies a
different dependency mechanism. The accepted mainline and the Stage 31
full-graph ranking remain unchanged.

## Artifacts

- `hypothesis.md`: preregistered hypothesis and gates
- `build_candidate.py`: hash-guarded source transformation and build
- `control.generated.cu`: exact current AOT control translation unit
- `candidate_a_reuse.generated.cu`: sole candidate translation unit
- `build-manifest.json`: commands and hashes
- `build.stderr`: ptxas resource and spill output
- `microbench.py`: deterministic and interleaved timing harness
- `raw-s1-s2.json`: complete samples and gate results, SHA256
  `b0916d2a47a0dbc4a40a6b9788c92aa5095e217676e1fd5b6c0e78b6dac7fc1f`

