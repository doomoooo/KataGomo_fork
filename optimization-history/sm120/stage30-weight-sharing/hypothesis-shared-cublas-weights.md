# H30: shared versus private ordinary cuBLAS GEMM weights

Date: 2026-08-06 UTC

## Scope and frozen target

- Read-only integration exploration: no KataGo shared C++/CUDA source is
  modified. The experiment is a standalone CUDA/cuBLAS microbenchmark under
  `results/rebuild/stage30-weight-sharing/`.
- GPU: RTX 5090 D (SM120), CUDA runtime device 2, exclusively held by
  `gpu-lock` for every GPU run.
- Fixed logical shape: exact 19x19, B13, so `M = 13 * 361 = 4693` rows.
- FP16 `cublasHgemm`, with one independent nonblocking stream, cuBLAS handle,
  input allocation, and output allocation per simulated NN server.
- Control gives the two streams pointer-distinct but byte-identical weight
  allocations. Candidate gives both streams the same weight pointer. This is
  the only timed-path difference.
- Both streams, handles, inputs, outputs, and weight 0 are allocated in the
  same order before the control-only weight 1 allocation. Thus the extra
  private allocation cannot shift any common timed operand's allocation
  address. An earlier pilot that allocated weight 1 before the inputs is
  retained only as superseded raw evidence and is not used for the decision.
- Three ordinary library-GEMM boundaries are measured separately:

| boundary | logical M/N/K | beta | weight bytes |
|---|---:|---:|---:|
| attention out-projection | 4693 / 384 / 384 | 1 | 294,912 |
| outer pre-projection | 4693 / 384 / 768 | 0 | 589,824 |
| outer post-projection | 4693 / 768 / 384 | 1 | 589,824 |

## Prior evidence

RTX 4090 Stage23 implemented process-local sharing for all ordinary matmul
weights. Startup evidence proved physical deduplication: 476 allocations and
H2D copies disappeared, totaling 254,435,328 bytes. Its representative
out-projection was already about 99.3% L2-hit and remained flat in NCU; the
completed long forward/reverse full-graph result regressed by 0.716%, so the
route was rejected.

The 5080 history reports only `+0.126%` for ordinary-weight sharing, while
sharing additional special QKV/outer weights reduced the signal to `+0.028%`.
The 5090D has 96 MiB total L2, so duplicate weights create less relative cache
pressure than on either older target. These results make a large benefit
unlikely, but they do not answer whether Blackwell's concurrent cuBLAS launch
behavior benefits from the two streams reading the same physical lines.

## Mechanism

With private weights, corresponding CTAs on the two streams read equal values
from two physical regions. With a shared allocation, both streams address the
same L2 lines. If duplicate weight residency contributes to the current
`library_gemm` S2 interference bucket, sharing should reduce combined L2/DRAM
traffic or let the two cuBLAS launches overlap more effectively.

The tested matrices occupy only 0.28-0.56 MiB each, far below the 5090D L2.
Repeated single-shape microbenchmarks therefore isolate immediate cross-stream
reuse but do not reproduce the complete model's aggregate live weight set. A
positive micro signal is only a reason to build a lifetime-safe integration
candidate; it is never an end-to-end acceptance result. A flat result is
strong evidence against immediate integration, but cannot prove aggregate
deduplication has zero whole-model effect.

## Falsifiable predictions

1. S1 private/shared results should be indistinguishable. Stream 0 uses the
   same allocation pattern and GEMM call in both modes; a stable S1 difference
   indicates measurement or setup contamination.
2. Kernel name, launch geometry, count, math mode, input/output ownership, and
   submission order must be identical in S2. Only the stream-1 weight pointer
   aliases stream 0 in the candidate.
3. A useful S2 result requires consistent improvement in both ABBA and reverse
   BAAB order beyond within-arm repeat dispersion for at least the dominant
   out-projection and no material regression in outer pre/post projections.
4. Nsys must attribute any signal to lower two-stream union/per-stream elapsed
   time or greater cross-stream overlap, not a tactic or launch-count change.
5. NCU is run only if timing/overlap needs explanation. The supporting
   mechanism would be higher L2 hit, fewer DRAM bytes/sectors, or lower kernel
   time under the shared pointer. Serialized NCU replay cannot by itself prove
   S2 benefit.

## Protocol and decision

1. Build a standalone extension of Stage27
   `cublas_outproj_dual_micro.cu`. Initialize private weights from the same host
   vector and verify pointer identity/distinction in structured output.
2. For every shape, run long S1 and S2 in ABBA and reverse BAAB process order,
   retaining stdout, stderr, command metadata, GPU state, source hash, and
   binary hash.
3. Capture paired Nsys traces with shorter runs and calculate kernel counts,
   summed duration, two-stream union, and cross-stream overlap.
4. Decide only whether integration is worth its ownership/lifetime complexity.
   Even a positive decision still requires a per-device/model-keyed shared
   owner, full B13/S2 Nsys, symmetric whole-network long tests, and 8,192-row
   replay before KataGo acceptance.
