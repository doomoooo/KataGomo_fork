# Stage 30 hypothesis: exact B13 fused policy P1

Date: 2026-08-06

## Frozen target

- GPU: RTX 5090 D, device 2
- Shape: batch 13, full 19x19 board, 361 spatial rows
- Topology: two NN servers / two CUDA streams
- Precision and layout: FP16 trunk/projection, FP32 policy P1 post-processing,
  NHWC
- Control: current accepted target configuration with
  `cudaUseFusedPolicyP1=false`
- Candidate: identical configuration with only
  `cudaUseFusedPolicyP1=true`
- Explicitly disabled in both arms: `cudaUseWideHeadProjection=false` and
  `cudaUseHeadBNHalfToFloat=false`

## Evidence

Stage27 S2 attribution places the policy P1 pointwise boundary at ordinals
324-326:

| Ordinal | Operation | Isolated median | S2 median | S2 excess / 60 |
| ---: | --- | ---: | ---: | ---: |
| 324 | half-to-float | 1.536 us | 1.760 us | 0.014944 ms |
| 325 | per-batch global-bias add | 1.824 us | 2.064 us | 0.016577 ms |
| 326 | FP32 affine + SiLU | 2.208 us | 2.464 us | 0.016257 ms |
| total | complete boundary | 5.568 us | 6.288 us | 0.047778 ms |

The actual aggregate S2 boundary is 0.381858 ms / 60 calls, or 6.364 us per
forward. It writes a 1,802,112-byte FP32 intermediate before two more
pointwise passes.

The exact B13 implementation on RTX 4090 reduced the corresponding local
boundary by about 63%, to 3.20-3.23 us with no spills. Its short S2 ABBA gain
was only +0.074%, with 3/4 positive adjacent pairs and a slightly negative
reverse aggregate, so the local mechanism is established but the whole-network
effect is near the scheduling-noise floor. The 5080 B19 implementation showed
+0.191% forward and +0.169% reverse and was accepted.

## Falsifiable mechanism

Fuse these operations without changing their order:

```text
float value = half_to_float(p1)
value = value + global_bias
value = value * scale + bias
out = value / (1 + exp(-value))
```

The kernel maps one CTA to five spatial rows and all 96 channels:

```text
grid=(ceil(361/5), 13), block=(96, 5)
```

It should replace three launches with one and eliminate the FP32 intermediate
round trip. The primary expected profiler change is 324-326 becoming one
kernel below about 3.5 us with no spill. Expected whole-network S2 gain is
+0.03% to +0.15%.

## Exact dispatch and fallback

The candidate may run only when all of these hold:

- option enabled explicitly;
- SM120 model active;
- batch size exactly 13;
- `nnXLen == nnYLen == 19` and `xySize == 361`;
- FP16 input, FP32 output, NHWC;
- p1 channels exactly 96;
- full-board path with no runtime board mask.

Every failed gate executes the complete official three-operation path. The
option defaults to false. Wide-head stride and BN-to-float are outside this
experiment.

## Validation gates

1. Build and smoke both control and candidate.
2. Exercise a non-B13 case to prove official fallback.
3. Thermally prime, then run short S2 A/B/B/A with 400 iterations per arm.
4. Continue only if adjacent-pair direction is positive and larger than the
   observed short-run noise.
5. On a positive screen, use Nsys to verify two launches removed and inspect
   the complete boundary/dual-stream union; use NCU to record duration,
   registers, occupancy and spills.
6. Only then run longer forward and reverse ABBA plus the fixed 8192-position
   full-output replay against the accepted CUDA output and FP32 reference.

Reject immediately if the local boundary is not clearly faster, if S2 direction
depends on order, or if exact arithmetic/shape semantics cannot retain the
official fallback.
