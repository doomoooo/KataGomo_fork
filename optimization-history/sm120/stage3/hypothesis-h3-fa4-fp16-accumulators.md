# Hypothesis H3: FA4 FP16 accumulators on SM120

Created: 2026-08-05 (UTC), before the first successful FP16-accumulator
candidate build. Target regime is RTX 5090D, B13, two independent server
streams, `requireExactNNLen=true`, FP16 I/O, unlocked clocks.

## Evidence

The accepted Stage-1 FA4 kernel reduces attention from 49 us/block to about
19 us/block, but NCU still reports only 8.3% achieved occupancy, 83% cycles
with no eligible warp, and 37% of issue stalls on the MMA pipe. The kernel
uses FP32 QK and PV accumulators even though its inputs and output are FP16.

The installed flash-attn 4.0.0b25 SM120 path exposes independent
`qk_acc_dtype` and `pv_acc_dtype` parameters. The first both16 build attempts
showed two implementation gaps rather than a hardware limitation: typed tail
masking for QK and an FP32-to-FP16 conversion missing from PV rescaling.

## Mechanism and prediction

FP16 QK and/or PV MMA accumulation should reduce accumulator register pressure
and use the lower-precision MMA form. Four otherwise identical AOT objects
(`fp32`, `qk16`, `pv16`, and `both16`) will isolate each contribution.

Expected signal:

- attention kernel median falls from about 19 us/block to at most 17 us/block;
- B13/S2 whole-network throughput improves by at least 1% outside run-order
  drift;
- NCU shows the expected accumulator instruction/resource change rather than
  an unrelated launch or clock effect.

## Risks and predeclared accuracy gates

FP16 accumulation changes arithmetic and can amplify tail-mask, softmax, or
long reduction error. Every performance candidate must be compared directly
with the Stage-1 full-FP32 reference over all 8,192 fixed rows and all output
heads. It is rejected on NaN/Inf or if any of these gates fail:

| metric | gate |
|---|---:|
| policy top-1 agreement | >= 99.70% |
| optimistic-policy top-1 agreement | >= 99.60% |
| policy probability RMSE | <= 1.5e-4 |
| policy total variation | <= 2.5e-3 |
| policy Jensen-Shannon divergence | <= 8e-6 |
| max policy absolute error | <= 0.03 |
| weighted p0loss delta vs reference | <= 5e-4 |
| outcome RMSE | <= 1.5e-2 |
| score-mean RMSE | <= 1.0e-2 |
| ownership sigmoid RMSE | <= 4.0e-4 |

These gates are broader than the Stage-1 FP32-accumulator FA4 result but keep
the candidate in the same numerical quality class as the accepted FP16
backend. Passing them is necessary, not sufficient: a slower mode is rejected.

## Validation order

1. Generate and hash all four AOT objects with fixed toolchain and shape.
2. Run the standalone C++ smoke against `attention_ref` for every object.
3. Replay a small corpus for crash/NaN screening.
4. Run an equal-duration kernel microbenchmark and Nsys scan.
5. Run full 8,192-row all-head regression for modes with a performance signal.
6. Run ordered whole-network A/B (and reverse order if retained).
7. Use NCU on the surviving kernel and update `results/rebuild/HISTORY.md`.
