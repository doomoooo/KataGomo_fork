# Stage 3: FA4 FP16 QK/PV accumulators on RTX 5090D

Status: accepted (2026-08-05 UTC). Scope is only the fixed 19x19 attention
shape: runtime batch B1-B13, S361, H12, D32, FP16, non-causal. All other
shapes and precisions retain the official backend fallback.

## Regime and artifacts

- GPU: RTX 5090D, CUDA runtime device 2, SM120, driver 595.80.
- Toolchain: CUDA 13.2.86, flash-attn 4.0.0b25, cutlass-dsl 4.7.0,
  quack-kernels 0.5.3.
- Network: `b11c768h12nbt3tflrs-fson-silu.bin.gz`, SHA256
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- AOT object: `fa4_sm120_b13.o`, SHA256
  `9b0c9b33c7bee1e61350cb0c3f422a58a5aaf2b20c2b426e638c0253f84398ac`.
- AOT cubin: SHA256
  `f02e5ea1e383b4e629c03a1e5b8397173241ef35e95cc8487cd745c00a353a87`.
- Performance regime: B13 per server, two independent CUDA streams, FP16 I/O,
  `requireExactNNLen=true`, unlocked clocks, per-GPU lock.
- Accuracy corpus: 8,192 fixed 19x19 rows, SHA256
  `0b2f2838df51ff98847f5bf595f9670350e993c5e178a92855c21e80e75762c5`.
- Full-FP32 reference: SHA256
  `5472b4a986726d6675b0f8746fa3cef48d230e3f6167f24143e3bb54948bd3bf`.

The predeclared hypothesis and gates are in
`hypothesis-h3-fa4-fp16-accumulators.md`. The AOT matrix, smoke, microbench,
accuracy, Nsys, NCU, and ABBA commands are retained as scripts in this
directory.

## Implementation

`build_aot.py` now supports independent FP16/FP32 QK and PV accumulators and
defaults to the accepted both16 object. flash-attn 4.0.0b25 omitted the
FP32-to-FP16 conversion when rescaling an FP16 PV accumulator, so the build
script applies a local typed store at that boundary. PTX confirms the accepted
object uses only
`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` for QK and PV.

The newer DSL AOT runtime keeps CUDA library/kernel handles in generated
process-global state. `replaynn` exposed a lifecycle bug when its warmup models
unloaded those handles before constructing the real replay models. The module
is now loaded once with `std::call_once` and retained for process lifetime.
This changes resource lifetime only, not arithmetic.

## Isolated results

All four objects passed the standalone 19x19 attention smoke without NaN/Inf.

| mode | median kernel time (us) | delta vs FP32 | max abs error vs `attention_ref` |
|---|---:|---:|---:|
| FP32 accumulators | 16.3599 | control | 1.22e-4 |
| QK16 | 13.4522 | -17.8% | 1.22e-4 |
| PV16 | 12.7965 | -21.8% | 2.44e-4 |
| both16 | 11.8000 | -27.9% | 2.44e-4 |

The QK16 and PV16 ablations show that both accumulator changes contribute;
both16 is not an un-attributed bundle.

## Nsys and NCU

Nsys under the real B13/S2 topology reports the fixed-B13 FA4 launches:

| metric | FP32 accumulators | both16 |
|---|---:|---:|
| average FA4 duration | 22.387 us | 17.946 us |
| registers/thread | 255 | 247 |
| streams observed (including warmup instances) | 4 | 4 |

This is a 19.8% attention-kernel reduction under concurrent execution.

Same-shape standalone NCU confirms the mechanism:

| metric | FP32 accumulators | both16 |
|---|---:|---:|
| duration | 22.34 us | 17.02 us |
| registers/thread | 255 | 247 |
| no eligible warp | 80.11% | 69.63% |
| issued warp/scheduler | 0.20 | 0.30 |
| issue slots busy | 17.09% | 25.91% |
| achieved occupancy | 14.73% | 14.24% |

The speedup comes from lower-latency FP16 accumulator MMA and improved issue
availability. Register use falls, but the two-block register limit and
theoretical occupancy do not change; an occupancy increase is not claimed.

## Full accuracy regression

Every mode was compared directly with the full-FP32 reference. The accepted
both16 result passes every predeclared gate:

| metric | FP32-accum FA4 | QK16 | PV16 | both16 |
|---|---:|---:|---:|---:|
| policy top-1 | 99.7681% | 99.7437% | 99.7559% | 99.7437% |
| optimistic top-1 | 99.7070% | 99.6582% | 99.6948% | 99.7314% |
| policy probability RMSE | 9.857e-5 | 9.928e-5 | 1.001e-4 | 1.031e-4 |
| total variation | 1.633e-3 | 1.651e-3 | 1.672e-3 | 1.707e-3 |
| JSD | 3.379e-6 | 3.469e-6 | 3.545e-6 | 3.718e-6 |
| weighted p0loss abs delta | 6.080e-5 | 5.269e-5 | 1.056e-4 | 4.172e-5 |
| outcome RMSE | 2.367e-3 | 2.350e-3 | 2.271e-3 | 2.287e-3 |
| score mean RMSE | 2.012e-3 | 2.186e-3 | 1.899e-3 | 1.996e-3 |
| ownership sigmoid RMSE | 2.404e-4 | 2.449e-4 | 2.422e-4 | 2.464e-4 |

Raw comparisons are in `accuracy/*-vs-fp32.json`.

## Whole-network A/B

After one 1,500-iteration thermal precondition, the ordered sequence was
`A-B-B-A / B-A-A-B`, where A is the same-toolchain FP32-accumulator object and
B is both16. Each measured run used 1,000 iterations.

| mode | values (nnEval/s) | mean | median | range |
|---|---|---:|---:|---:|
| FP32 accumulators | 3136.572 / 3118.673 / 3112.490 / 3104.291 | 3118.006 | 3115.581 | 32.281 |
| both16 | 3146.603 / 3136.521 / 3128.586 / 3115.910 | 3131.905 | 3132.553 | 30.693 |

Mean improvement is 0.446%; median improvement is 0.545%. The platform drifted
down monotonically, but both symmetric four-run blocks independently give
about the same 0.446% paired benefit. The whole-network result is below the
pre-experiment 1% prediction, so the prediction is recorded as too optimistic;
the direction remains supported by the paired order and profiler evidence.

## Decision

Accept both16 for the fixed 19x19 FA4 path. It passes full accuracy, has
independent QK/PV ablations, reduces the kernel under both isolated and
two-stream execution, and gives a repeatable whole-network improvement.
QK16 and PV16 remain recorded ablations but are not retained because both16 is
faster and passes the same gates.

The next single-variable candidate is a fixed-S361 tail-mask specialization:
only the final K tile has 105 valid columns, and no dynamic board-size support
is required in the FA4 path.
