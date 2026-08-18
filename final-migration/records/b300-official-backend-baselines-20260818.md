# NVIDIA B300 b11 official-backend baselines (2026-08-18)

This record establishes the pre-SM103-optimization controls for the
`b11c768h12nbt3tflrs-fson-silu` network on one NVIDIA B300 SXM6 AC.  The
production target remains KataGo's CUDA backend. TensorRT is a comparison
backend only.

## Fixed target and measurement contract

- GPU: NVIDIA B300 SXM6 AC, compute capability 10.3 (`sm_103`), 148 SMs,
  275040 MiB, driver 610.57.04.
- Model SHA-256:
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- Shape: exact 19x19, FP16 trunk, NHWC transformer, two inference lanes on
  CUDA device 0 (`S2`).
- `B` always means the physical batch **per lane**. Total concurrent physical
  work is `2 * B`.
- Metric: `benchmarknn` device-forward `combinedNNEvalsPerSec`; feature
  generation, H2D/D2H, output decoding, and search are excluded.
- Every subprocess was monitored with `nvidia-smi pmon`. All retained result
  files report no foreign active-SM PID, monitor error, or conflict retry.
- Short CUDA scan: two samples, 200 timed iterations and 80 warmups.
- CUDA confirmation: two samples, 1000 timed iterations and 80 warmups.
- TensorRT discovery: three samples, 500 timed iterations and 20 warmups.
- TensorRT confirmation: five samples, 1000 timed iterations and 20 warmups.

The managed stack used CPython 3.14.7, CUDA toolkit 13.3.1 (nvcc 13.3.73),
cuDNN 9.25.0.15, PyTorch 2.13.0+cu132, TileLang 0.1.13, and CUTLASS DSL
4.7.0. PyTorch's wheel reports CUDA 13.2 while loading the qualified cuDNN
9.25 DSO from the managed prefix.

## Official CUDA fallback

The dedicated binary was compiled for `sm_103`; all SM89/SM120 custom
backends were explicitly disabled in every benchmark override.

- Binary:
  `.final-migration-env/katago-builds/cuda-official-sm103/katago`
- SHA-256:
  `e820f3070969e7027ab49e0106546d30a714aa0f00ef229ad79872cbf74f357e`

The standard B4-B32 interval still rose into its right boundary:

| B | nnEval/s | B | nnEval/s |
| ---: | ---: | ---: | ---: |
| 4 | 2282.2 | 20 | 4676.5 |
| 8 | 3401.4 | 24 | 4770.6 |
| 12 | 4156.0 | 28 | 4996.0 |
| 16 | 4320.5 | 30 | 5118.8 |
| 19 | 4626.3 | 32 | 5106.5 |

An extended scan found a higher, shape-dependent plateau. Representative
short-scan values were B56=5137.7, B64=5148.4, B68=5237.6, B80=5086.2,
B96=5236.3, B100=5227.5, and B128=5187.2 nnEval/s. Exact shapes exhibit
discrete cuDNN tactic changes rather than a smooth batch curve.

Long confirmation:

| B | samples (nnEval/s) | median | relative spread |
| ---: | --- | ---: | ---: |
| **68** | 5268.152 / 5273.318 | **5270.735** | 0.098% |
| 96 | 5206.905 / 5232.446 | 5219.676 | 0.489% |
| 100 | 5227.497 / 5171.221 | 5199.359 | 1.082% |

The selected measured CUDA control is therefore **B68 / S2**, 5270.735
physical nnEval/s. Its effective two-lane batch wall time is about 25.80 ms.

Raw bundles:

- `.final-migration-env/baselines/b300-20260818/cuda-scan/result.json`
- `.final-migration-env/baselines/b300-20260818/cuda-extended-coarse/result.json`
- `.final-migration-env/baselines/b300-20260818/cuda-extended-refine/result.json`
- `.final-migration-env/baselines/b300-20260818/cuda-extended-confirm-final/result.json`

## Upstream-compatible TensorRT 10.16.1.11

This is the official comparison. It uses the user-supplied CUDA-13.2
TensorRT tar with the upstream weakly typed `kFP16` and per-layer FP32
precision-constraint path. TensorRT documents this package as CUDA-13.x
compatible; it ran against the managed CUDA 13.3 runtime.

- Source tar SHA-256:
  `08334948c57c3bcf2de171ef4bd53d15f48a61a942b048abea12e32f892284e8`.
- Binary:
  `.final-migration-env/katago-builds/tensorrt-10.16.1.11-sm103/katago`
- Binary SHA-256:
  `883024dc8bbc02e7f6b05b0431034652931acc760b76e7fd455dc996af278612`.
- C++ protobuf/protoc: 35.1.

Only wave-model peak candidates were built and measured:

| B | discovery median nnEval/s | effective S2 batch ms |
| ---: | ---: | ---: |
| 23 | 6505.5 | 7.07 |
| 24 | 6668.5 | 7.20 |
| 29 | 6716.2 | 8.64 |
| 34 | 6721.6 | 10.12 |
| 40 | 6657.8 | 12.02 |
| 46 | 6725.6 | 13.68 |
| 52 | 6737.7 | 15.44 |

Long confirmation of the main plateau:

| B | five-sample median nnEval/s | relative spread | effective S2 batch ms |
| ---: | ---: | ---: | ---: |
| **29** | **6733.719** | 0.442% | **8.61** |
| 34 | 6722.285 | 1.386% | 10.12 |
| 46 | 6738.884 | 1.153% | 13.65 |
| 52 | 6739.509 | 0.050% | 15.43 |

B52 was the numerical maximum, but exceeded B29 by only 0.086% while using a
79% larger per-lane batch. Following the repository's established policy of
choosing the smaller batch when throughput is effectively tied, the official
TensorRT control is **B29 / S2**, 6733.719 physical nnEval/s. B24 is another
latency option, but its discovery median was 0.97% below the confirmed B29
median.

At the selected points TensorRT 10 is 27.76% faster than the official CUDA
fallback. This is a backend control, not an expected SM103 optimization gain.

Raw bundles:

- `.final-migration-env/baselines/b300-20260818/trt10-peak-scan/result.json`
- `.final-migration-env/baselines/b300-20260818/trt10-peak-confirm/result.json`
- `.final-migration-env/baselines/b300-20260818/trt10-left-peaks/result.json`

## Experimental TensorRT 11.2.1.2 port

Upstream KataGo master does not currently compile against TensorRT 11; its
strongly typed support is still represented by an unmerged upstream pull
request. This branch independently adds explicit ONNX FP16 trunk casts and
FP32 RMSNorm/head boundaries. Therefore these numbers are retained as
experimental port evidence and are **not** the official upstream comparison.

- Binary SHA-256:
  `51d1cae2a64aa1f432c60310246d547f5af4cf43982dd1afeed79a936cccde6e`.
- Mixed graph parse evidence: 535 FP16 casts, 67 FP32 casts, 66 internal
  RMSNorm boundaries, and 3356 parsed TensorRT layers.
- Best confirmed point: B52 / S2, 6674.378 nnEval/s (five 1000-iteration
  samples, 1.795% spread).
- It was 0.88% slower than the selected TensorRT 10 B29 control.

## Predicting useful batch candidates

For the dominant FFN projection, let `R = B * 361`, with `N=1152`. A tactic
using `128x128` output tiles has approximately

```text
q = ceil(R / 128)
CTA(B) = q * ceil(1152 / 128) = 9q
```

On 148 SMs, useful candidates occur just before `CTA(B)` crosses another
multiple of 148. This predicts B17, B23, B29, B34, B40, B46, B52, B58, B63,
and so on. The observed TensorRT 11 curve strongly supports this explanation:
B34 reached 6459.0 nnEval/s, while B35 immediately fell to 6259.7 after
crossing a wave boundary. The model is a candidate generator, not a substitute
for measurement, because TensorRT can switch tactics and the whole graph also
contains other GEMM widths, attention, convolution, and different occupancy.

## Remaining qualification boundary

These are throughput baselines. The binaries passed compilation, dynamic-link,
ONNX parse, engine-build, and real b11 inference smokes. An immutable 8192-row
FP32 replay comparison is still required before treating the experimental
TensorRT 11 port or any future SM103 CUDA plan as production-correct.
