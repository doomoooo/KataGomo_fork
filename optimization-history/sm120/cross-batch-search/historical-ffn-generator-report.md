# SM120 historical tanh-half2 FFN generator report

Date: 2026-08-06 UTC

## Outcome

A reproducible, source-only generator now covers every exact B1 through B32
shape for the fixed 19x19 board.  It is isolated under
`katago/python/sm120_historical_ffn/` and does not edit the current tactic
runner, AOT registry, active slot, backend source, or configuration.

The generated candidate is exactly
`ffn-m128-n64-k32-s2-mb3-tanh-half2`: M128, N64, K32, two pipeline stages,
128 threads, launch bound min-blocks 3, FP16 MMA accumulation, and the packed
half2 tanh epilogue.  Constants and candidate identity are frozen in
[generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:32).

This is not the generic exponential SwiGLU path.  The frozen upstream schedule
calls `tilelang_h2tanh_approx`, forms the sigmoid with half2 FMA, then performs
the two half2 multiplies in the historical order
([onnx_kernels.py](/workspace/katago/python/sm120_historical_ffn/upstream/onnx_kernels.py:1606)).
The generator requires those markers and rejects `expf`, `T.exp`, or the
division-form exponential implementation
([generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:202)).

## Frozen provenance

The three requested historical files are vendored byte-for-byte under
`upstream/`:

| Historical file | SHA256 |
|---|---|
| `export_ffn_aot_sm120.py` | `f0d3bfe96d263d9a8e9b0cc756830928c6d39e8085f901f88b482c0884ce1466` |
| `onnx_kernels.py` | `6386d94cf2ded04728bb8139c1f35339e9c5969981e786b7d3e2b8685c29ba48` |
| `tune_ffn_swiglu.py` | `56bef6f7dd36b2f3921e9cc25e067df1edfbbb6dce2e14843bf60487d7c35901` |

The complete provenance record is in
[manifest.json](/workspace/katago/python/sm120_historical_ffn/manifest.json:1).
It freezes TileLang 0.1.13 revision `8001cc4ccf6149382d2019654a19f59c1d4d0482`,
Git tree `49f50071da5a0a0dd46296965847446d9ad727cf`, installed-package tree
SHA256 `f20f03fdf1f9924f6d1540aac7202035be8ff2839f3e3d72b28783bd11e879e0`,
the complete historical pip-freeze hash, CUDA 13.2.78, and the historical nvcc
binary hash.  The exact package list is preserved in
[requirements-historical.lock.txt](/workspace/katago/python/sm120_historical_ffn/requirements-historical.lock.txt:1).

Generation refuses a source-snapshot mismatch or TileLang installation-tree
mismatch before lowering
([generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:100),
[generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:111)).
The current local TileLang tree has the same frozen tree hash.  The current
Python/numpy environment differs from the historical full lock, but B19 still
reproduces the historical device source byte-for-byte; every artifact records
the actual code-generation dependency versions.

## Device-free lowering

The original historical script used `execution_backend="cython"`.  That path
generates CUDA source and then initializes a runtime adapter, whose
`cudaFuncSetAttribute` call probes a CUDA device.  The new path temporarily
recovers the same historical PrimFunc and invokes TileLang's own device-source
lowering with host codegen and device compilation disabled
([generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:257)).
`CUDA_VISIBLE_DEVICES` is cleared before importing TileLang
([generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:24)).

This pure-lowering path generated the renamed B19 device source as exactly
15,496 bytes with SHA256
`905d8068911c4c1e08408ebf4c8968dd4c6e5e97189abb2057c2aec2669aaa81`,
matching the historical source prefix.  This golden comparison is mandatory on
every B19 generation.

## ABI and exact shapes

Each generated source exports the existing search-slot ABI:

```cpp
extern "C" int sm120_search_ffn_batch();
extern "C" const char* sm120_search_ffn_id();
extern "C" cudaError_t sm120_search_ffn_launch(
  const half* input,
  const half* linear_weights,
  const half* gate_weights,
  half* output,
  cudaStream_t stream);
```

The wrapper is defined at
[generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:59).
The public ABI is `(input, linear, gate, output, stream)`, while TileLang emits
the kernel arguments as `(input, gate, linear, output)`; the launcher performs
that intentional reorder.  Tensor layouts are contiguous row-major FP16:

- input: `[batch*361, 384]`;
- linear and gate weights: `[384, 1152]`;
- output: `[batch*361, 1152]`.

The exact launch is `grid=(18, ceil(batch*361/128), 1)`,
`block=(128,1,1)`, and 32,768 dynamic shared bytes.  Every source has identical
search-slot symbols, so only one exact-batch TU may be linked into the active
slot at once.  Its CUDA kernel symbol is batch-qualified.  A compile-time SM120
guard permits inclusion in KataGo's multi-architecture fat binary
([generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:182)).

## Commands

Generate and statically verify all batches:

```bash
/workspace/venv/bin/python \
  /workspace/katago/python/sm120_historical_ffn/generate.py \
  --all-batches \
  --space /workspace/results/rebuild/cross-batch-search/space-5090d-b1-32-s2.json \
  --output-dir /workspace/results/rebuild/cross-batch-search/historical-ffn-b1-b32

/workspace/venv/bin/python \
  /workspace/katago/python/sm120_historical_ffn/verify.py \
  --artifact-root /workspace/results/rebuild/cross-batch-search/historical-ffn-b1-b32
```

Generate one candidate directly to the existing stable active-slot path:

```bash
/workspace/venv/bin/python \
  /workspace/katago/python/sm120_historical_ffn/generate.py \
  --batch 13 \
  --space /workspace/results/rebuild/cross-batch-search/space-5090d-b1-32-s2.json \
  --candidate-id ffn-m128-n64-k32-s2-mb3-tanh-half2 \
  --output-dir RESULTS_FOR_B13 \
  --source-path ACTIVE_FFN_SOURCE.cu
```

This command has no `--device` argument by design.

## Static self-test evidence

The retained self-test artifacts are at
[historical-ffn-static-selftest](/workspace/results/rebuild/cross-batch-search/historical-ffn-static-selftest/manifest-b1-b32.json:1),
and all 32 ABI/shape/arithmetic checks passed in
[static-verification-b1-b32.json](/workspace/results/rebuild/cross-batch-search/historical-ffn-static-selftest/static-verification-b1-b32.json:1).
The machine-readable overall result is
[selftest-summary.json](/workspace/results/rebuild/cross-batch-search/historical-ffn-static-selftest/selftest-summary.json:1).

Representative stable source hashes are:

| Batch | Source SHA256 |
|---:|---|
| B1 | `65d69f8122d0f9ad9010ba093b742eb773b131c577ba7eba14b8d2defdd6ed73` |
| B13 | `7f41a9ea0a7d6ce642e1645e453d196f52ce7adec2526b612031a017e6e479d2` |
| B19 | `749044844add83a62349050a31f16129e2b615d5a5e4c08f1fc1d234b939f300` |
| B32 | `58cb5c176845e1fe2d9cfe808f2f487193cc757b3fd9bc7d1fdefc6d01dd9bea` |

A second independent B1-B32 generation produced identical hashes for 32/32
sources.  B1, B13, B19, and B32 sources also compiled successfully with local
CUDA 13.2.86 as a fat object containing both `sm_89` and `sm_120`.  No GPU was
selected or executed during generation, verification, or nvcc compilation.

## Runner integration recommendation

When the tactic runner sees FFN candidate implementation
`historical_tilelang` (or this exact candidate ID in an older schema-2 space),
dispatch to this generator instead of `sm120_generate_tilelang_aot.py`.  Pass
`--batch`, `--space`, `--candidate-id`, `--output-dir`, and the current
`--source-path`; do not pass a CUDA device.  Keep the existing build and stable
search registry ABI unchanged.

Do not link all 32 sources together because their three public search symbols
are intentionally identical.  Materialize all 32 for caching, then copy or
regenerate only the selected batch into the single active FFN slot before each
build.

After integration, static generation is only the first gate.  Each batch still
requires:

1. compile/smoke and full-output accuracy against the fixed FP32 reference;
2. optional S1/NCU mechanism evidence;
3. acceptance or rejection from natural whole-graph S2 throughput;
4. no homogeneous/mixed local S2 proxy gate.

The generator metadata explicitly records this acceptance contract
([generate.py](/workspace/katago/python/sm120_historical_ffn/generate.py:370)).
