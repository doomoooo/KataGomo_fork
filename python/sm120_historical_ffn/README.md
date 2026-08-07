# Historical SM120 tanh-half2 FFN generator

This directory preserves the accepted historical TileLang FFN code-generation
path and extends its fixed shape from B19 to every exact batch B1 through B32 on
a fixed 19x19 board.  It is isolated from the current general-purpose TileLang
generator because the two epilogues are not interchangeable:

- historical candidate: FP16 MMA plus packed-half2 `h2tanh_approx` sigmoid;
- current generic candidate: scalar/exponential SiLU.

`generate.py` fails if the generated CUDA contains `expf`, loses the half2 tanh
intrinsic, or fails to reproduce the frozen B19 device source byte for byte.  It
also hashes all installed TileLang package files before generation.  CUDA device
visibility is cleared before TileLang is imported; this path performs codegen
only and never selects or runs a GPU.  In particular, it recovers the historical
PrimFunc and calls TileLang's device-source lowering directly, bypassing the
historical Cython runtime adapter whose initialization probes a CUDA device.

## Generate

Use the code-generation environment that contains the frozen TileLang 0.1.13
package tree:

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

For one active search slot, pass `--batch B --output-dir DIR` and optionally
`--source-path PATH`.  The latter is intended for the CMake active-slot source
path, but this generator never chooses or edits that path on its own.

## ABI

Each exact-batch source exports the existing search-slot ABI:

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

The launcher maps the public `(linear, gate)` order to TileLang's emitted
`(input, gate, linear, output)` order.  All tensors are contiguous row-major
FP16 with `M=batch*361`, `K=384`, and `N=1152`.  The launch is
`grid=(18, ceil(M/128), 1)`, `block=(128,1,1)`, and 32768 dynamic shared bytes.

Every source defines the same three search-slot symbols, so only one generated
translation unit may be active in a binary.  The CUDA kernel symbol itself is
batch-qualified.  Non-SM120 device bodies are compiled empty so the source can
coexist in KataGo's multi-architecture fat binary.

## Acceptance

Static generation only proves provenance, ABI, exact shape bounds, and the
historical arithmetic path.  It does not accept a tactic.  Each batch still
requires full-output accuracy and natural whole-graph S2 A/B measurement; local
homogeneous or mixed two-kernel proxies are not acceptance metrics.
