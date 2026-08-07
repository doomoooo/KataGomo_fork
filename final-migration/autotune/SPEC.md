# Autotune SDK specification

## 1. Environment artifact

The outer tar is relocatable and has one top-level directory.  Its payload is
complete before release: target setup is offline by default and rejects a
missing file or checksum mismatch.  All mutable state lives in `runtime/`,
`build/`, and `results/` below the chosen prefix.

The non-invasive host ABI is Linux x86-64/glibc >= 2.28. The setup script
checks, but does not overwrite, the baseline OS compiler and shell utilities.
It never assumes Ubuntu 24.04 and is qualified on Ubuntu 22.04 and 24.04.

Common locked environment:

- CPython 3.12.13 from python-build-standalone release 20260807;
- CUDA toolkit 12.8 (new enough for SM120 and supported by both validation
  drivers);
- cuDNN 9.25.0 for CUDA 12;
- FlashAttention `69e1bcbe`, with the SM89 C++ and minimal SM120 both16
  patches applied before its wheel is built;
- CUTLASS submodule `71275920` for SM89 FlashAttention and the independently
  pinned latest CUTLASS source for CuTe generation;
- exact revisions in `source-lock.tsv` for Triton, TileLang, Quack, TVM-FFI
  and cuDNN frontend;
- pinned wheels for build tools, PyTorch, CUTLASS DSL and small runtime
  dependencies.

The package manifest binds every carried payload by SHA-256.  Source builds
produce a second manifest containing source archive, patch, wheel and installed
module hashes.  No ambient Python package is accepted.

## 2. Architecture selection and scan

Device selection is by CUDA ordinal after `CUDA_VISIBLE_DEVICES` remapping.
The CUDA Runtime supplies compute capability and resource properties.  Product
names, SM counts and historical ordinals are provenance only.

Supported mappings:

| CUDA CC | workflow | default batches | streams |
| --- | --- | --- | --- |
| 8.9 | portable SM89, 20 families | 4-32 | 2 |
| 12.0 | SM120 coordinate, 5 families | 4-32 | 2 |

All AOT candidates for the selected batch domain are generated first and one
fat search binary is linked.  Candidate measurements must only switch runtime
tactics in that binary.  Discovery uses 100 timed iterations, 50 warmup and
one repeat with a 0.1% acceptance threshold.  The final joint gate uses at
least 1000 timed iterations, 50 warmup and two repeats with at most 10%
relative spread.

## 3. Plan contract

A release plan is data rather than compile-time state.  It binds at least:

- schema/workflow and exact batch/stream domain;
- source model and benchmark config hashes;
- CUDA Runtime device capabilities and CC;
- search-space, generator, source, patch, AOT manifest and binary hashes;
- every selected exact-batch component and its parameters;
- discovery decision chain and long-gate evidence.

Unsupported components, incomplete batch coverage, binary mismatch or
resource incompatibility are hard errors.  Discovery output is never labeled
final.  `ready_for_scan_bypass` requires complete long-stable evidence;
`production_ready` additionally requires the immutable 8192-row FP32
certificate.  The latter must remain false when the golden is unavailable.
