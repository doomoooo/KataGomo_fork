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
- CUDA toolkit 13.0.3 from fixed PyPI packages, including nvcc, CRT, CCCL and
  NVVM; the C++ backend and Python generators share this one installation;
- cuDNN 9.20.0 for CUDA 13 from the PyTorch-compatible PyPI package;
- a clean compatible FlashAttention source with the SM89 C++ and minimal SM120
  both16 patches applied before its wheel is built;
- the FlashAttention-declared CUTLASS submodule for SM89 and a clean compatible
  CUTLASS source for CuTe generation; resolved revisions are provenance only;
- published TileLang 0.1.13, Quack 0.6.4 and compatible TVM-FFI 0.1.12 wheels,
  plus KataGo's vendored cuDNN frontend headers;
- versioned wheels for build tools, PyTorch, CUTLASS DSL and small runtime
  dependencies, including PyTorch's Triton dependency. Triton is not a source
  component or a KataGo kernel generator;

The package manifest binds every carried payload by SHA-256.  Source builds
produce a second manifest containing source archive, patch, wheel and installed
module hashes.  No ambient Python package is accepted.

Default source-build concurrency is `min(nproc, memory_limit)`, where the
memory limit reserves 25% of current Linux/cgroup headroom and budgets 2 GiB
per compiler process. Explicit `--jobs` is authoritative. Historical SM120
FFN device sources are separately frozen and hash-verified, so upgrading the
main TileLang source revision cannot silently rewrite an accepted candidate.

## 2. Architecture selection and scan

Device selection is by CUDA ordinal after `CUDA_VISIBLE_DEVICES` remapping.
The CUDA Runtime product name is the plan registry's unique lookup key. After
lookup, the receiver's recorded capability and resource contract must match
the selected plan exactly. Only the historical CUDA ordinal is provenance.

Supported mappings:

| CUDA CC | workflow | default batches | streams |
| --- | --- | --- | --- |
| 8.9 | shared CUDA workflow, SM89 catalogs / 10 decision groups | optimized B4-B32 prescan, then top 3 | 2 |
| 12.0 | shared CUDA workflow, SM120 catalogs / 10 decision groups | optimized B4-B32 prescan, then top 3 | 2 |

The prescan uses an explicit, artifact-free stable optimized graph and ranks
physical `nnEval/s`; it is only a shape selector. All AOT candidates for the
selected three-batch domain are then generated and one fat search binary is
linked. Candidate measurements must only switch runtime tactics in that
binary. `--full-batch-scan` instead selects all 29 shapes without changing the
per-batch flow. Discovery uses 100 timed iterations, 50 warmup and one repeat
with a 0.1% acceptance threshold. The final joint gate uses at least 1000 timed
iterations, 50 warmup and two repeats with at most 10% relative spread.
Every measurement monitors external SM activity. Memory-only processes are
ignored. Non-zero foreign SM work discards only the affected measurement; the
workflow waits 30 seconds and retries instead of aborting the scan.

An implementation catalog is not an independent operator claim. The static
space contract groups catalogs by shared config ownership and declared runtime
dependencies. No config key or dependency may cross a decision-group boundary.
Within a group a historical bundle is measured first and later stages may only
refine explicitly declared keys; FA tile/accumulation keys are exclusively
owned by the FA catalog.

## 3. Plan contract

A release plan is data rather than compile-time state.  It binds at least:

- schema/workflow and exact batch/stream domain;
- source model and benchmark config hashes;
- CUDA Runtime device capabilities and CC;
- search-space, generator, source, patch, AOT manifest and binary hashes;
- every selected exact-batch component and its parameters;
- discovery decision chain and long-gate evidence.

Every scan starts from an explicit official-equivalent runtime baseline that
sets every architecture tactic key to `false`, `disabled`, or its neutral
scalar value. `keep-incumbent` therefore retains only earlier measured family
winners and never a parser default. Plan apply carries that full baseline plus
the selected changes and explicitly enables the chosen architecture backend;
`auto` and implicit B13 winners are not valid runtime states.

Unsupported components, incomplete batch coverage, binary mismatch or
resource incompatibility are hard errors.  Discovery output is never labeled
final.  `ready_for_scan_bypass` requires complete long-stable evidence;
`production_ready` additionally requires the immutable 8192-row FP32
certificate.  The latter must remain false when the golden is unavailable.

The build-only receiver path consumes one carried production plan. It
materializes the normal space for that exact receiver and batch, verifies every
selected candidate against it, then retains only selected candidates and their
recursive artifact dependencies before generation and compilation. It runs no
prescan, benchmark, refinement, long gate, or accuracy replay. A content-hashed
`plan-build.json` binds the locally compiled binary, artifact bundle, restricted
space, and source plan; the runtime launcher rejects an unbound rebuilt binary.

The release-qualification `reference` phase may create that golden only by
forcing `useFP16=false` and disabling both optimized architecture backends;
its sidecar binds the binary, model, corpus, command and output SHA-256. The
`accuracy` phase first requires complete long-gate coverage for the selected
domain (top 3 by default, B4-B32 in exhaustive mode), then
selects the batch with the highest stable physical `nnEval/s` and replays only
that accumulated override against the same file. It pads the physical tail
batch with repeated real inputs without serializing the padding, deletes the
candidate dump after comparison, and emits only that certified batch in
`best-tactic-plan.json`.
Golden reuse additionally requires the current model and corpus hashes. Each
comparison verifies KRNN exact-batch/tail-padding metadata and byte-identical
targets and inputs before computing output tolerances. Certification checks
both aggregate metrics and the GTP-shaped verifier's worst-per-request
max-absolute/per-head RMSE envelope. A single failing row is a hard accuracy
failure even when the corpus-wide RMSE passes.

The 8192-row input corpus is a maintained correctness fixture, not a moving
"latest training data" target. `corpus.lock.sh` fixes the official archive URL,
archive SHA-256, and sampled-corpus SHA-256. Reconstruction safely extracts that
archive and uniformly samples exactly 8192 full 19x19 rows with seed 20260803.
The sampled NPZ and full provenance manifest are carried in the tar and
revalidated by `setup.sh`. Changing the fixture requires intentionally
regenerating the FP32 reference and certified plans together. Candidate
inference output is never accepted as the FP32 reference.
