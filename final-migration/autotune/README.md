# Unified SM89 / SM120 autotune SDK

This directory defines the source-based, non-invasive autotune distribution.
It is separate from `environment/package-distribution.sh`, which packages an
already-built inference runtime.

The release artifact is one uncompressed outer `.tar`. Release construction
uses the maintained training archive and corpus identities in `corpus.lock.sh`.
The current gate contains exactly 8192 deterministically sampled full 19x19
rows with seed 20260803. The corpus changes only when maintainers intentionally
regenerate the FP32 reference and certified plans together. The tar carries the
corpus and its complete source/row manifest. It also carries a pinned
Python runtime, fixed CUDA 13.0.3 and cuDNN 9.20 PyPI wheels, the complete
KataGo source tree, materialized third-party source trees, build-prerequisite
payloads, the model, and integrity manifests. The CUDA compiler and C++ backend
use the same NVIDIA wheel tree as PyTorch, so the release does not carry a
second CUDA toolkit. The target does not clone GitHub repositories or resolve
dependency versions.

After extracting the release in a writable persistent directory:

```bash
./setup.sh
./build-for-plan.sh --device 0
./run.sh
```

This is the build-only path for the certified plan carried by the tar.
`build-for-plan.sh` looks up the plan by CUDA product name, validates its
receiver and model contract, generates only the selected plan's single-batch
tactics and recursive artifact dependencies, and compiles KataGo. It performs no
prescan, candidate benchmark, refinement, long gate, or accuracy replay. The
content-hashed `plan-build.json` binds the rebuilt binary and generated
artifacts to the certified plan.

To tune a new plan for the receiver instead:

```bash
./setup.sh
./run-autotune.sh --device 0
./run.sh
```

`run.sh` selects the certified plan for the requested CUDA Runtime device,
verifies the plan/model/measured-binary identity, and starts `katago gtp` with
the exact batch, two inference lanes, batch-aware dispatch, asynchronous event
pipeline, and the default search-thread budget. It does not reuse scan-host
paths from the plan.

The host baseline is Linux x86-64 with glibc 2.28 or newer, an NVIDIA driver
compatible with CUDA 13.0, and the small OS bootstrap set checked by `setup.sh`
(`bash`, GNU tar/coreutils, and GCC/G++). Everything above that
bootstrap is carried in the tar; setup performs no APT transaction, Git clone,
or network access. Setup validates the carried corpus before building. The
same `prepare_accuracy_corpus.py` path can reconstruct a missing pair from the
locked official archive and rejects a different archive or corpus hash. It
never changes the correctness gate merely because newer training data exists.
This deliberately
supports both validated Ubuntu 22.04 and
24.04 hosts instead of encoding one Ubuntu release.

`setup.sh` writes only below the extracted directory unless `--prefix` is
given. It installs the carried published TVM-FFI, TileLang, Quack and CUTLASS
DSL wheels, then builds only the locally patched FlashAttention package from
carried source. PyTorch's Triton wheel dependency is carried
as a binary package only; this workflow does not generate or benchmark Triton
kernels. CUDA, nvcc and cuDNN are fixed NVIDIA PyPI packages shared by Python
code generation and the C++ backend.

Build parallelism defaults to the lower of `nproc`, 8, and a memory-aware limit
(75% of current `MemAvailable`/cgroup headroom at 2 GiB per heavy compiler
process). This avoids fixed `-j4`/`-j8` values while protecting memory-limited
hosts. `--jobs N` remains an explicit override.

`run-autotune.sh` queries the selected device through the CUDA Runtime.  CC
8.9 dispatches the SM89 workflow and CC 12.0 dispatches the SM120 workflow.
The selection domain is exact B4-B32 with two inference streams. By default an
artifact-free stable optimized graph first measures all 29 batches, and only its
three highest-throughput shapes receive complete tactic generation,
batch-outer discovery, and the 1000-iteration/two-repeat long gate. Use
`--full-batch-scan` to optimize every B4-B32 shape; exhaustive mode is
supported but default-off.

The 19 backend implementation catalogs are organized into 10 decision groups
on both architectures. Shared runtime keys and declarative dependencies may not
cross a group boundary. Discovery is short; a plan is only marked
scan-bypass-ready after every selected batch passes the long gate. If the tar
carries the immutable full-FP32 golden, `all` selects the highest-throughput
long-gate batch, replays that one plan over the 8,192-row corpus, and emits a
single-batch `best-tactic-plan.json` with `production_ready=true`. Replay pads only
the physical tail batch by repeating real rows and serializes exactly 8,192
rows, so an exact-batch AOT route never escapes the accuracy gate through a
short-tail fallback. Candidate `.krnn` dumps are deleted immediately after
comparison; the reference and the one selected report are retained.
The comparator also requires exact-batch/tail-padding metadata and
byte-identical target/input sections; a golden from a different model or
corpus is rejected rather than relabeled. It applies the ordinary GTP
verifier's worst-per-request max-absolute and per-head RMSE limits as well as
the aggregate 8,192-row metrics, so a single bad position cannot disappear in
an average.

Release qualification can create the reference explicitly, through the
official CUDA FP32 path with both optimized backends disabled:

```bash
./run-autotune.sh --device 0 --phase reference
./run-autotune.sh --device 0 --phase accuracy
```

The first command records the binary/model/corpus hashes and the exact
disabled-backend overrides next to the golden. It never treats a candidate
backend's output as its own expected result. If the tar has no reference,
`all` leaves `production_ready=false` and prints that accuracy was skipped.

The accepted historical SM120 tanh-half2 FFN is preserved as hash-addressed
B1-B32 device sources. New candidates use the compatible published TileLang
version; historical materialization verifies and wraps the frozen source
instead of asking a newer compiler to reproduce old bytes.

Each benchmark subprocess records SM occupancy with `nvidia-smi pmon` while it
runs. A process that only holds device memory but has zero SM activity is not
treated as contention. External non-zero SM activity invalidates only the
affected measurement result; the workflow waits 30 seconds and retries rather
than aborting the autotune run. Occupancy monitoring remains frequent while a
measurement is active, but conflict rechecks are deliberately low-frequency.

See [SPEC.md](SPEC.md) for the packaging and plan contracts.
