# Unified SM89 / SM120 autotune SDK and SM103 qualification

This directory defines the source-based, non-invasive autotune distribution.
It is separate from `environment/package-distribution.sh`, which packages an
already-built inference runtime.

The release artifact is one uncompressed outer `.tar`. Release construction
uses the maintained training archive and corpus identities in `corpus.lock.sh`.
The current gate contains exactly 8192 deterministically sampled full 19x19
rows with seed 20260803. The corpus changes only when maintainers intentionally
regenerate the FP32 reference and certified plans together. The tar carries the
corpus and its complete source/row manifest. It also carries CPython 3.14.7,
PyTorch 2.13.0+cu132, native CUDA toolkit 13.3.1 and cuDNN 9.25.0.15 in one
managed prefix. The complete KataGo source tree, materialized
third-party source trees, build prerequisites, model, and integrity manifests
are carried as well. The target does not clone GitHub repositories or resolve
dependency versions.

PyTorch 2.13.0 has no official `cu133` wheel. Its `cu132` build ABI and metadata
(CUDA 13.2.1/cuDNN 9.20.0.48) are recorded separately from the active
CUDA 13.3.1/cuDNN 9.25.0.15 DSOs, but these are not two physical runtime
closures: the managed prefix contains one active DSO set. The three exact
metadata conflicts are fail-closed exceptions, not a general `pip check`
waiver. TensorRT 10.16.1.11 is the fixed B300 comparison baseline and is not
installed, linked or shipped by this environment.

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
meeting the conservative CUDA 13.3 packaging policy (610.43.02), and the small
OS bootstrap set checked by `setup.sh`
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
given. It installs TileLang 0.1.13 with its newest compatible TVM-FFI 0.1.12
and z3-solver 4.15.4.0, Quack 0.6.4, and CUTLASS DSL 4.7.0, then builds only
the locally patched FlashAttention package from carried source. The Quack DSL
metadata mismatch is a narrow, smoke-tested exception; it is not permission to
waive unrelated dependency errors. PyTorch's Triton wheel dependency is carried
as a binary package and may be used for isolated AOT research, but no retained
production tactic is implied by installing it. All Python generators resolve
the one active managed CUDA 13.3/cuDNN 9.25 DSO set described above.

Build parallelism defaults to the lower of `nproc`, 8, and a memory-aware limit
(75% of current `MemAvailable`/cgroup headroom at 2 GiB per heavy compiler
process). This avoids fixed `-j4`/`-j8` values while protecting memory-limited
hosts. `--jobs N` remains an explicit override.

`run-autotune.sh` queries the selected device through the CUDA Runtime. CC 8.9
dispatches the SM89 workflow and CC 12.0 dispatches the SM120 workflow. NVIDIA
B300 reports CC 10.3. Generic dependency smokes use `sm_103`, while accelerated
CuTe artifacts use exact `sm_103a`. The checked-in SM103 workflow remains
fail-closed outside its explicitly qualified B29/S2 contract; it is never
misrouted through the SM89 or SM120 catalog.
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
