# Environment setup

Use `setup.sh` as the public entry point.

For an interactive B300 code-generation shell, run:

```bash
source final-migration/environment/activate-sm103.sh
```

This selects the managed Python/CUDA stack, `sm_103a`, the system-independent
managed `ptxas`, and private Triton/CuTe/FlashInfer caches. It also sets
`FLASHINFER_NO_DOWNLOAD=1`; missing artifacts must be generated explicitly and
may not trigger an implicit network fetch during a benchmark. It also keeps
the prebuilt MSLK wheel available only as an API/source oracle; no production
path links its native kernels because that wheel has no `sm_103a` image.

```text
setup.sh install   private Python + published wheels + patched FA4 source
setup.sh audit     version/library/path audit (no installation)
setup.sh verify    compile/import smokes for third-party dependencies
setup.sh build     build the KataGo CUDA backend
setup.sh package   create a checked, non-invasive `.tar` distribution
setup.sh extract   verify/extract the tar into one empty isolated prefix
setup.sh deploy    optionally install archived Python tools below that prefix
setup.sh all       install, audit, verify, build
```

Configuration variables:

- `KATAGO_LOCAL_ARCHIVE`: local archive root; default
  `final-migration/archive`.
- `KATAGO_PYPI_MIRROR`: domestic PyPI index; default Tsinghua mirror.
- `KATAGO_PIP_EXTRA_INDEX_URLS`: additional indexes needed by the managed stack;
  defaults to NVIDIA PyPI and PyTorch's official `cu132` wheel index.
- `KATAGO_PIP_CACHE_DIR`: persistent pip cache; default
  `.final-migration-env/cache/pip` and exported as `PIP_CACHE_DIR`.
- `KATAGO_ENV_ROOT`: venv/source/build data root; default
  `.final-migration-env` in the repository.
- `KATAGO_FINAL_VENV`: isolated migration venv override. Ambient
  `KATAGO_VENV` is deliberately ignored so the setup cannot modify an active
  optimization session's environment.
- `KATAGO_PYTHON_ARCHIVE`: optional local override for the locked
  python-build-standalone archive. The archive's fixed SHA-256 is always
  checked before extraction.
- `KATAGO_THIRD_PARTY_ROOT`: override managed source location.
- `KATAGO_REFRESH_SOURCES=1`: explicitly refetch the two pinned source commits.
  The default reuses a clean checkout only when its commit matches the lock.
- `KATAGO_INCLUDE_RESEARCH=1`: also acquire reference-only repositories.
- `KATAGO_BUILD_JOBS`: explicit parallel compile override. By default the
  scripts take the lower of `nproc` and a memory-aware limit: 75% of current
  `MemAvailable`/cgroup headroom divided by 2 GiB per job. This leaves room for
  native code generators without hard-coding `-j4`/`-j8`.
- `KATAGO_SMOKE_ARCHS`: space-separated CUDA smoke architectures; default
  detected architectures, falling back to `89 103 120`.
- `KATAGO_KEEP_SOURCE_BUILD_TREES=1`: retain ignored compiler intermediates;
  default discards them after the hashed wheel is installed to protect limited
  workspace capacity.
- `KATAGO_RESUME_SOURCE_BUILD`: resume a specific interrupted source-build
  directory; every reused wheel is checked against its recorded SHA-256.
- `KATAGO_MIN_DRIVER`: minimum target NVIDIA driver recorded in a tar; defaults
  to the conservative CUDA 13.3 packaging policy (`610.43.02`). The current
  B300 qualification host runs 610.57.04; this value is not a claim that every
  driver at the policy floor was independently tested.

Only CUTLASS and FlashAttention are source inputs. CUTLASS supplies headers and
the CuTe dense-GEMM generator; FlashAttention carries the checked-in SM89 and
SM120 patches. Both source inputs are pinned in `third-party.lock.tsv`. B300
reports CC 10.3. Generic CUDA, Torch and dependency smokes use
`sm_103`/`10.3`, while accelerated CuTe artifacts use the exact `sm_103a`
target and FlashInfer spells that target `10.3a`.
FlashAttention initializes only its required `csrc/cutlass`
submodule. A matching clean checkout is reused without network access. A first
clone or explicit refetch may require a GitHub proxy. Example:

```bash
HTTPS_PROXY=http://proxy.example:7890 \
  ./final-migration/environment/setup.sh all
```

Each run verifies and records the two pinned source revisions. API-shape checks
and the two local patches add another fail-closed boundary. Binary packages
first use `archive/wheels`, then the configured indexes and pip cache. Every
line in the two requirement files is an unconditional exact pin, and the shared
environment checker compares all installed versions against them. A repeated
setup with a matching local archive/cache performs no index lookup.

CUTLASS DSL 4.7.0, TileLang 0.1.13, Quack 0.6.4, FlashInfer 0.6.17,
cuDNN Frontend Python 1.27.0, Liger Kernel 0.8.1, MSLK 1.3.0+cu132,
compatible TVM-FFI 0.1.12, and z3-solver 4.15.4.0 use published wheels.
TVM-FFI and z3 are the newest
versions admitted by TileLang's `<0.1.13` and `<4.15.5` constraints. Quack's
older CUTLASS DSL metadata constraint is handled only by the existing narrow,
smoke-tested waiver. Protobuf remains at 6.33.6 because CUTLASS DSL 4.7
requires `<7`. TileLang's wheel already carries the native library and the
CUTLASS/template headers needed to compile generated sources, so cloning its
roughly gigabyte recursive repository adds no production capability. The C++
cuDNN frontend remains the reviewed copy vendored under `cpp/external`; the
separately locked Python wheel supplies open-source CuTe-DSL operators and is
also required by FlashInfer. Only the patched `flash-attn-4` Python package is
built locally.

Triton is not cloned or built from source. The exact binary version required
by PyTorch is carried as a pinned wheel and is admitted as the SM103 AOT
research compiler. FlashInfer, Liger, and MSLK are candidate generators or
oracles, not enabled runtime paths merely because their wheels are installed.
Every generated cubin still requires an activation proof and plan-selected
tactic before it can be linked into the CUDA backend. FlashInfer 0.6.17 does
not list native CUDA 13.3 in its upstream qualification matrix, so its SM103
CuTe-DSL path remains compile-and-correctness gated on this stack. The MSLK
wheel is an API/source oracle: its shipped native architecture list does not
contain `sm_103a`.

fbtriton/TLX, Transformer Engine's PyTorch extension, DeepGEMM, and
ThunderKittens are excluded from the main venv. They replace the Triton
namespace, carry a separate Torch/CUDA native ABI, or require a source-only
per-kernel build, and therefore belong in isolated research environments.

The public setup never invokes APT, `sudo`, or a driver installer. A source
checkout requires an operational NVIDIA driver, a compiler, and zlib
development files. The active venv has one set of runtime DSOs: native CUDA
toolkit 13.3.1 (nvcc/CRT/NVVM 13.3.73) with cuDNN 9.25.0.15. PyTorch
2.13.0+cu132 was built and published against CUDA 13.2.1/cuDNN 9.20 metadata,
but it resolves the managed 13.3/9.25 DSOs in this venv. This is a deliberately
recorded ABI/runtime compatibility crossing, not two physically isolated
runtime closures. PyTorch publishes no cu133 wheel, so every new Torch native
extension remains qualification-gated. The environment scripts verify exact
package pins, cross-boundary imports, source generation and compile-only native
smokes. GPU execution and model correctness are separate qualification records,
not claims made by the environment smoke. Setup obtains the locked Python
3.14.7 standalone archive from the local archive first and otherwise from its
recorded upstream URL. The source-complete release tar
carries the complete fixed wheel set, Python runtime and sources, so target
setup remains fully offline.

The 423 MB cuBLAS wheel and the other PyTorch CUDA libraries are real runtime
dependencies, not duplicate toolchains. A fresh machine must obtain each once;
the local archive/cache and the release tar prevent repeated downloads.

The distributable path is separate from source development setup. It bundles
the compiled executable, a private ELF loader and user-space runtime libraries
in a plain tar. `setup.sh extract ARCHIVE PREFIX` accepts only
an empty, non-system prefix and all verification runs in place. Python wheels
are archival/optional and require the recorded Python ABI; KataGo itself has no
Python runtime dependency.

TensorRT 10.16.1.11 is the fixed B300 comparison baseline. It is not installed
by these scripts, linked into the optimized CUDA product, or shipped in its
runtime artifact. Eigen, OpenCL, and distributed-build dependencies remain out
of scope.
