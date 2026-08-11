# Environment setup

Use `setup.sh` as the public entry point.

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
- `KATAGO_ENV_ROOT`: venv/source/build data root; default
  `.final-migration-env` in the repository.
- `KATAGO_FINAL_VENV`: isolated migration venv override. Ambient
  `KATAGO_VENV` is deliberately ignored so the setup cannot modify an active
  optimization session's environment.
- `KATAGO_PYTHON_ARCHIVE`: optional local override for the locked
  python-build-standalone archive. The archive's fixed SHA-256 is always
  checked before extraction.
- `KATAGO_THIRD_PARTY_ROOT`: override managed source location.
- `KATAGO_REFRESH_SOURCES=1`: explicitly refresh the two cached upstream
  source trees. The default reuses a clean checkout and performs no HEAD check.
- `KATAGO_INCLUDE_RESEARCH=1`: also acquire reference-only repositories.
- `KATAGO_BUILD_JOBS`: explicit parallel compile override. By default the
  scripts take the lower of `nproc` and a memory-aware limit: 75% of current
  `MemAvailable`/cgroup headroom divided by 2 GiB per job. This leaves room for
  native code generators without hard-coding `-j4`/`-j8`.
- `KATAGO_SMOKE_ARCHS`: space-separated CUDA smoke architectures; default
  detected architectures, falling back to `89 120`.
- `KATAGO_KEEP_SOURCE_BUILD_TREES=1`: retain ignored compiler intermediates;
  default discards them after the hashed wheel is installed to protect limited
  workspace capacity.
- `KATAGO_RESUME_SOURCE_BUILD`: resume a specific interrupted source-build
  directory; every reused wheel is checked against its recorded SHA-256.
- `KATAGO_MIN_DRIVER`: minimum target NVIDIA driver recorded in a tar; defaults
  to the CUDA 13.0 baseline (`580.65.06`) and must be updated when moving to a
  CUDA release with a different compatibility floor.

Only CUTLASS and FlashAttention are source inputs. CUTLASS supplies headers and
the CuTe dense-GEMM generator; FlashAttention carries the checked-in SM89 and
SM120 patches. FlashAttention initializes only its required `csrc/cutlass`
submodule. Clean cached checkouts are reused without network access; use
`KATAGO_REFRESH_SOURCES=1` to request a new upstream snapshot. A first clone or
explicit refresh may require a GitHub proxy. Example:

```bash
HTTPS_PROXY=http://proxy.example:7890 \
  ./final-migration/environment/setup.sh all
```

Each run records resolved revisions for provenance, but neither the generator
nor CMake requires one hard-coded commit. API-shape checks and the two local
patches fail closed if a newer source is incompatible. Binary packages first
use `archive/wheels`, then the domestic PyPI mirror and pip cache. A hash of the
two requirement files plus installed-version checks makes repeated setup
entirely local instead of contacting the index again.

CUTLASS DSL, TileLang 0.1.13, Quack 0.6.4 and compatible TVM-FFI 0.1.12 use
published wheels. TileLang's wheel already carries the native library and the
CUTLASS/template headers needed to compile generated sources, so cloning its
roughly gigabyte recursive repository adds no production capability. The
cuDNN frontend is the copy already vendored by KataGo under `cpp/external`.
Only the patched `flash-attn-4` Python package is built locally.

Triton is not a KataGo code generator in this workflow and is not cloned or
built from source. The exact binary version required by PyTorch is carried as
a pinned transitive wheel; no Triton kernel is generated, benchmarked, or
linked into the CUDA backend.

The public setup never invokes APT, `sudo`, or a driver installer. A source
checkout requires an operational NVIDIA driver, a compiler, and zlib
development files. CUDA 13.0.3, nvcc, cuDNN 9.20 and the CUDA math/runtime
libraries are PyPI packages installed once into the same private Python
environment used by PyTorch, TileLang and FlashAttention. The C++ backend uses
that same wheel layout; there is no second CUDA toolkit tree. Setup also obtains
the locked Python 3.12.13 standalone archive from the local archive first and
otherwise from its recorded upstream URL. The source-complete release tar
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

TensorRT, Eigen, and OpenCL are intentionally out of scope and are not installed
or tested by these scripts.
