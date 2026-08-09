# Environment setup

Use `setup.sh` as the public entry point.

```text
setup.sh install   system + Python + latest locally built source dependencies
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
- `KATAGO_SYSTEM_PYTHON`: interpreter used only to create the isolated venv;
  defaults to `/usr/bin/python3` rather than an ambient activated environment.
- `KATAGO_THIRD_PARTY_ROOT`: override managed source location.
- `KATAGO_INCLUDE_RESEARCH=1`: also acquire reference-only repositories.
- `KATAGO_ALLOW_STALE_BINARY=1`: explicitly accept a fully local binary seed
  when the mirror cannot be reached; normal development requires a latest
  version check.
- `KATAGO_INSTALL_DRIVER=0`: forbid driver installation on a fresh host.
- `KATAGO_CUDA_TOOLKIT_PACKAGE`, `KATAGO_CUDNN_PACKAGE`,
  `KATAGO_NSIGHT_SYSTEMS_PACKAGE`, `KATAGO_NSIGHT_COMPUTE_PACKAGE`, and
  `KATAGO_DRIVER_PACKAGE`: explicit compatibility overrides. Defaults resolve
  current packages from the NVIDIA repository.
- `KATAGO_BUILD_JOBS`: explicit parallel compile override. By default the
  scripts take the lower of `nproc` and a memory-aware limit: 75% of current
  `MemAvailable`/cgroup headroom divided by 2 GiB per job. This leaves room for
  memory-heavy Triton C++ translation units without hard-coding `-j4`/`-j8`.
- `KATAGO_SMOKE_ARCHS`: space-separated CUDA smoke architectures; default
  detected architectures, falling back to `89 120`.
- `KATAGO_KEEP_SOURCE_BUILD_TREES=1`: retain ignored compiler intermediates;
  default discards them after the hashed wheel is installed to protect limited
  workspace capacity.
- `KATAGO_RESUME_SOURCE_BUILD`: resume a specific interrupted source-build
  directory; every reused wheel is checked against its recorded SHA-256.
- `KATAGO_MIN_DRIVER`: minimum target NVIDIA driver recorded in a tar; defaults
  to the CUDA 13.2 build baseline (`595.45`) and must be updated when moving to
  a CUDA release with a different compatibility floor.

For source-capable dependencies, the setup checks current upstream HEAD and
builds it locally. A local Git bundle seeds that checkout but does not prevent
the latest-commit check. GitHub access may require a proxy. Example:

```bash
HTTPS_PROXY=http://proxy.example:7890 \
  ./final-migration/environment/setup.sh all
```

Each run captures resolved source commits and built wheel hashes. Those
artifacts are what deployment consumes, so a later upstream change cannot
silently alter an already packaged build. Binary-only/bootstrap packages first
use `archive/wheels`, then the domestic PyPI mirror.

CUTLASS is cloned at current HEAD and its headers are used directly. CuTe DSL's
compiled MLIR payload is not published as buildable source by that repository;
the matching official CUDA-13 wheel selected by the source tree is therefore a
documented binary exception.

TileLang and FlashAttention are cloned at current HEAD, but their shared native
ABI dependency is constrained: TVM-FFI is built from the highest stable source
tag satisfying both projects (`v0.1.12` for this source snapshot). Following
TVM-FFI's independent HEAD can compile successfully while breaking TileLang's
reflection registry at import time. The compatibility pin must be revalidated
when either top-level project's metadata constraint changes.

On a fresh machine the bootstrap reads `/etc/os-release` and uses the matching
NVIDIA Ubuntu repository (for example `ubuntu2204` or `ubuntu2404`); it does not
hard-code one Ubuntu release. CUDA/cuDNN/profiler meta packages resolve their
current repository versions. An already operational driver is deliberately
left untouched. On a shared optimization host, use explicit package overrides
instead of changing its compiler or driver underneath active sessions.

The distributable path is separate from development bootstrap. It never uses
APT: it bundles the compiled executable, a private ELF loader and user-space
runtime libraries in a plain tar. `setup.sh extract ARCHIVE PREFIX` accepts only
an empty, non-system prefix and all verification runs in place. Python wheels
are archival/optional and require the recorded Python ABI; KataGo itself has no
Python runtime dependency.

TensorRT, Eigen, and OpenCL are intentionally out of scope and are not installed
or tested by these scripts.
