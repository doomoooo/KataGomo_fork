# Environment setup

Use `setup.sh` as the public entry point.

```text
setup.sh install   system + Python + latest locally built source dependencies
setup.sh audit     version/library/path audit (no installation)
setup.sh verify    compile/import smokes for third-party dependencies
setup.sh build     build the KataGo CUDA backend
setup.sh package   create a checked prebuilt distribution bundle
setup.sh deploy    install a checked prebuilt bundle without source builds
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
- `KATAGO_BUILD_JOBS`: parallel compile count; default is capped at eight and
  approximately one job per 4 GiB RAM because Triton C++ translation units are
  memory-heavy.
- `KATAGO_SMOKE_ARCHS`: space-separated CUDA smoke architectures; default
  detected architectures, falling back to `89 120`.
- `KATAGO_PACKAGE_TAR=1`: additionally emit a `.tar.zst` distribution and an
  external SHA-256 file for transfer.
- `KATAGO_KEEP_SOURCE_BUILD_TREES=1`: retain ignored compiler intermediates;
  default discards them after the hashed wheel is installed to protect limited
  workspace capacity.
- `KATAGO_RESUME_SOURCE_BUILD`: resume a specific interrupted source-build
  directory; every reused wheel is checked against its recorded SHA-256.
- `KATAGO_SKIP_SYSTEM_BOOTSTRAP=1`: prebuilt deployment only; skip APT after
  verifying that the bundle's exact CUDA toolkit and cuDNN package names are
  already installed.

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

On a fresh machine the NVIDIA CUDA/cuDNN/driver meta packages resolve their
current repository versions. An already operational driver is deliberately
left untouched. On a shared optimization host, use explicit package overrides
instead of changing its compiler or driver underneath active sessions.

TensorRT, Eigen, and OpenCL are intentionally out of scope and are not installed
or tested by these scripts.
