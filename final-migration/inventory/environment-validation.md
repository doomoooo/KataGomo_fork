# Phase 1 validation

Validated on 2026-08-07 in the isolated final-migration workspace. No GPU
kernel was launched by these checks; the active optimization worktrees were not
modified.

## Source builds

The source-capable Python components produced the following distributable
wheels. CUTLASS DSL is the documented upstream-binary exception because its
compiled MLIR payload is not source-published in the CUTLASS repository.

| Component | Version | Source commit | Wheel SHA-256 |
| --- | --- | --- | --- |
| CUTLASS DSL | 4.7.0 | `dcf215af68a2d08d305076c152a06f201728cd53` | upstream binary |
| Compatible TVM-FFI | 0.1.12 | `3050b0a7bd48e04f853027c5fa1f5ab7bc20b856` | `94401b0488761d06be3891efd4d84c63f1c1d9093c0038d22b9c58b0b6a1d6df` |
| Triton | 3.8.0+git5bcfc513 | `5bcfc513ddbbc64f2688dfb15a4d824c56a9649a` | `57efe48b2efcc7ba3c1e4c29447afc91de012f946906ce5c6531f4bdc103f685` |
| Quack | 0.6.3 | `050387bde3d3f03a26c87279bff2df3173640127` | `63412dd22959bc2d6a66297868f044e0da9a4c755c051a3b1643c1b5a3e89e9e` |
| FlashAttention CuTe | 0.0.1.dev1+g69e1bcbe7 | `69e1bcbe77c359c84b3a4589e92a7c076e33a202` | `6ba3976f77e2e67bd6f85ca2b68f7f1b85261908e79e986eb419997ddde9213a` |
| TileLang | 0.1.13+cuda.git12dbf3e9 | `12dbf3e9d30d84b5c27d7b8b672c268457f7eb27` | `9152b9eae45c77d59cb8d5f42517954357638a6e9dae1e2b7958c18dc854ba30` |

The import audit passed for CUTLASS, TVM-FFI, Triton, Quack, FlashAttention
CuTe, and TileLang. Triton and TileLang both compiled no-run kernels for SM89
and SM120. CUDA/cuBLAS/cuDNN link checks, CUTLASS/CuTe header checks, and the
current cuDNN frontend header check also passed for the relevant targets.

An independently built TVM-FFI HEAD (`620fece9`) was rejected after it caused
`import tilelang` to fail at `tl.SwizzleMode`. Stable tag `v0.1.12` is the
highest version satisfying both the current TileLang and FlashAttention
constraints; both imports pass and `pip check` reports no FFI conflict with it.

## KataGo and distribution

Official KataGo `v1.17.2` commit
`6a1fc5de9fc253723ac475a0683bf0b9d9b7bd19` compiled successfully with only the
CUDA backend, CUDA 13.2.86, distributed build disabled, and TCMalloc disabled.
TensorRT, OpenCL, and Eigen were not built.

The binary's dynamic dependencies resolve only through the versioned CUDA
toolkit and Ubuntu system library directories. A first build accidentally found
Nsight Systems' private zlib; the build script now pins
`/usr/lib/<multiarch>/libz.so` and packaging rejects `/opt/nvidia` or
`/workspace` runtime resolutions.

The release form is a plain, non-invasive tar: the KataGo executable, locally
built source wheels, the exact binary wheel closure, a private ELF loader and
user-space CUDA/cuDNN/C++ runtime, manifests, licenses, installer, and
`SHA256SUMS`. It writes only an explicitly supplied empty prefix and never runs
APT on a target. The concrete file count, size, checksum, local empty-prefix
test, and Ubuntu 22.04 remote test are regenerated when the final tar is cut;
the current artifact is recorded outside Git at
`.final-migration-env/state/latest-distribution`.

## Non-invasive tar qualification

Artifact cut from commit `0c9d6237658f6ffd20de92645d3ef60bc799c357`:

- tar: `.final-migration-env/distributions/20260807T205459Z.tar`;
- size: 4,942,848,000 bytes;
- SHA-256: `7720e91da425e839d62b3ac560fd01b4656c8b4f374399584de6f6cb382b4801`;
- extracted payload: 127 regular files, 4,942,719,639 bytes, including 28
  recorded native runtime libraries.

Local qualification extracted it into a new empty prefix, verified both
external artifact hashes and all internal hashes/symlinks, resolved CUDA 13,
cuDNN 9, C++ and glibc through the private loader, ran `katago version`, then
restored/imported the complete cp312 wheel archive without network access.
The dpkg package-list hash was identical before and after deployment.

Remote qualification used `wangyize@10.101.3.156`, Ubuntu 22.04 amd64, glibc
2.35, an RTX 5080 and NVIDIA driver 595.84. The build-host executable requires
newer glibc symbols, so this specifically exercised the bundled glibc 2.39
loader rather than the host loader. Installation completed below only:

`/data/wangyize/katago-tar-validation-20260807T205459Z/runtime`

`katago version` passed from an unrelated working directory. The target dpkg
hash remained
`21cb9dcace4b9f02271626ef218c17bb183c6e86f432f1603f1771ad63e9d991`
before and after. The optional Python installer correctly rejected the host's
cp310 ABI before creating an environment; the standalone CUDA backend remained
usable. No APT command, sudo elevation, or system-directory write occurred.
