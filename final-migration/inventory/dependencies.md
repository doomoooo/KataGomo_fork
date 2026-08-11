# Dependency observations

## System and Python CUDA stacks

The current host intentionally has two related but non-identical stacks:

| Layer | CUDA | cuDNN | Purpose |
| --- | --- | --- | --- |
| system toolkit | 13.2 | 9.25 | nvcc and C++ KataGo backends |
| PyTorch wheels | 13.0 | 9.20 | Python search/codegen tooling |

The environment audit must report both. `LD_LIBRARY_PATH` must not cause the
Python wheel libraries to replace system libraries for the C++ executable.

## Historical optimization toolchain observed

- CUTLASS/CuTe 4.6.1 source headers;
- cuDNN frontend 1.26.0 source headers (official KataGo also vendors a copy);
- FlashAttention FA4/CuTe source near `fa4-v4.0.0.beta21`, with local 4090
  modifications in the active source checkout;
- TileLang 0.1.13;
- Triton 3.7.1;
- PyTorch 2.13.0 with CUDA 13.0 wheels;
- NVIDIA CUTLASS DSL 4.7.0 in the captured environment;
- QFlash, SageAttention, and Luminal as research/reference inputs.

These versions describe the active optimization environments and are evidence,
not the clean migration dependency policy. The active FlashAttention checkout
is dirty in Hopper launch, SM80 mainloop,
softmax, and tile-size files and contains generated AOT directories. Those edits
must later be archived as a reviewed patch associated with the frozen 4090
result. A clean upstream checkout is not equivalent to the observed result.

The captured `/workspace/venv` is accidentally mixed: FA4 4.0.0b25 and Quack
0.5.3 retain CUTLASS DSL 4.6 constraints, while the DSL/base packages were
upgraded to 4.7.0 and only the CUDA 13 library package remained 4.6.0.dev0.
Imports succeed, but `pip check` fails for both the expected upstream metadata
constraint and the unintended base/CUDA-library mismatch.

## Clean migration source policy

The clean environment resolves upstream default-branch HEAD and builds locally
instead of copying packages from `/workspace/venv`. The 2026-08-07 source
manifest resolved:

| Source | Resolved commit | Description |
| --- | --- | --- |
| CUTLASS | `dcf215af68a2d08d305076c152a06f201728cd53` | `dcf215af` |
| Quack | `050387bde3d3f03a26c87279bff2df3173640127` | `v0.6.3` |
| FlashAttention | `69e1bcbe77c359c84b3a4589e92a7c076e33a202` | `69e1bcb` |
| TileLang | `12dbf3e9d30d84b5c27d7b8b672c268457f7eb27` | `12dbf3e9` |
| Compatible TVM FFI | `3050b0a7bd48e04f853027c5fa1f5ab7bc20b856` | `v0.1.12` |
| cuDNN frontend | `ec139877e51f17d6b1d7520d9789f34d1c65f77e` | `ec13987` |

The complete generated manifest also records recursive submodule commits. A
future setup run may resolve newer commits; the compiled distribution captures
the exact resolved tuple and every wheel hash so deployment remains immutable.

TVM-FFI is deliberately not resolved from its independent default-branch HEAD.
TileLang's native libraries and Python enum/reflection registry share an ABI,
while FlashAttention also declares a minimum FFI release. The build therefore
uses the highest stable source tag in their current constraint intersection:
`v0.1.12`. An experiment with independent TVM-FFI HEAD
`620fece9f8d81dded637cec9fc52e388f7bd0ae1` built successfully but failed at
`import tilelang` because the `tl.SwizzleMode` registration was incompatible.
Both TileLang and FlashAttention imports pass with `v0.1.12`, and their FFI
metadata requirements are satisfied.

PyTorch 2.13.0/CUDA 13.0 and other Python bootstrap wheels are binary inputs.
CUTLASS headers come from current source. CUTLASS CuTe DSL 4.7's compiled MLIR
libraries are not source-published in the CUTLASS repository, so the official
CUDA-13 DSL wheel is an explicit binary exception. PyTorch's pinned Triton
dependency is also carried as a binary wheel, but no Triton source, LLVM
toolchain, or KataGo Triton kernel is part of the workflow.

## Explicit exclusions

The final product only uses KataGo's CUDA backend. TensorRT, Eigen, OpenCL, and
distributed-build dependencies are not installed or compiled by the migration
environment. TCMalloc is also disabled unless a later measured CUDA/GTP build
explicitly establishes it as part of the accepted configuration.
