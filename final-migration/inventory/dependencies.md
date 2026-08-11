# Dependency observations

## Unified PyPI CUDA stack

The build and code-generation environment uses one fixed CUDA stack:

| Layer | CUDA | cuDNN | Purpose |
| --- | --- | --- | --- |
| fixed PyPI wheels | 13.0.3 | 9.20 | nvcc, C++ backend, PyTorch and codegen |

The CUDA compiler, headers and libraries live below the private venv's
`site-packages/nvidia` tree. TileLang and the C++ build must not fall back to
`/usr/local/cuda`; the resulting executable is packaged with the same PyPI
runtime libraries.

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

Only two upstream source trees remain in the production setup:

- CUTLASS, for C++ headers and the CuTe dense-GEMM generator;
- FlashAttention, plus only its required `csrc/cutlass` submodule, because the
  SM89 and SM120 accepted paths carry checked-in patches.

The resolved revisions and source hashes are recorded in every release, but
are not hard-coded compatibility keys. The generator validates the required
source layout, each source rewrite requires its expected API fragments, and
CMake verifies the applied patch markers. A compatible newer upstream source
is therefore allowed; an incompatible one fails before code generation.

TileLang 0.1.13, Quack 0.6.4 and TVM-FFI 0.1.12 use their published wheels.
TVM-FFI remains at 0.1.12 because TileLang 0.1.13 declares `<0.1.13` and an
independent newer FFI can break TileLang's reflection registry despite a
successful native build. KataGo's existing `cpp/external/cudnn-frontend` copy
replaces a redundant standalone clone.

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
