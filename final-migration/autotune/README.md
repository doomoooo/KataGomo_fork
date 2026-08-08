# Unified SM89 / SM120 autotune SDK

This directory defines the source-based, non-invasive autotune distribution.
It is separate from `environment/package-distribution.sh`, which packages an
already-built inference runtime.

The release artifact is one uncompressed outer `.tar`.  It carries a pinned
Python runtime, CUDA 13.2 build toolkit, cuDNN 9.25 for CUDA 13, the complete
KataGo source tree, materialized third-party source trees, build-prerequisite
payloads, pinned binary wheels, the model, and integrity manifests.  The
target does not clone GitHub repositories or resolve dependency versions.

After extracting the release in a writable persistent directory:

```bash
./setup.sh
./run-autotune.sh --device 0
```

The host baseline is Linux x86-64 with glibc 2.28 or newer, an NVIDIA driver
compatible with CUDA 13.2, and the small OS bootstrap set checked by `setup.sh`
(`bash`, GNU tar/coreutils, GCC/G++, and `flock`). Everything above that
bootstrap is carried in the tar; setup performs no APT transaction, Git clone,
or network access. This deliberately supports both validated Ubuntu 22.04 and
24.04 hosts instead of encoding one Ubuntu release.

`setup.sh` writes only below the extracted directory unless `--prefix` is
given.  It builds TVM-FFI, Triton, TileLang, Quack and FlashAttention from the
carried source into the locked Python 3.12 environment.  PyTorch and NVIDIA
CUTLASS DSL are explicit upstream-binary exceptions; their exact wheels are
carried because neither project exposes a practical equivalent source-only
Python payload for this workflow.  CUDA and cuDNN are NVIDIA binary
toolchains, not Python source dependencies.

`run-autotune.sh` queries the selected device through the CUDA Runtime.  CC
8.9 dispatches the SM89 workflow and CC 12.0 dispatches the SM120 workflow.
The default domain is exact B4-B32 with two inference streams.  Discovery is
short; a final plan is only marked scan-bypass-ready after the 1000-iteration,
two-repeat long gate.  Absence of a retained FP32 golden leaves numerical
`production_ready` false rather than manufacturing a new reference from the
candidate under test.

The release includes its own per-GPU lock wrapper. CUDA ordinal to physical
`nvidia-smi` ordinal mapping is done by PCI identity before the lock is taken,
so `CUDA_VISIBLE_DEVICES` remapping remains safe. A compatible system
`gpu-lock` takes precedence when one is installed.

See [SPEC.md](SPEC.md) for the packaging and plan contracts.
