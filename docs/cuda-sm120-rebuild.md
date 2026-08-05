# CUDA SM120 backend rebuild

Status: stage 1 FA4 AOT checkpoint, FP32 accumulator (both16 pending) (2026-08-05)

Goal: rebuild the accepted SM120 optimization from `cuda-optimization-history.md`
(5080 machine) on the current 5090D/4090 machines, with every SM120-specific
piece isolated in `cpp/neuralnet/cudabackend_sm120.h/cpp`. The official backend
files (`cudabackend.cpp`, `cudahelpers.cu`, `cudautils.cpp`, ...) contain only a
thin dispatch and no SM120 kernels/branches.

## Architecture

`ComputeHandle` always builds the official `Model` (weight upload, buffer layout,
fallback). On a SM120 device (`major==12 && minor==0`) it additionally builds
`Sm120Backend::Sm120Model` and routes every `apply()` through it. The official
apply is reachable through a trampoline (`applyOfficialModel`), so the SM120 path
starts bit-identical and each stage can land behind its config switch. The only
SM120 kernel landing so far is the FA4 AOT attention hook; the rest of the forward
still runs through the official apply.

Config switches are parsed in `Sm120Backend::parseOptions` and stored on
`ComputeContext`. `cudaSm120Backend=false` restores the official path for A/B.

## Rebuild stages

1. FA4 both16 attention: fixed B19/S361/H12/D32, noncausal, CuTe AOT; fall back to
   official attention for unsupported shapes/precision. (Checkpoint: fixed B13 was
   compiled first with FP32 accumulators; both16 AOT is the next artifact.)
2. Wide QKV CuTe AOT (C384 -> QKV1152), batch-shared fused Q/K RoPE, fused
   residual epilogues (beta=1 GEMM).
3. TileLang fused FFN (input projection + SwiGLU) and linear2/out-projection AOT.
4. Custom RMSNorm / SiLU / head kernels, initial-conv frontend, initial
   global matmul-add, fused policy P1, wide head projection, persisting-L2
   windows, per-GPU weight sharing.
5. Final batch/stream scan and full accuracy regression per stage.

Every stage must follow the SKILL.md evidence chain: freeze protocol, FP32
reference comparison on the full 8192-row corpus, Nsys/NCU evidence, single-
variable A/B, and entries in the optimization history.

## Verification

- Build: `cmake --build build-cuda-replayfix -j` (CUDA) and
  `cmake --build build-trt-replayfix -j` (TensorRT must stay unaffected).
- SM89 (4090): must use the official path (no `SM120 backend` log).
- SM120 (5090D): must log `SM120 backend: stage-0 official fallback active` and
  produce identical `benchmarknn`/replay results to the official path.
- All GPU runs go through `gpu-lock.sh` (CUDA 0/1 = 4090s, CUDA 2 = 5090D).
