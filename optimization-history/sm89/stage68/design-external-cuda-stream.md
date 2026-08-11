# Stage 68: externally owned CUDA stream

## Problem

The CUDA backend currently relies on `cudaStreamPerThread` in three different ways:

1. `CudaHandles` binds cuBLAS and cuDNN to PTDS.
2. the independent SM89 forward creates a second handle set bound to PTDS, and a few SM89 kernels name PTDS directly;
3. the generic kernels in `cudahelpers.cu` omit the launch stream, so the build-wide
   `CUDA_API_PER_THREAD_DEFAULT_STREAM` definition silently places them on PTDS.

Changing only the stream stored in `CudaHandles` would therefore be incorrect: library work could run on the external stream while generic kernels continued on PTDS, removing ordering guarantees.

## Ownership and interface

- The NN server/caller owns one explicit, non-blocking CUDA stream per CUDA compute handle.
- The stream is created after selecting the handle's CUDA device, passed to
  `NeuralNet::createComputeHandle`, and destroyed only after the compute handle is freed.
- `ComputeHandle` and all subordinate CUDA objects borrow the stream; none destroys it.
- The backend-neutral interface carries the backend-native stream as an opaque `void*`.
  CUDA requires a non-null value and casts it to `cudaStream_t`; non-CUDA backends ignore it.
- The CUDA backend does not fall back to PTDS. A missing stream is an error, so future call sites cannot silently reintroduce the old behavior.

## Execution rules

- `CudaHandles`, its cuBLAS handle, and its cuDNN handle use the borrowed stream.
- `Sm89Forward` and its private cuBLAS/cuDNN handles borrow that same stream.
- Every custom CUDA helper launch receives an explicit `cudaStream_t`; no launch relies on the compiler's default-stream mode.
- Optimized SM89 kernels receive `Sm89Ctx::stream`; there are no direct PTDS references.
- Runtime H2D/D2H copies use `cudaMemcpyAsync` on the borrowed stream. `getOutput`
  synchronizes that stream before CPU postprocessing. Pure-device benchmark events and
  synchronization use the borrowed stream.
- Initialization-only allocation and weight upload may remain synchronous, but construction
  finishes with synchronization of the borrowed stream, not a device-wide steady-state dependency.

## Lifetime and failure handling

- Callers use RAII so stream destruction also happens when handle construction or inference throws.
- A compute handle must be freed before its stream.
- The handle remains single-host-thread/non-thread-safe, as before.
- No stream may be shared across CUDA devices.

## Verification

1. CUDA build and backend unit/smoke tests pass.
2. Exact B13 19x19 S1 and S2 produce the accepted Stage62/65 outputs.
3. Static audit finds no `cudaStreamPerThread` in the CUDA backend or SM89 forward and no
   streamless launch in `cudahelpers.cu`.
4. Nsys shows each server's CUDA API, custom kernels, cuBLAS and cuDNN work on its explicitly
   created stream, with no forward kernels on PTDS/default.
5. Short S1/S2 throughput is checked for regression. This change is primarily an interface and
   ordering fix, so it is committed only after correctness and stream-placement evidence pass.

## Verification result

- CUDA Release build completed successfully; `runnnlayertests` passed all 28 configurations.
- Exact-19 B13 replay on the fixed 26-row corpus is bit-identical to the accepted Stage62 output
  for policy, optimistic policy, value, score and ownership (all reported errors zero, top-1 1.0).
- Static audit finds no `cudaStreamPerThread` or
  `CUDA_API_PER_THREAD_DEFAULT_STREAM` in the CUDA backend, SM89 forward, CUDA helpers, or the
  CUDA CMake target. TensorRT remains a separate backend and is deliberately outside this change.
- Natural S2 Nsys execution places all 5,916 captured forward/warmup kernels on four explicit
  non-blocking streams. The two measured forward streams are 81 and 82; each contains its own
  H2D and D2H copies. No forward kernel executes on a default stream.
- Three 100-iteration S2 runs produced 3462.633, 3460.038 and 3450.634 nnEval/s
  (mean 3457.768). This is a no-regression check, not a causal speedup claim, because the accepted
  Stage62 checkpoint was not rerun as a same-binary paired control.

Decision: accept the interface/lifetime fix. A `ComputeHandle` now borrows the opaque stream
passed by its caller, and every CUDA execution primitive is ordered on that stream.
