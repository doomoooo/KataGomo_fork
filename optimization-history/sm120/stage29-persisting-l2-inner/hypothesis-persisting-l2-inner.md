# H29-L2-inner: fixed-B13/S2 persisting-L2 C384 inner window

Date: 2026-08-06 UTC

## Frozen target and control

- GPU: RTX 5090 D (SM120), CUDA device 2, held by `gpu-lock` for every GPU
  command.
- Shape/topology: exact 19x19, B13, FP16 NHWC, two NN server threads and two
  independent per-thread streams.
- Model SHA256:
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- The accepted Stage28 C768 trunk window is enabled in both arms with
  `cudaUsePersistingL2Trunk=true`.
- Control/candidate differ only in `cudaUsePersistingL2Inner=false/true`.
  The option remains false by code default.
- Existing SM120 kernels, including affine/RMSNorm vec8 paths, remain unchanged.

## Prior evidence and mechanism

Each of the 11 nested bottleneck residual blocks owns a long-lived C384 `mid`
residual across six alternating transformer attention/FFN inner blocks. At B13:

```text
C768 trunk window per stream = 13 * 361 * 768 * 2 = 7,208,448 bytes
C384 inner window per stream = 13 * 361 * 384 * 2 = 3,604,224 bytes
combined S2 set-aside request = 2 * (7,208,448 + 3,604,224)
                              = 21,625,344 bytes
```

CUDA exposes one access-policy window per stream. The candidate therefore
switches the stream window from C768 to C384 during an inner block and restores
C768 afterward; it does not attempt to install both windows simultaneously.
The larger device-global set-aside is intended to let recently used trunk lines
survive while the active stream policy protects `mid`.

RTX 4090 Stage21 accepted the identical B13/S2 lifetime optimization on top of
its trunk window. Its clean split NCU evidence showed the C384 consumer at
`6.62 -> 6.59 us`, L2 hit `52.51% -> 97.99%`, and evict-last sectors
`0 -> 112,632`; the C768 guard remained stable at `11.36 -> 11.30 us` and
`82.50% -> 82.97%` L2 hit. Three locked forward/reverse ABBA rounds improved
pooled throughput by `+1.2903%`. This is transfer evidence only, not a 5090D
acceptance result.

## Exact lifecycle boundary

For each `NestedBottleneckResidualBlock::apply`:

1. Run `normActConv1.norm`, producing the normalized C768 input in
   `trunkScratchBuf` while the accepted trunk window remains active.
2. Immediately before `normActConv1.conv` writes `mid.buf`, switch this stream's
   access-policy window to the exact 3,604,224-byte C384 `mid` allocation.
3. Retain that window across all six inner transformer blocks and
   `normActConv2.norm`, the final direct consumer of `mid`.
4. Immediately after enqueueing `normActConv2.norm`, before
   `normActConv2.conv` reads `midScratch` and accumulates into the outer
   residual, restore the exact C768 window over `trunkBuf`.

This adds 22 `cudaStreamSetAttribute` calls per forward per stream: one switch
and one restore for each of 11 nested blocks. Together with Stage28 trunk
set/clear, the candidate should have 24 calls per forward per stream. It changes
no arithmetic, kernel geometry, allocation, copy, or synchronization.

## Falsifiable predictions

1. Smoke and short symmetric ABBA must be positive before profiling or long
   testing. A flat/negative screen rejects the candidate even if cache counters
   move in the predicted direction.
2. Nsys must show unchanged kernel, memcpy, memset, and synchronization counts;
   only the predicted stream-attribute calls may increase. Complete-forward
   kernel union must not regress materially.
3. Exact B13 C384 `affineSiluHalf2Kernel<384>` consumers (`grid=4693`,
   `block=192`) must show nonzero evict-last sectors plus higher L2 hit or lower
   DRAM traffic.
4. Exact B13 C768 consumers (`grid=4693`, `block=384`) are a guard: protecting
   C384 must not materially damage the accepted outer-residual residency.
5. If long ABBA/BAAB accepts the candidate, the 8,192-row replay must be
   byte-identical to Stage28 because the arithmetic path is unchanged.

## Risks and rejection conditions

- The 5090D already has 96 MiB total L2, so C384 may naturally reside and the
  older 4090 benefit can collapse to noise.
- A stream can designate only one active window. C768 receives no persisting
  priority while six inner blocks execute, even though the global set-aside has
  capacity for both working sets.
- The 44 extra host API calls across S2 can perturb launch pacing and stream
  phase. NCU improvement alone is insufficient without full-graph S2 evidence.
- The fixed quota assumes exactly two streams and this exact model/shape. Mixed
  models, more streams, or other batches require a separate per-device quota
  policy and are outside this candidate.

## Decision protocol

1. Implement the exact switch/restore boundary behind the existing inner flag;
   compile and statically verify no kernel/arithmetic changes.
2. Run control/candidate smoke and short ABBA under the device-2 lock.
3. Only if positive, capture Nsys and focused NCU for C384 plus the C768 guard.
4. Only if mechanism and full-graph evidence agree, run long symmetric
   ABBA/BAAB and the complete 8,192-row replay.

