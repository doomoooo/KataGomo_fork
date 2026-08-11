# H28-L2: fixed-B13/S2 persisting-L2 trunk window

Date: 2026-08-06 UTC

## Frozen target

- GPU: RTX 5090 D (SM120), CUDA device 2, exclusively held by `gpu-lock` for
  every GPU command.
- Shape/topology: exact 19x19, B13, FP16 NHWC, two NN server threads and two
  independent per-thread streams.
- Model SHA256:
  `1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`.
- Control/candidate differ only in
  `cudaUsePersistingL2Trunk=false/true`. `cudaUsePersistingL2Inner` has no
  runtime hook in this experiment.
- Current accepted SM120 kernels, including Stage 27 RMSNorm vec8, remain
  unchanged. In particular, this stage does not modify
  `cudabackend_sm120_kernels.cu`.

## Evidence and mechanism

The long-lived C768 residual is `trunkScratch.buf` allocated inside
`Trunk::apply`, not the top-level final-output `trunkBuf`. For fixed B13 it is:

```text
13 * 361 * 768 * sizeof(half) = 7,208,448 bytes per stream
S2 protected working set      = 14,416,896 bytes
```

The current 5090D Nsys target information reports `100,663,296` bytes (96 MiB)
total L2. The implementation queries `cudaDevAttrMaxPersistingL2CacheSize`,
requests at most `14,416,896` bytes via `cudaDeviceSetLimit`, reads back the
actual limit, and scales the stream hit ratio down only if the runtime grants
less than the exact S2 working set.

The same model/shape/topology route was accepted on RTX 4090 Stage 20:

- C768 L2 hit: `51.953% -> 82.636%`;
- DRAM bytes: `7.218304 MB -> 2.686976 MB` (`-62.776%`);
- representative consumer: `13.728 -> 12.704 us` (`-7.459%`);
- locked S2 ABBA: `3082.698 -> 3113.886 nn/s` (`+1.0117%`);
- accuracy replay byte-identical.

The older RTX 5080 history also accepted the route at `+0.305%`, but that B19
percentage is weaker evidence than the exact B13/S2 4090 result and is not
transferred to 5090D.

## Exact lifecycle

1. In `Trunk::apply`, allocate `trunkScratch` as today.
2. Immediately before `initialConv` starts writing it, install this stream's
   access-policy window over the full 7,208,448-byte allocation.
3. Keep the window active across initial feature accumulation, all 11 nested
   blocks, all 33 attention/FFN pairs, and the final trunk-tip RMSNorm.
4. Immediately after enqueueing the final trunk-tip norm, clear the stream
   access-policy window back to `cudaAccessPropertyNormal`.

This adds exactly one `cudaStreamSetAttribute` set and one clear per forward per
stream. It changes no arithmetic, allocation, copy, kernel geometry, or
synchronization. The window is stream-local; the set-aside is device-global.

## Falsifiable predictions

1. Nsys shows two additional stream-attribute calls per forward per stream,
   unchanged kernel/memcpy/memset/synchronization counts, and no new allocation.
2. NCU on a representative C768 trunk consumer (current
   `affineSiluHalf2Kernel<768>`, `grid=4693`, `block=384`) shows nonzero
   evict-last sectors. Acceptance also requires a directionally useful L2-hit
   increase or DRAM-byte reduction; attribute activity alone is insufficient.
3. Short B13/S2 ABBA is positive beyond run dispersion. If it is negative or
   flat, reject even if NCU confirms eviction priority.
4. If performance is accepted, full 8192-row replay must be byte-identical to
   the current accepted candidate because the arithmetic path is unchanged.

## Risks

- Two stream-local windows compete in the same L2 and device-global set-aside.
  Reserving 14.4 MB can evict higher-value GEMM weights even though it is only
  about 14.3% of total L2.
- The 5090D has substantially more L2 than the 4090, so the residual may already
  be resident and the older benefit may collapse to noise.
- The fixed S2 quota assumes exactly two active server streams on this device.
  This candidate is gated to SM120, B13, 19x19, FP16 NHWC and is not evidence
  for S1, another batch, or split streams across multiple GPUs.
- `cudaDeviceSetLimit` is context/device-global. Multiple compute handles set
  the same idempotent requested value; mixed models/topologies in one process
  would need a per-device quota owner before generalization.

## Decision protocol

1. Compile and CPU/static inspect the single-variable diff.
2. Under `gpu-lock`, run candidate/control smoke and record the runtime-granted
   persisting limit/window/hit ratio.
3. Run short ABBA with control `false` and candidate `true`.
4. Only if the short screen is positive, capture Nsys and focused NCU, followed
   by a longer ABBA/BAAB and full accuracy replay.

