# Stage 20 hypothesis: persist the long-lived C768 trunk residual in L2

Date: 2026-08-06 UTC

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2.
- Current accepted Stage16 binary/config is the control.
- SM clock 2400 MHz for attribution runs; independently primed forward and
  reverse ABBA; 500 timed forwards per sample.
- `cudaUsePersistingL2Inner=false`: this stage tests only the outer C768 trunk
  residual and does not bundle the inner C384 policy.

## Baseline evidence and mechanism

- Stage16 Nsys measures 311 kernels per forward per stream and a 255.423 ms
  union for the last 30 complete forwards on both streams. The trunk repeatedly
  crosses RMSNorm, nested pre/post projections, attention residual, and FFN
  residual boundaries, so its residency can affect several kernel classes.
- The actual long-lived residual is `trunkScratch` inside `Sm89Trunk::apply`,
  not the top-level final-output `trunkBuf`. At B13 it is
  `13 * 361 * 768 * 2 = 7,208,448` bytes (6.88 MiB) per stream, or 13.75 MiB
  for S2. The RTX 4090 reports 72 MiB total L2.
- The accepted Stage16 QKV+RoPE kernel already has 96.45% aggregate L2 hit
  rate, so this experiment does not predict a QKV-local speedup. The falsifiable
  mechanism is fewer DRAM misses for kernels that repeatedly read or update the
  long-lived residual while large weights and short-lived inner activations
  compete for L2.
- CUDA access-policy windows mark the target range evict-last on the owning
  stream. A single window is installed immediately after allocating
  `trunkScratch` and cleared after the final trunk normalization. No arithmetic,
  launch geometry, or buffer layout changes.

## Pre-implementation profiler gate

1. Nsys must confirm that residual consumers remain on the Stage16 critical
   path and that no hidden copy/allocation is required for the policy.
2. NCU with cache flushing disabled must show non-zero device-memory misses on
   representative residual consumers. If the relevant boundaries are already
   wholly L2-resident, do not implement the policy.
3. Record L2 hit/miss sectors, DRAM bytes, duration, and evict-last sectors.

## Post-implementation falsifiable test

1. NCU must show evict-last sectors on the target accesses and a directionally
   consistent reduction in device-memory misses or DRAM bytes for at least one
   representative residual consumer. A throughput change without the predicted
   cache mechanism is not accepted.
2. Nsys must show no new synchronization, memcpy, allocation, or launch and
   must not increase the complete-forward S2 kernel union.
3. A screen must be positive before three independent locked 2400 MHz
   forward/reverse ABBA rounds. Final acceptance requires directionally
   consistent orders, a positive adjacent-pair majority, and full 8192-row
   all-head/p0loss comparison against the frozen FP32 reference.

## Risks

- Protecting 13.75 MiB across S2 can evict weights that have higher reuse than
  the residual and reduce throughput even if the residual hit rate rises.
- Access-policy windows are stream-local but the L2 set-aside is device-global;
  the requested set-aside and hit ratio must account for both streams.
- NCU replay perturbs concurrency. It proves the cache mechanism only; the
  complete-forward Nsys and locked ABBA decide production value.

## Result

Accepted for exact 19x19, B13, FP16, S2 on RTX 4090.

- Paired NCU (`cache-control none`, five matching C768 samples) confirmed the
  proposed mechanism. Median duration changed from 13.728us to 12.704us
  (-7.459%), L2 hit rate from 51.953% to 82.636%, DRAM bytes from 7.218304MB
  to 2.686976MB (-62.776%), and evict-last sectors from 0 to 225264.
- Paired Nsys found identical memcpy/memset counts and bytes and no additional
  synchronization. The last 30 complete forwards on each of the two streams
  changed from 255.150329ms to 251.585792ms kernel union (-1.397%). The exact
  C768 kernel was flat under concurrent Nsys (14.778us to 14.846us), so the
  full-graph result, rather than the NCU replay duration, decides the value.
- Three independently primed locked-2400 forward/reverse ABBA rounds changed
  the pooled median from 3082.697701 to 3113.885787 nnEval/s (+1.0117%).
  Forward and reverse orders were both positive (+0.9181% and +1.1543%), all
  three round medians were positive, and 10/12 adjacent pairs were positive.
- The 8192-row all-head replay is byte-identical to the accepted Stage16 replay
  (SHA256 `7dde3f6b36e240eb4e92ffc632ecc578d059052fb2c13816b043d0a7093ba484`)
  and therefore preserves its complete FP32 error envelope and p0loss.

Artifacts: `ncu-paired/`, `nsys-paired/summary.json`,
`abba-pooled-3r-summary.json`, and
`replay-persisting-l2-trunk-vs-fp32.json`.
