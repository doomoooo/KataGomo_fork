# Stage 60B hypothesis: half2 scaling in folded-RMS QKV epilogue

## Frozen target and parent candidate

- Target remains RTX 4090 SM89, exact 19x19 B13 FP16 NHWC, with real S2 as
  the deployment topology.
- Parent candidate is Stage60 attention-only RMSNorm-to-QKV folding. No graph
  boundary, tile, launch geometry, weight folding, or other operator changes.
- Single variable: replace scalar FP32 invRMS application to the already-FP16
  QKV output fragment with one FP16 row scale and vectorized half2 multiply.

## Evidence and mechanism

Stage60 NCU disproved the original FP32-scale implementation:

- Real S2-source boundary: 34.592 us control versus 35.200 us candidate
  (+1.76%).
- Single-stream intrinsic boundary: 34.944 us control versus 35.264 us
  candidate (+0.92%).
- The reduction improved from 4.640 to 2.816 us in the intrinsic capture, but
  QKV regressed from 30.304 to 32.448 us. Grid, registers (240/thread), shared
  memory (50.176 KiB/block), and spills remained unchanged, isolating the cost
  to the epilogue's per-element FP16-to-FP32 conversion, multiply, and FP16
  conversion.

For 13*361*3*384 QKV output elements, the revised epilogue converts the FP32
invRMS to FP16 once per logical row handled by the iterator and applies it with
half2 multiplies. This halves the arithmetic instruction count for scaling and
removes scalar FP32 conversions around every output element. The operation is
performed after the GEMM has already rounded its output fragment to FP16, so it
matches the accepted Stage59 both16 accuracy policy rather than weakening a
remaining FP32 accumulation boundary.

## Prediction and gates

- QKV must recover at least the 0.32 us intrinsic boundary deficit, with no
  register, shared-memory, occupancy, or spill regression.
- A strict intrinsic `invRMS + folded-QKV` win permits short real-S2 ABBA/BAAB.
- A reproducible S2 regression prevents deployment; an inconclusive S2 result
  may only be retained default-off if local NCU is strictly better.
- Build, 26-row all-head smoke, disabled-path equality/fallback, then NCU and
  only then S2 throughput. If local NCU still loses, revert Stage60 entirely.

## Numerical risk

Rounding invRMS to FP16 adds one approximation before scaling. NaN/Inf,
policy/optimistic top-1 loss, or a material all-head error increase rejects the
candidate. Full 8,192-row accuracy remains mandatory before deployment.
