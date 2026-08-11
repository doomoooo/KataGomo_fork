# Stage 67 hypothesis: M128xN64 stage-3 plain QKV

## Evidence

- Stage 65's native-half plain-QKV control uses M128xN128xK32, warp 64x64,
  stage 3, 168 registers/thread and 49.15 KiB shared memory. It is limited to
  two CTA/SM.
- Stage 66 proved that lowering to stage 2 is counterproductive: registers rose
  to 206, occupancy stayed 16.67%, and the natural boundary regressed 12.55%.
- N=384 is exactly divisible by both 128 and 64. An N64 tile introduces no
  padded MMA. It doubles grid size from 333 to 666 CTA but halves each CTA's N
  accumulator and B shared tile.

## Falsifiable mechanism

Variant 2 keeps stage 3, FP16 accumulator, native-half epilogue, K32, M128, and
the batched Q/K/V layout, changing only threadblock N128->N64 and warp
64x64->64x32. It should reduce both registers and shared memory enough for three
CTA/SM. More CTA and repeated A loading may offset that benefit.

Gates:

1. Variant 1/2 same-binary 26-row output must be byte-identical.
2. NCU must verify no padding/spill, lower resources, and the intended occupancy
   tier. A slower kernel with no concurrency mechanism is rejected.
3. Natural S1 Nsys directly measures QKV launch through RoPE completion.
4. Any credible local/resource win proceeds immediately to short natural S2
   Nsys, because multi-stream contention is the purpose of the tile.
5. Stable S2 gain is required for deployment; a strict local-only win may be
   retained default-off under the split-path workflow.
