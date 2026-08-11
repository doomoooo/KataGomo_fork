# Stage 50 hypothesis: retained postConv fusion on the Stage 47 graph

## Frozen target

- GPU: RTX 5090 D, CUDA device 2 under `gpu-lock`.
- Shape/topology: exact 19x19, B13, FP16/NHWC, two NN servers and two natural
  CUDA streams.
- Control: clean commit `acf588c`, Stage 47 CuTe packed-QKV + fused-FFN graph,
  `cudaUsePostConvBNSiluSm120=false`.
- Candidate: the identical binary and configuration with only
  `cudaUsePostConvBNSiluSm120=true`.
- `CUDA_DEVICE_MAX_CONNECTIONS` is unset.

## Existing mechanism evidence

The retained Stage 45 implementation is unchanged in the current source.  It
fuses each of 11 outer C384-to-C768 residual projections with the immediately
following C768 affine-SiLU and preserves the rounded FP16 residual output.

Existing targeted NCU and correctness evidence remains valid because Stage 47
changed only the QKV and FFN runtime kernels:

- registers/thread: accepted library GEMM 154, fused candidate 108;
- dynamic shared memory: 73.728 KiB to 50.176 KiB;
- zero spills;
- 11 standalone affine-SiLU launches and 11 full C768 reads are removed;
- Stage 45 S1: `3267.010 -> 3272.912 nnEval/s` (`+0.181%`);
- full 8,192-row replay was byte-identical.

The old natural-S2 test was negative by 0.438% on the Stage 44 graph.  Stage 47
subsequently replaced both QKV and fused FFN kernels and changed the natural
stream phase/interference pattern.  Under the revised workflow, this retained
S1-positive and resource-strict candidate merits one current-graph retest.

## Falsifiable gate

1. Rebuild cleanly from `acf588c` before testing; do not use the Stage 49
   experimental linear2 object.
2. Run natural-S2 ABBA and reverse BAAB with 100 timed iterations and 20
   warmups per arm under one GPU lock.
3. Promote only if the pooled mean is positive and both order aggregates are
   non-negative.  A sign conflict closes this current-graph retest.
4. If positive, confirm with 400 timed iterations, then run the full accuracy
   replay and a fresh accepted whole-graph S2 Nsys plus complete 344-ordinal S1
   NCU before enabling the default and committing.
5. If rejected, reuse the Stage 47 accepted full profile and proceed directly
   to exact-19 mask preprocessing elision.
