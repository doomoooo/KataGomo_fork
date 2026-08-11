# H23: Fixed-B13 single-stream composition may approach two-stream throughput

## Frozen protocol

- GPU: RTX 5090 D, CUDA device 2, exclusively held by `gpu-lock`.
- Model: `/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz`.
- Shape: fixed 19x19, batch 13 only.
- Backend/precision: CUDA SM120, FP16.
- S1 topology: one NN server thread and one CUDA stream.
- S2 reference: two NN server threads on the same GPU, current accepted kernel set.
- Initial screen: 100 timed iterations after 20 warmup iterations.
- Confirmation: controlled ABBA runs on the best S1 candidate and its nearest control.

## Evidence

- Under S2, the isolated out-projection residual kernel improved dual-stream micro latency
  from 30.545 us to 18.243 us, yet whole-network throughput fell from about 3830 to
  3718 nnEval/s.
- Several earlier candidates showed the same sign reversal between S1/local measurements
  and S2 whole-network throughput.
- The current accepted S2 topology therefore includes a material scheduling/phase-control
  term that is absent under S1.

## Mechanism

With one stream, a locally faster kernel shortens the serial critical path directly. It
cannot delay a peer stream by changing CTA residency, launch phase, or shared-resource
competition. Fixed-shape AOT kernels may therefore compose more monotonically in S1,
including kernels rejected under S2 for interference rather than serial latency.

## Falsifiable prediction

1. S1 will prefer at least one kernel choice that S2 rejects, especially the AOT
   out-projection residual epilogue.
2. The best S1 bundle will be materially faster than the historical S1-specific bundle.
3. If the best confirmed S1 result reaches at least 95% of the current confirmed S2
   throughput, S1 becomes a credible primary topology because it removes phase-control
   sensitivity. Below 90%, topology alone leaves too large a gap; further serial-kernel
   work is required before reconsidering S1 as the primary topology.

## Variables and ablations

Screen these independently at fixed B13:

- fused FFN AOT versus historical wide single-GEMM FFN;
- wide QKV AOT versus historical strided-batched QKV;
- linear2 residual AOT on/off;
- out-projection residual AOT on/off.

The best bundle must be followed by leave-one-out ablations so its gain is not attributed
to an untested interaction.

## Risks and validation

- Short runs may be affected by thermal or clock drift: confirm with ABBA.
- Kernel-local wins may still alter library autotactics or cache state: inspect Nsys on
  the best S1 candidate.
- Any accepted new arithmetic path must pass the fixed 8192-sample all-head comparison
  against the FP32 reference before acceptance.
