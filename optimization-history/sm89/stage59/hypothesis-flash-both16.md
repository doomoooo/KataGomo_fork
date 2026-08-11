# Stage 59 hypothesis: SM89 FlashAttention both16 accumulation

## Scope

- Target: RTX 4090, exact 19x19, B13, FP16 NHWC, two NN-server streams.
- Control: deployed Stage56/57 binary with `cudaUseFlashAttentionSm89=true` and
  FP32 QK/PV Tensor Core accumulation.
- Candidate: identical graph and `M64 x N96`, four-warp FlashAttention tile,
  changing only QK and PV Tensor Core accumulation to FP16. Online-softmax
  `row_max`, `row_sum`, exponentiation and LSE remain FP32.
- Runtime switch: `cudaUseFlashAttentionBoth16Sm89`, default false until the
  candidate passes the acceptance chain.

## Evidence

- Current SM89 Flash NCU/SASS: 168 registers/thread, 16,768 B dynamic shared
  memory, three resident CTAs/SM, 25% theoretical occupancy, about 70k cycles,
  and 96 `HMMA.16816.F32` instructions.
- TensorRT Myelin MHA for the same B13/S361/H12/D32 shape: 118
  registers/thread, about 12.3 KiB dynamic shared memory, four resident
  CTAs/SM, about 51k cycles, and 48 `HMMA.16816.F16` instructions.
- Stage56 S2 Nsys assigns FlashAttention 9.91% of busy-union exclusive time.
- The neighboring SM120 backend accepted FA4 both16 after all-head accuracy;
  its isolated attention improved from 16.3599 to 11.8000 us and its S2 mean
  throughput improved by 0.446%. This is supporting evidence, not a projected
  SM89 result.

## Mechanism and falsifiable prediction

Using the SM80 `F16F16F16F16` MMA atom for both QK and PV halves each MMA
accumulator register payload. The FA2 online-softmax object already stores its
row maxima and sums as float, so the candidate can preserve FP32 scalar
statistics while allowing the score and output MMA fragments to round to FP16.

The candidate is supported only if NCU/SASS confirms all of the following:

1. QK/PV instructions change from `HMMA.16816.F32` to
   `HMMA.16816.F16`.
2. Registers/thread and no-eligible cycles strictly decrease with no local or
   shared spill.
3. Isolated attention latency strictly decreases under the same clock and
   cache protocol.

The initial full-graph expectation is +1% to +3% S2 throughput from the
Stage56 3288.9709 nnEval/s baseline, but the SM120 result shows that overlap can
compress the realized gain below 1%. The measured S2 result, not this estimate,
controls deployment.

## Risks and gates

- Numerical: FP16 QK/PV accumulation changes attention outputs. Run smoke first,
  then the established 8,192-row all-head comparison against the frozen FP32
  reference. The user explicitly accepts the existing both16 loss envelope, but
  NaN/Inf or a regression outside the established gates is still a rejection.
- Performance: fewer registers do not guarantee a fourth resident CTA or lower
  dependency stalls; a local non-winner is rejected before a whole-graph run.
- S2 scheduling: a strict local/NCU improvement may be retained default-off
  even if a short S2 run is inconclusive. A clear S2 regression prevents
  deployment.
- Shape and backend: only exact FP16 B13/S361/H12/D32 SM89 dispatch changes;
  all fallback paths remain unchanged.

## Validation order

1. Build and B13 smoke/fallback check.
2. Short accuracy sanity check.
3. Isolated microbenchmark plus NCU/SASS against the frozen FP32-accum control.
4. If locally supported, short ordered S2 Nsys A/B.
5. If retained, full 8,192-row all-head accuracy, locked-clock S2 ABBA/BAAB,
   then post-change full-graph Nsys and broad NCU.

## Result

Accepted and deployed in commit `7d299d0`.

- S2-source NCU median: 28.224 -> 20.864 us (-26.08%). Registers/thread:
  168 -> 117; register occupancy limit: 3 -> 4 CTA/SM; eligible warps/cycle:
  0.348 -> 0.870. SASS changed from `HMMA.16816.F32` to
  `HMMA.16816.F16`, with no spill.
- Locked-2400 S2 ABBA+BAAB: mean 3274.600 -> 3417.873 nnEval/s
  (+4.375%); median +4.700%.
- The 8,192-row all-head comparison passed the accepted both16 envelope.
- Post-change Nsys: Flash raw time 71.786 -> 57.614 ms (-19.74%) and
  exclusive time 25.688 -> 14.148 ms (-44.92%). The next macro priority is
  FFN dual projection + SwiGLU, followed by QKV + RoPE.

See `final-decision-summary.json` and
`post-stage59-current-best-checkpoint.md` for the frozen measurements.
