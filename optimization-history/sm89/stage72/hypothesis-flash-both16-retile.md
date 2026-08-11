# Stage 72 hypothesis: retile SM89 FlashAttention after both16 accumulation

## Frozen target and control

- GPU: RTX 4090 (SM89), device 0 only, protected by `gpu-lock`.
- Shape: exact B13/S361/H12/D32 FP16, 19x19, non-causal attention.
- Service topology: S2 is the deployment target; S1 is a local diagnostic.
- Control: deployed both16 FlashAttention, `M64 x N96`, four warps, one stage.
- Timing regime: locked SM clock with observed clock recorded; short natural
  execution boundaries are used for screening. NCU replay is used only to
  explain a single Flash kernel.

## Existing Nsys/NCU evidence

- The current eager S2 full-graph tail contains 1,980 Flash launches totaling
  52.507 ms, or 26.519 us per launch in the contended trace. The last complete
  union attribution before the deployment investigation assigned Flash 22.89%
  of busy-union exclusive time, behind dual-FFN and QKV.
- Stage 59 changed only accumulator precision. NCU measured 168 -> 117
  registers/thread, the register occupancy limit increased from 3 -> 4
  CTA/SM, eligible warps/cycle increased 0.348 -> 0.870, and duration improved
  28.224 -> 20.864 us without spilling.
- The Stage 7 tile sweep selected M64xN96 under FP32 accumulation. It predates
  the 30% register reduction and additional resident CTA, so its resource and
  wave tradeoff is no longer the same experiment.

## Mechanism and falsifiable prediction

At S=361, N96 executes four K/V loop tiles and processes 384 padded positions.
N128 executes three loop tiles and also processes exactly 384 padded positions.
It can therefore remove one online-softmax update, synchronization interval,
and loop-control step without adding Tensor Core padding work. Before both16,
the wider tile could lose through register pressure; after Stage 59 that risk is
materially lower.

The first focused candidates are:

1. M64xN128/W4: isolates the predicted K/V-loop reduction while retaining the
   accepted query tile and CTA count.
2. M128xN128/W4 only if M64xN128 is locally competitive: tests whether lower
   query-grid overhead becomes viable after the accumulator reduction.

Support requires the same-shape isolated natural-event time to improve beyond
run dispersion, no spill, and NCU to show that the loop reduction is not offset
by a worse occupancy/stall regime. A supported tile then advances to short S1
and S2 natural Nsys boundaries. It is deployed only after S2 improvement and
all-head accuracy validation.

## Risks and stop conditions

- Wider N can increase shared memory, live fragments, or barrier cost enough to
  lose despite one fewer loop.
- M128 halves the CTA grid and may expose a tail wave on 128 SMs.
- Cross-kernel NCU durations will not be added. Whole-network decisions use one
  natural execution trace/event boundary.
- If M64xN128 is clearly slower locally and NCU explains the loss, the stage
  stops without a full-graph re-profile; the clean eager profile remains the
  next-hotspot source.

## Result

Rejected without changing the KataGo source tree.

- Locked-2205 natural-event ABBA/BAAB screen (10,000 launches/sample):
  M64xN96 mean/median `20.8662/20.8597 us`; M64xN128
  `21.4539/21.4488 us`, or `+2.817%/+2.824%` slower. All four paired
  comparisons were slower by 2.70-2.99%.
- Three-launch NCU medians: approximately `21.50 -> 22.18 us`. N128 raised
  registers/thread `117 -> 168`, dynamic shared memory `16.77 -> 20.86 KiB`,
  reduced the register block limit `4 -> 3`, theoretical occupancy
  `33.33% -> 25.00%`, achieved occupancy `29.32% -> 22.28%`, and eligible
  warps/scheduler `0.87 -> 0.69`. Neither variant spilled.
- The predicted loop reduction is real, but the wider live score/output
  fragments recreate the old register bottleneck. M128xN128 is not attempted
  because the prerequisite M64xN128 candidate is already strictly worse.
- No S1/S2 full-network run, accuracy pass, full-graph Nsys/NCU, or code commit
  is warranted. Reuse the clean eager full-graph profile for the next hotspot.
