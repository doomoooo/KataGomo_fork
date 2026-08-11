# Hypothesis H4: specialize the FA4 tail mask for S361

Created: 2026-08-05 UTC, before implementation. The accepted Stage-3 both16
kernel is the control. Scope is deliberately limited to the existing 19x19
dispatch: B1-B13, S361, H12, D32, FP16, non-causal.

## Evidence

The C++ dispatch already rejects every sequence length except 361. Inside the
AOT kernel, flash-attn still computes the tail-mask limit from dynamic
`seqlen_k`, current `n_block`, tile size, and per-thread column offset. For
S361 and tile N=128, masking is needed only on K tile 2 and the fixed valid
column count is `361 - 2*128 = 105`.

Stage-3 NCU reports 7,328,880 executed instructions and 10,920 branch
instructions for the both16 kernel. The tail-mask arithmetic is small relative
to MMA, so this is a low-upside experiment, but it directly matches the fixed
production shape and removes runtime work that cannot vary.

## Mechanism and prediction

Replace only `AttentionMask.apply_mask` in this AOT specialization with a
non-causal S361 implementation. When `mask_seqlen=true`, it applies the same
R2P bit mask using the constant tail limit 105; when false, it does nothing.
The AOT entry remains gated by `seqLen == 361`, and all other shapes keep the
official fallback.

Expected signal:

- output is byte-identical to the accepted generic-mask both16 object;
- standalone B13 kernel median improves by at least 0.5% or 0.05 us;
- PTX/SASS or NCU shows fewer integer/control instructions;
- if the isolated signal passes, B13/S2 whole-network throughput is retested.

## Risks and rejection rule

An incorrect fixed limit would leak padded K/V values into softmax. Therefore
the standalone attention smoke and full 8,192-row replay must be byte-identical
to the accepted both16 result, not merely within the broader FP32 gates.

Reject if it crashes, differs numerically, or does not meet the isolated
performance signal. A no-signal result is retained in history and is not sent
to whole-network A/B.
