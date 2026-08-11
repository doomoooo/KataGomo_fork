# Stage 53 hypothesis: add the value-terminal fusion to the cumulative bundle

Target remains RTX 4090 exact 19x19, B13, FP16, S2. The deployed control is the
frozen Stage 50 binary/config. The Stage 52 candidate enables initial-global,
wide-head, and head-BN-to-FP32 fusions; all three are intrinsic-accepted but the
combined S2 result is phase-sensitive.

Stage 51's value/score terminal fusion is not intrinsically accepted: its S1
ABBA was throughput-neutral. It nevertheless qualifies for strict-local
accumulation because it is byte-identical on 8,192 rows, replaces four launches
with two, and cuts the complete terminal boundary by about 49% in both orders
without a stable S1 regression.

## Falsifiable test

Enable all four mechanisms together and compare against the deployed Stage 50
control with 20-forward, locked-2400-MHz S2 Nsys in forward and reverse order.
The candidate must remove eight launches per forward. Only if throughput is
positive in both orders does it proceed to one 100-forward S2 ABBA. Otherwise
the four implementations remain in the strict-local bundle, default-off, and
the deployed current best does not change.
