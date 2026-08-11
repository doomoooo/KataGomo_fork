# RTX 5080 B16/S2 CUDA tactic plan

This directory contains the current production-ready SM120 plan for the tested
RTX 5080. It is bound to exact 19x19 FP16/NHWC inference, model SHA-256
`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`, exact
batch 16, and two inference streams per device.

The coupling-audited search contains 19 implementation catalogs in 10 ordered
decision groups and closes all 63 retained SM120 positive-history records. The
stable optimized B4-B32 prescan selected B12, B14, and B16 for full search.
B16 won the stable long whole-graph gate at `2836.211933` physical nnEval/s,
from samples `2835.319044` and `2837.104822` at 1000 timed iterations each.
Every measurement recorded an empty foreign-SM PID set. The separately
requested B19 full search reached `2832.459599`, below the B16 result.

One 8192-row all-head replay then passed the immutable full-FP32 aggregate and
per-request gates. The plan has complete backend, scan-candidate, post-launch
activation, and plan-apply closure. It contains no mask component; full 19x19
execution is a fail-closed backend invariant.

A normal GTP startup loaded the plan on both NN-server lanes, observed every
selected post-launch marker on both lanes, enabled fixed-B16 batch dispatch,
started the event-gated single-slot scheduler, and completed a 19x19 `genmove`.

The recorded CUDA ordinal is provenance only. A receiver may use another
ordinal, but the loader still requires SM120, the exact model/board/batch,
FP16/NHWC, and two streams, and fails closed on any incompatible tactic.
