# RTX 4090 D B12/S2 CUDA tactic plan

This directory contains the current production-ready SM89 plan for the tested
RTX 4090 D. It is bound to exact 19x19 FP16/NHWC inference, model SHA-256
`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`, exact
batch 12, and two inference streams per device.

The coupling-audited search contains 19 implementation catalogs in 10 ordered
decision groups and closes all 60 retained positive-history records. Its long
whole-graph gate measured `3110.690824` physical nnEval/s from samples
`3110.484420` and `3110.897228` at 1000 timed iterations each. One 8192-row
all-head replay then passed the immutable full-FP32 aggregate and per-request
gates. The plan contains no mask search component; full 19x19 is a backend
invariant.

The recorded CUDA ordinal is provenance only. A receiver may use another
ordinal, but the loader still requires SM89, the exact model/board/batch,
FP16/NHWC, and two streams, and fails closed on any incompatible tactic.
