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

The stable optimized B4-B32 prescan independently selected B12, B13, and B14
for full search and recorded no foreign PID with nonzero SM activity. A normal
GTP startup loaded this plan on two NN-server lanes, observed all planned
post-launch activation markers on both lanes, started fixed-B12 dispatch and
the event-gated single-slot scheduler, and completed basic 19x19 commands.
A 64-thread search benchmark over ten 800-visit positions measured `3075.81`
visits/s, `2973.42` logical nnEvals/s, `255.46` launched batches/s, and an
average logical batch of `11.64`. The corresponding fixed-B12 physical rate is
`3065.52` nnEval/s, 1.45% below the compute-only long gate. No foreign PID used
nonzero SM time during the benchmark.

The recorded CUDA ordinal is provenance only. A receiver may use another
ordinal, but the loader still requires the exact RTX 4090 D product and CUDA
capability fingerprint, model/board/batch, FP16/NHWC, and two streams, and
fails closed on any mismatch.
