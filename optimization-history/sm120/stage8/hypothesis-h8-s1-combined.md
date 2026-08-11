# Hypothesis H8: combined S1 projection optimizations

Created: 2026-08-05 (UTC), before combined measurement. Target is RTX 5090D,
fixed 19x19, B13, one server stream, FP16.

Stage 6 independently accepted the single-wide FFN projection (+1.035% long
S1 mean) and Stage 7 independently accepted strided-batched QKV (+1.405%). Both
passed full accuracy and affect disjoint projection subgraphs. Their
percentages must not be arithmetically chained because clock and scheduling
interactions can overlap.

The combined candidate enables only those two accepted switches. It must pass
the same full-FP32 8,192-row gates and show at least 1.5% whole-network gain in
a symmetric 1,000-iteration S1 A/B. S2 is measured as a rejection/control
boundary and is not expected to improve because both independent S2 tests
regressed.
