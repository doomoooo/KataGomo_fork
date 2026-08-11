# Stage 51 hypothesis: fuse value terminal projections

Scope is RTX 4090 exact 19x19, B13, FP16 trunk, FP32 heads, and S2 only.
The Stage 50 current-best full-graph trace shows the value head ending with two
independent projections from the same B13x384 `v2Out` activation:

1. `v3Mul` (384 -> 3), then an identity bias kernel.
2. `sv3Mul` (384 -> 6), then an identity bias kernel.

Across the two busy streams, the 64 complete observed head pairs contain 256
target launches. Their raw time is 0.677953 ms, or 10.593 us per head. Broad
NCU measures the common small-N SGEMM geometry at 3.520 us and the two bias
geometries at 1.824/1.792 us. Every target launch is at most 0.01 waves/SM and
below 0.20% SM throughput, so this is launch/low-wave limited rather than a
compute-throughput problem.

The candidate concatenates the 3-row and 6-row FP32 weight matrices once at
initialization, runs one 384 -> 9 SGEMM into a tiny B13x9 scratch buffer, and
uses one kernel to split the result into the existing value/score buffers while
adding their biases. This changes four launches to two without changing any
upstream head or trunk schedule.

Expected local result: about 10.6 us -> 5-6 us for the complete terminal
boundary and two fewer launches per forward per stream. Risks are a different
cuBLAS small-N tactic/accumulation order for N=9 and S2 phase movement. The
candidate must pass the 26-row smoke gate, targeted NCU, and paired full-graph
S2 Nsys in both orders before the single short 100-iteration ABBA. Full 8192
accuracy runs only after the performance gate passes.
