# Stage 50 hypothesis: eliminate exact-board mask preprocessing

Scope is exact 19x19, B13, S2 on RTX 4090. `requireExactNNLen` guarantees all
361 locations are valid, yet the current SM89 forward still extracts channel 0,
converts the mask to float, and sums it before discarding the mask pointers.

The post-Stage-49 full-graph Nsys sees one instance of each launch per forward;
their average durations are 1.457, 2.233, and 2.038 us. Broad NCU measures
2.080, 1.888, and 2.880 us, with only 0.145%, 0.162%, and 1.326% SM throughput.

The candidate uploads a B13 vector filled with 361 once during handle creation.
Exact forwards pass this constant mask sum to the unchanged policy/value pooling
kernels and allocate no mask scratch. Non-exact calls retain the original path.

Expected local result: all three launches disappear, reducing the target boundary
from 6.848 us in isolated NCU to zero. Acceptance still requires a consistent
full-graph S2 improvement; no long test is run unless the short paired evidence
is positive in both orders.
