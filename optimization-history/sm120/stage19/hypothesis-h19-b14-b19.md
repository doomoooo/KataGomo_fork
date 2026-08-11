# Hypothesis H19: remove the artificial B13 topology ceiling (out of scope)

Status: stopped and excluded from optimization decisions. The target is fixed
19x19, B13, S2; B14-B19 results are retained only as an experimental record.

Created: 2026-08-06 UTC, before B14-B19 measurement.

The final Stage-17 scan stopped at B13 because the benchmark configuration and
AOT build example used B13. The FA4 C ABI carries runtime tensor shapes and the
build script states that batch is runtime. The 5080 retained path was tuned at
B19, so B14-B19 must be measured rather than excluded by assumption.

Run fixed 19x19 FP16 S1 and S2 at B14-B19 using the current accepted kernels.
First smoke B19 and verify the FA4 dispatch log, launch correctness, and finite
JSON output. Then scan all six batches. No code or arithmetic changes are made,
so accuracy is unchanged. If a batch exceeds B13, confirm it with 1000-iteration
runs before changing the recommended configuration. A crash, AOT shape error,
or lower throughput rejects only the larger-batch topology, not the kernels.
