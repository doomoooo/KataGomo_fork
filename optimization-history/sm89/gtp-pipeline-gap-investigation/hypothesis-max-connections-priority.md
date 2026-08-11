# Hypotheses: Hyper-Q connections and asymmetric stream priority

Date: 2026-08-06 UTC

Target: RTX 4090 (GPU 0), exact 19x19, FP16 NHWC, B13, two CUDA compute
streams. Core and memory clocks are locked to 2205 and 10501 MHz.

## H1: `CUDA_DEVICE_MAX_CONNECTIONS`

Evidence: S2 throughput depends strongly on the relative phase of two streams.
`CUDA_DEVICE_MAX_CONNECTIONS` controls the number of CUDA work queues available
to the context, so changing it can alter how the two streams are mapped and
interleaved. It does not control the number of host-enqueued kernels per stream.

Prediction: if insufficient or unfavorable Hyper-Q queue mapping is a material
bottleneck, the common-wall B13 physical row rate will show a repeatable trend or
an interior optimum over 1, 2, 4, 8, 16, and 32 connections.

Short screen: accepted eager event pipeline, queue depth 1, 80 timed launches per
lane, 10 warmups, ascending then descending order. Only a repeatable gain larger
than run-to-run phase noise proceeds to GTP ABBA and Nsys.

Falsification: no repeatable trend, or all pair means remain within ordinary S2
phase variation. In that case the problem is not described as kernel launch-depth
starvation.

## H2: asymmetric CUDA stream priority

Evidence: CUDA Graph fixes a worse two-stream phase than eager submission. Its
profile has more simultaneous overlap and slower resource-contending hot kernels.
Giving exactly one compute stream a slightly higher priority may break symmetric
CTA scheduling and keep the pair out of the unfavorable fixed phase.

Prediction: one-lane priority reduces the graph penalty, changes simultaneous
overlap/union and hot-kernel duration in the predicted direction, without adding
kernel-union idle gaps. The leader lane must be swapped to distinguish a real
priority effect from lane/start-order bias.

Risk: priority may starve the lower-priority stream, increase unfairness, or make
phase locking worse. CUDA priority is advisory and only affects pending work; it
does not preempt an already running CTA.

Validation sequence: query the supported priority range; short common-wall graph
and eager screens with lane 0 versus lane 1 favored; Nsys only for a candidate
with a repeatable signal; then exact GTP ABBA and full-output replay before any
production recommendation.
