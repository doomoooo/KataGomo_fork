# Hypothesis H12: fixed-C1152 contiguous half8 SwiGLU

Created: 2026-08-05 (UTC), after profiling the accepted Stage11 baseline and
before implementation or measurement.

The accepted S2 trace reports the standalone SwiGLU at 45.471 ms across 4,422
launches (6.1% of summed kernel time). NCU on the B13 launch reports 16.32 us,
75.37% DRAM throughput, 44.17% SM throughput, 84.71% achieved occupancy,
19 registers/thread, and about 63.8% of issue distance in long-scoreboard
stalls. The official EPT4 kernel performs four separate half2 accesses per
input/output and retains four bounds predicates per thread.

For fixed 19x19, C1152, FP16, the element count is divisible by eight. H12
keeps eight scalar FP32 SiLU calculations per thread but accesses them as one
contiguous 128-bit half8 vector for each input and output, with no per-element
bounds predicates. Arithmetic expressions and FP16 conversion remain
unchanged.

Nsys must show a lower direct SwiGLU kernel total. Short A-B-B-A must improve a
target topology before full validation. A retained candidate requires at least
0.5% long whole-network gain and all full-FP32 8,192-row accuracy gates.
Unsupported shapes, channel counts, and precisions retain the official kernel.
