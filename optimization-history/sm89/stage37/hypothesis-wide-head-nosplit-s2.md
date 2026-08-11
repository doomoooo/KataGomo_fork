# Stage 37 hypothesis: no-split wide head projection under S2

## Frozen target

- RTX 4090 (SM89), exact 19x19, B13, FP16, S2 only.
- The accepted post-Stage-36 source and
  `/workspace/bench-cuda-gpu0-4090-s2.cfg` are the control.
- The only candidate variable is `cudaUseWideHeadProjection=true`.
- `numNNServerThreadsPerModel` remains 2 in every measurement. No S1 result is
  used as an end-to-end acceptance result.

## Pre-candidate profiler evidence

The current Stage35 control S2 Nsys trace still executes two
`ampere_h16816gemm_128x64` C768-to-C96 projections at 12.772 us average and one
`ampere_h1688gemm_256x64` C768-to-C192 projection at 20.559 us average. They
sum to about 46.103 us and three launches per logical head forward.

NCU collected before the implementation on these exact B13/19x19 projection
shapes measured 9.60/9.50/16.064 us in isolation. The C96 launches have only
0.58 waves/SM and the C192 launch only 0.29 waves/SM, all at about 8.3%
achieved occupancy and with no spill. This validates the mechanism: the three
narrow projections are launch-wave limited, not a saturated throughput path.

The already implemented, default-off candidate concatenates the three weights
as `[p1:96, g1:96, v1:192]`, runs one fixed C768-to-C384 Stage11 AOT GEMM, and
lets the first consumers read row-strided slices without materializing splits.
Prior candidate NCU measured that C384 GEMM at 15.94 us with 162 registers,
0.87 waves/SM, 8.32% occupancy, and zero spill. Stage37 independently tests
whether this locally valid mechanism improves the two-stream graph.

## Falsifiable S2 gates

1. The 26-row S2 replay must be byte-identical to control.
2. Two current S2 NCU samples of the candidate wide projection must retain the
   expected resource class, remain spill-free, and stay below 20 us median.
3. Run locked-2400, 20-iteration S2 Nsys in forward and reverse order. Both
   orders must remove two projection launches per forward, reduce the complete
   projection/first-consumer boundary, and improve end-to-end throughput.
   Summed kernel time and GPU busy union must also be reported.
4. Only if all short gates pass, run one locked-2400 100-iteration S2 ABBA.
5. Only if the ABBA gate is positive, run the full 8192-row FP32-reference
   replay and enable the option in the S2 configuration.

Any failed gate rejects the S2 enablement. The implementation remains
default-off because it is independently accepted historical code, but S2 will
not enable it without passing this sequence.

## Result

Rejected for S2; the default-off implementation is retained and the S2 config
remains disabled.

- The 26-row S2 smoke replay was byte-identical to control (SHA-256
  `00f4110d9b2770fc942f8ac88e1ecea6e17519dc99994bc6ca2f73233396be97`).
- Current S2 candidate NCU measured 16.00/16.10 us, 111 CTA, 162
  registers/thread, 81.92 KiB dynamic shared memory, 0.87 waves/SM, 8.32%
  achieved occupancy, and zero local/shared spill.
- Forward-order Nsys removed 132 launches over 66 captured forwards. The
  projection plus first-consumer boundary fell 53.506 -> 32.429 us (-39.39%),
  GPU busy union fell 271.948 -> 270.113 ms (-0.675%), and throughput improved
  3201.661 -> 3213.153 nnEval/s (+0.359%). Summed kernel time increased 0.386%.
- Reverse-order Nsys showed the same local mechanism: the boundary fell
  52.597 -> 30.862 us (-41.32%) and summed kernel time fell 1.032%. However,
  GPU busy union increased 269.390 -> 275.515 ms (+2.273%) and throughput fell
  3203.615 -> 3134.730 nnEval/s (-2.150%).

The local work and launch reduction are real, but the reverse order shows that
the changed head launch shape moves the two streams into a worse overlap phase.
The required two-order full-graph gate failed, so no ABBA or 8192-row replay was
run. Reopen for S2 after phase control is available, or with a head kernel shape
that preserves the incumbent overlap schedule.
