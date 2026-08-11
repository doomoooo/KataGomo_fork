# Stage 48 hypothesis: fold RMSNorm into QKV and FFN projections

## Frozen target

- RTX 4090 SM89, exact 19x19, B13, FP16, S2 only.
- Frozen accepted control binary:
  `results/4090/stage46/binaries/control-katago`, SHA256
  `2b0718dab575bcc80aad816bf60b093d2dcfcd8ca90dbfb69dfc2d4e8ff73b31`.
- Attention-only and FFN-only switches are required so a combined result can be
  disambiguated. S1 is not a performance gate.

## Pre-implementation evidence

This hypothesis is provisional until the Stage 48 current-best full-graph Nsys
and broad NCU checkpoint confirms that no larger actionable boundary has moved
ahead of it.

The accepted S2 Nsys trace contains 4,356 RMSNorm launches (66 per complete
forward across 66 forwards), averaging 5.614 us and summing to 24.456 ms. On
the two timed streams, RMSNorm covers 23.704 ms of busy union and is the only
active family for 3.657 ms (1.39% of the complete busy union).

Accepted S2 NCU measures 4.61 us median, 40 registers/thread, no spills, about
49% L2 throughput, 98% L2 hit, negligible DRAM traffic, and about 66.5%
no-eligible cycles. The kernel reads and retains twelve FP32 values per lane,
reduces the row, then writes the entire 4693x384 half tensor. A reduction-only
variant can discard those values after the sum and write 4693 floats instead of
1,802,112 half values.

The older SM120/B19 history measured FFN-only folding at -0.016% and QKV-only
at -0.353%. Those results prevent assuming a win but do not substitute for the
current SM89/B13 kernels, topology, and epilogues.

## Candidate mechanism

1. Pre-fold each block's FP32 RMS gamma into its QKV or dual-FFN host weights,
   then convert the folded weights to FP16 exactly once at model initialization.
2. Replace the full normalized-tensor kernel with a warp-per-row reduction that
   stores one FP32 invRMS scalar per token.
3. Feed the original residual tensor to the projection GEMM.
4. In QKV output coordinates, multiply the projected half values by the row's
   invRMS before Q/K RoPE.
5. In dual-FFN output coordinates, multiply both projections by invRMS before
   SiLU and gate multiplication.
6. Preserve the original path as fallback for masks, shapes, precision modes,
   unsupported kernels, and disabled options.

This intentionally changes FP16 rounding: gamma is rounded as part of the
folded weight and row scaling moves after tensor-core accumulation. It is an
accuracy-gated optimization, not a bit-exact transform.

## Falsifiable gates

1. Build and a 26-row S2 replay must be finite, retain policy/optimistic top-1,
   and stay inside the predeclared smoke error envelope.
2. NCU, 2-3 launches isolated from S2, must show a materially smaller reduction
   kernel and no spill/occupancy regression in QKV or dual-FFN. The complete
   `reduce + projection` boundary must improve by at least 3%.
3. Run attention-only, FFN-only, and combined short S2 Nsys in both binary
   orders, 20 timed iterations. Record kernel counts, family sums, busy union,
   and throughput. Any component that regresses in both orders is disabled.
4. Only a stable positive component/combination enters one locked 100-iteration
   S2 ABBA.
5. Only after the performance gates, run all 8,192 positions against the FP32
   reference and enforce every output-head threshold.
