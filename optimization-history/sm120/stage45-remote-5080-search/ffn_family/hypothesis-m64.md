# H45-FFN-M64: fixed-B13 SM120 FFN M-tile retune

- Target: RTX 5080, fixed B13, exact 19x19, FP16/NHWC, natural S2 graph.
- Control: accepted `M128/N64/K32/S2` fused dual-GEMM + SwiGLU with
  A-fragment reuse.
- Single variable: use `M64/N64/K32/S2`; preserve K traversal, FP16 MMA and
  SwiGLU arithmetic, output layout, N tile, stage count, thread count and
  A-fragment reuse.
- Evidence: interleaved S1 is `59.971 -> 58.030 us` (`-3.238%`), all
  5,406,336 output FP16 words compare bit-exact; ptxas is `136 -> 100`
  registers/thread, dynamic shared memory `32 -> 24 KiB`, zero spills.
  (The exact output word count is taken from the stored JSON shape rather than
  used as a gate.)
- Expected mechanism: lower per-CTA register/shared-memory pressure and shorter
  M tiles improve schedulability on the 84-SM 5080.
- Main risk: grid size doubles from 666 to 1332 CTAs, which can add scheduling
  overhead or worsen natural two-stream phasing despite better per-CTA
  resources.
- Validation: compile/smoke, same-mode S1 NCU, then direct natural full-graph
  S2 forward/reverse short A/B. Do not run homogeneous or synthetic mixed S2.
- Acceptance: only a repeatable positive natural-S2 whole-graph result advances
  to long ABBA/BAAB, 8192-row full-output replay, fresh whole-graph S2/S1 Nsys
  and all-344-ordinal S1 NCU. Otherwise retain the artifacts and reject.
