# Stage 47 hypothesis: issue the dual-FFN next-stage copy before MMA

## Frozen scope

- RTX 4090 SM89 CUDA FP16, exact 19x19 B13, S2 only.
- Target is one CUTLASS dual-GEMM SwiGLU launch isolated from an S2 forward.
- Frozen control binary is Stage 46 `binaries/control-katago`, SHA256
  `2b0718dab575bcc80aad816bf60b093d2dcfcd8ca90dbfb69dfc2d4e8ff73b31`.
- Accepted control NCU median is 41.28 us at locked 2400 MHz.

## Evidence and mechanism

Stage 46 SourceCounters attributes 802 of the mainloop barrier's 816 samples to
long scoreboard. Control SASS has 64 HMMAs and 8 mainloop LDGSTS instructions.
The first mainloop LDGSTS appears only after 9 HMMAs, the last after about 20,
and the wait/barrier after 22. The copies therefore have little issue-to-wait
distance even though independent tensor-core work exists.

For `kWarpGemmIterations == 2`, issue the same two copy groups and the same fence
after the look-ahead shared load but before group 0 MMA. Keep the original
wait/barrier, stage advancement, arithmetic, tile, stage count, cache policy,
launch geometry, and all non-two-kgroup paths unchanged.

Expected machine effect: the first mainloop LDGSTS moves from after 9 HMMAs to
no later than after 2, while 64 HMMA, 24 total LDGSTS and 19 barriers remain.
This gives the existing 22 pre-barrier HMMAs more opportunity to hide async-copy
latency. Risk is longer iterator/predicate live ranges, register growth, or
competition between LDGSTS issue and tensor-pipe scheduling.

## Gates

1. SASS must show the expected earlier LDGSTS issue and unchanged instruction
   class counts. Otherwise reject statically.
2. The 26-row S2 replay must be byte-identical to control.
3. Three target launches from S2 must have median <= 40.04 us (>=3% faster), with
   no register/shared-memory/occupancy regression.
4. A one-launch SourceCounters profile must reduce mainloop barrier long-scoreboard
   and/or no-eligible cycles in the predicted direction.
5. Only then run forward/reverse locked S2 Nsys with 20 timed iterations.
6. Only stable positive Nsys results enter one 100-iteration S2 ABBA, followed by
   the 8,192-row FP32 accuracy gate.

## Result

The source move compiled, but SASS did not realize the predicted issue timing.
Control/candidate both retain 64 HMMA, 24 LDGSTS and 19 barriers. The first
mainloop LDGSTS remains after 9 HMMAs; the last changes only from after 18 to
after 17, and the barrier from after 22 to after 21. This is normal `ptxas`
scheduling variation rather than the proposed full-copy head start. Stage 47 is
rejected at the static gate and fully reverted without smoke, NCU, Nsys, ABBA,
or full accuracy.
