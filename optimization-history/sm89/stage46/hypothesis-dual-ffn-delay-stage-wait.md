# Stage 46 hypothesis: delay the two-kgroup dual-FFN stage wait

## Frozen scope

- GPU/backend: RTX 4090 SM89 CUDA FP16.
- Shape/topology: exact 19x19, batch 13, two NN servers/two streams only.
- Target: CUTLASS dual-GEMM SwiGLU mainloop.
- Control binary: `binaries/control-katago`, SHA256
  `2b0718dab575bcc80aad816bf60b093d2dcfcd8ca90dbfb69dfc2d4e8ff73b31`.
- No S1 measurement is valid for this stage. A "single-kernel" result means a
  target launch isolated from an S2 invocation.

## Evidence before implementation

The accepted S2 Nsys profile has 2,178 target launches at 43.233 us average and
94.161 ms summed time, the largest summed kernel hotspot. The accepted S2 NCU
control has a 41.28 us median, 168 registers/thread, 49.15 KiB shared memory,
16.67% theoretical occupancy and about 78.3% no-eligible cycles.

Stage 46 SourceCounters on an S2 launch reports:

- 32.7% execution math-pipe stall;
- 571,464 excessive shared-memory wavefronts (14% of 3,957,152);
- mainloop `BAR.SYNC.DEFER_BLOCKING` PC: 816 samples, 802 long-scoreboard;
- prologue barrier PC: only 101 samples, 99 long-scoreboard.

For this instantiation `kWarpGemmIterations == 2`. The current schedule issues
both next-stage async-copy groups after warp K group 0, fences, then immediately
waits and synchronizes. Warp K group 1 therefore cannot hide the copy latency.
Its look-ahead shared-memory load targets the next stage, which is why the wait
cannot merely be moved without moving that load too.

## Falsifiable change

Only when `kWarpGemmIterations == 2`:

1. Keep issuing and fencing both next-stage async-copy groups after K group 0.
2. Run K group 1 tensor-core work while those copies are outstanding.
3. Wait/synchronize and advance circular-stage iterators after K group 1.
4. Load/transform the next stage's K group 0 only after the synchronization.

All tile shapes, stages, copies, arithmetic, epilogue, cache policies and launch
geometry remain unchanged. Other K-group counts retain the original schedule.

## Gates

1. Build and 26-row S2 replay must be byte-identical to the frozen control.
2. SASS must still contain 64 HMMA instructions, with the mainloop barrier moved
   after the second 32-HMMA group.
3. Three target launches isolated from S2 must have median <= 40.04 us (at least
   3% faster than 41.28 us), with lower barrier long-scoreboard/no-eligible signal.
4. Only then run forward/reverse S2 Nsys, 20 timed iterations per binary.
5. Only stable positive Nsys results justify one locked 100-iteration S2 ABBA.
6. The 8,192-row FP32 accuracy comparison runs only after the performance gates.

Failure at any gate means full revert and a recorded rejection.

## Implementation note

The first source-only revision compiled, but its SASS still placed the mainloop
barrier after exactly 22 HMMAs in both control and candidate (at 0x2490 and
0x2500 respectively). `ptxas` legally moved the register-only MMA work across
the source-level synchronization, so that revision did not realize the proposed
schedule and was not run. The second revision makes the wait and barrier consume
the tail word of `accum1` in one volatile PTX block. This is a scheduling
dependency only; it does not change the accumulator value. The same SASS gate
must prove that it actually delays the barrier before any dynamic measurement.

That second revision produced byte-identical target SASS to the first revision:
64 HMMA, 24 LDGSTS, 19 barriers, and still 22 HMMAs before the mainloop barrier.
The explicit self-move dependency was eliminated by `ptxas`. Stage 46 is
therefore rejected at the static gate without smoke, NCU, Nsys, ABBA, or full
accuracy. Reopening this exact route requires a non-removable machine-level
dependency whose added work can be justified independently.
