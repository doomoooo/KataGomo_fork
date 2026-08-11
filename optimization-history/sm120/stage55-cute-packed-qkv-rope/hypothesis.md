# Stage 55 hypothesis: packed CuTe QKV epilogue + precomputed RoPE

## Frozen control and opportunity

- Control source: commit `3526b13`; accepted/default performance graph remains
  Stage 47 (`acf588c`), exact 19x19 B13, FP16/NHWC, natural S2 on RTX 5090 D.
- Packed CuTe QKV is 13.92% of Stage-47 S2 work; the immediately following
  packed Q/K RoPE is 4.98%. Together they account for 1050.2 us/forward and
  391.6 us/forward of S2 interference excess.
- Current QKV uses 107 registers/thread, 99.3 KiB shared memory, 340 CTAs and
  1.00 waves/SM. Current packed RoPE uses 28 registers, no shared memory and
  costs about 5.6 us S1 per boundary.

## Distinction from the rejected historical AOT

The 5080 QKV+RoPE AOT lost 1.038% and the earlier 5090D audit correctly banned
repeating it unchanged. That implementation paid repeated trigonometric work
and did not use the current packed CuTe mainloop/epilogue. This stage is allowed
only because it tests the audit's explicit reopening condition: a low-live-
range epilogue preserving the current packed layout without materially raising
registers.

For fixed 19x19, learnable RoPE frequencies and board coordinates are constant
for a model. Precompute the 361x192 FP32 `(cos,sin)` table once with the same
CUDA `__sincosf` operation. In the existing CuTe 64x32 epilogue tile:

1. convert accumulator values to FP16 and store them to the existing epilogue
   shared-memory stage, preserving the current QKV rounding boundary;
2. use the 256 MMA threads to rotate the 1024 Q/K half2 pairs in-place, four
   pairs per thread, with short scalar live ranges;
3. issue the existing TMA store. V tiles are untouched.

No new per-forward kernel, mainloop stage, output buffer, or dynamic shared
memory is introduced.

## Gates

1. Independent boundary probe must be byte-identical to accepted packed QKV +
   packed half2 RoPE before graph integration.
2. Nsys must show the standalone RoPE launch removed and a smaller complete
   QKV-to-RoPE boundary.
3. NCU decides whether the epilogue meets the low-live-range premise. A large
   register increase, spill, shared-memory increase, or slower complete S1
   boundary rejects the implementation before graph integration.
4. Natural S2 both-order real graph decides default acceptance. A correct
   both-order S1/resource-positive result may be retained default-off.
5. Only an accepted source change triggers full accuracy, fresh full S2 Nsys,
   complete S1 NCU and one commit.
