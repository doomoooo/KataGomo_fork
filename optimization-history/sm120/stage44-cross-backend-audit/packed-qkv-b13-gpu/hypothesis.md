# H44-PQ: exact-B13 packed-QKV GPU gate

## Frozen target and control

- GPU: RTX 5090 D, CUDA runtime ordinal 2, NVML UUID
  `GPU-a05bf432-3f6b-83c1-98b1-aec8f8a6fd69`.
- Graph: exact 19x19, B13, FP16, natural two-server S2.
- Performance control: Stage 38 accepted graph. Stage 39 through Stage 43 did
  not change the accepted graph.
- Candidate binary: `/workspace/build-packed-qkv-b1b32/katago`, SHA256
  `cf3ff16a2ad3f23c30a4dd67e71fadeb95947faca1aa6e7d84eed3ee565a175e`.
- Candidate source: `/workspace/katago-packed-qkv-b1b32` at `01f140c`.

The linked search object is inert when the requested tactic is `auto`, so the
same binary is used for control and candidate. This avoids a cross-build A/B.
Both modes retain the accepted persisting-L2 windows and every other Stage 38
setting.

## Existing profiler evidence

Stage 38 attributes `785.1 us/forward` (13.77% of S2 work) and
`149.6 us/forward` of interference excess to QKV projection. The accepted
planar QKV kernel uses 136 registers, 65.5 KiB dynamic shared memory, 1.96
waves/SM, 8.3% occupancy, and only 6.9% eligible cycles. QK RoPE adds
`177.9 us/forward` and FA4 adds `592.0 us/forward`, so the complete layout
change must be judged as the QKV -> RoPE -> FA4 boundary rather than as a GEMM
alone.

## Candidate and falsifiable mechanism

Request `qkv-m128-n128-k64-s2-cute-atom4x2-packed`. It computes the same
M=4693, N=1152, K=384 projection, but writes row-packed `[token,Q,K,V]` output.
The following packed half2 RoPE and dynamically-strided FA4 consume that layout
without a reformat kernel. The hypothesis is useful only if the complete
boundary reduces isolated work or improves hard resources without moving more
cost into RoPE/FA4.

This is not an accepted optimization. Stage 44 so far contains only CPU-side
generation, linking, and address-contract checks.

## Gates

1. Runtime smoke must select the requested packed tactic, packed RoPE, and FA4
   on CUDA ordinal 2 without fallback, launch error, NaN, or Inf.
2. A short replay must establish numerical viability before performance work.
3. S1 Nsys plus targeted NCU must compare the whole QKV/RoPE/FA4 boundary.
   Kernel durations are never added across NCU captures. Reject on a coherent
   boundary slowdown or spills/resource blow-up with no compensating boundary
   gain.
4. If the mechanism survives, run short symmetric natural-S2 A/B against
   `auto` on the same binary. The portable tactic runner is not authoritative
   here because it disables the accepted L2 windows.
5. Only a positive natural-S2 signal is promoted to long A/B, full 8,192-row
   all-head accuracy, fresh full-graph S2 Nsys and matching 344-ordinal S1 NCU,
   history update, and one dedicated commit.

