# Fixed B13 SM89 dependencies

The optional SM89 path is compiled only when `SM89_FLASH_ATTN_ROOT` points to
the pinned Dao-AILab/flash-attention checkout below. It is deliberately AOT
specialized for FP16 `[B,S,H,D] = [13,361,12,32]` and falls back to cuDNN for
every other shape.

- FlashAttention commit: `5835c733e7e9c07606b045255768e8a7e9e851bd`
- CUTLASS submodule: `7127592069c2fe01b041e174ba4345ef9b279671`
- License: BSD-3-Clause, reproduced in `flash-attention-sm89.LICENSE`
- AOT tile: `M64 x N96`, four warps, native D32, `PackGQA=false`, SM89
- Dual GEMM + SwiGLU tile: `M128 x N64 x K32`, warp `M64 x N32 x K32`, three
  stages, swizzle 2, SM89; only `[M,N,K] = [4693,1152,384]`
- Linear2 residual GEMM tile: `M128 x N128 x K32`, warp `M64 x N64 x K32`,
  four stages, swizzle 1, `alpha=1`, `beta=1`, SM89; only
  `[M,N,K] = [4693,384,1152]`
- Nested preConv GEMM tile: `M128 x N128 x K32`, warp `M64 x N64 x K32`,
  five stages, swizzle 1, `alpha=1`, `beta=0`, SM89; only
  `[M,N,K] = [4693,384,768]`
- Fused learnable RoPE rotates Q and K in one SM89 launch for the accepted
  FP16 `[B,S,H,D] = [13,361,12,32]` path. It replaces 66 scalar Q/K launches
  with 33 fused launches per forward.
- QKV+RoPE uses a fixed CUTLASS batched GEMM at
  `[M,N,K,batch] = [4693,384,384,3]`, tile `M128 x N128 x K32`, warp
  `M64 x N64 x K32`, and three stages. Its output iterator applies learnable
  RoPE to Q/K after FP16 rounding and writes V unchanged. It recomputes angles
  from the 1.5KiB frequency vector, so it does not add the rejected large
  model-lifetime cos/sin tables. Other shapes fall back to batched QKV plus
  fused RoPE.
- Experimental out-projection residual GEMM uses the same tile at
  `[M,N,K] = [4693,384,384]`. Its isolated S2 boundary is faster, but fixed
  B13/S2 whole-network ABBA did not stabilize, so its runtime switch remains
  disabled.
- Experimental nested postConv residual GEMM uses a three-stage version of the
  same tile at `[M,N,K] = [4693,768,384]`. Its isolated S2 boundary is faster,
  but three locked-clock whole-network ABBA rounds were consistently slower,
  so its runtime switch remains disabled.
- Experimental precomputed Q/K RoPE stores model-lifetime float2 cos/sin tables.
  Its RoPE kernel is slightly faster and byte-exact, but paired ABBA and Nsys
  critical-path evidence did not support a whole-network gain, so
  `cudaUsePrecomputedQKRoPESm89` remains disabled.
- Experimental batch-grouped Q/K RoPE shares one `sincos` across multiple B13
  rows. Group 2 improves the isolated kernel by about 4.6%, but the whole-network
  signal is below stable resolution; `cudaRoPEBatchGroupSm89` remains 1.

Prepare the dependency from the KataGo worktree:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git /workspace/third_party/flash-attention
git -C /workspace/third_party/flash-attention checkout 5835c733e7e9c07606b045255768e8a7e9e851bd
git -C /workspace/third_party/flash-attention submodule update --init csrc/cutlass
git -C /workspace/third_party/flash-attention apply cpp/neuralnet/flash-attention-sm89.patch
cmake -S cpp -B build-cuda -DSM89_FLASH_ATTN_ROOT=/workspace/third_party/flash-attention
```

`SM89_DUAL_GEMM_SWIZZLE` is a compile-time experiment knob and defaults to the
accepted value 2.

CMake checks both commits and the FlashAttention tile patch. The runtime switches are
`cudaUseFlashAttentionSm89`, `cudaUseDualGemmSwiGLUSm89`,
`cudaUseLinear2GemmSm89`, `cudaUsePreConvGemmSm89`, `cudaUseFusedQKRoPE`, and
`cudaUseQKVRoPEGemmSm89`;
all default off in the SM89 code so the same executable can run controlled A/B tests. The rejected experimental
out-projection path is exposed as `cudaUseOutProjGemmSm89` and also defaults off.
The rejected postConv path is exposed as `cudaUsePostConvGemmSm89` and defaults
off as well.
