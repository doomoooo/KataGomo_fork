# Exact-B1..B32 packed-QKV CuTe integration (CPU-only phase)

## Result

The historical packed-QKV data contract is now implemented in an isolated
worktree and wired into the exact-19x19 natural whole-graph S2 search runner.
All B1 through B32 CuTe host objects were freshly generated from pinned source;
no historical `.o` was copied. The B13 variant was compiled and linked into a
complete KataGo CUDA binary without running a GPU.

- Isolated worktree: `/workspace/katago-packed-qkv-b1b32`
- Branch: `packed-qkv-b1b32-integration`
- Base commit: `a56308f40d343183f0dfbf2ab18131c7ebd2bb2e`
- Integration commit: `01f140c90d0be2e4ce770db2bddac5459a06e112`
- Candidate: `qkv-m128-n128-k64-s2-cute-atom4x2-packed`
- Generated artifacts: `b1/` through `b32/` in this directory
- Aggregate artifact manifest: `artifact-manifest-b1-b32.json`
- CPU-only full build: `/workspace/build-packed-qkv-b1b32/katago`
- Built binary SHA256: `cf3ff16a2ad3f23c30a4dd67e71fadeb95947faca1aa6e7d84eed3ee565a175e`

The shared main worktree `/workspace/katago` was not modified.

## Frozen provenance

Historical design was read remotely, read-only, from
`wangyize@10.101.3.156:/data/wangyize/katago/refactor-backup-20260805` and the
associated historical QKV build directory. No precompiled object was copied.

- Historical `cudaarch_sm120.cpp` SHA256:
  `e727edc2162c57d8bf0f899debb160b3502d12f02c9447b87f89249a13ff14fd`
- Historical `cudabackend.cpp` SHA256:
  `21f7a041c17afc4f97b694f1006500d72a1db4a0d22a6f780c3b97a27ebb4859`
- Historical `cudahelpers.cu` SHA256:
  `0a21b5a4e9dfad14657e9893a8740fcfb597b72a046d0e7363fdd8793679b944`
- Historical `cudahelpers.h` SHA256:
  `275d3ebe7c59e22a4890752b79543350d1f81d54e1b3681b58fea0f6070d9d18`
- Historical packed-QKV exporter SHA256:
  `4367a07f497e8206b2fb7ba261712c815a05e48a88ee4b161bbb11483c6304b2`
- Historical packed-RoPE exporter SHA256:
  `eefb1db78454babfdb0a3392cd72000e70ab28acc6652ebe219ea150f9d75aa1`

Fresh generation is pinned to:

- CUTLASS source commit: `e05f953a5b3d38adc240df2ff928e0421c2abba3`
  (clean v4.6.1 tree at `/workspace/third_party/cutlass`)
- Pinned dense example SHA256:
  `613052799aff35d5564d49c8bbb4bbac2e22bc58cb3e27499c4c9c3ee95c6e03`
- Deterministically patched dense example SHA256:
  `3e067e4f635c8bb7d09e641aa9a68c0898656c479a506e95b4255fe2ab157c26`
- Generator SHA256:
  `85191b822ded58ba8034741b3696a403e3d359e352a3bf63309b48217afd8e47`
- Python `3.12.3`
- `nvidia-cutlass-dsl==4.7.0`
- `nvidia-cutlass-dsl-libs-cu12==4.7.0`, DSL CUDA version `12.9`
- `cuda-python==13.3.1`
- System NVCC `CUDA 13.2, V13.2.86`

The deterministic patch removes the two `setmaxregister` hints from the pinned
CUTLASS SM120 example. SM120 libNVVM rejects these directives; the historical
KataGo source made the same adjustment. The patched Python source is emitted
beside every AOT object for auditing.

## Implemented contract

The generator uses inert static CUDA DLPack descriptors, so specialization and
object emission do not allocate, query, or execute on a GPU. For exact batch B:

- `M=B*361`, `N=1152`, `K=384`
- tile `128x128x64`, two computed AB stages, FP16 accumulation
- atom layout `(4,2,1)`, 8 MMA warps plus one DMA warp
- output row-major `[token,1152]`, interpreted as `[token,3,384]`
- Q/K/V base offsets in half elements: `0`, `384`, `768`

The C++ integration propagates a `packedOutput` property from the generated AOT
registry through the official attention block. Packed Q/K/V then use:

- packed batch-shared half2 RoPE with a 1152-half token stride;
- FA4 dynamic BSHD strides `{361*1152, 1152, 32}` for Q, K, and V;
- normal contiguous output strides for attention output.

Packed selection is gated to exact 19x19 FP16, no mask, learnable RoPE, fused
batch-shared RoPE, FA4 enabled, and FA4 `both16`. If packed buffers have already
been produced, RoPE or FA4 failure throws instead of falling through to a
planar-only implementation. For other shapes or masks, the packed tactic is
skipped before launch and the existing planar path remains valid.

The search runner now regenerates and reconfigures the exact object for each
batch and automatically adds the packed prerequisites to the natural S2 config.
The generated B1-B32 search space and generation plan are in
`space-b1-b32.json` and `generation-plan-qkv-b1-b32.json`.

## Validation performed without GPU use

1. Generated B1-B32 independently. Each metadata file reports exact
   `rows=B*361`, packed layout, the pinned hashes above, and hashes of its fresh
   `.h`, `.o`, and bridge.
2. Verified all B1-B32 metadata and artifacts. Host-object sizes are 41,224 to
   41,248 bytes; B13 object SHA256 is
   `534e0c8f810c3506c200fd1ea2e301eb210af31d9100f41a85854c0863702f6e`.
3. Ran CPU layout-contract checks for B1-B32 boundary coordinates:
   packed plane coverage, packed RoPE Q/K addresses, V exclusion, and equality
   between packed offsets and FA4 dynamic-stride offsets. All passed.
4. Compiled all 32 generated CUDA bridges with `nvcc -arch=sm_120`. All passed.
5. Configured and built a complete CUDA KataGo with the freshly generated B13
   bridge/object. Compilation and final link passed.
6. Inspected the linked binary: it contains
   `sm120_search_qkv_{batch,id,packed,launch}`, the packed RoPE launcher, and the
   packed candidate ID.
7. `git diff --check` and Python bytecode compilation passed. Pytest is not
   installed, so the three pure-Python test functions were imported and invoked
   directly; all passed.

Representative commands:

```sh
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2 \
  python3 python/sm120_generate_cute_qkv_aot.py \
  --batch 13 --output-dir .../b13 \
  --bridge-path .../b13/sm120_qkv_cute_active.cu

cmake -S cpp -B /workspace/build-packed-qkv-b1b32 \
  -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release -DNO_GIT_REVISION=1 \
  -DSM120_SEARCH_QKV_SOURCE=.../b13/sm120_qkv_cute_active.cu \
  -DSM120_SEARCH_QKV_OBJECT=.../b13/sm120_qkv_cute_active.o
cmake --build /workspace/build-packed-qkv-b1b32 -j2
```

No `gpu-lock`, KataGo inference, CUDA kernel launch, NCU, NSYS, or other GPU
command was run in this phase.

## Remaining integration risks / next gate

1. Numerical correctness and runtime module loading are not yet GPU-validated.
   The next gate should compare packed and planar Q/K/V, post-RoPE tensors, FA4
   output, and final network output at B13 before any performance judgment.
2. The generated CuTe header loads its embedded library across visible CUDA
   devices. Validate this under the actual `gpu-lock --gpu 2` visibility and the
   confirmed CUDA-ordinal-2 / NVML-index-1 mapping; do not change the KataGo
   CUDA ordinal.
3. FP16 accumulation matches the historical retained candidate but needs the
   normal KataGo numerical tolerance check against the current planar tactic.
4. `max_active_clusters=170` is intentionally frozen for the 170-SM RTX 5090 D.
   These objects are not portable performance choices for 5080 or another SM
   count even though they are valid SM120 code.
5. Packed buffers deliberately have no generic cuDNN/custom-attention fallback.
   The early gate and later hard failures are safety properties, but runtime
   testing must verify all intended B13 natural-S2 config prerequisites are
   present.
6. The runner reconfigures CMake when switching between TileLang planar and
   CuTe packed objects. This is slower than overwriting a single source slot,
   but makes the linked exact-batch object explicit and reproducible.

Nothing in this commit claims a throughput win. It is ready for the parent
thread's controlled GPU correctness gate and then natural whole-graph S2
measurement/profile cycle.
