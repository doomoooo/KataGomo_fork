#!/usr/bin/env python3
"""Regenerate the FA4 SM120 AOT artifacts (fa4_sm120_b13.h/.o) used by the
KataGo SM120 backend attention hook.

Prereqs (see /workspace/container-setup/third_party_env.sh):
  - Python venv with flash-attn 4.x + nvidia-cutlass-dsl installed
  - A GPU with compute capability 12.0 (RTX 5090D) reachable from CUDA device 2
  - ptxas from CUDA 13.x

Usage:
  source /workspace/container-setup/third_party_env.sh
  python cpp/neuralnet/fa4_aot/build_aot.py

The generated header/object are checked into the repo; only regenerate when
the FA4/DSL toolchain or the fixed tile config changes, and verify with
fa4_smoke.cpp against attention_ref before committing.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_120")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")
os.environ.setdefault("CUTE_DSL_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", OUT_DIR)
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0")
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", os.path.join(OUT_DIR, ".aot-cache"))

import torch  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass import Float16, Float32  # noqa: E402

torch.cuda.set_device(2)

from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120  # noqa: E402
from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: E402
from flash_attn.cute.utils import AuxData  # noqa: E402
from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator  # noqa: E402
from quack import layout_utils  # noqa: E402

# AuxData is a compile-time marker the AOT C header generator does not know.
# Skip it when emitting the C signature (no aux tensors in our kernel).
def _make_skipping_gen(cls):
    orig = cls._generate_arguments

    def gen(self, symbol_prefix, args_spec, args, kwargs):
        rectified = args_spec.get_rectified_args(args, kwargs)

        class _Spec:
            pass

        spec = _Spec()
        spec.signature = args_spec.signature
        spec.get_rectified_args = lambda a, kw: [
            None if isinstance(v, AuxData) else v for v in rectified
        ]
        return orig(self, symbol_prefix, spec, args, kwargs)

    return gen


CuteCHeaderGenerator._generate_arguments = _make_skipping_gen(CuteCHeaderGenerator)

# Fixed shape used by the AOT probe/smoke: 19x19 (S=361), H=12, D=32, FP16,
# non-causal, SM120 tile 128x128, 128 threads, 1 stage. Batch is runtime.
B, S, H, D = 13, 361, 12, 32
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
out = torch.empty_like(q)
scale = 1.0 / (D ** 0.5)

# Accumulator modes: fp32/qk16/pv16/both16 (QK/PV accumulator data types).
#   FA4_QK_ACC=fp16 -> QK MMA accumulator FP16 (qk16)
#   FA4_PV_ACC=fp16 -> PV MMA accumulator FP16 (pv16)
# The accepted checked-in artifact uses FP16 for both accumulators. Override
# either variable with fp32 to reproduce the Stage-1 control or an ablation.
QK_ACC = os.environ.get("FA4_QK_ACC", "fp16")
PV_ACC = os.environ.get("FA4_PV_ACC", "fp16")
if QK_ACC not in ("fp16", "fp32") or PV_ACC not in ("fp16", "fp32"):
    raise ValueError("FA4_QK_ACC and FA4_PV_ACC must be fp16 or fp32")
qk_acc_dtype = Float16 if QK_ACC == "fp16" else Float32
pv_acc_dtype = Float16 if PV_ACC == "fp16" else Float32

# flash-attn 4.0.0b25 supports selecting an FP16 PV accumulator, but its
# online-softmax rescale stores an FP32 product without converting it back to
# the accumulator type. Keep the compatibility fix local to this AOT build.
if PV_ACC == "fp16":
    from flash_attn.cute.softmax import Softmax  # noqa: E402

    @cute.jit
    def _rescale_o_with_accumulator_cast(
        self,
        acc_o: cute.Tensor,
        row_scale: cute.Tensor,
    ) -> None:
        acc_o_mn = layout_utils.reshape_acc_to_mn(acc_o)
        assert cute.size(row_scale) == cute.size(acc_o_mn, mode=[0])
        for row in cutlass.range(cute.size(row_scale), unroll_full=True):
            scaled = acc_o_mn[row, None].load() * row_scale[row]
            acc_o_mn[row, None].store(scaled.to(acc_o_mn.element_type))

    Softmax.rescale_O = _rescale_o_with_accumulator_cast

FIXED_S361_MASK = os.environ.get("FA4_FIXED_S361_MASK", "0")
if FIXED_S361_MASK not in ("0", "1"):
    raise ValueError("FA4_FIXED_S361_MASK must be 0 or 1")

if FIXED_S361_MASK == "1":
    from flash_attn.cute.mask import (  # noqa: E402
        AttentionMask,
        mask_r2p_lambda,
        r2p_bitmask_below,
        sm90_col_to_r2p_idx,
    )

    @cute.jit
    def _apply_fixed_s361_tail_mask(
        self,
        acc_s: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr = None,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
    ) -> None:
        # This AOT object is constructed only for non-causal S361 with no mask
        # modifier. The first N iteration is fixed tile 2, with 105 valid cols.
        if cutlass.const_expr(mask_seqlen):
            acc_s_mn = layout_utils.reshape_acc_to_mn(acc_s)
            coordinates = cute.make_identity_tensor((self.tile_m, self.tile_n))
            thread_coordinates = layout_utils.reshape_acc_to_mn(
                thr_mma.partition_C(coordinates)
            )
            thread_col_offset = thread_coordinates[0][1]
            col_limit_r2p = sm90_col_to_r2p_idx(
                cutlass.Int32(105) - thread_col_offset
            )
            mask_r2p_lambda(
                acc_s_mn,
                lambda chunk: r2p_bitmask_below(col_limit_r2p, chunk),
            )

    AttentionMask.apply_mask = _apply_fixed_s361_tail_mask

fa_fwd = FlashAttentionForwardSm120(
    Float16, D, D, 1,
    is_causal=False,
    is_local=False,
    pack_gqa=False,
    tile_m=128,
    tile_n=128,
    num_stages=1,
    num_threads=128,
    Q_in_regs=False,
    score_mod=None,
    mask_mod=None,
    has_aux_tensors=False,
    qk_acc_dtype=qk_acc_dtype,
    pv_acc_dtype=pv_acc_dtype,
)

q_t, k_t, v_t, o_t = [to_cute_tensor(t) for t in (q, k, v, out)]
stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)

compiled = cute.compile(
    fa_fwd,
    q_t, k_t, v_t, o_t, None, scale,
    None, None, None, None, None, None, None, None,
    None, AuxData(), None, None, stream,
)

compiled.export_to_c(OUT_DIR, "fa4_sm120_b13", "fa4")
print(
    f"AOT export OK (QK_ACC={QK_ACC}, PV_ACC={PV_ACC}, "
    f"FIXED_S361_MASK={FIXED_S361_MASK}) ->",
    OUT_DIR,
)
