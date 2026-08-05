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
import cutlass.cute as cute  # noqa: E402
from cutlass import Float16  # noqa: E402

torch.cuda.set_device(2)

from flash_attn.cute.flash_fwd_sm120 import FlashAttentionForwardSm120  # noqa: E402
from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: E402
from flash_attn.cute.utils import AuxData  # noqa: E402
from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator  # noqa: E402

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
print("AOT export OK ->", OUT_DIR)
