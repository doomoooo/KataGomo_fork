#!/usr/bin/env python3
"""Generate one exact-batch FA4 SM120 AOT search candidate.

Prereqs (see /workspace/container-setup/third_party_env.sh):
  - Python venv with flash-attn 4.x + nvidia-cutlass-dsl installed
  - A GPU with compute capability 12.0 (RTX 5090D) reachable from CUDA device 2
  - ptxas from CUDA 13.x

Usage:
  source /workspace/container-setup/third_party_env.sh
  python cpp/neuralnet/fa4_aot/build_aot.py --batch B --device CUDA_ORDINAL

The selected per-GPU/per-batch winner is checked into the tactic registry only
after correctness and natural whole-graph S2 validation.
"""

import argparse
import hashlib
import json
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--device", type=int, required=True)
parser.add_argument("--output-dir")
parser.add_argument("--artifact-stem")
parser.add_argument("--symbol-prefix", default="fa4")
parser.add_argument("--candidate-id")
parser.add_argument("--bridge-path")
parser.add_argument("--tile-m", type=int, default=128)
parser.add_argument("--tile-n", type=int, default=128)
parser.add_argument("--num-stages", type=int, default=1)
args = parser.parse_args()
if args.batch < 1:
    parser.error("--batch must be positive")
if args.tile_m <= 0 or args.tile_n <= 0 or args.num_stages <= 0:
    parser.error("attention tile and stage values must be positive")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(args.output_dir or HERE)
ARTIFACT_STEM = args.artifact_stem or f"fa4_sm120_b{args.batch}"
SYMBOL_PREFIX = args.symbol_prefix
os.makedirs(OUT_DIR, exist_ok=True)

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

torch.cuda.set_device(args.device)

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
B, S, H, D = args.batch, 361, 12, 32
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

fa_fwd = FlashAttentionForwardSm120(
    Float16, D, D, 1,
    is_causal=False,
    is_local=False,
    pack_gqa=False,
    tile_m=args.tile_m,
    tile_n=args.tile_n,
    num_stages=args.num_stages,
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

compiled.export_to_c(OUT_DIR, ARTIFACT_STEM, SYMBOL_PREFIX)

if args.bridge_path:
    if not args.candidate_id:
        parser.error("--bridge-path requires --candidate-id")
    bridge_path = os.path.abspath(args.bridge_path)
    if os.path.dirname(bridge_path) != OUT_DIR:
        raise ValueError("FA4 bridge and generated header must share --output-dir")
    bridge = f'''#include "{ARTIFACT_STEM}.h"

#include <cmath>
#include <mutex>

namespace {{

{SYMBOL_PREFIX}_Kernel_Module_t module = {{}};
std::once_flag loadOnce;

}} // namespace

extern "C" int sm120_search_fa4_batch() {{ return {B}; }}
extern "C" const char* sm120_search_fa4_id() {{ return {json.dumps(args.candidate_id)}; }}
extern "C" cudaError_t sm120_search_fa4_launch(
  void* q, void* k, void* v, void* output,
  int batch, int seq, int heads, int dim, float scale, cudaStream_t stream
) {{
  if(batch != {B} || seq != {S} || heads != {H} || dim != {D})
    return cudaErrorInvalidValue;
  std::call_once(loadOnce, []() {{ {SYMBOL_PREFIX}_Kernel_Module_Load(&module); }});
  {SYMBOL_PREFIX}_Tensor_mQ_t tq = {{q, {{batch, seq, heads, dim}}, {{seq * heads * dim, heads * dim, dim}}}};
  {SYMBOL_PREFIX}_Tensor_mK_t tk = {{k, {{batch, seq, heads, dim}}, {{seq * heads * dim, heads * dim, dim}}}};
  {SYMBOL_PREFIX}_Tensor_mV_t tv = {{v, {{batch, seq, heads, dim}}, {{seq * heads * dim, heads * dim, dim}}}};
  {SYMBOL_PREFIX}_Tensor_mO_t to = {{output, {{batch, seq, heads, dim}}, {{seq * heads * dim, heads * dim, dim}}}};
  int32_t status = cute_dsl_{SYMBOL_PREFIX}_wrapper(
    &module, &tq, &tk, &tv, &to, scale, stream);
  return status == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}}
'''
    with open(bridge_path, "w", encoding="utf-8") as bridge_file:
        bridge_file.write(bridge)

    artifacts = {}
    for suffix in (".h", ".o"):
        path = os.path.join(OUT_DIR, ARTIFACT_STEM + suffix)
        with open(path, "rb") as artifact_file:
            artifacts[suffix] = hashlib.sha256(artifact_file.read()).hexdigest()
    artifacts["bridge"] = hashlib.sha256(bridge.encode()).hexdigest()
    with open(os.path.join(OUT_DIR, ARTIFACT_STEM + ".json"), "w") as metadata_file:
        json.dump({
            "schema": 1,
            "candidate_id": args.candidate_id,
            "batch": B,
            "fixed_shape": {"sequence": S, "heads": H, "head_dim": D},
            "accumulation": {"qk": QK_ACC, "pv": PV_ACC},
            "tile": {
                "m": args.tile_m,
                "n": args.tile_n,
                "num_stages": args.num_stages,
            },
            "artifact_stem": ARTIFACT_STEM,
            "symbol_prefix": SYMBOL_PREFIX,
            "sha256": artifacts,
            "acceptance_metric": "natural whole-graph S2 total throughput",
        }, metadata_file, indent=2)
        metadata_file.write("\n")
print(
    f"AOT export OK (B={B}, QK_ACC={QK_ACC}, PV_ACC={PV_ACC}, "
    f"tile={args.tile_m}x{args.tile_n}/S{args.num_stages}, "
    "fixed full-board S361, no mask) ->",
    OUT_DIR,
)
