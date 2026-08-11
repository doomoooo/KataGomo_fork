#!/usr/bin/env python3
"""Generate one exact-batch SM120 CuTe paired-projection FFN tactic.

The generated kernel consumes paired ``linear64,gate64`` weights, performs
both C384->C1152 projections, and applies SwiGLU before the wide intermediate
is written. Batch and persistent-grid size are explicit generator inputs; no
device name or B13 condition is embedded in the backend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")
if "CUDA_TOOLKIT_PATH" not in os.environ and "CUDA_HOME" in os.environ:
    os.environ["CUDA_TOOLKIT_PATH"] = os.environ["CUDA_HOME"]

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cuda.bindings import driver as cuda

import sm120_generate_cute_qkv_aot as common


SEQUENCE = 361
INPUT_CHANNELS = 384
OUTPUT_CHANNELS = 1152
WIDE_CHANNELS = 2 * OUTPUT_CHANNELS
PAIR_CHANNELS = 64
TILE = (128, 128, 32)
ATOM_LAYOUT = (4, 2, 1)
AB_STAGES = 2
EPILOGUE_STAGES = 4


def patch_dense_source(source: str) -> str:
    replacements = {
        "            cute.arch.setmaxregister_increase(self.mma_register_requirement)":
            "            pass  # SM120 rejects setmaxregister increase",
        "            cute.arch.setmaxregister_decrease(self.load_register_requirement)":
            "            pass  # SM120 rejects setmaxregister decrease",
        '''        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )''': '''        gC_mnl = cute.local_tile(
            mC_mnl,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1] // 2),
            (None, None, None),
        )''',
        '''        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)''': '''        acc_shape = tiled_mma.partition_shape_C(self.tile_shape_mnk[:2])
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)''',
        '''                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]''': '''                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        linear = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                        gate = tRS_rAcc[
                            (epi_idx + epi_tile_num) * size_tRS_rD + epi_v
                        ]
                        tRS_rD[epi_v] = (
                            linear / (1.0 + cute.math.exp(-linear)) * gate
                        ).to(self.acc_dtype)''',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(
                f"unexpected pinned dense source near: {before[:80]}"
            )
        source = source.replace(before, after)
    return source


def load_kernel(dense_path: pathlib.Path, output_dir: pathlib.Path):
    patched_path = output_dir / "dense_gemm_sm120_fused_swiglu.py"
    patched_path.write_text(patch_dense_source(dense_path.read_text()))
    spec = importlib.util.spec_from_file_location(
        "katago_sm120_fused_swiglu_dense_gemm", patched_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {patched_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sm120GemmKernel, patched_path


def make_kernel(sm120_gemm_kernel):
    class Kernel(sm120_gemm_kernel):
        def __init__(self) -> None:
            super().__init__(cutlass.Float16, TILE)
            self.atom_layout = ATOM_LAYOUT
            self.num_mma_warps = ATOM_LAYOUT[0] * ATOM_LAYOUT[1] * ATOM_LAYOUT[2]
            self.threads_per_cta = (
                self.num_mma_warps + 1
            ) * self.num_threads_per_warp
            self.epilog_sync_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=self.num_mma_warps * self.num_threads_per_warp,
            )

        def _compute_stages(self, *unused):
            return AB_STAGES, EPILOGUE_STAGES

        @staticmethod
        def _compute_grid(c, tile_shape_mnk, max_active_clusters):
            output_tile = (
                tile_shape_mnk[0], tile_shape_mnk[1] // 2, tile_shape_mnk[2],
            )
            return sm120_gemm_kernel._compute_grid(
                c, output_tile, max_active_clusters,
            )

    return Kernel()


def render_bridge(
    artifact_stem: str, candidate_id: str, batch: int, launch_symbol: str,
) -> str:
    return f'''#include "{artifact_stem}.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mutex>

namespace {{
{artifact_stem}_Kernel_Module_t module = {{}};
std::once_flag loadOnce;
}} // namespace

extern "C" cudaError_t {launch_symbol}(
  const half* input, const half* pairedWeights, const half* unusedGateWeights,
  half* output, cudaStream_t stream
) {{
  (void)unusedGateWeights;
  std::call_once(loadOnce, []() {{ {artifact_stem}_Kernel_Module_Load(&module); }});
  {artifact_stem}_Tensor_a_arg_t a = {{const_cast<half*>(input)}};
  {artifact_stem}_Tensor_b_arg_t b = {{const_cast<half*>(pairedWeights)}};
  {artifact_stem}_Tensor_c_arg_t c = {{output}};
  int32_t status = cute_dsl_{artifact_stem}_wrapper(&module, &a, &b, &c, stream);
  return status == 0 ? cudaPeekAtLastError() : cudaErrorUnknown;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--space", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-stem", required=True)
    parser.add_argument("--bridge-path", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--launch-symbol", required=True)
    parser.add_argument("--fat-symbol-token", required=True)
    parser.add_argument("--max-active-clusters", type=int, required=True)
    parser.add_argument(
        "--cutlass-root", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "../third_party/cutlass",
    )
    args = parser.parse_args()
    if not 4 <= args.batch <= 32:
        parser.error("--batch must be in B4-B32")
    if args.max_active_clusters < 1:
        parser.error("--max-active-clusters must be positive")
    for value, label in (
        (args.artifact_stem, "--artifact-stem"),
        (args.launch_symbol, "--launch-symbol"),
        (args.fat_symbol_token, "--fat-symbol-token"),
    ):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            parser.error(f"{label} must be a C identifier")

    space_path = args.space.resolve()
    space = json.loads(space_path.read_text())
    if space.get("architecture") != "sm120":
        parser.error("--space is not an SM120 search space")
    candidate = None
    for batch_entry in space.get("batches", []):
        if int(batch_entry.get("batch", -1)) != args.batch:
            continue
        candidate = next((
            value for value in batch_entry.get("dual_ffn", [])
            if value.get("id") == args.candidate_id
        ), None)
    if candidate is None:
        parser.error("candidate is absent from --space")

    cutlass_root = args.cutlass_root.resolve()
    actual_commit = common.git_output(cutlass_root, "rev-parse", "HEAD")
    if common.git_output(cutlass_root, "status", "--short"):
        raise RuntimeError("CUTLASS source must be clean")
    dense_path = cutlass_root / (
        "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/"
        "dense_gemm/dense_gemm.py"
    )
    dense_sha256 = common.sha256_file(dense_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Sm120GemmKernel, patched_path = load_kernel(dense_path, output_dir)
    gemm = make_kernel(Sm120GemmKernel)
    rows = args.batch * SEQUENCE
    a = common.make_tensor(
        (rows, INPUT_CHANNELS, 1),
        (INPUT_CHANNELS, 1, rows * INPUT_CHANNELS), 0x10000, 0,
    )
    b = common.make_tensor(
        (WIDE_CHANNELS, INPUT_CHANNELS, 1),
        (1, WIDE_CHANNELS, WIDE_CHANNELS * INPUT_CHANNELS), 0x20000, 0,
    )
    c = common.make_tensor(
        (rows, OUTPUT_CHANNELS, 1),
        (OUTPUT_CHANNELS, 1, rows * OUTPUT_CHANNELS), 0x30000, 0,
    )

    @cute.jit
    def launch(a_arg, b_arg, c_arg, stream: cuda.CUstream):
        gemm(a_arg, b_arg, c_arg, args.max_active_clusters, stream)

    compiled = cute.compile(
        launch, a, b, c,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    compiled.export_to_c(str(output_dir), file_name=args.artifact_stem)

    bridge_path = args.bridge_path.resolve()
    if bridge_path.parent != output_dir:
        raise ValueError("bridge and AOT artifacts must share --output-dir")
    bridge = render_bridge(
        args.artifact_stem, args.candidate_id, args.batch, args.launch_symbol,
    )
    bridge_path.write_text(bridge)
    artifact_base = output_dir / args.artifact_stem
    metadata = {
        "schema": 1,
        "family": "dual_ffn",
        "architecture": "sm120",
        "space_sha256": common.sha256_file(space_path),
        "candidate": candidate,
        "candidate_id": args.candidate_id,
        "batch": args.batch,
        "fixed_board": [19, 19],
        "rows": rows,
        "paired_weights": True,
        "pair_channels": PAIR_CHANNELS,
        "tile": list(TILE),
        "effective_output_tile": [TILE[0], TILE[1] // 2],
        "atom_layout": list(ATOM_LAYOUT),
        "ab_stages": AB_STAGES,
        "epilogue_stages": EPILOGUE_STAGES,
        "max_active_clusters": args.max_active_clusters,
        "launch_symbol": args.launch_symbol,
        "fat_symbol_token": args.fat_symbol_token,
        "generation_environment": {
            "compute_capability": [12, 0],
            "gpu_used_for_generation": False,
        },
        "provenance": {
            "generator_sha256": common.sha256_file(pathlib.Path(__file__).resolve()),
            "cutlass_commit": actual_commit,
            "dense_gemm_sha256": dense_sha256,
            "patched_dense_gemm_sha256": common.sha256_file(patched_path),
            "python": sys.version.split()[0],
            "nvidia_cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "cuda_python": importlib.metadata.version("cuda-python"),
            "nvcc_version": subprocess.run(
                ["nvcc", "--version"], check=True, text=True,
                capture_output=True,
            ).stdout.strip().splitlines()[-1],
        },
        "sha256": {
            "header": common.sha256_file(artifact_base.with_suffix(".h")),
            "object": common.sha256_file(artifact_base.with_suffix(".o")),
            "bridge": hashlib.sha256(bridge.encode()).hexdigest(),
        },
    }
    metadata_path = artifact_base.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
