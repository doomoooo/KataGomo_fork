#!/usr/bin/env python3
"""Generate one exact-batch packed-QKV plus FP16 RoPE CuTe tactic."""

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
from cuda.bindings import driver as cuda

import sm120_generate_cute_qkv_aot as common


SEQUENCE = 361
INPUT_CHANNELS = 384
OUTPUT_CHANNELS = 3 * 384
ROPE_PAIRS = 192
TILE = (128, 128, 64)
ATOM_LAYOUT = (4, 2, 1)


def load_kernel(
    dense_path: pathlib.Path, output_dir: pathlib.Path, rows: int,
):
    source = dense_path.read_text()
    replacements = {
        "            cute.arch.setmaxregister_increase(self.mma_register_requirement)":
            "            pass  # SM120 rejects setmaxregister increase",
        "            cute.arch.setmaxregister_decrease(self.load_register_requirement)":
            "            pass  # SM120 rejects setmaxregister decrease",
        """        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,""": """        c: cute.Tensor,
        rope_table: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,""",
        """            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(""": """            self.epi_smem_layout_staged,
            rope_table,
            tile_sched_params,
        ).launch(""",
        """        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,""": """        epi_smem_layout_staged: cute.ComposedLayout,
        rope_table: cute.Tensor,
        tile_sched_params: utils.PersistentTileSchedulerParams,""",
        """                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                # Allocate D registers.""": f"""                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                coord_mnl = cute.make_identity_tensor(({rows}, 1152, 1))
                coord_tile = cute.local_tile(
                    coord_mnl,
                    cute.slice_(self.tile_shape_mnk, (None, None, 0)),
                    (tile_coord_mnl[0], tile_coord_mnl[1], tile_coord_mnl[2]),
                )
                tCgCoord = thr_mma.partition_C(coord_tile)
                tRS_rCoord = tiled_copy_r2s.retile(tCgCoord)

                # Allocate D registers.""",
        """                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Type conversion""": """                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    # Q/K are the first six full N128 tiles. The coordinate
                    # layout is derived from the accumulator retile, so this
                    # remains exact for every generated batch row count.
                    if tile_coord_mnl[1] < 6:
                        base_m = tRS_rCoord[0][0]
                        base_n = tRS_rCoord[0][1]
                        qk_n_offset = 384 if tile_coord_mnl[1] >= 3 else 0
                        epi_m = (epi_idx % 2) * 64
                        epi_n = (epi_idx // 2) * 32
                        for rope_pair in cutlass.range_constexpr(size_tRS_rD // 2):
                            idx0 = rope_pair * 2
                            idx1 = idx0 + 1
                            global_m = base_m + epi_m + (rope_pair % 2) * 8
                            global_n = base_n + epi_n + (rope_pair // 2) * 16
                            hp = (global_n - qk_n_offset) // 2
                            xy = global_m % 361
                            cos_v = rope_table[(xy, hp, 0)]
                            sin_v = rope_table[(xy, hp, 1)]
                            q0 = tRS_rD[idx0]
                            q1 = tRS_rD[idx1]
                            tRS_rD[idx0] = q0 * cos_v - q1 * sin_v
                            tRS_rD[idx1] = q0 * sin_v + q1 * cos_v

                    # Type conversion""",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError(
                f"unexpected pinned dense source near: {before[:100]}"
            )
        source = source.replace(before, after)
    patched = output_dir / "dense_gemm_sm120_packed_qkv_rope.py"
    patched.write_text(source)
    spec = importlib.util.spec_from_file_location(
        "katago_sm120_packed_qkv_rope", patched,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {patched}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sm120GemmKernel, patched


def render_bridge(artifact_stem: str, launch_symbol: str) -> str:
    return f'''#include "{artifact_stem}.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>

namespace {{
{artifact_stem}_Kernel_Module_t module = {{}};
std::once_flag loadOnce;
}}

extern "C" cudaError_t {launch_symbol}(
  const half* input, const half* weights, const half* ropeTable,
  half* output, cudaStream_t stream
) {{
  std::call_once(loadOnce, []() {{ {artifact_stem}_Kernel_Module_Load(&module); }});
  {artifact_stem}_Tensor_a_arg_t a = {{const_cast<half*>(input)}};
  {artifact_stem}_Tensor_b_arg_t b = {{const_cast<half*>(weights)}};
  {artifact_stem}_Tensor_c_arg_t c = {{output}};
  {artifact_stem}_Tensor_table_arg_t table = {{const_cast<half*>(ropeTable)}};
  int32_t status = cute_dsl_{artifact_stem}_wrapper(
    &module, &a, &b, &c, &table, stream);
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
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--cutlass-root", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] /
            "../third_party/cutlass",
    )
    args = parser.parse_args()
    if not 4 <= args.batch <= 32:
        parser.error("--batch must be in B4-B32")
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
        if int(batch_entry.get("batch", -1)) == args.batch:
            candidate = next((
                value for value in batch_entry.get("qkv_rope", [])
                if value.get("id") == args.candidate_id
            ), None)
    if candidate is None:
        parser.error("candidate is absent from --space")

    cutlass_root = args.cutlass_root.resolve()
    actual_commit = common.git_output(cutlass_root, "rev-parse", "HEAD")
    if actual_commit != common.CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS commit mismatch: {actual_commit} != {common.CUTLASS_COMMIT}"
        )
    if common.git_output(cutlass_root, "status", "--short"):
        raise RuntimeError("CUTLASS source must be clean")
    dense_path = cutlass_root / (
        "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/"
        "dense_gemm/dense_gemm.py"
    )
    dense_sha256 = common.sha256_file(dense_path)
    if dense_sha256 != common.DENSE_GEMM_SHA256:
        raise RuntimeError("dense_gemm.py hash mismatch")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = args.batch * SEQUENCE
    kernel_class, patched_path = load_kernel(dense_path, output_dir, rows)
    gemm = common.FixedAtomLayoutGemm.make(kernel_class)
    device_properties = common.query_cuda_device(args.device)
    max_active_clusters = int(device_properties["multiprocessor_count"])
    a = common.make_tensor(
        (rows, INPUT_CHANNELS, 1),
        (INPUT_CHANNELS, 1, rows * INPUT_CHANNELS), 0x10000, args.device,
    )
    b = common.make_tensor(
        (OUTPUT_CHANNELS, INPUT_CHANNELS, 1),
        (1, OUTPUT_CHANNELS, OUTPUT_CHANNELS * INPUT_CHANNELS),
        0x20000, args.device,
    )
    c = common.make_tensor(
        (rows, OUTPUT_CHANNELS, 1),
        (OUTPUT_CHANNELS, 1, rows * OUTPUT_CHANNELS), 0x30000, args.device,
    )
    table = common.make_tensor(
        (SEQUENCE, ROPE_PAIRS, 2),
        (ROPE_PAIRS * 2, 2, 1), 0x40000, args.device,
    )

    @cute.jit
    def launch(a_arg, b_arg, c_arg, table_arg, stream: cuda.CUstream):
        gemm(
            a_arg, b_arg, c_arg, table_arg, max_active_clusters, stream,
        )

    compiled = cute.compile(
        launch, a, b, c, table,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
    )
    compiled.export_to_c(str(output_dir), file_name=args.artifact_stem)

    bridge_path = args.bridge_path.resolve()
    if bridge_path.parent != output_dir:
        raise ValueError("bridge and AOT artifacts must share --output-dir")
    bridge = render_bridge(args.artifact_stem, args.launch_symbol)
    bridge_path.write_text(bridge)
    artifact_base = output_dir / args.artifact_stem
    metadata = {
        "schema": 1,
        "family": "qkv_rope",
        "architecture": "sm120",
        "space_sha256": common.sha256_file(space_path),
        "candidate": candidate,
        "candidate_id": args.candidate_id,
        "batch": args.batch,
        "fixed_board": [19, 19],
        "rows": rows,
        "layout": "packed-token-qkv",
        "rope": "fp16-register-fragment-epilogue",
        "max_active_clusters": max_active_clusters,
        "launch_symbol": args.launch_symbol,
        "fat_symbol_token": args.fat_symbol_token,
        "generation_environment": {
            "compute_capability": [12, 0],
            "gpu_used_for_generation": False,
        },
        "provenance": {
            "generator_sha256": common.sha256_file(
                pathlib.Path(__file__).resolve()
            ),
            "cutlass_commit": actual_commit,
            "dense_gemm_sha256": dense_sha256,
            "patched_dense_gemm_sha256": common.sha256_file(patched_path),
            "python": sys.version.split()[0],
            "nvidia_cutlass_dsl": importlib.metadata.version(
                "nvidia-cutlass-dsl"
            ),
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
