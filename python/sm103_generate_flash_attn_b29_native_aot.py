#!/usr/bin/env python3
"""Generate the exact B29/S361/H12/D32 SM103a FA4 forward control.

This is deliberately an isolated artifact generator.  It does not edit or
link KataGo, and it does not use the public FlashAttention dispatcher: the
constructor arguments below are the complete compile-time tactic contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


BATCH = 29
SEQUENCE = 361
HEADS = 12
HEAD_DIM = 32
SCALE = HEAD_DIM**-0.5
ARCH = "sm_103a"
TILE_M = 128
TILE_N = 128
Q_STAGES = 2
KV_STAGES_DERIVED = 24
THREADS = 512
FLASH_ATTN_COMMIT = "0251105a2fb19d2957484b7f023cd8c115286ced"
SUBTILING_SEED_COMMIT = "526c18d25bcbc7fc7d6740ab3c7c84ed2d42cb0b"
CANDIDATE_ID = "fa4-main-sm103a-b29-s361-h12-d32-m128n128-q2-kv24-fp32"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_environment(output_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cuda_root = Path(os.environ.get("CUDA_HOME", repo_root / ".final-migration-env/cuda"))
    os.environ.setdefault("CUDA_HOME", str(cuda_root))
    os.environ.setdefault("CUDA_PATH", str(cuda_root))
    os.environ["FLASH_ATTENTION_ARCH"] = ARCH
    os.environ["CUTE_DSL_ARCH"] = ARCH
    os.environ.setdefault("CUTE_DSL_PTXAS_PATH", str(cuda_root / "bin/ptxas"))
    os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
    os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
    os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(output_dir))
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0")
    os.environ.setdefault(
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", str(output_dir / ".jit-cache")
    )


def source_identity() -> dict[str, Any]:
    import flash_attn.cute.flash_fwd_sm100 as flash_fwd_sm100
    import flash_attn.cute.interface as interface
    import flash_attn.cute.mask as mask
    import flash_attn.cute.softmax as softmax
    import flash_attn.cute.tile_scheduler as tile_scheduler

    repo_root = Path(__file__).resolve().parents[1]
    checkout = repo_root / ".final-migration-env/third_party/flash-attention"

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(checkout), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    modules = {
        name: {
            "path": str(Path(module.__file__).resolve()),
            "sha256": sha256_file(Path(module.__file__).resolve()),
        }
        for name, module in {
            "flash_attn.cute.flash_fwd_sm100": flash_fwd_sm100,
            "flash_attn.cute.interface": interface,
            "flash_attn.cute.mask": mask,
            "flash_attn.cute.softmax": softmax,
            "flash_attn.cute.tile_scheduler": tile_scheduler,
        }.items()
    }
    return {
        "flash_attention_checkout": {
            "path": str(checkout.resolve()),
            "head": git("rev-parse", "HEAD"),
            "expected_head": FLASH_ATTN_COMMIT,
            "subtiling_seed": git("rev-parse", "origin/subtiling"),
            "audited_subtiling_seed": SUBTILING_SEED_COMMIT,
            "subtiling_decision": (
                "not used: the seed's M64 SMEM-P constructor explicitly rejects "
                "SM103 and was only tested for SM100"
            ),
        },
        "distributions": {
            name: importlib.metadata.version(name)
            for name in (
                "flash-attn-4",
                "nvidia-cutlass-dsl",
                "quack-kernels",
                "torch",
                "apache-tvm-ffi",
            )
        },
        "modules": modules,
    }


def make_operator():
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

    return FlashAttentionForwardSm100(
        HEAD_DIM,
        HEAD_DIM,
        qhead_per_kvhead=1,
        is_causal=False,
        is_local=False,
        is_split_kv=False,
        pack_gqa=False,
        q_subtile_factor=1,
        kv_subtile_factor=1,
        m_block_size=TILE_M,
        n_block_size=TILE_N,
        q_stage=Q_STAGES,
        is_static_persistent=True,
        score_mod=None,
        mask_mod=None,
        has_aux_tensors=False,
        paged_kv_non_tma=False,
        is_varlen_q=False,
        use_2cta_instrs=False,
        use_clc_scheduler=False,
        has_tile_count_semaphore=False,
        seqlen_k_per_split=None,
    )


def compile_arguments(q, k, v, output, *, explicit_stream: bool):
    import cutlass.cute as cute
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    from flash_attn.cute.utils import AuxData

    q_tensor, k_tensor, v_tensor, output_tensor = [
        to_cute_tensor(tensor) for tensor in (q, k, v, output)
    ]
    stream = cute.runtime.make_fake_stream(
        use_tvm_ffi_env_stream=not explicit_stream
    )
    return (
        q_tensor,
        k_tensor,
        v_tensor,
        output_tensor,
        None,  # LSE
        SCALE,
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # seqused_q
        None,  # seqused_k
        None,  # page table
        None,  # window left
        None,  # window right
        None,  # learnable sink
        None,  # descale tensors
        None,  # block sparse tensors
        AuxData(),
        None,  # dynamic split count
        None,  # tile-count semaphore
        None,  # virtual batch index
        None,  # number of heads in L2
        None,  # cumulative M blocks
        None,  # cumulative split M blocks
        None,  # blocks-to-batch map
        SEQUENCE,  # exact max sequence Q
        stream,
    )


def patch_c_header_generator() -> None:
    """Drop compile-time marker structs from the exported C ABI."""

    from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator
    from cutlass import Int32
    from flash_attn.cute.utils import AuxData

    original = CuteCHeaderGenerator._generate_arguments
    if getattr(original, "_sm103_fa4_marker_patch", False):
        return

    def generate(self, symbol_prefix, args_spec, args, kwargs):
        rectified = args_spec.get_rectified_args(args, kwargs)
        argument_names = tuple(args_spec.signature.parameters)
        if len(argument_names) != len(rectified):
            raise RuntimeError("CuTe C-header argument-name/value count drifted")

        class Spec:
            pass

        spec = Spec()
        # ``max_seqlen_q`` is a real runtime Int32 in the lowered function, but
        # upstream annotates it as ``Int32 | int | None``.  CuTe DSL 4.7's C
        # generator does not unwrap PEP-604 unions.  Preserve the argument and
        # rectify only its export annotation; dropping it would shift stream
        # and return slots in the C ABI.
        spec.signature = args_spec.signature.replace(
            parameters=[
                parameter.replace(annotation=Int32)
                if parameter.name == "max_seqlen_q"
                else parameter
                for parameter in args_spec.signature.parameters.values()
            ]
        )
        spec.get_rectified_args = lambda unused_args, unused_kwargs: [
            None
            if isinstance(value, AuxData)
            else value
            for name, value in zip(argument_names, rectified, strict=True)
        ]
        return original(self, symbol_prefix, spec, args, kwargs)

    generate._sm103_fa4_marker_patch = True
    CuteCHeaderGenerator._generate_arguments = generate


def candidate_contract() -> dict[str, Any]:
    return {
        "candidate_id": CANDIDATE_ID,
        "architecture": ARCH,
        "shape": {
            "batch": BATCH,
            "sequence_q": SEQUENCE,
            "sequence_kv": SEQUENCE,
            "heads_q": HEADS,
            "heads_kv": HEADS,
            "head_dim_qk": HEAD_DIM,
            "head_dim_v": HEAD_DIM,
        },
        "semantics": {
            "dtype": "float16",
            "layout": "BSHD planar Q/K/V",
            "scale": SCALE,
            "causal": False,
            "local": False,
            "mask": None,
            "bias": None,
            "dropout": 0.0,
            "output_dtype": "float16",
        },
        "schedule": {
            "tile_m": TILE_M,
            "tile_n": TILE_N,
            "q_stages": Q_STAGES,
            "kv_stages": KV_STAGES_DERIVED,
            "threads": THREADS,
            "cta_count": 1,
            "cluster": [1, 1, 1],
            "scheduler": "static_persistent",
            "qk_accumulator": "float32",
            "pv_accumulator": "float32",
            "exp2": "native",
            "tcgen05_ld_red": False,
        },
    }


def generate(output_dir: Path, artifact_stem: str, symbol_prefix: str) -> Path:
    configure_environment(output_dir)

    import cutlass.cute as cute
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.manual_seed(20260818)
    shape = (BATCH, SEQUENCE, HEADS, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    output = torch.empty_like(q)

    patch_c_header_generator()
    operator = make_operator()
    compiled = cute.compile(
        operator,
        *compile_arguments(q, k, v, output, explicit_stream=True),
    )
    compiled.export_to_c(str(output_dir), artifact_stem, symbol_prefix)

    artifact_hashes = {}
    for suffix in (".h", ".o"):
        path = output_dir / f"{artifact_stem}{suffix}"
        artifact_hashes[path.name] = sha256_file(path)
    import cutlass

    runtime_archive = (
        Path(cutlass.__file__).resolve().parents[2]
        / "cu13/lib/libcuda_dialect_runtime_static.a"
    )
    if not runtime_archive.is_file():
        raise RuntimeError(f"missing CuTe static CUDA runtime: {runtime_archive}")

    manifest = {
        "schema": 1,
        "kind": "isolated-sm103-fa4-forward-aot",
        "contract": candidate_contract(),
        "source_identity": source_identity(),
        "python": {"executable": sys.executable, "version": sys.version},
        "artifact": {
            "stem": artifact_stem,
            "symbol_prefix": symbol_prefix,
            "sha256": artifact_hashes,
        },
        "runtime": {
            "static_cuda_dialect_runtime": {
                "path": str(runtime_archive),
                "bytes": runtime_archive.stat().st_size,
                "sha256": sha256_file(runtime_archive),
            },
            "requires_python": False,
            "requires_tvm_ffi": False,
        },
        "integration": {
            "core_modified": False,
            "cmake_modified": False,
            "status": "isolated candidate only",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            ".final-migration-env/artifacts/fa4-sm103a-b29-d32-m128n128-control"
        ),
    )
    parser.add_argument("--artifact-stem", default="fa4_sm103a_b29_d32_control")
    parser.add_argument("--symbol-prefix", default="fa4_sm103a_b29")
    arguments = parser.parse_args()
    if arguments.batch != BATCH:
        parser.error("this generator is exact-batch B29; --batch must equal 29")
    return arguments


def main() -> int:
    arguments = parse_args()
    manifest = generate(
        arguments.output_dir.resolve(), arguments.artifact_stem, arguments.symbol_prefix
    )
    print(manifest)
    print(sha256_file(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
