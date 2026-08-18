#!/usr/bin/env python3
"""Correctness, isolated S1/S2 timing, and one-launch profiling for SM103 FA4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from sm103_generate_flash_attn_b29_native_aot import (
    ARCH,
    BATCH,
    CANDIDATE_ID,
    HEAD_DIM,
    HEADS,
    SCALE,
    SEQUENCE,
    TILE_M,
    TILE_N,
    candidate_contract,
    configure_environment,
    sha256_file,
    source_identity,
)


CUDNN_CONTROL_S1_US = 26.2951
CUDNN_CONTROL_S2_ROUND_US = 45.6243


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def compile_runtime_candidate(torch):
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.utils import AuxData

    torch.manual_seed(20260818)
    shape = (BATCH, SEQUENCE, HEADS, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    output = torch.empty_like(q)

    _flash_attn_fwd.compile_cache.clear()
    _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        softmax_scale=SCALE,
        causal=False,
        tile_mn=(TILE_M, TILE_N),
        num_splits=1,
        pack_gqa=False,
        _arch=103,
        out=output,
        return_lse=False,
    )
    torch.cuda.synchronize()
    if len(_flash_attn_fwd.compile_cache.cache) != 1:
        raise RuntimeError("expected exactly one FA4 forward compile-cache entry")
    compiled = next(iter(_flash_attn_fwd.compile_cache.cache.values()))

    def launch(destination):
        compiled(
            q,
            k,
            v,
            destination,
            None,
            SCALE,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AuxData(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SEQUENCE,
        )

    return q, k, v, output, launch


def correctness(torch, q, k, v, output, launch) -> dict[str, Any]:
    launch(output)
    torch.cuda.synchronize()
    with torch.no_grad():
        q_ref = q.float().permute(0, 2, 1, 3)
        k_ref = k.float().permute(0, 2, 1, 3)
        v_ref = v.float().permute(0, 2, 1, 3)
        scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * SCALE
        probabilities = torch.softmax(scores, dim=-1)
        reference = torch.matmul(probabilities, v_ref).permute(0, 2, 1, 3)
        difference = output.float() - reference
        absolute = difference.abs()
        reference_abs = reference.abs()
        metrics = {
            "max_abs": absolute.max().item(),
            "mean_abs": absolute.mean().item(),
            "rmse": difference.square().mean().sqrt().item(),
            "reference_max_abs": reference_abs.max().item(),
            "finite": bool(torch.isfinite(output).all().item()),
            "output_sha256": sha256_file_from_tensor(torch, output),
        }
    if not metrics["finite"] or metrics["max_abs"] > 0.05 or metrics["rmse"] > 0.005:
        raise RuntimeError(f"gross correctness failure: {metrics}")
    return metrics


def sha256_file_from_tensor(torch, tensor) -> str:
    import hashlib

    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def time_s1(torch, launch, output, warmup: int, iterations: int, repeats: int):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            launch(output)
    stream.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(iterations):
                launch(output)
            end.record(stream)
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return samples


def time_s2(torch, launch, output, warmup: int, iterations: int, repeats: int):
    output_other = torch.empty_like(output)
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    for _ in range(warmup):
        with torch.cuda.stream(stream0):
            launch(output)
        with torch.cuda.stream(stream1):
            launch(output_other)
    torch.cuda.synchronize()

    samples = []
    default_stream = torch.cuda.current_stream()
    for _ in range(repeats):
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        done0 = torch.cuda.Event()
        done1 = torch.cuda.Event()
        end = torch.cuda.Event(enable_timing=True)
        begin.record(default_stream)
        stream0.wait_event(begin)
        stream1.wait_event(begin)
        for _ in range(iterations):
            with torch.cuda.stream(stream0):
                launch(output)
            with torch.cuda.stream(stream1):
                launch(output_other)
        done0.record(stream0)
        done1.record(stream1)
        default_stream.wait_event(done0)
        default_stream.wait_event(done1)
        end.record(default_stream)
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / iterations)
    return samples


def summarize(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "minimum_us": min(samples),
        "maximum_us": max(samples),
        "p90_us": percentile(samples, 0.9),
        "relative_spread": (max(samples) - min(samples)) / statistics.median(samples),
    }


def run_profile_once(torch, launch, output) -> None:
    for _ in range(10):
        launch(output)
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    status = cudart.cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed: {status}")
    launch(output)
    torch.cuda.synchronize()
    status = cudart.cudaProfilerStop()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStop failed: {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate-bench", "profile-once"), default="validate-bench")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    output_path = arguments.output or Path(
        ".final-migration-env/profiles/sm103-b29/stage-06-fa4-native/control-validation.json"
    )
    configure_environment(output_path.resolve().parent)

    import torch

    if torch.cuda.get_device_capability(0) != (10, 3):
        raise RuntimeError("the isolated validator requires a physical SM103 GPU")
    q, k, v, output, launch = compile_runtime_candidate(torch)
    if arguments.mode == "profile-once":
        run_profile_once(torch, launch, output)
        print(CANDIDATE_ID)
        return 0

    accuracy = correctness(torch, q, k, v, output, launch)
    s1 = summarize(
        time_s1(
            torch,
            launch,
            output,
            arguments.warmup,
            arguments.iterations,
            arguments.repeats,
        )
    )
    s2 = summarize(
        time_s2(
            torch,
            launch,
            output,
            arguments.warmup,
            arguments.iterations,
            arguments.repeats,
        )
    )
    result = {
        "schema": 1,
        "candidate": candidate_contract(),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "source_identity": source_identity(),
        "correctness_vs_fp32": accuracy,
        "timing": {
            "method": "CUDA events around direct compiled TVM-FFI launches",
            "warmup": arguments.warmup,
            "iterations": arguments.iterations,
            "repeats": arguments.repeats,
            "s1": s1,
            "s2_round_two_streams": s2,
            "cudnn_control_us": {
                "s1": CUDNN_CONTROL_S1_US,
                "s2_round": CUDNN_CONTROL_S2_ROUND_US,
            },
            "candidate_over_control": {
                "s1": CUDNN_CONTROL_S1_US / s1["median_us"],
                "s2_round": CUDNN_CONTROL_S2_ROUND_US / s2["median_us"],
            },
        },
        "python": {"executable": sys.executable, "version": sys.version},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output_path)
    print(sha256_file(output_path))
    print(json.dumps(result["correctness_vs_fp32"], sort_keys=True))
    print(json.dumps(result["timing"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
