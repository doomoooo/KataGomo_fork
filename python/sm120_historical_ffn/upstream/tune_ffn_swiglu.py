#!/usr/bin/env python3
"""Reproducible fixed-B19 TileLang dual-GEMM + SwiGLU audit.

Compilation is isolated from timed workers. Every timed worker takes the shared
GPU lock and is killed by the parent after 15 seconds.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import torch

import onnx_kernels as kernels


GPU_LOCK = Path("/data/wangyize/katago/locks/gpu-benchmark.lock")
RESULT_DIR = Path("/data/wangyize/katago/results/tilelang-katago")
TIMEOUT_SECONDS = 15.0
COMPILE_TIMEOUT_SECONDS = 180.0
ROWS, COLUMNS, INNER = 19 * 19 * 19, 1152, 384
USEFUL_FLOPS = 4 * ROWS * COLUMNS * INNER
CUTLASS_FP16_US = 63.75


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    stages: int
    threads: int
    accumulation: str
    raster: bool = False
    policy: str = "square"
    warp_specialization: bool = True
    reuse_a_fragment: bool = False
    implementation: str = "generic"
    epilogue: str = "swiglu"
    fine_mma: bool = False
    min_blocks_per_sm: int = 1

    @property
    def label(self) -> str:
        accum = "A16" if self.accumulation == "float16" else "A32"
        policy = {"square": "SQ", "full_row": "FR", "full_col": "FC"}[
            self.policy
        ]
        return (
            f"M{self.block_m}-N{self.block_n}-K{self.block_k}-S{self.stages}-"
            f"T{self.threads}-{accum}-R{int(self.raster)}-{policy}-"
            f"WS{int(self.warp_specialization)}-AR{int(self.reuse_a_fragment)}-"
            f"{'MMA' if self.implementation == 'intrinsics' else 'GEN'}-"
            f"EPI-{self.epilogue.upper()}-FM{int(self.fine_mma)}-"
            f"MB{self.min_blocks_per_sm}"
        )

    def cli(self) -> str:
        values = (
            self.block_m,
            self.block_n,
            self.block_k,
            self.stages,
            self.threads,
            self.accumulation,
            int(self.raster),
            self.policy,
            int(self.warp_specialization),
            int(self.reuse_a_fragment),
            self.implementation,
            self.epilogue,
            int(self.fine_mma),
            self.min_blocks_per_sm,
        )
        return ",".join(str(value) for value in values)


def intrinsic_config(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    threads: int = 128,
    accumulation: str = "float16",
    raster: bool = False,
    policy: str = "square",
    epilogue: str = "swiglu",
    fine_mma: bool = False,
    min_blocks_per_sm: int = 1,
) -> Config:
    return Config(
        block_m,
        block_n,
        block_k,
        stages,
        threads,
        accumulation,
        raster,
        policy,
        False,
        False,
        "intrinsics",
        epilogue,
        fine_mma,
        min_blocks_per_sm,
    )


# The quick set spans the CUTLASS winner, the old TileLang winner, pipeline
# depth, compiler warp specialization, and both required precision tracks.
QUICK_CONFIGS = (
    Config(64, 64, 64, 1, 128, "float16", False, "square", True),
    Config(64, 64, 64, 1, 128, "float16", False, "square", False),
    Config(64, 64, 32, 2, 128, "float16", False, "square", True),
    Config(64, 64, 32, 2, 128, "float16", False, "square", False),
    Config(128, 64, 32, 2, 128, "float16", False, "square", True),
    Config(128, 64, 32, 2, 128, "float16", False, "square", False),
    Config(128, 64, 32, 3, 128, "float16", False, "square", True),
    Config(128, 64, 32, 3, 128, "float16", False, "square", False),
    Config(128, 64, 32, 3, 256, "float16", False, "square", False),
    Config(64, 128, 32, 2, 128, "float16", False, "square", False),
    Config(64, 64, 64, 1, 128, "float32", False, "square", True),
    Config(64, 64, 64, 1, 128, "float32", False, "square", False),
    Config(64, 64, 32, 2, 128, "float32", False, "square", False),
    Config(128, 64, 32, 2, 128, "float32", False, "square", False),
)


FULL_EXTRAS = (
    Config(32, 64, 32, 1, 128, "float16", False, "square", False),
    Config(64, 32, 32, 2, 128, "float16", False, "square", False),
    Config(64, 64, 32, 3, 128, "float16", False, "square", False),
    Config(64, 64, 64, 2, 128, "float16", False, "square", False),
    Config(128, 64, 64, 2, 128, "float16", False, "square", False),
    Config(128, 64, 32, 3, 128, "float16", True, "square", False),
    Config(128, 64, 32, 3, 128, "float16", False, "full_row", False),
    Config(128, 64, 32, 3, 128, "float16", False, "full_col", False),
    Config(128, 128, 32, 2, 256, "float16", False, "square", False),
    Config(64, 64, 32, 3, 128, "float32", False, "square", False),
    Config(64, 64, 64, 2, 128, "float32", False, "square", False),
    Config(64, 64, 32, 2, 128, "float32", True, "square", False),
    Config(64, 64, 32, 2, 128, "float32", False, "full_row", False),
)


INTRINSIC_CONFIGS = (
    intrinsic_config(128, 64, 32, 1),
    intrinsic_config(128, 64, 32, 2),
    intrinsic_config(128, 64, 32, 3),
    intrinsic_config(128, 64, 32, 2, raster=True),
    intrinsic_config(128, 64, 32, 2, policy="full_col"),
    intrinsic_config(128, 64, 32, 2, policy="full_row"),
    intrinsic_config(128, 64, 16, 2),
    intrinsic_config(128, 64, 64, 1),
    intrinsic_config(128, 64, 64, 2),
    intrinsic_config(64, 64, 32, 2),
    intrinsic_config(64, 128, 32, 2),
    intrinsic_config(128, 128, 32, 2, threads=256),
    intrinsic_config(128, 64, 32, 1, accumulation="float32"),
    intrinsic_config(128, 64, 32, 2, accumulation="float32"),
    intrinsic_config(
        128, 64, 32, 2, accumulation="float32", policy="full_col"
    ),
    intrinsic_config(64, 64, 32, 2, accumulation="float32"),
)


OPTIMIZED_CONFIGS = (
    intrinsic_config(128, 64, 32, 1, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(128, 64, 32, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(
        128,
        64,
        32,
        2,
        epilogue="swiglu_tanh_f16x2",
        min_blocks_per_sm=3,
    ),
    intrinsic_config(128, 64, 32, 3, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(128, 64, 16, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(128, 64, 64, 1, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(128, 64, 64, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(64, 64, 32, 1, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(64, 64, 32, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(64, 64, 32, 3, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(64, 128, 32, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(128, 32, 32, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(64, 32, 32, 2, epilogue="swiglu_tanh_f16x2"),
    intrinsic_config(
        128, 64, 32, 2, policy="full_col", epilogue="swiglu_tanh_f16x2"
    ),
    intrinsic_config(
        128, 64, 32, 2, policy="full_row", epilogue="swiglu_tanh_f16x2"
    ),
    intrinsic_config(
        128, 64, 32, 2, threads=256, epilogue="swiglu_tanh_f16x2"
    ),
    intrinsic_config(
        128, 128, 32, 2, threads=256, epilogue="swiglu_tanh_f16x2"
    ),
)


LAUNCH_BOUND_CONFIGS = tuple(
    intrinsic_config(
        128,
        64,
        32,
        stages,
        epilogue="swiglu_tanh_f16x2",
        min_blocks_per_sm=min_blocks,
    )
    for stages, maximum in ((1, 5), (2, 4))
    for min_blocks in range(1, maximum + 1)
)


def compile_kernel(config: Config):
    if config.implementation == "intrinsics":
        warps = config.threads // 32
        if config.policy == "full_col":
            row_warps, column_warps = 1, warps
        elif config.policy == "full_row":
            row_warps, column_warps = warps, 1
        elif warps == 4:
            row_warps, column_warps = 2, 2
        elif warps == 8:
            row_warps, column_warps = 4, 2
        else:
            raise ValueError(f"No intrinsic warp layout for {config.threads} threads")
        return kernels.ffn_swiglu_intrinsics(
            ROWS,
            COLUMNS,
            INNER,
            config.block_m,
            config.block_n,
            config.block_k,
            config.stages,
            config.accumulation,
            config.raster,
            row_warps,
            column_warps,
            config.epilogue,
            config.fine_mma,
            config.min_blocks_per_sm,
        )
    if config.implementation != "generic":
        raise ValueError(f"Unknown implementation: {config.implementation}")
    return kernels.ffn_swiglu(
        ROWS,
        COLUMNS,
        INNER,
        config.block_m,
        config.block_n,
        config.block_k,
        config.stages,
        config.threads,
        config.accumulation,
        config.raster,
        config.policy,
        config.warp_specialization,
        config.reuse_a_fragment,
    )


def tensors():
    torch.manual_seed(20260804)
    data = torch.randn((ROWS, INNER), dtype=torch.float16, device="cuda") * 0.1
    weights = tuple(
        torch.randn((INNER, COLUMNS), dtype=torch.float16, device="cuda") * 0.1
        for _ in range(2)
    )
    return data, weights


def source_metadata(source: str) -> dict:
    launch = re.search(r"__launch_bounds__\((\d+),\s*(\d+)\)", source)
    return {
        "launchBounds": int(launch.group(1)) if launch else None,
        "minBlocksPerSm": int(launch.group(2)) if launch else None,
        "mmaInstructionsInSource": source.count("tl::mma_sync<"),
        "usesFp16AccumulatorMma": "kFloat16, 16, 8, 16" in source,
        "usesFp32AccumulatorMma": "kFloat32, 16, 8, 16" in source,
        "usesScalarExpf": "expf(" in source,
        "usesTmaLoad": "tl::tma_load(" in source,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generatedSourceBytes": len(source.encode()),
    }


def measure(function) -> tuple[float, float, list[float]]:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 100)
    return statistics.median(samples), min(samples), samples


def run_timed(config: Config) -> dict:
    data, weights = tensors()
    output = torch.empty((ROWS, COLUMNS), dtype=torch.float16, device="cuda")
    kernel = compile_kernel(config)

    def function() -> None:
        kernels.invoke(kernel, data, weights[0], weights[1], output)

    function()
    torch.cuda.synchronize()
    linear_fp32 = data.float() @ weights[0].float()
    if config.epilogue == "linear":
        expected_fp32 = linear_fp32
    elif config.epilogue == "add":
        expected_fp32 = linear_fp32 + (data.float() @ weights[1].float())
    elif config.epilogue == "multiply":
        expected_fp32 = linear_fp32 * (data.float() @ weights[1].float())
    else:
        expected_fp32 = (
            linear_fp32
            * torch.sigmoid(linear_fp32)
            * (data.float() @ weights[1].float())
        )
    error = output.float() - expected_fp32
    first = output.clone()
    deterministic = True
    for _ in range(2):
        function()
        torch.cuda.synchronize()
        deterministic = deterministic and torch.equal(first, output)
    if not torch.isfinite(output).all():
        raise AssertionError("non-finite fused SwiGLU output")
    if not deterministic:
        raise AssertionError("non-deterministic fused SwiGLU output")

    median_ms, minimum_ms, samples_ms = measure(function)
    median_us = median_ms * 1000
    return {
        "config": asdict(config),
        "label": config.label,
        "precision": {
            "input": "float16",
            "accumulation": config.accumulation,
            "epilogue": config.epilogue,
            "output": "float16",
            "reference": "float32",
        },
        "correctness": {
            "finite": True,
            "deterministic": True,
            "meanAbs": error.abs().mean().item(),
            "rmse": error.square().mean().sqrt().item(),
            "maxAbs": error.abs().max().item(),
        },
        "timing": {
            "medianUs": median_us,
            "minimumUs": minimum_ms * 1000,
            "samplesUs": [sample * 1000 for sample in samples_ms],
            "usefulTflops": USEFUL_FLOPS / median_ms * 1e-9,
            "cutlassFp16Ratio": median_us / CUTLASS_FP16_US,
        },
        "source": source_metadata(kernel.get_kernel_source()),
    }


def run_profile(config: Config) -> None:
    data, weights = tensors()
    output = torch.empty((ROWS, COLUMNS), dtype=torch.float16, device="cuda")
    kernel = compile_kernel(config)

    def function() -> None:
        kernels.invoke(kernel, data, weights[0], weights[1], output)

    for _ in range(10):
        function()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    function()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("PROFILE complete", flush=True)


def parse_config(raw: str) -> Config:
    values = raw.split(",")
    if len(values) not in (11, 12, 13, 14):
        raise ValueError(
            "A config must contain eleven through fourteen comma-separated values"
        )
    return Config(
        int(values[0]),
        int(values[1]),
        int(values[2]),
        int(values[3]),
        int(values[4]),
        values[5],
        bool(int(values[6])),
        values[7],
        bool(int(values[8])),
        bool(int(values[9])),
        values[10],
        values[11] if len(values) >= 12 else "swiglu",
        bool(int(values[12])) if len(values) >= 13 else False,
        int(values[13]) if len(values) == 14 else 1,
    )


def worker_command(config: Config) -> list[str]:
    return [sys.executable, __file__, "--worker", "--config", config.cli()]


def result_from_stdout(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line.removeprefix("RESULT "))
    raise ValueError("worker produced no RESULT line")


def save_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def run_parent(args: argparse.Namespace) -> None:
    if args.config:
        configs = [parse_config(args.config)]
    else:
        if args.preset == "launch-bound":
            configs = list(LAUNCH_BOUND_CONFIGS)
        elif args.preset == "optimized":
            configs = list(OPTIMIZED_CONFIGS)
        elif args.preset == "intrinsics":
            configs = list(INTRINSIC_CONFIGS)
        else:
            configs = list(QUICK_CONFIGS)
        if args.preset == "full":
            configs.extend(FULL_EXTRAS)
        if args.track != "both":
            wanted = "float16" if args.track == "fp16" else "float32"
            configs = [c for c in configs if c.accumulation == wanted]

    output = Path(args.output) if args.output else RESULT_DIR / (
        f"ffn-swiglu-sm120-{args.preset}-{args.track}.json"
    )
    report = {
        "schemaVersion": 1,
        "created": datetime.now().astimezone().isoformat(),
        "shape": {"M": ROWS, "N": COLUMNS, "K": INNER},
        "usefulFlops": USEFUL_FLOPS,
        "cutlassFp16BaselineUs": CUTLASS_FP16_US,
        "timedWorkerTimeoutSeconds": TIMEOUT_SECONDS,
        "results": [],
    }

    for config in configs:
        command = worker_command(config)
        try:
            compiled = subprocess.run(
                command + ["--compile-only"],
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            row = {"label": config.label, "config": asdict(config), "status": "compile-timeout"}
            report["results"].append(row)
            save_report(report, output)
            print(f"[tl-ffn] {config.label} compile-timeout", flush=True)
            continue
        if compiled.returncode != 0:
            error = compiled.stderr.strip().splitlines()
            row = {
                "label": config.label,
                "config": asdict(config),
                "status": "compile-failed",
                "error": error[-1] if error else str(compiled.returncode),
            }
            report["results"].append(row)
            save_report(report, output)
            print(f"[tl-ffn] {config.label} compile-failed: {row['error']}", flush=True)
            continue

        timed_error = "worker failed without an error message"
        try:
            timed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            row = result_from_stdout(timed.stdout) if timed.returncode == 0 else None
        except (subprocess.TimeoutExpired, ValueError) as error:
            row = None
            timed_error = str(error)
        if row is None:
            stderr = timed.stderr.strip().splitlines() if "timed" in locals() else []
            row = {
                "label": config.label,
                "config": asdict(config),
                "status": "timed-failed",
                "error": stderr[-1] if stderr else timed_error,
            }
            print(f"[tl-ffn] {config.label} timed-failed: {row['error']}", flush=True)
        else:
            row["status"] = "ok"
            timing = row["timing"]
            correctness = row["correctness"]
            print(
                f"[tl-ffn] {config.label} {timing['medianUs']:.3f} us "
                f"{timing['usefulTflops']:.2f} TFLOP/s "
                f"MAE={correctness['meanAbs']:.7g} "
                f"RMSE={correctness['rmse']:.7g} "
                f"max={correctness['maxAbs']:.7g}",
                flush=True,
            )
        report["results"].append(row)
        save_report(report, output)

    winners = {}
    for accumulation in ("float16", "float32"):
        valid = [
            row
            for row in report["results"]
            if row.get("status") == "ok"
            and row["precision"]["accumulation"] == accumulation
        ]
        if valid:
            winners[accumulation] = min(valid, key=lambda row: row["timing"]["medianUs"])[
                "label"
            ]
    report["winners"] = winners
    save_report(report, output)
    print(f"[tl-ffn] report={output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--config")
    parser.add_argument(
        "--preset",
        choices=("quick", "full", "intrinsics", "optimized", "launch-bound"),
        default="quick",
    )
    parser.add_argument("--track", choices=("fp16", "fp32", "both"), default="both")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.worker:
        run_parent(args)
        return
    if not args.config:
        raise ValueError("--worker requires --config")
    config = parse_config(args.config)
    kernel = compile_kernel(config)
    if args.compile_only:
        print("RESULT " + json.dumps(source_metadata(kernel.get_kernel_source())), flush=True)
        return
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if args.profile:
            run_profile(config)
        else:
            print("RESULT " + json.dumps(run_timed(config)), flush=True)


if __name__ == "__main__":
    main()
