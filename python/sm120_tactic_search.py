#!/usr/bin/env python3
"""Low-cost fixed-19x19 SM120 batch/tactic search orchestration.

This tool intentionally keeps whole-graph measurements separate from kernel
microbenchmarks. ``grid`` characterizes the full requested batch range and may
also report a plateau for deployment analysis. ``space`` always materializes
the small candidate families for every explicitly requested batch; a plateau
boundary never prunes the tactic search. Generated AOT kernels and profiler
results can then refer to the stable candidate keys emitted here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shlex
import statistics
import subprocess
import sys


GPU_CLASSES = ("rtx5080", "rtx5090d")


def parse_int_set(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"invalid descending range {token}")
            result.extend(range(first, last + 1))
        else:
            result.append(int(token))
    result = sorted(set(result))
    if not result or result[0] < 1:
        raise ValueError("batch/stream sets must contain positive integers")
    return result


def last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("benchmark output did not contain a JSON object")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def override_for(device: int, streams: int, extra: str) -> str:
    values = [
        f"numNNServerThreadsPerModel={streams}",
        f"cudaPersistingL2StreamsSm120={streams}",
    ]
    for stream in range(streams):
        values.append(f"cudaDeviceToUseThread{stream}={device}")
    if extra.strip():
        values.extend(item.strip() for item in extra.split(",") if item.strip())
    return ",".join(values)


def plateau_left(rows: list[dict], threshold: float, width: int) -> dict:
    by_batch: dict[int, list[float]] = {}
    for row in rows:
        by_batch.setdefault(row["batch"], []).append(row["nn_evals_per_sec"])
    medians = {
        batch: statistics.median(values) for batch, values in by_batch.items()
    }
    peak_batch = max(medians, key=medians.get)
    peak = medians[peak_batch]
    floor = peak * threshold
    batches = sorted(medians)
    left = None
    for index, batch in enumerate(batches):
        window = batches[index : index + width]
        if len(window) < width:
            continue
        if window != list(range(batch, batch + width)):
            continue
        if all(medians[item] >= floor for item in window):
            left = batch
            break
    return {
        "threshold_fraction": threshold,
        "required_consecutive_batches": width,
        "peak_batch": peak_batch,
        "peak_nn_evals_per_sec": peak,
        "floor_nn_evals_per_sec": floor,
        "left_batch": left,
        "batch_medians": {str(key): medians[key] for key in batches},
    }


def run_grid(args: argparse.Namespace) -> None:
    batches = parse_int_set(args.batches)
    streams_set = parse_int_set(args.streams)
    output = pathlib.Path(args.output).resolve()
    raw_dir = output.parent / f"{output.stem}-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runner = shlex.split(args.runner) if args.runner else []
    rows: list[dict] = []
    started = utc_now()

    for streams in streams_set:
        override = override_for(args.device, streams, args.override_config)
        for batch in batches:
            for repeat in range(args.repeats):
                command = runner + [
                    args.binary,
                    "benchmarknn",
                    "-config",
                    args.config,
                    "-override-config",
                    override,
                    "-model",
                    args.model,
                    "-iterations",
                    str(args.iterations),
                    "-warmup",
                    str(args.warmup),
                    "-batch-size",
                    str(batch),
                    "-boardsize",
                    "19",
                    "-json",
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                prefix = raw_dir / f"s{streams}-b{batch}-r{repeat}"
                prefix.with_suffix(".out").write_text(completed.stdout)
                prefix.with_suffix(".err").write_text(completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"benchmark failed for S{streams}/B{batch}/R{repeat}; "
                        f"see {prefix}.err"
                    )
                result = last_json_object(completed.stdout)
                rows.append(
                    {
                        "streams": streams,
                        "batch": batch,
                        "repeat": repeat,
                        "nn_evals_per_sec": result["combinedNNEvalsPerSec"],
                        "combined_per_batch_ms": result["combinedPerBatchMs"],
                        "per_server_median_ms": result["perServerMedianMs"],
                        "result": result,
                        "command": command,
                    }
                )
                print(
                    f"S{streams} B{batch} R{repeat}: "
                    f"{result['combinedNNEvalsPerSec']:.3f} nn/s",
                    flush=True,
                )

    plateaus = {}
    for streams in streams_set:
        stream_rows = [row for row in rows if row["streams"] == streams]
        plateaus[str(streams)] = plateau_left(
            stream_rows, args.plateau_fraction, args.plateau_width
        )
    payload = {
        "schema": 1,
        "started_utc": started,
        "finished_utc": utc_now(),
        "fixed_shape": {"board": [19, 19], "precision": "FP16/NHWC"},
        "binary": str(pathlib.Path(args.binary).resolve()),
        "config": str(pathlib.Path(args.config).resolve()),
        "model": str(pathlib.Path(args.model).resolve()),
        "cuda_device_ordinal": args.device,
        "batches": batches,
        "streams": streams_set,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runner": runner,
        "override_config": args.override_config,
        "plateaus": plateaus,
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def candidate(candidate_id: str, **parameters: object) -> dict:
    return {"id": candidate_id, **parameters}


def deduplicate_candidates(values: list[dict]) -> list[dict]:
    result: list[dict] = []
    seen: set[str] = set()
    for value in values:
        candidate_id = value["id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        result.append(value)
    return result


def candidate_space(batch: int, gpu_class: str = "rtx5090d") -> dict:
    # The six FFN points are the pruned neighborhood in section 6.1.  Two
    # K64/S3 corners exceed the intended low-cost/resource envelope.
    ffn = [
        candidate("ffn-m128-n64-k32-s2-mb3-areuse-exp", m=128, n=64, k=32, stages=2, min_blocks=3, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m64-n64-k32-s2-mb4-areuse-exp", m=64, n=64, k=32, stages=2, min_blocks=4, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m128-n64-k32-s3-mb2-areuse-exp", m=128, n=64, k=32, stages=3, min_blocks=2, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m64-n64-k32-s3-mb2-areuse-exp", m=64, n=64, k=32, stages=3, min_blocks=2, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m128-n64-k64-s2-mb1-areuse-exp", m=128, n=64, k=64, stages=2, min_blocks=1, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m64-n64-k64-s2-mb2-areuse-exp", m=64, n=64, k=64, stages=2, min_blocks=2, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-fallback-cublas-swiglu", implementation="fallback"),
    ]
    qkv = [
        candidate("qkv-m128-n128-k64-s2-tilelang-planar", m=128, n=128, k=64, stages=2, threads=128, min_blocks=3, implementation="tilelang", output="planar"),
        candidate("qkv-m128-n128-k32-s3-tilelang-planar", m=128, n=128, k=32, stages=3, threads=128, min_blocks=3, implementation="tilelang", output="planar"),
        candidate(
            "qkv-m128-n128-k64-s2-cute-atom4x2-packed",
            m=128, n=128, k=64, stages=2, threads=288,
            implementation="cute", copy_atom="4x2", output="packed",
            max_active_clusters=84 if gpu_class == "rtx5080" else 170,
        ),
        candidate("qkv-m64-n128-k32-s3-tilelang-planar", m=64, n=128, k=32, stages=3, threads=128, min_blocks=3, implementation="tilelang", output="planar"),
        candidate("qkv-fallback-three-gemm", implementation="fallback"),
    ]
    linear2 = [
        candidate("linear2-m128-n128-k32-s4-tilelang-64k", m=128, n=128, k=32, stages=4, threads=128, min_blocks=3, implementation="tilelang", dynamic_smem_bytes=65536),
        candidate("linear2-m128-n128-k32-s3-mb2-tilelang-49k", m=128, n=128, k=32, stages=3, threads=128, min_blocks=2, implementation="tilelang", dynamic_smem_bytes=49152),
        candidate("linear2-m128-n96-k32-s4-tilelang", m=128, n=96, k=32, stages=4, threads=128, min_blocks=3, implementation="tilelang"),
        candidate("linear2-fallback-cublas-beta1", implementation="fallback"),
    ]
    l2 = [
        candidate(
            "l2-off", trunk=False, inner=False, hit_ratio=0.0,
            config={
                "cudaUsePersistingL2Trunk": False,
                "cudaUsePersistingL2Inner": False,
            },
        )
    ]
    for inner in (False, True):
        scope = "trunk-inner" if inner else "trunk"
        for ratio in (0.5, 0.75, 1.0):
            ratio_id = str(ratio).replace(".", "p")
            l2.append(candidate(
                f"l2-{scope}-ratio-{ratio_id}",
                trunk=True,
                inner=inner,
                hit_ratio=ratio,
                actual_grant_limited=True,
                config={
                    "cudaUsePersistingL2Trunk": True,
                    "cudaUsePersistingL2Inner": inner,
                    "cudaPersistingL2HitRatioSm120": ratio,
                },
            ))
    fa4 = []
    for tile_n in (64, 128):
        fa4.append(candidate(
            f"fa4-b{batch}-s361-h12-d32-tm128-tn{tile_n}-s1-both16",
            batch=batch,
            seq_len=361,
            heads=12,
            head_dim=32,
            tile_m=128,
            tile_n=tile_n,
            num_stages=1,
            accumulation="both16",
            exact_shape_aot=True,
            implementation="fa4_cute",
        ))
    fa4.append(candidate("fa4-official-attention", implementation="fallback"))

    # The previously accepted 5080 implementations are generator families,
    # not B19-only anchors. Materialize them for every requested batch so the
    # search can discover where each arithmetic/copy/mainloop choice wins.
    ffn.append(candidate("ffn-m128-n64-k32-s2-mb3-tanh-half2", m=128, n=64, k=32, stages=2, min_blocks=3, a_fragment_reuse=False, swiglu="tanh_half2", implementation="historical_tilelang"))
    return {
        "batch": batch,
        "tokens": batch * 361,
        "fa4": fa4,
        "ffn": deduplicate_candidates(ffn),
        "qkv": deduplicate_candidates(qkv),
        "linear2": deduplicate_candidates(linear2),
        "l2": l2,
    }


def selected_gpu_classes(value: str) -> tuple[str, ...]:
    if value == "all":
        return GPU_CLASSES
    if value not in GPU_CLASSES:
        raise ValueError(f"gpu class must be one of {', '.join(GPU_CLASSES)}, all")
    return (value,)


def load_json_value(value: str) -> object:
    path = pathlib.Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(value)


def load_extra_candidates(paths: list[str]) -> list[dict]:
    extras: list[dict] = []
    for path_text in paths:
        payload = json.loads(pathlib.Path(path_text).read_text())
        if payload.get("schema") != 1 or not isinstance(payload.get("entries"), list):
            raise ValueError(f"invalid extra candidate manifest {path_text}")
        extras.extend(payload["entries"])
    return extras


def entry_applies(entry: dict, gpu_classes: tuple[str, ...], batch: int, streams: int) -> bool:
    return (
        bool(set(entry["gpu_classes"]) & set(gpu_classes))
        and batch in entry["batches"]
        and streams in entry["streams"]
    )


def merge_extra_candidates(space: dict, extras: list[dict], gpu_classes: tuple[str, ...], streams: int) -> None:
    for entry in extras:
        family = entry["family"]
        if family not in ("ffn", "qkv", "linear2", "l2", "fa4", "elementwise"):
            raise ValueError(f"unsupported candidate family {family}")
        candidate_value = entry["candidate"]
        if not isinstance(candidate_value, dict) or not candidate_value.get("id"):
            raise ValueError("extra candidate requires candidate.id")
        for batch_space in space["batches"]:
            if entry_applies(entry, gpu_classes, batch_space["batch"], streams):
                batch_space.setdefault(family, []).append(candidate_value)
                batch_space[family] = deduplicate_candidates(batch_space[family])


def write_space(args: argparse.Namespace) -> None:
    gpu_classes = selected_gpu_classes(args.gpu_class)
    batches = parse_int_set(args.batches)
    payload = {
        "schema": 2,
        "gpu_class": args.gpu_class,
        "gpu_classes": gpu_classes,
        "streams": args.streams,
        "fixed_board": [19, 19],
        "workflow_gate": (
            "correctness + local CUDA-event timing -> greedy local-best bundle "
            "curve -> natural whole-graph S2 coordinate scans near the plateau; "
            "GPU performance counters are optional explanation aids"
        ),
        "forbidden_proxy_gates": ["homogeneous local S2", "mixed local S2"],
        "batch_policy": "only explicitly requested batches; no implicit anchors",
        "batches": [candidate_space(batch, args.gpu_class) for batch in batches],
    }
    extras = load_extra_candidates(args.extra_candidates)
    merge_extra_candidates(payload, extras, gpu_classes, args.streams)
    payload["extra_candidate_manifests"] = args.extra_candidates
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)


def register_candidate(args: argparse.Namespace) -> None:
    path = pathlib.Path(args.manifest)
    payload = {"schema": 1, "entries": []}
    if path.exists():
        payload = json.loads(path.read_text())
    if payload.get("schema") != 1 or not isinstance(payload.get("entries"), list):
        raise ValueError("candidate manifest must have schema=1 and entries=[]")
    candidate_value = load_json_value(args.candidate_json)
    if not isinstance(candidate_value, dict) or not candidate_value.get("id"):
        raise ValueError("--candidate-json must be an object with a stable id")
    entry = {
        "registered_utc": utc_now(),
        "gpu_classes": list(selected_gpu_classes(args.gpu_class)),
        "batches": parse_int_set(args.batches),
        "streams": parse_int_set(args.streams),
        "family": args.family,
        "candidate": candidate_value,
        "reason": args.reason,
        "profiler_artifact": args.profiler_artifact,
    }
    payload["entries"] = [
        item for item in payload["entries"]
        if not (
            item["family"] == entry["family"]
            and item["candidate"]["id"] == candidate_value["id"]
            and item["gpu_classes"] == entry["gpu_classes"]
            and item["batches"] == entry["batches"]
            and item["streams"] == entry["streams"]
        )
    ]
    payload["entries"].append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"registered {candidate_value['id']} in {path}")


def check_winner(args: argparse.Namespace) -> None:
    payload = json.loads(pathlib.Path(args.space).read_text())
    if payload.get("schema") != 2:
        raise ValueError("winner checks require a schema-2 generated space")
    batch_space = next((item for item in payload["batches"] if item["batch"] == args.batch), None)
    ids = set() if batch_space is None else {
        item["id"] for item in batch_space.get(args.family, [])
    }
    if args.candidate_id not in ids:
        raise RuntimeError(
            "integration forbidden: profiler winner is outside the materialized "
            "search space; register it with register-candidate, regenerate space, "
            f"then retry ({args.family}/{args.candidate_id}, B{args.batch})"
        )
    print(f"integration gate passed: {args.family}/{args.candidate_id}, B{args.batch}")


def build_generation_plan(args: argparse.Namespace) -> None:
    payload = json.loads(pathlib.Path(args.space).read_text())
    if payload.get("schema") != 2:
        raise ValueError("generation plans require a schema-2 space")
    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    supported = ("fa4", "ffn", "qkv", "linear2")
    if not families or any(item not in supported for item in families):
        raise ValueError(f"--families must be a subset of {supported}")

    seed_ids = {
        "ffn": "ffn-m128-n64-k32-s2-mb3-areuse-exp",
        "qkv": "qkv-m128-n128-k64-s2-tilelang-planar",
        "linear2": "linear2-m128-n128-k32-s4-tilelang-64k",
    }
    tasks = []
    for batch_space in payload["batches"]:
        batch = batch_space["batch"]
        for family in families:
            values = batch_space.get(family, [])
            if args.phase == "seed":
                if family == "fa4":
                    values = values[:1]
                else:
                    values = [item for item in values if item["id"] == seed_ids[family]]
            for item in values:
                if family == "fa4":
                    generator = "cpp/neuralnet/fa4_aot/build_aot.py"
                elif item.get("implementation") == "cute":
                    generator = "python/sm120_generate_cute_qkv_aot.py"
                else:
                    generator = "python/sm120_generate_tilelang_aot.py"
                tasks.append({
                    "gpu_classes": payload["gpu_classes"],
                    "streams": payload["streams"],
                    "batch": batch,
                    "family": family,
                    "candidate_id": item["id"],
                    "candidate": item,
                    "generator": generator,
                    "acceptance_metric": "natural whole-graph S2 total throughput",
                })
    planned = {
        "schema": 1,
        "source_space": str(pathlib.Path(args.space).resolve()),
        "phase": args.phase,
        "fixed_board": [19, 19],
        "batch_policy": "all materialized batches; no anchor-first special case",
        "tasks": tasks,
    }
    text_value = json.dumps(planned, indent=2) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(text_value)
    else:
        sys.stdout.write(text_value)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    grid = subparsers.add_parser("grid", help="scan natural whole-graph batches")
    grid.add_argument("--binary", required=True)
    grid.add_argument("--config", required=True)
    grid.add_argument("--model", required=True)
    grid.add_argument("--device", type=int, required=True)
    grid.add_argument("--batches", default="1-32")
    grid.add_argument("--streams", default="2")
    grid.add_argument("--iterations", type=int, default=200)
    grid.add_argument("--warmup", type=int, default=20)
    grid.add_argument("--repeats", type=int, default=3)
    grid.add_argument("--runner", default="")
    grid.add_argument("--override-config", default="")
    grid.add_argument("--plateau-fraction", type=float, default=0.99)
    grid.add_argument("--plateau-width", type=int, default=2)
    grid.add_argument("--output", required=True)
    grid.set_defaults(function=run_grid)

    space = subparsers.add_parser("space", help="emit the pruned tactic family")
    space.add_argument("--gpu-class", required=True)
    space.add_argument("--batches", required=True)
    space.add_argument("--streams", type=int, default=2)
    space.add_argument("--extra-candidates", action="append", default=[])
    space.add_argument("--output")
    space.set_defaults(function=write_space)

    register = subparsers.add_parser(
        "register-candidate",
        help="record a profiler-discovered out-of-space candidate",
    )
    register.add_argument("--manifest", required=True)
    register.add_argument("--gpu-class", required=True)
    register.add_argument("--batches", required=True)
    register.add_argument("--streams", default="2")
    register.add_argument("--family", required=True)
    register.add_argument("--candidate-json", required=True)
    register.add_argument("--reason", required=True)
    register.add_argument("--profiler-artifact", required=True)
    register.set_defaults(function=register_candidate)

    check = subparsers.add_parser(
        "check-winner",
        help="block integration until a winning candidate is in the exact space",
    )
    check.add_argument("--space", required=True)
    check.add_argument("--batch", type=int, required=True)
    check.add_argument("--family", required=True)
    check.add_argument("--candidate-id", required=True)
    check.set_defaults(function=check_winner)

    generation = subparsers.add_parser(
        "generation-plan",
        help="materialize seed or full AOT tasks for every selected batch",
    )
    generation.add_argument("--space", required=True)
    generation.add_argument("--phase", choices=("seed", "full"), default="seed")
    generation.add_argument("--families", default="fa4,ffn,qkv,linear2")
    generation.add_argument("--output")
    generation.set_defaults(function=build_generation_plan)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if hasattr(args, "plateau_fraction") and not 0.0 < args.plateau_fraction <= 1.0:
        parser.error("--plateau-fraction must be in (0,1]")
    if hasattr(args, "plateau_width") and args.plateau_width < 1:
        parser.error("--plateau-width must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
