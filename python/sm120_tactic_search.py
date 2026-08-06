#!/usr/bin/env python3
"""Low-cost fixed-19x19 SM120 batch/tactic search orchestration.

This tool intentionally keeps whole-graph measurements separate from kernel
microbenchmarks.  ``grid`` finds the left edge of the throughput plateau for a
GPU.  ``space`` materializes the small candidate families documented in
BATCH-GPU-PORTABILITY.md.  Generated AOT kernels and profiler results can then
refer to the stable candidate keys emitted here.
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


# These are deliberately executable search anchors, not documentation-only
# records.  A generated space is invalid unless every search member of an
# applicable anchor is present at the exact (GPU, batch, streams) key.
ANCHORS = (
    {
        "id": "current_5090d_b13",
        "gpu_class": "rtx5090d",
        "batch": 13,
        "streams": 2,
        "reference_nn_evals_per_sec": 3800.0,
        "reference_kind": "current reproducible engineering baseline",
        "search_members": {
            "ffn": "ffn-m128-n64-k32-s2-mb3-areuse-exp",
            "qkv": "qkv-m128-n128-k64-s2-tilelang-planar",
            "linear2": "linear2-m128-n128-k32-s4-tilelang-64k",
            "l2": "l2-trunk-inner-auto",
        },
        "source": "git commit 8c7d64c and CUDA-OPTIMIZATION-PORTABILITY.md",
    },
    {
        "id": "historical_5080_b19",
        "gpu_class": "rtx5080",
        "batch": 19,
        "streams": 2,
        "reference_nn_evals_per_sec": 2862.953,
        "minimum_required_nn_evals_per_sec": 2862.953,
        "reference_kind": "locked-clock accepted historical record",
        "search_members": {
            "ffn": "ffn-m128-n64-k32-s2-mb3-tanh-half2",
            "qkv": "qkv-m128-n128-k64-s2-cute-atom4x2-planar",
            "linear2": "linear2-m128-n128-k32-s3-cutlass-49k",
            "l2": "l2-trunk-inner-auto",
        },
        "source": "cuda-optimization-history.md, final locked B19/S2 bundle",
    },
)


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
    values = [f"numNNServerThreadsPerModel={streams}"]
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


def candidate_space(batch: int, gpu_classes: tuple[str, ...]) -> dict:
    # The six FFN points are the pruned neighborhood in section 6.1.  Two
    # K64/S3 corners exceed the intended low-cost/resource envelope.
    ffn = [
        candidate("ffn-m128-n64-k32-s2-mb3-areuse-exp", m=128, n=64, k=32, stages=2, min_blocks=3, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m64-n64-k32-s2-mb4-areuse-exp", m=64, n=64, k=32, stages=2, min_blocks=4, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m128-n64-k32-s3-mb2-areuse-exp", m=128, n=64, k=32, stages=3, min_blocks=2, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m64-n64-k32-s3-mb2-areuse-exp", m=64, n=64, k=32, stages=3, min_blocks=2, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m128-n64-k64-s2-mb1-areuse-exp", m=128, n=64, k=64, stages=2, min_blocks=1, a_fragment_reuse=True, swiglu="exp"),
        candidate("ffn-m64-n64-k64-s2-mb2-areuse-exp", m=64, n=64, k=64, stages=2, min_blocks=2, a_fragment_reuse=True, swiglu="exp"),
    ]
    qkv = [
        candidate("qkv-m128-n128-k64-s2-tilelang-planar", m=128, n=128, k=64, stages=2, implementation="tilelang", output="planar"),
        candidate("qkv-m128-n128-k32-s3-tilelang-planar", m=128, n=128, k=32, stages=3, implementation="tilelang", output="planar"),
    ]
    if batch <= 10:
        qkv.append(candidate("qkv-m64-n128-k32-s3-tilelang-planar", m=64, n=128, k=32, stages=3, implementation="tilelang", output="planar"))
    linear2 = [
        candidate("linear2-m128-n128-k32-s4-tilelang-64k", m=128, n=128, k=32, stages=4, implementation="tilelang", dynamic_smem_bytes=65536),
        candidate("linear2-m128-n128-k32-s3-tilelang-49k", m=128, n=128, k=32, stages=3, implementation="tilelang", dynamic_smem_bytes=49152),
        candidate("linear2-m128-n96-k32-s4-tilelang", m=128, n=96, k=32, stages=4, implementation="tilelang"),
    ]
    l2 = [
        candidate("l2-off", trunk=False, inner=False, hit_ratio=0.0),
        candidate("l2-trunk-auto", trunk=True, inner=False, hit_ratio="auto"),
        candidate("l2-trunk-inner-auto", trunk=True, inner=True, hit_ratio="auto"),
    ]

    # Exact historical implementations remain candidates at their anchor key.
    # They may use a different arithmetic or kernel generator from the generic
    # neighborhood and therefore must not be aliased to a superficially equal
    # tile shape.
    if batch == 19 and "rtx5080" in gpu_classes:
        ffn.append(candidate("ffn-m128-n64-k32-s2-mb3-tanh-half2", m=128, n=64, k=32, stages=2, min_blocks=3, a_fragment_reuse=False, swiglu="tanh_half2", implementation="historical_tilelang"))
        qkv.append(candidate("qkv-m128-n128-k64-s2-cute-atom4x2-planar", m=128, n=128, k=64, stages=2, implementation="cute", copy_atom="4x2", output="planar"))
        linear2.append(candidate("linear2-m128-n128-k32-s3-cutlass-49k", m=128, n=128, k=32, stages=3, implementation="cutlass", dynamic_smem_bytes=49152))
    return {
        "batch": batch,
        "tokens": batch * 361,
        "ffn": deduplicate_candidates(ffn),
        "qkv": deduplicate_candidates(qkv),
        "linear2": deduplicate_candidates(linear2),
        "l2_first_round": l2,
        "l2_positive_refinement_hit_ratios": [0.5, 0.75, 1.0],
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
        if family not in ("ffn", "qkv", "linear2", "l2_first_round", "fa4", "elementwise"):
            raise ValueError(f"unsupported candidate family {family}")
        candidate_value = entry["candidate"]
        if not isinstance(candidate_value, dict) or not candidate_value.get("id"):
            raise ValueError("extra candidate requires candidate.id")
        for batch_space in space["batches"]:
            if entry_applies(entry, gpu_classes, batch_space["batch"], streams):
                batch_space.setdefault(family, []).append(candidate_value)
                batch_space[family] = deduplicate_candidates(batch_space[family])


def validate_anchor_coverage(space: dict, gpu_classes: tuple[str, ...], streams: int) -> list[dict]:
    batches = {value["batch"]: value for value in space["batches"]}
    validations: list[dict] = []
    for anchor in ANCHORS:
        if anchor["gpu_class"] not in gpu_classes or anchor["streams"] != streams:
            continue
        missing: list[str] = []
        batch_space = batches.get(anchor["batch"])
        if batch_space is None:
            missing.append(f"batch:{anchor['batch']}")
        else:
            for family, candidate_id in anchor["search_members"].items():
                family_key = "l2_first_round" if family == "l2" else family
                ids = {item["id"] for item in batch_space.get(family_key, [])}
                if candidate_id not in ids:
                    missing.append(f"{family}:{candidate_id}")
        validations.append({
            "anchor_id": anchor["id"],
            "valid": not missing,
            "missing": missing,
        })
    return validations


def write_space(args: argparse.Namespace) -> None:
    gpu_classes = selected_gpu_classes(args.gpu_class)
    batches = parse_int_set(args.batches)
    # Hard anchor batches are always materialized for the selected GPU class.
    batches = sorted(set(batches) | {
        anchor["batch"] for anchor in ANCHORS
        if anchor["gpu_class"] in gpu_classes and anchor["streams"] == args.streams
    })
    payload = {
        "schema": 2,
        "gpu_class": args.gpu_class,
        "gpu_classes": gpu_classes,
        "streams": args.streams,
        "fixed_board": [19, 19],
        "workflow_gate": "correctness -> S1/NCU -> natural whole-graph S2",
        "forbidden_proxy_gates": ["homogeneous local S2", "mixed local S2"],
        "anchors": [anchor for anchor in ANCHORS if anchor["gpu_class"] in gpu_classes],
        "batches": [candidate_space(batch, gpu_classes) for batch in batches],
    }
    extras = load_extra_candidates(args.extra_candidates)
    merge_extra_candidates(payload, extras, gpu_classes, args.streams)
    payload["extra_candidate_manifests"] = args.extra_candidates
    payload["anchor_validation"] = validate_anchor_coverage(payload, gpu_classes, args.streams)
    if not all(item["valid"] for item in payload["anchor_validation"]):
        raise RuntimeError(f"hard anchor is outside generated space: {payload['anchor_validation']}")
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
    if not all(item["valid"] for item in payload.get("anchor_validation", [])):
        raise RuntimeError("integration forbidden: hard anchor validation failed")
    batch_space = next((item for item in payload["batches"] if item["batch"] == args.batch), None)
    family = "l2_first_round" if args.family == "l2" else args.family
    ids = set() if batch_space is None else {item["id"] for item in batch_space.get(family, [])}
    if args.candidate_id not in ids:
        raise RuntimeError(
            "integration forbidden: profiler winner is outside the materialized "
            "search space; register it with register-candidate, regenerate space, "
            f"then retry ({args.family}/{args.candidate_id}, B{args.batch})"
        )
    print(f"integration gate passed: {args.family}/{args.candidate_id}, B{args.batch}")


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
