#!/usr/bin/env python3
"""Validate a completed SM120 tactic plan with interleaved whole-graph runs.

The short search ranks candidates independently.  This tool compares the plan
winner (A) with a control candidate (B) in ABBA and/or BAAB order using the
normal ``benchmarknn`` graph.  It deliberately does not turn timing into a
proxy acceptance gate: the output is evidence for the final review and can be
resumed after an interruption.
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

try:
    from sm120_benchmark_metrics import benchmark_throughput, summarize_throughput
    from sm120_run_tactic_search import (
        collect_environment,
        full_override,
        last_json_object,
        load_plan,
        parse_int_set,
        sha256_file,
        utc_now,
    )
    from sm120_tactic_plan import FAMILIES, validate_plan
except ModuleNotFoundError:  # imported as ``python.sm120_validate_tactic_plan``
    from python.sm120_benchmark_metrics import benchmark_throughput, summarize_throughput
    from python.sm120_run_tactic_search import (
        collect_environment,
        full_override,
        last_json_object,
        load_plan,
        parse_int_set,
        sha256_file,
        utc_now,
    )
    from python.sm120_tactic_plan import FAMILIES, validate_plan


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run_benchmark(
    command: list[str], log_path: pathlib.Path,
) -> tuple[float, dict]:
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.with_suffix(".out").write_text(completed.stdout)
    log_path.with_suffix(".err").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}); see "
            f"{log_path.with_suffix('.err')}"
        )
    result = last_json_object(completed.stdout)
    return benchmark_throughput(result), result


def candidate_by_id(space: dict, family: str, batch: int, candidate_id: str) -> dict:
    for batch_space in space.get("batches", []):
        if int(batch_space.get("batch", -1)) != batch:
            continue
        for candidate in batch_space.get(family, []):
            if candidate.get("id") == candidate_id:
                return candidate
    raise ValueError(f"candidate {family}/B{batch}/{candidate_id} is absent from space")


def plan_candidate(plan: dict, family: str, batch: int) -> dict:
    entry = plan.get("families", {}).get(family, {}).get("batches", {}).get(
        str(batch)
    )
    if not isinstance(entry, dict) or not entry.get("candidate_id"):
        raise ValueError(f"plan has no {family}/B{batch} candidate")
    candidate = entry.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"plan has no candidate parameters for {family}/B{batch}")
    return candidate


def sequence_for(name: str) -> list[str]:
    if name == "abba":
        return ["A", "B", "B", "A"]
    if name == "baab":
        return ["B", "A", "A", "B"]
    if name == "both":
        return ["A", "B", "B", "A", "B", "A", "A", "B"]
    raise ValueError(f"unsupported validation order: {name}")


def median_for_arm(values: list[float], arm: str) -> float:
    selected = [value for value, observed_arm in values if observed_arm == arm]
    if not selected:
        raise ValueError(f"validation sequence has no {arm} observations")
    return statistics.median(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--control-candidate-id", default="")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batches", default="4-32")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order", choices=("abba", "baab", "both"), default="both")
    parser.add_argument("--runner", default="")
    parser.add_argument("--override-config", default="")
    parser.add_argument(
        "--cutlass-root", default="/workspace/third_party/cutlass",
    )
    parser.add_argument(
        "--fa4-python", default=sys.executable,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan_path = pathlib.Path(args.plan).resolve()
    space_path = pathlib.Path(args.space).resolve()
    plan = load_plan(plan_path)
    space = json.loads(space_path.read_text())
    space["_path"] = str(space_path)
    batches = parse_int_set(args.batches)
    config_path = pathlib.Path(args.config).resolve()
    model_path = pathlib.Path(args.model).resolve()
    binary_path = pathlib.Path(args.binary).resolve()
    repo = pathlib.Path(args.repo).resolve()
    if not binary_path.is_file():
        raise ValueError(f"binary does not exist: {binary_path}")

    # validate_plan also checks the model/config/search-space identity and the
    # full-plan readiness.  Validate all families because this tool pins the
    # untouched families to the plan while comparing one family.
    for family in FAMILIES:
        validate_plan(
            plan, space, model_path, family, batches, args.streams, config_path,
        )

    control_ids = {}
    for batch in batches:
        winner = plan_candidate(plan, args.family, batch)
        winner_id = winner["id"]
        control_id = args.control_candidate_id
        if not control_id:
            # Do not assume that the space is sorted or starts at B1.
            batch_space = next(
                (
                    item for item in space.get("batches", [])
                    if int(item.get("batch", -1)) == batch
                ),
                None,
            )
            candidates = [
                item for item in (batch_space or {}).get(args.family, [])
                if item.get("implementation") == "fallback"
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"cannot infer one fallback control for {args.family}/B{batch}"
                )
            control_id = candidates[0]["id"]
        control = candidate_by_id(space, args.family, batch, control_id)
        if winner_id == control["id"]:
            raise ValueError(f"winner and control are identical at B{batch}")
        control_ids[str(batch)] = control["id"]

    runner = shlex.split(args.runner)
    orders = sequence_for(args.order)
    output = pathlib.Path(args.output).resolve()
    logs = output.parent / f"{output.stem}-logs"
    regime = {
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "space": str(space_path),
        "space_sha256": sha256_file(space_path),
        "family": args.family,
        "batches": batches,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "binary": str(binary_path),
        "binary_sha256": sha256_file(binary_path),
        "cuda_device_ordinal": args.device,
        "streams": args.streams,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "orders": orders,
        "runner": runner,
        "extra_override_config": args.override_config,
    }
    if output.exists():
        payload = json.loads(output.read_text())
        if payload.get("regime") != regime:
            raise ValueError("validation output exists with a different regime")
    else:
        environment = collect_environment(
            repo, config_path, model_path,
            {"cutlass": pathlib.Path(args.cutlass_root).resolve()},
            args.fa4_python,
        )
        payload = {
            "schema": 1,
            "kind": "sm120-abba-baab-validation",
            "started_utc": utc_now(),
            "plan_id": plan["plan_id"],
            "regime": regime,
            "winner_by_batch": {
                str(batch): plan_candidate(plan, args.family, batch)["id"]
                for batch in batches
            },
            "control_by_batch": control_ids,
            "rows": [],
            "environment_snapshots": [environment],
            "acceptance_policy": {
                "metric": "long-stable natural whole-graph benchmarknn combinedNNEvalsPerSec",
                "orders": ["ABBA", "BAAB"] if args.order == "both" else [args.order.upper()],
                "decision": "report stable_long_nn_evals_per_sec only after long ABBA/BAAB; correctness replay remains required",
            },
        }

    completed_batches = {
        int(row["batch"]) for row in payload.get("rows", [])
        if row.get("status") == "measured"
    }
    for batch in batches:
        if batch in completed_batches:
            continue
        winner = plan_candidate(plan, args.family, batch)
        winner_candidate = candidate_by_id(
            space, args.family, batch, winner["id"]
        )
        control_candidate = candidate_by_id(
            space, args.family, batch, control_ids[str(batch)]
        )
        overrides = {
            "A": full_override(
                args.family, winner_candidate, args.device, args.streams,
                args.override_config, False, plan, batch,
            ),
            "B": full_override(
                args.family, control_candidate, args.device, args.streams,
                args.override_config, False, plan, batch,
            ),
        }
        row = {
            "batch": batch,
            "winner_candidate_id": winner_candidate["id"],
            "control_candidate_id": control_candidate["id"],
            "started_utc": utc_now(),
            "orders": [],
        }
        try:
            for repeat in range(args.repeats):
                order_values = []
                order_records = []
                for index, arm in enumerate(orders):
                    command = runner + [
                        str(binary_path), "benchmarknn",
                        "-config", str(config_path),
                        "-override-config", overrides[arm],
                        "-model", str(model_path),
                        "-iterations", str(args.iterations),
                        "-warmup", str(args.warmup),
                        "-batch-size", str(batch),
                        "-boardsize", "19", "-json",
                    ]
                    log_prefix = logs / f"b{batch}-r{repeat}-{index:02d}-{arm}"
                    value, record = run_benchmark(command, log_prefix)
                    order_values.append((value, arm))
                    order_records.append({
                        "arm": arm,
                        "value": value,
                        "command": command,
                        "benchmark_record": record,
                        "log_prefix": str(log_prefix),
                    })
                row["orders"].append({
                    "sequence": orders,
                    "records": order_records,
                    "A_median": median_for_arm(order_values, "A"),
                    "B_median": median_for_arm(order_values, "B"),
                })
            a_values = [item["A_median"] for item in row["orders"]]
            b_values = [item["B_median"] for item in row["orders"]]
            a_median = statistics.median(a_values)
            b_median = statistics.median(b_values)
            row.update({
                "status": "measured",
                "finished_utc": utc_now(),
                "A_median": a_median,
                "B_median": b_median,
                "winner_delta_percent_vs_control": (
                    (a_median / b_median - 1.0) * 100.0
                    if b_median else None
                ),
                "winner_beats_control": a_median > b_median,
            })
            row.update(
                summarize_throughput(
                    a_values, iterations=args.iterations, warmup=args.warmup,
                )
            )
            print(
                f"B{batch} {winner_candidate['id']} vs {control_candidate['id']}: "
                f"A={a_median:.3f} B={b_median:.3f} "
                f"delta={row['winner_delta_percent_vs_control']:.3f}%",
                flush=True,
            )
        except Exception as error:
            row.update({
                "status": "failed",
                "finished_utc": utc_now(),
                "error": str(error),
            })
            print(f"B{batch}: FAILED: {error}", flush=True)
        payload["rows"].append(row)
        payload["finished_utc"] = utc_now()
        write_json(output, payload)


if __name__ == "__main__":
    main()
