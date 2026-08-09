#!/usr/bin/env python3
"""Report the peak from a completed long-stability joint-plan measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

try:
    from sm120_benchmark_metrics import require_stable_throughput
except ModuleNotFoundError:  # imported as python.sm120_report_joint_plan
    from python.sm120_benchmark_metrics import require_stable_throughput


def report(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1 or payload.get("kind") != "sm120-joint-plan-whole-graph":
        raise ValueError(f"unsupported joint-plan result: {path}")
    rows = [row for row in payload.get("rows", []) if row.get("status") == "measured"]
    if not rows:
        raise ValueError("joint-plan result has no measured rows")
    measured = []
    for row in rows:
        measured.append({
            "batch": int(row["batch"]),
            "stable_long_nn_evals_per_sec": require_stable_throughput(row),
            "candidate_ids": {
                family: value["candidate_id"]
                for family, value in row.get("selected", {}).items()
            },
        })
    peak = max(measured, key=lambda row: row["stable_long_nn_evals_per_sec"])
    return {
        "schema": 1,
        "kind": "sm120-joint-plan-stable-report",
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plan_id": payload.get("plan_id"),
        "metric": "stable_long_nn_evals_per_sec",
        "peak_batch": peak["batch"],
        "peak_stable_long_nn_evals_per_sec": peak["stable_long_nn_evals_per_sec"],
        "peak_candidate_ids": peak["candidate_ids"],
        "curve": sorted(measured, key=lambda row: row["batch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("joint_plan")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = report(pathlib.Path(args.joint_plan).resolve())
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = pathlib.Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
