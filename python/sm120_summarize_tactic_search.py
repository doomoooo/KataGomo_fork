#!/usr/bin/env python3
"""Summarize resumable SM120 whole-graph tactic scans by exact batch.

Multiple result files may be supplied for follow-up candidate families.  For a
duplicate (GPU class, family, batch, candidate) key the newest completed
measurement wins.  Failed and unsupported rows remain visible in the JSON so a
partial scan cannot be mistaken for full coverage.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fallback_candidate(row: dict) -> bool:
    return (
        row.get("implementation") == "fallback"
        or "fallback" in row["candidate_id"]
        or row["candidate_id"] == "l2-off"
    )


def load_rows(paths: list[pathlib.Path]) -> tuple[list[dict], list[dict]]:
    latest: dict[tuple, dict] = {}
    files = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("schema") != 1:
            raise ValueError(f"unsupported result schema in {path}")
        gpu_class = payload["gpu_class"]
        family = payload["family"]
        files.append({
            "path": str(path.resolve()),
            "gpu_class": gpu_class,
            "family": family,
            "started_utc": payload.get("started_utc"),
            "finished_utc": payload.get("finished_utc"),
        })
        for row in payload["rows"]:
            item = dict(row)
            item["gpu_class"] = gpu_class
            item["family"] = family
            item["source_result"] = str(path.resolve())
            key = (gpu_class, family, row["batch"], row["candidate_id"])
            if key not in latest or item.get("finished_utc", "") >= latest[key].get("finished_utc", ""):
                latest[key] = item
    return list(latest.values()), files


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["gpu_class"], row["family"], row["batch"]), []).append(row)
    output = []
    for (gpu_class, family, batch), group in sorted(groups.items()):
        measured = [row for row in group if row.get("status") == "measured"]
        measured.sort(key=lambda row: row["nn_evals_per_sec_median"], reverse=True)
        fallbacks = [row for row in measured if fallback_candidate(row)]
        baseline = max(fallbacks, key=lambda row: row["nn_evals_per_sec_median"], default=None)
        winner = measured[0] if measured else None
        baseline_rate = None if baseline is None else baseline["nn_evals_per_sec_median"]
        winner_rate = None if winner is None else winner["nn_evals_per_sec_median"]
        output.append({
            "gpu_class": gpu_class,
            "family": family,
            "batch": batch,
            "candidate_count": len(group),
            "measured_count": len(measured),
            "failed": [row["candidate_id"] for row in group if row.get("status") == "failed"],
            "unsupported": [row["candidate_id"] for row in group if row.get("status") == "unsupported_generator"],
            "winner": None if winner is None else winner["candidate_id"],
            "winner_nn_evals_per_sec": winner_rate,
            "baseline": None if baseline is None else baseline["candidate_id"],
            "baseline_nn_evals_per_sec": baseline_rate,
            "winner_vs_baseline_percent": (
                None if baseline_rate is None or winner_rate is None
                else (winner_rate / baseline_rate - 1.0) * 100.0
            ),
            "ranking": [
                {
                    "candidate_id": row["candidate_id"],
                    "nn_evals_per_sec_median": row["nn_evals_per_sec_median"],
                    "sample_count": len(row.get("nn_evals_per_sec_samples", [])),
                }
                for row in measured
            ],
        })
    return output


def render_markdown(groups: list[dict]) -> str:
    lines = [
        "# SM120 exact-batch tactic scan summary",
        "",
        "All rates are natural whole-graph S2 total throughput. Short-scan winners are",
        "ranking inputs, not accepted optimizations; acceptance still requires long S2",
        "and accuracy replay. Nsys/NCU are optional explanation aids, not gates.",
        "",
        "| GPU | Family | Batch | Coverage | Winner | nnEval/s | Fallback | Delta |",
        "|---|---|---:|---:|---|---:|---|---:|",
    ]
    for item in groups:
        coverage = f"{item['measured_count']}/{item['candidate_count']}"
        rate = "n/a" if item["winner_nn_evals_per_sec"] is None else f"{item['winner_nn_evals_per_sec']:.3f}"
        delta = "n/a" if item["winner_vs_baseline_percent"] is None else f"{item['winner_vs_baseline_percent']:+.3f}%"
        lines.append(
            f"| {item['gpu_class']} | {item['family']} | {item['batch']} | {coverage} | "
            f"{item['winner'] or 'n/a'} | {rate} | {item['baseline'] or 'n/a'} | {delta} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    paths = [pathlib.Path(value).resolve() for value in args.results]
    rows, files = load_rows(paths)
    groups = summarize(rows)
    payload = {
        "schema": 1,
        "generated_utc": utc_now(),
        "acceptance_metric": "natural whole-graph S2 total throughput",
        "source_files": files,
        "groups": groups,
        "status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in ("measured", "failed", "unsupported_generator")
        },
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown:
        pathlib.Path(args.markdown).write_text(render_markdown(groups))


if __name__ == "__main__":
    main()
