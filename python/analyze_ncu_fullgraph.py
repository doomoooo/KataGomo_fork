#!/usr/bin/env python3
"""Join a fixed-forward NCU capture with the matching Nsys ordinal report."""

import argparse
import csv
import json
import math
import statistics


METRICS = {
    "registers_per_thread": "launch__registers_per_thread",
    "dynamic_smem_kib": "launch__shared_mem_per_block_dynamic",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_cycles_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_throughput_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "wait_per_issue": "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "long_scoreboard_per_issue": (
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
    ),
}


def numeric(row, column):
    value = row.get(column, "")
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def median(values):
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_ncu(path):
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        next(reader)  # NCU emits a units row after the header.
        rows = list(reader)

    by_ordinal = {}
    for row in rows:
        ordinal = int(row["ID"])
        if ordinal in by_ordinal:
            raise RuntimeError(f"duplicate NCU result ID {ordinal}")
        by_ordinal[ordinal] = row
    expected = list(range(len(rows)))
    if sorted(by_ordinal) != expected:
        raise RuntimeError("NCU result IDs are not one contiguous fixed forward")
    return by_ordinal


def summarize(nsys, ncu):
    calls_per_ordinal = nsys["iterations"] * len(nsys["streams"])
    if calls_per_ordinal <= 0:
        raise RuntimeError("invalid Nsys iteration/stream count")
    if len(ncu) != len(nsys["ordinals"]):
        raise RuntimeError(
            f"NCU has {len(ncu)} ordinals, Nsys has {len(nsys['ordinals'])}"
        )

    total_ms = sum(group["total_ms"] for group in nsys["logical_groups"])
    groups = []
    for group in nsys["logical_groups"]:
        rows = [ncu[ordinal] for ordinal in group["ordinals"]]
        resource = {
            name: median([numeric(row, column) for row in rows])
            for name, column in METRICS.items()
        }
        groups.append({
            "logical_group": group["logical_group"],
            "positions": group["positions"],
            "families": group["families"],
            "s2_work_share_pct": group["total_ms"] / total_ms * 100.0,
            "s2_total_us_per_stream_forward": group["total_ms"] / calls_per_ordinal * 1e3,
            "s2_excess_us_per_stream_forward": (
                group["summed_excess_ms"] / calls_per_ordinal * 1e3
            ),
            "s2_to_s1_ratio": group["total_slowdown"],
            "ncu_median": resource,
        })
    return groups


def markdown(payload, limit):
    groups = payload["groups"]
    lines = [
        "# Current full-graph Nsys + NCU ranking",
        "",
        f"- NCU fixed-forward coverage: {payload['ncu_ordinals']} ordinals.",
        f"- Nsys timed topology: {payload['nsys_iterations']} iterations x "
        f"{payload['nsys_streams']} streams.",
        "- Nsys supplies S2 timing and interference weight. NCU is replayed S1 evidence "
        "for resources and stalls; its replay duration is not used as S2 performance.",
        "",
        "## Largest S2 work",
        "",
        "| logical group | work us/fwd | work share | excess us/fwd | S2/S1 | regs | "
        "smem KiB | waves/SM | occ % | eligible % | tensor % | wait/issue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(
        groups, key=lambda item: item["s2_total_us_per_stream_forward"], reverse=True
    )[:limit]:
        ncu = group["ncu_median"]
        lines.append(
            f"| {group['logical_group']} | "
            f"{group['s2_total_us_per_stream_forward']:.1f} | "
            f"{group['s2_work_share_pct']:.2f}% | "
            f"{group['s2_excess_us_per_stream_forward']:.1f} | "
            f"{group['s2_to_s1_ratio']:.3f}x | "
            f"{fmt(ncu['registers_per_thread'], 0)} | "
            f"{fmt(ncu['dynamic_smem_kib'])} | {fmt(ncu['waves_per_sm'], 2)} | "
            f"{fmt(ncu['achieved_occupancy_pct'])} | "
            f"{fmt(ncu['eligible_cycles_pct'])} | "
            f"{fmt(ncu['tensor_throughput_pct'])} | {fmt(ncu['wait_per_issue'], 2)} |"
        )

    lines += [
        "",
        "## Largest S2 interference excess",
        "",
        "| logical group | excess us/fwd | work us/fwd | S2/S1 | waves/SM | "
        "eligible % | long-scoreboard/issue |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(
        groups, key=lambda item: item["s2_excess_us_per_stream_forward"], reverse=True
    )[:limit]:
        ncu = group["ncu_median"]
        lines.append(
            f"| {group['logical_group']} | "
            f"{group['s2_excess_us_per_stream_forward']:.1f} | "
            f"{group['s2_total_us_per_stream_forward']:.1f} | "
            f"{group['s2_to_s1_ratio']:.3f}x | {fmt(ncu['waves_per_sm'], 2)} | "
            f"{fmt(ncu['eligible_cycles_pct'])} | "
            f"{fmt(ncu['long_scoreboard_per_issue'], 2)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-json", required=True)
    parser.add_argument("--ncu-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    with open(args.nsys_json, encoding="utf-8") as json_file:
        nsys = json.load(json_file)
    ncu = load_ncu(args.ncu_csv)
    payload = {
        "nsys_json": args.nsys_json,
        "ncu_csv": args.ncu_csv,
        "nsys_iterations": nsys["iterations"],
        "nsys_streams": len(nsys["streams"]),
        "ncu_ordinals": len(ncu),
        "groups": summarize(nsys, ncu),
    }
    with open(args.output_json, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    with open(args.output_md, "w", encoding="utf-8") as md_file:
        md_file.write(markdown(payload, args.limit))


if __name__ == "__main__":
    main()
