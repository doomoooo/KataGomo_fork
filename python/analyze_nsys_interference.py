#!/usr/bin/env python3
"""Build a per-kernel cross-stream interference report from an Nsys SQLite export."""

import argparse
import bisect
import collections
import json
import math
import sqlite3
import statistics


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def family(name, short_name):
    lowered = name.lower()
    if "fused_ffn" in lowered:
        return "fused_ffn"
    if "linear2_residual" in lowered:
        return "linear2_residual"
    if "wide_qkv" in lowered:
        return "wide_qkv"
    if "flash_attn" in lowered or "flashattention" in lowered:
        return "fa4"
    if "rmsnorm384" in lowered:
        return "rmsnorm"
    if "fusedqkrope" in lowered:
        return "qk_rope"
    if "swiglu" in lowered:
        return "swiglu"
    if "affinesilu" in lowered:
        return "affine_silu"
    if ("cutlass::kernel" in lowered or "nvjet_" in lowered or
            "gemmsn_" in lowered or "gemvnsp_" in lowered or
            "splitkreduce" in lowered):
        return "library_gemm"
    if "cudnn" in lowered:
        return "cudnn"
    if "copy" in lowered or "padding" in lowered:
        return "copy_reformat"
    if "bias" in lowered or "pool" in lowered or "extractchannel" in lowered:
        return "head_elementwise"
    return short_name or name.split("(", 1)[0]


def kernel_signature(row):
    return "|".join([
        row["family"],
        row["short_name"],
        f'g{row["grid_x"]}x{row["grid_y"]}x{row["grid_z"]}',
        f'b{row["block_x"]}x{row["block_y"]}x{row["block_z"]}',
        f'r{row["registers"]}',
        f's{row["static_smem"] + row["dynamic_smem"]}',
    ])


def choose_streams(conn, count):
    rows = conn.execute(
        """
        SELECT streamId, COUNT(*) AS instances
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId
        ORDER BY instances DESC, streamId
        LIMIT ?
        """,
        (count,),
    ).fetchall()
    if len(rows) != count:
        raise RuntimeError(f"expected {count} active streams, found {len(rows)}")
    return [row[0] for row in rows]


def benchmark_thread_for_stream(conn, stream_id):
    row = conn.execute(
        """
        SELECT r.globalTid, COUNT(*) AS launches
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        WHERE k.streamId = ?
        GROUP BY r.globalTid
        ORDER BY launches DESC
        LIMIT 1
        """,
        (stream_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"could not map stream {stream_id} to a host thread")
    return row[0]


def iteration_launch_ranges(conn, global_tid, iterations):
    rows = conn.execute(
        """
        SELECT r.start, r.end
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE r.globalTid = ? AND s.value = 'cudaEventRecord_ptsz_v7000'
        ORDER BY r.start
        """,
        (global_tid,),
    ).fetchall()
    required = iterations * 2
    if len(rows) < required:
        raise RuntimeError(
            f"thread {global_tid} has {len(rows)} benchmark event records, need {required}"
        )
    rows = rows[-required:]
    ranges = []
    for iteration in range(iterations):
        start_record = rows[2 * iteration]
        end_record = rows[2 * iteration + 1]
        if start_record[0] >= end_record[0]:
            raise RuntimeError("benchmark event records are not ordered")
        ranges.append((start_record[1], end_record[0]))
    return ranges


def load_timed_kernels(conn, stream_id, iterations):
    global_tid = benchmark_thread_for_stream(conn, stream_id)
    launch_ranges = iteration_launch_ranges(conn, global_tid, iterations)
    range_starts = [item[0] for item in launch_ranges]
    rows = conn.execute(
        """
        SELECT
          k.start, k.end, k.streamId,
          demangled.value, short.value,
          k.gridX, k.gridY, k.gridZ,
          k.blockX, k.blockY, k.blockZ,
          k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
          r.start
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds demangled ON demangled.id = k.demangledName
        JOIN StringIds short ON short.id = k.shortName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
        WHERE k.streamId = ? AND r.globalTid = ?
        ORDER BY k.start
        """,
        (stream_id, global_tid),
    ).fetchall()

    kernels = []
    per_iteration = collections.Counter()
    for values in rows:
        launch_start = values[14]
        iteration = bisect.bisect_right(range_starts, launch_start) - 1
        if iteration < 0:
            continue
        launch_range = launch_ranges[iteration]
        if launch_start > launch_range[1]:
            continue
        row = {
            "start": values[0],
            "end": values[1],
            "duration": values[1] - values[0],
            "stream": values[2],
            "name": values[3],
            "short_name": values[4],
            "grid_x": values[5],
            "grid_y": values[6],
            "grid_z": values[7],
            "block_x": values[8],
            "block_y": values[9],
            "block_z": values[10],
            "registers": values[11],
            "static_smem": values[12],
            "dynamic_smem": values[13],
            "iteration": iteration,
            "ordinal": per_iteration[iteration],
        }
        row["family"] = family(row["name"], row["short_name"])
        row["signature"] = kernel_signature(row)
        kernels.append(row)
        per_iteration[iteration] += 1

    if len(per_iteration) != iterations:
        raise RuntimeError(
            f"stream {stream_id}: found kernels for {len(per_iteration)}/{iterations} iterations"
        )
    counts = list(per_iteration.values())
    if min(counts) != max(counts):
        raise RuntimeError(
            f"stream {stream_id}: timed kernel count varies from {min(counts)} to {max(counts)}"
        )
    return kernels, counts[0]


def isolated_medians(sqlite_path, iterations):
    if sqlite_path is None:
        return {}
    conn = sqlite3.connect(sqlite_path)
    try:
        stream = choose_streams(conn, 1)[0]
        kernels, _ = load_timed_kernels(conn, stream, iterations)
    finally:
        conn.close()
    grouped = collections.defaultdict(list)
    for kernel in kernels:
        grouped[(kernel["ordinal"], kernel["signature"])].append(kernel["duration"])
    return {key: statistics.median(values) for key, values in grouped.items()}


def find_overlaps(kernels, peers):
    peer_ends = [peer["end"] for peer in peers]
    for kernel in kernels:
        index = bisect.bisect_right(peer_ends, kernel["start"])
        overlaps = []
        while index < len(peers) and peers[index]["start"] < kernel["end"]:
            peer = peers[index]
            overlap = min(kernel["end"], peer["end"]) - max(kernel["start"], peer["start"])
            if overlap > 0:
                overlaps.append((peer, overlap))
            index += 1
        kernel["overlaps"] = overlaps
        kernel["overlap_ns"] = sum(item[1] for item in overlaps)
        if overlaps:
            kernel["dominant_peer"] = max(overlaps, key=lambda item: item[1])[0]["family"]
        else:
            kernel["dominant_peer"] = "idle"


def logical_positions(ordinal_families):
    """Infer stable model positions from the fixed per-forward kernel sequence."""
    positions = {
        ordinal: f"unclassified.ordinal_{ordinal}.{name}"
        for ordinal, name in ordinal_families.items()
    }

    frontend = {
        0: "input.extract_mask",
        1: "input.mask_half_to_float",
        2: "input.mask_sum",
        3: "frontend.initial_conv_nhwc_padding_0",
        4: "frontend.initial_conv_nhwc_padding_1",
        5: "frontend.initial_conv",
        6: "frontend.initial_global_matmul",
        7: "frontend.initial_global_matmul_splitk_reduce",
        8: "frontend.initial_global_broadcast_add",
    }
    for ordinal, name in frontend.items():
        if ordinal in positions:
            positions[ordinal] = name

    qkv_ordinals = sorted(
        ordinal for ordinal, name in ordinal_families.items() if name == "wide_qkv"
    )
    transformer_offsets = {
        -1: ("rmsnorm", "attention_rmsnorm"),
        0: ("wide_qkv", "attention_qkv_projection"),
        1: ("qk_rope", "attention_qk_rope"),
        2: ("fa4", "attention_fa4"),
        3: ("library_gemm", "attention_out_projection_residual"),
        4: ("rmsnorm", "ffn_rmsnorm"),
        5: ("fused_ffn", "ffn_linear1_gate_swiglu"),
        6: ("linear2_residual", "ffn_linear2_residual"),
    }
    for block, qkv_ordinal in enumerate(qkv_ordinals):
        outer = block // 3
        inner = block % 3
        prefix = f"outer_{outer:02d}.transformer_{inner}.block_{block:02d}"
        for offset, (expected_family, operation) in transformer_offsets.items():
            ordinal = qkv_ordinal + offset
            if ordinal_families.get(ordinal) == expected_family:
                positions[ordinal] = f"{prefix}.{operation}"

    # This network has three inner Transformer blocks per outer bottleneck.
    # Infer the two boundary projections from those inner-block boundaries and
    # only label them when their observed families match the expected sequence.
    for outer, first_block in enumerate(range(0, len(qkv_ordinals), 3)):
        group = qkv_ordinals[first_block:first_block + 3]
        if len(group) != 3:
            continue
        pre_rmsnorm = group[0] - 1
        post_linear2 = group[-1] + 6
        boundary_positions = {
            pre_rmsnorm - 2: ("affine_silu", f"outer_{outer:02d}.pre_norm_silu"),
            pre_rmsnorm - 1: ("library_gemm", f"outer_{outer:02d}.pre_projection_c768_to_c384"),
            post_linear2 + 1: ("affine_silu", f"outer_{outer:02d}.post_norm_silu"),
            post_linear2 + 2: ("library_gemm", f"outer_{outer:02d}.post_projection_c384_to_c768_residual"),
        }
        for ordinal, (expected_family, name) in boundary_positions.items():
            if ordinal_families.get(ordinal) == expected_family:
                positions[ordinal] = name

    tail = {
        317: "trunk.tip_norm_silu",
        318: "policy.p1_conv",
        319: "policy.g1_conv",
        320: "policy.g1_norm_silu",
        321: "policy.g1_half_to_float",
        322: "policy.g1_global_pool",
        323: "policy.gpool_to_bias_matmul",
        324: "policy.p1_half_to_float",
        325: "policy.gpool_bias_add",
        326: "policy.p1_norm_silu",
        327: "policy.p2_conv",
        328: "policy.gpool_to_pass_matmul",
        329: "policy.pass_bias_silu",
        330: "policy.gpool_to_pass_matmul2",
        331: "value.v1_conv",
        332: "value.v1_norm_silu",
        333: "value.v1_half_to_float",
        334: "value.v1_global_pool",
        335: "value.v2_matmul",
        336: "value.v2_bias_silu",
        337: "value.v3_matmul",
        338: "value.v3_bias",
        339: "value.score_matmul",
        340: "value.score_bias",
        341: "value.ownership_conv",
        342: "value.ownership_conv_splitk_reduce",
        343: "value.ownership_half_to_float",
    }
    for ordinal, name in tail.items():
        if ordinal in positions:
            positions[ordinal] = name
    return positions


def summarize_ordinals(kernels_by_stream, baseline):
    rows = collections.defaultdict(lambda: {
        "durations": [], "overlap_ns": 0, "excess_ns": 0,
        "peer_rows": collections.defaultdict(lambda: {
            "instances": 0, "overlap_ns": 0, "slowdowns": [], "excess_ns": 0,
        }),
    })
    ordinal_families = {}

    for kernels in kernels_by_stream.values():
        for kernel in kernels:
            key = (kernel["ordinal"], kernel["signature"])
            base = baseline.get(key)
            ordinal_families.setdefault(kernel["ordinal"], kernel["family"])
            if ordinal_families[kernel["ordinal"]] != kernel["family"]:
                raise RuntimeError(
                    f'ordinal {kernel["ordinal"]} changes family within a fixed forward'
                )
            row = rows[key]
            row["family"] = kernel["family"]
            row["short_name"] = kernel["short_name"]
            row["durations"].append(kernel["duration"])
            row["overlap_ns"] += kernel["overlap_ns"]
            if base is not None:
                row["excess_ns"] += max(0, kernel["duration"] - base)

            peer = row["peer_rows"][kernel["dominant_peer"]]
            peer["instances"] += 1
            peer["overlap_ns"] += kernel["overlap_ns"]
            if base is not None:
                peer["slowdowns"].append(kernel["duration"] / base)
                peer["excess_ns"] += max(0, kernel["duration"] - base)

    positions = logical_positions(ordinal_families)
    output = []
    for (ordinal, signature), row in rows.items():
        base = baseline.get((ordinal, signature))
        peer_breakdown = []
        for peer_family, peer in row["peer_rows"].items():
            peer_breakdown.append({
                "family": peer_family,
                "instances": peer["instances"],
                "overlap_ms": peer["overlap_ns"] / 1e6,
                "median_slowdown": (
                    statistics.median(peer["slowdowns"]) if peer["slowdowns"] else None
                ),
                "summed_excess_ms": peer["excess_ns"] / 1e6,
            })
        peer_breakdown.sort(key=lambda peer: (-peer["instances"], peer["family"]))
        common_peer = peer_breakdown[0]
        eligible_worst = [
            peer for peer in peer_breakdown
            if peer["instances"] >= 4 and peer["median_slowdown"] is not None
        ]
        worst_peer = max(
            eligible_worst,
            key=lambda peer: (peer["median_slowdown"], peer["instances"]),
            default=None,
        )
        largest_excess_peer = max(
            peer_breakdown,
            key=lambda peer: (peer["summed_excess_ms"], peer["instances"]),
        )
        duration_ns = sum(row["durations"])
        output.append({
            "ordinal": ordinal,
            "logical_position": positions[ordinal],
            "family": row["family"],
            "short_name": row["short_name"],
            "signature": signature,
            "instances": len(row["durations"]),
            "isolated_median_us": base / 1e3 if base is not None else None,
            "s2_median_us": statistics.median(row["durations"]) / 1e3,
            "median_slowdown": (
                statistics.median(row["durations"]) / base if base is not None else None
            ),
            "total_ms": duration_ns / 1e6,
            "overlap_fraction": row["overlap_ns"] / duration_ns,
            "summed_excess_ms": row["excess_ns"] / 1e6,
            "common_peer_family": common_peer["family"],
            "common_peer_instances": common_peer["instances"],
            "worst_peer_family": worst_peer["family"] if worst_peer else None,
            "worst_peer_instances": worst_peer["instances"] if worst_peer else 0,
            "worst_peer_median_slowdown": (
                worst_peer["median_slowdown"] if worst_peer else None
            ),
            "largest_excess_peer_family": largest_excess_peer["family"],
            "largest_excess_peer_ms": largest_excess_peer["summed_excess_ms"],
            "peer_breakdown": peer_breakdown,
        })
    output.sort(key=lambda row: (row["ordinal"], row["signature"]))
    return output


def logical_group(position):
    transformer_operations = (
        "attention_rmsnorm",
        "attention_qkv_projection",
        "attention_qk_rope",
        "attention_fa4",
        "attention_out_projection_residual",
        "ffn_rmsnorm",
        "ffn_linear1_gate_swiglu",
        "ffn_linear2_residual",
    )
    for operation in transformer_operations:
        if position.endswith("." + operation):
            return "transformer." + operation
    if position.startswith("outer_"):
        return "outer." + position.split(".", 1)[1]
    return position


def summarize_logical_groups(ordinals):
    groups = collections.defaultdict(lambda: {
        "ordinals": [], "families": set(), "instances": 0,
        "total_ms": 0.0, "isolated_reference_total_ms": 0.0,
        "summed_excess_ms": 0.0,
    })
    for row in ordinals:
        name = logical_group(row["logical_position"])
        group = groups[name]
        group["ordinals"].append(row["ordinal"])
        group["families"].add(row["family"])
        group["instances"] += row["instances"]
        group["total_ms"] += row["total_ms"]
        if row["isolated_median_us"] is not None:
            group["isolated_reference_total_ms"] += (
                row["isolated_median_us"] * row["instances"] / 1e3
            )
        group["summed_excess_ms"] += row["summed_excess_ms"]

    output = []
    for name, group in groups.items():
        isolated_total = group["isolated_reference_total_ms"]
        output.append({
            "logical_group": name,
            "families": sorted(group["families"]),
            "ordinals": sorted(group["ordinals"]),
            "positions": len(group["ordinals"]),
            "instances": group["instances"],
            "total_ms": group["total_ms"],
            "isolated_reference_total_ms": isolated_total,
            "total_slowdown": (
                group["total_ms"] / isolated_total if isolated_total else None
            ),
            "summed_excess_ms": group["summed_excess_ms"],
        })
    output.sort(key=lambda row: (row["summed_excess_ms"], row["total_ms"]), reverse=True)
    return output


def summarize(kernels_by_stream, baseline):
    family_rows = collections.defaultdict(lambda: {
        "instances": 0, "duration_ns": 0, "overlap_ns": 0,
        "durations": [], "slowdowns": [], "excess_ns": 0,
    })
    pair_rows = collections.defaultdict(lambda: {
        "instances": 0, "overlap_ns": 0, "overlap_fractions": [],
        "durations": [], "slowdowns": [],
    })

    for kernels in kernels_by_stream.values():
        for kernel in kernels:
            base = baseline.get((kernel["ordinal"], kernel["signature"]))
            slowdown = kernel["duration"] / base if base else None
            row = family_rows[kernel["family"]]
            row["instances"] += 1
            row["duration_ns"] += kernel["duration"]
            row["overlap_ns"] += kernel["overlap_ns"]
            row["durations"].append(kernel["duration"])
            if slowdown is not None:
                row["slowdowns"].append(slowdown)
                row["excess_ns"] += max(0, kernel["duration"] - base)

            pair_key = (kernel["family"], kernel["dominant_peer"])
            pair = pair_rows[pair_key]
            pair["instances"] += 1
            pair["overlap_ns"] += kernel["overlap_ns"]
            pair["overlap_fractions"].append(kernel["overlap_ns"] / kernel["duration"])
            pair["durations"].append(kernel["duration"])
            if slowdown is not None:
                pair["slowdowns"].append(slowdown)

    families = []
    for name, row in family_rows.items():
        families.append({
            "family": name,
            "instances": row["instances"],
            "total_ms": row["duration_ns"] / 1e6,
            "median_us": statistics.median(row["durations"]) / 1e3,
            "p90_us": percentile(row["durations"], 0.9) / 1e3,
            "overlap_fraction": row["overlap_ns"] / row["duration_ns"],
            "median_slowdown": statistics.median(row["slowdowns"]) if row["slowdowns"] else None,
            "matched_instances": len(row["slowdowns"]),
            "summed_excess_ms": row["excess_ns"] / 1e6,
        })
    families.sort(key=lambda row: row["total_ms"], reverse=True)

    pairs = []
    for (name, peer), row in pair_rows.items():
        if row["instances"] < 4:
            continue
        pairs.append({
            "family": name,
            "dominant_peer": peer,
            "instances": row["instances"],
            "median_overlap_fraction": statistics.median(row["overlap_fractions"]),
            "median_us": statistics.median(row["durations"]) / 1e3,
            "median_slowdown": statistics.median(row["slowdowns"]) if row["slowdowns"] else None,
            "matched_instances": len(row["slowdowns"]),
        })
    pairs.sort(
        key=lambda row: (
            row["median_slowdown"] if row["median_slowdown"] is not None else 0.0,
            row["instances"],
        ),
        reverse=True,
    )
    return families, pairs


def phase_summary(kernels_by_stream, iterations):
    streams = list(kernels_by_stream)
    if len(streams) != 2:
        return None
    starts = {}
    for stream, kernels in kernels_by_stream.items():
        starts[stream] = {}
        for iteration in range(iterations):
            starts[stream][iteration] = min(
                kernel["start"] for kernel in kernels if kernel["iteration"] == iteration
            )
    offsets = [
        (starts[streams[1]][iteration] - starts[streams[0]][iteration]) / 1e3
        for iteration in range(iterations)
    ]
    return {
        "stream_order": streams,
        "median_offset_us": statistics.median(offsets),
        "min_offset_us": min(offsets),
        "max_offset_us": max(offsets),
        "p10_offset_us": percentile(offsets, 0.1),
        "p90_offset_us": percentile(offsets, 0.9),
        "offsets_us": offsets,
    }


def markdown_report(report):
    lines = []
    lines.append("# Nsys stream interference report")
    lines.append("")
    lines.append(
        f'- Timed iterations: {report["iterations"]}; streams: '
        + ", ".join(str(stream) for stream in report["streams"])
    )
    lines.append(
        "- Kernels per forward: "
        + ", ".join(
            f'{stream}={count}' for stream, count in report["kernels_per_forward"].items()
        )
    )
    if report["phase"] is not None:
        phase = report["phase"]
        lines.append(
            f'- Iteration start offset stream {phase["stream_order"][1]} - '
            f'{phase["stream_order"][0]}: median {phase["median_offset_us"]:.2f} us, '
            f'p10..p90 {phase["p10_offset_us"]:.2f}..{phase["p90_offset_us"]:.2f} us, '
            f'range {phase["min_offset_us"]:.2f}..{phase["max_offset_us"]:.2f} us.'
        )
    lines.append("")
    lines.append("## Kernel families")
    lines.append("")
    lines.append("| family | instances | total ms | median us | p90 us | peer overlap | S2/S1 median | summed excess ms | matched |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["families"]:
        slowdown = "n/a" if row["median_slowdown"] is None else f'{row["median_slowdown"]:.3f}x'
        lines.append(
            f'| {row["family"]} | {row["instances"]} | {row["total_ms"]:.3f} | '
            f'{row["median_us"]:.3f} | {row["p90_us"]:.3f} | '
            f'{100.0 * row["overlap_fraction"]:.1f}% | {slowdown} | '
            f'{row["summed_excess_ms"]:.3f} | {row["matched_instances"]} |'
        )
    lines.append("")
    lines.append("## Dominant interference pairs")
    lines.append("")
    lines.append("| running family | dominant peer | instances | median overlap | median us | S2/S1 median | matched |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in report["pairs"][:40]:
        slowdown = "n/a" if row["median_slowdown"] is None else f'{row["median_slowdown"]:.3f}x'
        lines.append(
            f'| {row["family"]} | {row["dominant_peer"]} | {row["instances"]} | '
            f'{100.0 * row["median_overlap_fraction"]:.1f}% | {row["median_us"]:.3f} | '
            f'{slowdown} | {row["matched_instances"]} |'
        )
    lines.append("")
    lines.append("## Logical operation groups")
    lines.append("")
    lines.append(
        "Isolated reference total is the isolated median for each ordinal multiplied "
        "by its S2 call count; it is a normalized reference, not a second trace total."
    )
    lines.append("")
    lines.append("| logical group | families | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in report["logical_groups"]:
        slowdown = "n/a" if row["total_slowdown"] is None else f'{row["total_slowdown"]:.3f}x'
        lines.append(
            f'| {row["logical_group"]} | {", ".join(row["families"])} | '
            f'{row["positions"]} | {row["instances"]} | '
            f'{row["isolated_reference_total_ms"]:.3f} | {row["total_ms"]:.3f} | '
            f'{slowdown} | {row["summed_excess_ms"]:.3f} |'
        )
    lines.append("")
    lines.append("## `library_gemm` logical breakdown")
    lines.append("")
    lines.append("| logical group | positions | calls | isolated reference ms | S2 total ms | total ratio | excess ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["logical_groups"]:
        if row["families"] != ["library_gemm"]:
            continue
        slowdown = "n/a" if row["total_slowdown"] is None else f'{row["total_slowdown"]:.3f}x'
        lines.append(
            f'| {row["logical_group"]} | {row["positions"]} | '
            f'{row["instances"]} | {row["isolated_reference_total_ms"]:.3f} | '
            f'{row["total_ms"]:.3f} | {slowdown} | '
            f'{row["summed_excess_ms"]:.3f} |'
        )
    lines.append("")
    lines.append("## Top ordinal hotspots by summed excess")
    lines.append("")
    lines.append(
        "The worst peer is the highest median S2/S1 slowdown among peer families "
        "observed at least four times for that ordinal."
    )
    lines.append("")
    lines.append("| rank | ordinal | logical position | family | calls | isolated us | S2 us | S2/S1 | excess ms | common peer | worst peer |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|---|---|")
    hotspots = sorted(
        report["ordinals"],
        key=lambda row: (row["summed_excess_ms"], row["total_ms"]),
        reverse=True,
    )
    for rank, row in enumerate(hotspots[:40], start=1):
        isolated = (
            "n/a" if row["isolated_median_us"] is None
            else f'{row["isolated_median_us"]:.3f}'
        )
        slowdown = (
            "n/a" if row["median_slowdown"] is None
            else f'{row["median_slowdown"]:.3f}x'
        )
        worst = (
            "n/a" if row["worst_peer_family"] is None
            else f'{row["worst_peer_family"]} '
                 f'({row["worst_peer_median_slowdown"]:.3f}x; '
                 f'{row["worst_peer_instances"]})'
        )
        lines.append(
            f'| {rank} | {row["ordinal"]} | {row["logical_position"]} | '
            f'{row["family"]} | {row["instances"]} | {isolated} | '
            f'{row["s2_median_us"]:.3f} | {slowdown} | '
            f'{row["summed_excess_ms"]:.3f} | '
            f'{row["common_peer_family"]} ({row["common_peer_instances"]}) | '
            f'{worst} |'
        )
    lines.append("")
    lines.append("## Full fixed-forward ordinal map")
    lines.append("")
    lines.append("| ordinal | logical position | family | resource signature | calls | isolated us | S2 us | S2/S1 | overlap | excess ms | common peer | worst peer |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in report["ordinals"]:
        isolated = (
            "n/a" if row["isolated_median_us"] is None
            else f'{row["isolated_median_us"]:.3f}'
        )
        slowdown = (
            "n/a" if row["median_slowdown"] is None
            else f'{row["median_slowdown"]:.3f}x'
        )
        worst = (
            "n/a" if row["worst_peer_family"] is None
            else f'{row["worst_peer_family"]} '
                 f'({row["worst_peer_median_slowdown"]:.3f}x; '
                 f'{row["worst_peer_instances"]})'
        )
        signature = row["signature"].replace("|", "; ")
        lines.append(
            f'| {row["ordinal"]} | {row["logical_position"]} | {row["family"]} | '
            f'{signature} | {row["instances"]} | {isolated} | '
            f'{row["s2_median_us"]:.3f} | {slowdown} | '
            f'{100.0 * row["overlap_fraction"]:.1f}% | '
            f'{row["summed_excess_ms"]:.3f} | '
            f'{row["common_peer_family"]} ({row["common_peer_instances"]}) | '
            f'{worst} |'
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--isolated-sqlite")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--stream-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    args = parser.parse_args()

    baseline = isolated_medians(args.isolated_sqlite, args.iterations)
    conn = sqlite3.connect(args.sqlite)
    try:
        streams = choose_streams(conn, args.stream_count)
        kernels_by_stream = {}
        kernels_per_forward = {}
        for stream in streams:
            kernels, count = load_timed_kernels(conn, stream, args.iterations)
            kernels_by_stream[stream] = kernels
            kernels_per_forward[stream] = count
    finally:
        conn.close()

    if len(streams) == 2:
        find_overlaps(kernels_by_stream[streams[0]], kernels_by_stream[streams[1]])
        find_overlaps(kernels_by_stream[streams[1]], kernels_by_stream[streams[0]])
    else:
        for kernel in kernels_by_stream[streams[0]]:
            kernel["overlaps"] = []
            kernel["overlap_ns"] = 0
            kernel["dominant_peer"] = "idle"

    families, pairs = summarize(kernels_by_stream, baseline)
    ordinals = summarize_ordinals(kernels_by_stream, baseline)
    logical_groups = summarize_logical_groups(ordinals)
    report = {
        "sqlite": args.sqlite,
        "isolated_sqlite": args.isolated_sqlite,
        "iterations": args.iterations,
        "streams": streams,
        "kernels_per_forward": kernels_per_forward,
        "phase": phase_summary(kernels_by_stream, args.iterations),
        "families": families,
        "pairs": pairs,
        "ordinals": ordinals,
        "logical_groups": logical_groups,
    }

    markdown = markdown_report(report)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    if not args.json_output and not args.markdown_output:
        print(markdown, end="")


if __name__ == "__main__":
    main()
