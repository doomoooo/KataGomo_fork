#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def split_csv(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for item in values:
        for piece in item.split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


def parse_graph_modes(spec: str) -> List[bool]:
    modes: List[bool] = []
    mapping = {"off": False, "on": True, "false": False, "true": True, "0": False, "1": True}
    for token in split_csv([spec]):
        key = token.lower()
        if key not in mapping:
            raise ValueError(f"Invalid graph mode token: {token}")
        modes.append(mapping[key])
    if not modes:
        raise ValueError("graph modes cannot be empty")
    dedup: List[bool] = []
    for mode in modes:
        if mode not in dedup:
            dedup.append(mode)
    return dedup


def extract_metric(result: Dict[str, object], metric_name: str) -> Optional[float]:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    if metric_name == "throughput_qps":
        val = metrics.get("throughput_qps")
        return float(val) if isinstance(val, (int, float)) else None

    if metric_name in {"throughput_qps_x_batch", "nn_evals_per_sec"}:
        qps = metrics.get("throughput_qps")
        ib = result.get("infer_batch")
        if not isinstance(qps, (int, float)) or not isinstance(ib, int):
            return None
        return float(qps) * float(ib)

    if metric_name == "latency_mean_ms":
        part = metrics.get("latency", {})
        if not isinstance(part, dict):
            return None
        mean = part.get("mean", {})
        if not isinstance(mean, dict):
            return None
        val = mean.get("value")
        return float(val) if isinstance(val, (int, float)) else None

    if metric_name == "gpu_compute_mean_ms":
        part = metrics.get("gpu_compute", {})
        if not isinstance(part, dict):
            return None
        mean = part.get("mean", {})
        if not isinstance(mean, dict):
            return None
        val = mean.get("value")
        return float(val) if isinstance(val, (int, float)) else None

    raise ValueError(f"Unsupported metric: {metric_name}")


def metric_label(metric_name: str) -> str:
    if metric_name in {"throughput_qps_x_batch", "nn_evals_per_sec"}:
        return "nnEval/s"
    if metric_name == "throughput_qps":
        return "trtexec qps"
    if metric_name == "latency_mean_ms":
        return "Latency Mean (ms)"
    if metric_name == "gpu_compute_mean_ms":
        return "GPU Compute Mean (ms)"
    return metric_name


def build_case_table(
    data: Dict[str, object],
    metric_name: str,
    require_ok: bool,
) -> Dict[Tuple[int, int, int, bool], float]:
    table: Dict[Tuple[int, int, int, bool], float] = {}
    results = data.get("results", {})
    if not isinstance(results, dict):
        return table

    for _, result in results.items():
        if not isinstance(result, dict):
            continue
        if require_ok and result.get("status") != "ok":
            continue

        pb = result.get("plan_batch")
        ib = result.get("infer_batch")
        streams = result.get("streams")
        graph = result.get("cuda_graph")
        if not isinstance(pb, int) or not isinstance(ib, int) or not isinstance(streams, int) or not isinstance(graph, bool):
            continue

        metric_value = extract_metric(result, metric_name)
        if metric_value is None:
            continue

        table[(pb, ib, streams, graph)] = metric_value
    return table


def choose_plan_batch(
    table: Dict[Tuple[int, int, int, bool], float],
    requested_plan_batch: Optional[int],
    meta: object,
) -> int:
    if requested_plan_batch is not None:
        if requested_plan_batch <= 0:
            raise ValueError("plan-batch must be positive")
        return requested_plan_batch

    if isinstance(meta, dict):
        plan_batch = meta.get("plan_batch")
        if isinstance(plan_batch, int) and plan_batch > 0:
            return plan_batch

    pbs = sorted({pb for (pb, _ib, _s, _g), _v in table.items()})
    if not pbs:
        raise RuntimeError("No plan_batch value found in benchmark results")
    return pbs[-1]


def plot_fixed_plan(
    table: Dict[Tuple[int, int, int, bool], float],
    output_png: Path,
    metric_display_name: str,
    plan_batch: int,
    ib_values: List[int],
    stream_values: List[int],
    graph_modes: List[bool],
    title_suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.get_cmap("tab10")
    style_map = {False: (0, (6, 3)), True: "-"}

    plotted_any = False
    legend_handles: List[Line2D] = []

    for stream in stream_values:
        color = colors((stream - 1) % 10)
        for graph in graph_modes:
            xs: List[int] = []
            ys: List[float] = []
            for ib in ib_values:
                key = (plan_batch, ib, stream, graph)
                if key not in table:
                    continue
                xs.append(ib)
                ys.append(table[key])
            if not xs:
                continue

            plotted_any = True
            label = f"s={stream}, g={'on' if graph else 'off'}"
            ax.plot(
                xs,
                ys,
                linestyle=style_map[graph],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=color,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=style_map[graph],
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    label=label,
                )
            )

    ax.set_title(f"pb={plan_batch}, {title_suffix} ({metric_display_name})")
    ax.set_xlabel("infer batch (ib)")
    ax.set_ylabel(metric_display_name)
    ax.grid(True, alpha=0.35)
    if ib_values:
        ax.set_xlim(left=min(ib_values), right=max(ib_values))

    if plotted_any:
        ax.legend(handles=legend_handles, loc="best", ncol=2, handlelength=3.2)
    else:
        ax.text(0.5, 0.5, "No matching data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize fixed-plan benchmark.py JSON with matplotlib.")
    parser.add_argument("--input-json", default="build/trtexec_benchmark.json")
    parser.add_argument("--output-dir", default="build/benchmark_plots")
    parser.add_argument(
        "--metric",
        default="nn_evals_per_sec",
        choices=["nn_evals_per_sec", "throughput_qps_x_batch", "throughput_qps", "latency_mean_ms", "gpu_compute_mean_ms"],
    )
    parser.add_argument("--plan-batch", type=int, default=None)
    parser.add_argument("--infer-batch-min", type=int, default=1)
    parser.add_argument("--infer-batch-max", type=int, default=None)
    parser.add_argument("--stream-min", type=int, default=1)
    parser.add_argument("--stream-max", type=int, default=4)
    parser.add_argument("--graph-modes", default="off,on", help="Comma separated: off,on")
    parser.add_argument("--include-non-ok", action="store_true")
    args = parser.parse_args()

    if args.infer_batch_min <= 0:
        raise ValueError("infer-batch-min must be positive")
    if args.infer_batch_max is not None and args.infer_batch_max <= 0:
        raise ValueError("infer-batch-max must be positive")
    if args.infer_batch_max is not None and args.infer_batch_max < args.infer_batch_min:
        raise ValueError("infer-batch-max must be >= infer-batch-min")
    if args.stream_min <= 0 or args.stream_max < args.stream_min:
        raise ValueError("Invalid stream range")

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input json not found: {input_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    table = build_case_table(
        data,
        metric_name=args.metric,
        require_ok=(not args.include_non_ok),
    )
    if not table:
        raise RuntimeError("No usable benchmark results found in json")

    plan_batch = choose_plan_batch(table, args.plan_batch, data.get("meta"))
    infer_max = plan_batch
    if args.infer_batch_max is not None:
        infer_max = min(infer_max, args.infer_batch_max)
    if args.infer_batch_min > infer_max:
        raise ValueError("No infer batch to plot after applying range and plan-batch")

    stream_values = list(range(args.stream_min, args.stream_max + 1))
    graph_modes = parse_graph_modes(args.graph_modes)
    metric_name = metric_label(args.metric)

    ib_values_all = list(range(args.infer_batch_min, infer_max + 1))
    ib_values_lt = [ib for ib in ib_values_all if ib < plan_batch]

    output_all = output_dir / f"pb{plan_batch}_ib_le_pb_lines.png"
    plot_fixed_plan(
        table=table,
        output_png=output_all,
        metric_display_name=metric_name,
        plan_batch=plan_batch,
        ib_values=ib_values_all,
        stream_values=stream_values,
        graph_modes=graph_modes,
        title_suffix="ib<=pb",
    )
    print(f"[ok] {output_all} ({metric_name})")

    if ib_values_lt:
        output_lt = output_dir / f"pb{plan_batch}_ib_lt_pb_lines.png"
        plot_fixed_plan(
            table=table,
            output_png=output_lt,
            metric_display_name=metric_name,
            plan_batch=plan_batch,
            ib_values=ib_values_lt,
            stream_values=stream_values,
            graph_modes=graph_modes,
            title_suffix="ib<pb",
        )
        print(f"[ok] {output_lt} ({metric_name})")
    else:
        print("[warn] Skip ib<pb plot: infer-batch range has no value < plan-batch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

