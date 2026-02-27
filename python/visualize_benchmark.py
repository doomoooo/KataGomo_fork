#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def plot_pb_eq_ib(
    table: Dict[Tuple[int, int, int, bool], float],
    output_png: Path,
    metric_name: str,
    metric_display_name: str,
    min_pb: int,
    max_pb: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.get_cmap("tab10")
    style_map = {False: (0, (6, 3)), True: "-"}

    plotted_any = False
    present_series: Dict[Tuple[int, bool], bool] = {}
    pbs = [pb for pb in range(min_pb, max_pb + 1)]
    for streams in range(1, 5):
        for graph in (False, True):
            xs: List[int] = []
            ys: List[float] = []
            for pb in pbs:
                key = (pb, pb, streams, graph)
                if key in table:
                    xs.append(pb)
                    ys.append(table[key])
            if not xs:
                continue
            plotted_any = True
            present_series[(streams, graph)] = True
            label = f"s={streams}, g={'on' if graph else 'off'}"
            ax.plot(
                xs,
                ys,
                linestyle=style_map[graph],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=colors(streams - 1),
                label=label,
            )

    ax.set_title(f"pb=ib Benchmark ({metric_display_name})")
    ax.set_xlabel("Batch (pb = ib)")
    ax.set_ylabel(metric_display_name)
    ax.grid(True, alpha=0.35)
    if plotted_any:
        legend_handles: List[Line2D] = []
        for streams in range(1, 5):
            for graph in (False, True):
                if (streams, graph) not in present_series:
                    continue
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=colors(streams - 1),
                        linestyle=style_map[graph],
                        marker="o",
                        linewidth=2.0,
                        markersize=4.5,
                        label=f"s={streams}, g={'on' if graph else 'off'}",
                    )
                )
        ax.legend(handles=legend_handles, loc="best", ncol=2, handlelength=3.2)
    else:
        ax.text(0.5, 0.5, "No matching pb=ib data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def plot_pb_eq_ib_gif(
    table: Dict[Tuple[int, int, int, bool], float],
    frames_dir: Path,
    gif_path: Path,
    metric_display_name: str,
    min_pb: int,
    max_pb: int,
    fps: int,
) -> Tuple[int, List[Path]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10")

    pbs_all = sorted(
        {
            pb
            for (pb, ib, _s, _g), _v in table.items()
            if pb == ib and min_pb <= pb <= max_pb
        }
    )
    if not pbs_all:
        return 0, []

    global_x_left = min_pb
    global_x_right = max(max_pb, max(pbs_all))
    all_y = [
        v
        for (pb, ib, _s, _g), v in table.items()
        if pb == ib and min_pb <= pb <= max_pb
    ]
    if all_y:
        y_min = min(all_y)
        y_max = max(all_y)
        pad = max((y_max - y_min) * 0.05, 1.0) if y_min == y_max else (y_max - y_min) * 0.05
        global_y_min = y_min - pad
        global_y_max = y_max + pad
    else:
        global_y_min = None
        global_y_max = None

    frame_paths: List[Path] = []
    for pb_limit in pbs_all:
        fig, ax = plt.subplots(figsize=(12, 7))
        plotted_any = False
        for streams in range(1, 5):
            for graph in (False, True):
                xs: List[int] = []
                ys: List[float] = []
                for pb in pbs_all:
                    if pb > pb_limit:
                        break
                    key = (pb, pb, streams, graph)
                    if key in table:
                        xs.append(pb)
                        ys.append(table[key])
                if not xs:
                    continue
                plotted_any = True
                style = "-" if graph else "--"
                label = f"s={streams}, g={'on' if graph else 'off'}"
                ax.plot(
                    xs,
                    ys,
                    linestyle=style,
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    color=colors(streams - 1),
                    label=label,
                )

        ax.set_title(f"pb=ib Benchmark ({metric_display_name}) up to pb={pb_limit}")
        ax.set_xlabel("Batch (pb = ib)")
        ax.set_ylabel(metric_display_name)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(left=global_x_left, right=global_x_right)
        if global_y_min is not None and global_y_max is not None:
            ax.set_ylim(bottom=global_y_min, top=global_y_max)
        if plotted_any:
            ax.legend(loc="best", ncol=2)
        else:
            ax.text(0.5, 0.5, "No matching pb=ib data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()

        frame_path = frames_dir / f"pb_{pb_limit:02d}.png"
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        frame_paths.append(frame_path)

    try:
        from PIL import Image
    except Exception as e:
        print(f"[warn] PIL not available, skip pb=ib GIF generation: {e}")
        return len(frame_paths), frame_paths

    images = [Image.open(p) for p in frame_paths]
    duration_ms = max(1, int(1000 / max(1, fps)))
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for img in images:
        img.close()
    return len(frame_paths), frame_paths


def choose_pb_le_ib_value_for_stream(
    table: Dict[Tuple[int, int, int, bool], float],
    pb: int,
    ib: int,
    stream: int,
) -> Optional[float]:
    preferred = (pb, ib, stream, True)
    if preferred in table:
        return table[preferred]

    secondary = (pb, ib, stream, False)
    if secondary in table:
        return table[secondary]
    return None


def plot_ib_le_pb_stream_gif(
    table: Dict[Tuple[int, int, int, bool], float],
    frames_dir: Path,
    gif_path: Path,
    metric_display_name: str,
    min_pb: int,
    max_pb: int,
    fps: int,
    stream: int,
) -> Tuple[int, List[Path]]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    pb_values = sorted({pb for (pb, ib, s, _g), _val in table.items() if s == stream and ib <= pb and min_pb <= pb <= max_pb})

    series_by_pb: Dict[int, Tuple[List[int], List[float]]] = {}
    for pb in pb_values:
        xs: List[int] = []
        ys: List[float] = []
        for ib in range(1, pb + 1):
            val = choose_pb_le_ib_value_for_stream(table, pb, ib, stream)
            if val is None:
                continue
            xs.append(ib)
            ys.append(val)
        series_by_pb[pb] = (xs, ys)

    global_x_left = 1
    global_x_right = max([2] + [pb for pb in pb_values])
    all_y = [y for _, ys in series_by_pb.values() for y in ys]
    global_y_min: Optional[float] = None
    global_y_max: Optional[float] = None
    if all_y:
        y_min = min(all_y)
        y_max = max(all_y)
        if y_min == y_max:
            pad = max(abs(y_min) * 0.05, 1.0)
        else:
            pad = (y_max - y_min) * 0.05
        global_y_min = y_min - pad
        global_y_max = y_max + pad

    frame_paths: List[Path] = []
    for pb in pb_values:
        xs, ys = series_by_pb[pb]

        fig, ax = plt.subplots(figsize=(10, 6))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"pb={pb}, ib<=pb, s={stream} ({metric_display_name})")
        ax.set_xlabel("infer batch (ib)")
        ax.set_ylabel(metric_display_name)
        ax.grid(True, alpha=0.35)
        ax.set_xlim(left=global_x_left, right=global_x_right)
        if global_y_min is not None and global_y_max is not None:
            ax.set_ylim(bottom=global_y_min, top=global_y_max)
        fig.tight_layout()

        frame_path = frames_dir / f"pb_{pb:02d}.png"
        fig.savefig(frame_path, dpi=160)
        plt.close(fig)
        frame_paths.append(frame_path)

    if frame_paths:
        try:
            from PIL import Image
        except Exception as e:
            print(f"[warn] PIL not available, skip GIF generation: {e}")
            return len(frame_paths), frame_paths

        images = [Image.open(p) for p in frame_paths]
        duration_ms = max(1, int(1000 / max(1, fps)))
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        for img in images:
            img.close()

    return len(frame_paths), frame_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize benchmark.py JSON with matplotlib.")
    parser.add_argument("--input-json", default="build/trtexec_benchmark.json")
    parser.add_argument("--output-dir", default="build/benchmark_plots")
    parser.add_argument(
        "--line-metric",
        default="nn_evals_per_sec",
        choices=["nn_evals_per_sec", "throughput_qps_x_batch", "throughput_qps", "latency_mean_ms", "gpu_compute_mean_ms"],
        help="Metric for pb=ib multi-line chart.",
    )
    parser.add_argument(
        "--gif-metric",
        default="nn_evals_per_sec",
        choices=["nn_evals_per_sec", "throughput_qps_x_batch", "throughput_qps", "latency_mean_ms", "gpu_compute_mean_ms"],
        help="Metric for pb<=ib GIF frames.",
    )
    parser.add_argument("--min-pb", type=int, default=1)
    parser.add_argument("--max-pb", type=int, default=16)
    parser.add_argument("--gif-fps", type=int, default=2)
    parser.add_argument("--include-non-ok", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input json not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    line_metric_name = metric_label(args.line_metric)
    gif_metric_name = metric_label(args.gif_metric)

    line_table = build_case_table(data, metric_name=args.line_metric, require_ok=(not args.include_non_ok))
    gif_table = build_case_table(data, metric_name=args.gif_metric, require_ok=(not args.include_non_ok))

    if not line_table and not gif_table:
        raise RuntimeError("No usable benchmark results found in json")

    pb_eq_png = output_dir / "pb_eq_ib_lines.png"
    if line_table:
        plot_pb_eq_ib(
            table=line_table,
            output_png=pb_eq_png,
            metric_name=args.line_metric,
            metric_display_name=line_metric_name,
            min_pb=args.min_pb,
            max_pb=args.max_pb,
        )
        print(f"[ok] pb=ib line plot: {pb_eq_png} ({line_metric_name})")
    else:
        print("[warn] Skip pb=ib line plot: no data for line metric")

    if gif_table:
        for stream in [1, 2]:
            frames_dir = output_dir / f"frames_ib_le_pb_s{stream}"
            gif_path = output_dir / f"ib_le_pb_s{stream}.gif"
            nframes, _ = plot_ib_le_pb_stream_gif(
                table=gif_table,
                frames_dir=frames_dir,
                gif_path=gif_path,
                metric_display_name=gif_metric_name,
                min_pb=args.min_pb,
                max_pb=args.max_pb,
                fps=args.gif_fps,
                stream=stream,
            )
            if nframes > 0:
                print(f"[ok] s={stream} frames: {frames_dir} ({nframes} frames)")
                if gif_path.exists():
                    print(f"[ok] s={stream} gif: {gif_path} ({gif_metric_name})")
                else:
                    print(f"[warn] s={stream} GIF was not generated")
            else:
                print(f"[warn] s={stream} has no ib<=pb data, GIF skipped")
    else:
        print("[warn] Skip pb<=ib GIF: no data for gif metric")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
