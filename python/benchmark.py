#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


PLAN_PATH_RE = re.compile(
    r"(?:Saved new plan cache to|Using existing plan cache at)\s+(.+)$",
    re.MULTILINE,
)
THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9eE+\-.]+)\s*qps")
TOTAL_HOST_RE = re.compile(r"Total Host Walltime:\s*([0-9eE+\-.]+)\s*s")
TOTAL_GPU_RE = re.compile(r"Total GPU Compute Time:\s*([0-9eE+\-.]+)\s*s")
SUMMARY_RE = {
    "latency": re.compile(
        r"Latency:\s*min = ([^,]+), max = ([^,]+), mean = ([^,]+), median = ([^,]+), "
        r"percentile\(90%\) = ([^,]+), percentile\(95%\) = ([^,]+), percentile\(99%\) = ([^\n,]+)"
    ),
    "enqueue": re.compile(
        r"Enqueue Time:\s*min = ([^,]+), max = ([^,]+), mean = ([^,]+), median = ([^,]+), "
        r"percentile\(90%\) = ([^,]+), percentile\(95%\) = ([^,]+), percentile\(99%\) = ([^\n,]+)"
    ),
    "h2d": re.compile(
        r"H2D Latency:\s*min = ([^,]+), max = ([^,]+), mean = ([^,]+), median = ([^,]+), "
        r"percentile\(90%\) = ([^,]+), percentile\(95%\) = ([^,]+), percentile\(99%\) = ([^\n,]+)"
    ),
    "gpu_compute": re.compile(
        r"GPU Compute Time:\s*min = ([^,]+), max = ([^,]+), mean = ([^,]+), median = ([^,]+), "
        r"percentile\(90%\) = ([^,]+), percentile\(95%\) = ([^,]+), percentile\(99%\) = ([^\n,]+)"
    ),
    "d2h": re.compile(
        r"D2H Latency:\s*min = ([^,]+), max = ([^,]+), mean = ([^,]+), median = ([^,]+), "
        r"percentile\(90%\) = ([^,]+), percentile\(95%\) = ([^,]+), percentile\(99%\) = ([^\n,]+)"
    ),
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def join_cmd(cmd: List[str]) -> str:
    return shlex.join(cmd)


def detect_default_path(candidates: List[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return str(Path(candidate).resolve())
    return candidates[0]


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


def parse_positive_int_csv(spec: str, label: str) -> List[int]:
    values: List[int] = []
    for token in split_csv([spec]):
        try:
            num = int(token)
        except ValueError as e:
            raise ValueError(f"Invalid integer in {label}: {token}") from e
        if num <= 0:
            raise ValueError(f"{label} must contain positive integers, got: {num}")
        values.append(num)
    dedup: List[int] = []
    for value in values:
        if value not in dedup:
            dedup.append(value)
    if not dedup:
        raise ValueError(f"{label} cannot be empty")
    return dedup


def parse_value_with_unit(raw: str) -> Dict[str, object]:
    token = raw.strip()
    match = re.match(r"^([0-9eE+\-.]+)\s*([A-Za-z%/]+)?$", token)
    if not match:
        return {"text": token}
    value = float(match.group(1))
    unit = match.group(2) or ""
    return {"value": value, "unit": unit, "text": token}


def parse_trtexec_metrics(text: str) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    throughput = THROUGHPUT_RE.search(text)
    if throughput:
        metrics["throughput_qps"] = float(throughput.group(1))

    total_host = TOTAL_HOST_RE.search(text)
    if total_host:
        metrics["total_host_walltime_s"] = float(total_host.group(1))

    total_gpu = TOTAL_GPU_RE.search(text)
    if total_gpu:
        metrics["total_gpu_compute_s"] = float(total_gpu.group(1))

    for name, regex in SUMMARY_RE.items():
        match = regex.search(text)
        if not match:
            continue
        metrics[name] = {
            "min": parse_value_with_unit(match.group(1)),
            "max": parse_value_with_unit(match.group(2)),
            "mean": parse_value_with_unit(match.group(3)),
            "median": parse_value_with_unit(match.group(4)),
            "p90": parse_value_with_unit(match.group(5)),
            "p95": parse_value_with_unit(match.group(6)),
            "p99": parse_value_with_unit(match.group(7)),
        }
    return metrics


def output_tail(text: str, lines: int = 100) -> str:
    rows = text.splitlines()
    if len(rows) <= lines:
        return text
    return "\n".join(rows[-lines:])


def atomic_save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def load_or_init_state(path: Path, resume: bool, meta: Dict[str, object]) -> Dict[str, object]:
    if resume and path.exists():
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("meta", {})
        state.setdefault("plans", {})
        state.setdefault("results", {})
        state["meta"]["last_resume_at"] = now_iso()
        return state
    return {"meta": meta, "plans": {}, "results": {}}


def build_env(tensorrt_lib: str) -> Dict[str, str]:
    env = os.environ.copy()
    if tensorrt_lib:
        existing = env.get("LD_LIBRARY_PATH", "")
        if existing:
            env["LD_LIBRARY_PATH"] = f"{tensorrt_lib}:{existing}"
        else:
            env["LD_LIBRARY_PATH"] = tensorrt_lib
    return env


def run_command(
    cmd: List[str],
    env: Dict[str, str],
    timeout_sec: int,
    input_text: Optional[str] = None,
) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + ("\n" + e.stderr if e.stderr else "")
        raise RuntimeError(
            f"Command timeout after {timeout_sec}s\n{join_cmd(cmd)}\n--- output tail ---\n{output_tail(partial)}"
        ) from e
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, combined


def render_progress(done: int, total: int, status: str) -> None:
    width = 36
    ratio = 1.0 if total == 0 else done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    line = f"\r[{bar}] {done}/{total} {ratio * 100:6.2f}% {status[:100]}"
    sys.stdout.write(line)
    sys.stdout.flush()


def case_key(batch_size: int, streams: int) -> str:
    return f"b{batch_size}_s{streams}"


def normalize_key_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return token or "plan"


def case_key_with_plan(
    plan_label: str,
    batch_size: int,
    streams: int,
) -> str:
    return f"pl{normalize_key_token(plan_label)}_{case_key(batch_size, streams)}"


def extract_plot_value(metric_name: str, result: Dict[str, object]) -> Optional[float]:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    throughput = metrics.get("throughput_qps")
    if not isinstance(throughput, (int, float)):
        return None

    if metric_name == "throughput_qps":
        return float(throughput)
    if metric_name == "nn_evals_per_sec":
        batch_size = result.get("batch_size")
        if not isinstance(batch_size, int):
            return None
        return float(throughput) * float(batch_size)
    raise ValueError(f"Unsupported plot metric: {metric_name}")


def plot_metric_label(metric_name: str) -> str:
    if metric_name == "throughput_qps":
        return "trtexec qps"
    if metric_name == "nn_evals_per_sec":
        return "nnEval/s"
    return metric_name


def draw_plot(
    state: Dict[str, object],
    output_png: Path,
    metric_name: str,
    include_non_ok: bool,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] Skip plotting: matplotlib unavailable ({e})")
        return False

    results = state.get("results", {})
    if not isinstance(results, dict):
        return False

    stream_series: Dict[int, Dict[int, float]] = {}
    for result in results.values():
        if not isinstance(result, dict):
            continue
        if not include_non_ok and result.get("status") != "ok":
            continue
        stream = result.get("streams")
        batch_size = result.get("batch_size")
        if not isinstance(stream, int) or not isinstance(batch_size, int):
            continue
        value = extract_plot_value(metric_name, result)
        if value is None:
            continue
        stream_series.setdefault(stream, {})[batch_size] = value

    if not stream_series:
        print("[warn] Skip plotting: no usable benchmark points")
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    color_map = plt.get_cmap("tab10")
    any_point = False
    for idx, stream in enumerate(sorted(stream_series.keys())):
        points = stream_series[stream]
        xs = sorted(points.keys())
        ys = [points[x] for x in xs]
        if not xs:
            continue
        any_point = True
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=color_map(idx % 10),
            label=f"stream={stream}",
        )

    if not any_point:
        print("[warn] Skip plotting: no usable benchmark points")
        plt.close(fig)
        return False

    ax.set_xlabel("batch size")
    ax.set_ylabel(plot_metric_label(metric_name))
    ax.set_title(f"trtexec benchmark ({plot_metric_label(metric_name)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return True


def should_skip_result(existing: Optional[Dict[str, object]], rerun_failed: bool) -> bool:
    if not existing:
        return False
    status = existing.get("status")
    if status == "ok":
        return True
    if status == "error" and not rerun_failed:
        return True
    return False


def parse_plan_file_args(specs: Optional[List[str]]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    if not specs:
        return parsed

    for spec in specs:
        for token in split_csv([spec]):
            item = token.strip()
            if not item:
                continue

            if "=" in item:
                label, raw_path = item.split("=", 1)
                label = label.strip()
                raw_path = raw_path.strip()
                if not raw_path:
                    raise ValueError(f"Invalid --plan-file entry: {item}")
                if not label:
                    label = Path(raw_path).name
            else:
                raw_path = item
                label = Path(raw_path).name

            plan_path = Path(raw_path).expanduser()
            if not plan_path.exists():
                raise FileNotFoundError(f"plan path not found: {plan_path}")
            parsed.append((label, str(plan_path.resolve())))

    dedup_labels: Dict[str, int] = {}
    unique: List[Tuple[str, str]] = []
    for label, path in parsed:
        base = label
        suffix = dedup_labels.get(base, 0)
        if suffix == 0 and base not in dedup_labels:
            dedup_labels[base] = 1
            unique.append((base, path))
            continue

        while True:
            suffix += 1
            candidate = f"{base}_{suffix}"
            if candidate not in dedup_labels:
                dedup_labels[base] = suffix
                dedup_labels[candidate] = 1
                unique.append((candidate, path))
                break

    return unique


def ensure_plan(
    plan_batch: int,
    args: argparse.Namespace,
    state: Dict[str, object],
    output_path: Path,
    env: Dict[str, str],
) -> str:
    plans = state["plans"]
    existing_entry = plans.get(str(plan_batch))
    if isinstance(existing_entry, dict):
        cached_path = existing_entry.get("path", "")
        if cached_path and Path(cached_path).exists():
            return str(Path(cached_path).resolve())

    print(f"\n[plan] ensure batch={plan_batch}")
    override_parts = [
        f"nnMaxBatchSize={plan_batch}",
        "nnMinBatchSize=1",
        "trtMultiProfile=true",
        "numSearchThreads=1",
        "numNNServerThreadsPerModel=1",
        "ponderingEnabled=false",
        "logSearchInfo=false",
        "logAllGTPCommunication=false",
        "logToStderr=true",
    ]
    override_parts.extend(split_csv(args.gtp_extra_override))
    override_str = ",".join(override_parts)

    cmd = [
        args.katago_bin,
        "gtp",
        "-config",
        args.config,
        "-model",
        args.model,
        "-override-config",
        override_str,
    ]
    retcode, text = run_command(cmd, env=env, timeout_sec=args.plan_timeout_sec, input_text="quit\n")
    if retcode != 0:
        raise RuntimeError(
            f"katago gtp failed for batch={plan_batch} (exit={retcode})\n{join_cmd(cmd)}\n--- output tail ---\n{output_tail(text)}"
        )

    matches = PLAN_PATH_RE.findall(text)
    if not matches:
        raise RuntimeError(
            "Failed to parse plan path from katago output.\n"
            f"Command: {join_cmd(cmd)}\n--- output tail ---\n{output_tail(text)}"
        )
    plan_path = matches[-1].strip()
    plan_file = Path(plan_path).expanduser()
    if not plan_file.exists():
        raise RuntimeError(f"Parsed plan path does not exist: {plan_file}")

    plans[str(plan_batch)] = {
        "path": str(plan_file.resolve()),
        "updated_at": now_iso(),
        "source": "katago-gtp",
    }
    atomic_save_json(output_path, state)
    return str(plan_file.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate static-profile TensorRT plans via katago, benchmark with trtexec, then plot.",
    )
    parser.add_argument(
        "--katago-bin",
        default=detect_default_path(["build/katago", "/opt/katago/katago"]),
    )
    parser.add_argument(
        "--trtexec-bin",
        default="/opt/tensorrt/bin/trtexec",
    )
    parser.add_argument(
        "--config",
        default=detect_default_path(["/opt/katago/config/gtp_example.cfg", "cpp/tests/data/configs/analysis_example.cfg"]),
    )
    parser.add_argument(
        "--model",
        default=detect_default_path(["/opt/katago/weight/b18tf.onnx"]),
    )
    parser.add_argument("--output-json", default="build/trtexec_benchmark.json")
    parser.add_argument("--tensorrt-lib", default="/opt/tensorrt/lib")
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--plan-batch", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--plan-file",
        action="append",
        default=[],
        help="Use existing plan file(s), format: /path/to/plan or label=/path/to/plan. Can be passed multiple times.",
    )
    parser.add_argument("--batch-min", type=int, default=1)
    parser.add_argument("--batch-max", type=int, default=None)
    parser.add_argument("--stream-min", type=int, default=1)
    parser.add_argument("--stream-max", type=int, default=4)
    parser.add_argument(
        "--shape-template",
        action="append",
        default=None,
        help="Input shape template with {batch}, e.g. input_spatial:{batch}x22x19x19",
    )
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--avg-runs", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--plan-timeout-sec", type=int, default=1800)
    parser.add_argument("--trtexec-timeout-sec", type=int, default=600)
    parser.add_argument("--gtp-extra-override", action="append", default=[])
    parser.add_argument("--trtexec-extra-arg", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-png", default="build/trtexec_benchmark.png")
    parser.add_argument(
        "--plot-metric",
        default="nn_evals_per_sec",
        choices=["nn_evals_per_sec", "throughput_qps"],
    )
    parser.add_argument("--plot-include-non-ok", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke mode")
    args = parser.parse_args()

    if args.plan_batch is not None:
        args.max_batch = args.plan_batch

    if args.smoke:
        args.max_batch = 1
        args.batch_min = 1
        args.batch_max = 1
        args.stream_min = 1
        args.stream_max = 1
        args.duration_sec = min(args.duration_sec, 1.0)
        args.warmup_ms = 0
        args.iterations = min(args.iterations, 20)
        args.avg_runs = min(args.avg_runs, 10)

    if args.max_batch <= 0:
        raise ValueError("max-batch must be positive")
    if args.batch_min <= 0:
        raise ValueError("batch-min must be positive")
    if args.stream_min <= 0 or args.stream_max < args.stream_min:
        raise ValueError("Invalid stream range")
    if args.batch_max is not None and args.batch_max <= 0:
        raise ValueError("batch-max must be positive")
    if args.batch_max is not None and args.batch_max < args.batch_min:
        raise ValueError("batch-max must be >= batch-min")
    if args.batch_min > args.max_batch:
        raise ValueError("batch-min cannot exceed max-batch")

    parsed_plan_files = parse_plan_file_args(args.plan_file)
    use_existing_plan_files = len(parsed_plan_files) > 0

    katago_bin = Path(args.katago_bin)
    trtexec_bin = Path(args.trtexec_bin)
    config = Path(args.config)
    model = Path(args.model)
    if not trtexec_bin.exists():
        raise FileNotFoundError(f"trtexec path not found: {trtexec_bin}")
    if not use_existing_plan_files:
        for path_obj, label in [(katago_bin, "katago"), (config, "config"), (model, "model")]:
            if not path_obj.exists():
                raise FileNotFoundError(f"{label} path not found: {path_obj}")

    shape_templates = split_csv(args.shape_template)
    if not shape_templates:
        shape_templates = [
            "input_spatial:{batch}x22x19x19",
            "input_global:{batch}x19",
        ]

    stream_values = list(range(args.stream_min, args.stream_max + 1))
    batch_max = args.max_batch if args.batch_max is None else min(args.max_batch, args.batch_max)
    if args.batch_min > batch_max:
        raise ValueError("No batch to benchmark after applying batch range and max-batch")
    batch_values = list(range(args.batch_min, batch_max + 1))
    env = build_env(args.tensorrt_lib)
    output_path = Path(args.output_json)

    meta = {
        "created_at": now_iso(),
        "cmdline": sys.argv,
        "katago_bin": str(katago_bin.resolve()) if katago_bin.exists() else str(katago_bin),
        "trtexec_bin": str(trtexec_bin.resolve()),
        "config": str(config.resolve()) if config.exists() else str(config),
        "model": str(model.resolve()) if model.exists() else str(model),
        "max_batch": args.max_batch,
        "batch_range": [args.batch_min, batch_max],
        "stream_range": [args.stream_min, args.stream_max],
        "profile_mode": "all-static-min-eq-max",
        "shape_templates": shape_templates,
        "plan_files": [{"label": label, "path": path} for label, path in parsed_plan_files],
    }
    state = load_or_init_state(output_path, resume=(not args.no_resume), meta=meta)
    state["meta"]["last_run_at"] = now_iso()
    atomic_save_json(output_path, state)

    plan_entries: List[Dict[str, object]] = []
    if use_existing_plan_files:
        for label, plan_path in parsed_plan_files:
            plan_entries.append(
                {
                    "label": label,
                    "path": plan_path,
                    "max_batch": args.max_batch,
                    "source": "cli-plan-file",
                }
            )
    else:
        plan_path = ensure_plan(args.max_batch, args, state, output_path, env)
        plan_entries.append(
            {
                "label": f"batch{args.max_batch}",
                "path": plan_path,
                "max_batch": args.max_batch,
                "source": "katago-gtp",
            }
        )

    cases: List[Tuple[str, str, int, int]] = []
    for plan_entry in plan_entries:
        plan_label = str(plan_entry["label"])
        plan_path = str(plan_entry["path"])
        for batch_size in batch_values:
            for streams in stream_values:
                cases.append((plan_label, plan_path, batch_size, streams))

    skipped = 0
    for plan_label, _plan_path, batch_size, streams in cases:
        existing = state["results"].get(case_key_with_plan(plan_label, batch_size, streams))
        if should_skip_result(existing, args.rerun_failed):
            skipped += 1

    total = len(cases)
    done = skipped
    render_progress(done, total, f"resume skip={skipped}")
    plot_path = Path(args.plot_png)

    ok_count = 0
    err_count = 0
    for plan_label, plan_path, batch_size, streams in cases:
        key = case_key_with_plan(plan_label, batch_size, streams)
        existing = state["results"].get(key)
        if should_skip_result(existing, args.rerun_failed):
            if existing.get("status") == "ok":
                ok_count += 1
            elif existing.get("status") == "error":
                err_count += 1
            continue

        shapes = ",".join(t.replace("{batch}", str(batch_size)) for t in shape_templates)

        cmd = [
            str(trtexec_bin.resolve()),
            f"--loadEngine={plan_path}",
            f"--shapes={shapes}",
            f"--duration={args.duration_sec}",
            f"--warmUp={args.warmup_ms}",
            f"--iterations={args.iterations}",
            f"--avgRuns={args.avg_runs}",
            f"--infStreams={streams}",
            f"--device={args.device}",
            f"--useProfile={batch_size - 1}",
            "--threads",
        ]
        cmd.extend(split_csv(args.trtexec_extra_arg))

        started = time.time()
        started_at = now_iso()
        status = "ok"
        error_text = ""
        rc = -1
        raw = ""
        metrics: Dict[str, object] = {}
        try:
            rc, raw = run_command(cmd, env=env, timeout_sec=args.trtexec_timeout_sec)
            metrics = parse_trtexec_metrics(raw)
            passed = rc == 0 and "&&&& PASSED" in raw
            if not passed:
                status = "error"
                error_text = f"trtexec failed or did not report PASSED (exit={rc})"
        except Exception as e:
            status = "error"
            error_text = str(e)

        elapsed = time.time() - started
        result = {
            "status": status,
            "error": error_text,
            "plan_label": plan_label,
            "max_batch": args.max_batch,
            "batch_size": batch_size,
            "streams": streams,
            "plan_path": plan_path,
            "shapes": shapes,
            "command": join_cmd(cmd),
            "return_code": rc,
            "started_at": started_at,
            "ended_at": now_iso(),
            "elapsed_sec": elapsed,
            "metrics": metrics,
            "output_tail": output_tail(raw, lines=120),
        }
        state["results"][key] = result
        atomic_save_json(output_path, state)
        if not args.no_plot:
            plotted = draw_plot(
                state=state,
                output_png=plot_path,
                metric_name=args.plot_metric,
                include_non_ok=args.plot_include_non_ok,
            )
            if plotted:
                state["meta"]["plot_metric"] = args.plot_metric
                state["meta"]["plot_png"] = str(plot_path.resolve())
                atomic_save_json(output_path, state)

        done += 1
        if status == "ok":
            ok_count += 1
        else:
            err_count += 1
        render_progress(
            done,
            total,
            (
                f"plan={normalize_key_token(plan_label)} "
                f"batch={batch_size} s={streams} {status}"
            ),
        )

        if status == "error" and args.stop_on_error:
            print("\n[stop] stop-on-error enabled")
            break

    if not args.no_plot and done == skipped:
        plotted = draw_plot(
            state=state,
            output_png=plot_path,
            metric_name=args.plot_metric,
            include_non_ok=args.plot_include_non_ok,
        )
        state["meta"]["plot_metric"] = args.plot_metric
        state["meta"]["plot_png"] = str(plot_path.resolve()) if plotted else ""
        atomic_save_json(output_path, state)

    print()
    print(
        f"Done. total={total} skipped={skipped} ok={ok_count} error={err_count} output={output_path.resolve()}"
    )
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
