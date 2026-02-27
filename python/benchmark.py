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


def case_key(plan_batch: int, infer_batch: int, streams: int, cuda_graph: bool) -> str:
    return f"pb{plan_batch}_ib{infer_batch}_s{streams}_g{1 if cuda_graph else 0}"


def should_skip_result(existing: Optional[Dict[str, object]], rerun_failed: bool) -> bool:
    if not existing:
        return False
    status = existing.get("status")
    if status == "ok":
        return True
    if status == "error" and not rerun_failed:
        return True
    return False


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
        description="Generate KataGo TensorRT plans via katago and benchmark with trtexec.",
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
    parser.add_argument("--plan-batch-min", type=int, default=1)
    parser.add_argument("--plan-batch-max", type=int, default=16)
    parser.add_argument("--infer-batch-max", type=int, default=None)
    parser.add_argument("--stream-min", type=int, default=1)
    parser.add_argument("--stream-max", type=int, default=4)
    parser.add_argument("--graph-modes", default="off,on", help="Comma separated: off,on")
    parser.add_argument(
        "--mismatch-streams",
        default="1,2",
        help="Comma separated streams used when infer_batch != plan_batch (graph is fixed on).",
    )
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
    parser.add_argument("--smoke", action="store_true", help="Quick smoke mode")
    args = parser.parse_args()

    if args.smoke:
        args.plan_batch_min = 1
        args.plan_batch_max = min(args.plan_batch_max, 1)
        args.infer_batch_max = 1
        args.stream_min = 1
        args.stream_max = 1
        args.graph_modes = "off,on"
        args.duration_sec = min(args.duration_sec, 1.0)
        args.warmup_ms = 0
        args.iterations = min(args.iterations, 20)
        args.avg_runs = min(args.avg_runs, 10)

    if args.plan_batch_min <= 0 or args.plan_batch_max < args.plan_batch_min:
        raise ValueError("Invalid plan batch range")
    if args.stream_min <= 0 or args.stream_max < args.stream_min:
        raise ValueError("Invalid stream range")
    if args.infer_batch_max is not None and args.infer_batch_max <= 0:
        raise ValueError("infer-batch-max must be positive")

    katago_bin = Path(args.katago_bin)
    trtexec_bin = Path(args.trtexec_bin)
    config = Path(args.config)
    model = Path(args.model)
    for path_obj, label in [(katago_bin, "katago"), (trtexec_bin, "trtexec"), (config, "config"), (model, "model")]:
        if not path_obj.exists():
            raise FileNotFoundError(f"{label} path not found: {path_obj}")

    shape_templates = split_csv(args.shape_template)
    if not shape_templates:
        shape_templates = [
            "input_spatial:{batch}x22x19x19",
            "input_global:{batch}x19",
        ]

    graph_modes = parse_graph_modes(args.graph_modes)
    mismatch_streams = parse_positive_int_csv(args.mismatch_streams, "mismatch-streams")
    stream_values = list(range(args.stream_min, args.stream_max + 1))
    env = build_env(args.tensorrt_lib)
    output_path = Path(args.output_json)

    meta = {
        "created_at": now_iso(),
        "cmdline": sys.argv,
        "katago_bin": str(katago_bin.resolve()),
        "trtexec_bin": str(trtexec_bin.resolve()),
        "config": str(config.resolve()),
        "model": str(model.resolve()),
        "shape_templates": shape_templates,
    }
    state = load_or_init_state(output_path, resume=(not args.no_resume), meta=meta)
    state["meta"]["last_run_at"] = now_iso()
    atomic_save_json(output_path, state)

    cases: List[Tuple[int, int, int, bool]] = []
    for plan_batch in range(args.plan_batch_min, args.plan_batch_max + 1):
        infer_max = plan_batch
        if args.infer_batch_max is not None:
            infer_max = min(infer_max, args.infer_batch_max)
        for infer_batch in range(1, infer_max + 1):
            # For mismatched plan/infer batches, only test graph-on on selected streams.
            if infer_batch != plan_batch:
                for streams in mismatch_streams:
                    cases.append((plan_batch, infer_batch, streams, True))
                continue
            for streams in stream_values:
                for graph_mode in graph_modes:
                    cases.append((plan_batch, infer_batch, streams, graph_mode))

    skipped = 0
    for pb, ib, s, g in cases:
        existing = state["results"].get(case_key(pb, ib, s, g))
        if should_skip_result(existing, args.rerun_failed):
            skipped += 1

    total = len(cases)
    done = skipped
    render_progress(done, total, f"resume skip={skipped}")

    ok_count = 0
    err_count = 0
    for plan_batch, infer_batch, streams, graph_mode in cases:
        key = case_key(plan_batch, infer_batch, streams, graph_mode)
        existing = state["results"].get(key)
        if should_skip_result(existing, args.rerun_failed):
            if existing.get("status") == "ok":
                ok_count += 1
            elif existing.get("status") == "error":
                err_count += 1
            continue

        plan_path = ensure_plan(plan_batch, args, state, output_path, env)
        shapes = ",".join(t.replace("{batch}", str(infer_batch)) for t in shape_templates)

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
            "--threads",
        ]
        if graph_mode:
            cmd.append("--useCudaGraph")
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
            "plan_batch": plan_batch,
            "infer_batch": infer_batch,
            "streams": streams,
            "cuda_graph": graph_mode,
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

        done += 1
        if status == "ok":
            ok_count += 1
        else:
            err_count += 1
        render_progress(
            done,
            total,
            f"pb={plan_batch} ib={infer_batch} s={streams} g={1 if graph_mode else 0} {status}",
        )

        if status == "error" and args.stop_on_error:
            print("\n[stop] stop-on-error enabled")
            break

    print()
    print(
        f"Done. total={total} skipped={skipped} ok={ok_count} error={err_count} output={output_path.resolve()}"
    )
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
