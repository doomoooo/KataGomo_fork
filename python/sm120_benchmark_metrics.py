"""Common whole-graph benchmark metric handling for SM120 search tools."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable


MIN_LONG_ITERATIONS = 1000
MIN_STABLE_SAMPLES = 2


def benchmark_throughput(record: dict) -> float:
    """Return the natural two-server throughput from one benchmark record."""
    value = record.get("combinedNNEvalsPerSec")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("benchmark JSON has no finite combinedNNEvalsPerSec")
    return float(value)


def summarize_throughput(
    samples: Iterable[float], *, iterations: int, warmup: int,
) -> dict:
    """Summarize a run and mark it stable only when it is a long measurement."""
    values = [float(value) for value in samples]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("throughput samples must contain finite values")
    median = statistics.median(values)
    result = {
        "nn_evals_per_sec_median": median,
        "nn_evals_per_sec_min": min(values),
        "nn_evals_per_sec_max": max(values),
        "measurement_iterations": int(iterations),
        "measurement_warmup": int(warmup),
        "measurement_sample_count": len(values),
        "measurement_kind": "short_scan",
    }
    if (
        iterations >= MIN_LONG_ITERATIONS
        and len(values) >= MIN_STABLE_SAMPLES
    ):
        result.update({
            "measurement_kind": "long_stable",
            "stable_long_nn_evals_per_sec": median,
            "stable_long_nn_evals_per_sec_min": min(values),
            "stable_long_nn_evals_per_sec_max": max(values),
        })
    return result


def require_stable_throughput(row: dict) -> float:
    """Return the final-report metric, rejecting short-only measurements."""
    value = row.get("stable_long_nn_evals_per_sec")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(
            "final report requires long_stable measurements with at least "
            f"{MIN_LONG_ITERATIONS} iterations and {MIN_STABLE_SAMPLES} samples"
        )
    return float(value)
