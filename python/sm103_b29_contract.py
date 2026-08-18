#!/usr/bin/env python3
"""Device-free B29 development anchor for the first SM103 optimization pass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import pathlib
import re
from typing import Any, Mapping

try:
    from sm103_contract import (
        ACCELERATED_TARGET,
        COMPUTE_CAPABILITY,
        MODEL_NAME,
        build_manifest as build_search_manifest,
        rows_for_batch,
    )
except ModuleNotFoundError:
    from python.sm103_contract import (
        ACCELERATED_TARGET,
        COMPUTE_CAPABILITY,
        MODEL_NAME,
        build_manifest as build_search_manifest,
        rows_for_batch,
    )


DEVELOPMENT_BATCH = 29
DEVELOPMENT_STREAMS = 2
DEVELOPMENT_ROWS = rows_for_batch(DEVELOPMENT_BATCH)
EXPECTED_MODEL_SHA256 = "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"
FIXED_BASELINE_BACKEND = "tensorrt"
FIXED_BASELINE_BINARY_SHA256 = "883024dc8bbc02e7f6b05b0431034652931acc760b76e7fd455dc996af278612"
FIXED_BASELINE_CONFIG_SHA256 = "807c646c5e4f6f5a3193f6da4c216782486ec2a3c3015e543a34f605ce2c0e3f"
FIXED_BASELINE_NN_EVALS_PER_SEC = 6733.719141
FIXED_BASELINE_SAMPLES = 5
ALLOWED_BASELINE_BACKENDS = ("cuda", "tensorrt")

# The request-level acceptance control is the checked-in TensorRT 10.16 replay
# against the same CUDA FP32 reference and 8192-row corpus used for every B29
# candidate.  "Same numerical regime" is deliberately defined as no more than
# 2x the control for every per-head maximum-absolute and maximum-RMSE metric.
# This admits the official CUDA FP16 control (its largest ratio is < 1.65x),
# while remaining substantially stricter than a decimal-order comparison.
TRT16_REQUEST_GATE_MULTIPLIER = 2.0
TRT16_REQUEST_GATE_CONTROL = {
    "policyProbability": {
        "maximumAbs": 0.011099457740783691,
        "maximumRmse": 0.000622167659457773,
    },
    "valueProbability": {
        "maximumAbs": 0.03413337469100952,
        "maximumRmse": 0.027866413816809654,
    },
    "scoreRaw": {
        "maximumAbs": 0.4619755744934082,
        "maximumRmse": 0.19242839515209198,
    },
    "ownershipProbability": {
        "maximumAbs": 0.007325530052185059,
        "maximumRmse": 0.0015376248629763722,
    },
}


class B29AnchorError(ValueError):
    """Raised when a provisional baseline cannot anchor B29 development."""


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise B29AnchorError(f"{name} must be a lowercase SHA-256")
    return value


def trt16_request_gate_thresholds() -> dict[str, dict[str, float]]:
    """Return the immutable B29 request thresholds derived from TRT 10.16."""

    return {
        head: {
            metric: value * TRT16_REQUEST_GATE_MULTIPLIER
            for metric, value in metrics.items()
        }
        for head, metrics in TRT16_REQUEST_GATE_CONTROL.items()
    }


def evaluate_trt16_request_gate(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one replay comparison against the fixed TRT 10.16 regime.

    Aggregate accuracy remains a separate certification gate.  This function
    intentionally consumes only ``requestGate`` and fails closed on missing,
    boolean, negative, or non-finite metrics.
    """

    request_gate = report.get("requestGate")
    if not isinstance(request_gate, Mapping):
        raise B29AnchorError("comparison has no requestGate mapping")
    thresholds = trt16_request_gate_thresholds()
    checks: dict[str, dict[str, bool]] = {}
    ratios: dict[str, dict[str, float]] = {}
    observed: dict[str, dict[str, float]] = {}
    for head, control_metrics in TRT16_REQUEST_GATE_CONTROL.items():
        head_payload = request_gate.get(head)
        if not isinstance(head_payload, Mapping):
            raise B29AnchorError(f"requestGate is missing {head}")
        checks[head] = {}
        ratios[head] = {}
        observed[head] = {}
        for metric, control in control_metrics.items():
            value = head_payload.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise B29AnchorError(
                    f"requestGate {head}.{metric} must be finite and non-negative"
                )
            observed_value = float(value)
            observed[head][metric] = observed_value
            ratios[head][metric] = observed_value / control
            checks[head][metric] = observed_value <= thresholds[head][metric]
    return {
        "control": "TensorRT 10.16.1.11 vs CUDA FP32",
        "multiplier": TRT16_REQUEST_GATE_MULTIPLIER,
        "thresholds": thresholds,
        "observed": observed,
        "ratios_to_control": ratios,
        "checks": checks,
        "passed": all(
            passed
            for head_checks in checks.values()
            for passed in head_checks.values()
        ),
    }


@dataclass(frozen=True)
class B29BaselineAnchor:
    backend: str
    batch: int
    streams: int
    measurement_kind: str
    nn_evals_per_sec_median: float
    sample_count: int
    measurement_iterations: int
    measurement_relative_spread: float
    binary_sha256: str
    config_sha256: str
    model_sha256: str
    source: str
    report_status: str = "fixed_batch_long_confirmation"
    batch_selection_fixed: bool = True
    production_ready: bool = False

    def __post_init__(self) -> None:
        if self.backend != FIXED_BASELINE_BACKEND:
            raise B29AnchorError("fixed B29 baseline must use TensorRT")
        if self.batch != DEVELOPMENT_BATCH or self.streams != DEVELOPMENT_STREAMS:
            raise B29AnchorError("B29 development requires exact batch 29 and S2")
        if self.measurement_kind != "long_stable":
            raise B29AnchorError("fixed B29 baseline must be long-stable")
        if self.measurement_iterations < 1000 or self.sample_count < FIXED_BASELINE_SAMPLES:
            raise B29AnchorError("fixed B29 baseline lacks long-confirmation samples")
        if not 0 <= self.measurement_relative_spread <= 0.10:
            raise B29AnchorError("fixed B29 baseline spread exceeds the stability gate")
        if self.nn_evals_per_sec_median != FIXED_BASELINE_NN_EVALS_PER_SEC:
            raise B29AnchorError("fixed B29 baseline throughput identity changed")
        if _sha256("binary_sha256", self.binary_sha256) != FIXED_BASELINE_BINARY_SHA256:
            raise B29AnchorError("fixed B29 baseline binary identity changed")
        if _sha256("config_sha256", self.config_sha256) != FIXED_BASELINE_CONFIG_SHA256:
            raise B29AnchorError("fixed B29 baseline config identity changed")
        if _sha256("model_sha256", self.model_sha256) != EXPECTED_MODEL_SHA256:
            raise B29AnchorError("baseline belongs to a different model")
        if not self.batch_selection_fixed or self.production_ready:
            raise B29AnchorError(
                "B29 selection is fixed, but an optimization development anchor "
                "cannot be production-ready"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_baseline_anchor(path: pathlib.Path) -> B29BaselineAnchor:
    """Load one official baseline result without treating it as certification."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("schema") != 1 or payload.get("kind") != "official-backend-baseline":
        raise B29AnchorError("B29 anchor requires an official-backend-baseline schema-1 file")
    backend = payload.get("backend")
    if backend != FIXED_BASELINE_BACKEND:
        raise B29AnchorError("fixed B29 baseline must use TensorRT")
    if payload.get("status") != "completed" or payload.get("measurement_mode") != "long_confirmation":
        raise B29AnchorError("fixed B29 baseline requires a completed long confirmation")
    if payload.get("streams") != DEVELOPMENT_STREAMS:
        raise B29AnchorError("B29 baseline must use exactly two server streams")
    device = payload.get("device")
    if not isinstance(device, dict) or device.get("compute_capability") != list(COMPUTE_CAPABILITY):
        raise B29AnchorError("B29 baseline device must be exact compute capability 10.3")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise B29AnchorError("B29 baseline has no immutable identity")
    rows = [
        row for row in payload.get("rows", [])
        if isinstance(row, dict) and row.get("batch") == DEVELOPMENT_BATCH
    ]
    if len(rows) != 1 or rows[0].get("status") != "measured":
        raise B29AnchorError("B29 baseline must contain one measured B29 row")
    row = rows[0]
    samples = row.get("nn_evals_per_sec_samples")
    if not isinstance(samples, list) or not samples:
        raise B29AnchorError("B29 baseline row has no throughput samples")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in samples):
        raise B29AnchorError("B29 baseline samples must be positive numbers")
    median = row.get("nn_evals_per_sec_median")
    if isinstance(median, bool) or not isinstance(median, (int, float)) or median <= 0:
        raise B29AnchorError("B29 baseline median must be positive")
    iterations = row.get("measurement_iterations")
    spread = row.get("measurement_relative_spread")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise B29AnchorError("B29 baseline measurement_iterations must be an integer")
    if isinstance(spread, bool) or not isinstance(spread, (int, float)):
        raise B29AnchorError("B29 baseline spread must be numeric")
    return B29BaselineAnchor(
        backend=str(backend),
        batch=DEVELOPMENT_BATCH,
        streams=DEVELOPMENT_STREAMS,
        measurement_kind=str(row.get("measurement_kind")),
        nn_evals_per_sec_median=float(median),
        sample_count=len(samples),
        measurement_iterations=iterations,
        measurement_relative_spread=float(spread),
        binary_sha256=_sha256("binary_sha256", identity.get("binary_sha256")),
        config_sha256=_sha256("config_sha256", identity.get("config_sha256")),
        model_sha256=_sha256("model_sha256", identity.get("model_sha256")),
        source=str(resolved),
    )


def build_b29_development_manifest(
    baseline_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build the shared B29 work manifest while final certification is pending."""

    anchor: dict[str, Any]
    if baseline_path is None:
        anchor = {
            "report_status": "fixed_batch_report_not_attached",
            "batch_selection_fixed": True,
            "production_ready": False,
        }
    else:
        anchor = load_baseline_anchor(baseline_path).to_dict()
    return {
        "schema": 1,
        "kind": "katago-sm103-b29-development-anchor",
        "architecture": "sm103",
        "compute_capability": list(COMPUTE_CAPABILITY),
        "accelerated_target": ACCELERATED_TARGET,
        "model_name": MODEL_NAME,
        "batch": DEVELOPMENT_BATCH,
        "streams": DEVELOPMENT_STREAMS,
        "rows": DEVELOPMENT_ROWS,
        "selection_provenance": "user_fixed_after_long_confirmation",
        "batch_selection_fixed": True,
        "production_ready": False,
        "baseline": anchor,
        "search_contract": build_search_manifest(DEVELOPMENT_BATCH),
    }


__all__ = (
    "ALLOWED_BASELINE_BACKENDS",
    "B29AnchorError",
    "B29BaselineAnchor",
    "DEVELOPMENT_BATCH",
    "DEVELOPMENT_ROWS",
    "DEVELOPMENT_STREAMS",
    "EXPECTED_MODEL_SHA256",
    "FIXED_BASELINE_BACKEND",
    "FIXED_BASELINE_BINARY_SHA256",
    "FIXED_BASELINE_CONFIG_SHA256",
    "FIXED_BASELINE_NN_EVALS_PER_SEC",
    "FIXED_BASELINE_SAMPLES",
    "TRT16_REQUEST_GATE_CONTROL",
    "TRT16_REQUEST_GATE_MULTIPLIER",
    "build_b29_development_manifest",
    "evaluate_trt16_request_gate",
    "load_baseline_anchor",
    "trt16_request_gate_thresholds",
)
