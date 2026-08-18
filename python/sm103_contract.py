#!/usr/bin/env python3
"""Pure-data SM103/B300 search contract for KataGo's b11 transformer.

This module deliberately has no CUDA, PyTorch, compiler, or device-runtime
dependency.  It describes work that may be compiled and measured later; merely
importing it is safe while a GPU benchmark is running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any


ARCHITECTURE = "sm103"
GPU_CLASS = "b300"
COMPUTE_CAPABILITY = (10, 3)
ACCELERATED_TARGET = "sm_103a"
ISA_FAMILY = "sm100_tcgen05"
MODEL_NAME = "b11c768h12nbt3tflrs-fson-silu"

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32

BOARD_SIDE = 19
SPATIAL_TOKENS = BOARD_SIDE * BOARD_SIDE
MODEL_CHANNELS = 384
TRUNK_CHANNELS = 768
FFN_CHANNELS = 1152
ATTENTION_HEADS = 12
ATTENTION_HEAD_DIM = 32

FP16_INSTRUCTION_K = 16
FP4_ULTRA_INSTRUCTION_K = 96


class ContractValidationError(ValueError):
    """Raised when data cannot be represented by the SM103 search contract."""


def _require_plain_int(name: str, value: object) -> int:
    # bool is an int subclass, but accepting True as batch 1 is not a useful
    # contract and can silently select the wrong fixed-batch artifact.
    if type(value) is not int:  # noqa: E721 - exact type is intentional
        raise ContractValidationError(f"{name} must be an integer, got {value!r}")
    return value


def validate_batch(batch: object) -> int:
    """Validate and return a supported fixed physical batch size."""

    value = _require_plain_int("batch", batch)
    if not MIN_BATCH_SIZE <= value <= MAX_BATCH_SIZE:
        raise ContractValidationError(
            f"batch must be in [{MIN_BATCH_SIZE}, {MAX_BATCH_SIZE}], got {value}"
        )
    return value


def validate_target(target: object) -> str:
    """Reject generic or wrong-architecture artifacts for accelerated kernels."""

    if type(target) is not str or target != ACCELERATED_TARGET:  # noqa: E721
        raise ContractValidationError(
            f"accelerated target must be {ACCELERATED_TARGET!r}, got {target!r}"
        )
    return ACCELERATED_TARGET


def rows_for_batch(batch: object) -> int:
    """Return b11's flattened spatial row count, R = 361 * batch."""

    return SPATIAL_TOKENS * validate_batch(batch)


def _validate_candidate_id(candidate_id: object) -> str:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ContractValidationError("candidate_id must be a non-empty string")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    if any(character not in allowed for character in candidate_id):
        raise ContractValidationError(
            "candidate_id may contain only lowercase ASCII letters, digits, '-' and '_'"
        )
    return candidate_id


def _validate_cluster_shape(
    cluster_shape: object, cta_count: object
) -> tuple[int, int, int]:
    count = _require_plain_int("cta_count", cta_count)
    if count not in (1, 2):
        raise ContractValidationError(f"cta_count must be 1 or 2, got {count}")
    if not isinstance(cluster_shape, tuple) or len(cluster_shape) != 3:
        raise ContractValidationError(
            "cluster_shape must be an immutable (x, y, z) tuple"
        )
    if any(type(axis) is not int or axis <= 0 for axis in cluster_shape):
        raise ContractValidationError("cluster_shape axes must be positive integers")

    allowed = {
        1: ((1, 1, 1),),
        2: ((2, 1, 1), (1, 2, 1)),
    }
    if cluster_shape not in allowed[count]:
        raise ContractValidationError(
            f"illegal {count}CTA cluster_shape {cluster_shape!r}; "
            f"expected one of {allowed[count]!r}"
        )
    return cluster_shape


@dataclass(frozen=True)
class DeviceContract:
    architecture: str = ARCHITECTURE
    gpu_class: str = GPU_CLASS
    compute_capability: tuple[int, int] = COMPUTE_CAPABILITY
    accelerated_target: str = ACCELERATED_TARGET
    isa_family: str = ISA_FAMILY
    required_features: tuple[str, ...] = (
        "tcgen05",
        "tmem",
        "tma",
        "thread_block_clusters",
    )

    def __post_init__(self) -> None:
        expected = (
            ARCHITECTURE,
            GPU_CLASS,
            COMPUTE_CAPABILITY,
            ACCELERATED_TARGET,
            ISA_FAMILY,
        )
        actual = (
            self.architecture,
            self.gpu_class,
            self.compute_capability,
            self.accelerated_target,
            self.isa_family,
        )
        if actual != expected:
            raise ContractValidationError(
                f"invalid B300 device identity {actual!r}; expected {expected!r}"
            )
        validate_target(self.accelerated_target)
        if self.required_features != (
            "tcgen05",
            "tmem",
            "tma",
            "thread_block_clusters",
        ):
            raise ContractValidationError("B300 required feature set must not be changed")


@dataclass(frozen=True)
class GemmProblem:
    """One row-major logical GEMM boundary in M, N, K order."""

    boundary: str
    batch: int
    m: int
    n: int
    k: int
    input_dtype: str = "float16"
    output_dtype: str = "float16"

    def __post_init__(self) -> None:
        batch = validate_batch(self.batch)
        if self.boundary not in _BOUNDARY_NK:
            raise ContractValidationError(f"unknown b11 GEMM boundary {self.boundary!r}")
        expected_n, expected_k = _BOUNDARY_NK[self.boundary]
        expected = (rows_for_batch(batch), expected_n, expected_k)
        if (self.m, self.n, self.k) != expected:
            raise ContractValidationError(
                f"{self.boundary} must have MNK={expected!r}, got "
                f"{(self.m, self.n, self.k)!r}"
            )
        if (self.input_dtype, self.output_dtype) != ("float16", "float16"):
            raise ContractValidationError("first-pass b11 GEMMs must have FP16 I/O")

    @property
    def mnk(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.k)


# Dimensions are N, K. M is always R = 361 * fixed physical batch.
_BOUNDARY_NK: dict[str, tuple[int, int]] = {
    "wide_qkv": (1152, 384),
    "dual_ffn": (2304, 384),
    "linear2": (384, 1152),
    "outproj": (384, 384),
    "preconv": (384, 768),
    "postconv": (768, 384),
    "wide_head": (384, 768),
}
BOUNDARY_NAMES = tuple(_BOUNDARY_NK)


def gemm_problems(batch: object) -> tuple[GemmProblem, ...]:
    """Materialize all primary b11 FP16 GEMM boundaries for one batch."""

    fixed_batch = validate_batch(batch)
    rows = rows_for_batch(fixed_batch)
    return tuple(
        GemmProblem(name, fixed_batch, rows, n, k)
        for name, (n, k) in _BOUNDARY_NK.items()
    )


@dataclass(frozen=True)
class FlashProblem:
    batch: int
    sequence_length: int = SPATIAL_TOKENS
    heads: int = ATTENTION_HEADS
    head_dim: int = ATTENTION_HEAD_DIM
    causal: bool = False
    mask: str = "none"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        validate_batch(self.batch)
        actual = (
            self.sequence_length,
            self.heads,
            self.head_dim,
            self.causal,
            self.mask,
            self.dtype,
        )
        expected = (SPATIAL_TOKENS, ATTENTION_HEADS, ATTENTION_HEAD_DIM, False, "none", "float16")
        if actual != expected:
            raise ContractValidationError(
                f"b11 FlashAttention problem must be {expected!r}, got {actual!r}"
            )


def flash_problem(batch: object) -> FlashProblem:
    return FlashProblem(validate_batch(batch))


@dataclass(frozen=True)
class FlashCandidate:
    candidate_id: str
    tile_m: int
    tile_n: int
    q_stages: int
    kv_stages: int
    persistent: str
    cta_count: int = 1
    cluster_shape: tuple[int, int, int] = (1, 1, 1)
    threads: int = 512
    accumulator: str = "float32"
    qkv_transport: str = "tma"
    output_transport: str = "tma"
    p_residency: str = "tmem"
    use_tcgen05_ld_red: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_candidate_id(self.candidate_id)
        if self.tile_m not in (64, 128) or self.tile_n not in (64, 128):
            raise ContractValidationError("FA tile_m and tile_n must each be 64 or 128")
        if self.q_stages not in (1, 2):
            raise ContractValidationError("FA q_stages must be 1 or 2")
        if self.kv_stages not in (3, 4, 8, 16, 24):
            raise ContractValidationError(
                "FA kv_stages must be one of 3, 4, 8, 16, or 24"
            )
        if self.persistent not in ("static", "none"):
            raise ContractValidationError("FA persistent must be 'static' or 'none'")
        _validate_cluster_shape(self.cluster_shape, self.cta_count)
        if self.cta_count != 1:
            raise ContractValidationError("D32 first-round FA candidates are 1CTA")
        if self.threads != 512:
            raise ContractValidationError("SM103 upstream FA configuration uses 512 threads")
        if self.accumulator != "float32":
            raise ContractValidationError("QK and PV must use FP32 accumulation")
        if (self.qkv_transport, self.output_transport, self.p_residency) != (
            "tma",
            "tma",
            "tmem",
        ):
            raise ContractValidationError("SM103 FA transport must remain TMA/TMEM")
        if type(self.use_tcgen05_ld_red) is not bool:  # noqa: E721
            raise ContractValidationError("use_tcgen05_ld_red must be a bool")
        if self.use_tcgen05_ld_red:
            raise ContractValidationError("tcgen05.ld.red is disabled for b11 D32")


FLASH_UPSTREAM_CONTROL = FlashCandidate(
    candidate_id="fa-upstream-m128-n128-q2-kv24-static",
    tile_m=128,
    tile_n=128,
    q_stages=2,
    kv_stages=24,
    persistent="static",
)

# This is an axis-isolating first pass, not a Cartesian-product autotune.  It
# keeps the upstream winner and changes one major scheduling choice at a time.
FLASH_FIRST_ROUND_CANDIDATES = (
    FLASH_UPSTREAM_CONTROL,
    FlashCandidate("fa-m64-n128-q2-kv24-static", 64, 128, 2, 24, "static"),
    FlashCandidate("fa-m128-n64-q2-kv24-static", 128, 64, 2, 24, "static"),
    FlashCandidate("fa-m64-n64-q2-kv24-static", 64, 64, 2, 24, "static"),
    FlashCandidate("fa-m128-n128-q1-kv24-static", 128, 128, 1, 24, "static"),
    FlashCandidate("fa-m128-n128-q2-kv3-static", 128, 128, 2, 3, "static"),
    FlashCandidate("fa-m128-n128-q2-kv4-static", 128, 128, 2, 4, "static"),
    FlashCandidate("fa-m128-n128-q2-kv8-static", 128, 128, 2, 8, "static"),
    FlashCandidate("fa-m128-n128-q2-kv16-static", 128, 128, 2, 16, "static"),
    FlashCandidate("fa-m128-n128-q2-kv24-nonpersistent", 128, 128, 2, 24, "none"),
)


@dataclass(frozen=True)
class GemmCandidate:
    candidate_id: str
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    cta_count: int
    cluster_shape: tuple[int, int, int]
    accumulator: str
    persistent: str
    tma_store: bool
    mma: str = "tcgen05"
    input_dtype: str = "float16"
    output_dtype: str = "float16"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_candidate_id(self.candidate_id)
        _validate_cluster_shape(self.cluster_shape, self.cta_count)
        legal_m = {1: (64, 128), 2: (128, 256)}[self.cta_count]
        if self.tile_m not in legal_m:
            raise ContractValidationError(
                f"{self.cta_count}CTA FP16 tile_m must be one of {legal_m!r}"
            )
        if self.tile_n not in (64, 128, 192, 256):
            raise ContractValidationError("FP16 tile_n must be 64, 128, 192, or 256")
        if self.tile_k not in (32, 64) or self.tile_k % FP16_INSTRUCTION_K:
            raise ContractValidationError("FP16 tile_k must be 32 or 64 and K16 aligned")
        if type(self.stages) is not int or not 2 <= self.stages <= 8:
            raise ContractValidationError("FP16 stages must be an integer in [2, 8]")
        if self.accumulator not in ("float16", "float32"):
            raise ContractValidationError("accumulator must be float16 or float32")
        if self.persistent not in ("static", "dynamic"):
            raise ContractValidationError("persistent must be static or dynamic")
        if type(self.tma_store) is not bool:
            raise ContractValidationError("tma_store must be a bool")
        if self.mma != "tcgen05":
            raise ContractValidationError("SM103 GEMM candidates must use tcgen05")
        if (self.input_dtype, self.output_dtype) != ("float16", "float16"):
            raise ContractValidationError("first-round GEMM candidates require FP16 I/O")


def _gemm_candidate(
    candidate_id: str,
    tile_m: int,
    tile_n: int,
    *,
    tile_k: int = 64,
    stages: int = 3,
    cta_count: int = 1,
    cluster_shape: tuple[int, int, int] = (1, 1, 1),
    accumulator: str = "float32",
    persistent: str = "static",
    tma_store: bool = True,
) -> GemmCandidate:
    return GemmCandidate(
        candidate_id,
        tile_m,
        tile_n,
        tile_k,
        stages,
        cta_count,
        cluster_shape,
        accumulator,
        persistent,
        tma_store,
    )


GEMM_FP16_FIRST_ROUND_CANDIDATES = (
    _gemm_candidate("gemm-1cta-m64-n128-k64-a32", 64, 128),
    _gemm_candidate("gemm-1cta-m128-n128-k64-a32", 128, 128),
    _gemm_candidate("gemm-1cta-m128-n192-k64-a32", 128, 192),
    _gemm_candidate("gemm-1cta-m128-n256-k64-a32", 128, 256),
    _gemm_candidate(
        "gemm-1cta-m128-n128-k32-s4-dynamic-a32",
        128,
        128,
        tile_k=32,
        stages=4,
        persistent="dynamic",
    ),
    _gemm_candidate(
        "gemm-1cta-m128-n256-k64-a16", 128, 256, accumulator="float16"
    ),
    _gemm_candidate(
        "gemm-1cta-m128-n256-k64-a32-no-tma-store", 128, 256, tma_store=False
    ),
    _gemm_candidate(
        "gemm-2cta-m128-n128-k64-a32",
        128,
        128,
        cta_count=2,
        cluster_shape=(2, 1, 1),
    ),
    _gemm_candidate(
        "gemm-2cta-m256-n128-k64-a32",
        256,
        128,
        cta_count=2,
        cluster_shape=(2, 1, 1),
    ),
    _gemm_candidate(
        "gemm-2cta-m256-n192-k64-a32",
        256,
        192,
        cta_count=2,
        cluster_shape=(2, 1, 1),
    ),
    _gemm_candidate(
        "gemm-2cta-m256-n256-k64-a32",
        256,
        256,
        cta_count=2,
        cluster_shape=(2, 1, 1),
    ),
    _gemm_candidate(
        "gemm-2cta-m256-n256-k64-cluster1x2-a32",
        256,
        256,
        cta_count=2,
        cluster_shape=(1, 2, 1),
    ),
    _gemm_candidate(
        "gemm-2cta-m256-n256-k64-a16",
        256,
        256,
        cta_count=2,
        cluster_shape=(2, 1, 1),
        accumulator="float16",
    ),
    _gemm_candidate(
        "gemm-2cta-m256-n256-k64-dynamic-a32",
        256,
        256,
        cta_count=2,
        cluster_shape=(2, 1, 1),
        persistent="dynamic",
    ),
)


def _index_candidates(candidates: tuple[Any, ...]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for candidate in candidates:
        if candidate.candidate_id in index:
            raise RuntimeError(f"duplicate candidate ID {candidate.candidate_id!r}")
        index[candidate.candidate_id] = candidate
    return index


_FLASH_CANDIDATE_INDEX = _index_candidates(FLASH_FIRST_ROUND_CANDIDATES)
_GEMM_CANDIDATE_INDEX = _index_candidates(GEMM_FP16_FIRST_ROUND_CANDIDATES)


def flash_candidate(candidate_id: object) -> FlashCandidate:
    key = _validate_candidate_id(candidate_id)
    try:
        return _FLASH_CANDIDATE_INDEX[key]
    except KeyError as exc:
        raise ContractValidationError(f"unknown first-round FA candidate {key!r}") from exc


def gemm_candidate(candidate_id: object) -> GemmCandidate:
    key = _validate_candidate_id(candidate_id)
    try:
        return _GEMM_CANDIDATE_INDEX[key]
    except KeyError as exc:
        raise ContractValidationError(
            f"unknown first-round FP16 GEMM candidate {key!r}"
        ) from exc


def validate_candidate(candidate: object) -> None:
    """Validate a typed candidate and require exact first-round registration."""

    if isinstance(candidate, FlashCandidate):
        candidate.validate()
        registered = flash_candidate(candidate.candidate_id)
        if candidate != registered:
            raise ContractValidationError(
                f"FA candidate {candidate.candidate_id!r} differs from its registered config"
            )
        return
    if isinstance(candidate, GemmCandidate):
        candidate.validate()
        registered = gemm_candidate(candidate.candidate_id)
        if candidate != registered:
            raise ContractValidationError(
                f"GEMM candidate {candidate.candidate_id!r} differs from its registered config"
            )
        return
    raise ContractValidationError(
        f"candidate must be FlashCandidate or GemmCandidate, got {type(candidate).__name__}"
    )


@dataclass(frozen=True)
class Fp4UltraTrack:
    """Opt-in SM103-only quantization track, separate from FP16 bring-up."""

    track_id: str = "fp4-ultra-k96"
    accelerated_target: str = ACCELERATED_TARGET
    instruction_k: int = FP4_ULTRA_INSTRUCTION_K
    operand_dtype: str = "nvfp4"
    accumulator: str = "float32"
    enabled_by_default: bool = False
    requires_activation_quantization: bool = True
    requires_scale_factor_tma_pipeline: bool = True
    requires_precision_recertification: bool = True
    eligible_boundaries: tuple[str, ...] = BOUNDARY_NAMES

    def __post_init__(self) -> None:
        _validate_candidate_id(self.track_id)
        validate_target(self.accelerated_target)
        if self.instruction_k != FP4_ULTRA_INSTRUCTION_K:
            raise ContractValidationError("B300 FP4 Ultra instruction K must be 96")
        if (self.operand_dtype, self.accumulator) != ("nvfp4", "float32"):
            raise ContractValidationError("FP4 Ultra track must use NVFP4 with FP32 accumulation")
        flags = (
            self.enabled_by_default,
            self.requires_activation_quantization,
            self.requires_scale_factor_tma_pipeline,
            self.requires_precision_recertification,
        )
        if flags != (False, True, True, True):
            raise ContractValidationError(
                "FP4 must remain opt-in with quantization, scale, and accuracy gates"
            )
        if self.eligible_boundaries != BOUNDARY_NAMES:
            raise ContractValidationError("FP4 eligible boundaries must match the b11 GEMM set")


FP4_ULTRA_TRACK = Fp4UltraTrack()


@dataclass(frozen=True)
class Fp4UltraProblem:
    """One opt-in NVFP4 kernel boundary with explicit quantization semantics."""

    boundary: str
    batch: int
    m: int
    n: int
    k: int
    source_activation_dtype: str = "float16"
    operand_dtype: str = "nvfp4"
    output_dtype: str = "float16"
    accumulator: str = "float32"
    instruction_k: int = FP4_ULTRA_INSTRUCTION_K

    def __post_init__(self) -> None:
        batch = validate_batch(self.batch)
        if self.boundary not in _BOUNDARY_NK:
            raise ContractValidationError(
                f"unknown b11 FP4 boundary {self.boundary!r}"
            )
        expected_n, expected_k = _BOUNDARY_NK[self.boundary]
        expected = (rows_for_batch(batch), expected_n, expected_k)
        if (self.m, self.n, self.k) != expected:
            raise ContractValidationError(
                f"{self.boundary} FP4 problem must have MNK={expected!r}, got "
                f"{(self.m, self.n, self.k)!r}"
            )
        if (
            self.source_activation_dtype,
            self.operand_dtype,
            self.output_dtype,
            self.accumulator,
        ) != ("float16", "nvfp4", "float16", "float32"):
            raise ContractValidationError(
                "FP4 Ultra requires FP16 source activations, NVFP4 operands, "
                "FP16 output, and FP32 accumulation"
            )
        if self.instruction_k != FP4_ULTRA_INSTRUCTION_K:
            raise ContractValidationError("B300 FP4 Ultra instruction K must be 96")
        if self.k % self.instruction_k != 0:
            raise ContractValidationError(
                f"{self.boundary} K={self.k} is not aligned to K{self.instruction_k}"
            )

    @property
    def mnk(self) -> tuple[int, int, int]:
        return (self.m, self.n, self.k)


def fp4_ultra_problems(batch: object) -> tuple[Fp4UltraProblem, ...]:
    """Return K96-exact b11 boundaries or fail instead of silently padding K."""

    problems = gemm_problems(batch)
    misaligned = tuple(
        problem.boundary
        for problem in problems
        if problem.k % FP4_ULTRA_TRACK.instruction_k != 0
    )
    if misaligned:
        raise ContractValidationError(
            f"FP4 Ultra boundaries are not K96 aligned: {misaligned!r}"
        )
    return tuple(
        Fp4UltraProblem(
            problem.boundary,
            problem.batch,
            problem.m,
            problem.n,
            problem.k,
        )
        for problem in problems
    )


DEVICE = DeviceContract()


def _records(values: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [asdict(value) for value in values]


def build_manifest(
    batch: object, *, target: object = ACCELERATED_TARGET
) -> dict[str, Any]:
    """Build a fresh JSON-serializable manifest for a fixed physical batch."""

    fixed_batch = validate_batch(batch)
    validate_target(target)
    fp16_problems = gemm_problems(fixed_batch)
    fp4_problems = fp4_ultra_problems(fixed_batch)
    return {
        "schema_version": 1,
        "kind": "katago-b11-sm103-search-contract",
        "device": asdict(DEVICE),
        "network": {
            "name": MODEL_NAME,
            "board_side": BOARD_SIDE,
            "spatial_tokens": SPATIAL_TOKENS,
            "trunk_channels": TRUNK_CHANNELS,
            "transformer_channels": MODEL_CHANNELS,
            "ffn_channels": FFN_CHANNELS,
        },
        "fixed_batch": fixed_batch,
        "rows": rows_for_batch(fixed_batch),
        "flash_attention": {
            "problem": asdict(flash_problem(fixed_batch)),
            "upstream_control_candidate_id": FLASH_UPSTREAM_CONTROL.candidate_id,
            "first_round_candidates": _records(FLASH_FIRST_ROUND_CANDIDATES),
        },
        "fp16_gemm": {
            "problems": _records(fp16_problems),
            "first_round_candidates": _records(GEMM_FP16_FIRST_ROUND_CANDIDATES),
        },
        "experimental_tracks": {
            "fp4_ultra_k96": {
                "contract": asdict(FP4_ULTRA_TRACK),
                "problems": _records(fp4_problems),
            }
        },
    }


def manifest_json(
    batch: object, *, target: object = ACCELERATED_TARGET, indent: int | None = 2
) -> str:
    """Serialize the manifest without touching the filesystem or GPU runtime."""

    return json.dumps(
        build_manifest(batch, target=target),
        indent=indent,
        sort_keys=True,
    )


__all__ = (
    "ACCELERATED_TARGET",
    "ARCHITECTURE",
    "ATTENTION_HEAD_DIM",
    "ATTENTION_HEADS",
    "BOARD_SIDE",
    "BOUNDARY_NAMES",
    "COMPUTE_CAPABILITY",
    "ContractValidationError",
    "DEVICE",
    "DeviceContract",
    "FLASH_FIRST_ROUND_CANDIDATES",
    "FLASH_UPSTREAM_CONTROL",
    "FP4_ULTRA_INSTRUCTION_K",
    "FP4_ULTRA_TRACK",
    "Fp4UltraProblem",
    "FlashCandidate",
    "FlashProblem",
    "GEMM_FP16_FIRST_ROUND_CANDIDATES",
    "GPU_CLASS",
    "GemmCandidate",
    "GemmProblem",
    "ISA_FAMILY",
    "MODEL_NAME",
    "MODEL_CHANNELS",
    "SPATIAL_TOKENS",
    "TRUNK_CHANNELS",
    "build_manifest",
    "flash_candidate",
    "flash_problem",
    "fp4_ultra_problems",
    "gemm_candidate",
    "gemm_problems",
    "manifest_json",
    "rows_for_batch",
    "validate_batch",
    "validate_candidate",
    "validate_target",
)
