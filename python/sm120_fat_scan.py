#!/usr/bin/env python3
"""Pure helpers for exact-batch SM120 TileLang fat-scan bundles.

This module deliberately has no CUDA, torch, or TileLang imports.  It owns the
stable symbol naming, request validation, generated registry text, and the
TileLang ``debug.h`` symbol isolation needed when many generated translation
units are linked into one executable.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections.abc import Iterable


FAMILIES = (
    "dual_ffn", "wide_qkv", "qkv_rope", "linear2", "outproj", "fa4",
)
TILELANG_FAMILIES = ("dual_ffn", "wide_qkv", "linear2", "outproj")

_FAMILY_ABI = {
    "dual_ffn": {
        "tactic_type": "FusedFFNAotTactic",
        "getter": "getSm120SearchFfnFatTactics",
        "launch_prefix": "sm120_search_ffn_fat_launch",
        "launch_args": (
            "const half*, const half*, const half*, half*, cudaStream_t"
        ),
    },
    "wide_qkv": {
        "tactic_type": "WideQKVAotTactic",
        "getter": "getSm120SearchQkvFatTactics",
        "launch_prefix": "sm120_search_qkv_fat_launch",
        "launch_args": "const half*, const half*, half*, cudaStream_t",
    },
    "qkv_rope": {
        "tactic_type": "WideQKVRopeAotTactic",
        "getter": "getSm120SearchQkvRopeFatTactics",
        "launch_prefix": "sm120_search_qkv_rope_fat_launch",
        "launch_args": (
            "const half*, const half*, const half*, half*, cudaStream_t"
        ),
    },
    "linear2": {
        "tactic_type": "ResidualGemmAotTactic",
        "getter": "getSm120SearchLinear2FatTactics",
        "launch_prefix": "sm120_search_linear2_fat_launch",
        "launch_args": "const half*, const half*, half*, int, cudaStream_t",
    },
    "outproj": {
        "tactic_type": "ResidualGemmAotTactic",
        "getter": "getSm120SearchOutprojFatTactics",
        "launch_prefix": "sm120_search_outproj_fat_launch",
        "launch_args": "const half*, const half*, half*, int, cudaStream_t",
    },
    "fa4": {
        "tactic_type": "FA4AotTactic",
        "getter": "getSm120SearchFA4FatTactics",
        "launch_prefix": "sm120_search_fa4_fat_launch",
        "launch_args": (
            "void*, void*, void*, void*, int, int, int, int, float, "
            "bool, cudaStream_t"
        ),
    },
}

_DEBUG_NAMES = (
    "PrintTraits",
    "debug_print_var",
    "debug_print_buffer_value",
    "device_assert",
    "device_assert_with_msg",
    "debug_print_msg",
)
_DEBUG_INCLUDE = "#include <tl_templates/cuda/debug.h>"
_SYMBOL_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_family(family: str) -> None:
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(FAMILIES)}")


def validate_symbol_token(token: str) -> None:
    if not _SYMBOL_TOKEN_RE.fullmatch(token):
        raise ValueError(f"invalid C/C++ symbol token: {token!r}")


def symbol_token(family: str, batch: int, candidate_id: str) -> str:
    """Return a deterministic, collision-resistant C identifier fragment."""
    validate_family(family)
    if batch < 1:
        raise ValueError("batch must be positive")
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    return f"{family}_b{batch}_{digest}"


def launch_symbol(family: str, token: str) -> str:
    validate_family(family)
    validate_symbol_token(token)
    return f"{_FAMILY_ABI[family]['launch_prefix']}_{token}"


def isolate_tilelang_debug_symbols(source: str, token: str) -> str:
    """Give all externally visible TileLang debug helpers TU-local names.

    TileLang's ``debug.h`` defines explicit device specializations and ordinary
    device functions in a header.  Linking two generated TUs therefore fails
    with duplicate definitions.  Token-renaming every public helper before the
    include preserves generated call sites while making their linker names
    unique.  The macros stay active through the generated kernel and are
    undefined only at the end of the TU.
    """
    validate_symbol_token(token)
    occurrences = source.count(_DEBUG_INCLUDE)
    if occurrences != 1:
        raise ValueError(
            f"expected one TileLang debug include, found {occurrences}"
        )
    prefix = "\n".join(
        f"#define {name} sm120_tl_{name}_{token}" for name in _DEBUG_NAMES
    )
    suffix = "\n".join(f"#undef {name}" for name in reversed(_DEBUG_NAMES))
    return source.replace(_DEBUG_INCLUDE, f"{prefix}\n{_DEBUG_INCLUDE}") + (
        f"\n{suffix}\n"
    )


def select_tilelang_requests(
    space: dict,
    family: str,
    batches: Iterable[int],
    candidate_ids: Iterable[str] = (),
) -> list[dict]:
    """Select every exact (batch, tactic) request from a CUDA tactic space."""
    validate_family(family)
    if family not in TILELANG_FAMILIES:
        raise ValueError(f"{family} does not use the TileLang fat preparer")
    if (
        space.get("schema") != 1 or
        space.get("kind") != "cuda-tactic-search-space" or
        space.get("architecture") != "sm120"
    ):
        raise ValueError("fat scan requires an SM120 CUDA tactic search space")
    requested_batches = sorted(set(int(value) for value in batches))
    if not requested_batches or requested_batches[0] < 1:
        raise ValueError("batch set must contain positive integers")
    allowed_ids = {value for value in candidate_ids if value}
    batch_spaces = {int(item["batch"]): item for item in space.get("batches", [])}
    missing = sorted(set(requested_batches) - set(batch_spaces))
    if missing:
        raise ValueError(f"batches outside materialized space: {missing}")

    result: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for batch in requested_batches:
        for candidate in batch_spaces[batch].get(family, []):
            candidate_id = candidate["id"]
            if allowed_ids and candidate_id not in allowed_ids:
                continue
            if candidate.get("implementation", "tilelang") != "tilelang":
                continue
            key = (batch, candidate_id)
            if key in seen:
                raise ValueError(f"duplicate exact fat-scan request: B{batch}/{candidate_id}")
            seen.add(key)
            token = symbol_token(family, batch, candidate_id)
            result.append({
                "batch": batch,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "symbol_token": token,
                "launch_symbol": launch_symbol(family, token),
            })

    if allowed_ids:
        known_ids = {
            candidate["id"]
            for batch in requested_batches
            for candidate in batch_spaces[batch].get(family, [])
        }
        missing_ids = sorted(allowed_ids - known_ids)
        if missing_ids:
            raise ValueError(
                "requested IDs are absent from the selected batches: "
                f"{missing_ids}"
            )
    if not result:
        raise ValueError("fat scan selected no TileLang candidates")
    return result


def render_registry(family: str, requests: Iterable[dict]) -> str:
    """Render the one registry TU for a generated family bundle."""
    validate_family(family)
    values = list(requests)
    if not values:
        raise ValueError("cannot render an empty fat registry")
    keys: set[tuple[int, str]] = set()
    tokens: set[str] = set()
    for item in values:
        batch = int(item["batch"])
        candidate_id = str(item["candidate_id"])
        token = str(item["symbol_token"])
        expected_token = symbol_token(family, batch, candidate_id)
        if token != expected_token:
            raise ValueError(
                f"unstable symbol token for B{batch}/{candidate_id}: "
                f"expected {expected_token}, got {token}"
            )
        expected_launch = launch_symbol(family, token)
        if item.get("launch_symbol") != expected_launch:
            raise ValueError(f"invalid launch symbol for B{batch}/{candidate_id}")
        key = (batch, candidate_id)
        if key in keys:
            raise ValueError(f"duplicate exact fat registry key: B{batch}/{candidate_id}")
        if token in tokens:
            raise ValueError(f"duplicate fat registry symbol token: {token}")
        keys.add(key)
        tokens.add(token)

    abi = _FAMILY_ABI[family]
    declarations = "\n".join(
        f'extern "C" cudaError_t {item["launch_symbol"]}({abi["launch_args"]});'
        for item in values
    )
    entries = []
    for item in values:
        quoted_id = json.dumps(item["candidate_id"], ensure_ascii=True)
        launch = item["launch_symbol"]
        if family == "dual_ffn":
            paired = str(
                item.get("candidate", {}).get("paired_weights", False)
            ).lower()
            entry = (
                f'{{{item["batch"]}, {quoted_id}, '
                f'{paired}, {launch}}}'
            )
        elif family == "wide_qkv":
            packed = str(item.get("candidate", {}).get("output") == "packed").lower()
            entry = (
                f'{{{item["batch"]}, {quoted_id}, '
                f'{packed}, {launch}}}'
            )
        elif family == "qkv_rope":
            entry = f'{{{item["batch"]}, {quoted_id}, {launch}}}'
        elif family in ("linear2", "outproj"):
            input_channels = 1152 if family == "linear2" else 384
            entry = (
                f'{{{item["batch"]}, {input_channels}, {quoted_id}, '
                f'{launch}}}'
            )
        else:
            entry = f'{{{item["batch"]}, {quoted_id}, {launch}}}'
        entries.append(f"  {entry}")
    entry_text = ",\n".join(entries)
    return f'''// Generated by the SM120 fat-bundle preparer. Do not edit.
#include "cudabackend_sm120_kernels.h"

#include <cstddef>

{declarations}

namespace Sm120Backend {{
namespace {{

const {abi["tactic_type"]} fatTactics[] = {{
{entry_text}
}};

}} // namespace

const {abi["tactic_type"]}* {abi["getter"]}(std::size_t& count) {{
  count = sizeof(fatTactics) / sizeof(fatTactics[0]);
  return fatTactics;
}}

}} // namespace Sm120Backend
'''


def write_registry(path: pathlib.Path, family: str, requests: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_registry(family, requests), encoding="ascii")
