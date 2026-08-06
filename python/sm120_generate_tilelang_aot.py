#!/usr/bin/env python3
"""Generate an exact-batch SM120 TileLang candidate for whole-graph search.

The candidate must already exist in a schema-2 space emitted by
``sm120_tactic_search.py space``.  This prevents an NCU-inspired one-off from
silently escaping the recorded search region.  The output is a standalone
CUDA translation unit implementing one of the three search-slot ABIs consumed
by the SM120 AOT registry.

This generator performs no local two-stream proxy benchmark.  Optional timing
is single-stream only; acceptance is always natural whole-graph S2 throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import time

import tilelang
import tilelang.language as T
import torch

from sm120_fat_scan import (
    isolate_tilelang_debug_symbols,
    launch_symbol as fat_launch_symbol,
    validate_symbol_token,
)


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def fused_ffn(
    input_tensor,
    linear_weights,
    gate_weights,
    output,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    min_blocks: int,
):
    m, n, k = T.const("m, n, k")
    dtype = T.float16
    input_tensor: T.Tensor((m, k), dtype)
    linear_weights: T.Tensor((k, n), dtype)
    gate_weights: T.Tensor((k, n), dtype)
    output: T.Tensor((m, n), dtype)

    with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
        T.annotate_min_blocks_per_sm(min_blocks)
        input_shared = T.alloc_shared((block_m, block_k), dtype)
        linear_shared = T.alloc_shared((block_k, block_n), dtype)
        gate_shared = T.alloc_shared((block_k, block_n), dtype)
        linear_local = T.alloc_fragment((block_m, block_n), dtype)
        gate_local = T.alloc_fragment((block_m, block_n), dtype)
        T.use_swizzle(10)
        T.clear(linear_local)
        T.clear(gate_local)
        for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
            T.copy(input_tensor[by * block_m, ko * block_k], input_shared)
            T.copy(linear_weights[ko * block_k, bx * block_n], linear_shared)
            T.copy(gate_weights[ko * block_k, bx * block_n], gate_shared)
            T.gemm(input_shared, linear_shared, linear_local)
            T.gemm(input_shared, gate_shared, gate_local)
        for i, j in T.Parallel(block_m, block_n):
            linear_local[i, j] = (
                linear_local[i, j]
                / (1.0 + T.exp(-linear_local[i, j]))
                * gate_local[i, j]
            )
        T.copy(linear_local, output[by * block_m, bx * block_n])


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def wide_qkv(
    input_tensor,
    weights,
    output,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    min_blocks: int,
):
    m, n, k, q_dim = T.const("m, n, k, q_dim")
    dtype = T.float16
    input_tensor: T.Tensor((m, k), dtype)
    weights: T.Tensor((k, n), dtype)
    output: T.Tensor((3, m, q_dim), dtype)

    with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
        T.annotate_min_blocks_per_sm(min_blocks)
        input_shared = T.alloc_shared((block_m, block_k), dtype)
        weight_shared = T.alloc_shared((block_k, block_n), dtype)
        output_local = T.alloc_fragment((block_m, block_n), dtype)
        T.use_swizzle(10)
        T.clear(output_local)
        for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
            T.copy(input_tensor[by * block_m, ko * block_k], input_shared)
            T.copy(weights[ko * block_k, bx * block_n], weight_shared)
            T.gemm(input_shared, weight_shared, output_local)
        for i, j in T.Parallel(block_m, block_n):
            if by * block_m + i < m:
                output[
                    (bx * block_n + j) // q_dim,
                    by * block_m + i,
                    (bx * block_n + j) % q_dim,
                ] = output_local[i, j]


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def residual_gemm(
    input_tensor,
    weights,
    residual,
    output,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    threads: int,
    min_blocks: int,
):
    m, n, k = T.const("m, n, k")
    dtype = T.float16
    input_tensor: T.Tensor((m, k), dtype)
    weights: T.Tensor((k, n), dtype)
    residual: T.Tensor((m, n), dtype)
    output: T.Tensor((m, n), dtype)

    with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (bx, by):
        T.annotate_min_blocks_per_sm(min_blocks)
        input_shared = T.alloc_shared((block_m, block_k), dtype)
        weight_shared = T.alloc_shared((block_k, block_n), dtype)
        output_local = T.alloc_fragment((block_m, block_n), dtype)
        T.use_swizzle(10)
        T.clear(output_local)
        for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
            T.copy(input_tensor[by * block_m, ko * block_k], input_shared)
            T.copy(weights[ko * block_k, bx * block_n], weight_shared)
            T.gemm(input_shared, weight_shared, output_local)
        for i, j in T.Parallel(block_m, block_n):
            if by * block_m + i < m:
                output_local[i, j] += residual[by * block_m + i, bx * block_n + j]
        T.copy(output_local, output[by * block_m, bx * block_n])


def matching_brace(lines: list[str], start: int) -> int:
    depth = 0
    for index in range(start, len(lines)):
        depth += lines[index].count("{") - lines[index].count("}")
        if depth == 0:
            return index
    raise RuntimeError(f"unmatched brace at generated line {start + 1}")


def enclosing_scope(lines: list[str], declaration: str) -> tuple[int, int]:
    declaration_index = next(i for i, line in enumerate(lines) if declaration in line)
    start = declaration_index - 1
    if lines[start].strip() != "{":
        raise RuntimeError(f"expected standalone scope before {declaration}")
    return start, matching_brace(lines, start)


def merge_a_pair(
    lines: list[str], linear_name: str, gate_name: str,
) -> None:
    linear_suffix = linear_name.removeprefix("A_local")
    gate_suffix = gate_name.removeprefix("A_local")
    gate_start, gate_end = enclosing_scope(lines, f"half_t {gate_name}[")
    gate = lines[gate_start : gate_end + 1]
    ki_name = "ki" + gate_suffix
    ki_start = next(i for i, line in enumerate(gate) if f"for (int {ki_name}" in line)
    ki_end = matching_brace(gate, ki_start)
    b_loop = next(i for i, line in enumerate(gate) if "ptx_ldmatrix_x4_trans" in line)
    body_start = b_loop - 2
    if gate[body_start].strip() != "#pragma unroll":
        raise RuntimeError(f"failed to locate gate B loop for {gate_name}")
    gate_body = gate[body_start:ki_end]
    replacements = {
        gate_name: linear_name,
        "B_local" + gate_suffix: "B_local" + linear_suffix,
        ki_name: "ki" + linear_suffix,
    }
    gate_body = [
        next_line if old == new else next_line.replace(old, new)
        for next_line in gate_body
        for old, new in [next(iter(replacements.items()))]
    ]
    # Apply the remaining replacements separately to avoid substring suffix
    # interactions (for example ki_1 versus ki).
    for old, new in list(replacements.items())[1:]:
        gate_body = [line.replace(old, new) for line in gate_body]

    del lines[gate_start : gate_end + 1]
    linear_start, linear_end = enclosing_scope(lines, f"half_t {linear_name}[")
    linear = lines[linear_start : linear_end + 1]
    linear_ki = "ki" + linear_suffix
    linear_ki_start = next(i for i, line in enumerate(linear) if f"for (int {linear_ki}" in line)
    linear_ki_end = matching_brace(linear, linear_ki_start)
    lines[linear_start : linear_end + 1] = (
        linear[:linear_ki_end] + gate_body + linear[linear_ki_end:]
    )


def reuse_ffn_a_fragments(source: str) -> str:
    names = re.findall(r"half_t (A_local(?:_\d+)?)\[", source)
    indexed = {}
    for name in names:
        suffix = name.removeprefix("A_local")
        indexed[0 if not suffix else int(suffix[1:])] = name
    if len(indexed) < 2 or len(indexed) % 2:
        raise RuntimeError(f"unexpected generated A fragment set: {indexed}")
    expected = list(range(len(indexed)))
    if sorted(indexed) != expected:
        raise RuntimeError(f"non-contiguous generated A fragment set: {indexed}")
    lines = source.splitlines(keepends=True)
    for even in reversed(range(0, len(indexed), 2)):
        merge_a_pair(lines, indexed[even], indexed[even + 1])
    candidate_source = "".join(lines)
    remaining = re.findall(r"half_t (A_local(?:_\d+)?)\[", candidate_source)
    if len(remaining) * 2 != len(names):
        raise RuntimeError("A-fragment reuse transform did not halve A declarations")
    return candidate_source


def find_candidate(space: dict, batch: int, family: str, candidate_id: str) -> dict:
    if space.get("schema") != 2:
        raise ValueError("--space must be schema 2")
    batch_space = next((item for item in space["batches"] if item["batch"] == batch), None)
    if batch_space is None:
        raise ValueError(f"B{batch} is outside the materialized space")
    item = next((value for value in batch_space.get(family, []) if value["id"] == candidate_id), None)
    if item is None:
        raise ValueError(
            f"{family}/{candidate_id} is outside B{batch}; register it before generation"
        )
    return item


def event_bench(callable_kernel, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        callable_kernel()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            callable_kernel()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return samples


def common_tile(candidate_value: dict) -> dict:
    return {
        "block_m": int(candidate_value["m"]),
        "block_n": int(candidate_value["n"]),
        "block_k": int(candidate_value["k"]),
        "num_stages": int(candidate_value["stages"]),
        "threads": int(candidate_value.get("threads", 128)),
        "min_blocks": int(candidate_value.get("min_blocks", 1)),
    }


def append_wrapper(
    source: str, family: str, candidate_value: dict, batch: int,
    dynamic_smem: int,
    symbol_token: str | None = None,
) -> str:
    candidate_id = candidate_value["id"]
    m = batch * 361
    threads = int(candidate_value.get("threads", 128))
    block_m = int(candidate_value["m"])
    block_n = int(candidate_value["n"])
    grid_x = (1152 if family in ("ffn", "qkv") else 384) // block_n
    if family == "qkv" and 1152 % block_n:
        grid_x += 1
    if family == "ffn" and 1152 % block_n:
        grid_x += 1
    if family == "linear2" and 384 % block_n:
        grid_x += 1
    grid_y = (m + block_m - 1) // block_m
    kernel_old = {
        "ffn": "fused_ffn_kernel",
        "qkv": "wide_qkv_kernel",
        "linear2": "residual_gemm_kernel",
    }[family]
    if symbol_token is not None:
        validate_symbol_token(symbol_token)
    kernel_new = f"sm120_search_{family}_kernel"
    if symbol_token is not None:
        kernel_new += f"_{symbol_token}"
    source = source.replace(kernel_old, kernel_new)
    attribute = f"""
  static const cudaError_t attributeStatus = cudaFuncSetAttribute(
    {kernel_new}, cudaFuncAttributeMaxDynamicSharedMemorySize, {dynamic_smem});
  if(attributeStatus != cudaSuccess)
    return attributeStatus;
""" if dynamic_smem > 49152 else ""
    descriptors = ""
    if symbol_token is None:
        descriptors = f'''\nextern "C" int sm120_search_{family}_batch() {{ return {batch}; }}
extern "C" const char* sm120_search_{family}_id() {{ return "{candidate_id}"; }}
'''
    launch_name = (
        f"sm120_search_{family}_launch"
        if symbol_token is None
        else fat_launch_symbol(family, symbol_token)
    )
    if family == "ffn":
        launcher = f'''extern "C" cudaError_t {launch_name}(
  const half* input, const half* linear_weights, const half* gate_weights,
  half* output, cudaStream_t stream) {{
{attribute}  {kernel_new}<<<dim3({grid_x}, {grid_y}, 1), {threads}, {dynamic_smem}, stream>>>(
    reinterpret_cast<const half_t*>(gate_weights),
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<const half_t*>(linear_weights),
    reinterpret_cast<half_t*>(output));
  return cudaPeekAtLastError();
}}
'''
    elif family == "qkv":
        launcher = f'''extern "C" cudaError_t {launch_name}(
  const half* input, const half* weights, half* output,
  cudaStream_t stream) {{
{attribute}  {kernel_new}<<<dim3({grid_x}, {grid_y}, 1), {threads}, {dynamic_smem}, stream>>>(
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<half_t*>(output),
    reinterpret_cast<const half_t*>(weights));
  return cudaPeekAtLastError();
}}
'''
    else:
        launcher = f'''extern "C" cudaError_t {launch_name}(
  const half* input, const half* weights, half* residual,
  cudaStream_t stream) {{
{attribute}  {kernel_new}<<<dim3({grid_x}, {grid_y}, 1), {threads}, {dynamic_smem}, stream>>>(
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<half_t*>(residual),
    reinterpret_cast<const half_t*>(residual),
    reinterpret_cast<const half_t*>(weights));
  return cudaPeekAtLastError();
}}
'''
    return source + descriptors + launcher


def restrict_generated_kernel_to_sm120(source: str, kernel_name: str) -> str:
    """Keep an exact SM120 search TU buildable in KataGo's fat binary."""
    lines = source.splitlines(keepends=True)
    definition = next(
        index for index, line in enumerate(lines)
        if kernel_name in line and "__global__" in line and line.rstrip().endswith("{")
    )
    end = matching_brace(lines, definition)
    lines.insert(definition + 1, "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200\n")
    lines.insert(end + 1, "#endif\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--family", choices=("ffn", "qkv", "linear2"), required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-path",
        help="optional stable active-slot path; avoids CMake reconfiguration between candidates",
    )
    parser.add_argument(
        "--fat-symbol-token",
        help=(
            "emit a uniquely named fat-scan kernel/launcher instead of the "
            "legacy single-slot descriptor ABI"
        ),
    )
    parser.add_argument("--s1-warmup", type=int, default=20)
    parser.add_argument("--s1-iterations", type=int, default=100)
    args = parser.parse_args()

    space = json.loads(pathlib.Path(args.space).read_text())
    candidate_value = find_candidate(space, args.batch, args.family, args.candidate_id)
    if candidate_value.get("implementation", "tilelang") != "tilelang":
        raise ValueError("this generator only handles TileLang candidates")
    if args.family == "ffn" and candidate_value.get("swiglu") != "exp":
        raise ValueError("historical tanh-half2 FFN uses its reproducible legacy generator")

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260806)
    m = args.batch * 361
    tile = common_tile(candidate_value)
    started = time.time()
    correctness = {}

    if args.family == "ffn":
        n, k = 1152, 384
        kernel = fused_ffn.compile(m=m, n=n, k=k, **tile)
        linear = (torch.randn(k, n, device="cuda") * 0.05).half()
        gate = (torch.randn(k, n, device="cuda") * 0.05).half()
        input_tensor = (torch.randn(m, k, device="cuda") * 0.15).half()
        output = torch.empty(m, n, device="cuda", dtype=torch.float16)
        call = lambda: kernel(input_tensor, linear, gate, output)
        call()
        reference = torch.nn.functional.silu(input_tensor @ linear) * (input_tensor @ gate)
    elif args.family == "qkv":
        n, k, q_dim = 1152, 384, 384
        kernel = wide_qkv.compile(m=m, n=n, k=k, q_dim=q_dim, **tile)
        weights = (torch.randn(k, n, device="cuda") * 0.05).half()
        input_tensor = (torch.randn(m, k, device="cuda") * 0.15).half()
        output = torch.empty(3, m, q_dim, device="cuda", dtype=torch.float16)
        call = lambda: kernel(input_tensor, weights, output)
        call()
        reference = (input_tensor @ weights).reshape(m, 3, q_dim).permute(1, 0, 2)
    else:
        n, k = 384, 1152
        kernel = residual_gemm.compile(m=m, n=n, k=k, **tile)
        weights = (torch.randn(k, n, device="cuda") * 0.05).half()
        input_tensor = (torch.randn(m, k, device="cuda") * 0.15).half()
        output = (torch.randn(m, n, device="cuda") * 0.15).half()
        residual = output.clone()
        call = lambda: kernel(input_tensor, weights, residual, output)
        call()
        reference = torch.addmm(residual, input_tensor, weights)

    torch.cuda.synchronize()
    diff = (output.float() - reference.float()).abs()
    correctness = {
        "max_abs": float(diff.max().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    s1_samples = event_bench(call, args.s1_warmup, args.s1_iterations)

    source = kernel.get_kernel_source()
    if args.family == "ffn" and candidate_value.get("a_fragment_reuse"):
        source = reuse_ffn_a_fragments(source)
    source = restrict_generated_kernel_to_sm120(
        source,
        {
            "ffn": "fused_ffn_kernel",
            "qkv": "wide_qkv_kernel",
            "linear2": "residual_gemm_kernel",
        }[args.family],
    )
    block_m = int(candidate_value["m"])
    block_n = int(candidate_value["n"])
    block_k = int(candidate_value["k"])
    stages = int(candidate_value["stages"])
    weight_count = 2 if args.family == "ffn" else 1
    dynamic_smem = (block_m * block_k + weight_count * block_k * block_n) * 2 * stages
    output_columns = 1152 if args.family in ("ffn", "qkv") else 384
    grid_x = (output_columns + block_n - 1) // block_n
    grid_y = (m + block_m - 1) // block_m
    source = append_wrapper(
        source, args.family, candidate_value, args.batch, dynamic_smem,
        args.fat_symbol_token,
    )
    debug_token = args.fat_symbol_token or f"single_{args.family}"
    source = isolate_tilelang_debug_symbols(source, debug_token)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = (
        pathlib.Path(args.source_path).resolve()
        if args.source_path
        else output_dir / f"{args.family}-{args.candidate_id}.cu"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{args.family}-{args.candidate_id}.json"
    source_path.write_text(source, encoding="ascii")
    metadata = {
        "schema": 1,
        "candidate": candidate_value,
        "family": args.family,
        "batch": args.batch,
        "tokens": m,
        "fixed_board": [19, 19],
        "source": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
        "dynamic_smem_bytes": dynamic_smem,
        "launch": {
            "grid": [grid_x, grid_y, 1],
            "block": [int(candidate_value.get("threads", 128)), 1, 1],
            "cta_count": grid_x * grid_y,
        },
        "fat_symbol_token": args.fat_symbol_token,
        "launch_symbol": (
            fat_launch_symbol(args.family, args.fat_symbol_token)
            if args.fat_symbol_token is not None
            else f"sm120_search_{args.family}_launch"
        ),
        "correctness_against_torch": correctness,
        "s1_us_samples": s1_samples,
        "s1_us_median": statistics.median(s1_samples),
        "generator_seconds": time.time() - started,
        "acceptance_metric": "natural whole-graph S2 total throughput",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
