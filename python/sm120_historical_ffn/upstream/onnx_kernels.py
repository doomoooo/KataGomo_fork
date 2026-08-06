#!/usr/bin/env python3
"""Correctness-first TileLang kernels used by the KataGo ONNX prototype.

This module deliberately keeps ONNX operators separate.  It is a numerical
coverage scaffold, not the optimized backend: every arithmetic operation is
implemented by TileLang and PyTorch is only used to own CUDA allocations.
"""

import math
from functools import lru_cache
from typing import Sequence

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)
from tilelang.cuda.intrinsics.layout import mma_store_index_map
from tilelang.layout import make_swizzled_layout


TARGET = {"kind": "cuda", "arch": "sm_120"}
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
}
FLASH_ATTENTION_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}
THREADS = 256
FP16_TANH_INTRINSIC_SOURCE = r"""
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
__device__ __forceinline__ cutlass::half_t tilelang_htanh_approx(
    cutlass::half_t value) {
  return cutlass::half_t(htanh_approx(value.to_half()));
}
__device__ __forceinline__ unsigned tilelang_h2tanh_approx(unsigned bits) {
  union PackedHalf2 {
    unsigned bits;
    __half2 value;
  } input, output;
  input.bits = bits;
  output.value = h2tanh_approx(input.value);
  return output.bits;
}
"""


def _value_dtype(dtype: str):
    if dtype == "float16":
        return T.float16
    if dtype == "float32":
        return T.float32
    raise ValueError(f"Unsupported value dtype: {dtype}")


def _prod(shape: Sequence[int]) -> int:
    return math.prod(shape)


def _strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result = []
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _flat_broadcast_index(index, input_shape: tuple[int, ...], output_shape: tuple[int, ...]):
    padded = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    output_strides = _strides(output_shape)
    input_strides = _strides(padded)
    result = 0
    for dim, out_stride, in_stride in zip(padded, output_strides, input_strides):
        if dim != 1:
            result += ((index // out_stride) % dim) * in_stride
    return result


def _transpose_input_index(
    output_index,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    permutation: tuple[int, ...],
):
    input_strides = _strides(input_shape)
    output_strides = _strides(output_shape)
    result = 0
    for output_axis in range(len(output_shape)):
        coordinate = (
            output_index // output_strides[output_axis]
        ) % output_shape[output_axis]
        result += coordinate * input_strides[permutation[output_axis]]
    return result


@lru_cache(maxsize=None)
def unary(op: str, shape: tuple[int, ...], dtype: str = "float32"):
    numel = _prod(shape)
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((numel,), value_dtype),
            output: T.Tensor((numel,), value_dtype),
        ):
            with T.Kernel(T.ceildiv(numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    index = block * THREADS + thread
                    if index < numel:
                        if op == "Sqrt":
                            output[index] = T.sqrt(source[index])
                        elif op == "Sigmoid":
                            value_fp32 = T.cast(source[index], T.float32)
                            output[index] = T.cast(
                                1.0 / (1.0 + T.exp(-value_fp32)), value_dtype
                            )
                        else:
                            raise ValueError(f"Unsupported unary operator: {op}")

        return main

    return build()


@lru_cache(maxsize=None)
def binary(
    op: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    dtype: str = "float32",
):
    left_numel = _prod(left_shape)
    right_numel = _prod(right_shape)
    output_numel = _prod(output_shape)
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            left: T.Tensor((left_numel,), value_dtype),
            right: T.Tensor((right_numel,), value_dtype),
            output: T.Tensor((output_numel,), value_dtype),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    index = block * THREADS + thread
                    if index < output_numel:
                        left_index = _flat_broadcast_index(index, left_shape, output_shape)
                        right_index = _flat_broadcast_index(index, right_shape, output_shape)
                        if op == "Add":
                            output[index] = left[left_index] + right[right_index]
                        elif op == "Mul":
                            output[index] = left[left_index] * right[right_index]
                        elif op == "Div":
                            output[index] = left[left_index] / right[right_index]
                        else:
                            raise ValueError(f"Unsupported binary operator: {op}")

        return main

    return build()


@lru_cache(maxsize=None)
def transpose(
    input_shape: tuple[int, ...],
    permutation: tuple[int, ...],
    dtype: str = "float32",
):
    output_shape = tuple(input_shape[axis] for axis in permutation)
    numel = _prod(input_shape)
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((numel,), value_dtype),
            output: T.Tensor((numel,), value_dtype),
        ):
            with T.Kernel(T.ceildiv(numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < numel:
                        input_index = _transpose_input_index(
                            output_index, input_shape, output_shape, permutation
                        )
                        output[output_index] = source[input_index]

        return main

    return build()


def _gather_indices(
    output_index,
    indices,
    input_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axis: int,
):
    input_strides = _strides(input_shape)
    output_strides = _strides(output_shape)
    indices_strides = _strides(indices_shape)
    input_index = 0
    indices_index = 0
    output_axis = 0
    for input_axis in range(len(input_shape)):
        if input_axis == axis:
            for indices_axis in range(len(indices_shape)):
                coordinate = (
                    output_index // output_strides[output_axis]
                ) % output_shape[output_axis]
                indices_index += coordinate * indices_strides[indices_axis]
                output_axis += 1
            gathered = indices[indices_index]
            gathered = T.if_then_else(
                gathered < 0, gathered + input_shape[input_axis], gathered
            )
            input_index += gathered * input_strides[input_axis]
        else:
            coordinate = (
                output_index // output_strides[output_axis]
            ) % output_shape[output_axis]
            input_index += coordinate * input_strides[input_axis]
            output_axis += 1
    return input_index


@lru_cache(maxsize=None)
def gather(
    input_shape: tuple[int, ...],
    indices_shape: tuple[int, ...],
    axis: int,
    dtype: str = "float32",
):
    axis %= len(input_shape)
    output_shape = input_shape[:axis] + indices_shape + input_shape[axis + 1 :]
    input_numel = _prod(input_shape)
    indices_numel = _prod(indices_shape)
    output_numel = _prod(output_shape)
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((input_numel,), value_dtype),
            indices: T.Tensor((indices_numel,), T.int64),
            output: T.Tensor((output_numel,), value_dtype),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < output_numel:
                        input_index = _gather_indices(
                            output_index,
                            indices,
                            input_shape,
                            indices_shape,
                            output_shape,
                            axis,
                        )
                        output[output_index] = source[input_index]

        return main

    return build()


def _reduction_input_index(
    output_index,
    reduction_index,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
):
    input_strides = _strides(input_shape)
    output_strides = _strides(output_shape)
    reduction_shape = tuple(input_shape[axis] for axis in axes)
    reduction_strides = _strides(reduction_shape)
    result = 0
    output_axis = 0
    reduction_axis = 0
    for input_axis, (dim, input_stride) in enumerate(zip(input_shape, input_strides)):
        if input_axis in axes:
            coordinate = (
                reduction_index // reduction_strides[reduction_axis]
            ) % reduction_shape[reduction_axis]
            reduction_axis += 1
            if keepdims:
                output_axis += 1
        else:
            coordinate = (
                output_index // output_strides[output_axis]
            ) % output_shape[output_axis]
            output_axis += 1
        result += coordinate * input_stride
    return result


@lru_cache(maxsize=None)
def reduce(
    op: str,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
    dtype: str = "float32",
):
    axes = tuple(sorted(axis % len(input_shape) for axis in axes))
    if keepdims:
        output_shape = tuple(1 if axis in axes else dim for axis, dim in enumerate(input_shape))
    else:
        output_shape = tuple(dim for axis, dim in enumerate(input_shape) if axis not in axes)
    if not output_shape:
        output_shape = (1,)
    input_numel = _prod(input_shape)
    output_numel = _prod(output_shape)
    reduction_numel = _prod(tuple(input_shape[axis] for axis in axes))
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((input_numel,), value_dtype),
            output: T.Tensor((output_numel,), value_dtype),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < output_numel:
                        if op == "ReduceMean":
                            accumulator = T.alloc_var(T.float32, init=0)
                            for reduction_index in T.serial(reduction_numel):
                                input_index = _reduction_input_index(
                                    output_index,
                                    reduction_index,
                                    input_shape,
                                    output_shape,
                                    axes,
                                    keepdims,
                                )
                                accumulator += source[input_index]
                            output[output_index] = accumulator / reduction_numel
                        elif op == "ReduceMax":
                            accumulator = T.alloc_var(
                                T.float32, init=-T.infinity(T.float32)
                            )
                            for reduction_index in T.serial(reduction_numel):
                                input_index = _reduction_input_index(
                                    output_index,
                                    reduction_index,
                                    input_shape,
                                    output_shape,
                                    axes,
                                    keepdims,
                                )
                                accumulator = T.max(accumulator, source[input_index])
                            output[output_index] = accumulator
                        else:
                            raise ValueError(f"Unsupported reduction operator: {op}")

        return main

    return build(), output_shape


@lru_cache(maxsize=None)
def softmax(rows: int, columns: int, dtype: str = "float32"):
    block_columns = 1 << (columns - 1).bit_length()
    scale_log2e = 1.4426950408889634
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((rows, columns), value_dtype),
            output: T.Tensor((rows, columns), value_dtype),
        ):
            with T.Kernel(rows, threads=128) as row:
                values = T.alloc_fragment((1, block_columns), T.float32)
                exponentials = T.alloc_fragment((1, block_columns), T.float32)
                maximum = T.alloc_fragment((1,), T.float32)
                total = T.alloc_fragment((1,), T.float32)
                for column in T.Parallel(block_columns):
                    values[0, column] = T.if_then_else(
                        column < columns,
                        source[row, column],
                        -T.infinity(T.float32),
                    )
                T.reduce_max(values, maximum, dim=1, clear=True)
                for column in T.Parallel(block_columns):
                    exponentials[0, column] = T.if_then_else(
                        column < columns,
                        T.exp2((values[0, column] - maximum[0]) * scale_log2e),
                        0,
                    )
                T.reduce_sum(exponentials, total, dim=1, clear=True)
                for column in T.Parallel(block_columns):
                    if column < columns:
                        output[row, column] = exponentials[0, column] / total[0]

        return main

    return build()


@lru_cache(maxsize=None)
def rmsnorm(rows: int, columns: int, epsilon: float):
    """Row-wise FP16 RMSNorm with FP32 square reduction and normalization."""
    block_columns = 1 << (columns - 1).bit_length()

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((rows, columns), T.float16),
            weight: T.Tensor((columns,), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(rows, threads=128) as row:
                squares = T.alloc_fragment((1, block_columns), T.float32)
                total = T.alloc_fragment((1,), T.float32)
                for column in T.Parallel(block_columns):
                    value = T.if_then_else(
                        column < columns,
                        T.cast(source[row, column], T.float32),
                        0,
                    )
                    squares[0, column] = value * value
                T.reduce_sum(squares, total, dim=1, clear=True)
                denominator = T.sqrt(total[0] / columns + epsilon)
                for column in T.Parallel(block_columns):
                    if column < columns:
                        output[row, column] = (
                            T.cast(source[row, column], T.float32)
                            / denominator
                            * T.cast(weight[column], T.float32)
                        )

        return main

    return build()


@lru_cache(maxsize=None)
def rmsnorm_denominator(rows: int, columns: int, epsilon: float):
    """Compute one FP32 RMS denominator per row for projection-side fusion."""
    block_columns = 1 << (columns - 1).bit_length()

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            source: T.Tensor((rows, columns), T.float16),
            denominator: T.Tensor((rows,), T.float32),
        ):
            with T.Kernel(rows, threads=128) as row:
                squares = T.alloc_fragment((1, block_columns), T.float32)
                total = T.alloc_fragment((1,), T.float32)
                for column in T.Parallel(block_columns):
                    value = T.if_then_else(
                        column < columns,
                        T.cast(source[row, column], T.float32),
                        0,
                    )
                    squares[0, column] = value * value
                T.reduce_sum(squares, total, dim=1, clear=True)
                denominator[row] = T.sqrt(total[0] / columns + epsilon)

        return main

    return build()


@lru_cache(maxsize=None)
def matmul_2d(
    rows: int,
    columns: int,
    inner: int,
    dtype: str = "float32",
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 1,
    threads: int = 128,
    enable_rasterization: bool = False,
):
    value_dtype = _value_dtype(dtype)
    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            left: T.Tensor((rows, inner), value_dtype),
            right: T.Tensor((inner, columns), value_dtype),
            output: T.Tensor((rows, columns), value_dtype),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=threads,
            ) as (block_x, block_y):
                left_shared = T.alloc_shared((block_m, block_k), value_dtype)
                right_shared = T.alloc_shared((block_k, block_n), value_dtype)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.disable_warp_group_reg_alloc()
                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.clear(accumulator)
                for tile_k in T.Pipelined(
                    T.ceildiv(inner, block_k), num_stages=num_stages
                ):
                    T.copy(left[block_y * block_m, tile_k * block_k], left_shared)
                    T.copy(right[tile_k * block_k, block_x * block_n], right_shared)
                    T.gemm(left_shared, right_shared, accumulator)
                T.copy(accumulator, output[block_y * block_m, block_x * block_n])

        return main

    return build()


@lru_cache(maxsize=None)
def matmul_residual_2d(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
):
    """FP32-accum GEMM with the FP16 residual add fused into its epilogue."""

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            weight: T.Tensor((inner, columns), T.float16),
            residual: T.Tensor((rows, columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (block_x, block_y):
                T.disable_warp_group_reg_alloc()
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k], data_shared
                    )
                    T.copy(
                        weight[tile_k * block_k, block_x * block_n],
                        weight_shared,
                    )
                    T.gemm(data_shared, weight_shared, accumulator)
                for local_row, local_column in T.Parallel(block_m, block_n):
                    row = block_y * block_m + local_row
                    column = block_x * block_n + local_column
                    if row < rows and column < columns:
                        matmul_rounded = T.cast(
                            accumulator[local_row, local_column], T.float16
                        )
                        sum_rounded = T.cast(
                            T.cast(matmul_rounded, T.float32)
                            + T.cast(residual[row, column], T.float32),
                            T.float16,
                        )
                        accumulator[local_row, local_column] = T.cast(
                            sum_rounded, T.float32
                        )
                T.copy(
                    accumulator,
                    output[block_y * block_m, block_x * block_n],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def linear2_residual_intrinsics(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 2,
    enable_rasterization: bool = False,
    block_row_warps: int = 2,
    block_col_warps: int = 2,
    min_blocks_per_sm: int = 2,
    direct_epilogue: bool = False,
    enable_l2_prefetch: bool = False,
    enable_warp_specialization: bool = False,
):
    """FP16 Tensor Core GEMM with an in-place FP16 residual epilogue."""

    if inner % block_k or columns % block_n:
        raise ValueError("linear2 requires exact K and N tiles")
    if block_m % block_row_warps or block_n % block_col_warps:
        raise ValueError("block tile must divide evenly across its warp layout")

    warp_m = block_m // block_row_warps
    warp_n = block_n // block_col_warps
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.float16,
        b_dtype=T.float16,
        accum_dtype=T.float16,
        a_transposed=False,
        b_transposed=False,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_m,
        warp_col_tiles=warp_n,
        chunk=block_k,
    )
    threads = 32 * block_row_warps * block_col_warps
    a_local_size = emitter.warp_rows * emitter.local_size_a
    b_local_size = emitter.warp_cols * emitter.local_size_b
    c_local_size = (
        emitter.warp_rows * emitter.warp_cols * emitter.local_size_out
    )
    pass_configs = dict(FLASH_ATTENTION_PASS_CONFIGS)
    if not enable_warp_specialization:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True
    compile_flags = (
        ["-DTL_ENABLE_L2_PREFETCH=1"] if enable_l2_prefetch else None
    )
    kernel_target = {
        "kind": "cuda",
        "arch": "sm_120a" if enable_warp_specialization else "sm_120",
    }

    @tilelang.jit(
        out_idx=[],
        target=kernel_target,
        execution_backend="cython",
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            weight: T.Tensor((inner, columns), T.float16),
            output_residual: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                columns // block_n,
                T.ceildiv(rows, block_m),
                threads=threads,
            ) as (block_x, block_y):
                T.annotate_min_blocks_per_sm(min_blocks_per_sm)
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                if not direct_epilogue:
                    output_shared = T.alloc_shared(
                        (block_m, block_n), T.float16
                    )
                data_local = T.alloc_local((a_local_size,), T.float16)
                weight_local = T.alloc_local((b_local_size,), T.float16)
                accumulator = T.alloc_local((c_local_size,), T.float16)

                if direct_epilogue:
                    T.annotate_layout(
                        {
                            data_shared: make_swizzled_layout(data_shared),
                            weight_shared: make_swizzled_layout(weight_shared),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            data_shared: make_swizzled_layout(data_shared),
                            weight_shared: make_swizzled_layout(weight_shared),
                            output_shared: make_swizzled_layout(output_shared),
                        }
                    )
                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.clear(accumulator)
                for tile_k in T.Pipelined(
                    inner // block_k, num_stages=num_stages
                ):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k],
                        data_shared,
                    )
                    T.copy(
                        weight[tile_k * block_k, block_x * block_n],
                        weight_shared,
                    )
                    for ki in T.serial(block_k // emitter.micro_size_k):
                        emitter.ldmatrix_a(data_local, data_shared, ki)
                        emitter.ldmatrix_b(weight_local, weight_shared, ki)
                        for atom_m, atom_n in T.grid(
                            emitter.mma_num_inst_m, emitter.mma_num_inst_n
                        ):
                            emitter.mma_atom(
                                data_local,
                                weight_local,
                                accumulator,
                                atom_m,
                                atom_n,
                                ki,
                            )

                if direct_epilogue:
                    thread_binding = emitter.get_thread_binding()
                    lane, warp_n, warp_m = emitter.extract_thread_binding(
                        thread_binding
                    )
                    for mma_m, mma_n in T.grid(
                        emitter.warp_rows, emitter.warp_cols
                    ):
                        for pair in T.serial(emitter.local_size_out // 2):
                            for pair_lane in T.vectorized(2):
                                local_id = pair * 2 + pair_lane
                                row, column = T.meta_var(
                                    mma_store_index_map(lane, local_id)
                                )
                                global_row = (
                                    block_y * block_m
                                    + (warp_m * emitter.warp_rows + mma_m)
                                    * emitter.M_DIM
                                    + row
                                )
                                global_column = (
                                    block_x * block_n
                                    + (warp_n * emitter.warp_cols + mma_n)
                                    * emitter.n_dim
                                    + column
                                )
                                accumulator_index = (
                                    mma_m
                                    * emitter.warp_cols
                                    * emitter.local_size_out
                                    + mma_n * emitter.local_size_out
                                    + local_id
                                )
                                if global_row < rows:
                                    output_residual[
                                        global_row, global_column
                                    ] = T.cast(
                                        accumulator[accumulator_index]
                                        + output_residual[
                                            global_row, global_column
                                        ],
                                        T.float16,
                                    )
                else:
                    emitter.stmatrix(accumulator, output_shared)
                    for row, column in T.Parallel(block_m, block_n):
                        global_row = block_y * block_m + row
                        global_column = block_x * block_n + column
                        if global_row < rows:
                            output_residual[
                                global_row, global_column
                            ] = T.cast(
                                output_shared[row, column]
                                + output_residual[
                                    global_row, global_column
                                ],
                                T.float16,
                            )

        return main

    return build()


@lru_cache(maxsize=None)
def affine_silu_matmul_2d(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
):
    """Per-channel affine-SiLU fused into an FP16 GEMM input tile load."""

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            scale: T.Tensor((inner,), T.float16),
            bias: T.Tensor((inner,), T.float16),
            weight: T.Tensor((inner, columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (block_x, block_y):
                T.disable_warp_group_reg_alloc()
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    for local_row, local_k in T.Parallel(block_m, block_k):
                        row = block_y * block_m + local_row
                        k = tile_k * block_k + local_k
                        if row < rows and k < inner:
                            scaled = T.cast(
                                T.cast(data[row, k], T.float32)
                                * T.cast(scale[k], T.float32),
                                T.float16,
                            )
                            biased = T.cast(
                                T.cast(scaled, T.float32)
                                + T.cast(bias[k], T.float32),
                                T.float16,
                            )
                            sigmoid = T.cast(
                                1.0
                                / (
                                    1.0
                                    + T.exp(-T.cast(biased, T.float32))
                                ),
                                T.float16,
                            )
                            data_shared[local_row, local_k] = T.cast(
                                T.cast(biased, T.float32)
                                * T.cast(sigmoid, T.float32),
                                T.float16,
                            )
                        else:
                            data_shared[local_row, local_k] = 0
                    T.copy(
                        weight[tile_k * block_k, block_x * block_n],
                        weight_shared,
                    )
                    T.gemm(data_shared, weight_shared, accumulator)
                T.copy(
                    accumulator,
                    output[block_y * block_m, block_x * block_n],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def affine_silu_matmul_residual_2d(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
):
    """Affine-SiLU input fusion plus an exact FP16 residual epilogue."""

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            scale: T.Tensor((inner,), T.float16),
            bias: T.Tensor((inner,), T.float16),
            weight: T.Tensor((inner, columns), T.float16),
            residual: T.Tensor((rows, columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (block_x, block_y):
                T.disable_warp_group_reg_alloc()
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    for local_row, local_k in T.Parallel(block_m, block_k):
                        row = block_y * block_m + local_row
                        k = tile_k * block_k + local_k
                        if row < rows and k < inner:
                            scaled = T.cast(
                                T.cast(data[row, k], T.float32)
                                * T.cast(scale[k], T.float32),
                                T.float16,
                            )
                            biased = T.cast(
                                T.cast(scaled, T.float32)
                                + T.cast(bias[k], T.float32),
                                T.float16,
                            )
                            sigmoid = T.cast(
                                1.0
                                / (
                                    1.0
                                    + T.exp(-T.cast(biased, T.float32))
                                ),
                                T.float16,
                            )
                            data_shared[local_row, local_k] = T.cast(
                                T.cast(biased, T.float32)
                                * T.cast(sigmoid, T.float32),
                                T.float16,
                            )
                        else:
                            data_shared[local_row, local_k] = 0
                    T.copy(
                        weight[tile_k * block_k, block_x * block_n],
                        weight_shared,
                    )
                    T.gemm(data_shared, weight_shared, accumulator)
                for local_row, local_column in T.Parallel(block_m, block_n):
                    row = block_y * block_m + local_row
                    column = block_x * block_n + local_column
                    if row < rows and column < columns:
                        matmul_rounded = T.cast(
                            accumulator[local_row, local_column], T.float16
                        )
                        sum_rounded = T.cast(
                            T.cast(matmul_rounded, T.float32)
                            + T.cast(residual[row, column], T.float32),
                            T.float16,
                        )
                        accumulator[local_row, local_column] = T.cast(
                            sum_rounded, T.float32
                        )
                T.copy(
                    accumulator,
                    output[block_y * block_m, block_x * block_n],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def matmul_batched(
    groups: int,
    rows: int,
    columns: int,
    inner: int,
    dtype: str = "float32",
):
    block_m, block_n, block_k = 64, 64, 32
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            left: T.Tensor((groups, rows, inner), value_dtype),
            right: T.Tensor((groups, inner, columns), value_dtype),
            output: T.Tensor((groups, rows, columns), value_dtype),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                groups,
                threads=128,
            ) as (block_x, block_y, group):
                left_shared = T.alloc_shared((block_m, block_k), value_dtype)
                right_shared = T.alloc_shared((block_k, block_n), value_dtype)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.disable_warp_group_reg_alloc()
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    T.copy(left[group, block_y * block_m, tile_k * block_k], left_shared)
                    T.copy(right[group, tile_k * block_k, block_x * block_n], right_shared)
                    T.gemm(left_shared, right_shared, accumulator)
                T.copy(accumulator, output[group, block_y * block_m, block_x * block_n])

        return main

    return build()


@lru_cache(maxsize=None)
def qkv_matmul(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
):
    """Three FP16-input/FP32-accum projections sharing each activation tile."""
    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            weight_q: T.Tensor((inner, columns), T.float16),
            weight_k: T.Tensor((inner, columns), T.float16),
            weight_v: T.Tensor((inner, columns), T.float16),
            output_q: T.Tensor((rows, columns), T.float16),
            output_k: T.Tensor((rows, columns), T.float16),
            output_v: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (block_x, block_y):
                T.disable_warp_group_reg_alloc()
                linear_data_shared = T.alloc_shared(
                    (block_m, block_k), T.float16
                )
                gate_data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_q_shared = T.alloc_shared((block_k, block_n), T.float16)
                weight_k_shared = T.alloc_shared((block_k, block_n), T.float16)
                weight_v_shared = T.alloc_shared((block_k, block_n), T.float16)
                accumulator_q = T.alloc_fragment((block_m, block_n), T.float32)
                accumulator_k = T.alloc_fragment((block_m, block_n), T.float32)
                accumulator_v = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(accumulator_q)
                T.clear(accumulator_k)
                T.clear(accumulator_v)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k], data_shared
                    )
                    T.copy(
                        weight_q[tile_k * block_k, block_x * block_n],
                        weight_q_shared,
                    )
                    T.copy(
                        weight_k[tile_k * block_k, block_x * block_n],
                        weight_k_shared,
                    )
                    T.copy(
                        weight_v[tile_k * block_k, block_x * block_n],
                        weight_v_shared,
                    )
                    T.gemm(data_shared, weight_q_shared, accumulator_q)
                    T.gemm(data_shared, weight_k_shared, accumulator_k)
                    T.gemm(data_shared, weight_v_shared, accumulator_v)
                T.copy(
                    accumulator_q,
                    output_q[block_y * block_m, block_x * block_n],
                )
                T.copy(
                    accumulator_k,
                    output_k[block_y * block_m, block_x * block_n],
                )
                T.copy(
                    accumulator_v,
                    output_v[block_y * block_m, block_x * block_n],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def qkv_grouped_matmul(rows: int, columns: int, inner: int):
    """Group three projections in one launch while retaining one accumulator."""
    block_m, block_n, block_k = 64, 64, 32
    column_blocks = math.ceil(columns / block_n)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            weight_q: T.Tensor((inner, columns), T.float16),
            weight_k: T.Tensor((inner, columns), T.float16),
            weight_v: T.Tensor((inner, columns), T.float16),
            output_q: T.Tensor((rows, columns), T.float16),
            output_k: T.Tensor((rows, columns), T.float16),
            output_v: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                column_blocks * 3,
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (grouped_x, block_y):
                T.disable_warp_group_reg_alloc()
                projection = grouped_x // column_blocks
                block_x = grouped_x % column_blocks
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k], data_shared
                    )
                    if projection == 0:
                        T.copy(
                            weight_q[tile_k * block_k, block_x * block_n],
                            weight_shared,
                        )
                    elif projection == 1:
                        T.copy(
                            weight_k[tile_k * block_k, block_x * block_n],
                            weight_shared,
                        )
                    else:
                        T.copy(
                            weight_v[tile_k * block_k, block_x * block_n],
                            weight_shared,
                        )
                    T.gemm(data_shared, weight_shared, accumulator)
                if projection == 0:
                    T.copy(
                        accumulator,
                        output_q[block_y * block_m, block_x * block_n],
                    )
                elif projection == 1:
                    T.copy(
                        accumulator,
                        output_k[block_y * block_m, block_x * block_n],
                    )
                else:
                    T.copy(
                        accumulator,
                        output_v[block_y * block_m, block_x * block_n],
                    )

        return main

    return build()


@lru_cache(maxsize=None)
def qkv_grouped_rmsnorm_matmul(rows: int, columns: int, inner: int):
    """Grouped QKV projection with RMS scale fused into the shared-memory load."""
    block_m, block_n, block_k = 64, 64, 32
    column_blocks = math.ceil(columns / block_n)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            norm_weight: T.Tensor((inner,), T.float16),
            denominator: T.Tensor((rows,), T.float32),
            weight_q: T.Tensor((inner, columns), T.float16),
            weight_k: T.Tensor((inner, columns), T.float16),
            weight_v: T.Tensor((inner, columns), T.float16),
            output_q: T.Tensor((rows, columns), T.float16),
            output_k: T.Tensor((rows, columns), T.float16),
            output_v: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                column_blocks * 3,
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (grouped_x, block_y):
                T.disable_warp_group_reg_alloc()
                projection = grouped_x // column_blocks
                block_x = grouped_x % column_blocks
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    for local_row, local_k in T.Parallel(block_m, block_k):
                        row = block_y * block_m + local_row
                        k = tile_k * block_k + local_k
                        if row < rows and k < inner:
                            data_shared[local_row, local_k] = T.cast(
                                T.cast(data[row, k], T.float32)
                                / denominator[row]
                                * T.cast(norm_weight[k], T.float32),
                                T.float16,
                            )
                        else:
                            data_shared[local_row, local_k] = 0
                    if projection == 0:
                        T.copy(
                            weight_q[tile_k * block_k, block_x * block_n],
                            weight_shared,
                        )
                    elif projection == 1:
                        T.copy(
                            weight_k[tile_k * block_k, block_x * block_n],
                            weight_shared,
                        )
                    else:
                        T.copy(
                            weight_v[tile_k * block_k, block_x * block_n],
                            weight_shared,
                        )
                    T.gemm(data_shared, weight_shared, accumulator)
                if projection == 0:
                    T.copy(
                        accumulator,
                        output_q[block_y * block_m, block_x * block_n],
                    )
                elif projection == 1:
                    T.copy(
                        accumulator,
                        output_k[block_y * block_m, block_x * block_n],
                    )
                else:
                    T.copy(
                        accumulator,
                        output_v[block_y * block_m, block_x * block_n],
                    )

        return main

    return build()


@lru_cache(maxsize=None)
def ffn_swiglu(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 1,
    threads: int = 128,
    accumulation: str = "float32",
    enable_rasterization: bool = False,
    warp_policy: str = "square",
    enable_warp_specialization: bool = True,
    reuse_a_fragment: bool = False,
):
    """Fixed-shape dual GEMM with a fused SiLU-times-gate epilogue.

    ``accumulation`` controls both the MMA accumulator and epilogue arithmetic.
    Keeping those tied makes the FP16 mode directly comparable with CUTLASS
    example 45 while retaining a separate FP32-accumulation accuracy track.
    """

    accumulation_dtype = _value_dtype(accumulation)
    policies = {
        "square": T.GemmWarpPolicy.Square,
        "full_row": T.GemmWarpPolicy.FullRow,
        "full_col": T.GemmWarpPolicy.FullCol,
    }
    if warp_policy not in policies:
        raise ValueError(f"Unsupported GEMM warp policy: {warp_policy}")
    policy = policies[warp_policy]
    pass_configs = dict(FLASH_ATTENTION_PASS_CONFIGS)
    if not enable_warp_specialization:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=pass_configs,
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            linear_weight: T.Tensor((inner, columns), T.float16),
            gate_weight: T.Tensor((inner, columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=threads,
            ) as (block_x, block_y):
                T.disable_warp_group_reg_alloc()
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                linear_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                gate_weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                if reuse_a_fragment:
                    data_fragment = T.alloc_fragment((block_m, block_k), T.float16)
                linear = T.alloc_fragment((block_m, block_n), accumulation_dtype)
                gate = T.alloc_fragment((block_m, block_n), accumulation_dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.clear(linear)
                T.clear(gate)
                for tile_k in T.Pipelined(
                    T.ceildiv(inner, block_k), num_stages=num_stages
                ):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k], data_shared
                    )
                    T.copy(
                        linear_weight[tile_k * block_k, block_x * block_n],
                        linear_weight_shared,
                    )
                    T.copy(
                        gate_weight[tile_k * block_k, block_x * block_n],
                        gate_weight_shared,
                    )
                    if reuse_a_fragment:
                        T.copy(data_shared, data_fragment)
                        T.gemm(
                            data_fragment,
                            linear_weight_shared,
                            linear,
                            policy=policy,
                        )
                        T.gemm(
                            data_fragment,
                            gate_weight_shared,
                            gate,
                            policy=policy,
                        )
                    else:
                        T.gemm(
                            data_shared,
                            linear_weight_shared,
                            linear,
                            policy=policy,
                        )
                        T.gemm(
                            data_shared,
                            gate_weight_shared,
                            gate,
                            policy=policy,
                        )
                for row, column in T.Parallel(block_m, block_n):
                    linear[row, column] = (
                        linear[row, column]
                        / (1.0 + T.exp(-linear[row, column]))
                        * gate[row, column]
                    )
                T.copy(
                    linear,
                    output[block_y * block_m, block_x * block_n],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def ffn_swiglu_intrinsics(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 2,
    accumulation: str = "float16",
    enable_rasterization: bool = False,
    block_row_warps: int = 2,
    block_col_warps: int = 2,
    epilogue: str = "swiglu",
    fine_grained_mma: bool = False,
    min_blocks_per_sm: int = 1,
):
    """Dual GEMM schedule that reuses each MMA-sized A register fragment."""

    if epilogue not in (
        "linear",
        "add",
        "multiply",
        "swiglu",
        "swiglu_tanh",
        "swiglu_tanh_f16",
        "swiglu_tanh_f16x2",
    ):
        raise ValueError(f"Unsupported FFN epilogue: {epilogue}")
    accumulation_dtype = _value_dtype(accumulation)
    threads = 32 * block_row_warps * block_col_warps
    if block_m % block_row_warps != 0 or block_n % block_col_warps != 0:
        raise ValueError("Block tile must divide evenly across its warp layout")
    warp_m = block_m // block_row_warps
    warp_n = block_n // block_col_warps
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.float16,
        b_dtype=T.float16,
        accum_dtype=accumulation_dtype,
        a_transposed=False,
        b_transposed=False,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_m,
        warp_col_tiles=warp_n,
        chunk=block_k,
    )
    a_local_size = emitter.warp_rows * emitter.local_size_a
    b_local_size = emitter.warp_cols * emitter.local_size_b
    c_local_size = (
        emitter.warp_rows * emitter.warp_cols * emitter.local_size_out
    )
    pass_configs = dict(FLASH_ATTENTION_PASS_CONFIGS)
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=pass_configs,
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            linear_weight: T.Tensor((inner, columns), T.float16),
            gate_weight: T.Tensor((inner, columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=threads,
            ) as (block_x, block_y):
                T.annotate_min_blocks_per_sm(min_blocks_per_sm)
                T.import_source(FP16_TANH_INTRINSIC_SOURCE)
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                linear_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                gate_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                output_shared = T.alloc_shared((block_m, block_n), T.float16)
                data_local = T.alloc_local((a_local_size,), T.float16)
                linear_weight_local = T.alloc_local((b_local_size,), T.float16)
                gate_weight_local = T.alloc_local((b_local_size,), T.float16)
                linear_local = T.alloc_local((c_local_size,), accumulation_dtype)
                gate_local = T.alloc_local((c_local_size,), accumulation_dtype)

                T.annotate_layout(
                    {
                        data_shared: make_swizzled_layout(data_shared),
                        linear_weight_shared: make_swizzled_layout(
                            linear_weight_shared
                        ),
                        gate_weight_shared: make_swizzled_layout(
                            gate_weight_shared
                        ),
                        output_shared: make_swizzled_layout(output_shared),
                    }
                )
                T.use_swizzle(panel_size=10, enable=enable_rasterization)
                T.clear(linear_local)
                T.clear(gate_local)
                for tile_k in T.Pipelined(
                    T.ceildiv(inner, block_k), num_stages=num_stages
                ):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k], data_shared
                    )
                    T.copy(
                        linear_weight[tile_k * block_k, block_x * block_n],
                        linear_weight_shared,
                    )
                    T.copy(
                        gate_weight[tile_k * block_k, block_x * block_n],
                        gate_weight_shared,
                    )
                    for ki in T.serial(block_k // emitter.micro_size_k):
                        emitter.ldmatrix_a(data_local, data_shared, ki)
                        emitter.ldmatrix_b(
                            linear_weight_local, linear_weight_shared, ki
                        )
                        emitter.ldmatrix_b(
                            gate_weight_local, gate_weight_shared, ki
                        )
                        for atom_m, atom_n in T.grid(
                            emitter.mma_num_inst_m, emitter.mma_num_inst_n
                        ):
                            if fine_grained_mma:
                                a_offset = atom_m * emitter.local_size_a
                                b_offset = atom_n * emitter.local_size_b
                                c_offset = (
                                    atom_m
                                    * emitter.warp_cols
                                    * emitter.local_size_out
                                    + atom_n * emitter.local_size_out
                                )
                                T.ptx_mma(
                                    accumulation_dtype,
                                    emitter.mma_prefix,
                                    "row",
                                    "col",
                                    emitter.a_dtype_abbrv,
                                    emitter.b_dtype_abbrv,
                                    emitter.accum_dtype_abbrv,
                                    data_local.data,
                                    a_offset,
                                    linear_weight_local.data,
                                    b_offset,
                                    linear_local.data,
                                    c_offset,
                                    T.bool(False),
                                )
                                T.ptx_mma(
                                    accumulation_dtype,
                                    emitter.mma_prefix,
                                    "row",
                                    "col",
                                    emitter.a_dtype_abbrv,
                                    emitter.b_dtype_abbrv,
                                    emitter.accum_dtype_abbrv,
                                    data_local.data,
                                    a_offset,
                                    gate_weight_local.data,
                                    b_offset,
                                    gate_local.data,
                                    c_offset,
                                    T.bool(False),
                                )
                                T.ptx_mma(
                                    accumulation_dtype,
                                    emitter.mma_prefix,
                                    "row",
                                    "col",
                                    emitter.a_dtype_abbrv,
                                    emitter.b_dtype_abbrv,
                                    emitter.accum_dtype_abbrv,
                                    data_local.data,
                                    a_offset,
                                    linear_weight_local.data,
                                    b_offset + emitter.local_size_b // 2,
                                    linear_local.data,
                                    c_offset + emitter.local_size_out // 2,
                                    T.bool(False),
                                )
                                T.ptx_mma(
                                    accumulation_dtype,
                                    emitter.mma_prefix,
                                    "row",
                                    "col",
                                    emitter.a_dtype_abbrv,
                                    emitter.b_dtype_abbrv,
                                    emitter.accum_dtype_abbrv,
                                    data_local.data,
                                    a_offset,
                                    gate_weight_local.data,
                                    b_offset + emitter.local_size_b // 2,
                                    gate_local.data,
                                    c_offset + emitter.local_size_out // 2,
                                    T.bool(False),
                                )
                            else:
                                emitter.mma_atom(
                                    data_local,
                                    linear_weight_local,
                                    linear_local,
                                    atom_m,
                                    atom_n,
                                    ki,
                                )
                                emitter.mma_atom(
                                    data_local,
                                    gate_weight_local,
                                    gate_local,
                                    atom_m,
                                    atom_n,
                                    ki,
                                )

                if epilogue == "swiglu_tanh_f16x2":
                    half2 = T.float16x2(T.float16(0.5), T.float16(0.5))
                    for pair_index in T.unroll(c_local_size // 2):
                        linear_pair = T.float16x2(
                            linear_local[pair_index * 2],
                            linear_local[pair_index * 2 + 1],
                        )
                        gate_pair = T.float16x2(
                            gate_local[pair_index * 2],
                            gate_local[pair_index * 2 + 1],
                        )
                        tanh_pair = T.reinterpret(
                            T.call_extern(
                                "tilelang_h2tanh_approx",
                                T.reinterpret(
                                    T.mul2(linear_pair, half2), T.uint32
                                ),
                                dtype=T.uint32,
                            ),
                            T.float16x2,
                        )
                        sigmoid_pair = T.fma2(tanh_pair, half2, half2)
                        result_bits = T.reinterpret(
                            T.mul2(
                                T.mul2(linear_pair, sigmoid_pair), gate_pair
                            ),
                            T.uint32,
                        )
                        linear_local[pair_index * 2] = T.reinterpret(
                            T.cast(result_bits & T.uint32(0xFFFF), T.uint16),
                            T.float16,
                        )
                        linear_local[pair_index * 2 + 1] = T.reinterpret(
                            T.cast(result_bits >> 16, T.uint16), T.float16
                        )
                elif epilogue != "linear":
                    for outer in T.serial(c_local_size // 2):
                        for inner_index in T.vectorized(2):
                            index = outer * 2 + inner_index
                            if epilogue == "add":
                                linear_local[index] = (
                                    linear_local[index] + gate_local[index]
                                )
                            elif epilogue == "multiply":
                                linear_local[index] = (
                                    linear_local[index] * gate_local[index]
                                )
                            elif epilogue == "swiglu":
                                linear_local[index] = (
                                    linear_local[index]
                                    / (1.0 + T.exp(-linear_local[index]))
                                    * gate_local[index]
                                )
                            elif epilogue == "swiglu_tanh":
                                linear_local[index] = (
                                    linear_local[index]
                                    * (
                                        0.5
                                        * T.tanh(0.5 * linear_local[index])
                                        + 0.5
                                    )
                                    * gate_local[index]
                                )
                            else:
                                linear_local[index] = (
                                    linear_local[index]
                                    * (
                                        0.5
                                        * T.call_extern(
                                            "tilelang_htanh_approx",
                                            T.cast(
                                                0.5 * linear_local[index],
                                                T.float16,
                                            ),
                                            dtype=T.float16,
                                        )
                                        + 0.5
                                    )
                                    * gate_local[index]
                                )
                emitter.stmatrix(linear_local, output_shared)
                for row, column in T.Parallel(block_m, block_n):
                    if block_y * block_m + row < rows:
                        output[
                            block_y * block_m + row,
                            block_x * block_n + column,
                        ] = output_shared[row, column]

        return main

    return build()


@lru_cache(maxsize=None)
def ffn_swiglu_packed_wide_intrinsics(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 128,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 2,
    min_blocks_per_sm: int = 1,
    block_row_warps: int = 4,
    block_col_warps: int = 2,
    direct_epilogue: bool = False,
):
    """One packed 2N GEMM followed by the accepted FP16 half2 SwiGLU.

    The packed weight is block-interleaved as ``[linear N64, gate N64]``.
    A 128-column GEMM therefore computes the same output tile as the retained
    pair of 64-column GEMMs.  The shared-memory epilogue supports arbitrary
    compatible warp layouts.  The direct epilogue requires one column warp so
    that each warp owns matching linear/gate columns and can apply SwiGLU in
    registers without an inter-warp exchange.
    """

    packed_n = 2 * block_n
    threads = 32 * block_row_warps * block_col_warps
    if block_m % block_row_warps != 0 or packed_n % block_col_warps != 0:
        raise ValueError("Packed block tile must divide evenly across warps")
    if direct_epilogue and block_col_warps != 1:
        raise ValueError("Direct packed epilogue requires one column warp")
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.float16,
        b_dtype=T.float16,
        accum_dtype=T.float16,
        a_transposed=False,
        b_transposed=False,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=block_m // block_row_warps,
        warp_col_tiles=packed_n // block_col_warps,
        chunk=block_k,
    )
    a_local_size = emitter.warp_rows * emitter.local_size_a
    b_local_size = emitter.warp_cols * emitter.local_size_b
    c_local_size = (
        emitter.warp_rows * emitter.warp_cols * emitter.local_size_out
    )
    pass_configs = dict(FLASH_ATTENTION_PASS_CONFIGS)
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=pass_configs,
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            packed_weight: T.Tensor((inner, 2 * columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=threads,
            ) as (block_x, block_y):
                T.annotate_min_blocks_per_sm(min_blocks_per_sm)
                T.import_source(FP16_TANH_INTRINSIC_SOURCE)
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                weight_shared = T.alloc_shared((block_k, packed_n), T.float16)
                if not direct_epilogue:
                    packed_output_shared = T.alloc_shared(
                        (block_m, packed_n), T.float16
                    )
                data_local = T.alloc_local((a_local_size,), T.float16)
                weight_local = T.alloc_local((b_local_size,), T.float16)
                accumulator = T.alloc_local((c_local_size,), T.float16)
                if direct_epilogue:
                    T.annotate_layout(
                        {
                            data_shared: make_swizzled_layout(data_shared),
                            weight_shared: make_swizzled_layout(weight_shared),
                        }
                    )
                else:
                    T.annotate_layout(
                        {
                            data_shared: make_swizzled_layout(data_shared),
                            weight_shared: make_swizzled_layout(weight_shared),
                            packed_output_shared: make_swizzled_layout(
                                packed_output_shared
                            ),
                        }
                    )
                T.clear(accumulator)
                for tile_k in T.Pipelined(
                    T.ceildiv(inner, block_k), num_stages=num_stages
                ):
                    T.copy(
                        data[block_y * block_m, tile_k * block_k], data_shared
                    )
                    T.copy(
                        packed_weight[
                            tile_k * block_k, block_x * packed_n
                        ],
                        weight_shared,
                    )
                    for ki in T.serial(block_k // emitter.micro_size_k):
                        emitter.ldmatrix_a(data_local, data_shared, ki)
                        emitter.ldmatrix_b(weight_local, weight_shared, ki)
                        for atom_m, atom_n in T.grid(
                            emitter.mma_num_inst_m, emitter.mma_num_inst_n
                        ):
                            emitter.mma_atom(
                                data_local,
                                weight_local,
                                accumulator,
                                atom_m,
                                atom_n,
                                ki,
                            )

                half2 = T.float16x2(T.float16(0.5), T.float16(0.5))
                if direct_epilogue:
                    logical_warp_cols = emitter.warp_cols // 2
                    for mma_m, mma_n in T.grid(
                        emitter.warp_rows, logical_warp_cols
                    ):
                        for pair in T.unroll(emitter.local_size_out // 2):
                            linear_index = (
                                mma_m
                                * emitter.warp_cols
                                * emitter.local_size_out
                                + mma_n * emitter.local_size_out
                                + pair * 2
                            )
                            gate_index = (
                                linear_index
                                + logical_warp_cols * emitter.local_size_out
                            )
                            linear_pair = T.float16x2(
                                accumulator[linear_index],
                                accumulator[linear_index + 1],
                            )
                            gate_pair = T.float16x2(
                                accumulator[gate_index],
                                accumulator[gate_index + 1],
                            )
                            tanh_pair = T.reinterpret(
                                T.call_extern(
                                    "tilelang_h2tanh_approx",
                                    T.reinterpret(
                                        T.mul2(linear_pair, half2), T.uint32
                                    ),
                                    dtype=T.uint32,
                                ),
                                T.float16x2,
                            )
                            sigmoid_pair = T.fma2(tanh_pair, half2, half2)
                            result_bits = T.reinterpret(
                                T.mul2(
                                    T.mul2(linear_pair, sigmoid_pair),
                                    gate_pair,
                                ),
                                T.uint32,
                            )
                            accumulator[linear_index] = T.reinterpret(
                                T.cast(
                                    result_bits & T.uint32(0xFFFF), T.uint16
                                ),
                                T.float16,
                            )
                            accumulator[linear_index + 1] = T.reinterpret(
                                T.cast(result_bits >> 16, T.uint16), T.float16
                            )

                    thread_binding = emitter.get_thread_binding()
                    lane, warp_n, warp_m = emitter.extract_thread_binding(
                        thread_binding
                    )
                    for mma_m, mma_n in T.grid(
                        emitter.warp_rows, logical_warp_cols
                    ):
                        for pair in T.serial(emitter.local_size_out // 2):
                            for pair_lane in T.vectorized(2):
                                local_id = pair * 2 + pair_lane
                                row, column = T.meta_var(
                                    mma_store_index_map(lane, local_id)
                                )
                                global_row = (
                                    block_y * block_m
                                    + (warp_m * emitter.warp_rows + mma_m)
                                    * emitter.M_DIM
                                    + row
                                )
                                global_column = (
                                    block_x * block_n
                                    + mma_n * emitter.n_dim
                                    + column
                                )
                                accumulator_index = (
                                    mma_m
                                    * emitter.warp_cols
                                    * emitter.local_size_out
                                    + mma_n * emitter.local_size_out
                                    + local_id
                                )
                                if global_row < rows:
                                    output[global_row, global_column] = accumulator[
                                        accumulator_index
                                    ]
                else:
                    emitter.stmatrix(accumulator, packed_output_shared)
                    T.sync_threads()
                    for row, pair in T.Parallel(block_m, block_n // 2):
                        column = pair * 2
                        linear_pair = T.float16x2(
                            packed_output_shared[row, column],
                            packed_output_shared[row, column + 1],
                        )
                        gate_pair = T.float16x2(
                            packed_output_shared[row, block_n + column],
                            packed_output_shared[row, block_n + column + 1],
                        )
                        tanh_pair = T.reinterpret(
                            T.call_extern(
                                "tilelang_h2tanh_approx",
                                T.reinterpret(
                                    T.mul2(linear_pair, half2), T.uint32
                                ),
                                dtype=T.uint32,
                            ),
                            T.float16x2,
                        )
                        sigmoid_pair = T.fma2(tanh_pair, half2, half2)
                        result_bits = T.reinterpret(
                            T.mul2(
                                T.mul2(linear_pair, sigmoid_pair), gate_pair
                            ),
                            T.uint32,
                        )
                        if block_y * block_m + row < rows:
                            output[
                                block_y * block_m + row,
                                block_x * block_n + column,
                            ] = T.reinterpret(
                                T.cast(
                                    result_bits & T.uint32(0xFFFF), T.uint16
                                ),
                                T.float16,
                            )
                            output[
                                block_y * block_m + row,
                                block_x * block_n + column + 1,
                            ] = T.reinterpret(
                                T.cast(result_bits >> 16, T.uint16), T.float16
                            )

        return main

    return build()


@lru_cache(maxsize=None)
def ffn_persistent_residual(
    rows: int,
    inner: int,
    ffn_columns: int,
    output_columns: int,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 64,
    accumulation: str = "float32",
    epilogue: str = "swiglu",
    disable_warp_specialized: bool = False,
    num_stages: int = 1,
    threads: int = 128,
):
    """Materialize SwiGLU in shared memory, then persist across output N tiles."""
    if inner % block_k or ffn_columns % block_n or output_columns % block_n:
        raise ValueError("Persistent FFN currently requires exact K/N tiles")
    if epilogue not in (
        "swiglu",
        "swiglu_tanh_f16",
        "swiglu_tanh_f16x2",
    ):
        raise ValueError(f"Unsupported persistent FFN epilogue: {epilogue}")
    accumulation_dtype = _value_dtype(accumulation)
    pass_configs = dict(FLASH_ATTENTION_PASS_CONFIGS)
    if disable_warp_specialized:
        pass_configs[tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED] = True

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=pass_configs,
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            linear_weight: T.Tensor((inner, ffn_columns), T.float16),
            gate_weight: T.Tensor((inner, ffn_columns), T.float16),
            output_weight: T.Tensor(
                (ffn_columns, output_columns), T.float16
            ),
            residual: T.Tensor((rows, output_columns), T.float16),
            output: T.Tensor((rows, output_columns), T.float16),
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as block_y:
                T.disable_warp_group_reg_alloc()
                if epilogue in ("swiglu_tanh_f16", "swiglu_tanh_f16x2"):
                    T.import_source(FP16_TANH_INTRINSIC_SOURCE)
                swiglu_shared = T.alloc_shared(
                    (block_m, ffn_columns), T.float16
                )
                data_shared = T.alloc_shared((block_m, block_k), T.float16)
                linear_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                gate_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                # These buffers belong to the later linear2 pipeline. Keeping
                # them logically distinct prevents TileLang from inferring a
                # nested stage dimension when both mainloops are double-buffered.
                output_data_shared = T.alloc_shared(
                    (block_m, block_k), T.float16
                )
                output_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                linear = T.alloc_fragment(
                    (block_m, block_n), accumulation_dtype
                )
                gate = T.alloc_fragment(
                    (block_m, block_n), accumulation_dtype
                )
                output_accumulator = T.alloc_fragment(
                    (block_m, block_n), accumulation_dtype
                )

                for ffn_block in T.serial(ffn_columns // block_n):
                    T.clear(linear)
                    T.clear(gate)
                    for tile_k in T.Pipelined(
                        inner // block_k, num_stages=num_stages
                    ):
                        T.copy(
                            data[block_y * block_m, tile_k * block_k],
                            data_shared,
                        )
                        T.copy(
                            linear_weight[
                                tile_k * block_k, ffn_block * block_n
                            ],
                            linear_weight_shared,
                        )
                        T.copy(
                            gate_weight[
                                tile_k * block_k, ffn_block * block_n
                            ],
                            gate_weight_shared,
                        )
                        T.gemm(data_shared, linear_weight_shared, linear)
                        T.gemm(data_shared, gate_weight_shared, gate)
                    if epilogue == "swiglu_tanh_f16x2":
                        half2 = T.float16x2(T.float16(0.5), T.float16(0.5))
                        for local_row, pair_column in T.Parallel(
                            block_m, block_n // 2
                        ):
                            local_column = pair_column * 2
                            linear_pair = T.float16x2(
                                linear[local_row, local_column],
                                linear[local_row, local_column + 1],
                            )
                            gate_pair = T.float16x2(
                                gate[local_row, local_column],
                                gate[local_row, local_column + 1],
                            )
                            tanh_pair = T.reinterpret(
                                T.call_extern(
                                    "tilelang_h2tanh_approx",
                                    T.reinterpret(
                                        T.mul2(linear_pair, half2), T.uint32
                                    ),
                                    dtype=T.uint32,
                                ),
                                T.float16x2,
                            )
                            sigmoid_pair = T.fma2(tanh_pair, half2, half2)
                            result_bits = T.reinterpret(
                                T.mul2(
                                    T.mul2(linear_pair, sigmoid_pair),
                                    gate_pair,
                                ),
                                T.uint32,
                            )
                            linear[local_row, local_column] = T.reinterpret(
                                T.cast(
                                    result_bits & T.uint32(0xFFFF), T.uint16
                                ),
                                T.float16,
                            )
                            linear[
                                local_row, local_column + 1
                            ] = T.reinterpret(
                                T.cast(result_bits >> 16, T.uint16),
                                T.float16,
                            )
                    elif epilogue == "swiglu_tanh_f16":
                        for local_row, outer in T.Parallel(
                            block_m, block_n // 2
                        ):
                            for inner_index in T.vectorized(2):
                                local_column = outer * 2 + inner_index
                                linear[local_row, local_column] = (
                                    linear[local_row, local_column]
                                    * (
                                        T.float16(0.5)
                                        * T.call_extern(
                                            "tilelang_htanh_approx",
                                            T.cast(
                                                T.float16(0.5)
                                                * linear[
                                                    local_row, local_column
                                                ],
                                                T.float16,
                                            ),
                                            dtype=T.float16,
                                        )
                                        + T.float16(0.5)
                                    )
                                    * gate[local_row, local_column]
                                )
                    else:
                        for local_row, local_column in T.Parallel(
                            block_m, block_n
                        ):
                            linear[local_row, local_column] = (
                                linear[local_row, local_column]
                                / (
                                    1.0
                                    + T.exp(-linear[local_row, local_column])
                                )
                                * gate[local_row, local_column]
                            )
                    T.copy(
                        linear,
                        swiglu_shared[
                            0:block_m,
                            ffn_block
                            * block_n : (ffn_block + 1)
                            * block_n,
                        ],
                    )

                for output_block in T.serial(output_columns // block_n):
                    T.clear(output_accumulator)
                    for ffn_block in T.Pipelined(
                        ffn_columns // block_k, num_stages=num_stages
                    ):
                        T.copy(
                            swiglu_shared[
                                0:block_m,
                                ffn_block
                                * block_k : (ffn_block + 1)
                                * block_k,
                            ],
                            output_data_shared,
                        )
                        T.copy(
                            output_weight[
                                ffn_block * block_k, output_block * block_n
                            ],
                            output_weight_shared,
                        )
                        T.gemm(
                            output_data_shared,
                            output_weight_shared,
                            output_accumulator,
                        )
                    for local_row, local_column in T.Parallel(
                        block_m, block_n
                    ):
                        row = block_y * block_m + local_row
                        column = output_block * block_n + local_column
                        if row < rows:
                            matmul_rounded = T.cast(
                                output_accumulator[local_row, local_column],
                                T.float16,
                            )
                            sum_rounded = T.cast(
                                T.cast(matmul_rounded, T.float32)
                                + T.cast(residual[row, column], T.float32),
                                T.float16,
                            )
                            output_accumulator[
                                local_row, local_column
                            ] = T.cast(sum_rounded, T.float32)
                    T.copy(
                        output_accumulator,
                        output[
                            block_y * block_m, output_block * block_n
                        ],
                    )

        return main

    return build()


@lru_cache(maxsize=None)
def ffn_rmsnorm_swiglu(
    rows: int,
    columns: int,
    inner: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
):
    """RMS-scaled dual GEMM with a fused FP32 SiLU-times-gate epilogue."""

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=FLASH_ATTENTION_PASS_CONFIGS,
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((rows, inner), T.float16),
            norm_weight: T.Tensor((inner,), T.float16),
            denominator: T.Tensor((rows,), T.float32),
            linear_weight: T.Tensor((inner, columns), T.float16),
            gate_weight: T.Tensor((inner, columns), T.float16),
            output: T.Tensor((rows, columns), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(columns, block_n),
                T.ceildiv(rows, block_m),
                threads=128,
            ) as (block_x, block_y):
                T.disable_warp_group_reg_alloc()
                linear_data_shared = T.alloc_shared(
                    (block_m, block_k), T.float16
                )
                gate_data_shared = T.alloc_shared((block_m, block_k), T.float16)
                linear_weight_shared = T.alloc_shared(
                    (block_k, block_n), T.float16
                )
                gate_weight_shared = T.alloc_shared((block_k, block_n), T.float16)
                linear = T.alloc_fragment((block_m, block_n), T.float32)
                gate = T.alloc_fragment((block_m, block_n), T.float32)
                T.clear(linear)
                T.clear(gate)
                for tile_k in T.Pipelined(T.ceildiv(inner, block_k), num_stages=1):
                    for local_row, local_k in T.Parallel(block_m, block_k):
                        row = block_y * block_m + local_row
                        k = tile_k * block_k + local_k
                        if row < rows and k < inner:
                            linear_data_shared[local_row, local_k] = T.cast(
                                T.cast(data[row, k], T.float32)
                                / denominator[row]
                                * T.cast(norm_weight[k], T.float32),
                                T.float16,
                            )
                        else:
                            linear_data_shared[local_row, local_k] = 0
                    for local_row, local_k in T.Parallel(block_m, block_k):
                        row = block_y * block_m + local_row
                        k = tile_k * block_k + local_k
                        if row < rows and k < inner:
                            gate_data_shared[local_row, local_k] = T.cast(
                                T.cast(data[row, k], T.float32)
                                / denominator[row]
                                * T.cast(norm_weight[k], T.float32),
                                T.float16,
                            )
                        else:
                            gate_data_shared[local_row, local_k] = 0
                    T.copy(
                        linear_weight[tile_k * block_k, block_x * block_n],
                        linear_weight_shared,
                    )
                    T.copy(
                        gate_weight[tile_k * block_k, block_x * block_n],
                        gate_weight_shared,
                    )
                    T.gemm(linear_data_shared, linear_weight_shared, linear)
                    T.gemm(gate_data_shared, gate_weight_shared, gate)
                for local_row, local_column in T.Parallel(block_m, block_n):
                    linear[local_row, local_column] = (
                        linear[local_row, local_column]
                        / (1.0 + T.exp(-linear[local_row, local_column]))
                        * gate[local_row, local_column]
                    )
                T.copy(
                    linear,
                    output[block_y * block_m, block_x * block_n],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def flash_attention_bhsd(
    batch: int,
    heads: int,
    sequence: int,
    dimension: int,
    score_scale: float,
    accumulation: str = "both16",
):
    """Non-causal BHSD attention with selectable QK/PV MMA accumulation."""
    if accumulation not in ("fp32", "qk16", "pv16", "both16"):
        raise ValueError(f"Unsupported attention accumulation: {accumulation}")
    block_m, block_n = 64, 64
    threads = 128
    qk_accum_dtype = T.float16 if accumulation in ("qk16", "both16") else T.float32
    pv_accum_dtype = T.float16 if accumulation in ("pv16", "both16") else T.float32
    scale_log2e = score_scale * math.log2(math.e)

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=FLASH_ATTENTION_PASS_CONFIGS,
    )
    def build():
        @T.prim_func
        def main(
            query: T.Tensor((batch, heads, sequence, dimension), T.float16),
            key: T.Tensor((batch, heads, sequence, dimension), T.float16),
            value: T.Tensor((batch, heads, sequence, dimension), T.float16),
            output: T.Tensor((batch, heads, sequence, dimension), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(sequence, block_m), heads, batch, threads=threads
            ) as (query_block, head, batch_index):
                T.disable_warp_group_reg_alloc()
                query_shared = T.alloc_shared((block_m, dimension), T.float16)
                key_shared = T.alloc_shared((block_n, dimension), T.float16)
                value_shared = T.alloc_shared((block_n, dimension), T.float16)
                output_shared = T.alloc_shared((block_m, dimension), T.float16)
                scores = T.alloc_fragment((block_m, block_n), qk_accum_dtype)
                probabilities = T.alloc_fragment((block_m, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, dimension), pv_accum_dtype)
                scores_max = T.alloc_fragment((block_m,), T.float32)
                scores_max_previous = T.alloc_fragment((block_m,), T.float32)
                scores_scale = T.alloc_fragment((block_m,), T.float32)
                scores_sum = T.alloc_fragment((block_m,), T.float32)
                denominator = T.alloc_fragment((block_m,), T.float32)

                T.copy(
                    query[
                        batch_index,
                        head,
                        query_block * block_m : (query_block + 1) * block_m,
                        :,
                    ],
                    query_shared,
                )
                T.fill(accumulator, 0)
                T.fill(denominator, 0)
                T.fill(scores_max, -T.infinity(T.float32))

                for key_block in T.Pipelined(
                    T.ceildiv(sequence, block_n), num_stages=1
                ):
                    T.copy(
                        key[
                            batch_index,
                            head,
                            key_block * block_n : (key_block + 1) * block_n,
                            :,
                        ],
                        key_shared,
                    )
                    for row, column in T.Parallel(block_m, block_n):
                        scores[row, column] = T.if_then_else(
                            key_block * block_n + column < sequence,
                            0,
                            -T.infinity(qk_accum_dtype),
                        )
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(scores_max, scores_max_previous)
                    T.fill(scores_max, -T.infinity(T.float32))
                    T.reduce_max(scores, scores_max, dim=1, clear=False)
                    for row in T.Parallel(block_m):
                        scores_max[row] = T.max(
                            scores_max[row], scores_max_previous[row]
                        )
                        scores_scale[row] = T.exp2(
                            scores_max_previous[row] * scale_log2e
                            - scores_max[row] * scale_log2e
                        )
                    for row, column in T.Parallel(block_m, block_n):
                        scores[row, column] = T.exp2(
                            scores[row, column] * scale_log2e
                            - scores_max[row] * scale_log2e
                        )
                    T.reduce_sum(scores, scores_sum, dim=1)
                    for row in T.Parallel(block_m):
                        denominator[row] = (
                            denominator[row] * scores_scale[row] + scores_sum[row]
                        )
                    T.copy(scores, probabilities)

                    for row, column in T.Parallel(block_m, dimension):
                        accumulator[row, column] *= scores_scale[row]
                    T.copy(
                        value[
                            batch_index,
                            head,
                            key_block * block_n : (key_block + 1) * block_n,
                            :,
                        ],
                        value_shared,
                    )
                    T.gemm(
                        probabilities,
                        value_shared,
                        accumulator,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for row, column in T.Parallel(block_m, dimension):
                    accumulator[row, column] /= denominator[row]
                T.copy(accumulator, output_shared)
                T.copy(
                    output_shared,
                    output[
                        batch_index,
                        head,
                        query_block * block_m : (query_block + 1) * block_m,
                        :,
                    ],
                )

        return main

    return build()


@lru_cache(maxsize=None)
def flash_attention_nhwc_rope(
    batch: int,
    heads: int,
    sequence: int,
    dimension: int,
    score_scale: float,
    accumulation: str = "both16",
):
    """Attention with NHWC/BHSD transforms and RoPE fused into tile loads."""
    if accumulation not in ("fp32", "qk16", "pv16", "both16"):
        raise ValueError(f"Unsupported attention accumulation: {accumulation}")
    block_m, block_n = 64, 64
    channels = heads * dimension
    qk_accum_dtype = T.float16 if accumulation in ("qk16", "both16") else T.float32
    pv_accum_dtype = T.float16 if accumulation in ("pv16", "both16") else T.float32
    scale_log2e = score_scale * math.log2(math.e)

    @tilelang.jit(
        out_idx=[],
        target=TARGET,
        execution_backend="cython",
        pass_configs=FLASH_ATTENTION_PASS_CONFIGS,
    )
    def build():
        @T.prim_func
        def main(
            query: T.Tensor((batch, sequence, channels), T.float16),
            key: T.Tensor((batch, sequence, channels), T.float16),
            value: T.Tensor((batch, sequence, channels), T.float16),
            rope_cos: T.Tensor((heads, sequence, dimension), T.float16),
            rope_sin_signed: T.Tensor(
                (heads, sequence, dimension), T.float16
            ),
            output: T.Tensor((batch, sequence, channels), T.float16),
        ):
            with T.Kernel(
                T.ceildiv(sequence, block_m), heads, batch, threads=128
            ) as (query_block, head, batch_index):
                T.disable_warp_group_reg_alloc()
                query_shared = T.alloc_shared((block_m, dimension), T.float16)
                key_shared = T.alloc_shared((block_n, dimension), T.float16)
                value_shared = T.alloc_shared((block_n, dimension), T.float16)
                scores = T.alloc_fragment((block_m, block_n), qk_accum_dtype)
                probabilities = T.alloc_fragment((block_m, block_n), T.float16)
                accumulator = T.alloc_fragment((block_m, dimension), pv_accum_dtype)
                scores_max = T.alloc_fragment((block_m,), T.float32)
                scores_max_previous = T.alloc_fragment((block_m,), T.float32)
                scores_scale = T.alloc_fragment((block_m,), T.float32)
                scores_sum = T.alloc_fragment((block_m,), T.float32)
                denominator = T.alloc_fragment((block_m,), T.float32)

                for row, column in T.Parallel(block_m, dimension):
                    position = query_block * block_m + row
                    channel = head * dimension + column
                    if position < sequence:
                        term = T.cast(
                            T.cast(query[batch_index, position, channel], T.float32)
                            * T.cast(rope_cos[head, position, column], T.float32),
                            T.float16,
                        )
                        swapped_term = T.cast(
                            T.cast(
                                query[
                                    batch_index,
                                    position,
                                    head * dimension + (column ^ 1),
                                ],
                                T.float32,
                            )
                            * T.cast(
                                rope_sin_signed[head, position, column], T.float32
                            ),
                            T.float16,
                        )
                        query_shared[row, column] = T.cast(
                            T.cast(term, T.float32)
                            + T.cast(swapped_term, T.float32),
                            T.float16,
                        )
                    else:
                        query_shared[row, column] = 0

                T.fill(accumulator, 0)
                T.fill(denominator, 0)
                T.fill(scores_max, -T.infinity(T.float32))

                for key_block in T.Pipelined(
                    T.ceildiv(sequence, block_n), num_stages=1
                ):
                    for row, column in T.Parallel(block_n, dimension):
                        position = key_block * block_n + row
                        channel = head * dimension + column
                        if position < sequence:
                            term = T.cast(
                                T.cast(
                                    key[batch_index, position, channel], T.float32
                                )
                                * T.cast(
                                    rope_cos[head, position, column], T.float32
                                ),
                                T.float16,
                            )
                            swapped_term = T.cast(
                                T.cast(
                                    key[
                                        batch_index,
                                        position,
                                        head * dimension + (column ^ 1),
                                    ],
                                    T.float32,
                                )
                                * T.cast(
                                    rope_sin_signed[head, position, column],
                                    T.float32,
                                ),
                                T.float16,
                            )
                            key_shared[row, column] = T.cast(
                                T.cast(term, T.float32)
                                + T.cast(swapped_term, T.float32),
                                T.float16,
                            )
                        else:
                            key_shared[row, column] = 0
                    for row, column in T.Parallel(block_m, block_n):
                        scores[row, column] = T.if_then_else(
                            key_block * block_n + column < sequence,
                            0,
                            -T.infinity(qk_accum_dtype),
                        )
                    T.gemm(
                        query_shared,
                        key_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(scores_max, scores_max_previous)
                    T.fill(scores_max, -T.infinity(T.float32))
                    T.reduce_max(scores, scores_max, dim=1, clear=False)
                    for row in T.Parallel(block_m):
                        scores_max[row] = T.max(
                            scores_max[row], scores_max_previous[row]
                        )
                        scores_scale[row] = T.exp2(
                            scores_max_previous[row] * scale_log2e
                            - scores_max[row] * scale_log2e
                        )
                    for row, column in T.Parallel(block_m, block_n):
                        scores[row, column] = T.exp2(
                            scores[row, column] * scale_log2e
                            - scores_max[row] * scale_log2e
                        )
                    T.reduce_sum(scores, scores_sum, dim=1)
                    for row in T.Parallel(block_m):
                        denominator[row] = (
                            denominator[row] * scores_scale[row] + scores_sum[row]
                        )
                    T.copy(scores, probabilities)

                    for row, column in T.Parallel(block_m, dimension):
                        accumulator[row, column] *= scores_scale[row]
                    for row, column in T.Parallel(block_n, dimension):
                        position = key_block * block_n + row
                        if position < sequence:
                            value_shared[row, column] = value[
                                batch_index,
                                position,
                                head * dimension + column,
                            ]
                        else:
                            value_shared[row, column] = 0
                    T.gemm(
                        probabilities,
                        value_shared,
                        accumulator,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for row, column in T.Parallel(block_m, dimension):
                    position = query_block * block_m + row
                    if position < sequence:
                        output[
                            batch_index,
                            position,
                            head * dimension + column,
                        ] = accumulator[row, column] / denominator[row]

        return main

    return build()


@lru_cache(maxsize=None)
def matmul_2d_fp32(rows: int, columns: int, inner: int):
    """Strict FP32 SIMT MatMul used to validate graph semantics."""
    output_numel = rows * columns

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            left: T.Tensor((rows, inner), T.float32),
            right: T.Tensor((inner, columns), T.float32),
            output: T.Tensor((rows, columns), T.float32),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < output_numel:
                        row = output_index // columns
                        column = output_index % columns
                        accumulator = T.alloc_var(T.float32, init=0)
                        for reduction_index in T.serial(inner):
                            accumulator += (
                                left[row, reduction_index]
                                * right[reduction_index, column]
                            )
                        output[row, column] = accumulator

        return main

    return build()


@lru_cache(maxsize=None)
def matmul_batched_fp32(groups: int, rows: int, columns: int, inner: int):
    """Strict FP32 SIMT batched MatMul used to validate graph semantics."""
    output_numel = groups * rows * columns

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            left: T.Tensor((groups, rows, inner), T.float32),
            right: T.Tensor((groups, inner, columns), T.float32),
            output: T.Tensor((groups, rows, columns), T.float32),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < output_numel:
                        group = output_index // (rows * columns)
                        within_group = output_index % (rows * columns)
                        row = within_group // columns
                        column = within_group % columns
                        accumulator = T.alloc_var(T.float32, init=0)
                        for reduction_index in T.serial(inner):
                            accumulator += (
                                left[group, row, reduction_index]
                                * right[group, reduction_index, column]
                            )
                        output[group, row, column] = accumulator

        return main

    return build()


@lru_cache(maxsize=None)
def convolution_nhwc(
    batch: int,
    height: int,
    width: int,
    channels: int,
    filters: int,
    kernel_height: int,
    kernel_width: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    pad_h: int,
    pad_w: int,
    dtype: str = "float32",
):
    if kernel_height != kernel_width or stride_h != stride_w or dilation_h != dilation_w or pad_h != pad_w:
        raise ValueError("The correctness prototype currently expects symmetric square convolutions")
    output_height = (height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1) // stride_h + 1
    output_width = (width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1) // stride_w + 1
    block_m, block_n, block_k = 64, 64, 32
    reduction = kernel_height * kernel_width * channels
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((batch, height, width, channels), value_dtype),
            kernel: T.Tensor(
                (kernel_height, kernel_width, channels, filters), value_dtype
            ),
            output: T.Tensor(
                (batch, output_height, output_width, filters), value_dtype
            ),
        ):
            with T.Kernel(
                T.ceildiv(filters, block_n),
                T.ceildiv(batch * output_height * output_width, block_m),
                threads=128,
            ) as (block_x, block_y):
                data_shared = T.alloc_shared((block_m, block_k), value_dtype)
                kernel_shared = T.alloc_shared((block_k, block_n), value_dtype)
                accumulator = T.alloc_fragment((block_m, block_n), T.float32)
                kernel_flat = T.Tensor((reduction, filters), value_dtype, kernel.data)
                output_flat = T.Tensor(
                    (batch * output_height * output_width, filters),
                    value_dtype,
                    output.data,
                )
                T.disable_warp_group_reg_alloc()
                T.clear(accumulator)
                for tile_k in T.Pipelined(T.ceildiv(reduction, block_k), num_stages=1):
                    T.im2col(
                        data,
                        data_shared,
                        block_y,
                        tile_k,
                        kernel_height,
                        stride_h,
                        dilation_h,
                        pad_h,
                    )
                    T.copy(kernel_flat[tile_k * block_k, block_x * block_n], kernel_shared)
                    T.gemm(data_shared, kernel_shared, accumulator)
                T.copy(accumulator, output_flat[block_y * block_m, block_x * block_n])

        return main

    return build(), (batch, output_height, output_width, filters)


@lru_cache(maxsize=None)
def convolution_nhwc_fp32(
    batch: int,
    height: int,
    width: int,
    channels: int,
    filters: int,
    kernel_height: int,
    kernel_width: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    pad_h: int,
    pad_w: int,
):
    """Strict FP32 direct convolution used to validate graph semantics."""
    output_height = (
        height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1
    ) // stride_h + 1
    output_width = (
        width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1
    ) // stride_w + 1
    output_numel = batch * output_height * output_width * filters

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            data: T.Tensor((batch, height, width, channels), T.float32),
            kernel: T.Tensor(
                (kernel_height, kernel_width, channels, filters), T.float32
            ),
            output: T.Tensor(
                (batch, output_height, output_width, filters), T.float32
            ),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < output_numel:
                        filter_index = output_index % filters
                        after_filter = output_index // filters
                        output_x = after_filter % output_width
                        after_x = after_filter // output_width
                        output_y = after_x % output_height
                        batch_index = after_x // output_height
                        accumulator = T.alloc_var(T.float32, init=0)
                        for kernel_y in T.serial(kernel_height):
                            input_y = output_y * stride_h + kernel_y * dilation_h - pad_h
                            if input_y >= 0 and input_y < height:
                                for kernel_x in T.serial(kernel_width):
                                    input_x = (
                                        output_x * stride_w
                                        + kernel_x * dilation_w
                                        - pad_w
                                    )
                                    if input_x >= 0 and input_x < width:
                                        for channel in T.serial(channels):
                                            accumulator += (
                                                data[
                                                    batch_index,
                                                    input_y,
                                                    input_x,
                                                    channel,
                                                ]
                                                * kernel[
                                                    kernel_y,
                                                    kernel_x,
                                                    channel,
                                                    filter_index,
                                                ]
                                            )
                        output[
                            batch_index, output_y, output_x, filter_index
                        ] = accumulator

        return main

    return build(), (batch, output_height, output_width, filters)


def _concat_indices(
    output_index,
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
    third_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    axis: int,
):
    output_strides = _strides(output_shape)
    first_strides = _strides(first_shape)
    second_strides = _strides(second_shape)
    third_strides = _strides(third_shape)
    first_index = 0
    second_index = 0
    third_index = 0
    for dim in range(len(output_shape)):
        coordinate = (
            output_index // output_strides[dim]
        ) % output_shape[dim]
        if dim == axis:
            first_coordinate = coordinate
            second_coordinate = coordinate - first_shape[axis]
            third_coordinate = coordinate - first_shape[axis] - second_shape[axis]
        else:
            first_coordinate = coordinate
            second_coordinate = coordinate
            third_coordinate = coordinate
        first_index += first_coordinate * first_strides[dim]
        second_index += second_coordinate * second_strides[dim]
        third_index += third_coordinate * third_strides[dim]
    return first_index, second_index, third_index


@lru_cache(maxsize=None)
def concat_three(
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
    third_shape: tuple[int, ...],
    axis: int,
    dtype: str = "float32",
):
    axis %= len(first_shape)
    output_shape = list(first_shape)
    output_shape[axis] = first_shape[axis] + second_shape[axis] + third_shape[axis]
    output_shape = tuple(output_shape)
    output_numel = _prod(output_shape)
    first_numel, second_numel, third_numel = map(
        _prod, (first_shape, second_shape, third_shape)
    )
    value_dtype = _value_dtype(dtype)

    @tilelang.jit(
        out_idx=[], target=TARGET, execution_backend="cython", pass_configs=PASS_CONFIGS
    )
    def build():
        @T.prim_func
        def main(
            first: T.Tensor((first_numel,), value_dtype),
            second: T.Tensor((second_numel,), value_dtype),
            third: T.Tensor((third_numel,), value_dtype),
            output: T.Tensor((output_numel,), value_dtype),
        ):
            with T.Kernel(T.ceildiv(output_numel, THREADS), threads=THREADS) as block:
                for thread in T.Parallel(THREADS):
                    output_index = block * THREADS + thread
                    if output_index < output_numel:
                        axis_coordinate = (
                            output_index // _strides(output_shape)[axis]
                        ) % output_shape[axis]
                        first_index, second_index, third_index = _concat_indices(
                            output_index,
                            first_shape,
                            second_shape,
                            third_shape,
                            output_shape,
                            axis,
                        )
                        if axis_coordinate < first_shape[axis]:
                            output[output_index] = first[first_index]
                        elif axis_coordinate < first_shape[axis] + second_shape[axis]:
                            output[output_index] = second[second_index]
                        else:
                            output[output_index] = third[third_index]

        return main

    return build(), output_shape


def invoke(kernel, *tensors) -> None:
    """Launch a Cython-backed TileLang kernel on contiguous tensor views."""
    kernel(*tensors, skip_tensor_validation=True)
