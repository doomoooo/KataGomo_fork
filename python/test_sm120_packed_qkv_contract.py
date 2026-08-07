"""CPU-only layout checks for the exact-19x19 packed QKV pipeline."""

import itertools


SEQ = 361
HEADS = 12
DIM = 32
PLANE = HEADS * DIM
PACKED_ROW = 3 * PLANE


def packed_offset(batch: int, xy: int, head: int, dim: int, plane: int) -> int:
    return (batch * SEQ + xy) * PACKED_ROW + plane * PLANE + head * DIM + dim


def fa4_dynamic_stride_offset(
    batch: int, xy: int, head: int, dim: int, plane: int,
) -> int:
    base = plane * PLANE
    return base + batch * (SEQ * PACKED_ROW) + xy * PACKED_ROW + head * DIM + dim


def test_fa4_dynamic_strides_address_every_packed_qkv_value_b1_b32() -> None:
    for batch_size in range(1, 33):
        batches = sorted({0, batch_size - 1})
        for batch, xy, head, dim, plane in itertools.product(
            batches, (0, 1, SEQ - 1), (0, HEADS - 1), (0, 1, DIM - 1), range(3),
        ):
            assert fa4_dynamic_stride_offset(batch, xy, head, dim, plane) == packed_offset(
                batch, xy, head, dim, plane,
            )


def test_packed_rope_half2_addresses_only_q_and_k() -> None:
    packed_pairs_per_row = PACKED_ROW // 2
    for batch_size in range(1, 33):
        batches = sorted({0, batch_size - 1})
        for batch, xy, hp in itertools.product(
            batches, (0, 1, SEQ - 1), (0, 1, HEADS * (DIM // 2) - 1),
        ):
            token = batch * SEQ + xy
            q_half = 2 * (token * packed_pairs_per_row + hp)
            # kBuf is passed at packed base + 384 half values.
            k_half = PLANE + 2 * (token * packed_pairs_per_row + hp)
            assert q_half == packed_offset(batch, xy, hp // 16, 2 * (hp % 16), 0)
            assert k_half == packed_offset(batch, xy, hp // 16, 2 * (hp % 16), 1)
            row_start = token * PACKED_ROW
            assert q_half < row_start + PLANE
            assert row_start + PLANE <= k_half < row_start + 2 * PLANE
            assert k_half + 1 < row_start + 2 * PLANE


def test_packed_planes_cover_each_row_without_overlap() -> None:
    for plane in range(3):
        offsets = {
            packed_offset(0, 0, head, dim, plane)
            for head, dim in itertools.product(range(HEADS), range(DIM))
        }
        assert offsets == set(range(plane * PLANE, (plane + 1) * PLANE))
