#!/usr/bin/env python3
"""Stage03 B29 cuDNN OSS derivative with the unused AB12 data path removed.

This module is CPU-only.  It derives cumulatively from the exact accepted
projection-FP16 round-trip source produced by
:mod:`sm103_cudnn_oss_b29_roundtrip` and never edits the provider package or
the parent artifact.  The external ``__call__`` signature deliberately retains the
caller-owned AB12
tensor, and its shape still drives the persistent grid.  No AB12 pointer is
passed to the device kernel and no AB12 descriptor, shared/register fragment,
copy, store, or output pipeline remains.

The transform is fail-closed at three levels: the complete parent source SHA,
the SHA of every bounded replacement region, and a structural post-transform
audit.  GPU compilation and measurement are intentionally outside this file;
all resource changes below are hypotheses that Stage03 must verify with
NCU/NSYS before the variant can be retained.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import datetime
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import time
from types import ModuleType
from typing import Any, Final, Iterable

try:
    import sm103_cudnn_oss_b29_roundtrip as parent_candidate
except ModuleNotFoundError:
    from python import sm103_cudnn_oss_b29_roundtrip as parent_candidate


CANDIDATE_ID: Final = (
    "cudnn-fe-1_27-oss-dense-gemm-swiglu-proj-fp16-roundtrip-no-ab12-b29"
)
NUMERIC_SEMANTICS_ID: Final = parent_candidate.NUMERIC_SEMANTICS_ID
NUMERIC_SEMANTICS_SELECTOR: Final = "projection-fp16-roundtrip-no-ab12"
PARENT_CANDIDATE_ID: Final = parent_candidate.CANDIDATE_ID
PARENT_DERIVATIVE_SHA256: Final = (
    "99247c64d70a5f0b14ff75c08ba8d28fde31f159248e1e86c934cec6152777bc"
)
PARENT_GENERATOR_SHA256: Final = (
    "44d974d7653787cd3e24af5beb781ae71ff67b12337c50ca413785644e91b0bb"
)
EXPECTED_DERIVATIVE_SHA256: Final = (
    "f5a010550e7ba3581e7e1695e80e9a63f8203d57081390e67b4a62981c146081"
)
DERIVATIVE_FILENAME: Final = "dense_gemm_persistent_swiglu_no_ab12.py"
DERIVATIVE_PROVENANCE_FILENAME: Final = "no-ab12-provenance.json"

# Fixed B29 facts.  The parent NCU report measured 214,016 B dynamic shared
# memory.  Its four-stage, 128x32 FP16 AB12 ring occupies exactly 32,768 B.
# The transform holds the five A/B stages and two C stages fixed, so deleting
# that aligned field predicts an exact 32,768 B per-CTA reduction.
B29_ROWS: Final = 29 * 361
AB12_COLUMNS: Final = 2 * 1152
AB12_ELEMENT_BYTES: Final = 2
AB12_OUTPUT_BYTES_PER_STREAM: Final = (
    B29_ROWS * AB12_COLUMNS * AB12_ELEMENT_BYTES
)
PARENT_DYNAMIC_SHARED_MEMORY_BYTES: Final = 214_016
REMOVED_SHARED_MEMORY_BYTES_PER_CTA: Final = 32_768
EXPECTED_DYNAMIC_SHARED_MEMORY_BYTES: Final = (
    PARENT_DYNAMIC_SHARED_MEMORY_BYTES - REMOVED_SHARED_MEMORY_BYTES_PER_CTA
)


class NoAb12DerivativeError(RuntimeError):
    """Raised when the bounded transform or its static contract drifts."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class RegionPatch:
    """One exact source interval, selected by unique start/end anchors."""

    name: str
    start: str
    end: str
    expected_region_sha256: str
    replacement: str


@dataclass(frozen=True)
class ExactPatch:
    """One exact literal replacement whose old text must occur once."""

    name: str
    old: str
    replacement: str


SETUP_STAGES_REPLACEMENT = """\
# Setup fixed A/B/C stage counts.  Stage03 deliberately preserves the
        # finalized parent's five A/B stages; only the output scratch path is
        # removed, making the shared-memory delta deterministic.
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages()

        # Compute A/B/C shared memory layouts
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile_c,
            self.num_c_stage,
        )

        """

TMA_SETUP_REPLACEMENT = """\
# Setup the only output TMA descriptor: C.  ``ab12`` remains a host-side
        # ABI/shape argument and is intentionally absent from the device launch.
        c_cta_v_layout = cute.composition(cute.make_identity_layout(c.shape), self.epi_tile_c)
        epi_smem_layout_c = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout_c,
            c_cta_v_layout,
        )

        """

EPILOGUE_PARTITION_REPLACEMENT = """\
tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )

            tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(
                tma_atom_c,
                tCgC,
                epi_tile_c,
                sC,
            )
"""

EPILOGUE_TILE_SLICE_REPLACEMENT = """\
# Slice C to this persistent MMA tile.
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]
                """

OUTPUT_BUFFER_REPLACEMENT = """\
# C combines each pair of projection subtiles.
                    c_buffer = (num_prev_subtiles + subtile_idx // 2) % self.num_c_stage

                    """

TMEM_HELPER_REPLACEMENT = """\
    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        \"\"\"Partition TMEM and allocate two accumulator register fragments.\"\"\"
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.ab12_layout,
            self.ab12_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # Derive the register-side lane ownership from a compile-time identity
        # coordinate tensor.  This is the SM100 pattern used by Quack's GEMM
        # epilogue and replaces the parent's global-output partition without a
        # pointer, descriptor, load, or store.
        cAcc = cute.make_identity_tensor(
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1])
        )
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        fragment_shape = tTR_cAcc[None, None, None, 0, 0].shape
        tTR_rAcc = cute.make_rmem_tensor(fragment_shape, self.acc_dtype)
        tTR_rAcc1 = cute.make_rmem_tensor(fragment_shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_rAcc1

"""

SMEM_HELPER_REPLACEMENT = """\
    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        \"\"\"Partition only the C register-to-shared-memory copy.\"\"\"
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.ab12_layout,
            self.ab12_dtype,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

"""

GMEM_HELPER_REPLACEMENT = """\
    def epilog_gmem_copy_and_partition(
        self,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile_c: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        \"\"\"Partition only the C shared-to-global TMA store.\"\"\"
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile_c
        )
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

"""

STAGES_HELPER_REPLACEMENT = """\
    @staticmethod
    def _compute_stages() -> Tuple[int, int, int]:
        \"\"\"Return the finalized parent stages minus its four output stages.\"\"\"
        # ACC=2, A/B=5, C=2 are the exact finalized B29 parent values.  Keeping
        # A/B fixed prevents the freed output storage from being silently spent
        # on an extra input stage and makes the NCU resource hypothesis exact.
        return 2, 5, 2

"""


REGION_PATCHES: Final[tuple[RegionPatch, ...]] = (
    RegionPatch(
        "setup_stages_and_layouts",
        "# Setup A/B/AB12 stage count in shared memory and ACC stage count in tensor memory",
        "# Compute the number of tensor memory allocation columns",
        "5055fde093ab7acf0e08e6a648155bbc58573234ad9964b96f5d2a62c93bc88d",
        SETUP_STAGES_REPLACEMENT,
    ),
    RegionPatch(
        "output_tma_setup",
        "# Setup TMA store for AB12 and C",
        "# Compute grid size",
        "b65063ae06a3eebe8b973e93576e99e506bfad1f47021c823a616173f5bf04d1",
        TMA_SETUP_REPLACEMENT,
    ),
    RegionPatch(
        "epilogue_partition_setup",
        "tTR_rAB12 = None",
        "            #\n"
        "            # Persistent tile scheduling loop\n"
        "            #\n"
        "            tile_sched = utils.StaticPersistentTileScheduler.create(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())\n"
        "            work_tile = tile_sched.initial_work_tile_info()\n\n"
        "            acc_consumer_state",
        "b6e99ef093197d609189fddeb38f5e2987a3d9762a91e0ced53b28cde3566b22",
        EPILOGUE_PARTITION_REPLACEMENT,
    ),
    RegionPatch(
        "epilogue_output_tile_slice",
        "# Slice to per mma tile index\n"
        "                #\n"
        "                # ((ATOM_V, REST_V), EPI_M, EPI_N)\n"
        "                bSG_gAB12",
        "# Set tensor memory buffer for current tile\n"
        "                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)\n"
        "                tTR_tAcc",
        "f7244069f00263fda0bfe5237066c95f42646da011e0437f344927510111f016",
        EPILOGUE_TILE_SLICE_REPLACEMENT,
    ),
    RegionPatch(
        "output_buffer_selection",
        "# Store AB12 and C to shared memory",
        "cute.copy(\n                        tiled_copy_r2s,\n                        tRS_rC,",
        "7c2cdaa4e5bd7cf1fa75a3789b9de41ba7e5eb8e91d87bbe01ab480c90d8a598",
        OUTPUT_BUFFER_REPLACEMENT,
    ),
    RegionPatch(
        "tmem_helper",
        "    def epilog_tmem_copy_and_partition(",
        "    def epilog_smem_copy_and_partition(",
        "7f7f5be7f6c7f6166d4faa9eeb690810e41a1a1e11f16e522b4b1b7d4edd72ad",
        TMEM_HELPER_REPLACEMENT,
    ),
    RegionPatch(
        "smem_helper",
        "    def epilog_smem_copy_and_partition(",
        "    def epilog_gmem_copy_and_partition(",
        "775c929fa92282458aa1624141085def11c3c71152059643c7e0d35aeb14b4db",
        SMEM_HELPER_REPLACEMENT,
    ),
    RegionPatch(
        "gmem_helper",
        "    def epilog_gmem_copy_and_partition(",
        "    @staticmethod\n    def _compute_stages(",
        "881d77d1a9335fe5f6ee93e234092d1ed5e1ce7f6372702e00e8cfa0b7ecc517",
        GMEM_HELPER_REPLACEMENT,
    ),
    RegionPatch(
        "stages_helper",
        "    @staticmethod\n    def _compute_stages(",
        "    @staticmethod\n    def _compute_grid(",
        "8769552c15729621708914be34a476605feb4e0ed2e8227081a9883298ade33f",
        STAGES_HELPER_REPLACEMENT,
    ),
)


EXACT_PATCHES: Final[tuple[ExactPatch, ...]] = (
    ExactPatch(
        "remove_output_smem_size",
        "        ab12_smem_size = cute.cosize(self.ab12_smem_layout_staged.outer)\n"
        "        # ab12_smem_size: S<1,4,3> o 0 o ((8,16),(32,1),(1,8)):((32,256),(1,0),(0,4096))\n",
        "",
    ),
    ExactPatch(
        "remove_output_shared_field",
        "            # (EPI_TILE_M, EPI_TILE_N, STAGE)\n"
        "            sAB12: cute.struct.Align[\n"
        "                cute.struct.MemRange[\n"
        "                    self.ab12_dtype,\n"
        "                    ab12_smem_size,\n"
        "                ],\n"
        "                self.buffer_align_bytes,\n"
        "            ]\n"
        "            # (EPI_TILE_M, EPI_TILE_N, STAGE)\n\n",
        "            # (EPI_TILE_M, EPI_TILE_N, STAGE)\n",
    ),
    ExactPatch(
        "remove_output_launch_arguments",
        "            tma_atom_ab12,\n"
        "            tma_atom_c,\n"
        "            tma_tensor_ab12,\n"
        "            tma_tensor_c,",
        "            tma_atom_c,\n            tma_tensor_c,",
    ),
    ExactPatch(
        "remove_output_layout_launch_argument",
        "            self.b_smem_layout_staged,\n"
        "            self.ab12_smem_layout_staged,\n"
        "            self.c_smem_layout_staged,",
        "            self.b_smem_layout_staged,\n"
        "            self.c_smem_layout_staged,",
    ),
    ExactPatch(
        "remove_output_kernel_arguments",
        "        mB_nkl: cute.Tensor,\n"
        "        tma_atom_ab12: Optional[cute.CopyAtom],\n"
        "        tma_atom_c: Optional[cute.CopyAtom],\n"
        "        mAB12_mnl: cute.Tensor,\n"
        "        mC_mnl: cute.Tensor,",
        "        mB_nkl: cute.Tensor,\n"
        "        tma_atom_c: Optional[cute.CopyAtom],\n"
        "        mC_mnl: cute.Tensor,",
    ),
    ExactPatch(
        "remove_output_layout_kernel_argument",
        "        b_smem_layout_staged: cute.ComposedLayout,\n"
        "        ab12_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],\n"
        "        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],",
        "        b_smem_layout_staged: cute.ComposedLayout,\n"
        "        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],",
    ),
    ExactPatch(
        "remove_output_descriptor_prefetch",
        "            cpasync.prefetch_descriptor(tma_atom_ab12)\n",
        "",
    ),
    ExactPatch(
        "remove_output_smem_tensor",
        "        # (EPI_TILE_M, EPI_TILE_N, STAGE)\n"
        "        sAB12 = storage.sAB12.get_tensor(ab12_smem_layout_staged.outer, swizzle=ab12_smem_layout_staged.inner)\n"
        "        # (EPI_TILE_M, EPI_TILE_N, STAGE)\n",
        "        # (EPI_TILE_M, EPI_TILE_N, STAGE)\n",
    ),
    ExactPatch(
        "remove_output_global_tile",
        "        # (bM, bN, RestM, RestN, RestL)\n"
        "        gAB12_mnl = cute.local_tile(mAB12_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))\n",
        "",
    ),
    ExactPatch(
        "remove_output_mma_partition",
        "        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)\n"
        "        tCgAB12 = thr_mma.partition_C(gAB12_mnl)\n"
        "        tCgC = thr_mma.partition_C(gC_mnl)",
        "        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)\n"
        "        tCgC = thr_mma.partition_C(gC_mnl)",
    ),
    ExactPatch(
        "remove_output_tmem_helper_argument",
        "                tCtAcc_base,\n"
        "                tCgAB12,\n"
        "                tCgC,\n"
        "                epi_tile,\n"
        "                epi_tile_c,\n"
        "                use_2cta_instrs,",
        "                tCtAcc_base,\n"
        "                epi_tile,\n"
        "                use_2cta_instrs,",
    ),
    ExactPatch(
        "detach_c_pipeline_from_removed_output_stage_name",
        "                num_stages=self.num_ab12_stage,",
        "                num_stages=4,  # preserve the parent's C TMA flight depth",
    ),
    ExactPatch(
        "remove_output_global_grouping",
        "                bSG_gAB12 = cute.group_modes(bSG_gAB12, 1, cute.rank(bSG_gAB12))\n",
        "",
    ),
    ExactPatch(
        "remove_output_register_stores",
        "                    acc_vec0 = acc_vec0_ab12\n"
        "                    acc_vec1 = acc_vec1_ab12\n\n"
        "                    tRS_rAB12.store(acc_vec0)  # both of them are pure Gemm Output.\n"
        "                    tRS_rAB12_1.store(acc_vec1)\n"
        "                    tRS_rC.store(acc_vec_c)",
        "                    tRS_rC.store(acc_vec_c)",
    ),
    ExactPatch(
        "remove_output_tma_stores",
        "                        cute.copy(\n"
        "                            tma_atom_ab12,\n"
        "                            bSG_sAB12[(None, ab12_buffer0)],\n"
        "                            bSG_gAB12[(None, subtile_idx)],\n"
        "                        )\n"
        "                        cute.copy(\n"
        "                            tma_atom_ab12,\n"
        "                            bSG_sAB12[(None, ab12_buffer1)],\n"
        "                            bSG_gAB12[(None, subtile_idx + 1)],\n"
        "                        )\n\n",
        "",
    ),
)


ROUNDTRIP_MATH_BLOCK: Final = """\
                    acc_vec0 = acc_vec0 * alpha
                    acc_vec1 = acc_vec1 * alpha

                    # KataGo B29 Variant A: reproduce the legacy FP16 GEMM
                    # projection boundary before the otherwise unchanged
                    # FP32 fast-exp2/rcp SwiGLU epilogue. TensorSSA.to uses
                    # arith.truncf's default round-to-nearest-even mode.
                    acc_vec0_ab12 = acc_vec0.to(self.ab12_dtype)
                    acc_vec1_ab12 = acc_vec1.to(self.ab12_dtype)
                    acc_vec0 = acc_vec0_ab12.to(self.acc_dtype)
                    acc_vec1 = acc_vec1_ab12.to(self.acc_dtype)
                    # Use exp2 with log2(e) conversion since cute.math.exp is not available
                    # exp(x) = 2^(x * log2(e))
                    gate_rcp = (1 + cute.math.exp2(-1 * acc_vec1 * LOG2_E, True)).to(self.acc_dtype)

                    res = cute.make_rmem_tensor(gate_rcp.shape, cutlass.Float32)
                    res.store(gate_rcp)
                    for i in cutlass.range_constexpr(cute.size(res.shape)):
                        res[i] = cute.arch.rcp_approx(res[i])

                    gate = res.load()
                    gate = gate * acc_vec1

                    acc_vec_c = (acc_vec0 * gate).to(self.c_dtype)
"""


FORBIDDEN_DEVICE_DATA_PATH_TOKENS: Final = (
    "tma_atom_ab12",
    "tma_tensor_ab12",
    "mAB12_mnl",
    "ab12_smem_layout_staged",
    "ab12_smem_size",
    "num_ab12_stage",
    "sAB12",
    "gAB12",
    "tCgAB12",
    "rAB12",
    "ab12_buffer",
)


def _locate_patches(source: str) -> list[tuple[int, int, str, str]]:
    """Resolve every patch against the untouched parent and prove uniqueness."""

    located: list[tuple[int, int, str, str]] = []
    for patch in REGION_PATCHES:
        start_count = source.count(patch.start)
        end_count = source.count(patch.end)
        if start_count != 1 or end_count != 1:
            raise NoAb12DerivativeError(
                f"{patch.name} anchors must each occur once, got "
                f"start={start_count}, end={end_count}"
            )
        begin = source.index(patch.start)
        finish = source.index(patch.end, begin)
        if finish <= begin:
            raise NoAb12DerivativeError(f"{patch.name} region is inverted")
        region = source[begin:finish]
        actual = _sha256_text(region)
        if actual != patch.expected_region_sha256:
            raise NoAb12DerivativeError(
                f"{patch.name} region SHA-256 mismatch: expected "
                f"{patch.expected_region_sha256}, got {actual}"
            )
        located.append((begin, finish, patch.name, patch.replacement))

    for patch in EXACT_PATCHES:
        count = source.count(patch.old)
        if count != 1:
            raise NoAb12DerivativeError(
                f"{patch.name} exact context must occur once, got {count}"
            )
        begin = source.index(patch.old)
        located.append(
            (begin, begin + len(patch.old), patch.name, patch.replacement)
        )

    ordered = sorted(located)
    for left, right in zip(ordered, ordered[1:]):
        if left[1] > right[0]:
            raise NoAb12DerivativeError(
                f"patch regions overlap: {left[2]} and {right[2]}"
            )
    return ordered


def patch_context_counts(parent_source: bytes) -> dict[str, int]:
    """Return the verified one-count contract for all bounded transforms."""

    if _sha256_bytes(parent_source) != PARENT_DERIVATIVE_SHA256:
        raise NoAb12DerivativeError("finalized round-trip parent SHA-256 mismatch")
    try:
        source = parent_source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoAb12DerivativeError("round-trip parent is not UTF-8") from error
    located = _locate_patches(source)
    return {name: 1 for _, _, name, _ in located}


def _bounded_slice(source: str, start: str, end: str) -> str:
    if source.count(start) != 1 or source.count(end) != 1:
        raise NoAb12DerivativeError("invariant slice anchors drifted")
    begin = source.index(start)
    finish = source.index(end, begin)
    if finish <= begin:
        raise NoAb12DerivativeError("invariant slice is inverted")
    return source[begin:finish]


def audit_derivative(parent_source: bytes, derivative_source: bytes) -> dict[str, Any]:
    """Prove source-level data-path removal and preserved C computation."""

    parent = parent_source.decode("utf-8")
    derivative = derivative_source.decode("utf-8")

    forbidden_counts = {
        token: derivative.count(token) for token in FORBIDDEN_DEVICE_DATA_PATH_TOKENS
    }
    nonzero = {token: count for token, count in forbidden_counts.items() if count}
    if nonzero:
        raise NoAb12DerivativeError(
            "AB12 device data-path tokens remain: " + json.dumps(nonzero, sort_keys=True)
        )

    if parent.count(ROUNDTRIP_MATH_BLOCK) != 1:
        raise NoAb12DerivativeError("parent round-trip math block drifted")
    if derivative.count(ROUNDTRIP_MATH_BLOCK) != 1:
        raise NoAb12DerivativeError("derivative did not preserve round-trip C math")

    invariant_slices = {
        "tma_a_b_setup": (
            "# Setup TMA load for A",
            "# Setup TMA store for AB12 and C",
        ),
        "tcgen05_mma_mainloop": (
            "# Specialized MMA warp",
            "# Specialized epilogue warps",
        ),
    }
    invariant_sha256: dict[str, str] = {}
    for name, (start, end) in invariant_slices.items():
        parent_slice = _bounded_slice(parent, start, end)
        # The output-TMA heading was replaced, so the derivative A/B slice ends
        # at its new C-only heading while retaining the exact parent payload.
        derivative_end = (
            "# Setup the only output TMA descriptor: C."
            if name == "tma_a_b_setup"
            else end
        )
        derivative_slice = _bounded_slice(derivative, start, derivative_end)
        if parent_slice != derivative_slice:
            raise NoAb12DerivativeError(f"protected invariant changed: {name}")
        invariant_sha256[name] = _sha256_text(parent_slice)

    exact_token_counts: dict[str, int] = {}
    unchanged_tokens: Iterable[str] = (
        "cute.nvgpu.make_tiled_tma_atom_A(",
        "cute.nvgpu.make_tiled_tma_atom_B(",
        "cute.copy(\n                            tma_atom_c,",
        "tiled_mma.set(tcgen05.Field.ACCUMULATE, False)",
        "tiled_mma.set(tcgen05.Field.ACCUMULATE, True)",
        "tcgen05.make_tmem_copy(",
        "cute.arch.alloc_tmem(",
        "cute.arch.dealloc_tmem(",
    )
    for token in unchanged_tokens:
        parent_count = parent.count(token)
        derivative_count = derivative.count(token)
        if parent_count == 0 or derivative_count != parent_count:
            raise NoAb12DerivativeError(
                f"protected operation count changed for {token!r}: "
                f"parent={parent_count}, derivative={derivative_count}"
            )
        exact_token_counts[token] = derivative_count

    # One argument belongs to __call__ (the preserved external ABI), and one
    # belongs to the host-only grid helper (the preserved shape contract).
    if derivative.count("        ab12: cute.Tensor,") != 2:
        raise NoAb12DerivativeError("AB12 compatibility argument changed")
    if derivative.count("self._compute_grid(ab12,") != 1:
        raise NoAb12DerivativeError("AB12 compatibility shape path changed")
    if derivative.count("cpasync.make_tiled_tma_atom(") != 1:
        raise NoAb12DerivativeError("expected exactly one output TMA descriptor (C)")
    if derivative.count("cpasync.prefetch_descriptor(tma_atom_c)") != 1:
        raise NoAb12DerivativeError("C descriptor prefetch changed")
    if derivative.count("tRS_rC.store(acc_vec_c)") != 1:
        raise NoAb12DerivativeError("C register store changed")
    if derivative.count(
        "num_stages=4,  # preserve the parent's C TMA flight depth"
    ) != 1:
        raise NoAb12DerivativeError("C store pipeline flight depth changed")

    return {
        "parent_sha256": _sha256_bytes(parent_source),
        "derivative_sha256": _sha256_bytes(derivative_source),
        "patch_contexts": patch_context_counts(parent_source),
        "forbidden_device_data_path_counts": forbidden_counts,
        "roundtrip_math_block_sha256": _sha256_text(ROUNDTRIP_MATH_BLOCK),
        "protected_invariant_sha256": invariant_sha256,
        "protected_operation_counts": exact_token_counts,
        "abi_shape_contract": {
            "ab12_call_argument_count": 1,
            "ab12_grid_shape_use_count": 1,
            "ab12_device_kernel_argument_count": 0,
        },
    }


def derive_no_ab12_source(parent_source: bytes) -> bytes:
    """Apply all non-overlapping exact patches to the finalized parent bytes."""

    actual_parent_sha = _sha256_bytes(parent_source)
    if actual_parent_sha != PARENT_DERIVATIVE_SHA256:
        raise NoAb12DerivativeError(
            "finalized round-trip parent SHA-256 mismatch: expected "
            f"{PARENT_DERIVATIVE_SHA256}, got {actual_parent_sha}"
        )
    try:
        source = parent_source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NoAb12DerivativeError("round-trip parent is not UTF-8") from error

    located = _locate_patches(source)
    for begin, finish, _name, replacement in reversed(located):
        source = source[:begin] + replacement + source[finish:]
    derivative = source.encode("utf-8")
    audit_derivative(parent_source, derivative)
    if EXPECTED_DERIVATIVE_SHA256 != "TO_BE_FIXED_AFTER_PATCH_AUDIT":
        actual = _sha256_bytes(derivative)
        if actual != EXPECTED_DERIVATIVE_SHA256:
            raise NoAb12DerivativeError(
                "no-AB12 derivative SHA-256 mismatch: expected "
                f"{EXPECTED_DERIVATIVE_SHA256}, got {actual}"
            )
    return derivative


def resource_hypothesis() -> dict[str, Any]:
    """Return only source-derived predictions; GPU counters remain required."""

    return {
        "status": "predicted_requires_ncu_nsys_validation",
        "fixed_problem": {
            "batch": 29,
            "rows": B29_ROWS,
            "projection_columns": AB12_COLUMNS,
            "dtype": "FP16",
            "streams": 2,
        },
        "global_memory": {
            "removed_output_bytes_per_stream_launch": AB12_OUTPUT_BYTES_PER_STREAM,
            "removed_output_bytes_per_dual_stream_round": 2
            * AB12_OUTPUT_BYTES_PER_STREAM,
            "removed_output_tma_store_path": "all AB12 S2G stores",
            "remaining_output_tma_store_path": "C only",
            "caller_allocation": "retained for C ABI compatibility",
        },
        "shared_memory": {
            "parent_dynamic_bytes_per_cta_ncu": PARENT_DYNAMIC_SHARED_MEMORY_BYTES,
            "removed_bytes_per_cta": REMOVED_SHARED_MEMORY_BYTES_PER_CTA,
            "expected_dynamic_bytes_per_cta": EXPECTED_DYNAMIC_SHARED_MEMORY_BYTES,
            "removed_output_stages": 4,
            "preserved_ab_input_stages": 5,
            "preserved_c_output_stages": 2,
        },
        "descriptors": {
            "removed_output_tma_descriptors": 1,
            "removed_output_descriptor_prefetches": 1,
            "remaining_descriptors": ["A load", "B load", "C store"],
        },
        "registers": {
            "removed_explicit_fp16_output_fragments_per_epilogue_thread": 2,
            "compiled_register_delta": "measure with NCU; do not infer",
        },
        "synchronization": {
            "removed_output_only_barriers": 0,
            "reason": "parent shares both epilogue barriers with C; both remain",
            "store_pipeline_stages": {"parent": 4, "candidate": 4},
        },
    }


@dataclass(frozen=True)
class DerivativeEvidence:
    candidate_id: str
    numeric_semantics_id: str
    parent_candidate_id: str
    parent_derivative_sha256: str
    parent_generator_sha256: str
    parent_upstream_sha256: str
    parent_patch_spec_sha256: str
    removal_patch_spec_sha256: str
    derivative_sha256: str
    upstream_project: str
    upstream_distribution: str
    upstream_version: str
    upstream_license: str
    site_packages_modified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _patch_spec_sha256() -> str:
    payload = {
        "regions": [asdict(patch) for patch in REGION_PATCHES],
        "exact": [asdict(patch) for patch in EXACT_PATCHES],
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


PATCH_SPEC_SHA256: Final = _patch_spec_sha256()


def inspect_derivative() -> tuple[bytes, DerivativeEvidence, dict[str, Any]]:
    """Derive the Stage03 source and return complete chained provenance."""

    parent_source, parent_evidence = parent_candidate.inspect_derivative()
    if parent_evidence.derivative_sha256 != PARENT_DERIVATIVE_SHA256:
        raise NoAb12DerivativeError("round-trip evidence/source identity mismatch")
    derivative = derive_no_ab12_source(parent_source)
    audit = audit_derivative(parent_source, derivative)
    evidence = DerivativeEvidence(
        candidate_id=CANDIDATE_ID,
        numeric_semantics_id=NUMERIC_SEMANTICS_ID,
        parent_candidate_id=PARENT_CANDIDATE_ID,
        parent_derivative_sha256=PARENT_DERIVATIVE_SHA256,
        parent_generator_sha256=PARENT_GENERATOR_SHA256,
        parent_upstream_sha256=parent_evidence.upstream_sha256,
        parent_patch_spec_sha256=parent_evidence.patch_spec_sha256,
        removal_patch_spec_sha256=PATCH_SPEC_SHA256,
        derivative_sha256=_sha256_bytes(derivative),
        upstream_project=parent_evidence.upstream_project,
        upstream_distribution=parent_evidence.upstream_distribution,
        upstream_version=parent_evidence.upstream_version,
        upstream_license=parent_evidence.upstream_license,
    )
    return derivative, evidence, audit


def materialize_derivative(
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, DerivativeEvidence]:
    """Materialize outside site-packages and refuse mismatched overwrites."""

    derivative, evidence, audit = inspect_derivative()
    resolved = output_dir.resolve()
    if resolved == pathlib.Path("/"):
        raise NoAb12DerivativeError("unsafe derivative output directory")
    parent_path = pathlib.Path(
        parent_candidate.inspect_derivative()[1].upstream_installed_path
    )
    site_packages_root = parent_path.resolve()
    for _ in pathlib.PurePosixPath(
        parent_candidate.UPSTREAM_KERNEL_RELATIVE_PATH
    ).parts:
        site_packages_root = site_packages_root.parent
    if resolved == site_packages_root or site_packages_root in resolved.parents:
        raise NoAb12DerivativeError("output must remain outside site-packages")

    resolved.mkdir(parents=True, exist_ok=True)
    source_path = resolved / DERIVATIVE_FILENAME
    provenance_path = resolved / DERIVATIVE_PROVENANCE_FILENAME
    provenance = {
        "evidence": evidence.to_dict(),
        "audit": audit,
        "resource_hypothesis": resource_hypothesis(),
        "production_ready": False,
        "gpu_validation": "not_run",
    }
    payloads = (
        (source_path, derivative),
        (
            provenance_path,
            (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        ),
    )
    for path, expected in payloads:
        if path.exists() and path.read_bytes() != expected:
            raise NoAb12DerivativeError(
                f"refusing to overwrite mismatched derivative artifact: {path}"
            )
        if not path.exists():
            path.write_bytes(expected)
    return source_path, provenance_path, evidence


def load_derivative_kernel_class(
    output_dir: pathlib.Path,
) -> tuple[type[Any], DerivativeEvidence, pathlib.Path, pathlib.Path]:
    """Materialize and import the verified derivative for explicit GPU export."""

    source_path, provenance_path, evidence = materialize_derivative(output_dir)
    module_name = "katago_sm103_cudnn_oss_swiglu_no_ab12"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise NoAb12DerivativeError("could not construct derivative module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    kernel_class = getattr(module, "PersistentDenseGemmKernel", None)
    if not isinstance(kernel_class, type):
        raise NoAb12DerivativeError("derivative kernel class is missing")
    return kernel_class, evidence, source_path, provenance_path


def numeric_semantics() -> dict[str, Any]:
    semantics = dict(parent_candidate.numeric_semantics())
    semantics["id"] = NUMERIC_SEMANTICS_ID
    semantics["ab12"] = (
        "caller pointer/shape retained for native ABI compatibility; device "
        "kernel performs no AB12 load, store, descriptor, or staging"
    )
    semantics["changed_factors"] = [
        *semantics["changed_factors"],
        "remove unused AB12 device output data path",
    ]
    return semantics


def build_candidate_manifest(
    provider: Any | None = None,
    baseline_path: pathlib.Path | None = None,
    repo_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build the cumulative, nonproduction Stage03 candidate contract."""

    if provider is None:
        provider = parent_candidate.upstream_candidate.inspect_installed_provider()
    base = parent_candidate.validate_candidate_manifest(
        parent_candidate.build_candidate_manifest(
            provider=provider,
            baseline_path=baseline_path,
            repo_root=repo_root,
        )
    )
    _, derivative, audit = inspect_derivative()
    base["kind"] = "katago-sm103-b29-cudnn-oss-no-ab12-candidate"
    base["candidate_id"] = CANDIDATE_ID
    base["operation"]["numeric_semantics"] = numeric_semantics()
    base["static_support"]["status"] = "cpu_no_ab12_derivative_verified"
    base["static_support"]["derivative"] = derivative.to_dict()
    base["static_support"]["no_ab12_audit"] = audit
    base["static_support"]["resource_hypothesis"] = resource_hypothesis()
    base["correctness"]["required_reference"] = (
        "Variant-A FP16 projection-roundtrip C output; AB12 must remain "
        "untouched because it is ABI-only"
    )
    base["benchmark"]["artifact_runtime"] = "native C ABI only"
    base["production_ready"] = False
    return base


def validate_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on every field, including complete chained provenance."""

    if manifest.get("kind") != "katago-sm103-b29-cudnn-oss-no-ab12-candidate":
        raise NoAb12DerivativeError("unexpected no-AB12 manifest kind")
    if manifest.get("candidate_id") != CANDIDATE_ID:
        raise NoAb12DerivativeError("no-AB12 candidate identity changed")
    expected = build_candidate_manifest()
    if manifest != expected:
        raise NoAb12DerivativeError("full no-AB12 manifest identity changed")
    return manifest


projection_fp16_roundtrip_reference = (
    parent_candidate.projection_fp16_roundtrip_reference
)
strengthen_probe_signal = parent_candidate.strengthen_probe_signal


def _tensor_error_metrics(
    torch: ModuleType, actual: Any, reference: Any
) -> dict[str, float]:
    difference = actual.float() - reference.float()
    absolute = difference.abs()
    return {
        "max_abs_error": float(absolute.max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_rel_error": float(
            (absolute / reference.float().abs().clamp_min(1.0e-3)).max().item()
        ),
    }


def validate_correctness_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply Variant-A's tight C gate without requiring an AB12 output."""

    signal = summary.get("reference_signal")
    output = summary.get("output")
    if not isinstance(signal, dict) or not isinstance(output, dict):
        raise NoAb12DerivativeError("no-AB12 correctness fields are missing")
    limits = parent_candidate.CORRECTNESS_LIMITS
    checks = {
        "reference_max_abs_signal": signal.get("max_abs", 0.0)
        >= limits["reference_max_abs_minimum"],
        "reference_rms_signal": signal.get("rms", 0.0)
        >= limits["reference_rms_minimum"],
        "output_max_abs": output.get("max_abs_error", math.inf)
        <= limits["output_max_abs_maximum"],
        "output_rmse": output.get("rmse", math.inf)
        <= limits["output_rmse_maximum"],
        "ab12_abi_only": summary.get("ab12_untouched") is True,
    }
    numeric_values = (
        signal.get("max_abs"),
        signal.get("rms"),
        output.get("max_abs_error"),
        output.get("rmse"),
    )
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in numeric_values
    ):
        raise NoAb12DerivativeError("correctness summary contains non-finite values")
    summary["limits"] = {
        key: value for key, value in limits.items() if key.startswith("output_")
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    if not summary["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise NoAb12DerivativeError(
            "no-AB12 tight correctness failed: " + ", ".join(failed)
        )
    return summary


def build_gpu_correctness_summary(
    torch: ModuleType,
    *,
    actual_output: Any,
    reference_output: Any,
    ab12_untouched: bool,
) -> dict[str, Any]:
    reference_float = reference_output.float()
    summary = {
        "reference_signal": {
            "max_abs": float(reference_float.abs().max().item()),
            "rms": float(reference_float.square().mean().sqrt().item()),
        },
        "output": _tensor_error_metrics(torch, actual_output, reference_output),
        "ab12_untouched": bool(ab12_untouched),
    }
    return validate_correctness_summary(summary)


def _load_native_library(library_path: pathlib.Path) -> Any:
    library = ctypes.CDLL(str(library_path.resolve()), mode=ctypes.RTLD_LOCAL)
    library.katagoCudnnOssB29Create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.katagoCudnnOssB29Create.restype = ctypes.c_void_p
    library.katagoCudnnOssB29Launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.katagoCudnnOssB29Launch.restype = ctypes.c_int32
    library.katagoCudnnOssB29Destroy.argtypes = [ctypes.c_void_p]
    library.katagoCudnnOssB29Destroy.restype = None
    return library


def authenticate_aot_library(library_path: pathlib.Path) -> dict[str, Any]:
    resolved = library_path.resolve()
    manifest_path = resolved.parent / "aot-manifest.json"
    if not resolved.is_file() or not manifest_path.is_file():
        raise NoAb12DerivativeError("AOT library or sibling manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NoAb12DerivativeError("AOT manifest is unreadable") from error
    _, evidence, _ = inspect_derivative()
    bridge = manifest.get("artifacts", {}).get("bridge_shared_library", {})
    derivative = manifest.get("derivative", {}).get("evidence")
    launch = manifest.get("launch_validation", {})
    checks = {
        "candidate_id": manifest.get("candidate_id") == CANDIDATE_ID,
        "numeric_semantics_selector": manifest.get("numeric_semantics_selector")
        == NUMERIC_SEMANTICS_SELECTOR,
        "compile_target": manifest.get("compile_target") == "sm_103a",
        "library_path": pathlib.Path(bridge.get("path", "")).resolve() == resolved,
        "library_sha256": bridge.get("sha256") == _sha256_bytes(resolved.read_bytes()),
        "derivative_evidence": derivative == evidence.to_dict(),
        "tight_correctness": launch.get("status") == "passed"
        and launch.get("tight_correctness", {}).get("passed") is True
        and launch.get("tight_correctness", {}).get("ab12_untouched") is True,
        "nonproduction": manifest.get("production_ready") is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise NoAb12DerivativeError(
            "AOT library authentication failed: " + ", ".join(failed)
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "library_path": str(resolved),
        "library_sha256": _sha256_bytes(resolved.read_bytes()),
        "checks": checks,
    }


def _timing_summary(
    samples: list[float], iterations: int, streams: int
) -> dict[str, Any]:
    iteration_ms = [sample * 1000.0 / iterations for sample in samples]
    median_ms = statistics.median(iteration_ms)
    return {
        "cuda_event_seconds_samples": samples,
        "milliseconds_per_concurrent_iteration_samples": iteration_ms,
        "median_stream_call_wall_milliseconds": median_ms,
        "median_effective_milliseconds_per_call": median_ms / streams,
        "calls_per_iteration": streams,
        "calls_per_second": 1000.0 * streams / median_ms,
        "relative_spread": (max(iteration_ms) - min(iteration_ms)) / median_ms,
    }


def benchmark_aot(
    *,
    allow_gpu: bool,
    device: int,
    library_path: pathlib.Path,
    warmup: int = 100,
    iterations: int = 1000,
    repeats: int = 5,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Tight-check and time the standalone native C ABI on S1 and S2."""

    if not allow_gpu:
        raise NoAb12DerivativeError(
            "AOT benchmark requires the explicit --allow-gpu acknowledgement"
        )
    for name, value in (
        ("device", device),
        ("warmup", warmup),
        ("iterations", iterations),
        ("repeats", repeats),
        ("seed", seed),
    ):
        if type(value) is not int:  # noqa: E721
            raise NoAb12DerivativeError(f"{name} must be an integer")
    if device < 0 or warmup < 1 or iterations < 1 or repeats < 1:
        raise NoAb12DerivativeError("invalid device or timing count")
    authentication = authenticate_aot_library(library_path)

    import torch

    if tuple(torch.cuda.get_device_capability(device)) != (10, 3):
        raise NoAb12DerivativeError("AOT benchmark requires exact SM103")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = parent_candidate.upstream_candidate._allocate_gpu_benchmark_inputs(
        torch, device, seed
    )
    signal_scaling = strengthen_probe_signal(torch, tensors)
    problem = tensors["problem"]

    def allocate_ab12() -> Any:
        return torch.empty_strided(
            (problem.m, problem.n_packed, 1),
            (problem.n_packed, 1, problem.m * problem.n_packed),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )

    def allocate_c() -> Any:
        return torch.empty_strided(
            (problem.m, problem.n_output, 1),
            (problem.n_output, 1, problem.m * problem.n_output),
            dtype=torch.float16,
            device=f"cuda:{device}",
        )

    ab12 = [allocate_ab12(), allocate_ab12()]
    outputs = [allocate_c(), allocate_c()]
    library = _load_native_library(library_path)
    status = ctypes.c_int32(-999)
    context = library.katagoCudnnOssB29Create(device, ctypes.byref(status))
    if not context or status.value != 0:
        raise NoAb12DerivativeError(
            f"C ABI context creation failed with status {status.value}"
        )

    def launch(index: int) -> Any:
        stream = torch.cuda.current_stream(device)
        launch_status = library.katagoCudnnOssB29Launch(
            context,
            ctypes.c_void_p(tensors["a_tensor"].data_ptr()),
            ctypes.c_void_p(tensors["b_tensor"].data_ptr()),
            ctypes.c_void_p(ab12[index].data_ptr()),
            ctypes.c_void_p(outputs[index].data_ptr()),
            ctypes.c_float(1.0),
            ctypes.c_void_p(stream.cuda_stream),
            problem.m,
            problem.k,
            problem.n_packed,
            problem.n_output,
            1,
        )
        if launch_status != 0:
            raise NoAb12DerivativeError(
                f"C ABI launch failed with status {launch_status}"
            )
        return outputs[index]

    try:
        ab12[0].fill_(float("nan"))
        outputs[0].fill_(float("nan"))
        launch(0)
        torch.cuda.synchronize(device)
        reference = projection_fp16_roundtrip_reference(
            torch,
            tensors["input_2d"],
            tensors["linear1"],
            tensors["linear_gate"],
        )
        correctness = build_gpu_correctness_summary(
            torch,
            actual_output=outputs[0][:, :, 0],
            reference_output=reference,
            ab12_untouched=bool(torch.isnan(ab12[0]).all().item()),
        )

        def measure(stream_count: int) -> dict[str, Any]:
            streams = [torch.cuda.Stream(device=device) for _ in range(stream_count)]
            coordinator = torch.cuda.Stream(device=device)
            live: list[Any] = [None] * stream_count
            for _ in range(warmup):
                for index, stream in enumerate(streams):
                    with torch.cuda.stream(stream):
                        live[index] = launch(index)
            torch.cuda.synchronize(device)
            samples: list[float] = []
            for _ in range(repeats):
                torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if stream_count == 1:
                    with torch.cuda.stream(streams[0]):
                        start.record()
                        for _ in range(iterations):
                            live[0] = launch(0)
                        end.record()
                else:
                    done = [torch.cuda.Event() for _ in streams]
                    with torch.cuda.stream(coordinator):
                        start.record()
                    for stream in streams:
                        stream.wait_event(start)
                    for _ in range(iterations):
                        for index, stream in enumerate(streams):
                            with torch.cuda.stream(stream):
                                live[index] = launch(index)
                    for event, stream in zip(done, streams, strict=True):
                        event.record(stream)
                    with torch.cuda.stream(coordinator):
                        for event in done:
                            coordinator.wait_event(event)
                        end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 1000.0)
            return _timing_summary(samples, iterations, stream_count)

        timings = {f"s{count}": measure(count) for count in (1, 2)}
    finally:
        try:
            torch.cuda.synchronize(device)
        finally:
            library.katagoCudnnOssB29Destroy(context)

    return {
        "schema": 1,
        "kind": "katago-sm103-b29-cudnn-oss-no-ab12-aot-timing",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "candidate_id": CANDIDATE_ID,
        "numeric_semantics_selector": NUMERIC_SEMANTICS_SELECTOR,
        "device": {
            "ordinal": device,
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "library": {
            "path": str(library_path.resolve()),
            "sha256": _sha256_bytes(library_path.read_bytes()),
            "bytes": library_path.stat().st_size,
        },
        "authentication": authentication,
        "method": {
            "clock": "CUDA coordinator-stream events spanning all worker streams",
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "repeats": repeats,
            "allocation": "all caller-owned buffers preallocated before timing",
            "s2": "two independent streams with per-stream ABI-only AB12 and C",
        },
        "correctness": {
            "status": "passed",
            "signal_scaling": signal_scaling,
            "tight_correctness": correctness,
        },
        "timings": timings,
        "production_ready": False,
    }


def stage_hypothesis() -> dict[str, Any]:
    derivative, evidence, audit = inspect_derivative()
    return {
        "kind": "katago-sm103-b29-stage03-no-ab12-hypothesis",
        "candidate_id": CANDIDATE_ID,
        "parent_candidate_id": PARENT_CANDIDATE_ID,
        "numeric_semantics_id": NUMERIC_SEMANTICS_ID,
        "derivative": evidence.to_dict(),
        "audit": audit,
        "resource_hypothesis": resource_hypothesis(),
        "source_bytes": len(derivative),
        "production_ready": False,
        "next_gate": "AOT compile, local NCU/NSYS, S2 whole-graph long test, precision",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--aot-benchmark", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--library", type=pathlib.Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--benchmark-output", type=pathlib.Path)
    args = parser.parse_args()
    if args.aot_benchmark:
        if args.library is None:
            raise NoAb12DerivativeError("--aot-benchmark requires --library")
        if args.output is not None:
            raise NoAb12DerivativeError(
                "--output materialization is invalid with --aot-benchmark"
            )
        payload = benchmark_aot(
            allow_gpu=args.allow_gpu,
            device=args.device,
            library_path=args.library,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    elif args.output is None:
        if args.allow_gpu or args.library is not None or args.benchmark_output:
            raise NoAb12DerivativeError(
                "GPU/benchmark options require --aot-benchmark"
            )
        payload = stage_hypothesis()
    else:
        if args.allow_gpu or args.library is not None or args.benchmark_output:
            raise NoAb12DerivativeError(
                "GPU/benchmark options require --aot-benchmark"
            )
        source, provenance, evidence = materialize_derivative(args.output)
        payload = {
            "source": str(source),
            "provenance": str(provenance),
            "evidence": evidence.to_dict(),
        }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.benchmark_output is not None:
        args.benchmark_output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
