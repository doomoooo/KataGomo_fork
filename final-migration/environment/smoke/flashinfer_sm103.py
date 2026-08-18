#!/usr/bin/env python3

"""Static FlashInfer SM103/D32 contract smoke; compiles or launches nothing."""

import os
import pathlib

from cutlass import Float32
from flashinfer.cute_dsl.attention.config import AttentionConfig
from flashinfer.cute_dsl.attention.fusion.mask import MaskSpec


config = AttentionConfig(
    qk_acc_dtype=Float32,
    pv_acc_dtype=Float32,
    mma_tiler=(128, 128, 32),
    is_persistent=True,
    mask_spec=MaskSpec(),
)
config.can_implement(dtype_width=16)

assert os.environ.get("CUTE_DSL_ARCH") == "sm_103a"
assert os.environ.get("FLASHINFER_CUDA_ARCH_LIST") == "10.3a"
assert os.environ.get("FLASHINFER_NO_DOWNLOAD") == "1"
for variable in (
    "TRITON_PTXAS_PATH",
    "TRITON_PTXAS_BLACKWELL_PATH",
    "CUTE_DSL_PTXAS_PATH",
    "FLASHINFER_NVCC",
):
    path = pathlib.Path(os.environ[variable])
    assert path.is_file(), (variable, path)

assert config.qk_mma_tiler == (128, 128, 32)
assert config.pv_mma_tiler == (128, 32, 128)
assert config.cluster_shape_mn == (1, 1)

print("FLASHINFER_SM103_D32_STATIC_OK")
