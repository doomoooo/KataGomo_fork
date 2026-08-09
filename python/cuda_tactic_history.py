#!/usr/bin/env python3
"""Machine-readable closure contract for historically positive CUDA tactics.

An entry belongs here only when the archived optimization history recorded a
real positive result and the route remained numerically valid.  A route may be
default-off because a later dual-stream phase test regressed; it is still a
required scan coordinate when its local/S1 mechanism was positive.  Strictly
negative, accuracy-failed, or never-performance-measured experiments are not
listed. Byte-identical output is the strongest available precision evidence,
and later reversion does not erase an earlier real positive measurement.

Every listed entry must close four links for every materialized exact batch:
backend source, a distinct scan candidate, a runtime activation marker, and a
non-empty runtime config that plan apply can reproduce verbatim.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any


def _entry(
    history_id: str,
    family: str,
    evidence: str,
    backend_file: str,
    backend_symbol: str,
    *,
    backend_symbols: Sequence[str] | None = None,
    candidate_id: str | None = None,
    candidate_prefix: str | None = None,
    config: Mapping[str, object] | None = None,
    fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selectors = sum(value is not None for value in (candidate_id, candidate_prefix, config, fields))
    if selectors == 0:
        raise ValueError(f"history entry {history_id} has no candidate selector")
    return {
        "history_id": history_id,
        "family": family,
        "evidence": evidence,
        "precision_status": "passed",
        "backend": {
            "file": backend_file,
            "symbol": backend_symbol,
            **({"symbols": list(backend_symbols)} if backend_symbols else {}),
        },
        "match": {
            **({"candidate_id": candidate_id} if candidate_id is not None else {}),
            **({"candidate_prefix": candidate_prefix} if candidate_prefix is not None else {}),
            **({"config": dict(config)} if config is not None else {}),
            **({"fields": dict(fields)} if fields is not None else {}),
        },
    }


SM89_POSITIVE_HISTORY = (
    _entry("sm89-stage2-wide-projection-bundle", "wide_projection", "4090/HISTORY stage2 wide-QKV + wide-FFN bundle +4.8%; no unsupported individual attribution", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "useWideQKV_, useWideFFN_", candidate_id="wide-projection-both"),
    _entry("sm89-stage3-residual-beta1", "fused_residual", "4090/HISTORY stage3", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "useFusedResidual", candidate_id="fused_residual-on"),
    _entry("sm89-stage4-rms-warps4", "rmsnorm", "4090/HISTORY stage4", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89RMSNormNHWCHalfKernel<4>", candidate_id="rmsnorm-warps4"),
    _entry("sm89-stage39-rms-warps8", "rmsnorm", "4090/HISTORY stage39 local -0.65%", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89RMSNormNHWCHalfKernel<8>", candidate_id="rmsnorm-warps8"),
    _entry("sm89-stage5-fused-qk-rope", "qkv_rope", "4090/HISTORY stage5/13", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "useFusedQKRoPE", candidate_id="qkv-rope-fused"),
    _entry("sm89-stage14-precomputed-rope", "qkv_rope", "4090/INTRINSIC-S2-AUDIT stage14 local -3.6%", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "usePrecomputedQKRoPE", candidate_id="qkv-rope-precomputed"),
    _entry("sm89-stage15-group2-rope", "qkv_rope", "4090/INTRINSIC-S2-AUDIT stage15 local -4.5%", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "ropeBatchGroup", candidate_id="qkv-rope-group-2"),
    _entry("sm89-stage16-qkv-rope-epilogue", "qkv_rope", "4090/HISTORY stage16", "cpp/neuralnet/cudabackend_sm89_qkv_rope_gemm.cu", "Sm89QKVRoPEGemm", candidate_id="qkv-rope-gemm-epilogue"),
    _entry("sm89-stage30-qkv-rope-epilogue-table", "qkv_rope", "4090/HISTORY stage30 local epilogue positive; model-lifetime float2 table", "cpp/neuralnet/cudabackend_sm89_qkv_rope_gemm.cu", "cosSinTable[(size_t)xy", candidate_id="qkv-rope-gemm-epilogue-precomputed"),
    _entry("sm89-stage64-split-qkv-rope", "qkv_rope", "4090/HISTORY stage64 S1 +10.77%", "cpp/neuralnet/cudabackend_sm89_qkv_rope_gemm.cu", "plainFp32Params", candidate_id="qkv-rope-gemm-split-v0"),
    _entry("sm89-stage65-native-half-qkv", "qkv_rope", "4090/HISTORY stage65 local -0.34%", "cpp/neuralnet/cudabackend_sm89_qkv_rope_gemm.cu", "plainVariant == 1", candidate_id="qkv-rope-gemm-split-v1"),
    _entry("sm89-stage6-exact-mask-downstream", "exact_mask", "4090/HISTORY stage6 +19.07%, 8192 FP32-reference gate passed", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "useExactMaskDownstreamElision", candidate_id="exact-mask-downstream-on"),
    _entry("sm89-stage50-exact-mask-preprocess", "exact_mask", "4090/HISTORY stage50 +1.061%, byte-identical", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "elideExactMask", candidate_id="exact-mask-preprocess-on"),
    _entry("sm89-stage7-fa4-native-d32-m128n112", "fa4", "4090 Stage7 53.12 to 29.46 us, CPU FP32 checked", "cpp/neuralnet/cudabackend_sm89_flash.cu", "launchFlashTactic<128,112,false,float>", candidate_id="fa4-d32-m128-n112-w4-pack0-fp32"),
    _entry("sm89-stage7-fa4-m128n96", "fa4", "4090 Stage7 24.19-24.21 us, CPU FP32 checked", "cpp/neuralnet/cudabackend_sm89_flash.cu", "launchFlashTactic<128,96,false,float>", candidate_id="fa4-d32-m128-n96-w4-pack0-fp32"),
    _entry("sm89-stage7-fa4-m64n96-pack-gqa", "fa4", "4090 Stage7 23.51-23.56 us, CPU FP32 checked", "cpp/neuralnet/cudabackend_sm89_flash.cu", "launchFlashTactic<64,96,true,float>", candidate_id="fa4-d32-m64-n96-w4-pack1-fp32"),
    _entry("sm89-stage7-fa4-m64n96-unpacked", "fa4", "4090 Stage7 23.28-23.34 us, 8192 passed", "cpp/neuralnet/cudabackend_sm89_flash.cu", "launchFlashTactic<64,96,false,float>", candidate_id="fa4-d32-m64-n96-w4-pack0-fp32"),
    _entry("sm89-stage59-fa4-both16", "fa4", "4090/HISTORY stage59 +4.375%", "cpp/neuralnet/cudabackend_sm89_flash.cu", "launchFlashTactic<64,96,false,cutlass::half_t>", candidate_id="fa4-d32-m64-n96-w4-pack0-both16"),
    _entry("sm89-stage8-dual-ffn-sw2", "dual_ffn", "4090 Stage8 accepted CUTLASS shared-A dual GEMM, swizzle2", "cpp/neuralnet/cudabackend_sm89_dual_gemm.cu", "m128-n64-k32-w64-n32-s3-sw2-exp", candidate_id="dual-cutlass-m128-n64-k32-w64-n32-s3-sw2-exp"),
    _entry("sm89-stage8-dual-ffn-sw4", "dual_ffn", "4090 Stage8 local +0.46%; later padded topology regression retained by contract", "cpp/neuralnet/cudabackend_sm89_dual_gemm.cu", "m128-n64-k32-w64-n32-s3-sw4-exp", candidate_id="dual-cutlass-m128-n64-k32-w64-n32-s3-sw4-exp"),
    _entry("sm89-stage62-dual-ffn-half2-tanh", "dual_ffn", "4090/HISTORY stage62 +1.033%", "cpp/neuralnet/cudabackend_sm89_dual_gemm.cu", "m128-n64-k32-w64-n32-s3-sw2-tanh-half2", candidate_id="dual-cutlass-m128-n64-k32-w64-n32-s3-sw2-tanh-half2"),
    _entry("sm89-stage9-linear2-w64n32-s3", "linear2", "4090 Stage9 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s3-sw1", candidate_id="linear2-cutlass-m128-n128-k32-w64-n32-s3-sw1"),
    _entry("sm89-stage9-linear2-w64n32-s4", "linear2", "4090 Stage9 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s4-sw1", candidate_id="linear2-cutlass-m128-n128-k32-w64-n32-s4-sw1"),
    _entry("sm89-stage9-linear2-w64n64-s3", "linear2", "4090 Stage9 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s3-sw1", candidate_id="linear2-cutlass-m128-n128-k32-w64-n64-s3-sw1"),
    _entry("sm89-stage9-linear2-w64n64-s4", "linear2", "4090 Stage9 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s4-sw1", candidate_id="linear2-cutlass-m128-n128-k32-w64-n64-s4-sw1"),
    _entry("sm89-stage9-linear2-w64n64-s5", "linear2", "4090 Stage9 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s5-sw1", candidate_id="linear2-cutlass-m128-n128-k32-w64-n64-s5-sw1"),
    _entry("sm89-stage9-linear2-w64n64-s6", "linear2", "4090 Stage9 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s6-sw1", candidate_id="linear2-cutlass-m128-n128-k32-w64-n64-s6-sw1"),
    _entry("sm89-stage57-linear2-postbn", "linear2", "4090/HISTORY stage57 S1 +0.799%", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "Sm89Linear2BnGemm", candidate_id="linear2-cutlass-m128-n128-k32-w64-n64-s4-sw1-postbn"),
    _entry("sm89-stage10-outproj-w64n32-s2", "outproj", "4090 Stage10 positive preliminary, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s2-sw1", candidate_id="outproj-cutlass-m128-n128-k32-w64-n32-s2-sw1"),
    _entry("sm89-stage10-outproj-w64n32-s3", "outproj", "4090 Stage10 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s3-sw1", candidate_id="outproj-cutlass-m128-n128-k32-w64-n32-s3-sw1"),
    _entry("sm89-stage10-outproj-w64n32-s4", "outproj", "4090 Stage10 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s4-sw1", candidate_id="outproj-cutlass-m128-n128-k32-w64-n32-s4-sw1"),
    _entry("sm89-stage10-outproj-w64n64-s3", "outproj", "4090 Stage10 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s3-sw1", candidate_id="outproj-cutlass-m128-n128-k32-w64-n64-s3-sw1"),
    _entry("sm89-stage10-outproj-w64n64-s4", "outproj", "4090 Stage10 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s4-sw1", candidate_id="outproj-cutlass-m128-n128-k32-w64-n64-s4-sw1"),
    _entry("sm89-stage11-preconv-w64n32-s3", "preconv", "4090 Stage11 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s3-sw1", candidate_id="preconv-cutlass-m128-n128-k32-w64-n32-s3-sw1"),
    _entry("sm89-stage11-preconv-w64n32-s4", "preconv", "4090 Stage11 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s4-sw1", candidate_id="preconv-cutlass-m128-n128-k32-w64-n32-s4-sw1"),
    _entry("sm89-stage11-preconv-w64n64-s3", "preconv", "4090 Stage11 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s3-sw1", candidate_id="preconv-cutlass-m128-n128-k32-w64-n64-s3-sw1"),
    _entry("sm89-stage11-preconv-w64n64-s4", "preconv", "4090 Stage11 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s4-sw1", candidate_id="preconv-cutlass-m128-n128-k32-w64-n64-s4-sw1"),
    _entry("sm89-stage11-preconv-w64n64-s5", "preconv", "4090 Stage11 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s5-sw1", candidate_id="preconv-cutlass-m128-n128-k32-w64-n64-s5-sw1"),
    _entry("sm89-stage11-preconv-w64n64-s6", "preconv", "4090 Stage11 positive ABBA, exact_fraction=1", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s6-sw1", candidate_id="preconv-cutlass-m128-n128-k32-w64-n64-s6-sw1"),
    _entry("sm89-stage12-postconv-w64n32-s2-sw1", "postconv_bn", "4090 Stage12 positive and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s2-sw1", candidate_id="postconv-cutlass-m128-n128-k32-w64-n32-s2-sw1"),
    _entry("sm89-stage12-postconv-w64n32-s3-sw1", "postconv_bn", "4090 Stage12 positive finalist and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s3-sw1", candidate_id="postconv-cutlass-m128-n128-k32-w64-n32-s3-sw1"),
    _entry("sm89-stage12-postconv-w64n32-s3-sw2", "postconv_bn", "4090 Stage12 positive finalist and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n32-s3-sw2", candidate_id="postconv-cutlass-m128-n128-k32-w64-n32-s3-sw2"),
    _entry("sm89-stage12-postconv-w64n64-s3-sw1", "postconv_bn", "4090 Stage12 positive finalist and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s3-sw1", candidate_id="postconv-cutlass-m128-n128-k32-w64-n64-s3-sw1"),
    _entry("sm89-stage12-postconv-w64n64-s3-sw2", "postconv_bn", "4090 Stage12 positive finalist and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s3-sw2", candidate_id="postconv-cutlass-m128-n128-k32-w64-n64-s3-sw2"),
    _entry("sm89-stage12-postconv-w64n64-s3-sw4", "postconv_bn", "4090 Stage12 positive finalist and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n128-k32-w64-n64-s3-sw4", candidate_id="postconv-cutlass-m128-n128-k32-w64-n64-s3-sw4"),
    _entry("sm89-stage12-postconv-m128n256", "postconv_bn", "4090 Stage12 positive preliminary and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m128-n256-k32-w64-n64-s2-sw2", candidate_id="postconv-cutlass-m128-n256-k32-w64-n64-s2-sw2"),
    _entry("sm89-stage12-postconv-m256n128-sw1", "postconv_bn", "4090 Stage12 positive preliminary and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m256-n128-k32-w64-n64-s2-sw1", candidate_id="postconv-cutlass-m256-n128-k32-w64-n64-s2-sw1"),
    _entry("sm89-stage12-postconv-m256n128-sw2", "postconv_bn", "4090 Stage12 positive preliminary and bit exact", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "m256-n128-k32-w64-n64-s2-sw2", candidate_id="postconv-cutlass-m256-n128-k32-w64-n64-s2-sw2"),
    _entry("sm89-stage56-postconv-bn-silu", "postconv_bn", "4090/HISTORY stage56 +0.365% S2", "cpp/neuralnet/cudabackend_sm89_linear2_gemm.cu", "Sm89PostConvBnGemm", candidate_id="postconv-cutlass-m128-n128-k32-w64-n64-s3-sw1-bn-silu"),
    _entry("sm89-stage22-c768-vec8", "pointwise", "4090/HISTORY stage22", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89ScaleBiasSiluNHWCHalfVec8", candidate_id="pointwise-c768-vec8"),
    _entry("sm89-stage34-c384-vec8", "pointwise", "4090/HISTORY stage34 local -31.82%", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89ScaleBiasSiluNHWCHalfVec8C384", candidate_id="pointwise-c384-vec8"),
    _entry("sm89-stage54-c384-vec4", "pointwise", "4090/HISTORY stage54 S1 +0.543%", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89ScaleBiasSiluNHWCHalfVec4C384", candidate_id="pointwise-c384-vec4"),
    _entry("sm89-stage20-l2-trunk", "l2", "4090/HISTORY stage20", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "setPersistingL2Window", config={"cudaUsePersistingL2Trunk": True, "cudaUsePersistingL2Inner": False, "cudaPersistingL2HitRatioSm89": 1.0}),
    _entry("sm89-stage21-l2-trunk-inner", "l2", "4090/HISTORY stage21 added inner retention on top of retained trunk", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "setPersistingL2Window", config={"cudaUsePersistingL2Trunk": True, "cudaUsePersistingL2Inner": True, "cudaPersistingL2HitRatioSm89": 1.0}),
    _entry("sm89-stage24-initial-conv", "initial_conv", "4090/HISTORY stage24 +0.160%", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "deselect_workspace_greater_than", candidate_id="initial_conv-on"),
    _entry("sm89-stage27-initial-global", "initial_global", "4090/HISTORY stage27 S1 +0.118%", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89InitialGlobalMatMulAdd", candidate_id="initial_global-on"),
    _entry("sm89-stage25-policy-p1-v1", "policy_p1", "4090 stage25 v1 block96x1 local positive", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "rowsPerBlock", candidate_id="policy-p1-block96x1"),
    _entry("sm89-stage25-policy-p1-v2", "policy_p1", "4090 stage25 v2 block96x5 local positive", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "rowsPerBlock", candidate_id="policy-p1-block96x5"),
    _entry("sm89-stage28-wide-head", "wide_head", "4090/HISTORY stage28 S1 +0.599%", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "Sm89WideHeadProjection", candidate_id="wide-head-on"),
    _entry("sm89-stage52-intrinsic-head-bundle", "wide_head", "4090/HISTORY stage52 initial-global + wide-head + head-BN S1 ABBA +0.623%", "cpp/neuralnet/cudabackend_sm89_forward.cpp", "Sm89WideHeadProjection", candidate_id="wide-head-stage52-intrinsic-bundle"),
    _entry("sm89-stage29-head-bn", "head_bn", "4090/HISTORY stage29 S1 +0.078%", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89HeadBNSiluStrided", candidate_id="head_bn-on"),
    _entry("sm89-stage51-value-terminal", "value_terminal", "4090/HISTORY stage51 local -49%", "cpp/neuralnet/cudabackend_sm89_kernels.cu", "sm89SplitValueTerminal", candidate_id="value_terminal-on"),
)


SM120_POSITIVE_HISTORY = (
    _entry("sm120-stage-qkv-strided", "wide_qkv", "5090D rebuild S1 +1.405%, full replay passed", "cpp/neuralnet/cudabackend_sm120.cpp", "cublasHgemmStridedBatched", candidate_id="wide_qkv-strided-batched"),
    _entry("sm120-stage-wide-qkv-tilelang", "wide_qkv", "5090D rebuild planar wide-QKV Stage21 +3.806%, 8192 passed", "python/sm120_generate_tilelang_aot.py", "wide_qkv", candidate_id="wide_qkv-m128-n128-k64-s2-tilelang-planar"),
    _entry("sm120-stage-wide-qkv-lower-smem", "wide_qkv", "5090D direct QKV 18.840 to 17.128 us, exact arithmetic", "python/sm120_generate_tilelang_aot.py", "wide_qkv", candidate_id="wide_qkv-m128-n128-k32-s3-tilelang-planar"),
    _entry("sm120-stage-wide-qkv-atom2x2", "wide_qkv", "5080 QKV CuTe baseline", "python/sm120_generate_cute_qkv_aot.py", "ATOM_LAYOUT", candidate_id="wide_qkv-m128-n128-k64-s2-cute-atom2x2-packed"),
    _entry("sm120-stage-wide-qkv-atom4x2", "wide_qkv", "5080 history row96 +0.373%", "python/sm120_generate_cute_qkv_aot.py", "--atom-layout", candidate_id="wide_qkv-m128-n128-k64-s2-cute-atom4x2-packed"),
    _entry("sm120-stage-wide-ffn", "wide_ffn", "5090D FINAL-S1-B13", "cpp/neuralnet/cudabackend_sm120.cpp", "single-wide FFN projection active", candidate_id="wide_ffn-single-projection"),
    _entry("sm120-combined-s1-projections", "wide_projection", "5090D combined S1 single-wide FFN plus strided QKV +2.564%, replay byte-identical", "cpp/neuralnet/cudabackend_sm120.cpp", "cublasHgemmStridedBatched", backend_symbols=["single-wide FFN projection active"], candidate_id="wide-projections-s1-bundle"),
    _entry("sm120-stage-residual-beta1", "fused_residual", "5080 history row68 +4.03%", "cpp/neuralnet/cudabackend_sm120.cpp", "fusedResidualGemm", candidate_id="fused_residual-on"),
    _entry("sm120-stage-rms-ordered-ept3", "rmsnorm", "5080 history row72", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "rmsNorm384OrderedEpt3Kernel", candidate_id="rmsnorm-ordered-ept3"),
    _entry("sm120-stage-rms-one-warp", "rmsnorm", "5090D accepted exact-tree one-warp", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "rmsNorm384Half2Kernel", candidate_id="rmsnorm-one-warp"),
    _entry("sm120-stage-rms-vec8", "rmsnorm", "5080 history row88 +4.053%", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "rmsNorm384Vec8Kernel", candidate_id="rmsnorm-vec8"),
    _entry("sm120-stage51-exact-mask", "exact_mask", "5090D Stage51 S1 +0.110%", "cpp/neuralnet/cudabackend.cpp", "sm120ExactMask", candidate_id="exact_mask-on"),
    _entry("sm120-stage-fused-qk-rope", "qkv_rope", "5080 history row70", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "fusedQKRoPE19HalfKernel", candidate_id="qkv-rope-fused-scalar"),
    _entry("sm120-stage-rope-half2", "qkv_rope", "5080 history row91 +0.630%", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "fusedQKRoPE19Half2Kernel", candidate_id="qkv-rope-fused-half2"),
    _entry("sm120-stage-rope-batch-shared", "qkv_rope", "5080 history row81 +1.52%", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "batchSharedPackedFusedQKRoPE19Half2Kernel", candidate_id="qkv-rope-batch-shared"),
    _entry("sm120-stage-rope-unrolled", "qkv_rope", "5080 history row95 +0.284%", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "launchBatchSharedPackedFusedQKRoPEUnrolledExact", candidate_id="qkv-rope-batch-shared-unrolled"),
    _entry("sm120-stage55-packed-qkv-rope", "qkv_rope", "5090D Stage55 boundary -4.099%", "python/sm120_generate_cute_qkv_rope_aot.py", "rope", candidate_id="qkv-packed-cute-precomputed-rope-static-register-both16-epilogue"),
    _entry("sm120-fa4-n64-fp32", "fa4", "5080 accumulator sweep N64 fp32 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_QK_ACC", fields={"tile_n": 64, "accumulation": "fp32"}),
    _entry("sm120-fa4-n64-qk16", "fa4", "5080 accumulator sweep N64 qk16 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_QK_ACC", fields={"tile_n": 64, "accumulation": "qk16"}),
    _entry("sm120-fa4-n64-pv16", "fa4", "5080 accumulator sweep N64 pv16 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_PV_ACC", fields={"tile_n": 64, "accumulation": "pv16"}),
    _entry("sm120-fa4-n64-both16", "fa4", "5080 accumulator sweep N64 both16 positive and precise; 5090D Stage56 +0.437%", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_PV_ACC", fields={"tile_n": 64, "accumulation": "both16"}),
    _entry("sm120-stage57-fa4-n96", "fa4", "5090D Stage57 +0.625%", "cpp/neuralnet/fa4_aot/build_aot.py", "--tile-n", fields={"tile_n": 96, "accumulation": "both16"}),
    _entry("sm120-fa4-n128-fp32", "fa4", "5090D initial accumulator matrix N128 fp32 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_QK_ACC", fields={"tile_n": 128, "accumulation": "fp32"}),
    _entry("sm120-fa4-n128-qk16", "fa4", "5090D initial accumulator matrix N128 qk16 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_QK_ACC", fields={"tile_n": 128, "accumulation": "qk16"}),
    _entry("sm120-fa4-n128-pv16", "fa4", "5090D initial accumulator matrix N128 pv16 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_PV_ACC", fields={"tile_n": 128, "accumulation": "pv16"}),
    _entry("sm120-fa4-n128-both16", "fa4", "5090D initial accumulator matrix N128 both16 positive and precise", "cpp/neuralnet/fa4_aot/build_aot.py", "FA4_PV_ACC", fields={"tile_n": 128, "accumulation": "both16"}),
    _entry("sm120-stage-dual-ffn-tilelang", "dual_ffn", "5080 history row84 +1.991%", "python/sm120_historical_ffn/generate.py", "tanh", candidate_id="dual_ffn-m128-n64-k32-s2-mb3-tanh-half2"),
    _entry("sm120-stage20-dual-ffn-cutlass-shared-a", "dual_ffn", "5080 shared-A dual GEMM whole +15.77%, 8192 passed", "cpp/neuralnet/sm120_aot/dual_ffn_shared_a.cu", "DualGemm", candidate_id="dual_ffn-cutlass-shared-a-m128-n64-k32-s3-swizzle2"),
    _entry("sm120-stage20-dual-ffn-exp", "dual_ffn", "5090D Stage20 S2 +2.586%, 8192 full FP32 gate passed", "python/sm120_generate_tilelang_aot.py", "fused_ffn", candidate_id="dual_ffn-m128-n64-k32-s2-mb3-exp"),
    _entry("sm120-stage33b-dual-ffn-areuse", "dual_ffn", "5090D Stage33b +0.910%, 8192 byte-identical", "python/sm120_generate_tilelang_aot.py", "a_fragment_reuse", candidate_id="dual_ffn-m128-n64-k32-s2-mb3-areuse-exp"),
    _entry("sm120-stage-dual-ffn-three-stage", "dual_ffn", "5090D historical S1 3047.568 to 3089.095", "python/sm120_generate_tilelang_aot.py", "num_stages", candidate_id="dual_ffn-m128-n64-k32-s3-mb2-areuse-exp"),
    _entry("sm120-stage47-dual-ffn-cute-grid340", "dual_ffn", "5090D Stage47 grid340 +0.449%", "python/sm120_generate_cute_fused_ffn_aot.py", "paired-projection", candidate_id="dual_ffn-cute-m128-n64x2-k32-ab2-epi4-grid340"),
    _entry("sm120-stage-linear2", "linear2", "5080 history row85 +2.294%", "cpp/neuralnet/sm120_aot/linear2_residual_cutlass.cu", "launchLinear2ResidualCutlass", candidate_id="linear2-m128-n128-k32-s3-cutlass"),
    _entry("sm120-stage22-linear2-m128n128-s2", "linear2", "5090D Stage22 direct single-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="linear2-m128-n128-k32-s2-t128-mb3-tilelang-32k"),
    _entry("sm120-stage22-linear2-m128n128-s3-t128", "linear2", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="linear2-m128-n128-k32-s3-t128-mb3-tilelang-49k"),
    _entry("sm120-stage22-linear2-m128n128-s3-t256", "linear2", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="linear2-m128-n128-k32-s3-t256-mb3-tilelang-49k"),
    _entry("sm120-stage22-linear2-m128n128-s4", "linear2", "5090D Stage22 exact B13/S2 ABBA four adjacent comparisons positive, full FP32 replay passed", "python/sm120_generate_tilelang_aot.py", "linear2", candidate_id="linear2-m128-n128-k32-s4-tilelang-64k"),
    _entry("sm120-stage22-linear2-m128n128-k64-s2", "linear2", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="linear2-m128-n128-k64-s2-t128-mb3-tilelang-64k"),
    _entry("sm120-stage22-linear2-m128n64-s3", "linear2", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="linear2-m128-n64-k32-s3-t128-mb3-tilelang-36k"),
    _entry("sm120-stage22-linear2-m64n128-s3", "linear2", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="linear2-m64-n128-k32-s3-t128-mb4-tilelang-36k"),
    _entry("sm120-stage-linear2-m128n96-s4", "linear2", "5090D homogeneous S2 33.246 to 33.092 us, boundary bit-exact", "python/sm120_generate_tilelang_aot.py", "linear2", candidate_id="linear2-m128-n96-k32-s4-tilelang"),
    _entry("sm120-stage-outproj", "outproj", "5080 history row86 +1.187%", "cpp/neuralnet/sm120_aot/outproj_residual_cutlass.cu", "launchOutProjectionResidualCutlass", candidate_id="outproj-m128-n128-k32-s3-cutlass"),
    _entry("sm120-stage22-outproj-m128n128-s3", "outproj", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="outproj-m128-n128-k32-s3-t128-mb3-tilelang-49k"),
    _entry("sm120-stage22-outproj-m128n128-s4", "outproj", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="outproj-m128-n128-k32-s4-tilelang-64k"),
    _entry("sm120-stage22-outproj-m128n128-k64-s2", "outproj", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="outproj-m128-n128-k64-s2-t128-mb3-tilelang-64k"),
    _entry("sm120-stage22-outproj-m128n64-s3", "outproj", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="outproj-m128-n64-k32-s3-t128-mb3-tilelang-36k"),
    _entry("sm120-stage22-outproj-m64n128-s3", "outproj", "5090D Stage22 direct single and dual-stream positive; numerical smoke passed", "python/sm120_generate_tilelang_aot.py", "residual_gemm", candidate_id="outproj-m64-n128-k32-s3-t128-mb4-tilelang-36k"),
    _entry("sm120-stage-swiglu", "swiglu", "5090D rebuild Stage12", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "wideSwiGLUHalf2Kernel", candidate_id="swiglu-on"),
    _entry("sm120-stage93-outer-projection-bundle", "postconv_bn", "5080 history row93 +1.198% for joint C768->C384 down and C384->C768 up", "cpp/neuralnet/sm120_aot/outer_projection.cu", "katago_create_outer_projection_down_sm120", backend_symbols=["katago_create_outer_projection_up_sm120"], candidate_id="outer-projection-cutlass-warp64x64-bundle"),
    _entry("sm120-stage-postconv-warp64x32", "postconv_bn", "5080 history row97 +0.078%", "cpp/neuralnet/sm120_aot/outer_projection.cu", "GemmWarp64x32", candidate_id="postconv-cutlass-warp64x32"),
    _entry("sm120-stage45-postconv-bn-silu", "postconv_bn", "5090D Stage45 S1 +0.181%", "cpp/neuralnet/sm120_aot/postconv_residual_affine_silu.cu", "launchPostConvResidualAffineSilu", candidate_id="postconv-cutlass-bn-silu"),
    _entry("sm120-stage-pointwise-half2", "pointwise", "5080 history row73", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "affineSiluHalf2Kernel", candidate_id="pointwise-half2"),
    _entry("sm120-stage-pointwise-half2x3", "pointwise", "5080 history row74 positive both orders", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "affineSiluHalf2x3Kernel", candidate_id="pointwise-half2x3"),
    _entry("sm120-stage-pointwise-flat-vec8", "pointwise", "5080 history row89 +0.216%", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "affineSiluFlatVec8C768Kernel", candidate_id="pointwise-flat-vec8-c768"),
    _entry("sm120-stage-l2-trunk", "l2", "5080 history row90 +0.305%", "cpp/neuralnet/cudabackend_sm120.cpp", "persistingL2TrunkActive", config={"cudaUsePersistingL2Trunk": True, "cudaUsePersistingL2Inner": False, "cudaPersistingL2HitRatioSm120": 1.0}),
    _entry("sm120-stage-l2-trunk-inner", "l2", "5080 history row92 +0.806% after adding inner retention to retained trunk", "cpp/neuralnet/cudabackend_sm120.cpp", "persistingL2InnerActive", config={"cudaUsePersistingL2Trunk": True, "cudaUsePersistingL2Inner": True, "cudaPersistingL2HitRatioSm120": 1.0}),
    _entry("sm120-stage-weight-sharing", "weight_sharing", "5080 history row94 +0.126%", "cpp/neuralnet/cudabackend_sm120.cpp", "shareModelWeights", candidate_id="weight_sharing-on"),
    _entry("sm120-stage-initial-conv-eng45", "initial_conv", "5080 history row98 +0.461%", "cpp/neuralnet/cudabackend.cpp", "sm120InitialConvFrontendEngine == 45", candidate_id="initial-conv-eng45-tile0-stages2"),
    _entry("sm120-stage52-initial-conv-eng47", "initial_conv", "5090D Stage52 S1 +0.193%", "cpp/neuralnet/cudabackend.cpp", "sm120InitialConvFrontendEngine == 47", candidate_id="initial-conv-eng47-k2-2-k6-1-k13-1-k14-0-k22-2"),
    _entry("sm120-stage-initial-global", "initial_global", "5080 history row101 +0.139%", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "initialGlobalMatMulAddKernel", candidate_id="initial_global-on"),
    _entry("sm120-stage-policy-p1", "policy_p1", "5080 history row99 positive both orders", "cpp/neuralnet/cudabackend_sm120_kernels.cu", "fusedPolicyP1Kernel", candidate_id="policy_p1-on"),
    _entry("sm120-stage-wide-head", "wide_head", "5080 history row102 +0.255%", "cpp/neuralnet/cudabackend_sm120.cpp", "wideHeadProjectionHandle", candidate_id="wide-head-full-c384"),
    _entry("sm120-stage53-partial-wide-head", "wide_head", "5090D Stage53 local 16.93 to 14.66 us, B13 byte-identical", "cpp/neuralnet/cudabackend_sm120.cpp", "partial C288 no-split g1+v1 head active", candidate_id="wide-head-partial-c288-g1-v1"),
    _entry("sm120-stage-head-bn", "head_bn", "5080 history row100 positive both orders", "cpp/neuralnet/cudabackend.cpp", "head BN direct FP32 output active", candidate_id="head_bn-on"),
)


POSITIVE_HISTORY: dict[str, tuple[dict[str, object], ...]] = {
    "sm89": SM89_POSITIVE_HISTORY,
    "sm120": SM120_POSITIVE_HISTORY,
}


def _candidate_config(candidate: Mapping[str, object]) -> dict[str, object]:
    config = candidate.get("config", candidate.get("config_overrides", {}))
    return dict(config) if isinstance(config, Mapping) else {}


def _matches(candidate: Mapping[str, object], selector: Mapping[str, object]) -> bool:
    candidate_id = str(candidate.get("id", ""))
    exact = selector.get("candidate_id")
    if exact is not None and candidate_id != exact:
        return False
    prefix = selector.get("candidate_prefix")
    if prefix is not None and not candidate_id.startswith(str(prefix)):
        return False
    required_config = selector.get("config", {})
    config = _candidate_config(candidate)
    if not isinstance(required_config, Mapping) or any(
        config.get(key) != value for key, value in required_config.items()
    ):
        return False
    fields = selector.get("fields", {})
    if not isinstance(fields, Mapping) or any(
        candidate.get(key) != value for key, value in fields.items()
    ):
        return False
    return True


def history_contract_sha256(architecture: str) -> str:
    encoded = json.dumps(
        POSITIVE_HISTORY[architecture], sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_positive_history_closure(
    repo: pathlib.Path,
    architecture: str,
    batches: Mapping[int, Mapping[str, Sequence[Mapping[str, object]]]],
    runtime_keys: set[str] | frozenset[str],
) -> dict[str, object]:
    """Fail closed if any positive-history route loses one of its four links."""
    if architecture not in POSITIVE_HISTORY:
        raise ValueError(f"no positive-history contract for {architecture}")
    records = POSITIVE_HISTORY[architecture]
    seen_ids: set[str] = set()
    for record in records:
        history_id = str(record["history_id"])
        if history_id in seen_ids:
            raise ValueError(f"duplicate positive-history id: {history_id}")
        seen_ids.add(history_id)
        backend = record["backend"]
        if not isinstance(backend, Mapping):
            raise ValueError(f"malformed backend proof: {history_id}")
        source = repo / str(backend["file"])
        symbols = [str(backend["symbol"])]
        extra_symbols = backend.get("symbols", [])
        if not isinstance(extra_symbols, Sequence) or isinstance(
            extra_symbols, (str, bytes)
        ):
            raise ValueError(f"malformed backend symbols: {history_id}")
        symbols.extend(str(symbol) for symbol in extra_symbols)
        source_text = source.read_text(errors="replace") if source.is_file() else ""
        missing_symbols = [symbol for symbol in symbols if symbol not in source_text]
        if missing_symbols:
            raise ValueError(
                f"positive-history backend proof is missing: {history_id} "
                f"({source}:{missing_symbols})"
            )
        family = str(record["family"])
        selector = record["match"]
        if not isinstance(selector, Mapping):
            raise ValueError(f"malformed candidate selector: {history_id}")
        for batch, family_map in sorted(batches.items()):
            values = family_map.get(family)
            if values is None:
                raise ValueError(
                    f"positive-history family missing at B{batch}: "
                    f"{history_id}/{family}"
                )
            matches = [value for value in values if _matches(value, selector)]
            if not matches:
                raise ValueError(
                    f"positive-history scan candidate missing at B{batch}: "
                    f"{history_id}/{family}"
                )
            if len(matches) != 1:
                raise ValueError(
                    f"positive-history selector is not a distinct coordinate: "
                    f"{history_id}/{family}/B{batch} matched {len(matches)}"
                )
            for value in matches:
                config = _candidate_config(value)
                if not config:
                    raise ValueError(
                        f"positive-history plan apply mapping is empty: "
                        f"{history_id}/B{batch}/{value.get('id')}"
                    )
                unknown = sorted(set(config) - set(runtime_keys))
                if unknown:
                    raise ValueError(
                        f"positive-history plan apply uses unknown keys: "
                        f"{history_id}/B{batch}/{unknown}"
                    )
                markers = value.get("activation_markers", [])
                if not isinstance(markers, list) or not markers:
                    raise ValueError(
                        f"positive-history activation proof is missing: "
                        f"{history_id}/B{batch}/{value.get('id')}"
                    )
                if value.get("implementation") == "fallback":
                    raise ValueError(
                        f"positive-history route resolves to fallback: "
                        f"{history_id}/B{batch}/{value.get('id')}"
                    )
                if value.get("requires_artifact") and not value.get("generator"):
                    raise ValueError(
                        f"positive-history generated route has no generator: "
                        f"{history_id}/B{batch}/{value.get('id')}"
                    )
    return {
        "complete": True,
        "architecture": architecture,
        "record_count": len(records),
        "record_ids": [str(record["history_id"]) for record in records],
        "contract_sha256": history_contract_sha256(architecture),
        "validated_batches": sorted(batches),
        "links": ["backend", "scan_candidate", "activation", "plan_apply"],
    }
