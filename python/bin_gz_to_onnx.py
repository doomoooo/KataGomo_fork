#!/usr/bin/env python3
"""
Convert a KataGo-style .bin/.bin.gz model into an ONNX file that matches this
repository's current ONNX backend conventions:

- inputs:
  - input_spatial: [batch, C, H, W]
  - input_global:  [batch, G]
- outputs:
  - out_policy:        [batch, P, H*W+1]
  - out_value:         [batch, 3]
  - out_miscvalue:     [batch, 10]
  - out_moremiscvalue: [batch, 8]
  - out_ownership:     [batch, H*W]

The exported ONNX model uses a fixed board size and embeds the metadata fields
expected by cpp/neuralnet/onnxprotoreader.cpp.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import onnx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "Missing PyTorch. Install a CPU build first, for example:\n"
        "  conda install -c pytorch pytorch cpuonly\n"
        "or:\n"
        "  pip install torch"
    ) from exc


DEFAULT_SCORE_PARAMS = {
    "td_score_multiplier": 20.0,
    "score_mean_multiplier": 20.0,
    "score_stdev_multiplier": 20.0,
    "lead_multiplier": 20.0,
    "variance_time_multiplier": 40.0,
    "shortterm_value_error_multiplier": 0.25,
    "shortterm_score_error_multiplier": 30.0,
    "output_scale_multiplier": 1.0,
}


@dataclass
class PostProcessParams:
    td_score_multiplier: float = DEFAULT_SCORE_PARAMS["td_score_multiplier"]
    score_mean_multiplier: float = DEFAULT_SCORE_PARAMS["score_mean_multiplier"]
    score_stdev_multiplier: float = DEFAULT_SCORE_PARAMS["score_stdev_multiplier"]
    lead_multiplier: float = DEFAULT_SCORE_PARAMS["lead_multiplier"]
    variance_time_multiplier: float = DEFAULT_SCORE_PARAMS["variance_time_multiplier"]
    shortterm_value_error_multiplier: float = DEFAULT_SCORE_PARAMS["shortterm_value_error_multiplier"]
    shortterm_score_error_multiplier: float = DEFAULT_SCORE_PARAMS["shortterm_score_error_multiplier"]
    output_scale_multiplier: float = DEFAULT_SCORE_PARAMS["output_scale_multiplier"]


@dataclass
class ConvDesc:
    name: str
    weight: np.ndarray
    dilation_y: int
    dilation_x: int


@dataclass
class BatchNormDesc:
    name: str
    merged_scale: np.ndarray
    merged_bias: np.ndarray


@dataclass
class ActivationDesc:
    name: str
    kind: str


@dataclass
class MatMulDesc:
    name: str
    weight: np.ndarray


@dataclass
class MatBiasDesc:
    name: str
    bias: np.ndarray


@dataclass
class ResidualBlockDesc:
    name: str
    pre_bn: BatchNormDesc
    pre_act: ActivationDesc
    regular_conv: ConvDesc
    mid_bn: BatchNormDesc
    mid_act: ActivationDesc
    final_conv: ConvDesc


@dataclass
class GlobalPoolingResidualBlockDesc:
    name: str
    pre_bn: BatchNormDesc
    pre_act: ActivationDesc
    regular_conv: ConvDesc
    gpool_conv: ConvDesc
    gpool_bn: BatchNormDesc
    gpool_act: ActivationDesc
    gpool_to_bias_mul: MatMulDesc
    mid_bn: BatchNormDesc
    mid_act: ActivationDesc
    final_conv: ConvDesc


BlockDesc = Union["ResidualBlockDesc", "GlobalPoolingResidualBlockDesc", "NestedBottleneckResidualBlockDesc"]


@dataclass
class NestedBottleneckResidualBlockDesc:
    name: str
    num_blocks: int
    pre_bn: BatchNormDesc
    pre_act: ActivationDesc
    pre_conv: ConvDesc
    blocks: List[BlockDesc]
    post_bn: BatchNormDesc
    post_act: ActivationDesc
    post_conv: ConvDesc


@dataclass
class SGFMetadataEncoderDesc:
    name: str
    num_input_meta_channels: int
    mul1: MatMulDesc
    bias1: MatBiasDesc
    act1: ActivationDesc
    mul2: MatMulDesc
    bias2: MatBiasDesc
    act2: ActivationDesc
    mul3: MatMulDesc


@dataclass
class TrunkDesc:
    name: str
    num_blocks: int
    trunk_num_channels: int
    initial_conv: ConvDesc
    initial_matmul: MatMulDesc
    sgf_metadata_encoder: SGFMetadataEncoderDesc | None
    blocks: List[BlockDesc]
    trunk_tip_bn: BatchNormDesc
    trunk_tip_activation: ActivationDesc


@dataclass
class PolicyHeadDesc:
    name: str
    p1_conv: ConvDesc
    g1_conv: ConvDesc
    g1_bn: BatchNormDesc
    g1_activation: ActivationDesc
    gpool_to_bias_mul: MatMulDesc
    p1_bn: BatchNormDesc
    p1_activation: ActivationDesc
    p2_conv: ConvDesc
    gpool_to_pass_mul: MatMulDesc
    gpool_to_pass_bias: MatBiasDesc | None
    pass_activation: ActivationDesc | None
    gpool_to_pass_mul2: MatMulDesc | None


@dataclass
class ValueHeadDesc:
    name: str
    v1_conv: ConvDesc
    v1_bn: BatchNormDesc
    v1_activation: ActivationDesc
    v2_mul: MatMulDesc
    v2_bias: MatBiasDesc
    v2_activation: ActivationDesc
    v3_mul: MatMulDesc
    v3_bias: MatBiasDesc
    sv3_mul: MatMulDesc
    sv3_bias: MatBiasDesc
    v_ownership_conv: ConvDesc


@dataclass
class ModelDesc:
    name: str
    model_version: int
    num_input_channels: int
    num_input_global_channels: int
    num_input_meta_channels: int
    meta_encoder_version: int
    postprocess: PostProcessParams
    trunk: TrunkDesc
    policy_head: PolicyHeadDesc
    value_head: ValueHeadDesc


class TokenStream:
    def __init__(self, data: bytes):
        self.data = data
        self.idx = 0

    def _skip_ws(self) -> None:
        data = self.data
        n = len(data)
        i = self.idx
        while i < n and data[i] in b" \t\r\n":
            i += 1
        self.idx = i

    def read_token(self) -> str:
        self._skip_ws()
        start = self.idx
        data = self.data
        n = len(data)
        while self.idx < n and data[self.idx] not in b" \t\r\n":
            self.idx += 1
        if self.idx == start:
            raise ValueError("Unexpected end of model while reading token")
        return data[start:self.idx].decode("ascii")

    def read_int(self) -> int:
        return int(self.read_token())

    def read_float(self) -> float:
        return float(self.read_token())

    def read_binary_floats(self, count: int, name: str) -> np.ndarray:
        self._skip_ws()
        header = b"@BIN@"
        end_header = self.idx + len(header)
        if self.data[self.idx:end_header] != header:
            raise ValueError(f"{name}: expected @BIN@ block")
        self.idx = end_header
        end = self.idx + count * 4
        if end > len(self.data):
            raise ValueError(f"{name}: truncated float block")
        arr = np.frombuffer(self.data[self.idx:end], dtype="<f4").copy()
        self.idx = end
        return arr


class BinModelParser:
    def __init__(self, stream: TokenStream):
        self.s = stream

    def parse(self) -> ModelDesc:
        name = self.s.read_token()
        model_version = self.s.read_int()
        if model_version < 3:
            raise ValueError(f"Unsupported model version {model_version}")

        num_input_channels = self.s.read_int()
        num_input_global_channels = self.s.read_int()

        if model_version >= 13:
            postprocess = PostProcessParams(
                td_score_multiplier=self.s.read_float(),
                score_mean_multiplier=self.s.read_float(),
                score_stdev_multiplier=self.s.read_float(),
                lead_multiplier=self.s.read_float(),
                variance_time_multiplier=self.s.read_float(),
                shortterm_value_error_multiplier=self.s.read_float(),
                shortterm_score_error_multiplier=self.s.read_float(),
            )
        else:
            postprocess = PostProcessParams()

        meta_encoder_version = 0
        num_input_meta_channels = 0
        if model_version >= 15:
            meta_encoder_version = self.s.read_int()
            num_input_meta_channels = 192 if meta_encoder_version > 0 else 0
            for _ in range(7):
                _ = self.s.read_int()

        trunk = self._parse_trunk(model_version, meta_encoder_version)
        policy_head = self._parse_policy_head(model_version)
        value_head = self._parse_value_head(model_version)

        return ModelDesc(
            name=name,
            model_version=model_version,
            num_input_channels=num_input_channels,
            num_input_global_channels=num_input_global_channels,
            num_input_meta_channels=num_input_meta_channels,
            meta_encoder_version=meta_encoder_version,
            postprocess=postprocess,
            trunk=trunk,
            policy_head=policy_head,
            value_head=value_head,
        )

    def _parse_conv(self) -> ConvDesc:
        name = self.s.read_token()
        conv_y = self.s.read_int()
        conv_x = self.s.read_int()
        in_channels = self.s.read_int()
        out_channels = self.s.read_int()
        dilation_y = self.s.read_int()
        dilation_x = self.s.read_int()
        weight = self.s.read_binary_floats(conv_y * conv_x * in_channels * out_channels, name)
        weight = weight.reshape(conv_y, conv_x, in_channels, out_channels).transpose(3, 2, 0, 1).copy()
        return ConvDesc(name=name, weight=weight, dilation_y=dilation_y, dilation_x=dilation_x)

    def _parse_bn(self) -> BatchNormDesc:
        name = self.s.read_token()
        num_channels = self.s.read_int()
        epsilon = self.s.read_float()
        has_scale = bool(self.s.read_int())
        has_bias = bool(self.s.read_int())
        mean = self.s.read_binary_floats(num_channels, name)
        variance = self.s.read_binary_floats(num_channels, name)
        scale = self.s.read_binary_floats(num_channels, name) if has_scale else np.ones(num_channels, dtype=np.float32)
        bias = self.s.read_binary_floats(num_channels, name) if has_bias else np.zeros(num_channels, dtype=np.float32)
        merged_scale = scale / np.sqrt(variance + epsilon)
        merged_bias = bias - merged_scale * mean
        return BatchNormDesc(name=name, merged_scale=merged_scale.astype(np.float32), merged_bias=merged_bias.astype(np.float32))

    def _parse_activation(self, model_version: int) -> ActivationDesc:
        name = self.s.read_token()
        if model_version >= 11:
            kind = self.s.read_token()
        else:
            kind = "ACTIVATION_RELU"
        return ActivationDesc(name=name, kind=kind)

    def _parse_matmul(self) -> MatMulDesc:
        name = self.s.read_token()
        in_channels = self.s.read_int()
        out_channels = self.s.read_int()
        weight = self.s.read_binary_floats(in_channels * out_channels, name)
        weight = weight.reshape(in_channels, out_channels).transpose(1, 0).reshape(out_channels, in_channels, 1, 1).copy()
        return MatMulDesc(name=name, weight=weight)

    def _parse_matbias(self) -> MatBiasDesc:
        name = self.s.read_token()
        num_channels = self.s.read_int()
        bias = self.s.read_binary_floats(num_channels, name).astype(np.float32)
        return MatBiasDesc(name=name, bias=bias)

    def _parse_residual_block(self, model_version: int) -> ResidualBlockDesc:
        name = self.s.read_token()
        return ResidualBlockDesc(
            name=name,
            pre_bn=self._parse_bn(),
            pre_act=self._parse_activation(model_version),
            regular_conv=self._parse_conv(),
            mid_bn=self._parse_bn(),
            mid_act=self._parse_activation(model_version),
            final_conv=self._parse_conv(),
        )

    def _parse_gpool_block(self, model_version: int) -> GlobalPoolingResidualBlockDesc:
        name = self.s.read_token()
        return GlobalPoolingResidualBlockDesc(
            name=name,
            pre_bn=self._parse_bn(),
            pre_act=self._parse_activation(model_version),
            regular_conv=self._parse_conv(),
            gpool_conv=self._parse_conv(),
            gpool_bn=self._parse_bn(),
            gpool_act=self._parse_activation(model_version),
            gpool_to_bias_mul=self._parse_matmul(),
            mid_bn=self._parse_bn(),
            mid_act=self._parse_activation(model_version),
            final_conv=self._parse_conv(),
        )

    def _parse_nested_block(self, model_version: int) -> NestedBottleneckResidualBlockDesc:
        name = self.s.read_token()
        num_blocks = self.s.read_int()
        pre_bn = self._parse_bn()
        pre_act = self._parse_activation(model_version)
        pre_conv = self._parse_conv()
        blocks = self._parse_block_stack(model_version, num_blocks)
        post_bn = self._parse_bn()
        post_act = self._parse_activation(model_version)
        post_conv = self._parse_conv()
        return NestedBottleneckResidualBlockDesc(
            name=name,
            num_blocks=num_blocks,
            pre_bn=pre_bn,
            pre_act=pre_act,
            pre_conv=pre_conv,
            blocks=blocks,
            post_bn=post_bn,
            post_act=post_act,
            post_conv=post_conv,
        )

    def _parse_block_stack(self, model_version: int, num_blocks: int) -> List[BlockDesc]:
        blocks: List[BlockDesc] = []
        for _ in range(num_blocks):
            kind = self.s.read_token()
            if kind == "ordinary_block":
                blocks.append(self._parse_residual_block(model_version))
            elif kind == "gpool_block":
                blocks.append(self._parse_gpool_block(model_version))
            elif kind == "nested_bottleneck_block":
                blocks.append(self._parse_nested_block(model_version))
            else:
                raise ValueError(f"Unknown block kind {kind}")
        return blocks

    def _parse_metadata_encoder(self, model_version: int) -> SGFMetadataEncoderDesc:
        name = self.s.read_token()
        num_input_meta_channels = self.s.read_int()
        return SGFMetadataEncoderDesc(
            name=name,
            num_input_meta_channels=num_input_meta_channels,
            mul1=self._parse_matmul(),
            bias1=self._parse_matbias(),
            act1=self._parse_activation(model_version),
            mul2=self._parse_matmul(),
            bias2=self._parse_matbias(),
            act2=self._parse_activation(model_version),
            mul3=self._parse_matmul(),
        )

    def _parse_trunk(self, model_version: int, meta_encoder_version: int) -> TrunkDesc:
        name = self.s.read_token()
        num_blocks = self.s.read_int()
        trunk_num_channels = self.s.read_int()
        _mid_num_channels = self.s.read_int()
        _regular_num_channels = self.s.read_int()
        _dilated_num_channels = self.s.read_int()
        _gpool_num_channels = self.s.read_int()
        if model_version >= 15:
            for _ in range(6):
                _ = self.s.read_int()

        initial_conv = self._parse_conv()
        initial_matmul = self._parse_matmul()
        metadata_encoder = self._parse_metadata_encoder(model_version) if meta_encoder_version > 0 else None
        blocks = self._parse_block_stack(model_version, num_blocks)
        trunk_tip_bn = self._parse_bn()
        trunk_tip_activation = self._parse_activation(model_version)
        return TrunkDesc(
            name=name,
            num_blocks=num_blocks,
            trunk_num_channels=trunk_num_channels,
            initial_conv=initial_conv,
            initial_matmul=initial_matmul,
            sgf_metadata_encoder=metadata_encoder,
            blocks=blocks,
            trunk_tip_bn=trunk_tip_bn,
            trunk_tip_activation=trunk_tip_activation,
        )

    def _parse_policy_head(self, model_version: int) -> PolicyHeadDesc:
        name = self.s.read_token()
        p1_conv = self._parse_conv()
        g1_conv = self._parse_conv()
        g1_bn = self._parse_bn()
        g1_activation = self._parse_activation(model_version)
        gpool_to_bias_mul = self._parse_matmul()
        p1_bn = self._parse_bn()
        p1_activation = self._parse_activation(model_version)
        p2_conv = self._parse_conv()
        gpool_to_pass_mul = self._parse_matmul()
        gpool_to_pass_bias = None
        pass_activation = None
        gpool_to_pass_mul2 = None
        if model_version >= 15:
            gpool_to_pass_bias = self._parse_matbias()
            pass_activation = self._parse_activation(model_version)
            gpool_to_pass_mul2 = self._parse_matmul()
        return PolicyHeadDesc(
            name=name,
            p1_conv=p1_conv,
            g1_conv=g1_conv,
            g1_bn=g1_bn,
            g1_activation=g1_activation,
            gpool_to_bias_mul=gpool_to_bias_mul,
            p1_bn=p1_bn,
            p1_activation=p1_activation,
            p2_conv=p2_conv,
            gpool_to_pass_mul=gpool_to_pass_mul,
            gpool_to_pass_bias=gpool_to_pass_bias,
            pass_activation=pass_activation,
            gpool_to_pass_mul2=gpool_to_pass_mul2,
        )

    def _parse_value_head(self, model_version: int) -> ValueHeadDesc:
        name = self.s.read_token()
        return ValueHeadDesc(
            name=name,
            v1_conv=self._parse_conv(),
            v1_bn=self._parse_bn(),
            v1_activation=self._parse_activation(model_version),
            v2_mul=self._parse_matmul(),
            v2_bias=self._parse_matbias(),
            v2_activation=self._parse_activation(model_version),
            v3_mul=self._parse_matmul(),
            v3_bias=self._parse_matbias(),
            sv3_mul=self._parse_matmul(),
            sv3_bias=self._parse_matbias(),
            v_ownership_conv=self._parse_conv(),
        )


class FixedConv(nn.Module):
    def __init__(self, desc: ConvDesc):
        super().__init__()
        self.register_buffer("weight", torch.from_numpy(desc.weight.astype(np.float32)))
        self.padding = (
            desc.dilation_y * (desc.weight.shape[2] - 1) // 2,
            desc.dilation_x * (desc.weight.shape[3] - 1) // 2,
        )
        self.dilation = (desc.dilation_y, desc.dilation_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=self.padding, dilation=self.dilation)


class FixedScaleBias(nn.Module):
    def __init__(self, scale: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.register_buffer("bias", torch.from_numpy(bias.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FixedBatchNorm(FixedScaleBias):
    def __init__(self, desc: BatchNormDesc):
        super().__init__(desc.merged_scale, desc.merged_bias)


class FixedMatMul(nn.Module):
    def __init__(self, desc: MatMulDesc):
        super().__init__()
        self.register_buffer("weight", torch.from_numpy(desc.weight.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=0)


class FixedMatBias(nn.Module):
    def __init__(self, desc: MatBiasDesc):
        super().__init__()
        self.register_buffer("bias", torch.from_numpy(desc.bias.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias.view(1, -1, 1, 1)


class ActivationLayer(nn.Module):
    def __init__(self, desc: ActivationDesc):
        super().__init__()
        self.kind = desc.kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "ACTIVATION_IDENTITY":
            return x
        if self.kind == "ACTIVATION_RELU":
            return F.relu(x)
        if self.kind == "ACTIVATION_MISH":
            return x * torch.tanh(F.softplus(x))
        raise ValueError(f"Unsupported activation {self.kind}")


class SGFMetadataEncoderModule(nn.Module):
    def __init__(self, desc: SGFMetadataEncoderDesc):
        super().__init__()
        self.mul1 = FixedMatMul(desc.mul1)
        self.bias1 = FixedMatBias(desc.bias1)
        self.act1 = ActivationLayer(desc.act1)
        self.mul2 = FixedMatMul(desc.mul2)
        self.bias2 = FixedMatBias(desc.bias2)
        self.act2 = ActivationLayer(desc.act2)
        self.mul3 = FixedMatMul(desc.mul3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mul1(x)
        x = self.bias1(x)
        x = self.act1(x)
        x = self.mul2(x)
        x = self.bias2(x)
        x = self.act2(x)
        x = self.mul3(x)
        return x


class ExactGPool(nn.Module):
    def __init__(self, board_y: int, board_x: int, is_value_head: bool):
        super().__init__()
        mask_width = math.sqrt(board_y * board_x)
        mask_scale = np.array([mask_width * 0.1 - 1.4], dtype=np.float32)
        mask_quad = np.array([(mask_width - 14.0) * (mask_width - 14.0) * 0.01 - 0.1], dtype=np.float32)
        self.register_buffer("mask_scale", torch.from_numpy(mask_scale))
        self.register_buffer("mask_quad", torch.from_numpy(mask_quad))
        self.is_value_head = is_value_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        mean_scale = mean * self.mask_scale.view(1, 1, 1, 1)
        if self.is_value_head:
            third = mean * self.mask_quad.view(1, 1, 1, 1)
        else:
            third = torch.amax(x, dim=(2, 3), keepdim=True)
        return torch.cat([mean, mean_scale, third], dim=1)


class ResidualBlockModule(nn.Module):
    def __init__(self, desc: ResidualBlockDesc):
        super().__init__()
        self.pre_bn = FixedBatchNorm(desc.pre_bn)
        self.pre_act = ActivationLayer(desc.pre_act)
        self.regular_conv = FixedConv(desc.regular_conv)
        self.mid_bn = FixedBatchNorm(desc.mid_bn)
        self.mid_act = ActivationLayer(desc.mid_act)
        self.final_conv = FixedConv(desc.final_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_bn(x)
        y = self.pre_act(y)
        y = self.regular_conv(y)
        y = self.mid_bn(y)
        y = self.mid_act(y)
        y = self.final_conv(y)
        return x + y


class GlobalPoolingResidualBlockModule(nn.Module):
    def __init__(self, desc: GlobalPoolingResidualBlockDesc, board_y: int, board_x: int):
        super().__init__()
        self.pre_bn = FixedBatchNorm(desc.pre_bn)
        self.pre_act = ActivationLayer(desc.pre_act)
        self.regular_conv = FixedConv(desc.regular_conv)
        self.gpool_conv = FixedConv(desc.gpool_conv)
        self.gpool_bn = FixedBatchNorm(desc.gpool_bn)
        self.gpool_act = ActivationLayer(desc.gpool_act)
        self.gpool = ExactGPool(board_y, board_x, is_value_head=False)
        self.gpool_to_bias_mul = FixedMatMul(desc.gpool_to_bias_mul)
        self.mid_bn = FixedBatchNorm(desc.mid_bn)
        self.mid_act = ActivationLayer(desc.mid_act)
        self.final_conv = FixedConv(desc.final_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_bn(x)
        y = self.pre_act(y)
        reg = self.regular_conv(y)
        g = self.gpool_conv(y)
        g = self.gpool_bn(g)
        g = self.gpool_act(g)
        g = self.gpool(g)
        g = self.gpool_to_bias_mul(g)
        y = reg + g
        y = self.mid_bn(y)
        y = self.mid_act(y)
        y = self.final_conv(y)
        return x + y


def build_block_module(desc: BlockDesc, board_y: int, board_x: int) -> nn.Module:
    if isinstance(desc, ResidualBlockDesc):
        return ResidualBlockModule(desc)
    if isinstance(desc, GlobalPoolingResidualBlockDesc):
        return GlobalPoolingResidualBlockModule(desc, board_y, board_x)
    if isinstance(desc, NestedBottleneckResidualBlockDesc):
        return NestedBottleneckResidualBlockModule(desc, board_y, board_x)
    raise TypeError(f"Unsupported block type {type(desc)!r}")


class NestedBottleneckResidualBlockModule(nn.Module):
    def __init__(self, desc: NestedBottleneckResidualBlockDesc, board_y: int, board_x: int):
        super().__init__()
        self.pre_bn = FixedBatchNorm(desc.pre_bn)
        self.pre_act = ActivationLayer(desc.pre_act)
        self.pre_conv = FixedConv(desc.pre_conv)
        self.blocks = nn.ModuleList([build_block_module(block, board_y, board_x) for block in desc.blocks])
        self.post_bn = FixedBatchNorm(desc.post_bn)
        self.post_act = ActivationLayer(desc.post_act)
        self.post_conv = FixedConv(desc.post_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_bn(x)
        y = self.pre_act(y)
        y = self.pre_conv(y)
        for block in self.blocks:
            y = block(y)
        y = self.post_bn(y)
        y = self.post_act(y)
        y = self.post_conv(y)
        return x + y


class TrunkModule(nn.Module):
    def __init__(self, desc: TrunkDesc, board_y: int, board_x: int):
        super().__init__()
        self.initial_conv = FixedConv(desc.initial_conv)
        self.initial_matmul = FixedMatMul(desc.initial_matmul)
        self.metadata_encoder = SGFMetadataEncoderModule(desc.sgf_metadata_encoder) if desc.sgf_metadata_encoder is not None else None
        self.blocks = nn.ModuleList([build_block_module(block, board_y, board_x) for block in desc.blocks])
        self.trunk_tip_bn = FixedBatchNorm(desc.trunk_tip_bn)
        self.trunk_tip_activation = ActivationLayer(desc.trunk_tip_activation)

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor, input_meta: torch.Tensor | None) -> torch.Tensor:
        x = self.initial_conv(input_spatial) + self.initial_matmul(input_global)
        if self.metadata_encoder is not None:
            if input_meta is None:
                raise ValueError("Metadata encoder present but no metadata tensor was provided")
            x = x + self.metadata_encoder(input_meta)
        for block in self.blocks:
            x = block(x)
        x = self.trunk_tip_bn(x)
        x = self.trunk_tip_activation(x)
        return x


class PolicyHeadModule(nn.Module):
    def __init__(self, desc: PolicyHeadDesc, board_y: int, board_x: int):
        super().__init__()
        self.p1_conv = FixedConv(desc.p1_conv)
        self.g1_conv = FixedConv(desc.g1_conv)
        self.g1_bn = FixedBatchNorm(desc.g1_bn)
        self.g1_activation = ActivationLayer(desc.g1_activation)
        self.gpool = ExactGPool(board_y, board_x, is_value_head=False)
        self.gpool_to_bias_mul = FixedMatMul(desc.gpool_to_bias_mul)
        self.p1_bn = FixedBatchNorm(desc.p1_bn)
        self.p1_activation = ActivationLayer(desc.p1_activation)
        self.p2_conv = FixedConv(desc.p2_conv)
        self.gpool_to_pass_mul = FixedMatMul(desc.gpool_to_pass_mul)
        self.gpool_to_pass_bias = FixedMatBias(desc.gpool_to_pass_bias) if desc.gpool_to_pass_bias is not None else None
        self.pass_activation = ActivationLayer(desc.pass_activation) if desc.pass_activation is not None else None
        self.gpool_to_pass_mul2 = FixedMatMul(desc.gpool_to_pass_mul2) if desc.gpool_to_pass_mul2 is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.p1_conv(x)
        g = self.g1_conv(x)
        g = self.g1_bn(g)
        g = self.g1_activation(g)
        g = self.gpool(g)

        p = p + self.gpool_to_bias_mul(g)
        p = self.p1_bn(p)
        p = self.p1_activation(p)
        policy_spatial = self.p2_conv(p)

        policy_pass = self.gpool_to_pass_mul(g)
        if self.gpool_to_pass_bias is not None:
            policy_pass = self.gpool_to_pass_bias(policy_pass)
            policy_pass = self.pass_activation(policy_pass)
            policy_pass = self.gpool_to_pass_mul2(policy_pass)

        return policy_spatial, policy_pass


class ValueHeadModule(nn.Module):
    def __init__(self, desc: ValueHeadDesc, board_y: int, board_x: int):
        super().__init__()
        self.v1_conv = FixedConv(desc.v1_conv)
        self.v1_bn = FixedBatchNorm(desc.v1_bn)
        self.v1_activation = ActivationLayer(desc.v1_activation)
        self.gpool = ExactGPool(board_y, board_x, is_value_head=True)
        self.v2_mul = FixedMatMul(desc.v2_mul)
        self.v2_bias = FixedMatBias(desc.v2_bias)
        self.v2_activation = ActivationLayer(desc.v2_activation)
        self.v3_mul = FixedMatMul(desc.v3_mul)
        self.v3_bias = FixedMatBias(desc.v3_bias)
        self.sv3_mul = FixedMatMul(desc.sv3_mul)
        self.sv3_bias = FixedMatBias(desc.sv3_bias)
        self.v_ownership_conv = FixedConv(desc.v_ownership_conv)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = self.v1_conv(x)
        v = self.v1_bn(v)
        v = self.v1_activation(v)
        pooled = self.gpool(v)
        pooled = self.v2_mul(pooled)
        pooled = self.v2_bias(pooled)
        pooled = self.v2_activation(pooled)
        out_value = self.v3_bias(self.v3_mul(pooled)).squeeze(-1).squeeze(-1)
        out_score = self.sv3_bias(self.sv3_mul(pooled)).squeeze(-1).squeeze(-1)
        out_ownership = self.v_ownership_conv(v)
        return out_value, out_score, out_ownership


def approx_equal(a: float, b: float, eps: float = 1e-12) -> bool:
    return abs(a - b) <= eps


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 20.0,
        x + torch.log1p(-torch.exp(-x)),
        torch.log(torch.exp(x) - 1.0),
    )


def adjust_softplus_channel(raw: torch.Tensor, actual: float, default: float) -> torch.Tensor:
    if approx_equal(actual, default):
        return raw
    return softplus_inverse(F.softplus(raw) * (actual / default))


def adjust_shortterm_channel(raw: torch.Tensor, actual: float, default: float, model_version: int) -> torch.Tensor:
    if model_version < 10 or approx_equal(actual, default):
        return raw
    if model_version >= 14:
        return 2.0 * softplus_inverse(F.softplus(raw * 0.5) * math.sqrt(actual / default))
    return softplus_inverse(F.softplus(raw) * (actual / default))


class BinGzToOnnxModel(nn.Module):
    def __init__(self, desc: ModelDesc, board_y: int, board_x: int):
        super().__init__()
        self.desc = desc
        self.board_y = board_y
        self.board_x = board_x
        self.direct_policy_channels = int(desc.policy_head.p2_conv.weight.shape[0])
        self.direct_score_channels = int(desc.value_head.sv3_bias.bias.shape[0])
        self.trunk = TrunkModule(desc.trunk, board_y, board_x)
        self.policy_head = PolicyHeadModule(desc.policy_head, board_y, board_x)
        self.value_head = ValueHeadModule(desc.value_head, board_y, board_x)

    def _pack_policy(self, policy_spatial: torch.Tensor, policy_pass: torch.Tensor) -> torch.Tensor:
        batch = policy_spatial.shape[0]
        subset = torch.cat(
            [
                policy_spatial.reshape(batch, policy_spatial.shape[1], -1),
                policy_pass.squeeze(-1).squeeze(-1).unsqueeze(-1),
            ],
            dim=2,
        )
        hw1 = subset.shape[2]
        zeros1 = subset.new_zeros((batch, 1, hw1))

        if self.desc.model_version <= 11:
            return torch.cat([subset[:, 0:1, :], zeros1, zeros1, zeros1], dim=1)

        optimistic = subset[:, 1:2, :] if self.direct_policy_channels >= 2 else zeros1
        if self.direct_policy_channels >= 4:
            q_extras = subset[:, 2:4, :]
            zeros_mid = subset.new_zeros((batch, 2, hw1))
            return torch.cat([subset[:, 0:1, :], q_extras, zeros_mid, optimistic], dim=1)

        zeros_mid = subset.new_zeros((batch, 4, hw1))
        return torch.cat([subset[:, 0:1, :], zeros_mid, optimistic], dim=1)

    def _pack_score_outputs(self, out_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = out_score.shape[0]
        zero = out_score.new_zeros((batch, 1))
        pp = self.desc.postprocess

        if self.direct_score_channels >= 1:
            score_mean = out_score[:, 0:1] * (pp.score_mean_multiplier / DEFAULT_SCORE_PARAMS["score_mean_multiplier"])
        else:
            score_mean = zero
        if self.direct_score_channels >= 2:
            score_stdev = adjust_softplus_channel(
                out_score[:, 1:2],
                pp.score_stdev_multiplier,
                DEFAULT_SCORE_PARAMS["score_stdev_multiplier"],
            )
        else:
            score_stdev = zero
        if self.direct_score_channels >= 3:
            lead = out_score[:, 2:3] * (pp.lead_multiplier / DEFAULT_SCORE_PARAMS["lead_multiplier"])
        else:
            lead = zero
        if self.direct_score_channels >= 4:
            variance_time = adjust_softplus_channel(
                out_score[:, 3:4],
                pp.variance_time_multiplier,
                DEFAULT_SCORE_PARAMS["variance_time_multiplier"],
            )
        else:
            variance_time = zero
        if self.direct_score_channels >= 5:
            shortterm_value = adjust_shortterm_channel(
                out_score[:, 4:5],
                pp.shortterm_value_error_multiplier,
                DEFAULT_SCORE_PARAMS["shortterm_value_error_multiplier"],
                self.desc.model_version,
            )
        else:
            shortterm_value = zero
        if self.direct_score_channels >= 6:
            shortterm_score = adjust_shortterm_channel(
                out_score[:, 5:6],
                pp.shortterm_score_error_multiplier,
                DEFAULT_SCORE_PARAMS["shortterm_score_error_multiplier"],
                self.desc.model_version,
            )
        else:
            shortterm_score = zero

        misc = torch.cat([score_mean, score_stdev, lead, variance_time] + [zero] * 6, dim=1)
        more = torch.cat([shortterm_value, shortterm_score] + [zero] * 6, dim=1)
        return misc, more

    def forward(self, input_spatial: torch.Tensor, input_global: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_global_4d = input_global.unsqueeze(-1).unsqueeze(-1)
        input_meta = None
        if self.desc.num_input_meta_channels > 0:
            input_meta = input_global.new_zeros(
                (input_global.shape[0], self.desc.num_input_meta_channels, 1, 1)
            )

        trunk = self.trunk(input_spatial, input_global_4d, input_meta)
        policy_spatial, policy_pass = self.policy_head(trunk)
        out_value, out_score, out_ownership = self.value_head(trunk)

        out_policy = self._pack_policy(policy_spatial, policy_pass)
        out_miscvalue, out_moremiscvalue = self._pack_score_outputs(out_score)
        out_ownership = out_ownership.reshape(out_ownership.shape[0], -1)
        return out_policy, out_value, out_miscvalue, out_moremiscvalue, out_ownership


def load_model_desc(path: str) -> ModelDesc:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
    else:
        with open(path, "rb") as f:
            data = f.read()
    stream = TokenStream(data)
    parser = BinModelParser(stream)
    return parser.parse()


def build_model_config_metadata(desc: ModelDesc, board_y: int, board_x: int, zero_metadata: bool) -> str:
    payload = {
        "source_format": "katago_bin_gz",
        "source_model_name": desc.name,
        "source_model_version": desc.model_version,
        "board_x": board_x,
        "board_y": board_y,
        "fixed_exact_board_size": True,
        "onnx_input_names": ["input_spatial", "input_global"],
        "onnx_output_names": ["out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"],
        "policy_layout": "Kx(HW+1), compatible with cpp/neuralnet/onnxbackend.cpp",
        "misc_layout": "10 channels, first 4 mapped from .bin.gz score head, rest zero",
        "moremisc_layout": "8 channels, first 2 mapped from .bin.gz score head, rest zero",
        "meta_encoder_version": desc.meta_encoder_version,
        "meta_encoder_export_mode": "zeros_baked_in" if zero_metadata else "not_used",
        "postprocess_params_from_bin": {
            "td_score_multiplier": desc.postprocess.td_score_multiplier,
            "score_mean_multiplier": desc.postprocess.score_mean_multiplier,
            "score_stdev_multiplier": desc.postprocess.score_stdev_multiplier,
            "lead_multiplier": desc.postprocess.lead_multiplier,
            "variance_time_multiplier": desc.postprocess.variance_time_multiplier,
            "shortterm_value_error_multiplier": desc.postprocess.shortterm_value_error_multiplier,
            "shortterm_score_error_multiplier": desc.postprocess.shortterm_score_error_multiplier,
            "output_scale_multiplier": desc.postprocess.output_scale_multiplier,
        },
        "onnx_runtime_expected_postprocess_defaults": DEFAULT_SCORE_PARAMS,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def add_metadata(
    onnx_path: str,
    desc: ModelDesc,
    board_y: int,
    board_x: int,
    model_config_json: str,
) -> None:
    model = onnx.load(onnx_path)
    metadata = {
        "modelVersion": str(desc.model_version),
        "name": desc.name,
        "num_spatial_inputs": str(desc.num_input_channels),
        "num_global_inputs": str(desc.num_input_global_channels),
        "has_mask": "false",
        "pos_len_x": str(board_x),
        "pos_len_y": str(board_y),
        "is_qat": "false",
        "is_simplified": "false",
        "is_int8": "false",
        "model_config": model_config_json,
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    model.producer_name = "bin_gz_to_onnx.py"
    onnx.save(model, onnx_path)


def validate_export(
    onnx_path: str,
    desc: ModelDesc,
    board_y: int,
    board_x: int,
) -> None:
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipped runtime validation", file=sys.stderr)
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    spatial = np.zeros((1, desc.num_input_channels, board_y, board_x), dtype=np.float32)
    global_input = np.zeros((1, desc.num_input_global_channels), dtype=np.float32)
    outputs = sess.run(
        ["out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"],
        {"input_spatial": spatial, "input_global": global_input},
    )

    expected_policy_channels = 4 if desc.model_version <= 11 else 6
    hw1 = board_y * board_x + 1
    if tuple(outputs[0].shape) != (1, expected_policy_channels, hw1):
        raise RuntimeError(f"Unexpected out_policy shape: {outputs[0].shape}")
    if tuple(outputs[1].shape) != (1, 3):
        raise RuntimeError(f"Unexpected out_value shape: {outputs[1].shape}")
    if tuple(outputs[2].shape) != (1, 10):
        raise RuntimeError(f"Unexpected out_miscvalue shape: {outputs[2].shape}")
    if tuple(outputs[3].shape) != (1, 8):
        raise RuntimeError(f"Unexpected out_moremiscvalue shape: {outputs[3].shape}")
    if tuple(outputs[4].shape) != (1, board_y * board_x):
        raise RuntimeError(f"Unexpected out_ownership shape: {outputs[4].shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a KataGo .bin/.bin.gz model into an ONNX file compatible with this fork's ONNX backend."
    )
    parser.add_argument("--input", required=True, help="Path to the source .bin or .bin.gz model")
    parser.add_argument("--output", required=True, help="Path to the output .onnx file")
    parser.add_argument("--board-size", type=int, default=None, help="Square board size shortcut, e.g. 19")
    parser.add_argument("--board-x", type=int, default=None, help="Board width")
    parser.add_argument("--board-y", type=int, default=None, help="Board height")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the destination file if it already exists")
    parser.add_argument("--skip-verify", action="store_true", help="Skip onnx.checker and onnxruntime validation")
    return parser.parse_args()


def resolve_board_size(args: argparse.Namespace) -> tuple[int, int]:
    if args.board_size is not None:
        if args.board_x is not None or args.board_y is not None:
            raise SystemExit("Use either --board-size or --board-x/--board-y, not both")
        return args.board_size, args.board_size
    if args.board_x is not None and args.board_y is not None:
        return args.board_y, args.board_x
    return 19, 19


def main() -> None:
    args = parse_args()
    board_y, board_x = resolve_board_size(args)

    if os.path.exists(args.output) and not args.overwrite:
        raise SystemExit(f"Output file already exists: {args.output} (pass --overwrite to replace it)")
    if not args.output.endswith(".onnx"):
        raise SystemExit("Output path must end with .onnx")

    desc = load_model_desc(args.input)

    if desc.model_version < 9:
        raise SystemExit(
            f"Model version {desc.model_version} is not compatible with the current ONNX backend path in this fork. "
            "The backend only postprocesses ONNX outputs for version >= 9."
        )

    if desc.postprocess.output_scale_multiplier != 1.0:
        raise SystemExit(
            "Unexpected non-unit output_scale_multiplier in source model. "
            "This script currently expects raw .bin/.bin.gz exports before backend-side scale8 rewriting."
        )

    if desc.num_input_meta_channels > 0:
        print(
            "Warning: source model uses SGF metadata inputs. The exported ONNX file bakes a zero metadata tensor "
            "because the current ONNX backend path only feeds input_spatial and input_global.",
            file=sys.stderr,
        )

    model = BinGzToOnnxModel(desc, board_y, board_x)
    model.eval()

    dummy_spatial = torch.zeros((1, desc.num_input_channels, board_y, board_x), dtype=torch.float32)
    dummy_global = torch.zeros((1, desc.num_input_global_channels), dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_spatial, dummy_global),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=["input_spatial", "input_global"],
            output_names=["out_policy", "out_value", "out_miscvalue", "out_moremiscvalue", "out_ownership"],
            dynamic_axes={
                "input_spatial": {0: "batch"},
                "input_global": {0: "batch"},
                "out_policy": {0: "batch"},
                "out_value": {0: "batch"},
                "out_miscvalue": {0: "batch"},
                "out_moremiscvalue": {0: "batch"},
                "out_ownership": {0: "batch"},
            },
        )

    model_config_json = build_model_config_metadata(
        desc=desc,
        board_y=board_y,
        board_x=board_x,
        zero_metadata=desc.num_input_meta_channels > 0,
    )
    add_metadata(args.output, desc, board_y, board_x, model_config_json)

    if not args.skip_verify:
        validate_export(args.output, desc, board_y, board_x)

    expected_policy_channels = 4 if desc.model_version <= 11 else 6
    print(
        json.dumps(
            {
                "status": "ok",
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(args.output),
                "model_name": desc.name,
                "model_version": desc.model_version,
                "board_x": board_x,
                "board_y": board_y,
                "num_input_channels": desc.num_input_channels,
                "num_input_global_channels": desc.num_input_global_channels,
                "num_input_meta_channels": desc.num_input_meta_channels,
                "policy_output_channels": expected_policy_channels,
                "verify": not args.skip_verify,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
