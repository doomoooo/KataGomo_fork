/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed-shape wrapper around CUTLASS examples/45_dual_gemm.
 * CUTLASS commit: 7127592069c2fe01b041e174ba4345ef9b279671
 **************************************************************************************************/

#include "../neuralnet/cudabackend_sm89_dual_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

namespace Sm89Backend {
namespace {

constexpr int B = 13;
constexpr int S = 361;
constexpr int Tokens = B * S;
constexpr int Channels = 384;
constexpr int FfnChannels = 1152;

using Element = cutlass::half_t;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using SwiGLUOutputOp = cutlass::epilogue::thread::LeftSiLUAndMul<
  Element, 8, Element, float>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using DualGemm = cutlass::gemm::device::DualGemm<
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::layout::RowMajor,
  cutlass::layout::RowMajor,
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  EpilogueOutputOp,
  SwiGLUOutputOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<KATAGO_DUAL_GEMM_SWIZZLE>,
  3,
  false,
  false,
  false,
  8,
  8>;

DualGemm::Arguments makeArguments(
  const half* weights,
  const half* input,
  half* output
) {
  using Layout = cutlass::layout::RowMajor;
  DualGemm::TensorRefC nullC;
  DualGemm::TensorRefD nullD;
  return {
    cutlass::gemm::DualGemmMode::kGemm,
    {Tokens, FfnChannels, Channels},
    {reinterpret_cast<const Element*>(input), Layout(Channels)},
    {reinterpret_cast<const Element*>(weights), Layout(FfnChannels)},
    nullC,
    nullD,
    {reinterpret_cast<const Element*>(weights + (size_t)FfnChannels * Channels), Layout(FfnChannels)},
    nullC,
    nullD,
    {reinterpret_cast<Element*>(output), Layout(FfnChannels)},
    {1.0f, 0.0f},
    {1.0f, 0.0f},
    {},
    1
  };
}

} // namespace

struct Sm89DualGemmSwiGLUB13::Impl {
  const half* weights;
  DualGemm op;
  bool initialized;

  explicit Impl(const half* weights_)
    : weights(weights_), op(), initialized(false)
  {}

  bool apply(const half* input, half* output, cudaStream_t stream) {
    DualGemm::Arguments args = makeArguments(weights, input, output);
    cutlass::Status status;
    if(!initialized) {
      status = op.can_implement(args);
      if(status != cutlass::Status::kSuccess)
        return false;
      status = op.initialize(args, nullptr, stream);
      if(status != cutlass::Status::kSuccess)
        return false;
      initialized = true;
    }
    else {
      status = op.update(args, nullptr);
      if(status != cutlass::Status::kSuccess)
        return false;
    }
    return op.run(stream) == cutlass::Status::kSuccess;
  }
};

Sm89DualGemmSwiGLUB13::Sm89DualGemmSwiGLUB13(const half* weights)
  : impl(std::make_unique<Impl>(weights))
{}

Sm89DualGemmSwiGLUB13::~Sm89DualGemmSwiGLUB13() = default;

bool Sm89DualGemmSwiGLUB13::apply(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int ffnChannels,
  cudaStream_t stream
) {
  if(batchSize != B || seqLen != S || inChannels != Channels ||
     ffnChannels != FfnChannels || input == nullptr || output == nullptr)
    return false;
  return impl->apply(input, output, stream);
}

} // namespace Sm89Backend
