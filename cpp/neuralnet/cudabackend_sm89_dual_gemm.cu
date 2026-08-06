/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed-shape wrapper around CUTLASS examples/45_dual_gemm.
 * CUTLASS commit: 7127592069c2fe01b041e174ba4345ef9b279671
 **************************************************************************************************/

#include "../neuralnet/cudabackend_sm89_dual_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
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

template<int Count>
class Half2TanhSwiGLUOutputOp {
 public:
  using ElementOutput = Element;
  using ElementAccumulator = Element;
  using ElementCompute = Element;
  using FragmentOutput = cutlass::Array<Element, Count>;
  using FragmentAccumulator = cutlass::Array<Element, Count>;
  struct Params {};

  CUTLASS_HOST_DEVICE
  explicit Half2TanhSwiGLUOutputOp(Params const&) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return true; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) { assert(false); }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
    FragmentAccumulator const& lhs,
    FragmentAccumulator const& rhs
  ) const {
    cutlass::multiplies<FragmentOutput> mul;
    cutlass::multiply_add<FragmentOutput> fma;
    cutlass::fast_tanh_op<FragmentOutput> tanh;
    Element half = cutlass::constants::half<Element>();
    FragmentOutput sigmoid = fma(tanh(mul(lhs, half)), half, half);
    return mul(mul(lhs, sigmoid), rhs);
  }

  CUTLASS_HOST_DEVICE
  Element operator()(Element const& lhs, Element const& rhs) const {
    Element half = cutlass::constants::half<Element>();
    Element sigmoid = cutlass::fast_tanh(lhs * half) * half + half;
    return lhs * sigmoid * rhs;
  }
};

using SwiGLUHalf2TanhOutputOp = Half2TanhSwiGLUOutputOp<8>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

template<typename SwiGLUOp>
using DualGemmT = cutlass::gemm::device::DualGemm<
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
  SwiGLUOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<KATAGO_DUAL_GEMM_SWIZZLE>,
  3,
  false,
  false,
  false,
  8,
  8>;

using DualGemm = DualGemmT<SwiGLUOutputOp>;
using DualGemmHalf2Tanh = DualGemmT<SwiGLUHalf2TanhOutputOp>;

template<typename Gemm>
typename Gemm::Arguments makeArguments(
  const half* weights,
  const half* input,
  half* output
) {
  using Layout = cutlass::layout::RowMajor;
  typename Gemm::TensorRefC nullC;
  typename Gemm::TensorRefD nullD;
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
  DualGemmHalf2Tanh half2TanhOp;
  bool useHalf2Tanh;
  bool initialized;

  Impl(const half* weights_, bool useHalf2Tanh_)
    : weights(weights_), op(), half2TanhOp(),
      useHalf2Tanh(useHalf2Tanh_), initialized(false)
  {}

  template<typename Gemm>
  bool applyImpl(Gemm& gemm, const half* input, half* output, cudaStream_t stream) {
    typename Gemm::Arguments args = makeArguments<Gemm>(weights, input, output);
    cutlass::Status status;
    if(!initialized) {
      status = gemm.can_implement(args);
      if(status != cutlass::Status::kSuccess)
        return false;
      status = gemm.initialize(args, nullptr, stream);
      if(status != cutlass::Status::kSuccess)
        return false;
      initialized = true;
    }
    else {
      status = gemm.update(args, nullptr);
      if(status != cutlass::Status::kSuccess)
        return false;
    }
    return gemm.run(stream) == cutlass::Status::kSuccess;
  }

  bool apply(const half* input, half* output, cudaStream_t stream) {
    return useHalf2Tanh
      ? applyImpl(half2TanhOp, input, output, stream)
      : applyImpl(op, input, output, stream);
  }
};

Sm89DualGemmSwiGLUB13::Sm89DualGemmSwiGLUB13(
  const half* weights,
  bool useHalf2Tanh
)
  : impl(std::make_unique<Impl>(weights, useHalf2Tanh))
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
