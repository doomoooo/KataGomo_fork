/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed-shape CUTLASS GEMMs for B13 residual projections and nested preConv.
 * CUTLASS commit: 7127592069c2fe01b041e174ba4345ef9b279671
 **************************************************************************************************/

#include "../neuralnet/cudabackend_sm89_linear2_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace Sm89Backend {
namespace {

constexpr int B = 13;
constexpr int S = 361;
constexpr int Tokens = B * S;
constexpr int InChannels = 1152;
constexpr int OutChannels = 384;
constexpr int AttentionChannels = 384;
constexpr int PreConvInChannels = 768;
constexpr int PostConvInChannels = 384;
constexpr int PostConvOutChannels = 768;

using Element = cutlass::half_t;
using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, 8, Element, float>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using Gemm = cutlass::gemm::device::Gemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  4,
  8,
  8>;

using PreConvEpilogue = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using PreConvGemm = cutlass::gemm::device::Gemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  PreConvEpilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  5,
  8,
  8>;

using PostConvGemm = cutlass::gemm::device::Gemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,
  8,
  8>;

using DefaultPostConvKernel = typename PostConvGemm::GemmKernel;
using PostConvMma = typename DefaultPostConvKernel::Mma;
using DefaultPostConvEpilogue = typename DefaultPostConvKernel::Epilogue;
using DefaultPostConvIterator = typename DefaultPostConvEpilogue::OutputTileIterator;
using PostConvSwizzle = typename DefaultPostConvKernel::ThreadblockSwizzle;

template<typename BaseIterator>
class PostConvBnOutputTileIterator : public BaseIterator {
 public:
  using Base = BaseIterator;
  using ThreadMap = typename Base::ThreadMap;
  using Element = typename Base::Element;
  using Layout = typename Base::Layout;
  using TensorRef = typename Base::TensorRef;
  using ConstTensorRef = typename Base::ConstTensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using LongIndex = typename Base::LongIndex;
  using Fragment = typename Base::Fragment;
  using AccessType = typename Base::AccessType;
  using Mask = typename Base::Mask;

  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kIterations = Base::kIterations;

  struct Params : public Base::Params {
    Element* activatedOutput;
    const Element* scale;
    const Element* bias;

    CUTLASS_HOST_DEVICE
    Params()
      : Base::Params(), activatedOutput(nullptr), scale(nullptr), bias(nullptr)
    {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
      : Base::Params(layout), activatedOutput(nullptr), scale(nullptr), bias(nullptr)
    {}
  };

 private:
  Base activatedIterator;
  const Element* scale;
  const Element* bias;

 public:
  CUTLASS_DEVICE
  PostConvBnOutputTileIterator(
    Params const& params,
    Element* pointer,
    TensorCoord extent,
    int threadIdx,
    TensorCoord threadblockOffset = TensorCoord(),
    int const* indices = nullptr
  ) : Base(params, pointer, extent, threadIdx, threadblockOffset, indices),
      activatedIterator(
        params, params.activatedOutput, extent, threadIdx, threadblockOffset, indices
      ),
      scale(params.scale),
      bias(params.bias)
  {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment, int64_t byteOffset) const {
    Base::store_with_byte_offset(fragment, byteOffset);

    Fragment transformed = fragment;
    AccessType* accesses = reinterpret_cast<AccessType*>(&transformed);
    int startColumn = Base::thread_start_column();

    CUTLASS_PRAGMA_UNROLL
    for(int cluster = 0; cluster < ThreadMap::Iterations::kCluster; cluster++) {
      CUTLASS_PRAGMA_UNROLL
      for(int group = 0; group < ThreadMap::Iterations::kGroup; group++) {
        CUTLASS_PRAGMA_UNROLL
        for(int row = 0; row < ThreadMap::Iterations::kRow; row++) {
          int fragmentRow = row + ThreadMap::Iterations::kRow *
            (group + ThreadMap::Iterations::kGroup * cluster);
          CUTLASS_PRAGMA_UNROLL
          for(int column = 0; column < ThreadMap::Iterations::kColumn; column++) {
            int outputColumn = startColumn + column * ThreadMap::Delta::kColumn;
            if(outputColumn < PostConvOutChannels) {
              AccessType& access = accesses[
                fragmentRow * ThreadMap::Iterations::kColumn + column
              ];
              CUTLASS_PRAGMA_UNROLL
              for(int element = 0; element < kElementsPerAccess; element++) {
                int channel = outputColumn + element;
                Element xElement = access[element];
                half x = __ushort_as_half(xElement.storage);
                half s = __ushort_as_half(scale[channel].storage);
                half b = __ushort_as_half(bias[channel].storage);
                half affine = __hfma(x, s, b);
                float a = __half2float(affine);
                half activated = __float2half(a / (1.0f + expf(-a)));
                access[element] = Element::bitcast(__half_as_ushort(activated));
              }
            }
          }
        }
      }
    }
    activatedIterator.store_with_byte_offset(transformed, byteOffset);
  }

  CUTLASS_DEVICE
  void store(Fragment const& fragment) const {
    store_with_byte_offset(fragment, 0);
  }

  CUTLASS_HOST_DEVICE
  PostConvBnOutputTileIterator& operator++() {
    Base::operator++();
    ++activatedIterator;
    return *this;
  }
};

using PostConvBnIterator = PostConvBnOutputTileIterator<DefaultPostConvIterator>;
using PostConvBnEpilogue = cutlass::epilogue::threadblock::Epilogue<
  typename DefaultPostConvEpilogue::Shape,
  typename DefaultPostConvEpilogue::WarpMmaOperator,
  DefaultPostConvEpilogue::kPartitionsK,
  PostConvBnIterator,
  typename DefaultPostConvEpilogue::AccumulatorFragmentIterator,
  typename DefaultPostConvEpilogue::WarpTileIterator,
  typename DefaultPostConvEpilogue::SharedLoadIterator,
  typename DefaultPostConvEpilogue::OutputOp,
  typename DefaultPostConvEpilogue::Padding,
  DefaultPostConvEpilogue::Base::kFragmentsPerIteration>;
using PostConvBnKernel = cutlass::gemm::kernel::Gemm<
  PostConvMma, PostConvBnEpilogue, PostConvSwizzle, false>;

Gemm::Arguments makeArguments(
  const half* weights,
  const half* input,
  half* output,
  int inChannels
) {
  using Layout = cutlass::layout::RowMajor;
  return {
    {Tokens, OutChannels, inChannels},
    {reinterpret_cast<const Element*>(input), Layout(inChannels)},
    {reinterpret_cast<const Element*>(weights), Layout(OutChannels)},
    {reinterpret_cast<const Element*>(output), Layout(OutChannels)},
    {reinterpret_cast<Element*>(output), Layout(OutChannels)},
    {1.0f, 1.0f}
  };
}

} // namespace

struct Sm89Linear2GemmB13::Impl {
  const half* weights;
  Gemm op;
  bool initialized;

  explicit Impl(const half* weights_)
    : weights(weights_), op(), initialized(false)
  {}

  bool applyAccumulate(const half* input, half* output, cudaStream_t stream) {
    Gemm::Arguments args = makeArguments(weights, input, output, InChannels);
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

Sm89Linear2GemmB13::Sm89Linear2GemmB13(const half* weights)
  : impl(std::make_unique<Impl>(weights))
{}

Sm89Linear2GemmB13::~Sm89Linear2GemmB13() = default;

bool Sm89Linear2GemmB13::applyAccumulate(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize != B || seqLen != S || inChannels != InChannels ||
     outChannels != OutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->applyAccumulate(input, output, stream);
}

struct Sm89OutProjGemmB13::Impl {
  const half* weights;
  Gemm op;
  bool initialized;

  explicit Impl(const half* weights_)
    : weights(weights_), op(), initialized(false)
  {}

  bool applyAccumulate(const half* input, half* output, cudaStream_t stream) {
    Gemm::Arguments args = makeArguments(weights, input, output, AttentionChannels);
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

Sm89OutProjGemmB13::Sm89OutProjGemmB13(const half* weights)
  : impl(std::make_unique<Impl>(weights))
{}

Sm89OutProjGemmB13::~Sm89OutProjGemmB13() = default;

bool Sm89OutProjGemmB13::applyAccumulate(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize != B || seqLen != S || inChannels != AttentionChannels ||
     outChannels != OutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->applyAccumulate(input, output, stream);
}

struct Sm89PreConvGemmB13::Impl {
  const half* weights;
  PreConvGemm op;
  bool initialized;

  explicit Impl(const half* weights_)
    : weights(weights_), op(), initialized(false)
  {}

  bool apply(const half* input, half* output, cudaStream_t stream) {
    using Layout = cutlass::layout::RowMajor;
    PreConvGemm::Arguments args(
      {Tokens, OutChannels, PreConvInChannels},
      {reinterpret_cast<const Element*>(input), Layout(PreConvInChannels)},
      {reinterpret_cast<const Element*>(weights), Layout(OutChannels)},
      {reinterpret_cast<const Element*>(output), Layout(OutChannels)},
      {reinterpret_cast<Element*>(output), Layout(OutChannels)},
      {1.0f, 0.0f}
    );
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

Sm89PreConvGemmB13::Sm89PreConvGemmB13(const half* weights)
  : impl(std::make_unique<Impl>(weights))
{}

Sm89PreConvGemmB13::~Sm89PreConvGemmB13() = default;

bool Sm89PreConvGemmB13::apply(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize != B || seqLen != S || inChannels != PreConvInChannels ||
     outChannels != OutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->apply(input, output, stream);
}

struct Sm89PostConvGemmB13::Impl {
  const half* weights;
  PostConvGemm op;
  bool initialized;

  explicit Impl(const half* weights_)
    : weights(weights_), op(), initialized(false)
  {}

  bool applyAccumulate(const half* input, half* output, cudaStream_t stream) {
    using Layout = cutlass::layout::RowMajor;
    PostConvGemm::Arguments args(
      {Tokens, PostConvOutChannels, PostConvInChannels},
      {reinterpret_cast<const Element*>(input), Layout(PostConvInChannels)},
      {reinterpret_cast<const Element*>(weights), Layout(PostConvOutChannels)},
      {reinterpret_cast<const Element*>(output), Layout(PostConvOutChannels)},
      {reinterpret_cast<Element*>(output), Layout(PostConvOutChannels)},
      {1.0f, 1.0f}
    );
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

Sm89PostConvGemmB13::Sm89PostConvGemmB13(const half* weights)
  : impl(std::make_unique<Impl>(weights))
{}

Sm89PostConvGemmB13::~Sm89PostConvGemmB13() = default;

bool Sm89PostConvGemmB13::applyAccumulate(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize != B || seqLen != S || inChannels != PostConvInChannels ||
     outChannels != PostConvOutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->applyAccumulate(input, output, stream);
}

struct Sm89PostConvBnGemmB13::Impl {
  typename PostConvBnKernel::Params params;
  bool initialized;

  Impl(const half* weights, const half* bnScale, const half* bnBias)
    : initialized(true)
  {
    cutlass::gemm::GemmCoord problem(Tokens, PostConvOutChannels, PostConvInChannels);
    cutlass::gemm::GemmCoord gridShape = PostConvSwizzle::get_tiled_shape(
      problem,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      1
    );
    using Layout = cutlass::layout::RowMajor;
    Layout inputLayout(PostConvInChannels);
    Layout outputLayout(PostConvOutChannels);
    typename PostConvMma::IteratorA::TensorRef nullInput(nullptr, inputLayout);
    typename PostConvMma::IteratorB::TensorRef weightRef(
      reinterpret_cast<Element*>(const_cast<half*>(weights)),
      Layout(PostConvOutChannels)
    );
    typename PostConvBnIterator::TensorRef nullOutput(nullptr, outputLayout);
    params = typename PostConvBnKernel::Params(
      problem,
      gridShape,
      nullInput,
      weightRef,
      nullOutput,
      nullOutput,
      typename Epilogue::Params(1.0f, 1.0f)
    );
    params.params_D.scale = reinterpret_cast<const Element*>(bnScale);
    params.params_D.bias = reinterpret_cast<const Element*>(bnBias);

    int smemSize = int(sizeof(typename PostConvBnKernel::SharedStorage));
    if(smemSize >= 48 * 1024)
      initialized = cudaFuncSetAttribute(
        cutlass::Kernel<PostConvBnKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smemSize
      ) == cudaSuccess;
  }

  bool apply(
    const half* input,
    half* residualOutput,
    half* activatedOutput,
    cudaStream_t stream
  ) {
    if(!initialized)
      return false;
    params.ref_A.reset(reinterpret_cast<Element*>(const_cast<half*>(input)));
    params.ref_C.reset(reinterpret_cast<Element*>(residualOutput));
    params.ref_D.reset(reinterpret_cast<Element*>(residualOutput));
    params.params_D.activatedOutput = reinterpret_cast<Element*>(activatedOutput);
    dim3 grid = PostConvSwizzle::get_grid_shape(params.grid_tiled_shape);
    dim3 block(PostConvBnKernel::kThreadCount, 1, 1);
    int smemSize = int(sizeof(typename PostConvBnKernel::SharedStorage));
    cutlass::Kernel<PostConvBnKernel><<<grid, block, smemSize, stream>>>(params);
    return cudaPeekAtLastError() == cudaSuccess;
  }
};

Sm89PostConvBnGemmB13::Sm89PostConvBnGemmB13(
  const half* weights,
  const half* bnScale,
  const half* bnBias
) : impl(std::make_unique<Impl>(weights, bnScale, bnBias))
{}

Sm89PostConvBnGemmB13::~Sm89PostConvBnGemmB13() = default;

bool Sm89PostConvBnGemmB13::applyAccumulateAndActivate(
  const half* input,
  half* residualOutput,
  half* activatedOutput,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize != B || seqLen != S || inChannels != PostConvInChannels ||
     outChannels != PostConvOutChannels || input == nullptr ||
     residualOutput == nullptr || activatedOutput == nullptr)
    return false;
  return impl->apply(input, residualOutput, activatedOutput, stream);
}

} // namespace Sm89Backend
