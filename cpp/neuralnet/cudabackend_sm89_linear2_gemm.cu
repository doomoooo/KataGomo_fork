/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed-channel/board CUTLASS GEMMs for residual projections and nested preConv.
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

constexpr int S = 361;
constexpr int InChannels = 1152;
constexpr int OutChannels = 384;
constexpr int AttentionChannels = 384;
constexpr int PreConvInChannels = 768;
constexpr int PostConvInChannels = 384;
constexpr int PostConvOutChannels = 768;

using Element = cutlass::half_t;
using Epilogue = cutlass::epilogue::thread::LinearCombination<Element, 8, Element, float>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using PreConvEpilogue = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;

template<
  int ThreadblockM, int ThreadblockN, int WarpM, int WarpN,
  int Stages, int Swizzle, typename OutputOp>
using GemmT = cutlass::gemm::device::Gemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, 32>,
  cutlass::gemm::GemmShape<WarpM, WarpN, 32>,
  InstructionShape,
  OutputOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
  Stages,
  8,
  8>;

// Retained final geometries for the two fused BN epilogues. The standalone
// projection classes below expose every historically positive geometry through
// a runtime-selected, prelinked CUTLASS instantiation.
using Linear2BnBaseGemm = GemmT<128,128,64,64,4,1,Epilogue>;
using PostConvGemm = GemmT<128,128,64,64,3,1,Epilogue>;
using FusedThreadblockShape = cutlass::gemm::GemmShape<128,128,32>;

class GemmRunner {
 public:
  virtual ~GemmRunner() = default;
  virtual bool run(
    const half* input, half* output, int tokens,
    int inChannels, int outChannels, float beta, cudaStream_t stream) = 0;
};

template<typename DeviceGemm>
class GemmRunnerT final : public GemmRunner {
 public:
  explicit GemmRunnerT(const half* weights_)
    : weights(weights_), initialized(false) {}

  bool run(
    const half* input, half* output, int tokens,
    int inChannels, int outChannels, float beta, cudaStream_t stream) override {
    using Layout = cutlass::layout::RowMajor;
    typename DeviceGemm::Arguments args(
      {tokens, outChannels, inChannels},
      {reinterpret_cast<const Element*>(input), Layout(inChannels)},
      {reinterpret_cast<const Element*>(weights), Layout(outChannels)},
      {reinterpret_cast<const Element*>(output), Layout(outChannels)},
      {reinterpret_cast<Element*>(output), Layout(outChannels)},
      {1.0f, beta}
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

 private:
  const half* weights;
  DeviceGemm op;
  bool initialized;
};

template<
  int ThreadblockM, int ThreadblockN, int WarpM, int WarpN,
  int Stages, int Swizzle, typename OutputOp>
std::unique_ptr<GemmRunner> makeGemmRunner(const half* weights) {
  using DeviceGemm = GemmT<
    ThreadblockM,ThreadblockN,WarpM,WarpN,Stages,Swizzle,OutputOp>;
  return std::make_unique<GemmRunnerT<DeviceGemm>>(weights);
}

std::unique_ptr<GemmRunner> makeLinear2Runner(
  const half* weights, const std::string& tactic) {
  if(tactic == "m128-n128-k32-w64-n32-s3-sw1")
    return makeGemmRunner<128,128,64,32,3,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n32-s4-sw1")
    return makeGemmRunner<128,128,64,32,4,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s3-sw1")
    return makeGemmRunner<128,128,64,64,3,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s4-sw1")
    return makeGemmRunner<128,128,64,64,4,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s5-sw1")
    return makeGemmRunner<128,128,64,64,5,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s6-sw1")
    return makeGemmRunner<128,128,64,64,6,1,Epilogue>(weights);
  return nullptr;
}

std::unique_ptr<GemmRunner> makeOutProjRunner(
  const half* weights, const std::string& tactic) {
  if(tactic == "m128-n128-k32-w64-n32-s2-sw1")
    return makeGemmRunner<128,128,64,32,2,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n32-s3-sw1")
    return makeGemmRunner<128,128,64,32,3,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n32-s4-sw1")
    return makeGemmRunner<128,128,64,32,4,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s3-sw1")
    return makeGemmRunner<128,128,64,64,3,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s4-sw1")
    return makeGemmRunner<128,128,64,64,4,1,Epilogue>(weights);
  return nullptr;
}

std::unique_ptr<GemmRunner> makePreConvRunner(
  const half* weights, const std::string& tactic) {
  if(tactic == "m128-n128-k32-w64-n32-s3-sw1")
    return makeGemmRunner<128,128,64,32,3,1,PreConvEpilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n32-s4-sw1")
    return makeGemmRunner<128,128,64,32,4,1,PreConvEpilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s3-sw1")
    return makeGemmRunner<128,128,64,64,3,1,PreConvEpilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s4-sw1")
    return makeGemmRunner<128,128,64,64,4,1,PreConvEpilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s5-sw1")
    return makeGemmRunner<128,128,64,64,5,1,PreConvEpilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s6-sw1")
    return makeGemmRunner<128,128,64,64,6,1,PreConvEpilogue>(weights);
  return nullptr;
}

std::unique_ptr<GemmRunner> makePostConvRunner(
  const half* weights, const std::string& tactic) {
  if(tactic == "m128-n128-k32-w64-n32-s2-sw1")
    return makeGemmRunner<128,128,64,32,2,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n32-s3-sw1")
    return makeGemmRunner<128,128,64,32,3,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n32-s3-sw2")
    return makeGemmRunner<128,128,64,32,3,2,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s3-sw1")
    return makeGemmRunner<128,128,64,64,3,1,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s3-sw2")
    return makeGemmRunner<128,128,64,64,3,2,Epilogue>(weights);
  if(tactic == "m128-n128-k32-w64-n64-s3-sw4")
    return makeGemmRunner<128,128,64,64,3,4,Epilogue>(weights);
  if(tactic == "m128-n256-k32-w64-n64-s2-sw2")
    return makeGemmRunner<128,256,64,64,2,2,Epilogue>(weights);
  if(tactic == "m256-n128-k32-w64-n64-s2-sw1")
    return makeGemmRunner<256,128,64,64,2,1,Epilogue>(weights);
  if(tactic == "m256-n128-k32-w64-n64-s2-sw2")
    return makeGemmRunner<256,128,64,64,2,2,Epilogue>(weights);
  return nullptr;
}

using DefaultPostConvKernel = typename PostConvGemm::GemmKernel;
using PostConvMma = typename DefaultPostConvKernel::Mma;
using DefaultPostConvEpilogue = typename DefaultPostConvKernel::Epilogue;
using DefaultPostConvIterator = typename DefaultPostConvEpilogue::OutputTileIterator;
using PostConvSwizzle = typename DefaultPostConvKernel::ThreadblockSwizzle;

template<typename BaseIterator, int OutputChannels>
class ResidualBnOutputTileIterator : public BaseIterator {
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
  ResidualBnOutputTileIterator(
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
            if(outputColumn < OutputChannels) {
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
  ResidualBnOutputTileIterator& operator++() {
    Base::operator++();
    ++activatedIterator;
    return *this;
  }
};

using PostConvBnIterator = ResidualBnOutputTileIterator<
  DefaultPostConvIterator, PostConvOutChannels>;
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

using DefaultLinear2Kernel = typename Linear2BnBaseGemm::GemmKernel;
using Linear2Mma = typename DefaultLinear2Kernel::Mma;
using DefaultLinear2Epilogue = typename DefaultLinear2Kernel::Epilogue;
using DefaultLinear2Iterator = typename DefaultLinear2Epilogue::OutputTileIterator;
using Linear2Swizzle = typename DefaultLinear2Kernel::ThreadblockSwizzle;
using Linear2BnIterator = ResidualBnOutputTileIterator<DefaultLinear2Iterator, OutChannels>;
using Linear2BnEpilogue = cutlass::epilogue::threadblock::Epilogue<
  typename DefaultLinear2Epilogue::Shape,
  typename DefaultLinear2Epilogue::WarpMmaOperator,
  DefaultLinear2Epilogue::kPartitionsK,
  Linear2BnIterator,
  typename DefaultLinear2Epilogue::AccumulatorFragmentIterator,
  typename DefaultLinear2Epilogue::WarpTileIterator,
  typename DefaultLinear2Epilogue::SharedLoadIterator,
  typename DefaultLinear2Epilogue::OutputOp,
  typename DefaultLinear2Epilogue::Padding,
  DefaultLinear2Epilogue::Base::kFragmentsPerIteration>;
using Linear2BnKernel = cutlass::gemm::kernel::Gemm<
  Linear2Mma, Linear2BnEpilogue, Linear2Swizzle, false>;

} // namespace

struct Sm89Linear2Gemm::Impl {
  std::unique_ptr<GemmRunner> runner;

  Impl(const half* weights, const std::string& tactic)
    : runner(makeLinear2Runner(weights, tactic)) {}

  bool applyAccumulate(const half* input, half* output, int tokens, cudaStream_t stream) {
    return runner != nullptr &&
      runner->run(input, output, tokens, InChannels, OutChannels, 1.0f, stream);
  }
};

Sm89Linear2Gemm::Sm89Linear2Gemm(
  const half* weights, const std::string& tactic)
  : impl(std::make_unique<Impl>(weights, tactic))
{}

Sm89Linear2Gemm::~Sm89Linear2Gemm() = default;

bool Sm89Linear2Gemm::applyAccumulate(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != InChannels ||
     outChannels != OutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->applyAccumulate(input, output, batchSize * seqLen, stream);
}

struct Sm89Linear2BnGemm::Impl {
  const half* weights;
  const half* bnScale;
  const half* bnBias;
  bool initialized;

  Impl(const half* weights_, const half* bnScale_, const half* bnBias_)
    : weights(weights_), bnScale(bnScale_), bnBias(bnBias_), initialized(true)
  {
    int smemSize = int(sizeof(typename Linear2BnKernel::SharedStorage));
    if(smemSize >= 48 * 1024)
      initialized = cudaFuncSetAttribute(
        cutlass::Kernel<Linear2BnKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smemSize
      ) == cudaSuccess;
  }

  bool apply(
    const half* input,
    half* residualOutput,
    half* activatedOutput,
    int tokens,
    cudaStream_t stream
  ) {
    if(!initialized)
      return false;
    cutlass::gemm::GemmCoord problem(tokens, OutChannels, InChannels);
    cutlass::gemm::GemmCoord gridShape = Linear2Swizzle::get_tiled_shape(
      problem,
      {FusedThreadblockShape::kM, FusedThreadblockShape::kN,
       FusedThreadblockShape::kK},
      1
    );
    using Layout = cutlass::layout::RowMajor;
    Layout inputLayout(InChannels);
    Layout outputLayout(OutChannels);
    typename Linear2Mma::IteratorA::TensorRef nullInput(nullptr, inputLayout);
    typename Linear2Mma::IteratorB::TensorRef weightRef(
      reinterpret_cast<Element*>(const_cast<half*>(weights)), Layout(OutChannels)
    );
    typename Linear2BnIterator::TensorRef nullOutput(nullptr, outputLayout);
    typename Linear2BnKernel::Params params(
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
    params.ref_A.reset(reinterpret_cast<Element*>(const_cast<half*>(input)));
    params.ref_C.reset(reinterpret_cast<Element*>(residualOutput));
    params.ref_D.reset(reinterpret_cast<Element*>(residualOutput));
    params.params_D.activatedOutput = reinterpret_cast<Element*>(activatedOutput);
    dim3 grid = Linear2Swizzle::get_grid_shape(params.grid_tiled_shape);
    dim3 block(Linear2BnKernel::kThreadCount, 1, 1);
    int smemSize = int(sizeof(typename Linear2BnKernel::SharedStorage));
    cutlass::Kernel<Linear2BnKernel><<<grid, block, smemSize, stream>>>(params);
    return cudaPeekAtLastError() == cudaSuccess;
  }
};

Sm89Linear2BnGemm::Sm89Linear2BnGemm(
  const half* weights,
  const half* bnScale,
  const half* bnBias
) : impl(std::make_unique<Impl>(weights, bnScale, bnBias))
{}

Sm89Linear2BnGemm::~Sm89Linear2BnGemm() = default;

bool Sm89Linear2BnGemm::applyAccumulateAndActivate(
  const half* input,
  half* residualOutput,
  half* activatedOutput,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != InChannels ||
     outChannels != OutChannels || input == nullptr || residualOutput == nullptr ||
     activatedOutput == nullptr)
    return false;
  return impl->apply(input, residualOutput, activatedOutput, batchSize * seqLen, stream);
}

struct Sm89OutProjGemm::Impl {
  std::unique_ptr<GemmRunner> runner;

  Impl(const half* weights, const std::string& tactic)
    : runner(makeOutProjRunner(weights, tactic)) {}

  bool applyAccumulate(const half* input, half* output, int tokens, cudaStream_t stream) {
    return runner != nullptr && runner->run(
      input, output, tokens, AttentionChannels, OutChannels, 1.0f, stream);
  }
};

Sm89OutProjGemm::Sm89OutProjGemm(
  const half* weights, const std::string& tactic)
  : impl(std::make_unique<Impl>(weights, tactic))
{}

Sm89OutProjGemm::~Sm89OutProjGemm() = default;

bool Sm89OutProjGemm::applyAccumulate(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != AttentionChannels ||
     outChannels != OutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->applyAccumulate(input, output, batchSize * seqLen, stream);
}

struct Sm89PreConvGemm::Impl {
  std::unique_ptr<GemmRunner> runner;

  Impl(const half* weights, const std::string& tactic)
    : runner(makePreConvRunner(weights, tactic)) {}

  bool apply(const half* input, half* output, int tokens, cudaStream_t stream) {
    return runner != nullptr && runner->run(
      input, output, tokens, PreConvInChannels, OutChannels, 0.0f, stream);
  }
};

Sm89PreConvGemm::Sm89PreConvGemm(
  const half* weights, const std::string& tactic)
  : impl(std::make_unique<Impl>(weights, tactic))
{}

Sm89PreConvGemm::~Sm89PreConvGemm() = default;

bool Sm89PreConvGemm::apply(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != PreConvInChannels ||
     outChannels != OutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->apply(input, output, batchSize * seqLen, stream);
}

struct Sm89PostConvGemm::Impl {
  std::unique_ptr<GemmRunner> runner;

  Impl(const half* weights, const std::string& tactic)
    : runner(makePostConvRunner(weights, tactic)) {}

  bool applyAccumulate(const half* input, half* output, int tokens, cudaStream_t stream) {
    return runner != nullptr && runner->run(
      input, output, tokens, PostConvInChannels, PostConvOutChannels, 1.0f, stream);
  }
};

Sm89PostConvGemm::Sm89PostConvGemm(
  const half* weights, const std::string& tactic)
  : impl(std::make_unique<Impl>(weights, tactic))
{}

Sm89PostConvGemm::~Sm89PostConvGemm() = default;

bool Sm89PostConvGemm::applyAccumulate(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != PostConvInChannels ||
     outChannels != PostConvOutChannels || input == nullptr || output == nullptr)
    return false;
  return impl->applyAccumulate(input, output, batchSize * seqLen, stream);
}

struct Sm89PostConvBnGemm::Impl {
  const half* weights;
  const half* bnScale;
  const half* bnBias;
  bool initialized;

  Impl(const half* weights_, const half* bnScale_, const half* bnBias_)
    : weights(weights_), bnScale(bnScale_), bnBias(bnBias_), initialized(true)
  {
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
    int tokens,
    cudaStream_t stream
  ) {
    if(!initialized)
      return false;
    cutlass::gemm::GemmCoord problem(tokens, PostConvOutChannels, PostConvInChannels);
    cutlass::gemm::GemmCoord gridShape = PostConvSwizzle::get_tiled_shape(
      problem,
      {FusedThreadblockShape::kM, FusedThreadblockShape::kN,
       FusedThreadblockShape::kK},
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
    typename PostConvBnKernel::Params params(
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

Sm89PostConvBnGemm::Sm89PostConvBnGemm(
  const half* weights,
  const half* bnScale,
  const half* bnBias
) : impl(std::make_unique<Impl>(weights, bnScale, bnBias))
{}

Sm89PostConvBnGemm::~Sm89PostConvBnGemm() = default;

bool Sm89PostConvBnGemm::applyAccumulateAndActivate(
  const half* input,
  half* residualOutput,
  half* activatedOutput,
  int batchSize,
  int seqLen,
  int inChannels,
  int outChannels,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != PostConvInChannels ||
     outChannels != PostConvOutChannels || input == nullptr ||
     residualOutput == nullptr || activatedOutput == nullptr)
    return false;
  return impl->apply(input, residualOutput, activatedOutput, batchSize * seqLen, stream);
}

} // namespace Sm89Backend
