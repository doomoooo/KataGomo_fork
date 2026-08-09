/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed-channel/19x19 CUTLASS batched QKV GEMM with learnable RoPE in the output iterator.
 * CUTLASS commit: 7127592069c2fe01b041e174ba4345ef9b279671
 **************************************************************************************************/

#include "../neuralnet/cudabackend_sm89_qkv_rope_gemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_batched.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace Sm89Backend {
namespace {

constexpr int S = 361;
constexpr int Channels = 384;
constexpr int Heads = 12;
constexpr int HeadDim = 32;
constexpr int GemmBatch = 3;

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using OutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using PlainOutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, Element, cutlass::epilogue::thread::ScaleType::Nothing>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using Swizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

using DeviceGemm = cutlass::gemm::device::GemmBatched<
  Element, Layout,
  Element, Layout,
  Element, Layout,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  OutputOp,
  Swizzle,
  3,
  8,
  8>;

using PlainDeviceGemm = cutlass::gemm::device::GemmBatched<
  Element, Layout,
  Element, Layout,
  Element, Layout,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  PlainOutputOp,
  Swizzle,
  3,
  8,
  8>;

using DefaultKernel = typename DeviceGemm::DefaultGemmKernel;
using Mma = typename DefaultKernel::Mma;
using DefaultEpilogue = typename DefaultKernel::Epilogue;
using DefaultIterator = typename DefaultEpilogue::OutputTileIterator;
using PlainDefaultKernel = typename PlainDeviceGemm::DefaultGemmKernel;
using PlainMma = typename PlainDefaultKernel::Mma;
using PlainDefaultEpilogue = typename PlainDefaultKernel::Epilogue;
using PlainDefaultIterator = typename PlainDefaultEpilogue::OutputTileIterator;

template<typename BaseIterator>
class RoPEOutputTileIterator : public BaseIterator {
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
    const float* freqs;
    const float2* cosSinTable;
    int totalRows;

    CUTLASS_HOST_DEVICE
    Params() : Base::Params(), freqs(nullptr), cosSinTable(nullptr), totalRows(0) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
      : Base::Params(layout), freqs(nullptr), cosSinTable(nullptr), totalRows(0) {}
  };

 private:
  const float* freqs;
  const float2* cosSinTable;
  int totalRows;

 public:
  CUTLASS_DEVICE
  RoPEOutputTileIterator(
    Params const& params,
    Element* pointer,
    TensorCoord extent,
    int threadIdx,
    TensorCoord threadblockOffset = TensorCoord(),
    int const* indices = nullptr
  ) : Base(params, pointer, extent, threadIdx, threadblockOffset, indices),
      freqs(params.freqs), cosSinTable(params.cosSinTable),
      totalRows(params.totalRows) {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment, int64_t byteOffset) const {
    Fragment transformed = fragment;
    if((freqs != nullptr || cosSinTable != nullptr) && int(blockIdx.z) < 2) {
      AccessType* accesses = reinterpret_cast<AccessType*>(&transformed);
      int startRow = Base::thread_start_row();
      int startColumn = Base::thread_start_column();

      CUTLASS_PRAGMA_UNROLL
      for(int cluster = 0; cluster < ThreadMap::Iterations::kCluster; cluster++) {
        CUTLASS_PRAGMA_UNROLL
        for(int group = 0; group < ThreadMap::Iterations::kGroup; group++) {
          CUTLASS_PRAGMA_UNROLL
          for(int row = 0; row < ThreadMap::Iterations::kRow; row++) {
            int fragmentRow = row + ThreadMap::Iterations::kRow *
              (group + ThreadMap::Iterations::kGroup * cluster);
            int rowOffset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;
            int outputRow = startRow + rowOffset;

            CUTLASS_PRAGMA_UNROLL
            for(int column = 0; column < ThreadMap::Iterations::kColumn; column++) {
              int outputColumn = startColumn + column * ThreadMap::Delta::kColumn;
              if(outputRow < totalRows && outputColumn < Channels) {
                AccessType& access = accesses[
                  fragmentRow * ThreadMap::Iterations::kColumn + column];
                int xy = outputRow % S;

                CUTLASS_PRAGMA_UNROLL
                for(int element = 0; element < kElementsPerAccess; element += 2) {
                  int channel = outputColumn + element;
                  int hp = channel / 2;
                  float sinValue;
                  float cosValue;
                  if(cosSinTable != nullptr) {
                    // Model-lifetime table layout is [xy][head * pairs + pair].
                    // With the fixed 12xD32 shape, channel / 2 is exactly that
                    // flattened head-pair coordinate.
                    float2 cosSin = cosSinTable[(size_t)xy * (Channels / 2) + hp];
                    cosValue = cosSin.x;
                    sinValue = cosSin.y;
                  }
                  else {
                    int x = xy % 19;
                    int y = xy / 19;
                    float angle = x * freqs[2 * hp] + y * freqs[2 * hp + 1];
                    __sincosf(angle, &sinValue, &cosValue);
                  }
                  float v0 = static_cast<float>(access[element]);
                  float v1 = static_cast<float>(access[element + 1]);
                  access[element] = Element(v0 * cosValue - v1 * sinValue);
                  access[element + 1] = Element(v0 * sinValue + v1 * cosValue);
                }
              }
            }
          }
        }
      }
    }
    Base::store_with_byte_offset(transformed, byteOffset);
  }

  CUTLASS_DEVICE
  void store(Fragment const& fragment) const {
    store_with_byte_offset(fragment, 0);
  }
};

using RopeIterator = RoPEOutputTileIterator<DefaultIterator>;
using RopeEpilogue = cutlass::epilogue::threadblock::Epilogue<
  typename DefaultEpilogue::Shape,
  typename DefaultEpilogue::WarpMmaOperator,
  DefaultEpilogue::kPartitionsK,
  RopeIterator,
  typename DefaultEpilogue::AccumulatorFragmentIterator,
  typename DefaultEpilogue::WarpTileIterator,
  typename DefaultEpilogue::SharedLoadIterator,
  typename DefaultEpilogue::OutputOp,
  typename DefaultEpilogue::Padding,
  DefaultEpilogue::Base::kFragmentsPerIteration>;
using PlainFp32Kernel = cutlass::gemm::kernel::GemmBatched<
  Mma, DefaultEpilogue, Swizzle>;
using PlainHalfKernel = cutlass::gemm::kernel::GemmBatched<
  PlainMma, PlainDefaultEpilogue, Swizzle>;
using RopeKernel = cutlass::gemm::kernel::GemmBatched<Mma, RopeEpilogue, Swizzle>;

} // namespace

struct Sm89QKVRoPEGemm::Impl {
  const half* weights;
  const float* freqs;
  const float2* cosSinTable;
  typename RopeKernel::Params params;
  typename PlainFp32Kernel::Params plainFp32Params;
  typename PlainHalfKernel::Params plainHalfParams;
  bool splitRoPE;
  int plainVariant;
  bool initialized;

  Impl(
    const half* weights_, const float* freqs_, const float2* cosSinTable_,
    bool splitRoPE_, int plainVariant_)
    : weights(weights_), freqs(freqs_), cosSinTable(cosSinTable_), splitRoPE(splitRoPE_),
      plainVariant(plainVariant_), initialized(true) {
    int smemSize = splitRoPE
      ? (plainVariant == 1
          ? int(sizeof(typename PlainHalfKernel::SharedStorage))
          : int(sizeof(typename PlainFp32Kernel::SharedStorage)))
      : int(sizeof(typename RopeKernel::SharedStorage));
    if(smemSize >= 48 * 1024)
      initialized = splitRoPE ?
        (plainVariant == 1
          ? cudaFuncSetAttribute(
              cutlass::Kernel<PlainHalfKernel>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize) == cudaSuccess
          : cudaFuncSetAttribute(
              cutlass::Kernel<PlainFp32Kernel>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize) == cudaSuccess)
        : cudaFuncSetAttribute(
            cutlass::Kernel<RopeKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            smemSize) == cudaSuccess;
  }

  void configure(int tokens) {
    cutlass::gemm::GemmCoord problem(tokens, Channels, Channels);
    cutlass::gemm::GemmCoord gridShape = Swizzle::get_tiled_shape(
      problem,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      GemmBatch);
    Layout matrixLayout(Channels);
    typename Mma::IteratorA::TensorRef nullInput(nullptr, matrixLayout);
    typename Mma::IteratorB::TensorRef weightRef(
      reinterpret_cast<Element*>(const_cast<half*>(weights)), matrixLayout);
    typename RopeIterator::TensorRef nullOutput(nullptr, matrixLayout);
    params = typename RopeKernel::Params(
      problem,
      gridShape,
      nullInput,
      0,
      weightRef,
      (int64_t)Channels * Channels,
      nullOutput,
      (int64_t)tokens * Channels,
      nullOutput,
      (int64_t)tokens * Channels,
      typename OutputOp::Params(1.0f, 0.0f),
      GemmBatch);
    params.params_D.freqs = freqs;
    params.params_D.cosSinTable = cosSinTable;
    params.params_D.totalRows = tokens;

    typename DefaultIterator::TensorRef plainFp32NullOutput(nullptr, matrixLayout);
    plainFp32Params = typename PlainFp32Kernel::Params(
      problem,
      gridShape,
      nullInput,
      0,
      weightRef,
      (int64_t)Channels * Channels,
      plainFp32NullOutput,
      (int64_t)tokens * Channels,
      plainFp32NullOutput,
      (int64_t)tokens * Channels,
      typename OutputOp::Params(1.0f, 0.0f),
      GemmBatch);

    typename PlainDefaultIterator::TensorRef plainHalfNullOutput(nullptr, matrixLayout);
    plainHalfParams = typename PlainHalfKernel::Params(
      problem,
      gridShape,
      nullInput,
      0,
      weightRef,
      (int64_t)Channels * Channels,
      plainHalfNullOutput,
      (int64_t)tokens * Channels,
      plainHalfNullOutput,
      (int64_t)tokens * Channels,
      typename PlainOutputOp::Params(Element(1.0f), Element(0.0f)),
      GemmBatch);

  }

  bool apply(const half* input, half* output, int tokens, cudaStream_t stream) {
    if(!initialized)
      return false;
    configure(tokens);
    if(splitRoPE) {
      if(plainVariant == 1) {
        plainHalfParams.ref_A.reset(reinterpret_cast<Element*>(const_cast<half*>(input)));
        plainHalfParams.ref_C.reset(reinterpret_cast<Element*>(output));
        plainHalfParams.ref_D.reset(reinterpret_cast<Element*>(output));
        dim3 grid = Swizzle::get_grid_shape(plainHalfParams.grid_tiled_shape);
        dim3 block(PlainHalfKernel::kThreadCount, 1, 1);
        int smemSize = int(sizeof(typename PlainHalfKernel::SharedStorage));
        cutlass::Kernel<PlainHalfKernel><<<grid, block, smemSize, stream>>>(plainHalfParams);
      }
      else {
        plainFp32Params.ref_A.reset(reinterpret_cast<Element*>(const_cast<half*>(input)));
        plainFp32Params.ref_C.reset(reinterpret_cast<Element*>(output));
        plainFp32Params.ref_D.reset(reinterpret_cast<Element*>(output));
        dim3 grid = Swizzle::get_grid_shape(plainFp32Params.grid_tiled_shape);
        dim3 block(PlainFp32Kernel::kThreadCount, 1, 1);
        int smemSize = int(sizeof(typename PlainFp32Kernel::SharedStorage));
        cutlass::Kernel<PlainFp32Kernel><<<grid, block, smemSize, stream>>>(plainFp32Params);
      }
      return cudaPeekAtLastError() == cudaSuccess;
    }
    params.ref_A.reset(reinterpret_cast<Element*>(const_cast<half*>(input)));
    params.ref_C.reset(reinterpret_cast<Element*>(output));
    params.ref_D.reset(reinterpret_cast<Element*>(output));
    dim3 grid = Swizzle::get_grid_shape(params.grid_tiled_shape);
    dim3 block(RopeKernel::kThreadCount, 1, 1);
    int smemSize = int(sizeof(typename RopeKernel::SharedStorage));
    cutlass::Kernel<RopeKernel><<<grid, block, smemSize, stream>>>(params);
    return cudaPeekAtLastError() == cudaSuccess;
  }
};

Sm89QKVRoPEGemm::Sm89QKVRoPEGemm(
  const half* weights,
  const float* freqs,
  const float2* cosSinTable,
  bool splitRoPE,
  int plainVariant
) : impl(std::make_unique<Impl>(
      weights, freqs, cosSinTable, splitRoPE, plainVariant)) {}

Sm89QKVRoPEGemm::~Sm89QKVRoPEGemm() = default;

bool Sm89QKVRoPEGemm::apply(
  const half* input,
  half* output,
  int batchSize,
  int seqLen,
  int inChannels,
  int qkvChannels,
  int numHeads,
  int headDim,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || inChannels != Channels ||
     qkvChannels != Channels || numHeads != Heads || headDim != HeadDim ||
     input == nullptr || output == nullptr)
    return false;
  return impl->apply(input, output, batchSize * seqLen, stream);
}

} // namespace Sm89Backend
