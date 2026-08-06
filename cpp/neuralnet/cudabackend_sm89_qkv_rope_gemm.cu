/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed B13/19x19 CUTLASS batched QKV GEMM with learnable RoPE in the output iterator.
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

constexpr int B = 13;
constexpr int S = 361;
constexpr int Tokens = B * S;
constexpr int Channels = 384;
constexpr int Heads = 12;
constexpr int HeadDim = 32;
constexpr int GemmBatch = 3;

using Element = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;
using OutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
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

using DefaultKernel = typename DeviceGemm::DefaultGemmKernel;
using Mma = typename DefaultKernel::Mma;
using DefaultEpilogue = typename DefaultKernel::Epilogue;
using DefaultIterator = typename DefaultEpilogue::OutputTileIterator;

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

    CUTLASS_HOST_DEVICE
    Params() : Base::Params(), freqs(nullptr) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
      : Base::Params(layout), freqs(nullptr) {}
  };

 private:
  const float* freqs;

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
      freqs(params.freqs) {}

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& fragment, int64_t byteOffset) const {
    Fragment transformed = fragment;
    if(freqs != nullptr && int(blockIdx.z) < 2) {
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
              if(outputRow < Tokens && outputColumn < Channels) {
                AccessType& access = accesses[
                  fragmentRow * ThreadMap::Iterations::kColumn + column];
                int xy = outputRow % S;

                CUTLASS_PRAGMA_UNROLL
                for(int element = 0; element < kElementsPerAccess; element += 2) {
                  int channel = outputColumn + element;
                  int x = xy % 19;
                  int y = xy / 19;
                  int hp = channel / 2;
                  float angle = x * freqs[2 * hp] + y * freqs[2 * hp + 1];
                  float sinValue, cosValue;
                  __sincosf(angle, &sinValue, &cosValue);
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
using RopeKernel = cutlass::gemm::kernel::GemmBatched<Mma, RopeEpilogue, Swizzle>;

} // namespace

struct Sm89QKVRoPEGemmB13::Impl {
  typename RopeKernel::Params params;
  bool initialized;

  Impl(const half* weights, const float* freqs) : initialized(true) {
    cutlass::gemm::GemmCoord problem(Tokens, Channels, Channels);
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
      (int64_t)Tokens * Channels,
      nullOutput,
      (int64_t)Tokens * Channels,
      typename OutputOp::Params(1.0f, 0.0f),
      GemmBatch);
    params.params_D.freqs = freqs;

    int smemSize = int(sizeof(typename RopeKernel::SharedStorage));
    if(smemSize >= 48 * 1024)
      initialized = cudaFuncSetAttribute(
        cutlass::Kernel<RopeKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize) == cudaSuccess;
  }

  bool apply(const half* input, half* output, cudaStream_t stream) {
    if(!initialized)
      return false;
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

Sm89QKVRoPEGemmB13::Sm89QKVRoPEGemmB13(
  const half* weights,
  const float* freqs
) : impl(std::make_unique<Impl>(weights, freqs)) {}

Sm89QKVRoPEGemmB13::~Sm89QKVRoPEGemmB13() = default;

bool Sm89QKVRoPEGemmB13::apply(
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
  if(batchSize != B || seqLen != S || inChannels != Channels ||
     qkvChannels != Channels || numHeads != Heads || headDim != HeadDim ||
     input == nullptr || output == nullptr)
    return false;
  return impl->apply(input, output, stream);
}

} // namespace Sm89Backend
