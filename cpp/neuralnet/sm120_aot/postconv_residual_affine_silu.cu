/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Runtime-row SM120 outer post-projection with a residual plus following
 * affine-SiLU dual-output epilogue. The output iterator preserves the
 * historical FP16 rounding boundary: the residual output is stored first and
 * the following activation is derived from that rounded fragment.
 **************************************************************************************************/

#include "../cudabackend_sm120_kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

namespace Sm120Backend {
namespace {

constexpr int InputChannels = 384;
constexpr int OutputChannels = 768;

using Element = cutlass::half_t;
using OutputOp = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float>;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
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
  OutputOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,
  8,
  8>;

using DefaultKernel = typename PostConvGemm::GemmKernel;
using Mma = typename DefaultKernel::Mma;
using DefaultEpilogue = typename DefaultKernel::Epilogue;
using DefaultIterator = typename DefaultEpilogue::OutputTileIterator;
using Swizzle = typename DefaultKernel::ThreadblockSwizzle;

template<typename BaseIterator>
class ResidualAffineSiluOutputTileIterator : public BaseIterator {
 public:
  using Base = BaseIterator;
  using ThreadMap = typename Base::ThreadMap;
  using Element = typename Base::Element;
  using Layout = typename Base::Layout;
  using TensorRef = typename Base::TensorRef;
  using TensorCoord = typename Base::TensorCoord;
  using Fragment = typename Base::Fragment;
  using AccessType = typename Base::AccessType;

  static int const kElementsPerAccess = Base::kElementsPerAccess;

  struct Params : public Base::Params {
    Element* activatedOutput;
    const Element* scale;
    const Element* bias;

    CUTLASS_HOST_DEVICE
    Params() : Base::Params(), activatedOutput(nullptr), scale(nullptr), bias(nullptr) {}

    CUTLASS_HOST_DEVICE
    explicit Params(Layout const& layout)
      : Base::Params(layout), activatedOutput(nullptr), scale(nullptr), bias(nullptr) {}
  };

 private:
  Base activatedIterator;
  const Element* scale;
  const Element* bias;

 public:
  CUTLASS_DEVICE
  ResidualAffineSiluOutputTileIterator(
    Params const& params,
    Element* pointer,
    TensorCoord extent,
    int threadIdx,
    TensorCoord threadblockOffset = TensorCoord(),
    int const* indices = nullptr
  ) : Base(params, pointer, extent, threadIdx, threadblockOffset, indices),
      activatedIterator(
        params, params.activatedOutput, extent, threadIdx, threadblockOffset, indices),
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
                fragmentRow * ThreadMap::Iterations::kColumn + column];
              CUTLASS_PRAGMA_UNROLL
              for(int element = 0; element < kElementsPerAccess; element++) {
                int channel = outputColumn + element;
                Element xElement = access[element];
                half x = __ushort_as_half(xElement.storage);
                half s = __ushort_as_half(scale[channel].storage);
                half b = __ushort_as_half(bias[channel].storage);
                half affine = __hfma(x, s, b);
                float value = __half2float(affine);
                half activated = __float2half(value / (1.0f + expf(-value)));
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
  ResidualAffineSiluOutputTileIterator& operator++() {
    Base::operator++();
    ++activatedIterator;
    return *this;
  }
};

using FusedIterator = ResidualAffineSiluOutputTileIterator<DefaultIterator>;
using FusedEpilogue = cutlass::epilogue::threadblock::Epilogue<
  typename DefaultEpilogue::Shape,
  typename DefaultEpilogue::WarpMmaOperator,
  DefaultEpilogue::kPartitionsK,
  FusedIterator,
  typename DefaultEpilogue::AccumulatorFragmentIterator,
  typename DefaultEpilogue::WarpTileIterator,
  typename DefaultEpilogue::SharedLoadIterator,
  typename DefaultEpilogue::OutputOp,
  typename DefaultEpilogue::Padding,
  DefaultEpilogue::Base::kFragmentsPerIteration>;
using FusedKernel = cutlass::gemm::kernel::Gemm<
  Mma, FusedEpilogue, Swizzle, false>;

} // namespace

cudaError_t launchPostConvResidualAffineSilu(
  const half* input,
  const half* weights,
  half* residual,
  half* activated,
  const half* scale,
  const half* bias,
  int totalRows,
  cudaStream_t stream
) {
  if(totalRows <= 0)
    return cudaErrorInvalidValue;
  cutlass::gemm::GemmCoord problem(totalRows, OutputChannels, InputChannels);
  cutlass::gemm::GemmCoord gridShape = Swizzle::get_tiled_shape(
    problem,
    {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
    1);
  using Layout = cutlass::layout::RowMajor;
  Layout inputLayout(InputChannels);
  Layout outputLayout(OutputChannels);
  typename Mma::IteratorA::TensorRef inputRef(
    reinterpret_cast<Element*>(const_cast<half*>(input)), inputLayout);
  typename Mma::IteratorB::TensorRef weightRef(
    reinterpret_cast<Element*>(const_cast<half*>(weights)), outputLayout);
  typename FusedIterator::TensorRef residualRef(
    reinterpret_cast<Element*>(residual), outputLayout);
  typename FusedKernel::Params params(
    problem,
    gridShape,
    inputRef,
    weightRef,
    residualRef,
    residualRef,
    typename OutputOp::Params(1.0f, 1.0f));
  params.params_D.activatedOutput = reinterpret_cast<Element*>(activated);
  params.params_D.scale = reinterpret_cast<const Element*>(scale);
  params.params_D.bias = reinterpret_cast<const Element*>(bias);

  int smemSize = int(sizeof(typename FusedKernel::SharedStorage));
  static const cudaError_t attributeStatus = cudaFuncSetAttribute(
    cutlass::Kernel<FusedKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smemSize);
  if(attributeStatus != cudaSuccess)
    return attributeStatus;

  dim3 grid = Swizzle::get_grid_shape(params.grid_tiled_shape);
  dim3 block(FusedKernel::kThreadCount, 1, 1);
  cutlass::Kernel<FusedKernel><<<grid, block, smemSize, stream>>>(params);
  return cudaPeekAtLastError();
}

} // namespace Sm120Backend
