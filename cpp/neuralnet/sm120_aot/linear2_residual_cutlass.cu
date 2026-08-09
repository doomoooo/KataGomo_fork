#include "../cudabackend_sm120_kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#include <memory>
#include <unordered_map>

namespace Sm120Backend {
namespace {

constexpr int N = 384;
constexpr int K = 1152;

using Element = cutlass::half_t;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
  Element,
  128 / cutlass::sizeof_bits<Element>::value,
  Element,
  Element
>;
using Gemm = cutlass::gemm::device::Gemm<
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,128,32>,
  cutlass::gemm::GemmShape<64,64,32>,
  cutlass::gemm::GemmShape<16,8,16>,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,
  8,
  8,
  false
>;

struct Handle {
  Handle(const Element* weights_, int rows_) :
    weights(weights_), rows(rows_), initialized(false) {}

  typename Gemm::GemmKernel::Params params;
  const Element* weights;
  int rows;
  bool initialized;
};

// A launch slot is invoked by one persistent NN-server host thread. Keeping
// the CUTLASS Params cache thread-local avoids both repeated host-side
// initialization and cross-stream mutation of A/C/D pointers.
thread_local std::unordered_map<
  const Element*,std::unordered_map<int,std::unique_ptr<Handle>>> handles;

cudaError_t launch(
  const half* input,
  const half* weights,
  half* residual,
  int matBatchSize,
  cudaStream_t stream
) {
  if(matBatchSize <= 0 || matBatchSize % 361 != 0)
    return cudaErrorInvalidValue;
  using GemmKernel = typename Gemm::GemmKernel;
  using ThreadblockSwizzle = typename Gemm::ThreadblockSwizzle;
  constexpr int sharedMemoryBytes = sizeof(typename GemmKernel::SharedStorage);
  static const cudaError_t attributeStatus = cudaFuncSetAttribute(
    cutlass::Kernel<GemmKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    sharedMemoryBytes);
  if(attributeStatus != cudaSuccess)
    return attributeStatus;

  const Element* cutlassWeights = reinterpret_cast<const Element*>(weights);
  auto& byRows = handles[cutlassWeights];
  auto found = byRows.find(matBatchSize);
  if(found == byRows.end())
    found = byRows.emplace(
      matBatchSize, std::make_unique<Handle>(cutlassWeights,matBatchSize)).first;
  Handle* handle = found->second.get();
  Element* output = reinterpret_cast<Element*>(residual);
  const Element one = Element(1.0f);
  Gemm::Arguments arguments(
    cutlass::gemm::GemmCoord(handle->rows,N,K),
    {reinterpret_cast<const Element*>(input), K},
    {handle->weights, N},
    {output, N},
    {output, N},
    {one, one},
    1);

  if(!handle->initialized) {
    const cutlass::Status status = Gemm::can_implement(arguments);
    if(status != cutlass::Status::kSuccess)
      return cudaErrorInvalidValue;
    ThreadblockSwizzle swizzle;
    const cutlass::gemm::GemmCoord gridShape = swizzle.get_tiled_shape(
      arguments.problem_size,
      {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN,
       Gemm::ThreadblockShape::kK},
      arguments.split_k_slices);
    handle->params = typename GemmKernel::Params{
      arguments.problem_size,
      gridShape,
      arguments.ref_A.non_const_ref(),
      arguments.ref_B.non_const_ref(),
      arguments.ref_C.non_const_ref(),
      arguments.ref_D,
      arguments.epilogue,
      nullptr,
      arguments.gather_A_indices,
      arguments.gather_B_indices,
      arguments.scatter_D_indices
    };
    handle->initialized = true;
  }
  else {
    handle->params.ref_A.reset(arguments.ref_A.non_const_ref().data());
    handle->params.ref_B.reset(arguments.ref_B.non_const_ref().data());
    handle->params.ref_C.reset(arguments.ref_C.non_const_ref().data());
    handle->params.ref_D.reset(arguments.ref_D.data());
    handle->params.output_op = arguments.epilogue;
  }

  ThreadblockSwizzle swizzle;
  const dim3 grid = swizzle.get_grid_shape(handle->params.grid_tiled_shape);
  const dim3 block(GemmKernel::kThreadCount,1,1);
  cutlass::Kernel<GemmKernel><<<grid,block,sharedMemoryBytes,stream>>>(handle->params);
  return cudaPeekAtLastError();
}

} // namespace

cudaError_t launchLinear2ResidualCutlass(
  const half* input,
  const half* weights,
  half* residual,
  int matBatchSize,
  cudaStream_t stream
) {
  return launch(input,weights,residual,matBatchSize,stream);
}

} // namespace Sm120Backend
