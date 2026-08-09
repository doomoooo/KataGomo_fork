/***************************************************************************************************
 * Dynamic-shape SM120 wrapper for CUTLASS examples/45_dual_gemm.
 * Both projections share the same A tile and the epilogue writes SwiGLU
 * directly, preserving the historically positive 5080 implementation family.
 **************************************************************************************************/

#include "dual_ffn_shared_a.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "device/dual_gemm.h"
#include "thread/left_silu_and_mul.h"

#include <memory>
#include <new>
#include <unordered_map>

namespace {

constexpr int Channels = 384;
constexpr int FfnChannels = 1152;
using Element = cutlass::half_t;
using ProjectionOutput = cutlass::epilogue::thread::LinearCombination<
  Element, 8, Element, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using SwiGLU = cutlass::epilogue::thread::LeftSiLUAndMul<
  Element, 8, Element, float>;
using DualGemm = cutlass::gemm::device::DualGemm<
  Element, cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  cutlass::layout::RowMajor,
  Element, cutlass::layout::RowMajor,
  Element,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128,64,32>,
  cutlass::gemm::GemmShape<64,32,32>,
  cutlass::gemm::GemmShape<16,8,16>,
  ProjectionOutput, ProjectionOutput, SwiGLU,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
  3, false, false, false, 8, 8>;

DualGemm::Arguments makeArguments(
  const void* input,
  const void* linearWeights,
  const void* gateWeights,
  void* output,
  int tokens
) {
  using Layout = cutlass::layout::RowMajor;
  DualGemm::TensorRefC nullC;
  DualGemm::TensorRefD nullD;
  return {
    cutlass::gemm::DualGemmMode::kGemm,
    {tokens, FfnChannels, Channels},
    {reinterpret_cast<const Element*>(input), Layout(Channels)},
    {reinterpret_cast<const Element*>(linearWeights), Layout(FfnChannels)},
    nullC, nullD,
    {reinterpret_cast<const Element*>(gateWeights), Layout(FfnChannels)},
    nullC, nullD,
    {reinterpret_cast<Element*>(output), Layout(FfnChannels)},
    {1.0f, 0.0f}, {1.0f, 0.0f}, {}, 1
  };
}

struct State {
  DualGemm op;
  bool initialized = false;
};

struct Handle {
  std::unordered_map<int,std::unique_ptr<State>> byTokens;
};

int statusCode(cutlass::Status status) {
  return 100 + static_cast<int>(status);
}

}  // namespace

extern "C" void* katago_create_dual_ffn_shared_a_sm120() {
  // DualGemm::initialize() configures the dynamic shared-memory attribute for
  // its DualGemmKernel.  Keep that setup owned by CUTLASS so this wrapper does
  // not depend on private kernel aliases from the examples/45 API.
  return new(std::nothrow) Handle();
}

extern "C" void katago_destroy_dual_ffn_shared_a_sm120(void* opaque) {
  delete static_cast<Handle*>(opaque);
}

extern "C" int katago_launch_dual_ffn_shared_a_sm120(
  void* opaque,
  const void* input,
  const void* linearWeights,
  const void* gateWeights,
  void* output,
  int tokens,
  cudaStream_t stream
) {
  if(opaque == nullptr || input == nullptr || linearWeights == nullptr ||
     gateWeights == nullptr || output == nullptr || tokens <= 0 ||
     tokens % 361 != 0)
    return 1;
  Handle* handle = static_cast<Handle*>(opaque);
  auto& slot = handle->byTokens[tokens];
  if(slot == nullptr)
    slot = std::make_unique<State>();
  DualGemm::Arguments args = makeArguments(
    input, linearWeights, gateWeights, output, tokens);
  cutlass::Status status;
  if(!slot->initialized) {
    status = slot->op.can_implement(args);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
    status = slot->op.initialize(args, nullptr, stream);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
    slot->initialized = true;
  }
  else {
    status = slot->op.update(args, nullptr);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
  }
  status = slot->op.run(stream);
  if(status != cutlass::Status::kSuccess)
    return statusCode(status);
  cudaError_t cudaStatus = cudaPeekAtLastError();
  return cudaStatus == cudaSuccess ? 0 : 200 + static_cast<int>(cudaStatus);
}
