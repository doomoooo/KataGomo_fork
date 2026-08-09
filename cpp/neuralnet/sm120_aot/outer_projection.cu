#include "outer_projection.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include <memory>
#include <new>
#include <cstring>
#include <unordered_map>

namespace {

#ifndef KATAGO_GEMM_SWIZZLE
#define KATAGO_GEMM_SWIZZLE 1
#endif
#ifndef KATAGO_GEMM_WARP_M
#define KATAGO_GEMM_WARP_M 64
#endif
#ifndef KATAGO_GEMM_WARP_N
#define KATAGO_GEMM_WARP_N 64
#endif

constexpr int DOWN_INPUT_CHANNELS = 768;
constexpr int DOWN_OUTPUT_CHANNELS = 384;
constexpr int UP_INPUT_CHANNELS = 384;
constexpr int UP_OUTPUT_CHANNELS = 768;

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
  cutlass::gemm::GemmShape<KATAGO_GEMM_WARP_M,KATAGO_GEMM_WARP_N,32>,
  cutlass::gemm::GemmShape<16,8,16>,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<KATAGO_GEMM_SWIZZLE>,
  3,
  8,
  8,
  false
>;
using GemmWarp64x32 = cutlass::gemm::device::Gemm<
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
  cutlass::gemm::GemmShape<64,32,32>,
  cutlass::gemm::GemmShape<16,8,16>,
  Epilogue,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
  3,
  8,
  8,
  false
>;

using ThreadblockShape = cutlass::gemm::GemmShape<128,128,32>;
using WarpShape = cutlass::gemm::GemmShape<KATAGO_GEMM_WARP_M,KATAGO_GEMM_WARP_N,32>;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<KATAGO_GEMM_SWIZZLE>;
using OutputThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
  ThreadblockShape, WarpShape, Element, 8, 1>;
using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
using Residual = cutlass::epilogue::threadblock::VisitorAuxLoad<
  OutputThreadMap, Element, cute::Stride<int64_t,cute::_1,int64_t>>;
using Add = cutlass::epilogue::threadblock::VisitorCompute<
  cutlass::plus, Element, Element, cutlass::FloatRoundStyle::round_to_nearest>;
using Raw = cutlass::epilogue::threadblock::Sm80EVT<Add, Accum, Residual>;
using RawStore = cutlass::epilogue::threadblock::VisitorAuxStore<
  OutputThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
  cute::Stride<int64_t,cute::_1,int64_t>>;
using StoredRaw = cutlass::epilogue::threadblock::Sm80EVT<RawStore, Raw>;
using Scale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
  OutputThreadMap, Element, cute::Stride<cute::_0,cute::_1,int64_t>>;
using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
  OutputThreadMap, Element, cute::Stride<cute::_0,cute::_1,int64_t>>;
using Affine = cutlass::epilogue::threadblock::VisitorCompute<
  cutlass::homogeneous_multiply_add, Element, Element,
  cutlass::FloatRoundStyle::round_to_nearest>;
using AffineTree = cutlass::epilogue::threadblock::Sm80EVT<
  Affine, StoredRaw, Scale, Bias>;
using Silu = cutlass::epilogue::threadblock::VisitorCompute<
  cutlass::epilogue::thread::SiLu, Element, float,
  cutlass::FloatRoundStyle::round_to_nearest>;
using Activated = cutlass::epilogue::threadblock::Sm80EVT<Silu, AffineTree>;
using ActivatedStore = cutlass::epilogue::threadblock::VisitorAuxStore<
  OutputThreadMap, Element, cutlass::FloatRoundStyle::round_to_nearest,
  cute::Stride<int64_t,cute::_1,int64_t>>;
using Fusion = cutlass::epilogue::threadblock::Sm80EVT<ActivatedStore, Activated>;
using FusedKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
  Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
  Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
  Element, cutlass::layout::RowMajor, 8,
  Element,
  Element,
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  Fusion,
  Swizzle,
  3,
  cutlass::arch::OpMultiplyAdd,
  1>::GemmKernel;
using FusedGemm = cutlass::gemm::device::GemmUniversalAdapter<FusedKernel>;

struct Handle {
  Handle(
    const void* weights_, int inputChannels_, int outputChannels_, int beta_,
    bool warp64x32_
  ) :
    weights(reinterpret_cast<const Element*>(weights_)),
    inputChannels(inputChannels_),
    outputChannels(outputChannels_),
    beta(beta_),
    warp64x32(warp64x32_) {}

  struct RowState {
    typename Gemm::GemmKernel::Params params;
    typename GemmWarp64x32::GemmKernel::Params paramsWarp64x32;
    FusedGemm fusedGemm;
    bool initialized = false;
    bool fusedInitialized = false;
  };

  const Element* weights;
  int inputChannels;
  int outputChannels;
  int beta;
  bool warp64x32;
  std::unordered_map<int,std::unique_ptr<RowState>> byRows;
};

template<typename GemmType>
typename GemmType::Arguments makeArguments(
  Handle* handle, const void* input, void* output, int rows
) {
  const Element alpha = Element(1.0f);
  const Element beta = Element(static_cast<float>(handle->beta));
  Element* outputElements = reinterpret_cast<Element*>(output);
  return typename GemmType::Arguments(
    cutlass::gemm::GemmCoord(rows, handle->outputChannels, handle->inputChannels),
    {reinterpret_cast<const Element*>(input), handle->inputChannels},
    {handle->weights, handle->outputChannels},
    {outputElements, handle->outputChannels},
    {outputElements, handle->outputChannels},
    {alpha, beta},
    1);
}

FusedGemm::Arguments makeFusedArguments(
  Handle* handle,
  const void* input,
  void* residualAndOutput,
  const void* scale,
  const void* bias,
  void* activatedOutput,
  int rows
) {
  typename Fusion::Arguments callbacks{
    {
      {
        {
          {
            {},
            {reinterpret_cast<Element*>(residualAndOutput), Element(0),
             {UP_OUTPUT_CHANNELS, cute::_1{}, rows * UP_OUTPUT_CHANNELS}},
            {}
          },
          {reinterpret_cast<Element*>(residualAndOutput),
           {UP_OUTPUT_CHANNELS, cute::_1{}, rows * UP_OUTPUT_CHANNELS}}
        },
        {reinterpret_cast<const Element*>(scale), Element(0),
         {cute::_0{}, cute::_1{}, UP_OUTPUT_CHANNELS}},
        {reinterpret_cast<const Element*>(bias), Element(0),
         {cute::_0{}, cute::_1{}, UP_OUTPUT_CHANNELS}},
        {}
      },
      {}
    },
    {reinterpret_cast<Element*>(activatedOutput),
     {UP_OUTPUT_CHANNELS, cute::_1{}, rows * UP_OUTPUT_CHANNELS}}
  };
  return FusedGemm::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    cutlass::gemm::GemmCoord(rows, UP_OUTPUT_CHANNELS, UP_INPUT_CHANNELS),
    1,
    callbacks,
    reinterpret_cast<const Element*>(input),
    handle->weights,
    nullptr,
    nullptr,
    static_cast<int64_t>(rows) * UP_INPUT_CHANNELS,
    static_cast<int64_t>(UP_INPUT_CHANNELS) * UP_OUTPUT_CHANNELS,
    0,
    0,
    UP_INPUT_CHANNELS,
    UP_OUTPUT_CHANNELS,
    0,
    0,
    nullptr,
    nullptr,
    nullptr);
}

int statusCode(cutlass::Status status) {
  return 100 + static_cast<int>(status);
}

void* createHandle(
  const void* weights, int inputChannels, int outputChannels, int beta,
  bool warp64x32
) {
  if(weights == nullptr)
    return nullptr;
  using Kernel = typename Gemm::GemmKernel;
  using KernelWarp64x32 = typename GemmWarp64x32::GemmKernel;
  const int sharedMemoryBytes = warp64x32 ?
    (int)sizeof(typename KernelWarp64x32::SharedStorage) :
    (int)sizeof(typename Kernel::SharedStorage);
  cudaError_t status = cudaFuncSetAttribute(
    warp64x32 ? (const void*)cutlass::Kernel<KernelWarp64x32> :
      (const void*)cutlass::Kernel<Kernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    sharedMemoryBytes);
  if(status != cudaSuccess)
    return nullptr;
  constexpr int fusedSharedMemoryBytes = sizeof(typename FusedKernel::SharedStorage);
  status = cudaFuncSetAttribute(
    cutlass::Kernel<FusedKernel>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    fusedSharedMemoryBytes);
  if(status != cudaSuccess)
    return nullptr;
  return new(std::nothrow) Handle(
    weights, inputChannels, outputChannels, beta, warp64x32);
}

template<typename GemmType, typename Params>
int launchGemm(
  Handle* handle, Params& params, bool& initialized,
  const void* input, void* output, int rows, cudaStream_t stream
) {
  typename GemmType::Arguments arguments =
    makeArguments<GemmType>(handle, input, output, rows);
  using Kernel = typename GemmType::GemmKernel;
  using Swizzle = typename GemmType::ThreadblockSwizzle;
  if(!initialized) {
    cutlass::Status status = GemmType::can_implement(arguments);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
    Swizzle swizzle;
    cutlass::gemm::GemmCoord gridShape = swizzle.get_tiled_shape(
      arguments.problem_size,
      {GemmType::ThreadblockShape::kM, GemmType::ThreadblockShape::kN,
       GemmType::ThreadblockShape::kK},
      arguments.split_k_slices);
    params = typename Kernel::Params{
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
    initialized = true;
  }
  else {
    params.ref_A.reset(arguments.ref_A.non_const_ref().data());
    params.ref_B.reset(arguments.ref_B.non_const_ref().data());
    params.ref_C.reset(arguments.ref_C.non_const_ref().data());
    params.ref_D.reset(arguments.ref_D.data());
    params.output_op = arguments.epilogue;
  }

  Swizzle swizzle;
  dim3 grid = swizzle.get_grid_shape(params.grid_tiled_shape);
  dim3 block(Kernel::kThreadCount, 1, 1);
  constexpr int sharedMemoryBytes = sizeof(typename Kernel::SharedStorage);
  cutlass::Kernel<Kernel><<<grid, block, sharedMemoryBytes, stream>>>(params);
  cudaError_t status = cudaPeekAtLastError();
  return status == cudaSuccess ? 0 : 200 + static_cast<int>(status);
}

int launchHandle(
  void* opaqueHandle, const void* input, void* output, int rows,
  cudaStream_t stream
) {
  if(opaqueHandle == nullptr || input == nullptr || output == nullptr ||
     rows <= 0 || rows % 361 != 0)
    return 1;

  Handle* handle = static_cast<Handle*>(opaqueHandle);
  auto& stateValue = handle->byRows[rows];
  if(stateValue == nullptr)
    stateValue = std::make_unique<Handle::RowState>();
  Handle::RowState* state = stateValue.get();
  if(handle->warp64x32)
    return launchGemm<GemmWarp64x32>(
      handle, state->paramsWarp64x32, state->initialized,
      input, output, rows, stream);
  return launchGemm<Gemm>(
    handle, state->params, state->initialized,
    input, output, rows, stream);
}

int launchFusedUpHandle(
  void* opaqueHandle,
  const void* input,
  void* residualAndOutput,
  const void* scale,
  const void* bias,
  void* activatedOutput,
  int rows,
  cudaStream_t stream
) {
  if(
    opaqueHandle == nullptr || input == nullptr || residualAndOutput == nullptr ||
    scale == nullptr || bias == nullptr || activatedOutput == nullptr ||
    rows <= 0 || rows % 361 != 0
  )
    return 1;
  Handle* handle = static_cast<Handle*>(opaqueHandle);
  auto& stateValue = handle->byRows[rows];
  if(stateValue == nullptr)
    stateValue = std::make_unique<Handle::RowState>();
  Handle::RowState* state = stateValue.get();
  if(
    handle->inputChannels != UP_INPUT_CHANNELS ||
    handle->outputChannels != UP_OUTPUT_CHANNELS || handle->beta != 1
  )
    return 2;

  if(handle->warp64x32)
    return 3;
  FusedGemm::Arguments arguments = makeFusedArguments(
    handle, input, residualAndOutput, scale, bias, activatedOutput, rows);
  cutlass::Status status;
  if(!state->fusedInitialized) {
    status = FusedGemm::can_implement(arguments);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
    status = state->fusedGemm.initialize(arguments, nullptr, stream);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
    state->fusedInitialized = true;
  }
  else {
    status = state->fusedGemm.update(arguments);
    if(status != cutlass::Status::kSuccess)
      return statusCode(status);
  }
  status = state->fusedGemm.run(stream);
  if(status != cutlass::Status::kSuccess)
    return statusCode(status);
  cudaError_t cudaStatus = cudaPeekAtLastError();
  return cudaStatus == cudaSuccess ? 0 : 200 + static_cast<int>(cudaStatus);
}

}  // namespace

extern "C" void* katago_create_outer_projection_down_sm120(
  const void* weights, const char* tactic
) {
  if(tactic == nullptr)
    return nullptr;
  bool warp64x32 = std::strcmp(tactic,"warp64x32") == 0;
  if(!warp64x32 && std::strcmp(tactic,"warp64x64") != 0)
    return nullptr;
  return createHandle(
    weights, DOWN_INPUT_CHANNELS, DOWN_OUTPUT_CHANNELS, 0, warp64x32);
}

extern "C" void* katago_create_head_projection_sm120(
  const void* weights, int outputChannels, const char* tactic
) {
  if(tactic == nullptr || (outputChannels != 288 && outputChannels != 384))
    return nullptr;
  bool warp64x32 = std::strcmp(tactic,"warp64x32") == 0;
  if(!warp64x32 && std::strcmp(tactic,"warp64x64") != 0)
    return nullptr;
  return createHandle(
    weights, DOWN_INPUT_CHANNELS, outputChannels, 0, warp64x32);
}

extern "C" void katago_destroy_outer_projection_down_sm120(void* handle) {
  delete static_cast<Handle*>(handle);
}

extern "C" int katago_launch_outer_projection_down_sm120(
  void* handle,
  const void* input,
  void* output,
  int rows,
  cudaStream_t stream
) {
  return launchHandle(handle, input, output, rows, stream);
}

extern "C" void* katago_create_outer_projection_up_sm120(
  const void* weights, const char* tactic
) {
  if(tactic == nullptr)
    return nullptr;
  bool warp64x32 = std::strcmp(tactic,"warp64x32") == 0;
  if(!warp64x32 && std::strcmp(tactic,"warp64x64") != 0)
    return nullptr;
  return createHandle(
    weights, UP_INPUT_CHANNELS, UP_OUTPUT_CHANNELS, 1, warp64x32);
}

extern "C" void katago_destroy_outer_projection_up_sm120(void* handle) {
  delete static_cast<Handle*>(handle);
}

extern "C" int katago_launch_outer_projection_up_sm120(
  void* handle,
  const void* input,
  void* residualAndOutput,
  int rows,
  cudaStream_t stream
) {
  return launchHandle(handle, input, residualAndOutput, rows, stream);
}

extern "C" int katago_launch_outer_projection_up_silu_sm120(
  void* handle,
  const void* input,
  void* residualAndOutput,
  const void* scale,
  const void* bias,
  void* activatedOutput,
  int rows,
  cudaStream_t stream
) {
  return launchFusedUpHandle(
    handle, input, residualAndOutput, scale, bias, activatedOutput, rows, stream);
}
