#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

// cuDNN frontend SDPA support. The header is vendored under external/cudnn-frontend.
// Requires cuDNN >= 8.5 (header-only library declares this); the SDPA path additionally
// requires cuDNN backend support which is present in cuDNN 8.9.3+ for our use case.
// IMPORTANT: cudnn_frontend bundles nlohmann/json 3.11.3 which uses the same include guard
// (INCLUDE_NLOHMANN_JSON_HPP_) as KataGo's older nlohmann/json 3.8.0. Including cudnn_frontend.h
// first ensures the 3.11.3 version wins and that the template signatures cudnn_frontend expects
// are the ones actually available.
#if CUDNN_VERSION >= 8903
  #define KATAGO_CUDA_HAS_SDPA 1
  // Note: cudnn_frontend's Execution_plan_list::query_properties() trips GCC's -Wnull-dereference.
  // It's a benign issue in vendored third-party header code, but it can't be silenced with a
  // `#pragma GCC diagnostic ignored` here: GCC emits this one from the -O2 interprocedural-analysis
  // phase with no source location, so it ignores the per-region diagnostic pragma state. Instead it's
  // suppressed file-scoped via -Wno-null-dereference on this source in CMakeLists.txt.
  #include <cudnn_frontend.h>
#else
  #define KATAGO_CUDA_HAS_SDPA 0
#endif

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudnnquerymutex.h"
#include "../neuralnet/cudautils.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/cudabackend_sm120.h"
#include "../neuralnet/cudabackend_sm89.h"

#include "../core/simpleallocator.h"
#include "../core/test.h"
#include "../core/hash.h"

#include "../external/half-2.2.0/include/half.hpp"

#include <optional>
#include <cstring>
#include <mutex>
#include <unordered_map>

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;

//Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

void NeuralNet::globalInitialize() {
  //Empty for cudnn backend
}

void NeuralNet::globalCleanup() {
  cudaDeviceReset();
}

//---------------------------------------------------------------------------------
// cudnn SDPA support. Graphs + execution plans cached lazily per (batchSize, hasMask).
// Used only when useFP16=true and cudnn supports SDPA at runtime. Otherwise falls back to
// customCudaFlashAttention (see cudahelpers).
//
// Tensor layout: BSHD physical, with strides chosen so that the (B,H,S,D)-dim graph view matches
// the existing CUDA backend's Q/K/V/output buffers from MatMulLayer:
//   element at (n, xy, h, d) lives at offset (h*headDim + d) + (n*seqLen + xy) * (numHeads*headDim).
//
// Masking: when a mask is present, we build a fully-materialized additive attention bias of shape
// [B, 1, S, S] from the [B, S] mask: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -3e4). cudnn does not have
// plans for the [B,1,1,S] broadcast pattern that would let us avoid this materialization, but the
// full bias is correct for arbitrary (non-prefix) masks, which we need to support sub-board games.
// The bias is built once per inference (the mask is the same across all 20 attention blocks).
//
// The bias tensor dtype MUST match the io dtype (fp16). An fp32 bias with fp16 Q/K/V passes
// validate/check_support/build_plans on cudnn 9.x but silently misexecutes (nonfinite outputs -
// the fused kernel evidently reinterprets the buffer). Since cudnn adds the (converted) bias to the
// fp32 scores and computes the softmax in fp32, the fp16-range-limited -3e4 constant still masks
// exactly (exp underflow) for any model whose genuine logit spread is below ~3e4.
// See cudahelpers.cu for details.
//
// When mask is NULL (full-board, requireExactNNLen case), we build a no-bias graph instead, which
// avoids both the extra memory and the bias build kernel.

#if KATAGO_CUDA_HAS_SDPA
struct SDPAPlanForBatchSize {
  std::shared_ptr<cudnn_frontend::graph::Graph> graph;
  int64_t workspaceBytes;
  bool hasMask;  // true if the graph expects a bias variant-pack entry

  // UIDs for the variant pack, fixed at graph build time.
  static constexpr int64_t Q_UID = 1;
  static constexpr int64_t K_UID = 2;
  static constexpr int64_t V_UID = 3;
  static constexpr int64_t O_UID = 4;
  static constexpr int64_t BIAS_UID = 5;
};

// Full discriminating key for an SDPA execution plan. Every field that changes the cudnn graph shape
// must be here: if any attention layer in a future model differs in head count/dim/seqLen, it gets its
// own plan rather than incorrectly reusing another layer's. (batchSize and hasMask vary at runtime.)
struct SDPAGraphKey {
  int numHeads;
  int numKVHeads;
  int qHeadDim;
  int vHeadDim;
  int seqLen;
  int batchSize;
  bool hasMask;
  bool usingFP16;

  bool operator==(const SDPAGraphKey& o) const {
    return
      numHeads == o.numHeads &&
      numKVHeads == o.numKVHeads &&
      qHeadDim == o.qHeadDim &&
      vHeadDim == o.vHeadDim &&
      seqLen == o.seqLen &&
      batchSize == o.batchSize &&
      hasMask == o.hasMask &&
      usingFP16 == o.usingFP16;
  }
};
struct SDPAGraphKeyHash {
  uint64_t operator()(const SDPAGraphKey& k) const noexcept {
    uint64_t acc = (uint64_t)123456789;
    auto mix = [&acc](uint64_t x) {
      acc += x;
      acc += acc << 13;
      acc ^= acc >> 6;
    };
    mix((uint64_t)k.numHeads);
    mix((uint64_t)k.numKVHeads);
    mix((uint64_t)k.qHeadDim);
    mix((uint64_t)k.vHeadDim);
    mix((uint64_t)k.seqLen);
    mix((uint64_t)k.batchSize);
    mix(k.hasMask ? 1 : 0);
    mix(k.usingFP16 ? 1 : 0);
    acc = Hash::basicLCong(acc);
    return (size_t)(acc ^ (acc >> 32));
  }
};

struct SDPAGraphCache {
  std::unordered_map<SDPAGraphKey, std::shared_ptr<SDPAPlanForBatchSize>, SDPAGraphKeyHash> plansByKey;
  bool sdpaSupported;
  string disableReason;

  SDPAGraphCache() :
    plansByKey(),
    sdpaSupported(true),
    disableReason()
  {}

  // Build (or fetch from cache) an execution plan for the given attention shape + batchSize + hasMask.
  // Returns nullptr if SDPA is not supported for this configuration; caller should use fallback.
  // On a build failure during warmup, SDPA is disabled going forward and nullptr is returned (the
  // caller falls back to the custom kernel); outside of warmup such a failure is fatal. logger (if
  // non-NULL) is used to report a disable.
  std::shared_ptr<SDPAPlanForBatchSize> getOrBuildPlan(cudnnHandle_t cudnn, const SDPAGraphKey& key, Logger* logger, bool isWarmup) {
    if(!sdpaSupported)
      return nullptr;

    // Cuda graphs for SDPA path only well-tested for FP16/BF16; FP32 uses fallback
    if(!key.usingFP16)
      return nullptr;

    auto it = plansByKey.find(key);
    if(it != plansByKey.end())
      return it->second;

    namespace fe = cudnn_frontend;

    // Disable SDPA and report the reason. Outside of warmup a build failure is fatal; during warmup
    // we tolerate it and fall back to the custom kernel (returning nullptr to the caller).
    auto disable = [&](const string& reason) -> std::shared_ptr<SDPAPlanForBatchSize> {
      if(!isWarmup)
        throw StringError(reason);
      sdpaSupported = false;
      disableReason = reason;
      if(logger != NULL)
        logger->write("Cuda backend: disabling cudnn SDPA and falling back to custom attention kernel: " + reason);
      return nullptr;
    };
    auto plan = std::make_shared<SDPAPlanForBatchSize>();
    plan->hasMask = key.hasMask;
    auto graph = std::make_shared<fe::graph::Graph>();

    bool useFP16 = key.usingFP16;

    fe::DataType_t ioType = useFP16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT;
    graph->set_io_data_type(ioType)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t B = key.batchSize;
    int64_t Hq = key.numHeads;
    int64_t Hkv = key.numKVHeads;
    int64_t S = key.seqLen;
    int64_t Dq = key.qHeadDim;
    int64_t Dv = key.vHeadDim;

    // BSHD physical layout, with logical dim ordering (B, H, S, D):
    // stride for B = S * H_inner * D
    // stride for H = D
    // stride for S = H_inner * D
    // stride for D = 1
    // where H_inner is the number of heads packed for this tensor (numHeads or numKVHeads).
    int64_t qHinner = key.numHeads;
    int64_t kHinner = key.numKVHeads;
    int64_t vHinner = key.numKVHeads;

    auto Q = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("Q")
      .set_uid(SDPAPlanForBatchSize::Q_UID)
      .set_dim({B, Hq, S, Dq})
      .set_stride({S * qHinner * Dq, Dq, qHinner * Dq, 1})
    );
    auto K = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("K")
      .set_uid(SDPAPlanForBatchSize::K_UID)
      .set_dim({B, Hkv, S, Dq})
      .set_stride({S * kHinner * Dq, Dq, kHinner * Dq, 1})
    );
    auto V = graph->tensor(
      fe::graph::Tensor_attributes()
      .set_name("V")
      .set_uid(SDPAPlanForBatchSize::V_UID)
      .set_dim({B, Hkv, S, Dv})
      .set_stride({S * vHinner * Dv, Dv, vHinner * Dv, 1})
    );

    float scale = 1.0f / std::sqrt((float)key.qHeadDim);
    auto sdpa_options = (
      fe::graph::SDPA_attributes()
      .set_name("sdpa_fwd")
      .set_generate_stats(false)
      .set_attn_scale(scale)
    );

    if(key.hasMask) {
      // Full [B, 1, S, S] additive bias, broadcast over heads only. Per cudnn 9.8 empirical
      // testing the broadcast-over-q variant ([B,1,1,S]) has no supported plans for our shape.
      auto bias = graph->tensor(
        fe::graph::Tensor_attributes()
        .set_name("bias")
        .set_uid(SDPAPlanForBatchSize::BIAS_UID)
        .set_dim({B, 1, S, S})
        .set_stride({S * S, S * S, S, 1})
      );
      sdpa_options.set_bias(bias);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;

    // Output O also uses BSHD physical layout (matches what outProj expects).
    int64_t oHinner = key.numHeads;
    O->set_output(true)
      .set_dim({B, Hq, S, Dv})
      .set_stride({S * oHinner * Dv, Dv, oHinner * Dv, 1})
      .set_uid(SDPAPlanForBatchSize::O_UID);

    auto status = graph->validate();
    if(status.is_bad())
      return disable(string("cudnn SDPA graph validate failed: ") + status.get_message());
    status = graph->build_operation_graph(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA build_operation_graph failed: ") + status.get_message());
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if(status.is_bad())
      return disable(string("cudnn SDPA create_execution_plans failed: ") + status.get_message());
    status = graph->check_support(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA check_support failed: ") + status.get_message());
    status = graph->build_plans(cudnn);
    if(status.is_bad())
      return disable(string("cudnn SDPA build_plans failed: ") + status.get_message());

    int64_t ws = 0;
    status = graph->get_workspace_size(ws);
    if(status.is_bad())
      return disable(string("cudnn SDPA get_workspace_size failed: ") + status.get_message());

    plan->graph = graph;
    plan->workspaceBytes = ws;
    plansByKey[key] = plan;
    return plan;
  }
};
#else
struct SDPAGraphCache {
  SDPAGraphCache() {}
};
#endif


struct CudaHandles {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
  // Borrowed from the caller. The ComputeHandle and all CUDA work use this one explicit stream.
  cudaStream_t stream;
  const bool ownsStreamForTesting;
  const int majorComputeCapability;
  const int minorComputeCapability;
  std::unique_ptr<SDPAGraphCache> sdpaCache;
  // Logger for this handle's server thread; may be NULL. Used to report cudnn SDPA falling back.
  Logger* logger;
  // Set while warming up (see NNEvaluator::maybeWarmupComputeHandle). When true, a failed cudnn SDPA
  // execution is tolerated (fall back to the custom kernel); when false such a failure is fatal.
  bool isWarmup;
  // If true, the cudnn graph SDPA path is skipped entirely and the custom attention kernel is always used.
  bool cudaDisableGraphSDPA;
  // Set once we have logged that cudaDisableGraphSDPA actually suppressed an otherwise-usable SDPA path,
  // so the message is emitted only a single time per handle rather than on every attention block.
  bool loggedGraphSDPADisabled;
  // Read by ConvLayer construction before Sm120Model exists.
  // 0=disabled, 45=eng45/tile0/stages2, 47=retained eng47 knob plan.
  int sm120InitialConvFrontendEngine;
  // Controls whether 1x1 NHWC convs run as a cuBLAS GEMM (vs cuDNN). Auto = matmul iff FP16.
  // True/False force the choice regardless of precision.
  enabled_t use1x1MatmulMode;
  // SM120 attention hook (thin dispatch). Null on non-SM120 handles; the
  // implementation lives entirely in cudabackend_sm120.cpp.
  Sm120Backend::Sm120AttentionFn sm120Attention;
  void* sm120AttentionContext;
  Sm120Backend::Sm120FFNSingleGemmFn sm120FFNSingleGemm;
  void* sm120FFNSingleGemmContext;
  Sm120Backend::Sm120MatMulFn sm120MatMul;
  void* sm120MatMulContext;
  Sm120Backend::Sm120Conv1x1Fn sm120Conv1x1;
  void* sm120Conv1x1Context;
  Sm120Backend::Sm120InitialGlobalFn sm120InitialGlobal;
  void* sm120InitialGlobalContext;
  Sm120Backend::Sm120QKVStridedFn sm120QKVStrided;
  void* sm120QKVStridedContext;
  Sm120Backend::Sm120FusedResidualGemmFn sm120FusedResidualGemm;
  void* sm120FusedResidualGemmContext;
  Sm120Backend::Sm120RMSNormFn sm120RMSNorm;
  void* sm120RMSNormContext;
  Sm120Backend::Sm120FusedQKRoPEFn sm120FusedQKRoPE;
  void* sm120FusedQKRoPEContext;
  Sm120Backend::Sm120SwiGLUFn sm120SwiGLU;
  void* sm120SwiGLUContext;
  Sm120Backend::Sm120AffineSiluFn sm120AffineSilu;
  void* sm120AffineSiluContext;
  Sm120Backend::Sm120PostConvBNSiluFn sm120PostConvBNSilu;
  void* sm120PostConvBNSiluContext;
  float* sm120FullBoardAreaBuf;
  int sm120FullBoardCapacity;
  bool sm120UseHeadBNHalfToFloat;
  bool loggedSm120HeadBNHalfToFloat;
  bool sm120ShareModelWeights;
  bool sm120SharedModelWeightsActive;
  bool loggedSm120SharedModelWeights;
  bool sm120UseFusedValueTerminal;
  bool loggedSm120FusedValueTerminal;
  Sm120Backend::Sm120FusedPolicyP1Fn sm120FusedPolicyP1;
  void* sm120FusedPolicyP1Context;
  Sm120Backend::Sm120WideHeadProjectionFn sm120WideHeadProjection;
  void* sm120WideHeadProjectionContext;
  Sm120Backend::Sm120PersistingL2Fn sm120PersistingL2;
  void* sm120PersistingL2Context;
  Sm120Backend::Sm120PersistingL2Fn sm120PersistingL2Inner;
  void* sm120PersistingL2InnerContext;

  CudaHandles(int major, int minor, cudaStream_t stream_, bool ownsStreamForTesting_ = false)
    : stream(stream_),
      ownsStreamForTesting(ownsStreamForTesting_),
      majorComputeCapability(major),
      minorComputeCapability(minor),
      sdpaCache(std::make_unique<SDPAGraphCache>()),
      logger(NULL),
      isWarmup(false),
      cudaDisableGraphSDPA(false),
      loggedGraphSDPADisabled(false),
      sm120InitialConvFrontendEngine(0),
      use1x1MatmulMode(enabled_t::Auto),
      sm120Attention(NULL),
      sm120AttentionContext(NULL),
      sm120FFNSingleGemm(NULL),
      sm120FFNSingleGemmContext(NULL),
      sm120MatMul(NULL),
      sm120MatMulContext(NULL),
      sm120Conv1x1(NULL),
      sm120Conv1x1Context(NULL),
      sm120InitialGlobal(NULL),
      sm120InitialGlobalContext(NULL),
      sm120QKVStrided(NULL),
      sm120QKVStridedContext(NULL),
      sm120FusedResidualGemm(NULL),
      sm120FusedResidualGemmContext(NULL),
      sm120RMSNorm(NULL),
      sm120RMSNormContext(NULL),
      sm120FusedQKRoPE(NULL),
      sm120FusedQKRoPEContext(NULL),
      sm120SwiGLU(NULL),
      sm120SwiGLUContext(NULL),
      sm120AffineSilu(NULL),
      sm120AffineSiluContext(NULL),
      sm120PostConvBNSilu(NULL),
      sm120PostConvBNSiluContext(NULL),
      sm120FullBoardAreaBuf(NULL),
      sm120FullBoardCapacity(0),
      sm120UseHeadBNHalfToFloat(false),
      loggedSm120HeadBNHalfToFloat(false),
      sm120ShareModelWeights(false),
      sm120SharedModelWeightsActive(false),
      loggedSm120SharedModelWeights(false),
      sm120UseFusedValueTerminal(false),
      loggedSm120FusedValueTerminal(false),
      sm120FusedPolicyP1(NULL),
      sm120FusedPolicyP1Context(NULL),
      sm120WideHeadProjection(NULL),
      sm120WideHeadProjectionContext(NULL),
      sm120PersistingL2(NULL),
      sm120PersistingL2Context(NULL),
      sm120PersistingL2Inner(NULL),
      sm120PersistingL2InnerContext(NULL)
  {
    if(stream == NULL)
      throw StringError("CudaHandles: external CUDA stream must not be null");
    CUBLAS_ERR("CudaHandles",cublasCreate(&cublas));
    CUDNN_ERR("CudaHandles",cudnnCreate(&cudnn));
    CUBLAS_ERR("CudaHandles",cublasSetStream(cublas, stream));
    CUDNN_ERR("CudaHandles",cudnnSetStream(cudnn, stream));
  }

  ~CudaHandles() {
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
    if(ownsStreamForTesting)
      cudaStreamDestroy(stream);
  }

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuIdxForThisThread);
    cudaStream_t stream;
    CUDA_ERR("cudaHandlesTesting",cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    return new CudaHandles(prop.major, prop.minor, stream, true);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};

//---------------------------------------------------------------------------------

template<typename T>
struct ByBatchSize {
  const int maxBatchSize;
  T* data;
  cudnnStatus_t (*destroyFunc)(T);

  ByBatchSize()
    : maxBatchSize(0), data(nullptr), destroyFunc(nullptr)
  {}

  ByBatchSize(
    int maxBatchSize_
  ) : maxBatchSize(maxBatchSize_), data(nullptr), destroyFunc(nullptr) {
    data = new T[maxBatchSize];
  }

  ByBatchSize(const ByBatchSize&) = delete;
  ByBatchSize& operator=(const ByBatchSize&) = delete;

  ~ByBatchSize() {
    if(destroyFunc != nullptr && data != nullptr) {
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        (*destroyFunc)(data[batchSize-1]);
      }
    }
    if(data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

template<typename T>
struct ByBatchSizeView {
  int maxBatchSize;
  T* data;

  ByBatchSizeView()
    : maxBatchSize(0), data(nullptr)
  {}

  ByBatchSizeView(const ByBatchSize<T>& toView)
    : maxBatchSize(toView.maxBatchSize), data(toView.data)
  {}
  ByBatchSizeView& operator=(const ByBatchSize<T>& toView) {
    maxBatchSize = toView.maxBatchSize;
    data = toView.data;
  }

  ~ByBatchSizeView() {
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

//---------------------------------------------------------------------------------


//channels, useFP16, useNHWC
typedef std::tuple<int, bool, bool> CudnnTensorDesc4DKey;

struct CudnnManager {
  const string name;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  std::map<CudnnTensorDesc4DKey, ByBatchSize<cudnnTensorDescriptor_t>*> tensorDesc4DByBatchSizeByKey;

  CudnnManager(string name_, int maxBatchSize_, int nnXLen_, int nnYLen_)
    :name(name_),
     maxBatchSize(maxBatchSize_),
     nnXLen(nnXLen_),
     nnYLen(nnYLen_),
     tensorDesc4DByBatchSizeByKey()
  {
  }

  ~CudnnManager() {
    for(auto& iter: tensorDesc4DByBatchSizeByKey) {
      delete iter.second;
    }
  }

  ByBatchSizeView<cudnnTensorDescriptor_t> getTensorDesc4DByBatchSize(
    int channels, bool useFP16, bool useNHWC
  ) {
    auto iter = tensorDesc4DByBatchSizeByKey.find({channels, useFP16, useNHWC});
    if(iter != tensorDesc4DByBatchSizeByKey.end()) {
      return ByBatchSizeView<cudnnTensorDescriptor_t>(*(iter->second));
    }
    ByBatchSize<cudnnTensorDescriptor_t>* descs = new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize);
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& desc = (*descs)[batchSize];
      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&desc));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
                  desc,
                  (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
                  (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
                  batchSize,
                  channels,
                  nnYLen,
                  nnXLen
                ));
    }
    descs->destroyFunc = cudnnDestroyTensorDescriptor;
    tensorDesc4DByBatchSizeByKey[{channels, useFP16, useNHWC}] = descs;
    return ByBatchSizeView<cudnnTensorDescriptor_t>(*descs);
  }
};

//---------------------------------------------------------------------------------

struct ScratchBuffers {

  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  SimpleAllocator<void*>* allocator;

  // Not scratch, but convenient to have here
  void* zeroBuf;
  void* oneBuf;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSize, int nnXLen, int nnYLen, bool useFP16)
    : batchXYFloatBytes((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * nnXLen * nnYLen * (useFP16 ? sizeof(half_t) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (useFP16 ? sizeof(half_t) : sizeof(float)))
  {
    std::function<void*(size_t)> allocateFunc = [](size_t size) {
      void* buf;
      CUDA_ERR("ScratchBuffers",cudaMalloc(&buf, size));
      return buf;
    };
    std::function<void(void*)> releaseFunc = [](void* buf) {
      cudaFree(buf);
    };

    allocator = new SimpleAllocator<void*>(allocateFunc, releaseFunc);

    CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
  }
  ~ScratchBuffers() {
    delete allocator;
    free(zeroBuf);
    free(oneBuf);
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }

};


//---------------------------------------------------------------------------------

struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  ByBatchSizeView<cudnnTensorDescriptor_t> inputDescriptors;
  ByBatchSizeView<cudnnTensorDescriptor_t> outputDescriptors;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
#if CUDNN_MAJOR >= 8
  ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>* convolutionAlgorithms; //array of one for each batch size
#else
  ByBatchSize<cudnnConvolutionFwdAlgo_t>* convolutionAlgorithms; //array of one for each batch size
#endif
  void* filterBuf;
  // A 1x1 conv is equivalent to a matmul. When use1x1Matmul is set we run it as a cuBLAS GEMM over
  // batch*spatial tokens and build NO cuDNN objects. This is the default for 1x1 NHWC FP16 convs.
  // matmulWeightBuf is [inC, outC] column-major (cuBLAS order); matmulSpatialSize is the spatial length.
  bool use1x1Matmul;
  int matmulSpatialSize;
  void* matmulWeightBuf;
  bool usingFP16;
  bool initialConvFrontendRequired;
  // cuDNN frontend execution plans are shape-specific. Keep one plan per
  // exact batch instead of hiding the historically successful frontend
  // engine behind a B19 branch.
  std::vector<size_t> initialConvFrontendWorkspaceBytes;
  std::vector<int64_t> initialConvFrontendPlanIndexes;
  std::vector<string> initialConvFrontendMarkers;
  mutable std::vector<unsigned char> loggedInitialConvFrontend;
#if KATAGO_CUDA_HAS_SDPA
  std::vector<std::shared_ptr<cudnn_frontend::graph::Graph>> initialConvFrontendGraphs;
#endif

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWC
  ) : ConvLayer(cudaHandles, manager, desc, useFP16, useNHWC, useNHWC)
  {}

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels)
  {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    testAssert(convXSize % 2 == 1);
    testAssert(convYSize % 2 == 1);

    usingFP16 = useFP16;
    initialConvFrontendRequired = false;
    filterBuf = NULL;
    matmulWeightBuf = NULL;
    filterDescriptor = NULL;
    convolutionDescriptor = NULL;
    convolutionAlgorithms = NULL;
    initialConvFrontendWorkspaceBytes.clear();
    initialConvFrontendPlanIndexes.clear();
    initialConvFrontendMarkers.clear();
    loggedInitialConvFrontend.clear();
#if KATAGO_CUDA_HAS_SDPA
    initialConvFrontendGraphs.clear();
#endif

    // A 1x1 conv is a matmul, and cuBLAS is faster than cuDNN's conv in FP16 (tensor cores).
    // Benchmarked as slightly faster on convnets and neutral on transformers.
    // In FP32 there wasn't an improvement, so the default (cudaUse1x1Matmul=Auto) only uses GEMM in FP16.
    // The config flag can force it either way regardless of precision. Supports NHWC only (the GEMM assumes
    // channel-contiguous-per-position layout).
    use1x1Matmul = false;
    if(convXSize == 1 && convYSize == 1 && useNHWCIn && useNHWCOut) {
      enabled_t mode = cudaHandles->use1x1MatmulMode;
      use1x1Matmul = (mode == enabled_t::True) || (mode == enabled_t::Auto && useFP16);
    }
    matmulSpatialSize = use1x1Matmul ? (manager->nnYLen * manager->nnXLen) : 0;

    if(use1x1Matmul) {
      // 1x1 conv weights are [outC, inC]. cuBLAS GEMM wants column-major, i.e. [inC, outC] in row-major notation.
      // So transpose. No cuDNN objects are built.
      vector<float> wT((size_t)inChannels * outChannels);
      for(int oc = 0; oc < outChannels; oc++)
        for(int ic = 0; ic < inChannels; ic++)
          wT[(size_t)oc + (size_t)ic * outChannels] = desc->weights[(size_t)oc * inChannels + ic];
      CudaUtils::mallocAndCopyToDevice(name + ":matmulW", wT, matmulWeightBuf, useFP16);
      return;
    }

    inputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,useNHWCIn);
    outputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,useNHWCOut);
    int maxBatchSize = manager->maxBatchSize;
    initialConvFrontendWorkspaceBytes.assign(maxBatchSize + 1,0);
    initialConvFrontendPlanIndexes.assign(maxBatchSize + 1,-1);
    initialConvFrontendMarkers.resize(maxBatchSize + 1);
    loggedInitialConvFrontend.assign(maxBatchSize + 1,0);
#if KATAGO_CUDA_HAS_SDPA
    initialConvFrontendGraphs.resize(maxBatchSize + 1);
#endif

    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      (filterNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      outChannels,
      inChannels,
      convYSize,
      convXSize
    ));

    int yStride = 1;
    int xStride = 1;

    //NVIDIA compute capability 7 is when we first hit Volta architecture, with tensor cores
    //See https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    bool tensorCoresSupported = cudaHandles->majorComputeCapability >= 7;

    CUDNN_ERR(name.c_str(),cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingY,
      paddingX,
      yStride,
      xStride,
      dilationY,
      dilationX,
      CUDNN_CROSS_CORRELATION,
      (useFP16 && !tensorCoresSupported) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT
    ));
    if(useFP16 && tensorCoresSupported)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

#if CUDNN_MAJOR >= 8
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>(maxBatchSize);
#else
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgo_t>(maxBatchSize);
#endif

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
#if CUDNN_MAJOR >= 8
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#else
        (*convolutionAlgorithms)[batchSize] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
      }
      else {
        const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize];
        const cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize];

#if CUDNN_MAJOR >= 8
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        {
          std::lock_guard<std::mutex> lock(
            CudaBackendInternal::cudnnConvolutionAlgorithmQueryMutex()
          );
          CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm_v7(
            cudaHandles->cudnn,
            inputDescriptor,
            filterDescriptor,
            convolutionDescriptor,
            outputDescriptor,
            requestedAlgoCount,
            &returnedAlgoCount,
            results
          ));
        }
        if(returnedAlgoCount <= 0)
          throw StringError("cudnnGetConvolutionForwardAlgorithm_v7 returned no algorithms?");
        (*convolutionAlgorithms)[batchSize] = results[0];
#else
        size_t bytesMemoryLimit = 0;
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm(
           cudaHandles->cudnn,
           inputDescriptor,
           filterDescriptor,
           convolutionDescriptor,
           outputDescriptor,
           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
           bytesMemoryLimit,
           &((*convolutionAlgorithms)[batchSize])
         ));
#endif
      }
    }

    testAssert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);

    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc*convYSize + y)*convXSize + x)*inChannels + ic] =
                desc->weights[((oc*inChannels + ic)*convYSize + y)*convXSize + x];
            }
          }
        }
      }
      CudaUtils::mallocAndCopyToDevice(name,weightsTransposed,filterBuf,useFP16);
      cudaDeviceSynchronize();
    }
    else
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,filterBuf,useFP16);

    initialConvFrontendRequired =
      cudaHandles->sm120InitialConvFrontendEngine != 0 &&
      cudaHandles->majorComputeCapability == 12 &&
      cudaHandles->minorComputeCapability == 0 &&
      cudnnGetVersion() >= 92400 && useFP16 && useNHWCIn && useNHWCOut &&
      manager->nnXLen == 19 && manager->nnYLen == 19 &&
      inChannels == 22 && outChannels == 768 &&
      convXSize == 3 && convYSize == 3 && dilationX == 1 && dilationY == 1;
#if KATAGO_CUDA_HAS_SDPA
    if(initialConvFrontendRequired) {
      namespace fe = cudnn_frontend;
      constexpr int64_t C = 22;
      constexpr int64_t H = 19;
      constexpr int64_t W = 19;
      constexpr int64_t K = 768;
      constexpr int64_t R = 3;
      constexpr int64_t S = 3;
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        const int64_t B = batchSize;
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
        auto x = graph->tensor(
          fe::graph::Tensor_attributes()
            .set_name("initial_conv_x").set_uid(1)
            .set_dim({B,C,H,W}).set_stride({H*W*C,1,W*C,C}));
        auto weight = graph->tensor(
          fe::graph::Tensor_attributes()
            .set_name("initial_conv_w").set_uid(2)
            .set_dim({K,C,R,S}).set_stride({R*S*C,1,S*C,C}));
        auto y = graph->conv_fprop(
          x, weight,
          fe::graph::Conv_fprop_attributes()
            .set_name("initial_conv")
            .set_padding({1,1}).set_stride({1,1}).set_dilation({1,1}));
        y->set_output(true).set_uid(3)
          .set_dim({B,K,H,W}).set_stride({H*W*K,1,W*K,K});

        fe::error_t status = graph->validate();
        if(!status.is_bad())
          status = graph->build_operation_graph(cudaHandles->cudnn);
        int64_t planIndex = -1;
        string marker;
        if(!status.is_bad() && cudaHandles->sm120InitialConvFrontendEngine == 45) {
          status = graph->create_execution_plan(
            45,{{fe::KnobType_t::TILE_SIZE,0},{fe::KnobType_t::STAGES,2}});
          if(!status.is_bad())
            status = graph->check_support(cudaHandles->cudnn);
          if(!status.is_bad())
            status = graph->build_plans(fe::BuildPlanPolicy_t::ALL);
          if(!status.is_bad()) {
            planIndex = 0;
            marker = "SM120 backend: initial-conv frontend eng45/tile0/stages2 active";
          }
        }
        else if(!status.is_bad() && cudaHandles->sm120InitialConvFrontendEngine == 47) {
          static const string targetTag =
            "eng47_k2=2_k6=1_k13=1_k14=0_k22=2";
          status = graph->create_execution_plans({fe::HeurMode_t::A});
          if(!status.is_bad()) {
            const int64_t planCount = graph->get_execution_plan_count();
            for(int64_t index = 0; index < planCount; index++) {
              string tag;
              fe::error_t nameStatus = graph->get_plan_name_at_index(index,tag);
              if(!nameStatus.is_bad() && tag == targetTag) {
                planIndex = index;
                break;
              }
            }
            if(planIndex >= 0)
              status = graph->build_plan_at_index(cudaHandles->cudnn,planIndex);
          }
          if(!status.is_bad())
            marker = "SM120 backend: initial-conv frontend eng47/k2=2/k6=1/k13=1/k14=0/k22=2 active";
        }
        int64_t workspace = 0;
        if(!status.is_bad() && planIndex >= 0)
          status = graph->get_workspace_size_plan_at_index(planIndex,workspace);
        if(!status.is_bad() && planIndex >= 0) {
          initialConvFrontendGraphs[batchSize] = graph;
          initialConvFrontendWorkspaceBytes[batchSize] = (size_t)workspace;
          initialConvFrontendPlanIndexes[batchSize] = planIndex;
          initialConvFrontendMarkers[batchSize] = marker;
        }
      }
    }
#endif
  }

  ~ConvLayer() {
    if(matmulWeightBuf != NULL)
      cudaFree(matmulWeightBuf);
    if(!use1x1Matmul) {
      cudaFree(filterBuf);
      cudnnDestroyFilterDescriptor(filterDescriptor);
      cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
      delete convolutionAlgorithms;
    }
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    if(use1x1Matmul)
      return 0;
    size_t workspaceBytes = 0;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize].algo,
      &workspaceBytes
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize],
      &workspaceBytes
    ));
#endif
    if(
      batchSize >= 0 &&
      (size_t)batchSize < initialConvFrontendWorkspaceBytes.size()
    )
      workspaceBytes = std::max(
        workspaceBytes,initialConvFrontendWorkspaceBytes[batchSize]);
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    if(use1x1Matmul) {
      // out[outC, tokens] = W[outC, inC] x in[inC, tokens]
      // where tokens = batchSize * spatial. NHWC buffers are [tokens, C] row-major = [C, tokens] column-major
      // matching cuBLAS's expectation. Same as MatMulLayer.
      int tokens = batchSize * matmulSpatialSize;
      if(cudaHandles->sm120Conv1x1 != NULL &&
         cudaHandles->sm120Conv1x1(
           cudaHandles->sm120Conv1x1Context,
           matmulWeightBuf, inputBuf, outputBuf, tokens,
           inChannels, outChannels, accumulate, usingFP16,
           cudaHandles->stream))
        return;
      if(!usingFP16) {
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CUBLAS_ERR(name.c_str(),cublasSgemm(
          cudaHandles->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const float*)matmulWeightBuf, outChannels,
          (const float*)inputBuf, inChannels,
          &beta, (float*)outputBuf, outChannels));
      }
      else {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(accumulate ? 1.0f : 0.0f);
        CUBLAS_ERR(name.c_str(),cublasHgemm(
          cudaHandles->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const half*)matmulWeightBuf, outChannels,
          (const half*)inputBuf, inChannels,
          &beta, (half*)outputBuf, outChannels));
      }
      return;
    }
#if KATAGO_CUDA_HAS_SDPA
    if(
      !accumulate && batchSize >= 0 &&
      (size_t)batchSize < initialConvFrontendGraphs.size() &&
      initialConvFrontendGraphs[batchSize] != NULL
    ) {
      std::unordered_map<int64_t,void*> tensorUidToPointer = {
        {1,inputBuf}, {2,filterBuf}, {3,outputBuf},
      };
      cudnn_frontend::error_t status = initialConvFrontendGraphs[batchSize]->execute_plan_at_index(
        cudaHandles->cudnn,tensorUidToPointer,workspaceBuf,
        initialConvFrontendPlanIndexes[batchSize]);
      if(status.is_bad())
        throw StringError(
          "SM120 initial-conv frontend execute failed: " +
          status.get_message());
      if(!loggedInitialConvFrontend[batchSize]) {
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(initialConvFrontendMarkers[batchSize]);
        loggedInitialConvFrontend[batchSize] = 1;
      }
      return;
    }
#endif
    if(!accumulate && initialConvFrontendRequired)
      throw StringError(
        "SM120 initial-conv frontend plan is unavailable for batch " +
        Global::intToString(batchSize));
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize],
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#endif
  }

};


//---------------------------------------------------------------------------------

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const int activation;
  const int nnXLen;
  const int nnYLen;

  const bool usingFP16;
  const bool usingNHWC;

  void* mergedScaleBuf;
  void* mergedBiasBuf;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(
    CudaHandles* cudaHandles,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;

    testAssert(desc->mean.size() == numChannels);
    testAssert(desc->variance.size() == numChannels);
    testAssert(desc->scale.size() == numChannels);
    testAssert(desc->bias.size() == numChannels);
    testAssert(desc->mergedScale.size() == numChannels);
    testAssert(desc->mergedBias.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->mergedScale,mergedScaleBuf,useFP16);
    CudaUtils::mallocAndCopyToDevice(name,desc->mergedBias,mergedBiasBuf,useFP16);
  }
  ~BatchNormLayer() {
    cudaFree(mergedScaleBuf);
    cudaFree(mergedBiasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* inputBuf,
    const void* maskBuf, //ok to be null
    void* outputBuf
  ) const {
    if(usingFP16 && usingNHWC && cudaHandles->sm120AffineSilu != NULL &&
       cudaHandles->sm120AffineSilu(
         cudaHandles->sm120AffineSiluContext,
         inputBuf, outputBuf, mergedScaleBuf, mergedBiasBuf, maskBuf,
         batchSize, nnXLen * nnYLen, numChannels, activation, usingFP16,
         cudaHandles->stream)) {
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      return;
    }
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation, cudaHandles->stream);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation, cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,numChannels,nnXLen*nnYLen,activation, cudaHandles->stream);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,nnXLen*nnYLen,numChannels,activation, cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

  }

};


//---------------------------------------------------------------------------------

struct SharedMatMulWeight {
  void* ptr;
  int device;

  SharedMatMulWeight(void* ptr_, int device_) : ptr(ptr_), device(device_) {}
  ~SharedMatMulWeight() {
    if(ptr == NULL)
      return;
    int oldDevice = device;
    if(cudaGetDevice(&oldDevice) == cudaSuccess && oldDevice != device)
      cudaSetDevice(device);
    cudaFree(ptr);
    if(oldDevice != device)
      cudaSetDevice(oldDevice);
  }
};

struct SharedMatMulKey {
  const MatMulLayerDesc* desc;
  int device;
  bool useFP16;

  bool operator==(const SharedMatMulKey& other) const {
    return desc == other.desc && device == other.device && useFP16 == other.useFP16;
  }
};

struct SharedMatMulKeyHash {
  size_t operator()(const SharedMatMulKey& key) const noexcept {
    size_t hash = std::hash<const void*>()((const void*)key.desc);
    hash ^= std::hash<int>()(key.device) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>()(key.useFP16) + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    return hash;
  }
};

static std::shared_ptr<SharedMatMulWeight> getSharedMatMulWeight(
  const MatMulLayerDesc* desc,
  bool useFP16,
  const string& name,
  bool& cacheHit
) {
  cacheHit = false;
  static std::mutex mutex;
  static std::unordered_map<
    SharedMatMulKey,
    std::weak_ptr<SharedMatMulWeight>,
    SharedMatMulKeyHash
  > cache;

  int device = 0;
  CUDA_ERR(name.c_str(),cudaGetDevice(&device));
  const SharedMatMulKey key{desc,device,useFP16};
  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(key);
  if(found != cache.end()) {
    std::shared_ptr<SharedMatMulWeight> existing = found->second.lock();
    if(existing != nullptr) {
      cacheHit = true;
      return existing;
    }
  }

  void* ptr = NULL;
  CudaUtils::mallocAndCopyToDevice(name,desc->weights,ptr,useFP16);
  auto weight = std::make_shared<SharedMatMulWeight>(ptr,device);
  cache[key] = weight;
  return weight;
}

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const bool usingFP16;
  void* matBuf;
  std::shared_ptr<SharedMatMulWeight> sharedWeight;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    usingFP16(useFP16),
    matBuf(NULL),
    sharedWeight(nullptr)
  {
    if(inChannels > 0 && outChannels > 0) {
      testAssert(desc->weights.size() == inChannels * outChannels);
      if(cudaHandles->sm120ShareModelWeights) {
        bool cacheHit = false;
        sharedWeight = getSharedMatMulWeight(desc,useFP16,name,cacheHit);
        matBuf = sharedWeight->ptr;
        if(cacheHit)
          cudaHandles->sm120SharedModelWeightsActive = true;
      }
      else
        CudaUtils::mallocAndCopyToDevice(name,desc->weights,matBuf,useFP16);
    }
  }

  ~MatMulLayer() {
    if(matBuf != NULL && sharedWeight == nullptr)
      cudaFree(matBuf);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles
  ) const {
    (void)cudaHandles;
    size_t workspaceBytes = 0;
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    assert(inChannels > 0 && outChannels > 0);

    if(cudaHandles->sm120MatMul != NULL &&
       cudaHandles->sm120MatMul(
         cudaHandles->sm120MatMulContext,
         cudaHandles->stream,
         matBuf,
         inputBuf,
         outputBuf,
         workspaceBuf,
         workspaceBytes,
         batchSize,
         inChannels,
         outChannels,
         usingFP16
       ))
      return;

    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        (const float*)matBuf,outChannels,
        (const float*)inputBuf,inChannels,
        &beta,
        (float*)outputBuf,outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)scratch->zeroBuf;
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        alpha,
        (const half*)matBuf,outChannels,
        (const half*)inputBuf,inChannels,
        beta,
        (half*)outputBuf,outChannels
      ));
    }

  }

};

//---------------------------------------------------------------------------------

struct MatBiasLayer {
  const string name;
  const int numChannels;
  const bool usingFP16;
  const int activation;

  void* biasBuf;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    bool useFP16,
    int activation_
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    usingFP16(useFP16),
    activation(activation_)
  {
    (void)cudaHandles;
    if(numChannels > 0) {
      testAssert(desc->weights.size() == numChannels);
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,biasBuf,useFP16);
    }
    else
      biasBuf = NULL;
  }

  ~MatBiasLayer() {
    if(numChannels > 0)
      cudaFree(biasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* matBuf
  ) const {
    assert(numChannels > 0);
    if(!usingFP16) {
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf,batchSize,numChannels,activation, cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf,batchSize,numChannels,activation, cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }

};

//---------------------------------------------------------------------------------

struct NormActConv {
  const BatchNormLayer norm;
  const ConvLayer conv;

  const int inChannels;
  const int outChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): norm(cudaHandles,normDesc,actDesc,nnX,nnY,useFP16,useNHWC),
     conv(cudaHandles,manager,convDesc,useFP16,useNHWC),
     inChannels(norm.numChannels),
     outChannels(conv.outChannels),
     nnXLen(nnX),
     nnYLen(nnY),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
    testAssert(norm.numChannels == conv.inChannels);
  }

  ~NormActConv()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inBuf,
    void* inScratchBuf,
    void* outBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    norm.apply(cudaHandles,batchSize,inBuf,maskBuf,inScratchBuf);
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("AFTER NORM "), inScratchBuf, batchSize, inChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
#endif
    conv.apply(cudaHandles,batchSize,accumulate,inScratchBuf,outBuf,workspaceBuf,workspaceBytes);
  }

};


//---------------------------------------------------------------------------------

struct ResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const NormActConv normActConv2;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->regularConv,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~ResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> midIn(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,midIn.buf,maskBuf,workspaceBuf,workspaceBytes);
    normActConv2.apply(cudaHandles,batchSize,true,midIn.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------


struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const NormActConv normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int regularChannels;
  const int gpoolChannels;
  const bool usingFP16;
  const bool usingNHWC;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,&desc->preActivation,nnX,nnY,useFP16,useNHWC),
     regularConv(cudaHandles,manager,&desc->regularConv,useFP16,useNHWC),
     gpoolConv(cudaHandles,manager,&desc->gpoolConv,useFP16,useNHWC),
     gpoolBN(cudaHandles,&desc->gpoolBN,&desc->gpoolActivation,nnX,nnY,useFP16,useNHWC),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,useFP16),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16,useNHWC),
     nnXLen(nnX),
     nnYLen(nnY),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*gpoolChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> regularScratch(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolOut2(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolConcat(scratch->allocator, scratch->getBufSize(gpoolChannels*3));
    SizedBuf<void*> gpoolBias(scratch->allocator, scratch->getBufSize(regularChannels));

    preBN.apply(cudaHandles,batchSize,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,regularOut.buf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,gpoolOut.buf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,batchSize,gpoolOut.buf,maskBuf,gpoolOut2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const float*)maskBuf,maskSumBuf, cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const float*)maskBuf,maskSumBuf, cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,gpoolChannels,nnXLen*nnYLen,(const half*)maskBuf,maskSumBuf, cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,nnXLen*nnYLen,gpoolChannels,(const half*)maskBuf,maskSumBuf, cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,gpoolConcat.buf,gpoolBias.buf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen, cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels, cudaHandles->stream);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,regularChannels,nnXLen*nnYLen, cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,nnXLen*nnYLen,regularChannels, cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    normActConv2.apply(cudaHandles,batchSize,true,regularOut.buf,regularScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct BlockStack {
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  vector<pair<int,unique_ptr_void>> blocks;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    int nBlocks,
    int trunkChannels,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  );
  ~BlockStack();

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const;

  bool apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    const BatchNormLayer* finalBN
  ) const;

};

//------------------------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const BlockStack blocks;
  const NormActConv normActConv2;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->preConv,nnX,nnY,useFP16,useNHWC),
     blocks(cudaHandles,manager,desc->numBlocks,desc->preConv.outChannels,desc->blocks,nnX,nnY,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->postBN,&desc->postActivation,&desc->postConv,nnX,nnY,useFP16,useNHWC)
  {
  }

  ~NestedBottleneckResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  bool apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    const BatchNormLayer* followingBN,
    bool preBNReady
  ) const {
    SizedBuf<void*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    assert(normActConv1.outChannels == normActConv2.inChannels);
    if(!preBNReady)
      normActConv1.norm.apply(cudaHandles,batchSize,trunkBuf,maskBuf,trunkScratchBuf);
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("AFTER NORM "), trunkScratchBuf, batchSize, normActConv1.inChannels, normActConv1.nnXLen*normActConv1.nnYLen, normActConv1.usingNHWC, normActConv1.usingFP16);
#endif
    if(cudaHandles->sm120PersistingL2Inner != NULL) {
      cudaHandles->sm120PersistingL2Inner(
        cudaHandles->sm120PersistingL2InnerContext,
        cudaHandles->stream,
        mid.buf,
        scratch->getBufSizeXY(normActConv1.outChannels));
    }
    normActConv1.conv.apply(cudaHandles,batchSize,false,trunkScratchBuf,mid.buf,workspaceBuf,workspaceBytes);
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      mid.buf,
      midScratch.buf,
      workspaceBuf,
      workspaceBytes,
      NULL
    );
    normActConv2.norm.apply(cudaHandles,batchSize,mid.buf,maskBuf,midScratch.buf);
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("AFTER NORM "), midScratch.buf, batchSize, normActConv2.inChannels, normActConv2.nnXLen*normActConv2.nnYLen, normActConv2.usingNHWC, normActConv2.usingFP16);
#endif
    if(cudaHandles->sm120PersistingL2Inner != NULL) {
      cudaHandles->sm120PersistingL2Inner(
        cudaHandles->sm120PersistingL2InnerContext,
        cudaHandles->stream,
        trunkBuf,
        scratch->getBufSizeXY(normActConv1.inChannels));
    }
    bool usedPostConvBNSilu = false;
    if(followingBN != NULL && cudaHandles->sm120PostConvBNSilu != NULL &&
       normActConv2.conv.use1x1Matmul && normActConv2.conv.matmulWeightBuf != NULL &&
       followingBN->numChannels == normActConv2.conv.outChannels) {
      usedPostConvBNSilu = cudaHandles->sm120PostConvBNSilu(
        cudaHandles->sm120PostConvBNSiluContext,
        midScratch.buf,
        normActConv2.conv.matmulWeightBuf,
        trunkBuf,
        trunkScratchBuf,
        followingBN->mergedScaleBuf,
        followingBN->mergedBiasBuf,
        maskBuf,
        batchSize,
        normActConv2.nnXLen * normActConv2.nnYLen,
        normActConv2.conv.inChannels,
        normActConv2.conv.outChannels,
        followingBN->activation,
        normActConv2.usingFP16,
        normActConv2.usingNHWC,
        cudaHandles->stream);
    }
    if(!usedPostConvBNSilu)
      normActConv2.conv.apply(
        cudaHandles,batchSize,true,midScratch.buf,trunkBuf,workspaceBuf,workspaceBytes);
    return usedPostConvBNSilu;
  }

};

//------------------------------------------------------------------------------

struct TransformerRMSNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const bool usingFP16;
  void* weightBuf;
  void* zeroBetaBuf;

  TransformerRMSNormLayer() = delete;
  TransformerRMSNormLayer(const TransformerRMSNormLayer&) = delete;
  TransformerRMSNormLayer& operator=(const TransformerRMSNormLayer&) = delete;

  TransformerRMSNormLayer(
    CudaHandles* cudaHandles,
    const TransformerRMSNormDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    usingFP16(useFP16)
  {
    (void)cudaHandles;
    testAssert((int)desc->weight.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name, desc->weight, weightBuf, useFP16);
    // Allocate a zero buffer for beta (TransformerRMSNorm has no bias)
    vector<float> zeros(numChannels, 0.0f);
    CudaUtils::mallocAndCopyToDevice(name + ":zeroBeta", zeros, zeroBetaBuf, useFP16);
  }

  ~TransformerRMSNormLayer() {
    cudaFree(weightBuf);
    cudaFree(zeroBetaBuf);
  }

  // Apply RMSNorm on NHWC data [N, XY, C], applying mask [N, XY] to zero padded positions.
  // Uses the RMSNormGammaBeta kernel with gamma=weight, beta=0, no activation.
  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    int xySize,
    void* inputBuf,
    void* outputBuf,
    const void* maskBuf
  ) const {
    if(cudaHandles->sm120RMSNorm != NULL &&
       cudaHandles->sm120RMSNorm(
         cudaHandles->sm120RMSNormContext,
         inputBuf,
         outputBuf,
         weightBuf,
         zeroBetaBuf,
         maskBuf,
         batchSize,
         xySize,
         numChannels,
         epsilon,
         usingFP16,
         cudaHandles->stream)) {
      return;
    }
    // RMSNormGammaBetaNHWC with gamma=weight, beta=zero, mask, identity activation.
    if(!usingFP16) {
      customCudaRMSNormGammaBetaNHWC(
        (const float*)inputBuf, (float*)outputBuf,
        (const float*)weightBuf, (const float*)zeroBetaBuf,
        (const float*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY, cudaHandles->stream);
    }
    else {
      customCudaRMSNormGammaBetaNHWC(
        (const half*)inputBuf, (half*)outputBuf,
        (const half*)weightBuf, (const half*)zeroBetaBuf,
        (const half*)maskBuf,
        batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY, cudaHandles->stream);
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

//------------------------------------------------------------------------------

struct RMSNormLayer {
  const string name;
  const int numChannels;
  const bool spatial;
  const int activation;
  const float epsilon;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  void* gammaBuf;
  void* betaBuf;

  RMSNormLayer() = delete;
  RMSNormLayer(const RMSNormLayer&) = delete;
  RMSNormLayer& operator=(const RMSNormLayer&) = delete;

  RMSNormLayer(
    CudaHandles* cudaHandles,
    const RMSNormLayerDesc* desc,
    int act,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    spatial(desc->spatial),
    activation(act),
    epsilon(desc->epsilon),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC)
  {
    (void)cudaHandles;
    testAssert((int)desc->gamma.size() == numChannels);
    testAssert((int)desc->beta.size() == numChannels);
    // The device kernels apply only RELU/MISH/SILU explicitly and treat anything else as
    // identity; guard here so an unsupported kind (e.g. MISH_SCALE8, which applyScale8 can
    // produce for non-transformer nets) fails loudly instead of silently skipping activation.
    if(activation != ACTIVATION_IDENTITY && activation != ACTIVATION_RELU &&
       activation != ACTIVATION_MISH && activation != ACTIVATION_SILU)
      throw StringError(name + ": RMSNorm layer unsupported activation: " + Global::intToString(activation));
    CudaUtils::mallocAndCopyToDevice(name, desc->gamma, gammaBuf, useFP16);
    CudaUtils::mallocAndCopyToDevice(name, desc->beta, betaBuf, useFP16);
  }

  ~RMSNormLayer() {
    cudaFree(gammaBuf);
    cudaFree(betaBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    const void* maskBuf,
    const float* maskSumBuf
  ) const {
    int xySize = nnXLen * nnYLen;
    if(!spatial) {
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, numChannels, xySize, epsilon, activation, cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, batchSize, xySize, numChannels, epsilon, activation, cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaRMSNormGammaBetaNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, numChannels, xySize, epsilon, activation, cudaHandles->stream);
        else
          customCudaRMSNormGammaBetaNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, activation, cudaHandles->stream);
      }
    }
    else {
      // Allocate temp buffer for spatial reduction from scratch (float regardless of FP16 mode).
      // Holds per-block partial sums plus the final reduced value per batch element; see
      // SPATIAL_RMSNORM_BLOCKS_PER_BATCH in cudahelpers.cu (partialStride = that + 1).
      SizedBuf<void*> sumSqBuf(scratch->allocator, (size_t)batchSize * CUDA_SPATIAL_RMSNORM_SUMSQ_STRIDE * sizeof(float));
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const float*)inputBuf, (float*)outputBuf, (const float*)gammaBuf, (const float*)betaBuf,
            (const float*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaSpatialRMSNormNCHW(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, numChannels, xySize, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
        else
          customCudaSpatialRMSNormNHWC(
            (const half*)inputBuf, (half*)outputBuf, (const half*)gammaBuf, (const half*)betaBuf,
            (const half*)maskBuf, maskSumBuf, batchSize, xySize, numChannels, epsilon, activation, (float*)sumSqBuf.buf, cudaHandles->stream);
      }
    }
    CUDA_ERR(name.c_str(), cudaPeekAtLastError());
  }
};

//------------------------------------------------------------------------------

struct TransformerAttentionBlock {
  const string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const bool useRope;
  const bool learnableRope;
  const int inChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer qProj;
  const MatMulLayer kProj;
  const MatMulLayer vProj;
  const MatMulLayer outProj;

  // Fixed RoPE: precomputed cos/sin tables on device (NULL for learnable RoPE).
  // Learnable RoPE: per-head frequencies on device (ropeFreqsBuf, FP32), cos/sin recomputed in-kernel.
  void* ropeCosTable;
  void* ropeSinTable;
  float* ropeFreqsBuf;
  int ropeNumPairs;
  int ropeNumKVHeads;

  TransformerAttentionBlock() = delete;
  TransformerAttentionBlock(const TransformerAttentionBlock&) = delete;
  TransformerAttentionBlock& operator=(const TransformerAttentionBlock&) = delete;

  TransformerAttentionBlock(
    CudaHandles* cudaHandles,
    const TransformerAttentionDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numHeads(desc->numHeads),
    numKVHeads(desc->numKVHeads),
    qHeadDim(desc->qHeadDim),
    vHeadDim(desc->vHeadDim),
    useRope(desc->useRope),
    learnableRope(desc->learnableRope),
    inChannels(desc->qProj.inChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles, &desc->preLN, useFP16),
    qProj(cudaHandles, &desc->qProj, useFP16),
    kProj(cudaHandles, &desc->kProj, useFP16),
    vProj(cudaHandles, &desc->vProj, useFP16),
    outProj(cudaHandles, &desc->outProj, useFP16),
    ropeCosTable(NULL),
    ropeSinTable(NULL),
    ropeFreqsBuf(NULL),
    ropeNumPairs(0),
    ropeNumKVHeads(0)
  {
    if(!useNHWC) {
      throw StringError("Transformer blocks with NCHW layout are not yet supported by the CUDA backend");
    }
    if(useRope) {
      ropeNumPairs = qHeadDim / 2;
      ropeNumKVHeads = numKVHeads;
      if(learnableRope) {
        // Table-free: upload the tiny per-head frequencies (FP32) and recompute cos/sin in-kernel.
        // Avoids the numKVHeads-times-larger cos/sin table that otherwise spills L2 (see kernel comment).
        testAssert(desc->ropeFreqs.size() == (size_t)(numKVHeads * ropeNumPairs * 2));
        void* freqsVoid = NULL;
        CudaUtils::mallocAndCopyToDevice(name + ":ropeFreqs", desc->ropeFreqs.data(), (int)desc->ropeFreqs.size(), freqsVoid, false);
        ropeFreqsBuf = (float*)freqsVoid;
      }
      else {
        int seqLen = nnXLen * nnYLen;
        vector<float> cosTableData;
        vector<float> sinTableData;
        desc->computeRopeCosSin(nnXLen, nnYLen, seqLen, cosTableData, sinTableData);
        CudaUtils::mallocAndCopyToDevice(name + ":ropeCos", cosTableData.data(), (int)cosTableData.size(), ropeCosTable, useFP16);
        CudaUtils::mallocAndCopyToDevice(name + ":ropeSin", sinTableData.data(), (int)sinTableData.size(), ropeSinTable, useFP16);
      }
    }
  }

  ~TransformerAttentionBlock() {
    if(ropeCosTable != NULL) cudaFree(ropeCosTable);
    if(ropeSinTable != NULL) cudaFree(ropeSinTable);
    if(ropeFreqsBuf != NULL) cudaFree(ropeFreqsBuf);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)cudaHandles;
    (void)batchSize;
    return 0;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)maskSumBuf;
    (void)workspaceBuf;
    (void)workspaceBytes;

    int seqLen = nnXLen * nnYLen;
    int qTotalDim = numHeads * qHeadDim;
    int kTotalDim = numKVHeads * qHeadDim;
    int vTotalDim = numKVHeads * vHeadDim;
    size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);

    // NHWC: trunk is [N, XY, C]. RMSNorm + mask zeroing.
    preLN.apply(cudaHandles, batchSize, seqLen, trunkBuf, trunkScratchBuf, maskBuf);

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Attn RMSNorm out", trunkScratchBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Step 2: Q/K/V projections
    // trunkScratchBuf is [N, XY, C] NHWC = [C, N*seqLen] column-major.
    // MatMulLayer expects input as [inChannels, batchSize], which matches.
    int matBatchSize = batchSize * seqLen;

    std::optional<SizedBuf<void*> > qkvStorage;
    std::optional<SizedBuf<void*> > qStorage;
    std::optional<SizedBuf<void*> > kStorage;
    std::optional<SizedBuf<void*> > vStorage;
    void* qBuf;
    void* kBuf;
    void* vBuf;
    bool packedQKV = false;
    bool qkvRopeApplied = false;
    if(cudaHandles->sm120QKVStrided != NULL) {
      qkvStorage.emplace(
        scratch->allocator,
        (size_t)(qTotalDim + kTotalDim + vTotalDim) * matBatchSize * bytesPerElt);
      qBuf = qkvStorage->buf;
      kBuf = (char*)qBuf + (size_t)qTotalDim * matBatchSize * bytesPerElt;
      vBuf = (char*)kBuf + (size_t)kTotalDim * matBatchSize * bytesPerElt;
      const bool allowPackedOutput = useRope && learnableRope && maskBuf == NULL;
      const float* qkvRopeFreqs =
        useRope && learnableRope && numHeads == 12 && numKVHeads == 12 &&
        qHeadDim == 32 && ropeNumPairs == 16 ? ropeFreqsBuf : NULL;
      bool usedQKVStrided = cudaHandles->sm120QKVStrided(
        cudaHandles->sm120QKVStridedContext,
        cudaHandles->cublas,
        cudaHandles->stream,
        qProj.matBuf,
        kProj.matBuf,
        vProj.matBuf,
        trunkScratchBuf,
        qBuf,
        allowPackedOutput,
        &packedQKV,
        qkvRopeFreqs,
        &qkvRopeApplied,
        matBatchSize,
        inChannels,
        qTotalDim,
        kTotalDim,
        vTotalDim,
        usingFP16);
      if(usedQKVStrided && packedQKV) {
        // The AOT buffer is [token,Q384,K384,V384].
        kBuf = (char*)qBuf + (size_t)qTotalDim * bytesPerElt;
        vBuf = (char*)qBuf + (size_t)(qTotalDim + kTotalDim) * bytesPerElt;
      }
      else if(!usedQKVStrided) {
        qProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, qBuf, workspaceBuf, workspaceBytes);
        kProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, kBuf, workspaceBuf, workspaceBytes);
        vProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, vBuf, workspaceBuf, workspaceBytes);
      }
    }
    else {
      qStorage.emplace(scratch->allocator, (size_t)qTotalDim * matBatchSize * bytesPerElt);
      kStorage.emplace(scratch->allocator, (size_t)kTotalDim * matBatchSize * bytesPerElt);
      vStorage.emplace(scratch->allocator, (size_t)vTotalDim * matBatchSize * bytesPerElt);
      qBuf = qStorage->buf;
      kBuf = kStorage->buf;
      vBuf = vStorage->buf;
      qProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, qBuf, workspaceBuf, workspaceBytes);
      kProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, kBuf, workspaceBuf, workspaceBytes);
      vProj.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, vBuf, workspaceBuf, workspaceBytes);
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint2D("CUDA Attn Q", qBuf, matBatchSize, qTotalDim, usingFP16);
#endif

    // Step 3: Apply RoPE to Q and K
    // Q is [qTotalDim, seqLen*batchSize] column-major = [batchSize*seqLen, qTotalDim] row-major
    if(useRope) {
      bool usedFusedQKRoPE = qkvRopeApplied;
      if(!usedFusedQKRoPE && learnableRope &&
         cudaHandles->sm120FusedQKRoPE != NULL) {
        usedFusedQKRoPE = cudaHandles->sm120FusedQKRoPE(
          cudaHandles->sm120FusedQKRoPEContext,
          qBuf, kBuf, ropeFreqsBuf, batchSize, seqLen, numHeads, numKVHeads,
          qHeadDim, ropeNumPairs, nnXLen, packedQKV, usingFP16, cudaHandles->stream);
      }
      if(packedQKV && !usedFusedQKRoPE)
        throw StringError("CUDA attention: packed QKV requires the SM120 packed RoPE path");
      if(!usedFusedQKRoPE && learnableRope) {
        // Table-free recompute path (cos/sin computed in-kernel from ropeFreqsBuf).
        if(!usingFP16) {
          customCudaApplyRoPELearnableRecompute((float*)qBuf, ropeFreqsBuf,
            batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen, cudaHandles->stream);
          customCudaApplyRoPELearnableRecompute((float*)kBuf, ropeFreqsBuf,
            batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen, cudaHandles->stream);
        }
        else {
          customCudaApplyRoPELearnableRecompute((half*)qBuf, ropeFreqsBuf,
            batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen, cudaHandles->stream);
          customCudaApplyRoPELearnableRecompute((half*)kBuf, ropeFreqsBuf,
            batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen, cudaHandles->stream);
        }
      }
      else if(!usedFusedQKRoPE) {
        if(!usingFP16) {
          customCudaApplyRoPE((float*)qBuf, (const float*)ropeCosTable, (const float*)ropeSinTable,
            batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope, cudaHandles->stream);
          customCudaApplyRoPE((float*)kBuf, (const float*)ropeCosTable, (const float*)ropeSinTable,
            batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope, cudaHandles->stream);
        }
        else {
          customCudaApplyRoPE((half*)qBuf, (const half*)ropeCosTable, (const half*)ropeSinTable,
            batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope, cudaHandles->stream);
          customCudaApplyRoPE((half*)kBuf, (const half*)ropeCosTable, (const half*)ropeSinTable,
            batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, learnableRope, cudaHandles->stream);
        }
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

    // Step 4: Scaled dot-product attention.
    // We use cudnn SDPA (FlashAttention-style, fused, no score-matrix materialization) when available
    // (FP16 + cudnn >= 8.9.3 + supported GPU). Otherwise fall back to a custom online-softmax CUDA kernel.
    // Both paths consume Q/K/V in BSHD layout and produce attnOut in the same layout as expected by outProj:
    //   attnOut: [numHeads*vHeadDim, seqLen*batchSize] col-major = [batchSize*seqLen, numHeads*vHeadDim] row-major.

    SizedBuf<void*> attnOutBuf(scratch->allocator, (size_t)numHeads * vHeadDim * seqLen * batchSize * bytesPerElt);

    bool usedSDPA = false;
    // Thin SM120 dispatch: the FA4 AOT kernel lives in cudabackend_sm120.cpp.
    if(cudaHandles->sm120Attention != NULL &&
       cudaHandles->sm120Attention(
         cudaHandles->sm120AttentionContext,
         cudaHandles,
         scratch,
         qBuf,
         kBuf,
         vBuf,
         packedQKV,
         maskBuf,
         attnOutBuf.buf,
         batchSize,
         seqLen,
         numHeads,
         numKVHeads,
         qHeadDim,
         vHeadDim,
         usingFP16,
         cudaHandles->stream,
         workspaceBuf,
         workspaceBytes
       )) {
      usedSDPA = true;
    }
    if(packedQKV && !usedSDPA)
      throw StringError("CUDA attention: packed QKV requires the SM120 FA4 path");
#if KATAGO_CUDA_HAS_SDPA
    SDPAGraphCache* sdpaCache = cudaHandles->sdpaCache.get();
    // Report once if cudaDisableGraphSDPA is the only reason we are not taking the cudnn graph SDPA path,
    // i.e. FP16 and a cache are available so SDPA would otherwise have been used.
    if(usingFP16 && sdpaCache != NULL && cudaHandles->cudaDisableGraphSDPA && !cudaHandles->loggedGraphSDPADisabled) {
      cudaHandles->loggedGraphSDPADisabled = true;
      if(cudaHandles->logger != NULL)
        cudaHandles->logger->write(
          "Cuda backend: cudaDisableGraphSDPA is set, using the custom attention kernel instead of the cudnn graph SDPA path that would otherwise have been used");
    }
    if(!usedSDPA && usingFP16 && sdpaCache != NULL && !cudaHandles->cudaDisableGraphSDPA) {
      bool hasMask = (maskBuf != NULL);
      SDPAGraphKey sdpaKey = {numHeads, numKVHeads, qHeadDim, vHeadDim, seqLen, batchSize, hasMask, usingFP16};
      auto plan = sdpaCache->getOrBuildPlan(cudaHandles->cudnn, sdpaKey, cudaHandles->logger, cudaHandles->isWarmup);
      if(plan != nullptr) {
        std::unordered_map<int64_t, void*> variant_pack = {
          {SDPAPlanForBatchSize::Q_UID, qBuf},
          {SDPAPlanForBatchSize::K_UID, kBuf},
          {SDPAPlanForBatchSize::V_UID, vBuf},
          {SDPAPlanForBatchSize::O_UID, attnOutBuf.buf},
        };

        // When a mask is present, materialize a [B, 1, S, S] additive bias: bias[b,q,k] = (mask[b,k] != 0 ? 0 : -1e4).
        // For our test model (B=16, S=361) this is ~4 MB; the bias only depends on the mask, but
        // we rebuild it per attention block for simplicity (the mask kernel itself is cheap).
        SizedBuf<void*> biasBuf(scratch->allocator, hasMask ? (size_t)batchSize * seqLen * seqLen * bytesPerElt : 1);
        if(hasMask) {
          customCudaMaskToAttnBiasFull((const half*)maskBuf, (half*)biasBuf.buf, batchSize, seqLen, cudaHandles->stream);
          variant_pack[SDPAPlanForBatchSize::BIAS_UID] = biasBuf.buf;
        }

        // Workspace from cudnn (separate from the conv workspace - different shape and lifetime).
        SizedBuf<void*> sdpaWs(scratch->allocator, (size_t)plan->workspaceBytes);

        auto status = plan->graph->execute(cudaHandles->cudnn, variant_pack, sdpaWs.buf);
        if(status.is_bad()) {
          string reason = string("cudnn SDPA execute failed: ") + status.get_message();
          // During warmup we tolerate this: disable SDPA from here on and fall through to the custom
          // kernel. Outside of warmup a failure here is fatal - the plan was already validated and
          // built, so an execute failure means something is genuinely wrong.
          if(!cudaHandles->isWarmup)
            throw StringError(reason);
          sdpaCache->sdpaSupported = false;
          sdpaCache->disableReason = reason;
          if(cudaHandles->logger != NULL)
            cudaHandles->logger->write("Cuda backend: disabling cudnn SDPA and falling back to custom attention kernel: " + reason);
        }
        else {
          usedSDPA = true;
        }
      }
    }
#endif

    if(!usedSDPA) {
      if(!usingFP16) {
        customCudaFlashAttention(
          (const float*)qBuf, (const float*)kBuf, (const float*)vBuf,
          (const float*)maskBuf, (float*)attnOutBuf.buf,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim, cudaHandles->stream);
      }
      else {
        customCudaFlashAttention(
          (const half*)qBuf, (const half*)kBuf, (const half*)vBuf,
          (const half*)maskBuf, (half*)attnOutBuf.buf,
          batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim, cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

    // Steps 5-6: output projection and residual addition. On fixed full-board
    // SM120 inputs the projection can accumulate directly into trunk.
    // attnOutBuf is [numHeads*vHeadDim, seqLen*batchSize] col-major
    // outProj maps to [inChannels, seqLen*batchSize]
    bool usedFusedResidual = false;
    if(cudaHandles->sm120FusedResidualGemm != NULL) {
      usedFusedResidual = cudaHandles->sm120FusedResidualGemm(
        cudaHandles->sm120FusedResidualGemmContext,
        cudaHandles->cublas,
        cudaHandles->stream,
        outProj.matBuf,
        attnOutBuf.buf,
        trunkBuf,
        maskBuf,
        matBatchSize,
        outProj.inChannels,
        outProj.outChannels,
        usingFP16);
    }
    if(!usedFusedResidual)
      outProj.apply(cudaHandles, scratch, matBatchSize, attnOutBuf.buf, trunkScratchBuf, workspaceBuf, workspaceBytes);

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Attn outProj", trunkScratchBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Step 6: Residual addition: trunk += trunkScratch * mask
    // NHWC: trunk is [N, XY, C], mask is [N, XY]
    if(!usedFusedResidual) {
      if(!usingFP16) {
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, inChannels, cudaHandles->stream);
      }
      else {
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, inChannels, cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Attn residual", trunkBuf, batchSize, inChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif
  }
};

//------------------------------------------------------------------------------

struct TransformerFFNBlock {
  const string name;
  const int numChannels;
  const int ffnChannels;
  const bool useSwiGLU;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  const TransformerRMSNormLayer preLN;
  const MatMulLayer linear1;
  std::unique_ptr<MatMulLayer> linearGate;
  const MatMulLayer linear2;

  TransformerFFNBlock() = delete;
  TransformerFFNBlock(const TransformerFFNBlock&) = delete;
  TransformerFFNBlock& operator=(const TransformerFFNBlock&) = delete;

  TransformerFFNBlock(
    CudaHandles* cudaHandles,
    const TransformerFFNDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    ffnChannels(desc->ffnChannels),
    useSwiGLU(desc->useSwiGLU),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    preLN(cudaHandles, &desc->preLN, useFP16),
    linear1(cudaHandles, &desc->linear1, useFP16),
    linear2(cudaHandles, &desc->linear2, useFP16)
  {
    if(!useSwiGLU) {
      throw StringError("Non-SwiGLU transformer FFN is not yet supported in CUDA backend");
    }
    linearGate = std::make_unique<MatMulLayer>(cudaHandles, &desc->linearGate, useFP16);
    if(!useNHWC) {
      throw StringError("Transformer blocks with NCHW layout are not yet supported by the CUDA backend");
    }
  }

  ~TransformerFFNBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)cudaHandles;
    (void)batchSize;
    return 0;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)maskSumBuf;

    int seqLen = nnXLen * nnYLen;
    int matBatchSize = batchSize * seqLen;
    size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);

    // Step 1: RMSNorm
    preLN.apply(cudaHandles, batchSize, seqLen, trunkBuf, trunkScratchBuf, maskBuf);

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA FFN RMSNorm out", trunkScratchBuf, batchSize, numChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif

    // Step 2: linear1 projection
    SizedBuf<void*> ffnBuf(scratch->allocator, (size_t)ffnChannels * matBatchSize * bytesPerElt);
    bool usedWideFFNSingleGemm = false;
    if(cudaHandles->sm120FFNSingleGemm != NULL) {
      SizedBuf<void*> wideFFNBuf(
        scratch->allocator, (size_t)ffnChannels * 2 * matBatchSize * bytesPerElt);
      usedWideFFNSingleGemm = cudaHandles->sm120FFNSingleGemm(
        cudaHandles->sm120FFNSingleGemmContext,
        cudaHandles->cublas,
        cudaHandles->stream,
        linear1.matBuf,
        linearGate->matBuf,
        trunkScratchBuf,
        wideFFNBuf.buf,
        ffnBuf.buf,
        matBatchSize,
        numChannels,
        ffnChannels,
        usingFP16);
    }
    if(!usedWideFFNSingleGemm)
      linear1.apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, ffnBuf.buf, workspaceBuf, workspaceBytes);

    // Step 3: SwiGLU
    if(!usedWideFFNSingleGemm) {
      SizedBuf<void*> gateBuf(scratch->allocator, (size_t)ffnChannels * matBatchSize * bytesPerElt);
      linearGate->apply(cudaHandles, scratch, matBatchSize, trunkScratchBuf, gateBuf.buf, workspaceBuf, workspaceBytes);

      if((size_t)ffnChannels * (size_t)matBatchSize >= (size_t)2147483647)
        throw StringError("CUDA SwiGLU element count exceeds the 32-bit index limit used by the kernel");
      int totalSize = (int)((size_t)ffnChannels * matBatchSize);
      bool usedSm120SwiGLU = false;
      if(cudaHandles->sm120SwiGLU != NULL) {
        usedSm120SwiGLU = cudaHandles->sm120SwiGLU(
          cudaHandles->sm120SwiGLUContext,
          ffnBuf.buf, gateBuf.buf, ffnBuf.buf, matBatchSize, ffnChannels,
          usingFP16, cudaHandles->stream);
      }
      if(!usedSm120SwiGLU && !usingFP16) {
        customCudaSwiGLU((const float*)ffnBuf.buf, (const float*)gateBuf.buf, (float*)ffnBuf.buf, totalSize, cudaHandles->stream);
      }
      else if(!usedSm120SwiGLU) {
        customCudaSwiGLU((const half*)ffnBuf.buf, (const half*)gateBuf.buf, (half*)ffnBuf.buf, totalSize, cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint2D("CUDA FFN SwiGLU", ffnBuf.buf, matBatchSize, ffnChannels, usingFP16);
#endif

    // Steps 4-5: linear2 and residual addition.
    bool usedFusedResidual = false;
    if(cudaHandles->sm120FusedResidualGemm != NULL) {
      usedFusedResidual = cudaHandles->sm120FusedResidualGemm(
        cudaHandles->sm120FusedResidualGemmContext,
        cudaHandles->cublas,
        cudaHandles->stream,
        linear2.matBuf,
        ffnBuf.buf,
        trunkBuf,
        maskBuf,
        matBatchSize,
        linear2.inChannels,
        linear2.outChannels,
        usingFP16);
    }
    if(!usedFusedResidual)
      linear2.apply(cudaHandles, scratch, matBatchSize, ffnBuf.buf, trunkScratchBuf, workspaceBuf, workspaceBytes);

    // Step 5: Residual addition: trunk += trunkScratch * mask
    if(!usedFusedResidual) {
      if(!usingFP16) {
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, numChannels, cudaHandles->stream);
      }
      else {
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, numChannels, cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(), cudaPeekAtLastError());
    }

#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA FFN residual", trunkBuf, batchSize, numChannels, seqLen, usingNHWC, usingFP16, maskBuf);
#endif
  }
};

//------------------------------------------------------------------------------

BlockStack::BlockStack(
  CudaHandles* cudaHandles,
  CudnnManager* manager,
  int nBlocks,
  int trunkChannels,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int nnX,
  int nnY,
  bool useFP16,
  bool useNHWC
) :
  numBlocks(nBlocks),
  trunkNumChannels(trunkChannels),
  nnXLen(nnX),
  nnYLen(nnY),
  usingFP16(useFP16),
  usingNHWC(useNHWC)
{
  testAssert(numBlocks == descBlocks.size());
  for(int i = 0; i<numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new ResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(ORDINARY_BLOCK_KIND,std::move(blockPtr));
    }
    else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new GlobalPoolingResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(GLOBAL_POOLING_BLOCK_KIND,std::move(blockPtr));
    }
    else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* blockDesc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new NestedBottleneckResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr));
    }
    else if(descBlocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionDesc* blockDesc = (TransformerAttentionDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerAttentionBlock(
          cudaHandles,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(TRANSFORMER_ATTENTION_BLOCK_KIND, std::move(blockPtr));
    }
    else if(descBlocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNDesc* blockDesc = (TransformerFFNDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new TransformerFFNBlock(
          cudaHandles,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16,
          useNHWC
        )
      );
      blocks.emplace_back(TRANSFORMER_FFN_BLOCK_KIND, std::move(blockPtr));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}
BlockStack::~BlockStack() {
}

size_t BlockStack::requiredWorkspaceBytes(
  CudaHandles* cudaHandles,
  int batchSize
) const {
  size_t bytes = 0;
  size_t b;

  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      TransformerAttentionBlock* block = (TransformerAttentionBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles, batchSize);
      bytes = std::max(bytes, b);
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      TransformerFFNBlock* block = (TransformerFFNBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles, batchSize);
      bytes = std::max(bytes, b);
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return bytes;
}

bool BlockStack::apply(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  int batchSize,
  void* maskBuf,
  float* maskSumBuf,
  void* trunkBuf,
  void* trunkScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes,
  const BatchNormLayer* finalBN
) const {

  bool preBNReady = false;
  for(int i = 0; i<blocks.size(); i++) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D("CUDA Blockstack block " + Global::intToString(i), trunkBuf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16, maskBuf);
#endif

    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      preBNReady = false;
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      preBNReady = false;
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      const BatchNormLayer* followingBN = NULL;
      if(i + 1 < blocks.size() &&
         blocks[i+1].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        NestedBottleneckResidualBlock* nextBlock =
          (NestedBottleneckResidualBlock*)blocks[i+1].second.get();
        followingBN = &nextBlock->normActConv1.norm;
      }
      else if(i + 1 == blocks.size()) {
        followingBN = finalBN;
      }
      preBNReady = block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes,
        followingBN,
        preBNReady
      );
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      preBNReady = false;
      TransformerAttentionBlock* block = (TransformerAttentionBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      preBNReady = false;
      TransformerFFNBlock* block = (TransformerFFNBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return preBNReady;
}
//------------------------------------------------------------------------------

struct SGFMetadataEncoder {
  const string name;

  const bool usingFP16;

  const MatMulLayer mul1;
  const MatBiasLayer bias1;
  const MatMulLayer mul2;
  const MatBiasLayer bias2;
  const MatMulLayer mul3;

  SGFMetadataEncoder() = delete;
  SGFMetadataEncoder(const SGFMetadataEncoder&) = delete;
  SGFMetadataEncoder& operator=(const SGFMetadataEncoder&) = delete;

  SGFMetadataEncoder(
    CudaHandles* cudaHandles,
    const SGFMetadataEncoderDesc* desc,
    bool useFP16
  ) :
    name(desc->name),
    usingFP16(useFP16),
    mul1(cudaHandles,&desc->mul1,useFP16),
    bias1(cudaHandles,&desc->bias1,useFP16,desc->act1.activation),
    mul2(cudaHandles,&desc->mul2,useFP16),
    bias2(cudaHandles,&desc->bias2,useFP16,desc->act2.activation),
    mul3(cudaHandles,&desc->mul3,useFP16)
  {
  }

  ~SGFMetadataEncoder()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    (void)batchSize;
    size_t bytes = 0;
    size_t b;

    b = mul1.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = mul2.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = mul3.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> internalBuf1(scratch->allocator, scratch->getBufSizeFloat(std::max(mul1.outChannels,mul2.outChannels)));
    SizedBuf<void*> internalBuf2(scratch->allocator, scratch->getBufSizeFloat(std::max(mul1.outChannels,mul2.outChannels)));

    mul1.apply(cudaHandles,scratch,batchSize,inputBuf,internalBuf1.buf,workspaceBuf,workspaceBytes);
    bias1.apply(cudaHandles,batchSize,internalBuf1.buf);
    mul2.apply(cudaHandles,scratch,batchSize,internalBuf1.buf,internalBuf2.buf,workspaceBuf,workspaceBytes);
    bias2.apply(cudaHandles,batchSize,internalBuf2.buf);
    mul3.apply(cudaHandles,scratch,batchSize,internalBuf2.buf,outputBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------

struct Trunk {
  const string name;
  const int modelVersion;
  const int numBlocks;
  const int trunkNumChannels;

  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  std::unique_ptr<SGFMetadataEncoder> sgfMetadataEncoder;
  const BlockStack blocks;
  const int trunkNormKind;
  std::unique_ptr<BatchNormLayer> trunkTipBN;
  std::unique_ptr<RMSNormLayer> trunkTipRMSNorm;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const TrunkDesc* desc,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    numBlocks(desc->numBlocks),
    trunkNumChannels(desc->trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    blocks(cudaHandles,manager,desc->numBlocks,desc->trunkNumChannels,desc->blocks,nnX,nnY,useFP16,useNHWC),
    trunkNormKind(desc->trunkNormKind)
  {
    int midNumChannels = desc->midNumChannels;
    int regularNumChannels = desc->regularNumChannels;
    int gpoolNumChannels = desc->gpoolNumChannels;

    int maxBatchSize = manager->maxBatchSize;
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,trunkNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,midNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,regularNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->initialConv,useFP16,inputsUseNHWC,useNHWC);
    initialMatMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->initialMatMul,useFP16);
    if(desc->metaEncoderVersion > 0) {
      sgfMetadataEncoder = std::make_unique<SGFMetadataEncoder>(cudaHandles,&desc->sgfMetadataEncoder,useFP16);
      testAssert(sgfMetadataEncoder->mul3.outChannels == initialMatMul->outChannels);
    }

    if(desc->trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      trunkTipBN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->trunkTipBN,&desc->trunkTipActivation,nnXLen,nnYLen,useFP16,useNHWC);
    }
    else if(desc->trunkNormKind == TRUNK_NORM_KIND_RMSNORM) {
      trunkTipRMSNorm = std::make_unique<RMSNormLayer>(cudaHandles,&desc->trunkTipRMSNorm,desc->trunkTipActivation.activation,nnXLen,nnYLen,useFP16,useNHWC);
    }
    else {
      throw StringError("Unsupported trunk norm kind: " + Global::intToString(desc->trunkNormKind));
    }
    testAssert(desc->blocks.size() == numBlocks);
  }

  ~Trunk()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = initialConv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    b = initialMatMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    if(sgfMetadataEncoder != nullptr) {
      b = sgfMetadataEncoder->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> trunkScratch(scratch->allocator, scratch->getBufSizeXY(trunkNumChannels));

    if(cudaHandles->sm120PersistingL2 != NULL) {
      cudaHandles->sm120PersistingL2(
        cudaHandles->sm120PersistingL2Context,
        cudaHandles->stream,
        trunkScratch.buf,
        scratch->getBufSizeXY(trunkNumChannels));
    }

    //Feed the conv into trunkScratch.buf, not trunkBuf
    initialConv->apply(cudaHandles,batchSize,false,inputBuf,trunkScratch.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("After initial conv"), trunkScratch.buf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
    #endif

    const bool usedSm120InitialGlobal =
      cudaHandles->sm120InitialGlobal != NULL &&
      cudaHandles->sm120InitialGlobal(
        cudaHandles->sm120InitialGlobalContext,
        trunkScratch.buf, inputGlobalBuf, initialMatMul->matBuf,
        batchSize, initialMatMul->inChannels, initialMatMul->outChannels,
        usingFP16, usingNHWC, cudaHandles->stream);
    if(!usedSm120InitialGlobal) {
      //Feed the matmul into trunkBuf
      initialMatMul->apply(cudaHandles,scratch,batchSize,inputGlobalBuf,trunkBuf,workspaceBuf,workspaceBytes);
      //Then accumulate it into trunkScratch.buf, broadcasting during the process
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen, cudaHandles->stream);
        else
          customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels, cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen, cudaHandles->stream);
        else
          customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels, cudaHandles->stream);
      }
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    if(sgfMetadataEncoder != nullptr) {
      testAssert(inputMetaBuf != NULL);
      //Feed the result into trunkBuf
      sgfMetadataEncoder->apply(cudaHandles,scratch,batchSize,inputMetaBuf,trunkBuf,workspaceBuf,workspaceBytes);
      //Then accumulate it into trunkScratch.buf, broadcasting during the process
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen, cudaHandles->stream);
        else
          customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels, cudaHandles->stream);
      }
      else {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,nnXLen*nnYLen, cudaHandles->stream);
        else
          customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,nnXLen*nnYLen,trunkNumChannels, cudaHandles->stream);
      }
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      testAssert(inputMetaBuf == NULL);
    }

    //Flip trunkBuf and trunkScratch.buf so that the result gets accumulated in trunkScratch.buf
    bool trunkTipReady = blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkScratch.buf,
      trunkBuf,
      workspaceBuf,
      workspaceBytes,
      trunkNormKind == TRUNK_NORM_KIND_STANDARD ? trunkTipBN.get() : NULL
    );

    //And now with the final norm port it from trunkScratch.buf to trunkBuf.
    if(trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
      if(!trunkTipReady)
        trunkTipBN->apply(cudaHandles,batchSize,trunkScratch.buf,maskBuf,trunkBuf);
    }
    else {
      trunkTipRMSNorm->apply(cudaHandles,scratch,batchSize,trunkScratch.buf,trunkBuf,maskBuf,maskSumBuf);
    }

    if(cudaHandles->sm120PersistingL2 != NULL) {
      cudaHandles->sm120PersistingL2(
        cudaHandles->sm120PersistingL2Context,
        cudaHandles->stream,
        NULL,
        0);
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("Trunk tip"), trunkBuf, batchSize, trunkNumChannels, nnXLen*nnYLen, usingNHWC, usingFP16);
    #endif
  }

};

//------------------------------------------------------------------------------

static void fillMaskFloatBufAndMaskSumBuf(CudaHandles* cudaHandles, void* maskBuf, float*& maskFloatBuf, float*& maskSumBuf, bool usingFP16, int batchSize, int nnXLen, int nnYLen) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0, cudaHandles->stream);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*nnXLen*nnYLen, cudaHandles->stream);
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,nnXLen*nnYLen,1.0, cudaHandles->stream);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
}


//------------------------------------------------------------------------------

struct PolicyHead {
  const string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer p1Conv;
  const ConvLayer g1Conv;
  const BatchNormLayer g1BN;
  const MatMulLayer gpoolToBiasMul;
  const BatchNormLayer p1BN;
  const ConvLayer p2Conv;
  const MatMulLayer gpoolToPassMul;
  const MatBiasLayer gpoolToPassBias;
  const MatMulLayer gpoolToPassMul2;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnX),
    nnYLen(nnY),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    p1Conv(cudaHandles,manager,&desc->p1Conv,useFP16,useNHWC),
    g1Conv(cudaHandles,manager,&desc->g1Conv,useFP16,useNHWC),
    g1BN(cudaHandles,&desc->g1BN,&desc->g1Activation,nnX,nnY,useFP16,useNHWC),
    gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,false),
    p1BN(cudaHandles,&desc->p1BN,&desc->p1Activation,nnX,nnY,false,useNHWC),
    p2Conv(cudaHandles,manager,&desc->p2Conv,false,useNHWC),
    gpoolToPassMul(cudaHandles,&desc->gpoolToPassMul,false),
    gpoolToPassBias(cudaHandles,&desc->gpoolToPassBias,false,desc->passActivation.activation),
    gpoolToPassMul2(cudaHandles,&desc->gpoolToPassMul2,false)
  {
  }

  ~PolicyHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = p1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = g1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul2.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*g1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    const half* wideHeadBuf,
    int wideHeadRowStride,
    int wideHeadP1Offset,
    int wideHeadG1Offset,
    float* policyPassBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> p1Out(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> p1Out2(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> g1Out(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Out2(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Concat(scratch->allocator, scratch->getBufSizeFloat(g1Channels*3));
    SizedBuf<void*> g1Bias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<void*> p1Pass(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    const bool useWideP1 = wideHeadBuf != NULL && wideHeadP1Offset >= 0;
    const bool useWideG1 = wideHeadBuf != NULL && wideHeadG1Offset >= 0;
    if(!useWideP1)
      p1Conv.apply(cudaHandles,batchSize,false,trunkBuf,p1Out.buf,workspaceBuf,workspaceBytes);
    if(!useWideG1)
      g1Conv.apply(cudaHandles,batchSize,false,trunkBuf,g1Out.buf,workspaceBuf,workspaceBytes);
    bool usedHeadBNHalfToFloat = false;
    if(usingFP16 && usingNHWC && maskBuf == NULL &&
       cudaHandles->sm120UseHeadBNHalfToFloat) {
      usedHeadBNHalfToFloat = Sm120Backend::launchHeadBNHalfToFloat(
        useWideG1 ? wideHeadBuf : (const half*)g1Out.buf,
        NULL,(float*)workspaceBuf,
        (const half*)g1BN.mergedScaleBuf,(const half*)g1BN.mergedBiasBuf,
        batchSize,nnXLen*nnYLen,g1Channels,
        useWideG1 ? wideHeadRowStride : g1Channels,
        useWideG1 ? wideHeadG1Offset : 0,
        cudaHandles->stream);
      if(usedHeadBNHalfToFloat && !cudaHandles->loggedSm120HeadBNHalfToFloat) {
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(
            "SM120 backend: head BN direct FP32 output active");
        cudaHandles->loggedSm120HeadBNHalfToFloat = true;
      }
    }
    if(useWideG1)
      testAssert(usedHeadBNHalfToFloat);
    if(!usedHeadBNHalfToFloat)
      g1BN.apply(cudaHandles,batchSize,g1Out.buf,maskBuf,g1Out2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf, cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf, cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      if(!usedHeadBNHalfToFloat) {
        customCudaCopyFromHalf((const half*)g1Out2.buf,(float*)workspaceBuf,batchSize*g1Channels*nnXLen*nnYLen, cudaHandles->stream);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,g1Channels,nnXLen*nnYLen,maskFloatBuf,maskSumBuf, cudaHandles->stream);
      else
        customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,nnXLen*nnYLen,g1Channels,maskFloatBuf,maskSumBuf, cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Bias.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("p1 pre-gpool-sum"), p1Out.buf, batchSize, p1Channels, nnXLen*nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint3D(string("g1 pre-gpool"), g1Out.buf, batchSize, g1Channels, nnXLen*nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("g1 pooled"), g1Concat.buf, batchSize, g1Channels*3, false);
    CudaUtils::debugPrint2D(string("g1 biases"), g1Bias.buf, batchSize, p1Channels, false);
    #endif

    float* p1OutBufA = NULL;
    float* p1OutBufB = NULL;
    bool usedFusedPolicyP1 = false;
    if(usingFP16 && usingNHWC && maskFloatBuf == NULL &&
       cudaHandles->sm120FusedPolicyP1 != NULL) {
      usedFusedPolicyP1 = cudaHandles->sm120FusedPolicyP1(
        cudaHandles->sm120FusedPolicyP1Context,
        useWideP1 ? wideHeadBuf : p1Out.buf,
        (float*)p1Out2.buf, (const float*)g1Bias.buf,
        (const float*)p1BN.mergedScaleBuf, (const float*)p1BN.mergedBiasBuf,
        batchSize, nnXLen * nnYLen, p1Channels,
        useWideP1 ? wideHeadRowStride : p1Channels,
        useWideP1 ? wideHeadP1Offset : 0, usingFP16, usingNHWC,
        cudaHandles->stream);
    }
    if(usedFusedPolicyP1) {
      p1OutBufB = (float*)p1Out2.buf;
    }
    else {
      testAssert(!useWideP1);
      if(!usingFP16) {
        p1OutBufA = (float*)p1Out.buf;
        p1OutBufB = (float*)p1Out2.buf;
      }
      else {
        customCudaCopyFromHalf((const half*)p1Out.buf,(float*)p1Out2.buf,batchSize*p1Channels*nnXLen*nnYLen, cudaHandles->stream);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        p1OutBufA = (float*)p1Out2.buf;
        p1OutBufB = (float*)p1Out.buf;
      }

      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW(p1OutBufA,(float*)g1Bias.buf,batchSize,p1Channels,nnXLen*nnYLen, cudaHandles->stream);
      else
        customCudaAddNCBiasInplaceNHWC(p1OutBufA,(float*)g1Bias.buf,batchSize,nnXLen*nnYLen,p1Channels, cudaHandles->stream);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());

      p1BN.apply(cudaHandles,batchSize,p1OutBufA,maskFloatBuf,p1OutBufB);
    }
    p2Conv.apply(cudaHandles,batchSize,false,p1OutBufB,(float*)policyBuf,workspaceBuf,workspaceBytes);

    if(modelVersion >= 15) {
      gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,p1Pass.buf,workspaceBuf,workspaceBytes);
      gpoolToPassBias.apply(cudaHandles,batchSize,p1Pass.buf);
      gpoolToPassMul2.apply(cudaHandles,scratch,batchSize,p1Pass.buf,policyPassBuf,workspaceBuf,workspaceBytes);
    }
    else {
      gpoolToPassMul.apply(cudaHandles,scratch,batchSize,g1Concat.buf,policyPassBuf,workspaceBuf,workspaceBytes);
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(
      string(usedFusedPolicyP1 ? "p1 after fused-gpool-bn" : "p1 after-gpool-sum"),
      usedFusedPolicyP1 ? p1OutBufB : p1OutBufA,
      batchSize, p1Channels, nnXLen*nnYLen, usingNHWC, false);
    CudaUtils::debugPrint2D(string("policypass"), policyPassBuf, batchSize, 1, false);
    CudaUtils::debugPrint3D(string("policy"), policyBuf, batchSize, p2Channels, nnXLen*nnYLen, usingNHWC, false);
    #endif

  }

};

//------------------------------------------------------------------------------

struct ValueHead {
  const string name;
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int v1Channels;
  const int v2Channels;
  const int valueChannels;
  const int scoreValueChannels;
  const int ownershipChannels;
  const bool usingFP16;
  const bool usingNHWC;

  const ConvLayer v1Conv;
  const BatchNormLayer v1BN;
  const MatMulLayer v2Mul;
  const MatBiasLayer v2Bias;
  const MatMulLayer v3Mul;
  const MatBiasLayer v3Bias;
  const MatMulLayer sv3Mul;
  const MatBiasLayer sv3Bias;
  const ConvLayer vOwnershipConv;
  bool fusedValueTerminalActive;
  void* fusedValueTerminalWeights;
  void* fusedValueTerminalBiases;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    nnXLen(nnX),
    nnYLen(nnY),
    v1Channels(desc->v1Conv.outChannels),
    v2Channels(desc->v2Mul.outChannels),
    valueChannels(desc->v3Mul.outChannels),
    scoreValueChannels(desc->sv3Mul.outChannels),
    ownershipChannels(desc->vOwnershipConv.outChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    v1Conv(cudaHandles,manager,&desc->v1Conv,useFP16,useNHWC),
    v1BN(cudaHandles,&desc->v1BN,&desc->v1Activation,nnX,nnY,useFP16,useNHWC),
    v2Mul(cudaHandles,&desc->v2Mul,false),
    v2Bias(cudaHandles,&desc->v2Bias,false,desc->v2Activation.activation),
    v3Mul(cudaHandles,&desc->v3Mul,false),
    v3Bias(cudaHandles,&desc->v3Bias,false,ACTIVATION_IDENTITY),
    sv3Mul(cudaHandles,&desc->sv3Mul,false),
    sv3Bias(cudaHandles,&desc->sv3Bias,false,ACTIVATION_IDENTITY),
    vOwnershipConv(cudaHandles,manager,&desc->vOwnershipConv,useFP16,useNHWC),
    fusedValueTerminalActive(false),
    fusedValueTerminalWeights(NULL),
    fusedValueTerminalBiases(NULL)
  {
    fusedValueTerminalActive =
      cudaHandles->sm120UseFusedValueTerminal && modelVersion >= 9 &&
      valueChannels == 3 && scoreValueChannels == 6 &&
      desc->v3Mul.inChannels == desc->sv3Mul.inChannels;
    if(fusedValueTerminalActive) {
      const int inputChannels = desc->v3Mul.inChannels;
      const int combinedChannels = valueChannels + scoreValueChannels;
      vector<float> weights((size_t)inputChannels * combinedChannels);
      for(int inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
        std::copy_n(
          desc->v3Mul.weights.begin() + (size_t)inputChannel * valueChannels,
          valueChannels,
          weights.begin() + (size_t)inputChannel * combinedChannels);
        std::copy_n(
          desc->sv3Mul.weights.begin() +
            (size_t)inputChannel * scoreValueChannels,
          scoreValueChannels,
          weights.begin() + (size_t)inputChannel * combinedChannels +
            valueChannels);
      }
      vector<float> biases;
      biases.reserve(combinedChannels);
      biases.insert(
        biases.end(),desc->v3Bias.weights.begin(),desc->v3Bias.weights.end());
      biases.insert(
        biases.end(),desc->sv3Bias.weights.begin(),desc->sv3Bias.weights.end());
      CudaUtils::mallocAndCopyToDevice(
        name + ":fusedTerminalWeights",weights,fusedValueTerminalWeights,false);
      CudaUtils::mallocAndCopyToDevice(
        name + ":fusedTerminalBiases",biases,fusedValueTerminalBiases,false);
    }
  }

  ~ValueHead()
  {
    if(fusedValueTerminalWeights != NULL)
      cudaFree(fusedValueTerminalWeights);
    if(fusedValueTerminalBiases != NULL)
      cudaFree(fusedValueTerminalBiases);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = v1Conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = v2Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*v1Channels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    b = sv3Mul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = vOwnershipConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*ownershipChannels*nnXLen*nnYLen;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    const half* wideHeadBuf,
    int wideHeadRowStride,
    int wideHeadV1Offset,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Out2(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<void*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));
    SizedBuf<void*> ownershipScratch(scratch->allocator, scratch->getBufSizeXYFloat(ownershipChannels));

    const bool useWideV1 = wideHeadBuf != NULL && wideHeadV1Offset >= 0;
    if(!useWideV1)
      v1Conv.apply(cudaHandles,batchSize,false,trunkBuf,v1Out.buf,workspaceBuf,workspaceBytes);
    bool usedHeadBNHalfToFloat = false;
    if(usingFP16 && usingNHWC && maskBuf == NULL &&
       cudaHandles->sm120UseHeadBNHalfToFloat) {
      usedHeadBNHalfToFloat = Sm120Backend::launchHeadBNHalfToFloat(
        useWideV1 ? wideHeadBuf : (const half*)v1Out.buf,
        (half*)v1Out2.buf,(float*)workspaceBuf,
        (const half*)v1BN.mergedScaleBuf,(const half*)v1BN.mergedBiasBuf,
        batchSize,nnXLen*nnYLen,v1Channels,
        useWideV1 ? wideHeadRowStride : v1Channels,
        useWideV1 ? wideHeadV1Offset : 0,
        cudaHandles->stream);
      if(usedHeadBNHalfToFloat && !cudaHandles->loggedSm120HeadBNHalfToFloat) {
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(
            "SM120 backend: head BN direct FP32 output active");
        cudaHandles->loggedSm120HeadBNHalfToFloat = true;
      }
    }
    if(useWideV1)
      testAssert(usedHeadBNHalfToFloat);
    if(!usedHeadBNHalfToFloat)
      v1BN.apply(cudaHandles,batchSize,v1Out.buf,maskBuf,v1Out2.buf);

    void* bufToBePooled = v1Out2.buf;
    if(usingFP16) {
      if(!usedHeadBNHalfToFloat) {
        customCudaCopyFromHalf((const half*)v1Out2.buf,(float*)workspaceBuf,batchSize*v1Channels*nnXLen*nnYLen, cudaHandles->stream);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC)
      customCudaValueHeadPoolNCHW((float*)bufToBePooled,(float*)v1Mean.buf,batchSize,v1Channels,nnXLen*nnYLen,maskSumBuf, cudaHandles->stream);
    else
      customCudaValueHeadPoolNHWC((const float*)bufToBePooled,(float*)v1Mean.buf,batchSize,nnXLen*nnYLen,v1Channels,maskSumBuf, cudaHandles->stream);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    v2Mul.apply(cudaHandles,scratch,batchSize,v1Mean.buf,v2Out.buf,workspaceBuf,workspaceBytes);
    v2Bias.apply(cudaHandles,batchSize,v2Out.buf);
    bool usedFusedTerminal = false;
    if(fusedValueTerminalActive) {
      const int combinedChannels = valueChannels + scoreValueChannels;
      SizedBuf<void*> combined(
        scratch->allocator,scratch->getBufSizeFloat(combinedChannels));
      const float alpha = 1.0f;
      const float beta = 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,CUBLAS_OP_N,CUBLAS_OP_N,
        combinedChannels,batchSize,v2Channels,
        &alpha,(const float*)fusedValueTerminalWeights,combinedChannels,
        (const float*)v2Out.buf,v2Channels,
        &beta,(float*)combined.buf,combinedChannels));
      usedFusedTerminal = Sm120Backend::launchSplitValueTerminal(
        (const float*)combined.buf,(const float*)fusedValueTerminalBiases,
        valueBuf,scoreValueBuf,batchSize,valueChannels,scoreValueChannels,
        cudaHandles->stream);
      if(usedFusedTerminal && !cudaHandles->loggedSm120FusedValueTerminal) {
        if(cudaHandles->logger != NULL)
          cudaHandles->logger->write(
            "SM120 backend: fused value/score terminal active");
        cudaHandles->loggedSm120FusedValueTerminal = true;
      }
    }
    if(!usedFusedTerminal) {
      v3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,valueBuf,workspaceBuf,workspaceBytes);
      v3Bias.apply(cudaHandles,batchSize,valueBuf);

      sv3Mul.apply(cudaHandles,scratch,batchSize,v2Out.buf,scoreValueBuf,workspaceBuf,workspaceBytes);
      sv3Bias.apply(cudaHandles,batchSize,scoreValueBuf);
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("v1"), v1Out.buf, batchSize, v1Channels, nnXLen*nnYLen, usingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("v1 pooled"), v1Mean.buf, batchSize, v1Channels, false);
    CudaUtils::debugPrint2D(string("v2"), v2Out.buf, batchSize, v1Channels, false);
    #endif

    if(!usingFP16) {
      vOwnershipConv.apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipBuf,workspaceBuf,workspaceBytes);
    }
    else {
      vOwnershipConv.apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipScratch.buf,workspaceBuf,workspaceBytes);
      customCudaCopyFromHalf((const half*)ownershipScratch.buf,(float*)ownershipBuf,batchSize*ownershipChannels*nnXLen*nnYLen, cudaHandles->stream);
      CUDA_ERR("vOwnership copy",cudaPeekAtLastError());
    }

  }

};

//------------------------------------------------------------------------------

struct Model {
  const string name;
  const int modelVersion;
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const int numInputChannels;
  const int numInputGlobalChannels;
  const int numInputMetaChannels;
  const int numPolicyChannels;
  const int numValueChannels;
  const int numScoreValueChannels;
  const int numOwnershipChannels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool inputsUsingNHWC;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::unique_ptr<CudnnManager> manager;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) :
    name(desc->name),
    modelVersion(desc->modelVersion),
    maxBatchSize(maxBatchSz),
    nnXLen(nnX),
    nnYLen(nnY),
    numInputChannels(desc->numInputChannels),
    numInputGlobalChannels(desc->numInputGlobalChannels),
    numInputMetaChannels(desc->numInputMetaChannels),
    numPolicyChannels(desc->numPolicyChannels),
    numValueChannels(desc->numValueChannels),
    numScoreValueChannels(desc->numScoreValueChannels),
    numOwnershipChannels(desc->numOwnershipChannels),
    usingFP16(useFP16),
    usingNHWC(useNHWC),
    inputsUsingNHWC(inputsUseNHWC)
  {
    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnXLen, NNPos::MAX_BOARD_LEN
      ));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnYLen, NNPos::MAX_BOARD_LEN
      ));

    int numFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf("Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels, numFeatures
      ));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf("Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels, numGlobalFeatures
      ));
    if(numInputMetaChannels > 0) {
      if(numInputMetaChannels != SGFMetadata::METADATA_INPUT_NUM_CHANNELS)
        throw StringError(Global::strprintf("Neural net numInputMetaChannels (%d) was not the expected number (%d)",
          numInputMetaChannels, SGFMetadata::METADATA_INPUT_NUM_CHANNELS
        ));
    }

    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputMetaChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numPolicyChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    manager = std::make_unique<CudnnManager>(name, maxBatchSize, nnXLen, nnYLen);
    trunk = std::make_unique<Trunk>(cudaHandles,manager.get(),&desc->trunk,nnXLen,nnYLen,inputsUseNHWC,useFP16,useNHWC);
    policyHead = std::make_unique<PolicyHead>(cudaHandles,manager.get(),&desc->policyHead,nnXLen,nnYLen,useFP16,useNHWC);
    valueHead = std::make_unique<ValueHead>(cudaHandles,manager.get(),&desc->valueHead,nnXLen,nnYLen,useFP16,useNHWC);
  }

  ~Model()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = trunk->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool requireExactNNLen,

    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,

    float* policyPassBuf,
    float* policyBuf,

    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    if(cudaHandles->sm120SharedModelWeightsActive &&
       !cudaHandles->loggedSm120SharedModelWeights) {
      if(cudaHandles->logger != NULL)
        cudaHandles->logger->write(
          "SM120 backend: per-device model-weight sharing active");
      cudaHandles->loggedSm120SharedModelWeights = true;
    }
    SizedBuf<void*> mask(scratch->allocator, scratch->getBufSizeXY(1));
    SizedBuf<void*> maskFloat(scratch->allocator, scratch->getBufSizeXYFloat(1));
    SizedBuf<void*> maskSum(scratch->allocator, scratch->getBufSizeFloat(1));

    void* maskBuf = mask.buf;
    float* maskFloatBuf = (float*)maskFloat.buf;
    float* maskSumBuf = (float*)maskSum.buf;

    const bool useSm120FullBoardContract =
      batchSize <= cudaHandles->sm120FullBoardCapacity &&
      cudaHandles->sm120FullBoardAreaBuf != NULL;
    if(useSm120FullBoardContract) {
      maskBuf = NULL;
      maskFloatBuf = NULL;
      maskSumBuf = cudaHandles->sm120FullBoardAreaBuf;
    }
    else if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels, cudaHandles->stream);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen, cudaHandles->stream);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, nnXLen*nnYLen, numInputChannels, cudaHandles->stream);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, nnXLen*nnYLen, cudaHandles->stream);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    if(!useSm120FullBoardContract)
      fillMaskFloatBufAndMaskSumBuf(cudaHandles,maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,nnXLen,nnYLen);

    //Don't do any masking if we know the board is exactly the desired size
    if(requireExactNNLen) {
      //Set to NULL to signal downstream that this buf doesn't need to be used
      maskBuf = NULL;
      maskFloatBuf = NULL;
      //The global pooling structures need this no matter what, for normalizing based on this and its sqrt.
      //maskSumBuf = NULL;
    }

    #ifdef DEBUG_INTERMEDIATE_VALUES
    CudaUtils::debugPrint3D(string("Initial bin features"), inputBuf, batchSize, trunk->initialConv->inChannels, nnXLen*nnYLen, inputsUsingNHWC, usingFP16);
    CudaUtils::debugPrint2D(string("Initial global features"), inputGlobalBuf, batchSize, trunk->initialMatMul->inChannels, usingFP16);
    if(trunk->sgfMetadataEncoder != nullptr) {
      assert(inputMetaBuf != NULL);
      CudaUtils::debugPrint2D(string("Initial meta features"), inputMetaBuf, batchSize, trunk->sgfMetadataEncoder->mul1.inChannels, usingFP16);
    }
    #endif

    SizedBuf<void*> trunkBuf(scratch->allocator, scratch->getBufSizeXY(trunk->trunkNumChannels));

    trunk->apply(
      cudaHandles,
      scratch,
      batchSize,
      inputBuf,
      inputGlobalBuf,
      inputMetaBuf,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      workspaceBuf,
      workspaceBytes
    );
    std::unique_ptr<SizedBuf<void*>> wideHead;
    const half* wideHeadBuf = NULL;
    int wideHeadRowStride = 0;
    int wideHeadP1Offset = -1;
    int wideHeadG1Offset = -1;
    int wideHeadV1Offset = -1;
    if(cudaHandles->sm120WideHeadProjection != NULL) {
      wideHead = std::make_unique<SizedBuf<void*>>(
        scratch->allocator, scratch->getBufSizeXY(384));
      bool usedWideHead = cudaHandles->sm120WideHeadProjection(
        cudaHandles->sm120WideHeadProjectionContext,
        trunkBuf.buf, wideHead->buf, batchSize, nnXLen * nnYLen,
        trunk->trunkNumChannels, 384, usingFP16, usingNHWC,
        &wideHeadRowStride, &wideHeadP1Offset, &wideHeadG1Offset,
        &wideHeadV1Offset,
        cudaHandles->stream);
      if(usedWideHead)
        wideHeadBuf = (const half*)wideHead->buf;
    }
    policyHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskFloatBuf,
      maskSumBuf,
      trunkBuf.buf,
      wideHeadBuf,
      wideHeadRowStride,
      wideHeadP1Offset,
      wideHeadG1Offset,
      policyPassBuf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      wideHeadBuf,
      wideHeadRowStride,
      wideHeadV1Offset,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};

// Trampoline used by Sm120Backend::Sm120Model to delegate to the official backend apply().
// Keeps the internal Model type opaque to cudabackend_sm120.cpp.
static void applyOfficialModel(
  void* ctx,
  CudaHandles* cudaHandles_,
  ScratchBuffers* scratch,
  int batchSize,
  bool requireExactNNLen,
  void* inputBuf,
  void* inputGlobalBuf,
  void* inputMetaBuf,
  float* policyPassBuf,
  float* policyBuf,
  float* valueBuf,
  float* scoreValueBuf,
  void* ownershipBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) {
  Model* model = (Model*)ctx;
  model->apply(
    cudaHandles_, scratch, batchSize, requireExactNNLen,
    inputBuf, inputGlobalBuf, inputMetaBuf,
    policyPassBuf, policyBuf,
    valueBuf, scoreValueBuf, ownershipBuf,
    workspaceBuf, workspaceBytes
  );
}


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file,expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufFloat;
  void* inputBuf;
  float* inputGlobalBufFloat;
  void* inputGlobalBuf;
  float* inputMetaBufFloat;
  void* inputMetaBuf;
  size_t inputBufBytesFloat;
  size_t inputBufBytes;
  size_t inputGlobalBufBytesFloat;
  size_t inputGlobalBufBytes;
  size_t inputMetaBufBytesFloat;
  size_t inputMetaBufBytes;

  float* policyPassBuf;
  size_t policyPassBufBytes;
  float* policyBuf;
  size_t policyBufBytes;

  float* valueBuf;
  size_t valueBufBytes;
  float* scoreValueBuf;
  size_t scoreValueBufBytes;
  void* ownershipBuf;
  size_t ownershipBufBytes;

  void* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m, const ScratchBuffers& scratch) {
    size_t batchXYFloatBytes = (size_t)scratch.batchXYFloatBytes;
    size_t batchFloatBytes = (size_t)scratch.batchFloatBytes;
    size_t batchXYBytes = (size_t)scratch.batchXYBytes;
    size_t batchBytes = (size_t)scratch.batchBytes;

    inputBufBytesFloat = m.numInputChannels * batchXYFloatBytes;
    inputBufBytes = m.numInputChannels * batchXYBytes;
    inputGlobalBufBytesFloat = m.numInputGlobalChannels * batchFloatBytes;
    inputGlobalBufBytes = m.numInputGlobalChannels * batchBytes;
    inputMetaBufBytesFloat = m.numInputMetaChannels * batchFloatBytes;
    inputMetaBufBytes = m.numInputMetaChannels * batchBytes;

    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputBufFloat), inputBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputGlobalBufFloat), inputGlobalBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBuf, inputGlobalBufBytes));
    if(m.numInputMetaChannels > 0) {
      CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&inputMetaBufFloat), inputMetaBufBytesFloat));
      CUDA_ERR("Buffers",cudaMalloc(&inputMetaBuf, inputMetaBufBytes));
    }
    else {
      inputMetaBufFloat = NULL;
      inputMetaBuf = NULL;
    }

    if(m.modelVersion >= 17)
      testAssert(m.policyHead->p2Channels == 2 || m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 16)
      testAssert(m.policyHead->p2Channels == 4);
    else if(m.modelVersion >= 12)
      testAssert(m.policyHead->p2Channels == 2);
    else
      testAssert(m.policyHead->p2Channels == 1);

    policyPassBufBytes = m.policyHead->p2Channels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyPassBuf), policyPassBufBytes));
    policyBufBytes = m.policyHead->p2Channels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&policyBuf), policyBufBytes));

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&valueBuf), valueBufBytes));

    scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(reinterpret_cast<void**>(&scoreValueBuf), scoreValueBufBytes));

    //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
    ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));

    //In theory the requiredWorkspaceBytes calls could give us values non-monotone in batch size
    //such as if the convolution algorithm changes between batch size 1 and larger.
    //So we call it for all the batch sizes.
    size_t bytes = 0;
    size_t b;
    for(int batchSize = 1; batchSize <= m.maxBatchSize; batchSize++) {
      b = m.requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    CUDA_ERR("Buffers",cudaMalloc(&workspaceBuf, bytes));
    workspaceBytes = bytes;
  }

  ~Buffers() {
    cudaFree(inputBufFloat);
    cudaFree(inputBuf);
    cudaFree(inputGlobalBufFloat);
    cudaFree(inputGlobalBuf);
    if(inputMetaBufFloat != NULL)
      cudaFree(inputMetaBufFloat);
    if(inputMetaBuf != NULL)
      cudaFree(inputMetaBuf);

    cudaFree(policyPassBuf);
    cudaFree(policyBuf);

    cudaFree(valueBuf);
    cudaFree(scoreValueBuf);
    cudaFree(ownershipBuf);

    cudaFree(workspaceBuf);
  }

};

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
  // If true, skip the cudnn graph SDPA path entirely and always use the custom attention kernel.
  bool cudaDisableGraphSDPA;
  bool cudaEventPipelineUseGraph;
  // Whether 1x1 NHWC convs use the cuBLAS GEMM path. Auto = matmul iff FP16.
  enabled_t use1x1MatmulMode;
  // SM89-specific backend options; only used when a server thread is on a SM89 device.
  Sm89Backend::Options sm89Options;
  // SM120-specific backend options; only used when a server thread is on a SM120 device.
  Sm120Backend::Options sm120Options;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)logger;
  (void)homeDataDirOverride;
  (void)loadedModel;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;

  // NHWC layout is a CUDA-specific option read directly off of cfg. Auto means "NHWC if FP16" (see below).
  context->useNHWCMode =
    cfg.contains("cudaUseNHWC") ? cfg.getEnabled("cudaUseNHWC") : enabled_t::Auto;
  context->cudaDisableGraphSDPA =
    cfg.contains("cudaDisableGraphSDPA") ? cfg.getBool("cudaDisableGraphSDPA") : false;
  context->cudaEventPipelineUseGraph =
    cfg.contains("cudaEventPipelineUseGraph") ? cfg.getBool("cudaEventPipelineUseGraph") : false;
  context->use1x1MatmulMode =
    cfg.contains("cudaUse1x1Matmul") ? cfg.getEnabled("cudaUse1x1Matmul") : enabled_t::Auto;
  context->sm89Options = Sm89Backend::parseOptions(cfg);
  context->sm120Options = Sm120Backend::parseOptions(cfg);
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct CudaComputeStream {
  int gpuIdx;
  cudaStream_t stream;
};

struct ComputeHandle {
  std::unique_ptr<CudaHandles> cudaHandles;
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  const int gpuIdx;
  const bool usingFP16;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const bool inputsUseNHWC;
  const bool usingNHWC;
  const bool eventPipelineUseGraph;
  const int eventPipelineBatchSize;
  bool eventPipelineEnabled;
  cudaStream_t uploadStream;
  cudaStream_t downloadStream;
  cudaEvent_t inputReadyEvent;
  cudaEvent_t inputConsumedEvent;
  cudaEvent_t applyCompleteEvent;
  cudaEvent_t outputConsumedEvent;
  cudaEvent_t outputReadyEvent;
  cudaGraph_t eventPipelineGraph;
  cudaGraphExec_t eventPipelineGraphExec;
  // Set only on SM120 devices; routes apply() to the SM120-specific path.
  std::unique_ptr<Sm120Backend::Sm120Model> sm120Model;
  // Set only on SM89 devices; routes apply() to the SM89-specific path.
  std::unique_ptr<Sm89Backend::Sm89Model> sm89Model;

  ComputeHandle(
    const ComputeContext* context,
    const LoadedModel* loadedModel,
    int gpuIdx_,
    int majorComputeCapability,
    int minorComputeCapability,
    int maxBatchSize,
    bool requireExactNNLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC,
    cudaStream_t stream
  ) :
    gpuIdx(gpuIdx_),
    usingFP16(useFP16),
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    requireExactNNLen(requireExactNNLen_),
    inputsUseNHWC(inputsUseNHWC_),
    usingNHWC(useNHWC),
    eventPipelineUseGraph(context->cudaEventPipelineUseGraph),
    eventPipelineBatchSize(maxBatchSize),
    eventPipelineEnabled(false),
    uploadStream(NULL),
    downloadStream(NULL),
    inputReadyEvent(NULL),
    inputConsumedEvent(NULL),
    applyCompleteEvent(NULL),
    outputConsumedEvent(NULL),
    outputReadyEvent(NULL),
    eventPipelineGraph(NULL),
    eventPipelineGraphExec(NULL)
  {
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability,stream);
    // Must be set before building the model: ConvLayer reads it at construction to pick the 1x1 conv path.
    cudaHandles->use1x1MatmulMode = context->use1x1MatmulMode;
    cudaHandles->sm120InitialConvFrontendEngine = 0;
    cudaHandles->sm120ShareModelWeights =
      Sm120Backend::isSm120Arch(majorComputeCapability, minorComputeCapability) &&
      context->sm120Options.enabled && context->sm120Options.shareModelWeights;
    cudaHandles->sm120UseFusedValueTerminal =
      Sm120Backend::isSm120Arch(majorComputeCapability, minorComputeCapability) &&
      context->sm120Options.enabled &&
      context->sm120Options.useFusedValueTerminal;
    cudaHandles->sm120UseHeadBNHalfToFloat =
      Sm120Backend::isSm120Arch(majorComputeCapability, minorComputeCapability) &&
      context->sm120Options.enabled && context->sm120Options.useHeadBNHalfToFloat;
    if(Sm120Backend::isSm120Arch(majorComputeCapability, minorComputeCapability) &&
       context->sm120Options.enabled) {
      if(context->sm120Options.initialConvFrontendPlan == "eng45-tile0-stages2")
        cudaHandles->sm120InitialConvFrontendEngine = 45;
      else if(context->sm120Options.initialConvFrontendPlan ==
              "eng47-k2-2-k6-1-k13-1-k14-0-k22-2")
        cudaHandles->sm120InitialConvFrontendEngine = 47;
    }
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      nnXLen, nnYLen, inputsUseNHWC, useFP16, useNHWC
    );
    scratch = std::make_unique<ScratchBuffers>(maxBatchSize, nnXLen, nnYLen, useFP16);
    buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, *scratch);

    if(Sm120Backend::isSm120Arch(majorComputeCapability, minorComputeCapability) &&
       context->sm120Options.enabled) {
      sm120Model = std::make_unique<Sm120Backend::Sm120Model>(
        model.get(), &applyOfficialModel, cudaHandles.get(), &(loadedModel->modelDesc),
        maxBatchSize, nnXLen, nnYLen, inputsUseNHWC_, useFP16, useNHWC,
        context->sm120Options
      );
      cudaHandles->sm120FullBoardAreaBuf = sm120Model->getFullBoardAreaBuf();
      if(cudaHandles->sm120FullBoardAreaBuf != NULL)
        cudaHandles->sm120FullBoardCapacity = maxBatchSize;
      if(sm120Model->hasPersistingL2Trunk()) {
        cudaHandles->sm120PersistingL2 = &Sm120Backend::applyPersistingL2Window;
        cudaHandles->sm120PersistingL2Context = sm120Model.get();
      }
      if(sm120Model->hasPersistingL2Inner()) {
        cudaHandles->sm120PersistingL2Inner = &Sm120Backend::applyPersistingL2Window;
        cudaHandles->sm120PersistingL2InnerContext = sm120Model.get();
      }
      cudaHandles->sm120Attention = &Sm120Backend::applyAttention;
      cudaHandles->sm120AttentionContext = sm120Model.get();
      if(context->sm120Options.useFusedFFN || context->sm120Options.useWideFFNSingleGemm) {
        cudaHandles->sm120FFNSingleGemm = &Sm120Backend::applyFFNSingleGemm;
        cudaHandles->sm120FFNSingleGemmContext = sm120Model.get();
      }
      if(context->sm120Options.useProjectionGemmLt) {
        cudaHandles->sm120MatMul = &Sm120Backend::applyMatMulLt;
        cudaHandles->sm120MatMulContext = sm120Model.get();
      }
      if(context->sm120Options.outerProjectionDownTactic != "disabled" ||
         context->sm120Options.outerProjectionUpTactic != "disabled") {
        cudaHandles->sm120Conv1x1 = &Sm120Backend::applyConv1x1;
        cudaHandles->sm120Conv1x1Context = sm120Model.get();
      }
      if(context->sm120Options.useInitialGlobalMatMulAdd) {
        cudaHandles->sm120InitialGlobal = &Sm120Backend::applyInitialGlobal;
        cudaHandles->sm120InitialGlobalContext = sm120Model.get();
      }
      if(context->sm120Options.useQKVStrided ||
         (context->sm120Options.useWideQKV && context->sm120Options.useQKVGemmAot)) {
        cudaHandles->sm120QKVStrided = &Sm120Backend::applyQKVStrided;
        cudaHandles->sm120QKVStridedContext = sm120Model.get();
      }
      if(context->sm120Options.useFusedResidualGemm) {
        cudaHandles->sm120FusedResidualGemm = &Sm120Backend::applyFusedResidualGemm;
        cudaHandles->sm120FusedResidualGemmContext = sm120Model.get();
      }
      if(context->sm120Options.rmsNorm384Tactic != "disabled") {
        cudaHandles->sm120RMSNorm = &Sm120Backend::applyRMSNorm;
        cudaHandles->sm120RMSNormContext = sm120Model.get();
      }
      if(context->sm120Options.useFusedQKRoPE) {
        cudaHandles->sm120FusedQKRoPE = &Sm120Backend::applyFusedQKRoPE;
        cudaHandles->sm120FusedQKRoPEContext = sm120Model.get();
      }
      if(context->sm120Options.useSwiGLU1152) {
        cudaHandles->sm120SwiGLU = &Sm120Backend::applySwiGLU;
        cudaHandles->sm120SwiGLUContext = sm120Model.get();
      }
      if(context->sm120Options.affineSiluTactic != "disabled") {
        cudaHandles->sm120AffineSilu = &Sm120Backend::applyAffineSilu;
        cudaHandles->sm120AffineSiluContext = sm120Model.get();
      }
      if(context->sm120Options.usePostConvBNSilu) {
        cudaHandles->sm120PostConvBNSilu = &Sm120Backend::applyPostConvBNSilu;
        cudaHandles->sm120PostConvBNSiluContext = sm120Model.get();
      }
      if(context->sm120Options.useFusedPolicyP1) {
        cudaHandles->sm120FusedPolicyP1 = &Sm120Backend::applyFusedPolicyP1;
        cudaHandles->sm120FusedPolicyP1Context = sm120Model.get();
      }
      if(context->sm120Options.wideHeadProjectionTactic != "disabled" &&
         context->sm120Options.useFusedPolicyP1 &&
         context->sm120Options.useHeadBNHalfToFloat) {
        cudaHandles->sm120WideHeadProjection =
          &Sm120Backend::applyWideHeadProjection;
        cudaHandles->sm120WideHeadProjectionContext = sm120Model.get();
      }
    }
    if(Sm89Backend::isSm89Arch(majorComputeCapability, minorComputeCapability) &&
       context->sm89Options.enabled) {
      sm89Model = std::make_unique<Sm89Backend::Sm89Model>(
        model.get(), &applyOfficialModel, cudaHandles.get(), &(loadedModel->modelDesc),
        maxBatchSize, nnXLen, nnYLen, inputsUseNHWC_, useFP16, useNHWC,
        cudaHandles->stream,
        context->sm89Options
      );
    }

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaStreamSynchronize(cudaHandles->stream));
  }
  ~ComputeHandle() {
    // All following members own device resources, so destroy them with their
    // device current even when the scheduler thread last touched another GPU.
    (void)cudaSetDevice(gpuIdx);
    if(eventPipelineGraphExec != NULL) {
      cudaStreamSynchronize(cudaHandles->stream);
      cudaGraphExecDestroy(eventPipelineGraphExec);
    }
    if(eventPipelineGraph != NULL)
      cudaGraphDestroy(eventPipelineGraph);
    if(eventPipelineEnabled) {
      cudaEventSynchronize(outputReadyEvent);
      cudaEventDestroy(outputReadyEvent);
      cudaEventDestroy(outputConsumedEvent);
      cudaEventDestroy(applyCompleteEvent);
      cudaEventDestroy(inputConsumedEvent);
      cudaEventDestroy(inputReadyEvent);
      cudaStreamDestroy(downloadStream);
      cudaStreamDestroy(uploadStream);
    }
  }

  void makeCurrent(const char* opName) const {
    int currentGpuIdx;
    CUDA_ERR(opName,cudaGetDevice(&currentGpuIdx));
    if(currentGpuIdx != gpuIdx)
      CUDA_ERR(opName,cudaSetDevice(gpuIdx));
  }

  void enableEventPipeline() {
    testAssert(!eventPipelineEnabled);
    makeCurrent("enableEventPipeline");
    try {
      CUDA_ERR("enableEventPipeline",cudaStreamCreateWithFlags(&uploadStream,cudaStreamNonBlocking));
      CUDA_ERR("enableEventPipeline",cudaStreamCreateWithFlags(&downloadStream,cudaStreamNonBlocking));
      CUDA_ERR("enableEventPipeline",cudaEventCreateWithFlags(&inputReadyEvent,cudaEventDisableTiming));
      CUDA_ERR("enableEventPipeline",cudaEventCreateWithFlags(&inputConsumedEvent,cudaEventDisableTiming));
      CUDA_ERR("enableEventPipeline",cudaEventCreateWithFlags(&applyCompleteEvent,cudaEventDisableTiming));
      CUDA_ERR("enableEventPipeline",cudaEventCreateWithFlags(&outputConsumedEvent,cudaEventDisableTiming));
      CUDA_ERR("enableEventPipeline",cudaEventCreateWithFlags(&outputReadyEvent,cudaEventDisableTiming));
      CUDA_ERR("enableEventPipeline",cudaEventRecord(inputConsumedEvent,cudaHandles->stream));
      CUDA_ERR("enableEventPipeline",cudaEventRecord(outputConsumedEvent,cudaHandles->stream));
      CUDA_ERR("enableEventPipeline",cudaEventRecord(outputReadyEvent,cudaHandles->stream));
      CUDA_ERR("enableEventPipeline",cudaStreamSynchronize(cudaHandles->stream));
    }
    catch(...) {
      if(outputReadyEvent != NULL) cudaEventDestroy(outputReadyEvent);
      if(outputConsumedEvent != NULL) cudaEventDestroy(outputConsumedEvent);
      if(applyCompleteEvent != NULL) cudaEventDestroy(applyCompleteEvent);
      if(inputConsumedEvent != NULL) cudaEventDestroy(inputConsumedEvent);
      if(inputReadyEvent != NULL) cudaEventDestroy(inputReadyEvent);
      if(downloadStream != NULL) cudaStreamDestroy(downloadStream);
      if(uploadStream != NULL) cudaStreamDestroy(uploadStream);
      outputReadyEvent = NULL;
      outputConsumedEvent = NULL;
      applyCompleteEvent = NULL;
      inputConsumedEvent = NULL;
      inputReadyEvent = NULL;
      downloadStream = NULL;
      uploadStream = NULL;
      throw;
    }
    eventPipelineEnabled = true;
  }

  void apply(
    CudaHandles* cudaHandles_,
    ScratchBuffers* scratch_,
    int batchSize,
    bool requireExactNNLen_,

    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,

    float* policyPassBuf,
    float* policyBuf,

    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,

    void* workspaceBuf,
    size_t workspaceBytes,
    cudaEvent_t inputConsumedEvent_ = nullptr,
    cudaEvent_t outputConsumedEvent_ = nullptr
  ) const {
    if(sm120Model != nullptr) {
      if(outputConsumedEvent_ != nullptr)
        CUDA_ERR("ComputeHandle::apply",cudaStreamWaitEvent(cudaHandles_->stream,outputConsumedEvent_,0));
      sm120Model->apply(
        cudaHandles_, scratch_, batchSize, requireExactNNLen_,
        inputBuf, inputGlobalBuf, inputMetaBuf,
        policyPassBuf, policyBuf,
        valueBuf, scoreValueBuf, ownershipBuf,
        workspaceBuf, workspaceBytes
      );
      if(inputConsumedEvent_ != nullptr)
        CUDA_ERR("ComputeHandle::apply",cudaEventRecord(inputConsumedEvent_,cudaHandles_->stream));
    }
    else if(sm89Model != nullptr) {
      sm89Model->apply(
        cudaHandles_, scratch_, batchSize, requireExactNNLen_,
        inputBuf, inputGlobalBuf, inputMetaBuf,
        policyPassBuf, policyBuf,
        valueBuf, scoreValueBuf, ownershipBuf,
        workspaceBuf, workspaceBytes,
        inputConsumedEvent_,outputConsumedEvent_
      );
    }
    else {
      if(outputConsumedEvent_ != nullptr)
        CUDA_ERR("ComputeHandle::apply",cudaStreamWaitEvent(cudaHandles_->stream,outputConsumedEvent_,0));
      model->apply(
        cudaHandles_, scratch_, batchSize, requireExactNNLen_,
        inputBuf, inputGlobalBuf, inputMetaBuf,
        policyPassBuf, policyBuf,
        valueBuf, scoreValueBuf, ownershipBuf,
        workspaceBuf, workspaceBytes
      );
      if(inputConsumedEvent_ != nullptr)
        CUDA_ERR("ComputeHandle::apply",cudaEventRecord(inputConsumedEvent_,cudaHandles_->stream));
    }
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

void* NeuralNet::createComputeStream(int gpuIdxForThisThread) {
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;
  CUDA_ERR("createComputeStream",cudaSetDevice(gpuIdxForThisThread));
  std::unique_ptr<CudaComputeStream> computeStream = std::make_unique<CudaComputeStream>();
  computeStream->gpuIdx = gpuIdxForThisThread;
  computeStream->stream = NULL;
  CUDA_ERR("createComputeStream",cudaStreamCreateWithFlags(
    &computeStream->stream,cudaStreamNonBlocking
  ));
  return computeStream.release();
}

void NeuralNet::freeComputeStream(void* computeStream) {
  if(computeStream == NULL)
    throw StringError("freeComputeStream: null CUDA stream");
  std::unique_ptr<CudaComputeStream> ownedStream(
    reinterpret_cast<CudaComputeStream*>(computeStream)
  );
  CUDA_ERR("freeComputeStream",cudaSetDevice(ownedStream->gpuIdx));
  CUDA_ERR("freeComputeStream",cudaStreamDestroy(ownedStream->stream));
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx,
  void* computeStream
) {
  //Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;

  CUDA_ERR("createComputeHandle",cudaSetDevice(gpuIdxForThisThread));

  if(computeStream == NULL)
    throw StringError("CUDA backend requires an externally owned compute stream");
  CudaComputeStream* ownedStream = reinterpret_cast<CudaComputeStream*>(computeStream);
  if(ownedStream->gpuIdx != gpuIdxForThisThread)
    throw StringError("CUDA backend compute stream device does not match compute handle device");
  cudaStream_t stream = ownedStream->stream;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,gpuIdxForThisThread);

  bool useFP16 = false;
  bool useNHWC = false;
  //Old GPUs - use FP32 and explicitly fail if FP16 enabled
  if(prop.major < 5 || (prop.major == 5 && prop.minor < 3)) {
    if(context->useFP16Mode == enabled_t::True)
      throw StringError("Cuda device versions below 5.3 do not support useFP16=true");
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //In theory these GPUs support FP16, so allow if the user wants.
  else if(prop.major < 6) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Pascal architecture, default to using FP16 operations
  //Actually, just use FP32 - there's a risk that on certain cards this might just be a lot worse.
  //A user manually fine-tuning for performance can just enable it themselves if they know how.
  else if(prop.major < 7) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Volta and higher, use FP16 and NHWC together because we have tensor cores.
  else {
    if(context->useFP16Mode == enabled_t::True || context->useFP16Mode == enabled_t::Auto)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True || (context->useNHWCMode == enabled_t::Auto && useFP16))
      useNHWC = true;
  }

  //The CUDA transformer block implementation only supports NHWC (its channel projections, RoPE, and
  //attention all assume the channel dim is contiguous per position). Unlike convnets, NHWC here is not
  //tied to FP16/tensor-cores - the transformer kernels have FP32 paths too. So for transformer models
  //force NHWC regardless of the FP16/NHWC-mode decision above, otherwise FP32 (or NHWC=false) would hit
  //the "NCHW layout not supported" throw. No effect on convnets.
  if(!useNHWC && loadedModel->modelDesc.trunk.hasAnyTransformerBlocks()) {
    if(context->useNHWCMode == enabled_t::False)
      throw StringError("CUDA backend: transformer models require NHWC, but cudaUseNHWC=false was set");
    useNHWC = true;
  }

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion) +
      " useFP16 = " + Global::boolToString(useFP16) +
      " useNHWC = " + Global::boolToString(useNHWC)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name +
      " (" + loadedModel->modelDesc.getShortInfoString() + ")"
    );
  }

  ComputeHandle* gpuHandle = new ComputeHandle(
    context,loadedModel,gpuIdxForThisThread,prop.major,prop.minor,maxBatchSize,
    requireExactNNLen,inputsUseNHWC,useFP16,useNHWC,stream
  );
  gpuHandle->cudaHandles->logger = logger;
  gpuHandle->cudaHandles->cudaDisableGraphSDPA = context->cudaDisableGraphSDPA;
  if(gpuHandle->sm120Model != nullptr)
    gpuHandle->sm120Model->setLogger(logger);
  if(gpuHandle->sm89Model != nullptr)
    gpuHandle->sm89Model->setLogger(logger);
  return gpuHandle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return handle->usingFP16;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  CudaHandles* cudaHandles = handle->cudaHandles.get();
  bool prev = cudaHandles->isWarmup;
  cudaHandles->isWarmup = isWarmup;
  return prev;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  for(int i = 0; i<numDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Found CUDA device " << i << ": " << prop.name << endl;
  }
}


//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singleInputMetaElts;
  size_t singleInputMetaBytes;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t userInputMetaBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer
  float* userInputMetaBuffer; //Host pointer

  float* policyPassResults; //Host pointer
  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer
  bool usingPinnedMemory;
  half* eventInputBuffer;
  half* eventInputGlobalBuffer;
  half* eventInputMetaBuffer;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputBytes = (size_t)m.numInputChannels * nnXLen * nnYLen * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singleInputMetaBytes = (size_t)m.numInputMetaChannels * sizeof(float);
    singlePolicyPassResultElts = (size_t)(m.numPolicyChannels);
    singlePolicyPassResultBytes = (size_t)(m.numPolicyChannels) * sizeof(float);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singlePolicyResultBytes = (size_t)(m.numPolicyChannels * nnXLen * nnYLen) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * nnXLen * nnYLen * sizeof(float);

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    userInputMetaBufferBytes = (size_t)m.numInputMetaChannels * maxBatchSize * sizeof(float);
    policyPassResultBufferBytes = (size_t)maxBatchSize * m.numPolicyChannels * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];
    if(m.numInputMetaChannels > 0)
      userInputMetaBuffer = new float[(size_t)m.numInputMetaChannels * maxBatchSize];
    else
      userInputMetaBuffer = NULL;

    policyPassResults = new float[(size_t)maxBatchSize * m.numPolicyChannels];
    policyResults = new float[(size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels];
    usingPinnedMemory = false;
    eventInputBuffer = NULL;
    eventInputGlobalBuffer = NULL;
    eventInputMetaBuffer = NULL;
  }

  ~InputBuffers() {
    if(eventInputBuffer != NULL)
      cudaFreeHost(eventInputBuffer);
    if(eventInputGlobalBuffer != NULL)
      cudaFreeHost(eventInputGlobalBuffer);
    if(eventInputMetaBuffer != NULL)
      cudaFreeHost(eventInputMetaBuffer);
    if(usingPinnedMemory) {
      cudaFreeHost(userInputBuffer);
      cudaFreeHost(userInputGlobalBuffer);
      if(userInputMetaBuffer != NULL)
        cudaFreeHost(userInputMetaBuffer);
      cudaFreeHost(policyPassResults);
      cudaFreeHost(policyResults);
      cudaFreeHost(valueResults);
      cudaFreeHost(scoreValueResults);
      cudaFreeHost(ownershipResults);
    }
    else {
      delete[] userInputBuffer;
      delete[] userInputGlobalBuffer;
      if(userInputMetaBuffer != NULL)
        delete[] userInputMetaBuffer;
      delete[] policyPassResults;
      delete[] policyResults;
      delete[] valueResults;
      delete[] scoreValueResults;
      delete[] ownershipResults;
    }
  }

  void enablePinnedMemory() {
    testAssert(!usingPinnedMemory);
    vector<float*> allocated;
    auto allocate = [&allocated](size_t bytes) {
      if(bytes == 0)
        return (float*)NULL;
      float* ptr = NULL;
      cudaError_t error = cudaHostAlloc((void**)&ptr,bytes,cudaHostAllocPortable);
      if(error != cudaSuccess) {
        for(float* allocatedPtr : allocated)
          cudaFreeHost(allocatedPtr);
        throw StringError(
          "enableEventGatedPipeline: cudaHostAlloc failed: " + string(cudaGetErrorString(error))
        );
      }
      allocated.push_back(ptr);
      return ptr;
    };

    float* newUserInputBuffer = allocate(userInputBufferBytes);
    float* newUserInputGlobalBuffer = allocate(userInputGlobalBufferBytes);
    float* newUserInputMetaBuffer = allocate(userInputMetaBufferBytes);
    float* newPolicyPassResults = allocate(policyPassResultBufferBytes);
    float* newPolicyResults = allocate(policyResultBufferBytes);
    float* newValueResults = allocate(valueResultBufferBytes);
    float* newScoreValueResults = allocate(scoreValueResultBufferBytes);
    float* newOwnershipResults = allocate(ownershipResultBufferBytes);

    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    if(userInputMetaBuffer != NULL)
      delete[] userInputMetaBuffer;
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;

    userInputBuffer = newUserInputBuffer;
    userInputGlobalBuffer = newUserInputGlobalBuffer;
    userInputMetaBuffer = newUserInputMetaBuffer;
    policyPassResults = newPolicyPassResults;
    policyResults = newPolicyResults;
    valueResults = newValueResults;
    scoreValueResults = newScoreValueResults;
    ownershipResults = newOwnershipResults;
    usingPinnedMemory = true;
  }

  void enablePinnedHalfInputs() {
    testAssert(eventInputBuffer == NULL);
    auto allocate = [](size_t numElts) {
      if(numElts == 0)
        return (half*)NULL;
      half* ptr = NULL;
      CUDA_ERR("enablePinnedHalfInputs",cudaHostAlloc(
        (void**)&ptr,numElts*sizeof(half),cudaHostAllocPortable
      ));
      return ptr;
    };
    try {
      eventInputBuffer = allocate(singleInputElts*maxBatchSize);
      eventInputGlobalBuffer = allocate(singleInputGlobalElts*maxBatchSize);
      eventInputMetaBuffer = allocate(singleInputMetaElts*maxBatchSize);
    }
    catch(...) {
      if(eventInputBuffer != NULL) cudaFreeHost(eventInputBuffer);
      if(eventInputGlobalBuffer != NULL) cudaFreeHost(eventInputGlobalBuffer);
      if(eventInputMetaBuffer != NULL) cudaFreeHost(eventInputMetaBuffer);
      eventInputBuffer = NULL;
      eventInputGlobalBuffer = NULL;
      eventInputMetaBuffer = NULL;
      throw;
    }
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel,maxBatchSize,nnXLen,nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::getRawNNOutputs(InputBuffers* inputBuffers, RawNNOutputs& out) {
  out.policyPassResults = inputBuffers->policyPassResults;
  out.policyResults = inputBuffers->policyResults;
  out.valueResults = inputBuffers->valueResults;
  out.scoreValueResults = inputBuffers->scoreValueResults;
  out.ownershipResults = inputBuffers->ownershipResults;
  out.numPolicyChannels = inputBuffers->singlePolicyPassResultElts;
  out.numValueChannels = inputBuffers->singleValueResultElts;
  out.numScoreValueChannels = inputBuffers->singleScoreValueResultElts;
  out.numOwnershipChannels = inputBuffers->singleOwnershipResultElts;
}

void NeuralNet::enableEventGatedPipeline(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers
) {
  testAssert(gpuHandle != NULL);
  testAssert(inputBuffers != NULL);
  gpuHandle->makeCurrent("enableEventGatedPipeline");
  if(!gpuHandle->usingFP16)
    throw StringError("Event-gated CUDA pipeline requires FP16 inputs");
  if(gpuHandle->eventPipelineUseGraph && gpuHandle->sm89Model == nullptr)
    throw StringError("cudaEventPipelineUseGraph currently requires the SM89 custom backend");
  gpuHandle->enableEventPipeline();
  inputBuffers->enablePinnedMemory();
  inputBuffers->enablePinnedHalfInputs();
  std::memset(
    inputBuffers->eventInputBuffer,0,
    inputBuffers->singleInputElts*inputBuffers->maxBatchSize*sizeof(half)
  );
  std::memset(
    inputBuffers->eventInputGlobalBuffer,0,
    inputBuffers->singleInputGlobalElts*inputBuffers->maxBatchSize*sizeof(half)
  );
  if(inputBuffers->eventInputMetaBuffer != NULL) {
    std::memset(
      inputBuffers->eventInputMetaBuffer,0,
      inputBuffers->singleInputMetaElts*inputBuffers->maxBatchSize*sizeof(half)
    );
  }
  CUDA_ERR("enableEventGatedPipeline",cudaEventRecord(
    gpuHandle->inputReadyEvent,gpuHandle->uploadStream
  ));
  CUDA_ERR("enableEventGatedPipeline",cudaStreamSynchronize(gpuHandle->uploadStream));
}

static bool queryEventWithoutBlocking(const char* opName, cudaEvent_t event) {
  cudaError_t error = cudaEventQuery(event);
  if(error == cudaSuccess)
    return true;
  if(error == cudaErrorNotReady)
    return false;
  throw StringError(string(opName) + ": " + cudaGetErrorString(error));
}

bool NeuralNet::eventPipelineInputHostReusable(ComputeHandle* gpuHandle) {
  testAssert(gpuHandle != NULL && gpuHandle->eventPipelineEnabled);
  gpuHandle->makeCurrent("eventPipelineInputHostReusable");
  return queryEventWithoutBlocking("eventPipelineInputHostReusable",gpuHandle->inputReadyEvent);
}

//---------------------------------------------------------------------------------------

static void prepareHostInput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int batchSize,
  NNResultBuf** inputBufs
) {
  assert(batchSize > 0 && batchSize <= inputBuffers->maxBatchSize);
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->model->modelVersion;
  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + inputBuffers->singleInputElts * nIdx;
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + inputBuffers->singleInputGlobalElts * nIdx;
    float* rowMetaInput = numMetaFeatures > 0 ?
      inputBuffers->userInputMetaBuffer + inputBuffers->singleInputMetaElts * nIdx : NULL;
    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta,rowMeta+numMetaFeatures,rowMetaInput);
    }
    else
      testAssert(!hasRowMeta);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial,rowSpatialInput,1,nnYLen,nnXLen,numSpatialFeatures,
      gpuHandle->inputsUseNHWC,inputBufs[nIdx]->symmetry
    );
  }
}

static void finishHostOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int batchSize,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
);

void NeuralNet::prepareEventPipelineInput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs
) {
  testAssert(gpuHandle != NULL && gpuHandle->eventPipelineEnabled);
  testAssert(inputBuffers != NULL && inputBuffers->eventInputBuffer != NULL);
  gpuHandle->makeCurrent("launchEventPipelineInference");
  testAssert(eventPipelineInputHostReusable(gpuHandle));
  prepareHostInput(gpuHandle,inputBuffers,numBatchEltsFilled,inputBufs);

  const size_t inputElts = inputBuffers->singleInputElts*numBatchEltsFilled;
  const size_t globalElts = inputBuffers->singleInputGlobalElts*numBatchEltsFilled;
  const size_t metaElts = inputBuffers->singleInputMetaElts*numBatchEltsFilled;
  for(size_t i = 0; i < inputElts; i++)
    inputBuffers->eventInputBuffer[i] = __float2half_rn(inputBuffers->userInputBuffer[i]);
  for(size_t i = 0; i < globalElts; i++)
    inputBuffers->eventInputGlobalBuffer[i] = __float2half_rn(inputBuffers->userInputGlobalBuffer[i]);
  for(size_t i = 0; i < metaElts; i++)
    inputBuffers->eventInputMetaBuffer[i] = __float2half_rn(inputBuffers->userInputMetaBuffer[i]);
}

void NeuralNet::launchEventPipelineInference(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled
) {
  testAssert(gpuHandle != NULL && gpuHandle->eventPipelineEnabled);
  testAssert(inputBuffers != NULL && inputBuffers->eventInputBuffer != NULL);
  // Submission runs on a persistent worker thread, whose current device is
  // otherwise unrelated to the slot it owns. cuBLAS and CUDA stream/event
  // operations must execute with this handle's device current.
  gpuHandle->makeCurrent("launchEventPipelineInference");
  testAssert(numBatchEltsFilled > 0 && numBatchEltsFilled <= inputBuffers->maxBatchSize);
  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  const int batchSize = numBatchEltsFilled;

  CUDA_ERR("launchEventPipelineInference",cudaStreamWaitEvent(
    gpuHandle->uploadStream,gpuHandle->inputConsumedEvent,0
  ));
  CUDA_ERR("launchEventPipelineInference",cudaMemcpyAsync(
    buffers->inputBuf,inputBuffers->eventInputBuffer,
    inputBuffers->singleInputElts*batchSize*sizeof(half),
    cudaMemcpyHostToDevice,gpuHandle->uploadStream
  ));
  CUDA_ERR("launchEventPipelineInference",cudaMemcpyAsync(
    buffers->inputGlobalBuf,inputBuffers->eventInputGlobalBuffer,
    inputBuffers->singleInputGlobalElts*batchSize*sizeof(half),
    cudaMemcpyHostToDevice,gpuHandle->uploadStream
  ));
  if(inputBuffers->singleInputMetaElts > 0) {
    CUDA_ERR("launchEventPipelineInference",cudaMemcpyAsync(
      buffers->inputMetaBuf,inputBuffers->eventInputMetaBuffer,
      inputBuffers->singleInputMetaElts*batchSize*sizeof(half),
      cudaMemcpyHostToDevice,gpuHandle->uploadStream
    ));
  }
  CUDA_ERR("launchEventPipelineInference",cudaEventRecord(
    gpuHandle->inputReadyEvent,gpuHandle->uploadStream
  ));

  cudaStream_t computeStream = gpuHandle->cudaHandles->stream;
  auto enqueueCompute = [&](bool useExternalEventNodes) {
    CUDA_ERR("launchEventPipelineInference",cudaStreamWaitEvent(
      computeStream,gpuHandle->inputReadyEvent,
      useExternalEventNodes ? cudaEventWaitExternal : cudaEventWaitDefault
    ));
    gpuHandle->apply(
      gpuHandle->cudaHandles.get(),scratch,batchSize,gpuHandle->requireExactNNLen,
      buffers->inputBuf,buffers->inputGlobalBuf,buffers->inputMetaBuf,
      buffers->policyPassBuf,buffers->policyBuf,
      buffers->valueBuf,buffers->scoreValueBuf,buffers->ownershipBuf,
      buffers->workspaceBuf,buffers->workspaceBytes,
      gpuHandle->inputConsumedEvent,gpuHandle->outputConsumedEvent
    );
    CUDA_ERR("launchEventPipelineInference",cudaEventRecordWithFlags(
      gpuHandle->applyCompleteEvent,computeStream,
      useExternalEventNodes ? cudaEventRecordExternal : cudaEventRecordDefault
    ));
  };

  if(gpuHandle->eventPipelineUseGraph) {
    if(batchSize != gpuHandle->eventPipelineBatchSize)
      throw StringError("cudaEventPipelineUseGraph requires fixed max-batch launches");
    if(gpuHandle->eventPipelineGraphExec == NULL) {
      CUDA_ERR("launchEventPipelineInference",cudaStreamBeginCapture(
        computeStream,cudaStreamCaptureModeThreadLocal
      ));
      try {
        enqueueCompute(true);
        CUDA_ERR("launchEventPipelineInference",cudaStreamEndCapture(
          computeStream,&gpuHandle->eventPipelineGraph
        ));
      }
      catch(...) {
        cudaGraph_t discardedGraph = NULL;
        cudaStreamEndCapture(computeStream,&discardedGraph);
        if(discardedGraph != NULL)
          cudaGraphDestroy(discardedGraph);
        throw;
      }
      CUDA_ERR("launchEventPipelineInference",cudaGraphInstantiate(
        &gpuHandle->eventPipelineGraphExec,gpuHandle->eventPipelineGraph,NULL,NULL,0
      ));
    }
    CUDA_ERR("launchEventPipelineInference",cudaGraphLaunch(
      gpuHandle->eventPipelineGraphExec,computeStream
    ));
  }
  else
    enqueueCompute(false);
}

void NeuralNet::enqueueEventPipelineOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled
) {
  testAssert(gpuHandle != NULL && gpuHandle->eventPipelineEnabled);
  testAssert(inputBuffers != NULL && inputBuffers->usingPinnedMemory);
  gpuHandle->makeCurrent("enqueueEventPipelineOutput");
  testAssert(numBatchEltsFilled > 0 && numBatchEltsFilled <= inputBuffers->maxBatchSize);
  Buffers* buffers = gpuHandle->buffers.get();
  const int batchSize = numBatchEltsFilled;
  cudaStream_t stream = gpuHandle->downloadStream;
  CUDA_ERR("enqueueEventPipelineOutput",cudaStreamWaitEvent(
    stream,gpuHandle->applyCompleteEvent,0
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaMemcpyAsync(
    inputBuffers->policyPassResults,buffers->policyPassBuf,
    inputBuffers->singlePolicyPassResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaMemcpyAsync(
    inputBuffers->policyResults,buffers->policyBuf,
    inputBuffers->singlePolicyResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaMemcpyAsync(
    inputBuffers->valueResults,buffers->valueBuf,
    inputBuffers->singleValueResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaMemcpyAsync(
    inputBuffers->scoreValueResults,buffers->scoreValueBuf,
    inputBuffers->singleScoreValueResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaMemcpyAsync(
    inputBuffers->ownershipResults,buffers->ownershipBuf,
    inputBuffers->singleOwnershipResultBytes*batchSize,cudaMemcpyDeviceToHost,stream
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaEventRecord(
    gpuHandle->outputConsumedEvent,stream
  ));
  CUDA_ERR("enqueueEventPipelineOutput",cudaEventRecord(
    gpuHandle->outputReadyEvent,stream
  ));
}

bool NeuralNet::eventPipelineOutputReady(ComputeHandle* gpuHandle) {
  testAssert(gpuHandle != NULL && gpuHandle->eventPipelineEnabled);
  gpuHandle->makeCurrent("eventPipelineOutputReady");
  return queryEventWithoutBlocking("eventPipelineOutputReady",gpuHandle->outputReadyEvent);
}

void NeuralNet::finishEventPipelineOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  testAssert(eventPipelineOutputReady(gpuHandle));
  finishHostOutput(gpuHandle,inputBuffers,numBatchEltsFilled,inputBufs,outputs);
}


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
#ifndef NDEBUG
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->model->modelVersion;
  const int numPolicyChannels = gpuHandle->model->numPolicyChannels;
#endif
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  prepareHostInput(gpuHandle,inputBuffers,batchSize,inputBufs);

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();
  CudaHandles* cudaHandles = gpuHandle->cudaHandles.get();

  if(!gpuHandle->usingFP16) {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytes);
    assert(inputBuffers->policyPassResultBufferBytes == buffers->policyPassBufBytes);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singleInputMetaBytes == inputBuffers->singleInputMetaElts*4);
    assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == numPolicyChannels*nnXLen*nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == numPolicyChannels*nnXLen*nnYLen * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    if(numMetaFeatures > 0) {
      CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputMetaBuf, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    }
  }
  else {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytesFloat);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytesFloat);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytesFloat);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes*2);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes*2);
    assert(inputBuffers->userInputMetaBufferBytes == buffers->inputMetaBufBytes*2);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singleInputMetaBytes == inputBuffers->singleInputMetaElts*4);
    assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
    assert(inputBuffers->singlePolicyPassResultBytes == numPolicyChannels * sizeof(float));
    assert(inputBuffers->singlePolicyResultElts == numPolicyChannels*nnXLen*nnYLen);
    assert(inputBuffers->singlePolicyResultBytes == numPolicyChannels*nnXLen*nnYLen * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    if(numMetaFeatures > 0) {
      CUDA_ERR("getOutput",cudaMemcpyAsync(buffers->inputMetaBufFloat, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    }

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize, cudaHandles->stream);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize, cudaHandles->stream);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    if(numMetaFeatures > 0) {
      customCudaCopyToHalf((const float*)buffers->inputMetaBufFloat,(half*)buffers->inputMetaBuf,inputBuffers->singleInputMetaElts*batchSize, cudaHandles->stream);
      CUDA_ERR("getOutput",cudaPeekAtLastError());
    }
  }

  gpuHandle->apply(
    gpuHandle->cudaHandles.get(),
    scratch,
    batchSize,
    gpuHandle->requireExactNNLen,

    buffers->inputBuf,
    buffers->inputGlobalBuf,
    buffers->inputMetaBuf,

    buffers->policyPassBuf,
    buffers->policyBuf,

    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,

    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->policyPassResults, buffers->policyPassBuf, inputBuffers->singlePolicyPassResultBytes*batchSize, cudaMemcpyDeviceToHost, cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost, cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost, cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost, cudaHandles->stream));
  CUDA_ERR("getOutput",cudaMemcpyAsync(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost, cudaHandles->stream));
  CUDA_ERR("getOutput",cudaStreamSynchronize(cudaHandles->stream));

  finishHostOutput(gpuHandle,inputBuffers,batchSize,inputBufs,outputs);
}

static void finishHostOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int batchSize,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(batchSize > 0 && batchSize <= inputBuffers->maxBatchSize);
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->model->modelVersion;
  const int numPolicyChannels = gpuHandle->model->numPolicyChannels;

  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
    const float* policySrcBuf = inputBuffers->policyResults + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      if(gpuHandle->usingNHWC) {
        for(int i = 0; i<nnXLen*nnYLen; i++) {
          float p = policySrcBuf[i*numPolicyChannels];
          float pOpt = policySrcBuf[i*numPolicyChannels+1];
          policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
      else {
        for(int i = 0; i<nnXLen*nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i+nnXLen*nnYLen];
          policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
    }
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0];
    }

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(modelVersion >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    }
    else if(modelVersion >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

}

static void cudaUploadBenchmarkInputs(ComputeHandle* gpuHandle, InputBuffers* inputBuffers, int batchSize) {
  Buffers* buffers = gpuHandle->buffers.get();
  CudaHandles* cudaHandles = gpuHandle->cudaHandles.get();
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;
  if(!gpuHandle->usingFP16) {
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    if(numMetaFeatures > 0) {
      CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputMetaBuf, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    }
  }
  else {
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    if(numMetaFeatures > 0) {
      CUDA_ERR("benchmarkOutput",cudaMemcpyAsync(buffers->inputMetaBufFloat, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes*batchSize, cudaMemcpyHostToDevice, cudaHandles->stream));
    }

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize, cudaHandles->stream);
    CUDA_ERR("benchmarkOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize, cudaHandles->stream);
    CUDA_ERR("benchmarkOutput",cudaPeekAtLastError());
    if(numMetaFeatures > 0) {
      customCudaCopyToHalf((const float*)buffers->inputMetaBufFloat,(half*)buffers->inputMetaBuf,inputBuffers->singleInputMetaElts*batchSize, cudaHandles->stream);
      CUDA_ERR("benchmarkOutput",cudaPeekAtLastError());
    }
  }
}

bool NeuralNet::benchmarkOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int batchSize,
  int numWarmups,
  int numIterations,
  vector<double>& iterationSeconds,
  BenchmarkForwardBarrier* phaseBarrier,
  int serverThreadIdx,
  int phaseOffsetMicros
) {
  assert(batchSize > 0 && batchSize <= inputBuffers->maxBatchSize);
  if(numWarmups < 0 || numIterations <= 0)
    throw StringError("benchmarkOutput: invalid warmup/iteration count");

  iterationSeconds.clear();

  // One-time H2D preparation, excluded from the timed loop.
  cudaUploadBenchmarkInputs(gpuHandle, inputBuffers, batchSize);

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();

  for(int w = 0; w < numWarmups; w++) {
    gpuHandle->apply(
      gpuHandle->cudaHandles.get(),
      scratch,
      batchSize,
      gpuHandle->requireExactNNLen,
      buffers->inputBuf,
      buffers->inputGlobalBuf,
      buffers->inputMetaBuf,
      buffers->policyPassBuf,
      buffers->policyBuf,
      buffers->valueBuf,
      buffers->scoreValueBuf,
      buffers->ownershipBuf,
      buffers->workspaceBuf,
      buffers->workspaceBytes
    );
  }
  CUDA_ERR("benchmarkOutput",cudaStreamSynchronize(gpuHandle->cudaHandles->stream));

  std::vector<cudaEvent_t> startEvents(numIterations);
  std::vector<cudaEvent_t> endEvents(numIterations);
  for(int i = 0; i < numIterations; i++) {
    CUDA_ERR("benchmarkOutput",cudaEventCreate(&startEvents[i]));
    CUDA_ERR("benchmarkOutput",cudaEventCreate(&endEvents[i]));
  }

  if(phaseBarrier != NULL)
    phaseBarrier->arriveAndWait(serverThreadIdx,phaseOffsetMicros);

  try {
    for(int i = 0; i < numIterations; i++) {
      CUDA_ERR("benchmarkOutput",cudaEventRecord(startEvents[i], gpuHandle->cudaHandles->stream));
      gpuHandle->apply(
        gpuHandle->cudaHandles.get(),
        scratch,
        batchSize,
        gpuHandle->requireExactNNLen,
        buffers->inputBuf,
        buffers->inputGlobalBuf,
        buffers->inputMetaBuf,
        buffers->policyPassBuf,
        buffers->policyBuf,
        buffers->valueBuf,
        buffers->scoreValueBuf,
        buffers->ownershipBuf,
        buffers->workspaceBuf,
        buffers->workspaceBytes
      );
      CUDA_ERR("benchmarkOutput",cudaEventRecord(endEvents[i], gpuHandle->cudaHandles->stream));
    }
    CUDA_ERR("benchmarkOutput",cudaStreamSynchronize(gpuHandle->cudaHandles->stream));

    iterationSeconds.reserve(numIterations);
    for(int i = 0; i < numIterations; i++) {
      float milliseconds = 0.0f;
      CUDA_ERR("benchmarkOutput",cudaEventElapsedTime(&milliseconds,startEvents[i],endEvents[i]));
      iterationSeconds.push_back((double)milliseconds / 1000.0);
    }
  }
  catch(...) {
    for(int i = 0; i < numIterations; i++) {
      cudaEventDestroy(startEvents[i]);
      cudaEventDestroy(endEvents[i]);
    }
    throw;
  }

  for(int i = 0; i < numIterations; i++) {
    cudaEventDestroy(startEvents[i]);
    cudaEventDestroy(endEvents[i]);
  }
  return true;
}

//TESTING ----------------------------------------------------------------------------------


bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");

  void* deviceInput;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ConvLayer* convLayer = new ConvLayer(cudaHandles,manager,desc,useFP16,useNHWC);

  size_t workspaceBytes =
    convLayer->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));


  bool accumulate = false;
  convLayer->apply(
    cudaHandles,
    desiredBatchSize,
    accumulate,
    deviceInput,
    deviceOutput,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  cudaFree(deviceWorkspace);

  delete convLayer;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected mask buffer size");

  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  void* deviceInput;
  void* deviceMask;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  BatchNormLayer* batchNormLayer = new BatchNormLayer(cudaHandles,desc,&actDesc,nnXLen,nnYLen,useFP16,useNHWC);

  batchNormLayer->apply(
    cudaHandles,
    desiredBatchSize,
    deviceInput,
    deviceMask,
    deviceOutput
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  delete batchNormLayer;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  void* deviceScratch;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  ResidualBlock* residualBlock = new ResidualBlock(cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC);

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  size_t numInputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)desiredBatchSize;
  size_t numOutputFloats = (size_t)desiredBatchSize * nnXLen * nnYLen * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, nnXLen, nnYLen, useFP16);

  void* deviceInput;
  void* deviceMask;
  float* deviceMaskFloatOrig;
  float* deviceMaskFloat;
  float* deviceMaskSum;
  void* deviceScratch;

  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CUDA_ERR("deviceMaskFloat",cudaMalloc(reinterpret_cast<void**>(&deviceMaskFloat), numMaskFloats * sizeof(float)));
  CUDA_ERR("deviceMaskSum",cudaMalloc(reinterpret_cast<void**>(&deviceMaskSum), numMaskSumFloats * sizeof(float)));
  deviceMaskFloatOrig = deviceMaskFloat;
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  fillMaskFloatBufAndMaskSumBuf(cudaHandles, deviceMask, deviceMaskFloat, deviceMaskSum, useFP16, desiredBatchSize, nnXLen, nnYLen);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,nnXLen,nnYLen);
  GlobalPoolingResidualBlock* residualBlock = new GlobalPoolingResidualBlock(
    cudaHandles,manager,desc,nnXLen,nnYLen,useFP16,useNHWC
  );

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(
      cudaHandles,desiredBatchSize
    );

  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceMaskSum,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceMaskFloatOrig);
  cudaFree(deviceMaskSum);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}


#endif  // USE_CUDA_BACKEND
