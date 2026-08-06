#include "../neuralnet/cudaincludes.h"
#if CUDNN_VERSION >= 8903
#include <cudnn_frontend.h>
#endif

#include "../neuralnet/cudabackend_sm89_forward.h"
#include "../neuralnet/cudabackend_sm89_dual_gemm.h"
#include "../neuralnet/cudabackend_sm89_flash.h"
#include "../neuralnet/cudabackend_sm89_kernels.h"
#include "../neuralnet/cudabackend_sm89_linear2_gemm.h"
#include "../neuralnet/cudabackend_sm89_qkv_rope_gemm.h"

#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudautils.h"

#include "../core/global.h"
#include "../core/logger.h"
#include "../core/test.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <unordered_map>

using namespace std;

namespace Sm89Backend {

// --------------------------------------------------------------------------------------
// Context / scratch

static void setPersistingL2Window(
  cudaStream_t stream,
  void* basePtr,
  size_t numBytes,
  float hitRatio
) {
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = basePtr;
  attr.accessPolicyWindow.num_bytes = numBytes;
  attr.accessPolicyWindow.hitRatio = hitRatio;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CUDA_ERR("Sm89PersistingL2",cudaStreamSetAttribute(
    stream, cudaStreamAttributeAccessPolicyWindow, &attr
  ));
}

static void clearPersistingL2Window(cudaStream_t stream) {
  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CUDA_ERR("Sm89PersistingL2",cudaStreamSetAttribute(
    stream, cudaStreamAttributeAccessPolicyWindow, &attr
  ));
}

Sm89Ctx::Sm89Ctx()
  : cublas(NULL), cudnn(NULL), stream(cudaStreamPerThread)
{
  CUBLAS_ERR("Sm89Ctx",cublasCreate(&cublas));
  CUDNN_ERR("Sm89Ctx",cudnnCreate(&cudnn));
  CUBLAS_ERR("Sm89Ctx",cublasSetStream(cublas, stream));
  CUDNN_ERR("Sm89Ctx",cudnnSetStream(cudnn, stream));
}

Sm89Ctx::~Sm89Ctx() {
  cublasDestroy(cublas);
  cudnnDestroy(cudnn);
}

Sm89Scratch::Sm89Scratch(
  bool useFP16, bool useExactMaskElision, int maxBatchSize, int xySize
)
  : allocator(
      [](size_t size) {
        void* buf = NULL;
        CUDA_ERR("Sm89Scratch",cudaMalloc(&buf, size));
        return buf;
      },
      [](void* buf) {
        cudaFree(buf);
      }
    ),
    zeroBuf(NULL),
    oneBuf(NULL),
    fullBoardMaskSumBuf(NULL)
{
  CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
  if(useExactMaskElision) {
    vector<float> fullBoardMaskSums(maxBatchSize, (float)xySize);
    CudaUtils::mallocAndCopyToDevice(
      "Sm89Scratch:fullBoardMaskSum", fullBoardMaskSums,
      fullBoardMaskSumBuf, false
    );
  }
}

Sm89Scratch::~Sm89Scratch() {
  if(fullBoardMaskSumBuf != NULL)
    cudaFree(fullBoardMaskSumBuf);
  free(zeroBuf);
  free(oneBuf);
}

size_t Sm89Scratch::getBufSizeXY(int channels, int maxBatchSize, int xySize, bool useFP16) const {
  return (size_t)channels * maxBatchSize * xySize * (useFP16 ? sizeof(half) : sizeof(float));
}
size_t Sm89Scratch::getBufSizeXYFloat(int channels, int maxBatchSize, int xySize) const {
  return (size_t)channels * maxBatchSize * xySize * sizeof(float);
}
size_t Sm89Scratch::getBufSizeFloat(int channels, int maxBatchSize) const {
  return (size_t)channels * maxBatchSize * sizeof(float);
}
size_t Sm89Scratch::getBufSize(int channels, int maxBatchSize, bool useFP16) const {
  return (size_t)channels * maxBatchSize * (useFP16 ? sizeof(half) : sizeof(float));
}

// --------------------------------------------------------------------------------------
// cuDNN SDPA plan cache (same graph shape/order as the official backend)

#if CUDNN_VERSION >= 8903
struct Sm89SDPAPlan {
  std::shared_ptr<cudnn_frontend::graph::Graph> graph;
  int64_t workspaceBytes;
  bool hasMask;
};

struct Sm89SDPAKey {
  int numHeads;
  int numKVHeads;
  int qHeadDim;
  int vHeadDim;
  int seqLen;
  int batchSize;
  bool hasMask;

  bool operator==(const Sm89SDPAKey& o) const {
    return numHeads == o.numHeads && numKVHeads == o.numKVHeads &&
           qHeadDim == o.qHeadDim && vHeadDim == o.vHeadDim &&
           seqLen == o.seqLen && batchSize == o.batchSize && hasMask == o.hasMask;
  }
};

struct Sm89SDPAKeyHash {
  size_t operator()(const Sm89SDPAKey& k) const {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](size_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix((size_t)k.numHeads);
    mix((size_t)k.numKVHeads);
    mix((size_t)k.qHeadDim);
    mix((size_t)k.vHeadDim);
    mix((size_t)k.seqLen);
    mix((size_t)k.batchSize);
    mix(k.hasMask ? 1 : 0);
    return h;
  }
};

class Sm89SDPACache {
 public:
  explicit Sm89SDPACache(cudnnHandle_t cudnn) : cudnn(cudnn), supported(true) {}

  std::shared_ptr<Sm89SDPAPlan> getPlan(const Sm89SDPAKey& key) {
    if(!supported)
      return nullptr;
    auto it = plans.find(key);
    if(it != plans.end())
      return it->second;

    namespace fe = cudnn_frontend;
    auto plan = std::make_shared<Sm89SDPAPlan>();
    plan->hasMask = key.hasMask;
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t B = key.batchSize;
    int64_t Hq = key.numHeads;
    int64_t Hkv = key.numKVHeads;
    int64_t S = key.seqLen;
    int64_t Dq = key.qHeadDim;
    int64_t Dv = key.vHeadDim;

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
      .set_name("Q").set_uid(1)
      .set_dim({B, Hq, S, Dq})
      .set_stride({S * Hq * Dq, Dq, Hq * Dq, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
      .set_name("K").set_uid(2)
      .set_dim({B, Hkv, S, Dq})
      .set_stride({S * Hkv * Dq, Dq, Hkv * Dq, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
      .set_name("V").set_uid(3)
      .set_dim({B, Hkv, S, Dv})
      .set_stride({S * Hkv * Dv, Dv, Hkv * Dv, 1}));

    auto sdpa_options = (
      fe::graph::SDPA_attributes()
      .set_name("sdpa_fwd")
      .set_generate_stats(false)
      .set_attn_scale(1.0f / std::sqrt((float)key.qHeadDim))
    );
    if(key.hasMask) {
      auto bias = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("bias").set_uid(5)
        .set_dim({B, 1, S, S})
        .set_stride({S * S, S * S, S, 1}));
      sdpa_options.set_bias(bias);
    }
    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
    (void)Stats;
    O->set_output(true)
      .set_dim({B, Hq, S, Dv})
      .set_stride({S * Hq * Dv, Dv, Hq * Dv, 1})
      .set_uid(4);

    auto status = graph->validate();
    if(status.is_bad())
      return nullptr;
    status = graph->build_operation_graph(cudnn);
    if(status.is_bad())
      return nullptr;
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if(status.is_bad())
      return nullptr;
    status = graph->check_support(cudnn);
    if(status.is_bad())
      return nullptr;
    status = graph->build_plans(cudnn);
    if(status.is_bad())
      return nullptr;
    int64_t ws = 0;
    status = graph->get_workspace_size(ws);
    if(status.is_bad())
      return nullptr;
    plan->graph = graph;
    plan->workspaceBytes = ws;
    plans[key] = plan;
    return plan;
  }

 private:
  cudnnHandle_t cudnn;
  bool supported;
  std::unordered_map<Sm89SDPAKey, std::shared_ptr<Sm89SDPAPlan>, Sm89SDPAKeyHash> plans;
};
#else
class Sm89SDPACache {
 public:
  explicit Sm89SDPACache(cudnnHandle_t) {}
};
#endif

// --------------------------------------------------------------------------------------
// Small per-batch descriptor holders (same shape/order as the official backend)

template<typename T>
struct Sm89ByBatchSize {
  int maxBatchSize;
  T* data;
  cudnnStatus_t (*destroyFunc)(T);

  Sm89ByBatchSize()
    : maxBatchSize(0), data(nullptr), destroyFunc(nullptr)
  {}
  explicit Sm89ByBatchSize(int maxBatchSize_)
    : maxBatchSize(maxBatchSize_), data(new T[maxBatchSize_]), destroyFunc(nullptr)
  {}
  ~Sm89ByBatchSize() {
    if(destroyFunc != nullptr && data != nullptr) {
      for(int i = 0; i < maxBatchSize; i++)
        (*destroyFunc)(data[i]);
    }
    delete[] data;
  }
  Sm89ByBatchSize(const Sm89ByBatchSize&) = delete;
  Sm89ByBatchSize& operator=(const Sm89ByBatchSize&) = delete;
  Sm89ByBatchSize(Sm89ByBatchSize&& other) noexcept
    : maxBatchSize(other.maxBatchSize), data(other.data), destroyFunc(other.destroyFunc)
  {
    other.maxBatchSize = 0;
    other.data = nullptr;
    other.destroyFunc = nullptr;
  }
  Sm89ByBatchSize& operator=(Sm89ByBatchSize&& other) noexcept {
    if(this != &other) {
      if(destroyFunc != nullptr && data != nullptr) {
        for(int i = 0; i < maxBatchSize; i++)
          (*destroyFunc)(data[i]);
      }
      delete[] data;
      maxBatchSize = other.maxBatchSize;
      data = other.data;
      destroyFunc = other.destroyFunc;
      other.maxBatchSize = 0;
      other.data = nullptr;
      other.destroyFunc = nullptr;
    }
    return *this;
  }
  T& operator[](int batchSize) {
    return data[batchSize - 1];
  }
};

template<typename T>
struct Sm89ByBatchSizeView {
  int maxBatchSize;
  T* data;

  Sm89ByBatchSizeView() : maxBatchSize(0), data(nullptr) {}
  explicit Sm89ByBatchSizeView(const Sm89ByBatchSize<T>& src)
    : maxBatchSize(src.maxBatchSize), data(src.data)
  {}
  Sm89ByBatchSizeView& operator=(const Sm89ByBatchSize<T>& src) {
    maxBatchSize = src.maxBatchSize;
    data = src.data;
    return *this;
  }
  T& operator[](int batchSize) const {
    return data[batchSize - 1];
  }
};

// --------------------------------------------------------------------------------------
// MatMul

struct Sm89SharedMatMulWeight {
  void* ptr;
  int device;

  Sm89SharedMatMulWeight(void* ptr_, int device_) : ptr(ptr_), device(device_) {}
  ~Sm89SharedMatMulWeight() {
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

struct Sm89SharedMatMulKey {
  const MatMulLayerDesc* desc;
  int device;
  bool useFP16;

  bool operator==(const Sm89SharedMatMulKey& other) const {
    return desc == other.desc && device == other.device && useFP16 == other.useFP16;
  }
};

struct Sm89SharedMatMulKeyHash {
  size_t operator()(const Sm89SharedMatMulKey& key) const noexcept {
    size_t h = std::hash<const void*>()((const void*)key.desc);
    h ^= std::hash<int>()(key.device) + 0x9e3779b9U + (h << 6) + (h >> 2);
    h ^= std::hash<bool>()(key.useFP16) + 0x9e3779b9U + (h << 6) + (h >> 2);
    return h;
  }
};

static std::shared_ptr<Sm89SharedMatMulWeight> getSharedMatMulWeight(
  const MatMulLayerDesc* desc, bool useFP16, const string& name
) {
  static std::mutex mutex;
  static std::unordered_map<
    Sm89SharedMatMulKey,
    std::weak_ptr<Sm89SharedMatMulWeight>,
    Sm89SharedMatMulKeyHash
  > cache;

  int device = 0;
  CUDA_ERR(name.c_str(),cudaGetDevice(&device));
  const Sm89SharedMatMulKey key{desc, device, useFP16};
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(key);
  if(it != cache.end()) {
    std::shared_ptr<Sm89SharedMatMulWeight> existing = it->second.lock();
    if(existing != nullptr)
      return existing;
  }

  void* ptr = NULL;
  CudaUtils::mallocAndCopyToDevice(name, desc->weights, ptr, useFP16);
  auto weight = std::make_shared<Sm89SharedMatMulWeight>(ptr, device);
  cache[key] = weight;
  return weight;
}

struct Sm89MatMul {
  const string name;
  const int inChannels;
  const int outChannels;
  const bool usingFP16;
  void* matBuf;
  std::shared_ptr<Sm89SharedMatMulWeight> sharedWeight;

  Sm89MatMul() = delete;
  Sm89MatMul(const Sm89MatMul&) = delete;
  Sm89MatMul& operator=(const Sm89MatMul&) = delete;

  Sm89MatMul(const MatMulLayerDesc* desc, bool useFP16, bool shareWeights = false)
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      usingFP16(useFP16),
      matBuf(NULL),
      sharedWeight(nullptr)
  {
    if(inChannels > 0 && outChannels > 0) {
      testAssert((int)desc->weights.size() == inChannels * outChannels);
      if(shareWeights) {
        sharedWeight = getSharedMatMulWeight(desc, useFP16, name);
        matBuf = sharedWeight->ptr;
      }
      else {
        CudaUtils::mallocAndCopyToDevice(name, desc->weights, matBuf, useFP16);
      }
    }
  }

  ~Sm89MatMul() {
    if(matBuf != NULL && sharedWeight == nullptr)
      cudaFree(matBuf);
  }

  void apply(Sm89Ctx* ctx, Sm89Scratch* scratch, int batchSize, void* inputBuf, void* outputBuf) const {
    assert(inChannels > 0 && outChannels > 0);
    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        outChannels, batchSize, inChannels,
        &alpha, (const float*)matBuf, outChannels,
        (const float*)inputBuf, inChannels,
        &beta, (float*)outputBuf, outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)scratch->zeroBuf;
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        outChannels, batchSize, inChannels,
        alpha, (const half*)matBuf, outChannels,
        (const half*)inputBuf, inChannels,
        beta, (half*)outputBuf, outChannels
      ));
    }
  }

  // C = A*B + C (beta=1), for fused residual epilogues.
  void applyAccumulate(Sm89Ctx* ctx, Sm89Scratch* scratch, int batchSize, void* inputBuf, void* outputBuf) const {
    assert(inChannels > 0 && outChannels > 0);
    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = 1.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        outChannels, batchSize, inChannels,
        &alpha, (const float*)matBuf, outChannels,
        (const float*)inputBuf, inChannels,
        &beta, (float*)outputBuf, outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)scratch->oneBuf;
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        outChannels, batchSize, inChannels,
        alpha, (const half*)matBuf, outChannels,
        (const half*)inputBuf, inChannels,
        beta, (half*)outputBuf, outChannels
      ));
    }
  }
};

struct Sm89MatBias {
  const string name;
  const int numChannels;
  const bool usingFP16;
  const int activation;
  void* biasBuf;

  Sm89MatBias() = delete;
  Sm89MatBias(const Sm89MatBias&) = delete;
  Sm89MatBias& operator=(const Sm89MatBias&) = delete;

  Sm89MatBias(const MatBiasLayerDesc* desc, bool useFP16, int activation_)
    : name(desc->name),
      numChannels(desc->numChannels),
      usingFP16(useFP16),
      activation(activation_),
      biasBuf(NULL)
  {
    if(numChannels > 0) {
      testAssert((int)desc->weights.size() == numChannels);
      CudaUtils::mallocAndCopyToDevice(name, desc->weights, biasBuf, useFP16);
    }
  }
  ~Sm89MatBias() {
    if(biasBuf != NULL)
      cudaFree(biasBuf);
  }
  void apply(int batchSize, void* matBuf) const {
    assert(numChannels > 0);
    if(!usingFP16)
      customCudaAddCBiasInplaceNC((float*)matBuf, (const float*)biasBuf, batchSize, numChannels, activation);
    else
      customCudaAddCBiasInplaceNC((half*)matBuf, (const half*)biasBuf, batchSize, numChannels, activation);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
  }
};

// --------------------------------------------------------------------------------------
// BatchNorm / RMSNorm

struct Sm89BatchNorm {
  const string name;
  const int numChannels;
  const int activation;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  const bool useScaleBiasSiluVec8;
  const bool useScaleBiasSiluVec4C384;
  void* mergedScaleBuf;
  void* mergedBiasBuf;

  Sm89BatchNorm() = delete;
  Sm89BatchNorm(const Sm89BatchNorm&) = delete;
  Sm89BatchNorm& operator=(const Sm89BatchNorm&) = delete;

  Sm89BatchNorm(
    const BatchNormLayerDesc* desc, const ActivationLayerDesc* actDesc,
    int nnX, int nnY, bool useFP16, bool useNHWC,
    bool useScaleBiasSiluVec8_ = false,
    bool useScaleBiasSiluVec4C384_ = false
  )
    : name(desc->name),
      numChannels(desc->numChannels),
      activation(actDesc->activation),
      nnXLen(nnX),
      nnYLen(nnY),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      useScaleBiasSiluVec8(useScaleBiasSiluVec8_),
      useScaleBiasSiluVec4C384(useScaleBiasSiluVec4C384_),
      mergedScaleBuf(NULL),
      mergedBiasBuf(NULL)
  {
    testAssert((int)desc->mergedScale.size() == numChannels);
    testAssert((int)desc->mergedBias.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name, desc->mergedScale, mergedScaleBuf, useFP16);
    CudaUtils::mallocAndCopyToDevice(name, desc->mergedBias, mergedBiasBuf, useFP16);
  }
  ~Sm89BatchNorm() {
    cudaFree(mergedScaleBuf);
    cudaFree(mergedBiasBuf);
  }
  void apply(int batchSize, void* inputBuf, const void* maskBuf, void* outputBuf) const {
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const float*)inputBuf, (float*)outputBuf, (const float*)mergedScaleBuf, (const float*)mergedBiasBuf, (const float*)maskBuf, batchSize, numChannels, nnXLen * nnYLen, activation);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf, (float*)outputBuf, (const float*)mergedScaleBuf, (const float*)mergedBiasBuf, (const float*)maskBuf, batchSize, nnXLen * nnYLen, numChannels, activation);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf, (half*)outputBuf, (const half*)mergedScaleBuf, (const half*)mergedBiasBuf, (const half*)maskBuf, batchSize, numChannels, nnXLen * nnYLen, activation);
      else {
        bool handled = false;
        if(useScaleBiasSiluVec8 && maskBuf == NULL && activation == ACTIVATION_SILU) {
          handled = sm89ScaleBiasSiluNHWCHalfVec8B13(
            (const half*)inputBuf, (half*)outputBuf,
            (const half*)mergedScaleBuf, (const half*)mergedBiasBuf,
            batchSize, nnXLen * nnYLen, numChannels, cudaStreamPerThread
          );
        }
        if(!handled && useScaleBiasSiluVec4C384 && maskBuf == NULL && activation == ACTIVATION_SILU) {
          handled = sm89ScaleBiasSiluNHWCHalfVec4C384B13(
            (const half*)inputBuf, (half*)outputBuf,
            (const half*)mergedScaleBuf, (const half*)mergedBiasBuf,
            batchSize, nnXLen * nnYLen, numChannels, cudaStreamPerThread
          );
        }
        if(!handled)
          customCudaApplyCScaleBiasNHWC((const half*)inputBuf, (half*)outputBuf, (const half*)mergedScaleBuf, (const half*)mergedBiasBuf, (const half*)maskBuf, batchSize, nnXLen * nnYLen, numChannels, activation);
      }
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }
};

struct Sm89TransformerRMSNorm {
  const string name;
  const int numChannels;
  const float epsilon;
  const bool usingFP16;
  const bool useOptimized;
  void* weightBuf;
  void* zeroBetaBuf;

  Sm89TransformerRMSNorm() = delete;
  Sm89TransformerRMSNorm(const Sm89TransformerRMSNorm&) = delete;
  Sm89TransformerRMSNorm& operator=(const Sm89TransformerRMSNorm&) = delete;

  Sm89TransformerRMSNorm(const TransformerRMSNormDesc* desc, bool useFP16, bool useOptimized_)
    : name(desc->name),
      numChannels(desc->numChannels),
      epsilon(desc->epsilon),
      usingFP16(useFP16),
      useOptimized(useOptimized_),
      weightBuf(NULL),
      zeroBetaBuf(NULL)
  {
    testAssert((int)desc->weight.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name, desc->weight, weightBuf, useFP16);
    vector<float> zeros(numChannels, 0.0f);
    CudaUtils::mallocAndCopyToDevice(name + ":zeroBeta", zeros, zeroBetaBuf, useFP16);
  }
  ~Sm89TransformerRMSNorm() {
    cudaFree(weightBuf);
    cudaFree(zeroBetaBuf);
  }
  void apply(int batchSize, int xySize, void* inputBuf, void* outputBuf, const void* maskBuf) const {
    if(useOptimized && usingFP16 && sm89RMSNormNHWCHalf(
      (const half*)inputBuf, (half*)outputBuf, (const half*)weightBuf, (const half*)zeroBetaBuf,
      (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, cudaStreamPerThread
    ))
      return;
    if(!usingFP16)
      customCudaRMSNormGammaBetaNHWC((const float*)inputBuf, (float*)outputBuf, (const float*)weightBuf, (const float*)zeroBetaBuf, (const float*)maskBuf, batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY);
    else
      customCudaRMSNormGammaBetaNHWC((const half*)inputBuf, (half*)outputBuf, (const half*)weightBuf, (const half*)zeroBetaBuf, (const half*)maskBuf, batchSize, xySize, numChannels, epsilon, ACTIVATION_IDENTITY);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
  }
};

// --------------------------------------------------------------------------------------
// Conv (cuDNN path + 1x1 cuBLAS matmul path)

struct Sm89Conv {
  const string name;
  const int inChannels;
  const int outChannels;
  std::unique_ptr<Sm89ByBatchSize<cudnnTensorDescriptor_t>> inputDescriptors;
  std::unique_ptr<Sm89ByBatchSize<cudnnTensorDescriptor_t>> outputDescriptors;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  Sm89ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>* convolutionAlgorithms;
  void* filterBuf;
  bool use1x1Matmul;
  int matmulSpatialSize;
  void* matmulWeightBuf;
  bool usingFP16;
  bool useInitialConvFrontend;
#if CUDNN_VERSION >= 8903
  std::shared_ptr<cudnn_frontend::graph::Graph> frontendGraph;
  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> frontendX;
  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> frontendW;
  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> frontendY;
  int64_t frontendWorkspaceBytes;
#endif

  Sm89Conv() = delete;
  Sm89Conv(const Sm89Conv&) = delete;
  Sm89Conv& operator=(const Sm89Conv&) = delete;

  Sm89Conv(
    Sm89Ctx* ctx,
    const ConvLayerDesc* desc,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut,
    bool useInitialConvFrontend_ = false
  )
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      filterDescriptor(NULL),
      convolutionDescriptor(NULL),
      convolutionAlgorithms(NULL),
      filterBuf(NULL),
      use1x1Matmul(false),
      matmulSpatialSize(0),
      matmulWeightBuf(NULL),
      usingFP16(useFP16),
      useInitialConvFrontend(false)
#if CUDNN_VERSION >= 8903
      , frontendGraph(nullptr),
      frontendX(nullptr),
      frontendW(nullptr),
      frontendY(nullptr),
      frontendWorkspaceBytes(0)
#endif
  {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    testAssert(convXSize % 2 == 1 && convYSize % 2 == 1);
    if(convXSize == 1 && convYSize == 1 && useNHWCIn && useNHWCOut && useFP16) {
      use1x1Matmul = true;
      matmulSpatialSize = nnXLen * nnYLen;
      vector<float> wT((size_t)inChannels * outChannels);
      for(int oc = 0; oc < outChannels; oc++)
        for(int ic = 0; ic < inChannels; ic++)
          wT[(size_t)oc + (size_t)ic * outChannels] = desc->weights[(size_t)oc * inChannels + ic];
      CudaUtils::mallocAndCopyToDevice(name + ":matmulW", wT, matmulWeightBuf, useFP16);
      return;
    }

    inputDescriptors = std::make_unique<Sm89ByBatchSize<cudnnTensorDescriptor_t>>(
      makeTensorDescs(desc->inChannels, maxBatchSize, useFP16, useNHWCIn, nnXLen, nnYLen)
    );
    outputDescriptors = std::make_unique<Sm89ByBatchSize<cudnnTensorDescriptor_t>>(
      makeTensorDescs(desc->outChannels, maxBatchSize, useFP16, useNHWCOut, nnXLen, nnYLen)
    );

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor));
    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
      filterNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      outChannels, inChannels, convYSize, convXSize
    ));

    CUDNN_ERR(name.c_str(),cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingY, paddingX, 1, 1, dilationY, dilationX,
      CUDNN_CROSS_CORRELATION,
      (useFP16 ? CUDNN_DATA_FLOAT : CUDNN_DATA_FLOAT)
    ));
    if(useFP16)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

    convolutionAlgorithms = new Sm89ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>(maxBatchSize);
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      }
      else {
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm_v7(
          ctx->cudnn,
          (*inputDescriptors)[batchSize],
          filterDescriptor,
          convolutionDescriptor,
          (*outputDescriptors)[batchSize],
          requestedAlgoCount,
          &returnedAlgoCount,
          results
        ));
        if(returnedAlgoCount <= 0)
          throw StringError(name + ": cudnn returned no conv algorithms");
        (*convolutionAlgorithms)[batchSize] = results[0];
      }
    }

    testAssert((int)desc->weights.size() == convYSize * convXSize * inChannels * outChannels);
    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc * convYSize + y) * convXSize + x) * inChannels + ic] =
                desc->weights[((oc * inChannels + ic) * convYSize + y) * convXSize + x];
            }
          }
        }
      }
      CudaUtils::mallocAndCopyToDevice(name, weightsTransposed, filterBuf, useFP16);
    }
    else {
      CudaUtils::mallocAndCopyToDevice(name, desc->weights, filterBuf, useFP16);
    }

#if CUDNN_VERSION >= 8903
    const int64_t maxFrontendWorkspaceBytes = 64 * 1024 * 1024;
    if(
      useInitialConvFrontend_ && maxBatchSize == 13 && nnXLen == 19 && nnYLen == 19 &&
      useFP16 && useNHWCIn && useNHWCOut && filterNHWC &&
      convXSize == 3 && convYSize == 3 && dilationX == 1 && dilationY == 1
    ) {
      namespace fe = cudnn_frontend;
      auto initializeGraph = [&]() {
        frontendGraph = std::make_shared<fe::graph::Graph>();
        frontendGraph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
        frontendX = frontendGraph->tensor(fe::graph::Tensor_attributes()
          .set_name("initialConvX").set_uid(1)
          .set_dim({13, inChannels, 19, 19})
          .set_stride({19 * 19 * inChannels, 1, 19 * inChannels, inChannels}));
        frontendW = frontendGraph->tensor(fe::graph::Tensor_attributes()
          .set_name("initialConvW").set_uid(2)
          .set_dim({outChannels, inChannels, 3, 3})
          .set_stride({3 * 3 * inChannels, 1, 3 * inChannels, inChannels}));
        frontendY = frontendGraph->conv_fprop(
          frontendX, frontendW,
          fe::graph::Conv_fprop_attributes()
            .set_name("initialConv")
            .set_padding({1, 1})
            .set_stride({1, 1})
            .set_dilation({1, 1})
        );
        frontendY->set_output(true).set_uid(3)
          .set_dim({13, outChannels, 19, 19})
          .set_stride({19 * 19 * outChannels, 1, 19 * outChannels, outChannels});
      };
      auto finishGraph = [&]() -> bool {
        auto status = frontendGraph->check_support(ctx->cudnn);
        if(status.is_bad())
          return false;
        status = frontendGraph->build_plans(ctx->cudnn);
        if(status.is_bad())
          return false;
        int64_t workspaceBytes = 0;
        status = frontendGraph->get_workspace_size(workspaceBytes);
        if(status.is_bad() || workspaceBytes > maxFrontendWorkspaceBytes)
          return false;
        frontendWorkspaceBytes = workspaceBytes;
        return true;
      };

      initializeGraph();
      auto status = frontendGraph->validate();
      if(status.is_good())
        status = frontendGraph->build_operation_graph(ctx->cudnn);
      if(status.is_good())
        status = frontendGraph->create_execution_plan(
          45,
          {{fe::KnobType_t::TILE_SIZE, 0}, {fe::KnobType_t::STAGES, 2}}
        );
      bool built = status.is_good() && finishGraph();

      if(!built) {
        initializeGraph();
        status = frontendGraph->validate();
        if(status.is_good())
          status = frontendGraph->build_operation_graph(ctx->cudnn);
        if(status.is_good())
          status = frontendGraph->create_execution_plans({fe::HeurMode_t::A});
        if(status.is_good())
          frontendGraph->deselect_workspace_greater_than(maxFrontendWorkspaceBytes);
        built = status.is_good() && finishGraph();
      }

      if(built)
        useInitialConvFrontend = true;
      else {
        frontendGraph.reset();
        frontendX.reset();
        frontendW.reset();
        frontendY.reset();
      }
    }
#endif
  }

  ~Sm89Conv() {
    if(matmulWeightBuf != NULL)
      cudaFree(matmulWeightBuf);
    if(!use1x1Matmul) {
      cudaFree(filterBuf);
      cudnnDestroyFilterDescriptor(filterDescriptor);
      cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
      delete convolutionAlgorithms;
    }
  }

  void apply(Sm89Ctx* ctx, int batchSize, bool accumulate, void* inputBuf, void* outputBuf, void* workspaceBuf, size_t workspaceBytes) const {
    if(use1x1Matmul) {
      int tokens = batchSize * matmulSpatialSize;
      if(!usingFP16) {
        const float alpha = 1.0f;
        const float beta = accumulate ? 1.0f : 0.0f;
        CUBLAS_ERR(name.c_str(),cublasSgemm(
          ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const float*)matmulWeightBuf, outChannels,
          (const float*)inputBuf, inChannels,
          &beta, (float*)outputBuf, outChannels
        ));
      }
      else {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(accumulate ? 1.0f : 0.0f);
        CUBLAS_ERR(name.c_str(),cublasHgemm(
          ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          outChannels, tokens, inChannels,
          &alpha, (const half*)matmulWeightBuf, outChannels,
          (const half*)inputBuf, inChannels,
          &beta, (half*)outputBuf, outChannels
        ));
      }
      return;
    }
#if CUDNN_VERSION >= 8903
    if(
      useInitialConvFrontend && batchSize == 13 && !accumulate &&
      frontendGraph != nullptr && workspaceBytes >= (size_t)frontendWorkspaceBytes
    ) {
      std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> ptrs = {
        {frontendX, inputBuf}, {frontendW, filterBuf}, {frontendY, outputBuf}
      };
      auto status = frontendGraph->execute(ctx->cudnn, ptrs, workspaceBuf);
      if(status.is_good())
        return;
    }
#endif
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      ctx->cudnn,
      &alpha,
      (*inputDescriptors)[batchSize], inputBuf,
      filterDescriptor, filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf, workspaceBytes,
      &beta,
      (*outputDescriptors)[batchSize], outputBuf
    ));
  }

 private:
  static Sm89ByBatchSize<cudnnTensorDescriptor_t> makeTensorDescs(
    int channels, int maxBatchSize, bool useFP16, bool nhwc, int nnXLen, int nnYLen
  ) {
    Sm89ByBatchSize<cudnnTensorDescriptor_t> descs(maxBatchSize);
    descs.destroyFunc = cudnnDestroyTensorDescriptor;
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& d = descs[batchSize];
      CUDNN_ERR("Sm89Conv",cudnnCreateTensorDescriptor(&d));
      CUDNN_ERR("Sm89Conv",cudnnSetTensor4dDescriptor(
        d,
        nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        batchSize, channels, nnYLen, nnXLen
      ));
    }
    return descs;
  }
};

// --------------------------------------------------------------------------------------
// Transformer blocks

struct Sm89AttentionBlock {
  const string name;
  const int numHeads;
  const int numKVHeads;
  const int qHeadDim;
  const int vHeadDim;
  const int inChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  const bool useFusedResidual;
  const bool useRMSNormOpt;
  const bool useFusedQKRoPE;
  const bool usePrecomputedQKRoPE;
  const bool useQKVRoPEGemm;
  const int ropeBatchGroup;
  const bool useFlashAttention;
  const bool useOutProjGemm;
  const Sm89TransformerRMSNorm preLN;
  const Sm89MatMul qProj;
  const Sm89MatMul kProj;
  const Sm89MatMul vProj;
  const Sm89MatMul outProj;
  void* qkvWeightsBuf;
  bool useQKVBatched;
  void* ropeCosTable;
  void* ropeSinTable;
  float* ropeFreqsBuf;
  float2* ropeCosSinTable;
  int ropeNumPairs;
  std::shared_ptr<Sm89SDPACache> sdpaCache;
#ifdef KATAGO_ENABLE_SM89_OUTPROJ_GEMM
  std::unique_ptr<Sm89Backend::Sm89OutProjGemmB13> outProjGemm;
#endif
#ifdef KATAGO_ENABLE_SM89_QKV_ROPE_GEMM
  std::unique_ptr<Sm89Backend::Sm89QKVRoPEGemmB13> qkvRopeGemm;
#endif

  Sm89AttentionBlock() = delete;
  Sm89AttentionBlock(const Sm89AttentionBlock&) = delete;
  Sm89AttentionBlock& operator=(const Sm89AttentionBlock&) = delete;

  Sm89AttentionBlock(Sm89Ctx* ctx, const TransformerAttentionDesc* desc, int nnX, int nnY, bool useFP16, bool useNHWC, bool useWideQKV, bool useFusedResidual_, bool useRMSNormOpt_, bool useFusedQKRoPE_, bool usePrecomputedQKRoPE_, bool useQKVRoPEGemm_, int ropeBatchGroup_, bool useFlashAttention_, bool useOutProjGemm_, bool shareModelWeights_)
    : name(desc->name),
      numHeads(desc->numHeads),
      numKVHeads(desc->numKVHeads),
      qHeadDim(desc->qHeadDim),
      vHeadDim(desc->vHeadDim),
      inChannels(desc->qProj.inChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      useFusedResidual(useFusedResidual_),
      useRMSNormOpt(useRMSNormOpt_),
      useFusedQKRoPE(useFusedQKRoPE_),
      usePrecomputedQKRoPE(usePrecomputedQKRoPE_),
      useQKVRoPEGemm(useQKVRoPEGemm_),
      ropeBatchGroup(ropeBatchGroup_),
      useFlashAttention(useFlashAttention_),
      useOutProjGemm(useOutProjGemm_),
      preLN(&desc->preLN, useFP16, useRMSNormOpt_),
      qProj(&desc->qProj, useFP16, shareModelWeights_),
      kProj(&desc->kProj, useFP16, shareModelWeights_),
      vProj(&desc->vProj, useFP16, shareModelWeights_),
      outProj(&desc->outProj, useFP16, shareModelWeights_),
      qkvWeightsBuf(NULL),
      useQKVBatched(false),
      ropeCosTable(NULL),
      ropeSinTable(NULL),
      ropeFreqsBuf(NULL),
      ropeCosSinTable(NULL),
      ropeNumPairs(desc->qHeadDim / 2),
      sdpaCache(std::make_shared<Sm89SDPACache>(ctx->cudnn))
#ifdef KATAGO_ENABLE_SM89_OUTPROJ_GEMM
      , outProjGemm(nullptr)
#endif
#ifdef KATAGO_ENABLE_SM89_QKV_ROPE_GEMM
      , qkvRopeGemm(nullptr)
#endif
  {
    if(!useNHWC)
      throw StringError("Sm89AttentionBlock: transformer blocks require NHWC");
#ifdef KATAGO_ENABLE_SM89_OUTPROJ_GEMM
    if(useOutProjGemm && useFP16 && useFusedResidual_)
      outProjGemm = std::make_unique<Sm89Backend::Sm89OutProjGemmB13>((const half*)outProj.matBuf);
#endif
    int qTotalDim = numHeads * qHeadDim;
    int kTotalDim = numKVHeads * qHeadDim;
    int vTotalDim = numKVHeads * vHeadDim;
    if(useWideQKV && useFP16 && qTotalDim == kTotalDim && kTotalDim == vTotalDim) {
      int outTotal = qTotalDim + kTotalDim + vTotalDim;
      MatMulLayerDesc wideDesc;
      wideDesc.name = name + ":wideQKV";
      wideDesc.inChannels = inChannels;
      wideDesc.outChannels = outTotal;
      wideDesc.weights.reserve((size_t)outTotal * inChannels);
      wideDesc.weights.insert(wideDesc.weights.end(), desc->qProj.weights.begin(), desc->qProj.weights.end());
      wideDesc.weights.insert(wideDesc.weights.end(), desc->kProj.weights.begin(), desc->kProj.weights.end());
      wideDesc.weights.insert(wideDesc.weights.end(), desc->vProj.weights.begin(), desc->vProj.weights.end());
      CudaUtils::mallocAndCopyToDevice(name + ":qkvW", wideDesc.weights, qkvWeightsBuf, useFP16);
      useQKVBatched = true;
    }
    if(desc->useRope) {
      if(desc->learnableRope) {
        testAssert((int)desc->ropeFreqs.size() == (size_t)desc->numKVHeads * ropeNumPairs * 2);
        void* freqsVoid = NULL;
        CudaUtils::mallocAndCopyToDevice(name + ":ropeFreqs", desc->ropeFreqs, freqsVoid, false);
        ropeFreqsBuf = (float*)freqsVoid;
        if(useFusedQKRoPE && usePrecomputedQKRoPE && useFP16) {
          int seqLen = nnXLen * nnYLen;
          size_t tableElts = (size_t)seqLen * numHeads * ropeNumPairs;
          CUDA_ERR(name.c_str(),cudaMalloc(&ropeCosSinTable, tableElts * sizeof(float2)));
          if(!sm89PrecomputeRoPECosSin(
               ropeFreqsBuf, ropeCosSinTable,
               seqLen, numHeads, numKVHeads, qHeadDim, nnXLen, ctx->stream
             )) {
            cudaFree(ropeCosSinTable);
            ropeCosSinTable = NULL;
          }
        }
      }
      else {
        int seqLen = nnXLen * nnYLen;
        vector<float> cosTableData, sinTableData;
        desc->computeRopeCosSin(nnXLen, nnYLen, seqLen, cosTableData, sinTableData);
        CudaUtils::mallocAndCopyToDevice(name + ":ropeCos", cosTableData, ropeCosTable, useFP16);
        CudaUtils::mallocAndCopyToDevice(name + ":ropeSin", sinTableData, ropeSinTable, useFP16);
      }
    }
#ifdef KATAGO_ENABLE_SM89_QKV_ROPE_GEMM
    if(useQKVRoPEGemm && useQKVBatched && ropeFreqsBuf != NULL &&
       nnXLen == 19 && nnYLen == 19 && inChannels == 384 &&
      numHeads == 12 && numKVHeads == 12 && qHeadDim == 32 && vHeadDim == 32) {
      qkvRopeGemm = std::make_unique<Sm89Backend::Sm89QKVRoPEGemmB13>(
        (const half*)qkvWeightsBuf, ropeFreqsBuf
      );
    }
#endif
  }

  ~Sm89AttentionBlock() {
    if(ropeCosTable != NULL) cudaFree(ropeCosTable);
    if(ropeSinTable != NULL) cudaFree(ropeSinTable);
    if(ropeFreqsBuf != NULL) cudaFree(ropeFreqsBuf);
    if(ropeCosSinTable != NULL) cudaFree(ropeCosSinTable);
    if(qkvWeightsBuf != NULL) cudaFree(qkvWeightsBuf);
  }

  void apply(
    Sm89Ctx* ctx,
    Sm89Scratch* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;
    int seqLen = nnXLen * nnYLen;
    int qTotalDim = numHeads * qHeadDim;
    int kTotalDim = numKVHeads * qHeadDim;
    int vTotalDim = numKVHeads * vHeadDim;
    int matBatchSize = batchSize * seqLen;
    size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);

    preLN.apply(batchSize, seqLen, trunkBuf, trunkScratchBuf, maskBuf);

    SizedBuf<void*> qkvBuf(&scratch->allocator, (size_t)(qTotalDim + kTotalDim + vTotalDim) * matBatchSize * bytesPerElt);
    void* qBuf = qkvBuf.buf;
    void* kBuf = (char*)qkvBuf.buf + (size_t)qTotalDim * matBatchSize * bytesPerElt;
    void* vBuf = (char*)qkvBuf.buf + (size_t)(qTotalDim + kTotalDim) * matBatchSize * bytesPerElt;
    bool usedQKVRoPEGemm = false;
#ifdef KATAGO_ENABLE_SM89_QKV_ROPE_GEMM
    if(qkvRopeGemm != nullptr) {
      usedQKVRoPEGemm = qkvRopeGemm->apply(
        (const half*)trunkScratchBuf, (half*)qkvBuf.buf,
        batchSize, seqLen, inChannels, qTotalDim, numHeads, qHeadDim, ctx->stream
      );
    }
#endif
    if(useQKVBatched) {
      if(!usedQKVRoPEGemm) {
        const half alpha = __float2half(1.0f);
        const half beta = __float2half(0.0f);
        CUBLAS_ERR(name.c_str(),cublasHgemmStridedBatched(
          ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
          qTotalDim, matBatchSize, inChannels,
          &alpha,
          (const half*)qkvWeightsBuf, qTotalDim, (int64_t)qTotalDim * inChannels,
          (const half*)trunkScratchBuf, inChannels, 0,
          &beta,
          (half*)qkvBuf.buf, qTotalDim, (int64_t)qTotalDim * matBatchSize,
          3
        ));
      }
    }
    else {
      qProj.apply(ctx, scratch, matBatchSize, trunkScratchBuf, qBuf);
      kProj.apply(ctx, scratch, matBatchSize, trunkScratchBuf, kBuf);
      vProj.apply(ctx, scratch, matBatchSize, trunkScratchBuf, vBuf);
    }

    if(ropeFreqsBuf != NULL) {
      bool usedFusedRoPE = usedQKVRoPEGemm;
      if(!usedFusedRoPE && useFusedQKRoPE && usingFP16) {
        if(ropeCosSinTable != NULL) {
          usedFusedRoPE = sm89ApplyRoPEQKHalfPrecomputed(
            (half*)qBuf, (half*)kBuf, ropeCosSinTable,
            batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ctx->stream
          );
        }
        else {
          if(ropeBatchGroup > 1) {
            usedFusedRoPE = sm89ApplyRoPEQKHalfBatchGrouped(
              (half*)qBuf, (half*)kBuf, ropeFreqsBuf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, nnXLen,
              ropeBatchGroup, ctx->stream
            );
          }
          if(!usedFusedRoPE) {
            usedFusedRoPE = sm89ApplyRoPEQKHalf(
              (half*)qBuf, (half*)kBuf, ropeFreqsBuf,
              batchSize, seqLen, numHeads, numKVHeads, qHeadDim, nnXLen, ctx->stream
            );
          }
        }
      }
      if(usedFusedRoPE) {
        // handled
      }
      else if(!usingFP16) {
        customCudaApplyRoPELearnableRecompute((float*)qBuf, ropeFreqsBuf, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen);
        customCudaApplyRoPELearnableRecompute((float*)kBuf, ropeFreqsBuf, batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen);
      }
      else {
        customCudaApplyRoPELearnableRecompute((half*)qBuf, ropeFreqsBuf, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen);
        customCudaApplyRoPELearnableRecompute((half*)kBuf, ropeFreqsBuf, batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, nnXLen);
      }
    }
    else if(ropeCosTable != NULL) {
      if(!usingFP16) {
        customCudaApplyRoPE((float*)qBuf, (const float*)ropeCosTable, (const float*)ropeSinTable, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, false);
        customCudaApplyRoPE((float*)kBuf, (const float*)ropeCosTable, (const float*)ropeSinTable, batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, false);
      }
      else {
        customCudaApplyRoPE((half*)qBuf, (const half*)ropeCosTable, (const half*)ropeSinTable, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, ropeNumPairs, false);
        customCudaApplyRoPE((half*)kBuf, (const half*)ropeCosTable, (const half*)ropeSinTable, batchSize, seqLen, numKVHeads, numKVHeads, qHeadDim, ropeNumPairs, false);
      }
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    SizedBuf<void*> attnOutBuf(&scratch->allocator, (size_t)numHeads * vHeadDim * seqLen * batchSize * bytesPerElt);
    bool usedSDPA = false;
#ifdef KATAGO_ENABLE_SM89_FLASH_ATTN
    if(useFlashAttention && usingFP16 && maskBuf == NULL) {
      SizedBuf<void*> lseBuf(&scratch->allocator, sm89FlashAttentionLseBytesB13D32());
      usedSDPA = sm89FlashAttentionB13D32(
        (const half*)qBuf, (const half*)kBuf, (const half*)vBuf, (half*)attnOutBuf.buf,
        (float*)lseBuf.buf, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim, ctx->stream
      );
    }
#endif
#if CUDNN_VERSION >= 8903
    if(!usedSDPA && usingFP16) {
      Sm89SDPAKey key{numHeads, numKVHeads, qHeadDim, vHeadDim, seqLen, batchSize, maskBuf != NULL};
      auto plan = sdpaCache->getPlan(key);
      if(plan != nullptr) {
        std::unordered_map<int64_t, void*> variant_pack = {
          {1, qBuf},
          {2, kBuf},
          {3, vBuf},
          {4, attnOutBuf.buf},
        };
        SizedBuf<void*> biasBuf(&scratch->allocator, maskBuf != NULL ? (size_t)batchSize * seqLen * seqLen * sizeof(half) : 1);
        if(maskBuf != NULL) {
          customCudaMaskToAttnBiasFull((const half*)maskBuf, (half*)biasBuf.buf, batchSize, seqLen);
          variant_pack[5] = biasBuf.buf;
        }
        SizedBuf<void*> sdpaWs(&scratch->allocator, (size_t)plan->workspaceBytes);
        auto status = plan->graph->execute(ctx->cudnn, variant_pack, sdpaWs.buf);
        if(!status.is_bad())
          usedSDPA = true;
      }
    }
#endif
    if(!usedSDPA) {
      if(!usingFP16)
        customCudaFlashAttention((const float*)qBuf, (const float*)kBuf, (const float*)vBuf, (const float*)maskBuf, (float*)attnOutBuf.buf, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim);
      else
        customCudaFlashAttention((const half*)qBuf, (const half*)kBuf, (const half*)vBuf, (const half*)maskBuf, (half*)attnOutBuf.buf, batchSize, seqLen, numHeads, numKVHeads, qHeadDim, vHeadDim);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    bool usedOutProjGemm = false;
#ifdef KATAGO_ENABLE_SM89_OUTPROJ_GEMM
    if(outProjGemm != nullptr && usingFP16) {
      usedOutProjGemm = outProjGemm->applyAccumulate(
        (const half*)attnOutBuf.buf, (half*)trunkBuf,
        batchSize, seqLen, numHeads * vHeadDim, inChannels, ctx->stream
      );
    }
#endif
    if(useFusedResidual && usingFP16) {
      if(!usedOutProjGemm)
        outProj.applyAccumulate(ctx, scratch, matBatchSize, attnOutBuf.buf, trunkBuf);
      if(maskBuf != NULL)
        sm89MaskZeroNHWC((half*)trunkBuf, (const half*)maskBuf, batchSize, seqLen, inChannels, ctx->stream);
    }
    else {
      outProj.apply(ctx, scratch, matBatchSize, attnOutBuf.buf, trunkScratchBuf);
      if(!usingFP16)
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, inChannels);
      else
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, inChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
  }
};

struct Sm89FFNBlock {
  const string name;
  const int numChannels;
  const int ffnChannels;
  const int nnXLen;
  const int nnYLen;
  const bool usingFP16;
  const bool usingNHWC;
  const bool useFusedResidual;
  const bool useRMSNormOpt;
  const bool useDualGemmSwiGLU;
  const bool useLinear2Gemm;
  const Sm89TransformerRMSNorm preLN;
  const Sm89MatMul linear1;
  const Sm89MatMul linearGate;
  const Sm89MatMul linear2;
  void* ffnWeightsBuf;
  bool useFFNBatched;
#ifdef KATAGO_ENABLE_SM89_DUAL_GEMM
  std::unique_ptr<Sm89Backend::Sm89DualGemmSwiGLUB13> dualGemmSwiGLU;
#endif
#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
  std::unique_ptr<Sm89Backend::Sm89Linear2GemmB13> linear2Gemm;
  std::unique_ptr<Sm89Backend::Sm89Linear2BnGemmB13> linear2PostBnGemm;
#endif

  Sm89FFNBlock() = delete;
  Sm89FFNBlock(const Sm89FFNBlock&) = delete;
  Sm89FFNBlock& operator=(const Sm89FFNBlock&) = delete;

  Sm89FFNBlock(
    const TransformerFFNDesc* desc,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    bool useWideFFN,
    bool useFusedResidual_,
    bool useRMSNormOpt_,
    bool useDualGemmSwiGLU_,
    bool useLinear2Gemm_,
    bool useLinear2PostBNSilu_,
    const Sm89BatchNorm* followingBN,
    bool shareModelWeights_
  )
    : name(desc->name),
      numChannels(desc->numChannels),
      ffnChannels(desc->ffnChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      useFusedResidual(useFusedResidual_),
      useRMSNormOpt(useRMSNormOpt_),
      useDualGemmSwiGLU(useDualGemmSwiGLU_),
      useLinear2Gemm(useLinear2Gemm_),
      preLN(&desc->preLN, useFP16, useRMSNormOpt_),
      linear1(&desc->linear1, useFP16, shareModelWeights_),
      linearGate(&desc->linearGate, useFP16, shareModelWeights_),
      linear2(&desc->linear2, useFP16, shareModelWeights_),
      ffnWeightsBuf(NULL),
      useFFNBatched(false)
#ifdef KATAGO_ENABLE_SM89_DUAL_GEMM
      , dualGemmSwiGLU(nullptr)
#endif
#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
      , linear2Gemm(nullptr)
      , linear2PostBnGemm(nullptr)
#endif
  {
    if(!desc->useSwiGLU)
      throw StringError("Sm89FFNBlock: non-SwiGLU FFN not supported");
    if(!useNHWC)
      throw StringError("Sm89FFNBlock: transformer blocks require NHWC");
    if(useWideFFN && useFP16) {
      MatMulLayerDesc wideDesc;
      wideDesc.name = name + ":wideLinear1Gate";
      wideDesc.inChannels = numChannels;
      wideDesc.outChannels = ffnChannels * 2;
      wideDesc.weights.reserve((size_t)ffnChannels * 2 * numChannels);
      wideDesc.weights.insert(wideDesc.weights.end(), desc->linear1.weights.begin(), desc->linear1.weights.end());
      wideDesc.weights.insert(wideDesc.weights.end(), desc->linearGate.weights.begin(), desc->linearGate.weights.end());
      CudaUtils::mallocAndCopyToDevice(name + ":ffnW", wideDesc.weights, ffnWeightsBuf, useFP16);
      useFFNBatched = true;
#ifdef KATAGO_ENABLE_SM89_DUAL_GEMM
      if(useDualGemmSwiGLU)
        dualGemmSwiGLU = std::make_unique<Sm89Backend::Sm89DualGemmSwiGLUB13>((const half*)ffnWeightsBuf);
#endif
    }
#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
    if(useLinear2Gemm && useFP16 && useFusedResidual_) {
      linear2Gemm = std::make_unique<Sm89Backend::Sm89Linear2GemmB13>((const half*)linear2.matBuf);
      if(
        useLinear2PostBNSilu_ && followingBN != nullptr &&
        followingBN->usingFP16 && followingBN->usingNHWC &&
        followingBN->numChannels == numChannels &&
        followingBN->activation == ACTIVATION_SILU &&
        followingBN->mergedScaleBuf != nullptr && followingBN->mergedBiasBuf != nullptr
      ) {
        linear2PostBnGemm = std::make_unique<Sm89Backend::Sm89Linear2BnGemmB13>(
          (const half*)linear2.matBuf,
          (const half*)followingBN->mergedScaleBuf,
          (const half*)followingBN->mergedBiasBuf
        );
      }
    }
#else
    (void)useLinear2PostBNSilu_;
    (void)followingBN;
#endif
  }

  ~Sm89FFNBlock() {
    if(ffnWeightsBuf != NULL)
      cudaFree(ffnWeightsBuf);
  }

  bool apply(
    Sm89Ctx* ctx,
    Sm89Scratch* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf
  ) const {
    int seqLen = nnXLen * nnYLen;
    int matBatchSize = batchSize * seqLen;
    size_t bytesPerElt = usingFP16 ? sizeof(half) : sizeof(float);
    preLN.apply(batchSize, seqLen, trunkBuf, trunkScratchBuf, maskBuf);

    SizedBuf<void*> ffnGateBuf(&scratch->allocator, (size_t)ffnChannels * 2 * matBatchSize * bytesPerElt);
    bool usedDualGemmSwiGLU = false;
#ifdef KATAGO_ENABLE_SM89_DUAL_GEMM
    if(dualGemmSwiGLU != nullptr && usingFP16) {
      usedDualGemmSwiGLU = dualGemmSwiGLU->apply(
        (const half*)trunkScratchBuf, (half*)ffnGateBuf.buf,
        batchSize, seqLen, numChannels, ffnChannels, ctx->stream
      );
    }
#endif
    if(!usedDualGemmSwiGLU && useFFNBatched) {
      const half alpha = __float2half(1.0f);
      const half beta = __float2half(0.0f);
      CUBLAS_ERR(name.c_str(),cublasHgemmStridedBatched(
        ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        ffnChannels, matBatchSize, numChannels,
        &alpha,
        (const half*)ffnWeightsBuf, ffnChannels, (int64_t)ffnChannels * numChannels,
        (const half*)trunkScratchBuf, numChannels, 0,
        &beta,
        (half*)ffnGateBuf.buf, ffnChannels, (int64_t)ffnChannels * matBatchSize,
        2
      ));
    }
    else if(!usedDualGemmSwiGLU) {
      linear1.apply(ctx, scratch, matBatchSize, trunkScratchBuf, ffnGateBuf.buf);
      linearGate.apply(ctx, scratch, matBatchSize, trunkScratchBuf, (char*)ffnGateBuf.buf + (size_t)ffnChannels * matBatchSize * bytesPerElt);
    }
    void* ffnBuf = ffnGateBuf.buf;
    void* gateBuf = (char*)ffnGateBuf.buf + (size_t)ffnChannels * matBatchSize * bytesPerElt;

    int totalSize = (int)((size_t)ffnChannels * matBatchSize);
    if(!usedDualGemmSwiGLU) {
      if(!usingFP16)
        customCudaSwiGLU((const float*)ffnBuf, (const float*)gateBuf, (float*)ffnBuf, totalSize);
      else
        customCudaSwiGLU((const half*)ffnBuf, (const half*)gateBuf, (half*)ffnBuf, totalSize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    bool usedLinear2PostBN = false;
#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
    if(linear2PostBnGemm != nullptr && usingFP16 && maskBuf == NULL) {
      usedLinear2PostBN = linear2PostBnGemm->applyAccumulateAndActivate(
        (const half*)ffnBuf, (half*)trunkBuf, (half*)trunkScratchBuf,
        batchSize, seqLen, ffnChannels, numChannels, ctx->stream
      );
    }
#endif
    bool usedLinear2Gemm = usedLinear2PostBN;
#ifdef KATAGO_ENABLE_SM89_LINEAR2_GEMM
    if(!usedLinear2PostBN && linear2Gemm != nullptr && usingFP16) {
      usedLinear2Gemm = linear2Gemm->applyAccumulate(
        (const half*)ffnBuf, (half*)trunkBuf,
        batchSize, seqLen, ffnChannels, numChannels, ctx->stream
      );
    }
#endif
    if(useFusedResidual && usingFP16) {
      if(!usedLinear2Gemm)
        linear2.applyAccumulate(ctx, scratch, matBatchSize, ffnBuf, trunkBuf);
      if(maskBuf != NULL)
        sm89MaskZeroNHWC((half*)trunkBuf, (const half*)maskBuf, batchSize, seqLen, numChannels, ctx->stream);
    }
    else {
      linear2.apply(ctx, scratch, matBatchSize, ffnBuf, trunkScratchBuf);
      if(!usingFP16)
        customCudaMaskedResidualAddNHWC((float*)trunkBuf, (const float*)trunkScratchBuf, (const float*)maskBuf, batchSize, seqLen, numChannels);
      else
        customCudaMaskedResidualAddNHWC((half*)trunkBuf, (const half*)trunkScratchBuf, (const half*)maskBuf, batchSize, seqLen, numChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    return usedLinear2PostBN;
  }
};

// --------------------------------------------------------------------------------------
// Nested bottleneck block + trunk

struct Sm89NestedBlock {
  const string name;
  const int nnXLen;
  const int nnYLen;
  const int maxBatchSize;
  const bool usingFP16;
  const bool usingNHWC;
  const bool useWideQKV;
  const bool useWideFFN;
  const bool usePersistingL2Trunk;
  const float persistingL2TrunkHitRatio;
  const bool usePersistingL2Inner;
  const float persistingL2InnerHitRatio;
  const Sm89BatchNorm preBN;
  const Sm89Conv preConv;
  const Sm89BatchNorm postBN;
  const Sm89Conv postConv;
  vector<std::function<bool(Sm89Ctx*, Sm89Scratch*, int, void*, void*, void*, void*, size_t)>> innerBlocks;
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
  std::unique_ptr<Sm89Backend::Sm89PreConvGemmB13> preConvGemm;
#endif
#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
  std::unique_ptr<Sm89Backend::Sm89PostConvGemmB13> postConvGemm;
  std::unique_ptr<Sm89Backend::Sm89PostConvBnGemmB13> postConvBnGemm;
#endif

  Sm89NestedBlock() = delete;
  Sm89NestedBlock(const Sm89NestedBlock&) = delete;
  Sm89NestedBlock& operator=(const Sm89NestedBlock&) = delete;

  Sm89NestedBlock(
    Sm89Ctx* ctx,
    const NestedBottleneckResidualBlockDesc* desc,
    int maxBatchSize,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    bool useWideQKV_,
    bool useWideFFN_,
    bool useFusedResidual_,
    bool useRMSNormOpt_,
    bool useFusedQKRoPE_,
    bool usePrecomputedQKRoPE_,
    bool useQKVRoPEGemm_,
    int ropeBatchGroup_,
    bool useFlashAttention_,
    bool useDualGemmSwiGLU_,
    bool useLinear2Gemm_,
    bool useOutProjGemm_,
    bool usePreConvGemm_,
    bool usePostConvGemm_,
    bool useLinear2PostBNSilu_,
    bool usePersistingL2Trunk_,
    float persistingL2TrunkHitRatio_,
    bool usePersistingL2Inner_,
    float persistingL2InnerHitRatio_,
    bool useScaleBiasSiluVec8_,
    bool useScaleBiasSiluVec4C384_,
    bool shareModelWeights_
  )
    : name(desc->name),
      nnXLen(nnX),
      nnYLen(nnY),
      maxBatchSize(maxBatchSize),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      useWideQKV(useWideQKV_),
      useWideFFN(useWideFFN_),
      usePersistingL2Trunk(usePersistingL2Trunk_),
      persistingL2TrunkHitRatio(persistingL2TrunkHitRatio_),
      usePersistingL2Inner(usePersistingL2Inner_),
      persistingL2InnerHitRatio(persistingL2InnerHitRatio_),
      preBN(&desc->preBN, &desc->preActivation, nnX, nnY, useFP16, useNHWC, useScaleBiasSiluVec8_, useScaleBiasSiluVec4C384_),
      preConv(ctx, &desc->preConv, maxBatchSize, nnX, nnY, useFP16, useNHWC, useNHWC),
      postBN(&desc->postBN, &desc->postActivation, nnX, nnY, useFP16, useNHWC, useScaleBiasSiluVec8_, useScaleBiasSiluVec4C384_),
      postConv(ctx, &desc->postConv, maxBatchSize, nnX, nnY, useFP16, useNHWC, useNHWC)
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
      , preConvGemm(nullptr)
#endif
#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
      , postConvGemm(nullptr)
      , postConvBnGemm(nullptr)
#endif
  {
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
    if(usePreConvGemm_ && useFP16 && preConv.use1x1Matmul && preConv.matmulWeightBuf != nullptr)
      preConvGemm = std::make_unique<Sm89Backend::Sm89PreConvGemmB13>((const half*)preConv.matmulWeightBuf);
#endif
#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
    if(usePostConvGemm_ && useFP16 && postConv.use1x1Matmul && postConv.matmulWeightBuf != nullptr)
      postConvGemm = std::make_unique<Sm89Backend::Sm89PostConvGemmB13>((const half*)postConv.matmulWeightBuf);
#endif
    for(size_t i = 0; i < desc->blocks.size(); i++) {
      int kind = desc->blocks[i].first;
      if(kind == TRANSFORMER_ATTENTION_BLOCK_KIND) {
        auto block = std::make_shared<Sm89AttentionBlock>(
          ctx, (const TransformerAttentionDesc*)desc->blocks[i].second.get(), nnX, nnY,
          useFP16, useNHWC, useWideQKV_, useFusedResidual_, useRMSNormOpt_,
          useFusedQKRoPE_, usePrecomputedQKRoPE_, useQKVRoPEGemm_,
          ropeBatchGroup_, useFlashAttention_,
          useOutProjGemm_, shareModelWeights_
        );
        innerBlocks.push_back([block](Sm89Ctx* ctx, Sm89Scratch* scratch, int batchSize, void* trunkBuf, void* trunkScratchBuf, void* maskBuf, void* workspaceBuf, size_t workspaceBytes) {
          block->apply(ctx, scratch, batchSize, trunkBuf, trunkScratchBuf, maskBuf, workspaceBuf, workspaceBytes);
          return false;
        });
      }
      else if(kind == TRANSFORMER_FFN_BLOCK_KIND) {
        auto block = std::make_shared<Sm89FFNBlock>(
          (const TransformerFFNDesc*)desc->blocks[i].second.get(), nnX, nnY, useFP16,
          useNHWC, useWideFFN_, useFusedResidual_, useRMSNormOpt_, useDualGemmSwiGLU_,
          useLinear2Gemm_, useLinear2PostBNSilu_ && i + 1 == desc->blocks.size(),
          &postBN, shareModelWeights_
        );
        innerBlocks.push_back([block](Sm89Ctx* ctx, Sm89Scratch* scratch, int batchSize, void* trunkBuf, void* trunkScratchBuf, void* maskBuf, void* workspaceBuf, size_t workspaceBytes) {
          (void)workspaceBuf;
          (void)workspaceBytes;
          return block->apply(ctx, scratch, batchSize, trunkBuf, trunkScratchBuf, maskBuf);
        });
      }
      else {
        throw StringError("Sm89NestedBlock: unsupported inner block kind " + Global::intToString(kind));
      }
    }
  }

  void configurePostConvBN(const Sm89BatchNorm& followingBN) {
#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
    if(
      usingFP16 && usingNHWC &&
      postConv.use1x1Matmul && postConv.matmulWeightBuf != nullptr &&
      followingBN.usingFP16 && followingBN.usingNHWC &&
      followingBN.numChannels == postConv.outChannels &&
      followingBN.activation == ACTIVATION_SILU &&
      followingBN.mergedScaleBuf != nullptr && followingBN.mergedBiasBuf != nullptr
    ) {
      postConvBnGemm = std::make_unique<Sm89Backend::Sm89PostConvBnGemmB13>(
        (const half*)postConv.matmulWeightBuf,
        (const half*)followingBN.mergedScaleBuf,
        (const half*)followingBN.mergedBiasBuf
      );
    }
#else
    (void)followingBN;
#endif
  }

  bool apply(
    Sm89Ctx* ctx,
    Sm89Scratch* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    bool preBNReady
  ) const {
    int xySize = nnXLen * nnYLen;
    const size_t midBytes =
      scratch->getBufSizeXY(preConv.outChannels, maxBatchSize, xySize, usingFP16);
    SizedBuf<void*> mid(&scratch->allocator, midBytes);
    SizedBuf<void*> midScratch(&scratch->allocator, midBytes);

    if(!preBNReady)
      preBN.apply(batchSize, trunkBuf, maskBuf, trunkScratchBuf);
    if(usePersistingL2Inner)
      setPersistingL2Window(
        ctx->stream, mid.buf, midBytes, persistingL2InnerHitRatio
      );
    bool usedPreConvGemm = false;
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
    if(preConvGemm != nullptr && usingFP16) {
      usedPreConvGemm = preConvGemm->apply(
        (const half*)trunkScratchBuf, (half*)mid.buf,
        batchSize, xySize, preConv.inChannels, preConv.outChannels, ctx->stream
      );
    }
#endif
    if(!usedPreConvGemm)
      preConv.apply(ctx, batchSize, false, trunkScratchBuf, mid.buf, workspaceBuf, workspaceBytes);

    bool postBNReady = false;
    for(const auto& fn : innerBlocks)
      postBNReady = fn(
        ctx, scratch, batchSize, mid.buf, midScratch.buf,
        maskBuf, workspaceBuf, workspaceBytes
      );

    if(!postBNReady)
      postBN.apply(batchSize, mid.buf, maskBuf, midScratch.buf);
    if(usePersistingL2Inner) {
      if(usePersistingL2Trunk) {
        const size_t trunkBytes = scratch->getBufSizeXY(
          preConv.inChannels, maxBatchSize, xySize, usingFP16
        );
        setPersistingL2Window(
          ctx->stream, trunkBuf, trunkBytes, persistingL2TrunkHitRatio
        );
      }
      else {
        clearPersistingL2Window(ctx->stream);
      }
    }
    bool usedPostConvBN = false;
#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
    if(postConvBnGemm != nullptr && usingFP16) {
      usedPostConvBN = postConvBnGemm->applyAccumulateAndActivate(
        (const half*)midScratch.buf, (half*)trunkBuf, (half*)trunkScratchBuf,
        batchSize, xySize, postConv.inChannels, postConv.outChannels, ctx->stream
      );
    }
#endif
    bool usedPostConvGemm = usedPostConvBN;
#ifdef KATAGO_ENABLE_SM89_POSTCONV_GEMM
    if(!usedPostConvBN && postConvGemm != nullptr && usingFP16) {
      usedPostConvGemm = postConvGemm->applyAccumulate(
        (const half*)midScratch.buf, (half*)trunkBuf,
        batchSize, xySize, postConv.inChannels, postConv.outChannels, ctx->stream
      );
    }
#endif
    if(!usedPostConvGemm)
      postConv.apply(ctx, batchSize, true, midScratch.buf, trunkBuf, workspaceBuf, workspaceBytes);
    return usedPostConvBN;
  }

};

struct Sm89Trunk {
  const string name;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;
  const int maxBatchSize;
  const bool usingFP16;
  const bool usingNHWC;
  const bool usePersistingL2Trunk;
  const float persistingL2TrunkHitRatio;
  const bool useInitialGlobalMatMulAdd;
  const Sm89Conv initialConv;
  const Sm89MatMul initialMatMul;
  vector<shared_ptr<Sm89NestedBlock>> blocks;
  unique_ptr<Sm89BatchNorm> trunkTipBN;

  Sm89Trunk() = delete;
  Sm89Trunk(const Sm89Trunk&) = delete;
  Sm89Trunk& operator=(const Sm89Trunk&) = delete;

  Sm89Trunk(
    Sm89Ctx* ctx,
    const TrunkDesc* desc,
    int maxBatchSize_,
    int nnX,
    int nnY,
    bool useFP16,
    bool useNHWC,
    bool useWideQKV_,
    bool useWideFFN_,
    bool useFusedResidual_,
    bool useRMSNormOpt_,
    bool useFusedQKRoPE_,
    bool usePrecomputedQKRoPE_,
    bool useQKVRoPEGemm_,
    int ropeBatchGroup_,
    bool useFlashAttention_,
    bool useDualGemmSwiGLU_,
    bool useLinear2Gemm_,
    bool useOutProjGemm_,
    bool usePreConvGemm_,
    bool usePostConvGemm_,
    bool usePostConvBNSilu_,
    bool useLinear2PostBNSilu_,
    bool usePersistingL2Trunk_,
    float persistingL2TrunkHitRatio_,
    bool usePersistingL2Inner_,
    float persistingL2InnerHitRatio_,
    bool useScaleBiasSiluVec8_,
    bool useScaleBiasSiluVec4C384_,
    bool useInitialConvFrontend_,
    bool useInitialGlobalMatMulAdd_,
    bool shareModelWeights_
  )
    : name(desc->name),
      trunkNumChannels(desc->trunkNumChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      maxBatchSize(maxBatchSize_),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      usePersistingL2Trunk(usePersistingL2Trunk_),
      persistingL2TrunkHitRatio(persistingL2TrunkHitRatio_),
      useInitialGlobalMatMulAdd(useInitialGlobalMatMulAdd_),
      initialConv(
        ctx, &desc->initialConv, maxBatchSize_, nnX, nnY,
        useFP16, useNHWC, useNHWC, useInitialConvFrontend_
      ),
      initialMatMul(&desc->initialMatMul, useFP16, shareModelWeights_)
  {
    if(desc->metaEncoderVersion > 0)
      throw StringError("Sm89Trunk: SGF metadata encoder not supported yet");
    for(size_t i = 0; i < desc->blocks.size(); i++) {
      if(desc->blocks[i].first != NESTED_BOTTLENECK_BLOCK_KIND)
        throw StringError("Sm89Trunk: only nested-bottleneck trunk blocks are supported");
      blocks.push_back(std::make_shared<Sm89NestedBlock>(
        ctx,
        (const NestedBottleneckResidualBlockDesc*)desc->blocks[i].second.get(),
        maxBatchSize_, nnX, nnY, useFP16, useNHWC, useWideQKV_, useWideFFN_,
        useFusedResidual_, useRMSNormOpt_, useFusedQKRoPE_, usePrecomputedQKRoPE_,
        useQKVRoPEGemm_, ropeBatchGroup_, useFlashAttention_,
        useDualGemmSwiGLU_, useLinear2Gemm_, useOutProjGemm_, usePreConvGemm_,
        usePostConvGemm_, useLinear2PostBNSilu_, usePersistingL2Trunk_,
        persistingL2TrunkHitRatio_,
        usePersistingL2Inner_, persistingL2InnerHitRatio_, useScaleBiasSiluVec8_,
        useScaleBiasSiluVec4C384_,
        shareModelWeights_
      ));
    }
    if(desc->trunkNormKind != TRUNK_NORM_KIND_STANDARD)
      throw StringError("Sm89Trunk: only standard trunk BN is supported");
    trunkTipBN = std::make_unique<Sm89BatchNorm>(
      &desc->trunkTipBN, &desc->trunkTipActivation,
      nnX, nnY, useFP16, useNHWC, useScaleBiasSiluVec8_, useScaleBiasSiluVec4C384_
    );
    if(usePostConvBNSilu_) {
      for(size_t i = 0; i < blocks.size(); i++) {
        const Sm89BatchNorm& followingBN =
          i + 1 < blocks.size() ? blocks[i + 1]->preBN : *trunkTipBN;
        blocks[i]->configurePostConvBN(followingBN);
      }
    }
  }

  void apply(
    Sm89Ctx* ctx,
    Sm89Scratch* scratch,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* maskBuf,
    void* trunkBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    int xySize = nnXLen * nnYLen;
    const size_t trunkScratchBytes =
      scratch->getBufSizeXY(trunkNumChannels, maxBatchSize, xySize, usingFP16);
    SizedBuf<void*> trunkScratch(&scratch->allocator, trunkScratchBytes);
    if(usePersistingL2Trunk)
      setPersistingL2Window(
        ctx->stream, trunkScratch.buf, trunkScratchBytes, persistingL2TrunkHitRatio
      );

    initialConv.apply(ctx, batchSize, false, inputBuf, trunkScratch.buf, workspaceBuf, workspaceBytes);
    bool fusedInitialGlobal = false;
    if(useInitialGlobalMatMulAdd && usingFP16 && usingNHWC) {
      fusedInitialGlobal = sm89InitialGlobalMatMulAddB13(
        (const half*)inputGlobalBuf, (const half*)initialMatMul.matBuf,
        (half*)trunkScratch.buf, batchSize, xySize,
        initialMatMul.inChannels, initialMatMul.outChannels, ctx->stream
      );
    }
    if(!fusedInitialGlobal) {
      initialMatMul.apply(ctx, scratch, batchSize, inputGlobalBuf, trunkBuf);
      if(!usingFP16)
        customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf, (const float*)trunkBuf, batchSize, xySize, trunkNumChannels);
      else
        customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf, (const half*)trunkBuf, batchSize, xySize, trunkNumChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    // Mirror official buffer flip: blocks write into trunkScratch and use trunkBuf as temp.
    bool preBNReady = false;
    for(const auto& block : blocks)
      preBNReady = block->apply(
        ctx, scratch, batchSize, trunkScratch.buf, trunkBuf, maskBuf,
        workspaceBuf, workspaceBytes, preBNReady
      );

    if(!preBNReady)
      trunkTipBN->apply(batchSize, trunkScratch.buf, maskBuf, trunkBuf);
    if(usePersistingL2Trunk)
      clearPersistingL2Window(ctx->stream);
  }
};

// --------------------------------------------------------------------------------------
// Policy / value heads

struct Sm89WideHeadProjection {
  bool available;
  void* weightBuf;
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
  std::unique_ptr<Sm89Backend::Sm89PreConvGemmB13> gemm;
#endif

  Sm89WideHeadProjection(
    const PolicyHeadDesc* policy,
    const ValueHeadDesc* value,
    bool useFP16,
    bool enabled
  )
    : available(false), weightBuf(nullptr)
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
      , gemm(nullptr)
#endif
  {
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
    const ConvLayerDesc* convs[3] = {&policy->p1Conv, &policy->g1Conv, &value->v1Conv};
    const int offsets[3] = {0, 96, 192};
    const int expectedOutChannels[3] = {96, 96, 192};
    if(!enabled || !useFP16)
      return;
    for(int i = 0; i < 3; i++) {
      const ConvLayerDesc& conv = *convs[i];
      if(
        conv.convXSize != 1 || conv.convYSize != 1 ||
        conv.inChannels != 768 || conv.outChannels != expectedOutChannels[i]
      )
        return;
    }

    vector<float> weights((size_t)768 * 384);
    for(int i = 0; i < 3; i++) {
      const ConvLayerDesc& conv = *convs[i];
      for(int ic = 0; ic < 768; ic++) {
        for(int oc = 0; oc < conv.outChannels; oc++) {
          weights[(size_t)ic * 384 + offsets[i] + oc] =
            conv.weights[(size_t)oc * 768 + ic];
        }
      }
    }
    CudaUtils::mallocAndCopyToDevice("wideHeadProjection", weights, weightBuf, true);
    gemm = std::make_unique<Sm89Backend::Sm89PreConvGemmB13>((const half*)weightBuf);
    available = true;
#else
    (void)policy;
    (void)value;
    (void)useFP16;
    (void)enabled;
#endif
  }

  ~Sm89WideHeadProjection() {
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
    gemm.reset();
#endif
    if(weightBuf != nullptr)
      cudaFree(weightBuf);
  }

  bool apply(const half* input, half* output, int batchSize, int xySize, cudaStream_t stream) {
#ifdef KATAGO_ENABLE_SM89_PRECONV_GEMM
    return available && gemm != nullptr &&
      gemm->apply(input, output, batchSize, xySize, 768, 384, stream);
#else
    (void)input;
    (void)output;
    (void)batchSize;
    (void)xySize;
    (void)stream;
    return false;
#endif
  }
};

struct Sm89PolicyHead {
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int maxBatchSize;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool useFusedPolicyP1;
  const bool useHeadBNHalfToFloat;
  const Sm89Conv p1Conv;
  const Sm89Conv g1Conv;
  const Sm89BatchNorm g1BN;
  const Sm89MatMul gpoolToBiasMul;
  const Sm89BatchNorm p1BN;
  const Sm89Conv p2Conv;
  const Sm89MatMul gpoolToPassMul;
  const Sm89MatBias gpoolToPassBias;
  const Sm89MatMul gpoolToPassMul2;

  Sm89PolicyHead() = delete;
  Sm89PolicyHead(const Sm89PolicyHead&) = delete;
  Sm89PolicyHead& operator=(const Sm89PolicyHead&) = delete;

  Sm89PolicyHead(Sm89Ctx* ctx, const PolicyHeadDesc* desc, int maxBatchSize, int nnX, int nnY, bool useFP16, bool useNHWC, bool useFusedPolicyP1_, bool useHeadBNHalfToFloat_, bool shareModelWeights)
    : modelVersion(desc->modelVersion),
      nnXLen(nnX),
      nnYLen(nnY),
      maxBatchSize(maxBatchSize),
      p1Channels(desc->p1Conv.outChannels),
      g1Channels(desc->g1Conv.outChannels),
      p2Channels(desc->p2Conv.outChannels),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      useFusedPolicyP1(useFusedPolicyP1_),
      useHeadBNHalfToFloat(useHeadBNHalfToFloat_),
      p1Conv(ctx, &desc->p1Conv, maxBatchSize, nnX, nnY, useFP16, useNHWC, useNHWC),
      g1Conv(ctx, &desc->g1Conv, maxBatchSize, nnX, nnY, useFP16, useNHWC, useNHWC),
      g1BN(&desc->g1BN, &desc->g1Activation, nnX, nnY, useFP16, useNHWC),
      gpoolToBiasMul(&desc->gpoolToBiasMul, false, shareModelWeights),
      p1BN(&desc->p1BN, &desc->p1Activation, nnX, nnY, false, useNHWC),
      p2Conv(ctx, &desc->p2Conv, maxBatchSize, nnX, nnY, false, useNHWC, useNHWC),
      gpoolToPassMul(&desc->gpoolToPassMul, false, shareModelWeights),
      gpoolToPassBias(&desc->gpoolToPassBias, false, desc->passActivation.activation),
      gpoolToPassMul2(&desc->gpoolToPassMul2, false, shareModelWeights)
  {}

  void apply(
    Sm89Ctx* ctx,
    Sm89Scratch* scratch,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    const half* wideHeadBuf,
    float* policyPassBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    int xySize = nnXLen * nnYLen;
    SizedBuf<void*> p1Out(&scratch->allocator, scratch->getBufSizeXYFloat(p1Channels, maxBatchSize, xySize));
    SizedBuf<void*> p1Out2(&scratch->allocator, scratch->getBufSizeXYFloat(p1Channels, maxBatchSize, xySize));
    SizedBuf<void*> g1Out(&scratch->allocator, scratch->getBufSizeXY(g1Channels, maxBatchSize, xySize, usingFP16));
    SizedBuf<void*> g1Out2(&scratch->allocator, scratch->getBufSizeXY(g1Channels, maxBatchSize, xySize, usingFP16));
    SizedBuf<void*> g1Concat(&scratch->allocator, scratch->getBufSizeFloat(g1Channels * 3, maxBatchSize));
    SizedBuf<void*> g1Bias(&scratch->allocator, scratch->getBufSizeFloat(p1Channels, maxBatchSize));
    SizedBuf<void*> p1Pass(&scratch->allocator, scratch->getBufSizeFloat(p1Channels, maxBatchSize));

    const bool useWideHead = wideHeadBuf != nullptr;
    if(!useWideHead) {
      p1Conv.apply(ctx, batchSize, false, trunkBuf, p1Out.buf, workspaceBuf, workspaceBytes);
      g1Conv.apply(ctx, batchSize, false, trunkBuf, g1Out.buf, workspaceBuf, workspaceBytes);
    }

    if(!usingFP16) {
      g1BN.apply(batchSize, g1Out.buf, maskBuf, g1Out2.buf);
      customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf, (float*)g1Concat.buf, batchSize, xySize, g1Channels, maskFloatBuf, maskSumBuf);
    }
    else {
      SizedBuf<void*> g1Float(&scratch->allocator, scratch->getBufSizeXYFloat(g1Channels, maxBatchSize, xySize));
      bool fusedHeadBN = false;
      if(useWideHead) {
        if(useHeadBNHalfToFloat)
          fusedHeadBN = sm89HeadBNHalfToFloatB13(
            wideHeadBuf, nullptr, (float*)g1Float.buf,
            (const half*)g1BN.mergedScaleBuf, (const half*)g1BN.mergedBiasBuf,
            batchSize, xySize, g1Channels, 384, 96, ctx->stream
          );
        else {
          fusedHeadBN = sm89HeadBNSiluStridedB13(
            wideHeadBuf, (half*)g1Out2.buf,
            (const half*)g1BN.mergedScaleBuf, (const half*)g1BN.mergedBiasBuf,
            batchSize, xySize, g1Channels, 384, 96, ctx->stream
          );
          if(fusedHeadBN)
            customCudaCopyFromHalf(
              (const half*)g1Out2.buf, (float*)g1Float.buf,
              batchSize * g1Channels * xySize
            );
        }
        testAssert(fusedHeadBN);
      }
      else if(usingNHWC && useHeadBNHalfToFloat && maskBuf == NULL) {
        fusedHeadBN = sm89HeadBNHalfToFloatB13(
          (const half*)g1Out.buf, nullptr, (float*)g1Float.buf,
          (const half*)g1BN.mergedScaleBuf, (const half*)g1BN.mergedBiasBuf,
          batchSize, xySize, g1Channels, g1Channels, 0, ctx->stream
        );
      }
      if(!fusedHeadBN) {
        g1BN.apply(batchSize, g1Out.buf, maskBuf, g1Out2.buf);
        customCudaCopyFromHalf((const half*)g1Out2.buf, (float*)g1Float.buf, batchSize * g1Channels * xySize);
      }
      customCudaPoolRowsGPoolNHWC((const float*)g1Float.buf, (float*)g1Concat.buf, batchSize, xySize, g1Channels, maskFloatBuf, maskSumBuf);
    }
    CUDA_ERR("Sm89PolicyHead",cudaPeekAtLastError());

    gpoolToBiasMul.apply(ctx, scratch, batchSize, g1Concat.buf, g1Bias.buf);

    bool fusedP1 = false;
    if(usingFP16 && usingNHWC && useFusedPolicyP1 && maskFloatBuf == NULL) {
      fusedP1 = sm89FusedPolicyP1B13(
        useWideHead ? wideHeadBuf : (const half*)p1Out.buf,
        (float*)p1Out2.buf, (const float*)g1Bias.buf,
        (const float*)p1BN.mergedScaleBuf, (const float*)p1BN.mergedBiasBuf,
        batchSize, xySize, p1Channels,
        useWideHead ? 384 : p1Channels, 0, ctx->stream
      );
    }
    if(fusedP1) {
      p2Conv.apply(ctx, batchSize, false, p1Out2.buf, policyBuf, workspaceBuf, workspaceBytes);
    }
    else {
      testAssert(!useWideHead);
      float* p1OutBufA;
      float* p1OutBufB;
      if(!usingFP16) {
        p1OutBufA = (float*)p1Out.buf;
        p1OutBufB = (float*)p1Out2.buf;
      }
      else {
        customCudaCopyFromHalf((const half*)p1Out.buf, (float*)p1Out2.buf, batchSize * p1Channels * xySize);
        p1OutBufA = (float*)p1Out2.buf;
        p1OutBufB = (float*)p1Out.buf;
      }
      customCudaAddNCBiasInplaceNHWC(p1OutBufA, (float*)g1Bias.buf, batchSize, xySize, p1Channels);
      CUDA_ERR("Sm89PolicyHead",cudaPeekAtLastError());
      p1BN.apply(batchSize, p1OutBufA, maskFloatBuf, p1OutBufB);
      p2Conv.apply(ctx, batchSize, false, p1OutBufB, policyBuf, workspaceBuf, workspaceBytes);
    }

    if(modelVersion >= 15) {
      gpoolToPassMul.apply(ctx, scratch, batchSize, g1Concat.buf, p1Pass.buf);
      gpoolToPassBias.apply(batchSize, p1Pass.buf);
      gpoolToPassMul2.apply(ctx, scratch, batchSize, p1Pass.buf, policyPassBuf);
    }
    else {
      gpoolToPassMul.apply(ctx, scratch, batchSize, g1Concat.buf, policyPassBuf);
    }
  }

};

struct Sm89FusedValueTerminal {
  const bool active;
  const int inChannels;
  const int valueChannels;
  const int scoreValueChannels;
  void* weightBuf;
  void* biasBuf;

  Sm89FusedValueTerminal() = delete;
  Sm89FusedValueTerminal(const Sm89FusedValueTerminal&) = delete;
  Sm89FusedValueTerminal& operator=(const Sm89FusedValueTerminal&) = delete;

  Sm89FusedValueTerminal(const ValueHeadDesc* desc, bool enabled)
    : active(
        enabled && desc->modelVersion >= 9 &&
        desc->v3Mul.outChannels == 3 && desc->sv3Mul.outChannels == 6 &&
        desc->v3Mul.inChannels == desc->sv3Mul.inChannels
      ),
      inChannels(desc->v3Mul.inChannels),
      valueChannels(desc->v3Mul.outChannels),
      scoreValueChannels(desc->sv3Mul.outChannels),
      weightBuf(NULL),
      biasBuf(NULL)
  {
    if(!active)
      return;

    testAssert((int)desc->v3Mul.weights.size() == inChannels * valueChannels);
    testAssert((int)desc->sv3Mul.weights.size() == inChannels * scoreValueChannels);
    testAssert((int)desc->v3Bias.weights.size() == valueChannels);
    testAssert((int)desc->sv3Bias.weights.size() == scoreValueChannels);

    const int combinedChannels = valueChannels + scoreValueChannels;
    vector<float> weights((size_t)inChannels * combinedChannels);
    for(int k = 0; k < inChannels; k++) {
      std::copy_n(
        desc->v3Mul.weights.begin() + (size_t)k * valueChannels,
        valueChannels,
        weights.begin() + (size_t)k * combinedChannels
      );
      std::copy_n(
        desc->sv3Mul.weights.begin() + (size_t)k * scoreValueChannels,
        scoreValueChannels,
        weights.begin() + (size_t)k * combinedChannels + valueChannels
      );
    }
    vector<float> biases;
    biases.reserve(combinedChannels);
    biases.insert(biases.end(), desc->v3Bias.weights.begin(), desc->v3Bias.weights.end());
    biases.insert(biases.end(), desc->sv3Bias.weights.begin(), desc->sv3Bias.weights.end());
    CudaUtils::mallocAndCopyToDevice("fusedValueTerminal:weights", weights, weightBuf, false);
    CudaUtils::mallocAndCopyToDevice("fusedValueTerminal:biases", biases, biasBuf, false);
  }

  ~Sm89FusedValueTerminal() {
    if(weightBuf != NULL)
      cudaFree(weightBuf);
    if(biasBuf != NULL)
      cudaFree(biasBuf);
  }

  bool apply(
    Sm89Ctx* ctx, Sm89Scratch* scratch, int batchSize,
    const float* input, float* value, float* scoreValue
  ) const {
    if(!active || batchSize != 13)
      return false;
    const int combinedChannels = valueChannels + scoreValueChannels;
    SizedBuf<void*> combined(
      &scratch->allocator, scratch->getBufSizeFloat(combinedChannels, batchSize)
    );
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR("fusedValueTerminal",cublasSgemm(
      ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      combinedChannels, batchSize, inChannels,
      &alpha, (const float*)weightBuf, combinedChannels,
      input, inChannels,
      &beta, (float*)combined.buf, combinedChannels
    ));
    return sm89SplitValueTerminalB13(
      (const float*)combined.buf, (const float*)biasBuf,
      value, scoreValue,
      batchSize, valueChannels, scoreValueChannels, ctx->stream
    );
  }
};

struct Sm89ValueHead {
  const int modelVersion;
  const int nnXLen;
  const int nnYLen;
  const int maxBatchSize;
  const int v1Channels;
  const int v2Channels;
  const int valueChannels;
  const int scoreValueChannels;
  const int ownershipChannels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool useHeadBNHalfToFloat;
  const bool useFusedValueTerminal;
  const Sm89Conv v1Conv;
  const Sm89BatchNorm v1BN;
  const Sm89MatMul v2Mul;
  const Sm89MatBias v2Bias;
  const Sm89MatMul v3Mul;
  const Sm89MatBias v3Bias;
  const Sm89MatMul sv3Mul;
  const Sm89MatBias sv3Bias;
  const Sm89FusedValueTerminal fusedValueTerminal;
  const Sm89Conv vOwnershipConv;

  Sm89ValueHead() = delete;
  Sm89ValueHead(const Sm89ValueHead&) = delete;
  Sm89ValueHead& operator=(const Sm89ValueHead&) = delete;

  Sm89ValueHead(Sm89Ctx* ctx, const ValueHeadDesc* desc, int maxBatchSize_, int nnX, int nnY, bool useFP16, bool useNHWC, bool useHeadBNHalfToFloat_, bool useFusedValueTerminal_, bool shareModelWeights)
    : modelVersion(desc->modelVersion),
      nnXLen(nnX),
      nnYLen(nnY),
      maxBatchSize(maxBatchSize_),
      v1Channels(desc->v1Conv.outChannels),
      v2Channels(desc->v2Mul.outChannels),
      valueChannels(desc->v3Mul.outChannels),
      scoreValueChannels(desc->sv3Mul.outChannels),
      ownershipChannels(desc->vOwnershipConv.outChannels),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      useHeadBNHalfToFloat(useHeadBNHalfToFloat_),
      useFusedValueTerminal(useFusedValueTerminal_),
      v1Conv(ctx, &desc->v1Conv, maxBatchSize_, nnX, nnY, useFP16, useNHWC, useNHWC),
      v1BN(&desc->v1BN, &desc->v1Activation, nnX, nnY, useFP16, useNHWC),
      v2Mul(&desc->v2Mul, false, shareModelWeights),
      v2Bias(&desc->v2Bias, false, desc->v2Activation.activation),
      v3Mul(&desc->v3Mul, false, shareModelWeights),
      v3Bias(&desc->v3Bias, false, ACTIVATION_IDENTITY),
      sv3Mul(&desc->sv3Mul, false, shareModelWeights),
      sv3Bias(&desc->sv3Bias, false, ACTIVATION_IDENTITY),
      fusedValueTerminal(desc, useFusedValueTerminal_),
      vOwnershipConv(ctx, &desc->vOwnershipConv, maxBatchSize_, nnX, nnY, useFP16, useNHWC, useNHWC)
  {}

  void apply(
    Sm89Ctx* ctx,
    Sm89Scratch* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    const half* wideHeadBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    int xySize = nnXLen * nnYLen;
    SizedBuf<void*> v1Out(&scratch->allocator, scratch->getBufSizeXY(v1Channels, maxBatchSize, xySize, usingFP16));
    SizedBuf<void*> v1Out2(&scratch->allocator, scratch->getBufSizeXY(v1Channels, maxBatchSize, xySize, usingFP16));
    SizedBuf<void*> v1Mean(&scratch->allocator, scratch->getBufSizeFloat(v1Channels * 3, maxBatchSize));
    SizedBuf<void*> v2Out(&scratch->allocator, scratch->getBufSizeFloat(v2Channels, maxBatchSize));
    SizedBuf<void*> ownershipScratch(&scratch->allocator, scratch->getBufSizeXYFloat(ownershipChannels, maxBatchSize, xySize));

    const bool useWideHead = wideHeadBuf != nullptr;
    if(!useWideHead)
      v1Conv.apply(ctx, batchSize, false, trunkBuf, v1Out.buf, workspaceBuf, workspaceBytes);

    void* bufToBePooled;
    if(!usingFP16) {
      v1BN.apply(batchSize, v1Out.buf, maskBuf, v1Out2.buf);
      bufToBePooled = v1Out2.buf;
    }
    else {
      bool fusedHeadBN = false;
      if(useWideHead) {
        if(useHeadBNHalfToFloat)
          fusedHeadBN = sm89HeadBNHalfToFloatB13(
            wideHeadBuf, (half*)v1Out2.buf, (float*)workspaceBuf,
            (const half*)v1BN.mergedScaleBuf, (const half*)v1BN.mergedBiasBuf,
            batchSize, xySize, v1Channels, 384, 192, ctx->stream
          );
        else {
          fusedHeadBN = sm89HeadBNSiluStridedB13(
            wideHeadBuf, (half*)v1Out2.buf,
            (const half*)v1BN.mergedScaleBuf, (const half*)v1BN.mergedBiasBuf,
            batchSize, xySize, v1Channels, 384, 192, ctx->stream
          );
          if(fusedHeadBN)
            customCudaCopyFromHalf(
              (const half*)v1Out2.buf, (float*)workspaceBuf,
              batchSize * v1Channels * xySize
            );
        }
        testAssert(fusedHeadBN);
      }
      else if(usingNHWC && useHeadBNHalfToFloat && maskBuf == NULL) {
        fusedHeadBN = sm89HeadBNHalfToFloatB13(
          (const half*)v1Out.buf, (half*)v1Out2.buf, (float*)workspaceBuf,
          (const half*)v1BN.mergedScaleBuf, (const half*)v1BN.mergedBiasBuf,
          batchSize, xySize, v1Channels, v1Channels, 0, ctx->stream
        );
      }
      if(!fusedHeadBN) {
        v1BN.apply(batchSize, v1Out.buf, maskBuf, v1Out2.buf);
        customCudaCopyFromHalf((const half*)v1Out2.buf, (float*)workspaceBuf, batchSize * v1Channels * xySize);
      }
      bufToBePooled = workspaceBuf;
    }
    customCudaValueHeadPoolNHWC((const float*)bufToBePooled, (float*)v1Mean.buf, batchSize, xySize, v1Channels, maskSumBuf);
    CUDA_ERR("Sm89ValueHead",cudaPeekAtLastError());

    v2Mul.apply(ctx, scratch, batchSize, v1Mean.buf, v2Out.buf);
    v2Bias.apply(batchSize, v2Out.buf);
    bool fusedTerminal = false;
    if(useFusedValueTerminal)
      fusedTerminal = fusedValueTerminal.apply(
        ctx, scratch, batchSize, (const float*)v2Out.buf, valueBuf, scoreValueBuf
      );
    if(!fusedTerminal) {
      v3Mul.apply(ctx, scratch, batchSize, v2Out.buf, valueBuf);
      v3Bias.apply(batchSize, valueBuf);
      sv3Mul.apply(ctx, scratch, batchSize, v2Out.buf, scoreValueBuf);
      sv3Bias.apply(batchSize, scoreValueBuf);
    }

    if(!usingFP16) {
      vOwnershipConv.apply(ctx, batchSize, false, v1Out2.buf, ownershipBuf, workspaceBuf, workspaceBytes);
    }
    else {
      vOwnershipConv.apply(ctx, batchSize, false, v1Out2.buf, ownershipScratch.buf, workspaceBuf, workspaceBytes);
      customCudaCopyFromHalf((const half*)ownershipScratch.buf, (float*)ownershipBuf, batchSize * ownershipChannels * xySize);
      CUDA_ERR("Sm89ValueHead",cudaPeekAtLastError());
    }
  }
};

// --------------------------------------------------------------------------------------
// Forward implementation

struct Sm89Forward::Impl {
  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const int numInputChannels;
  const bool usingFP16;
  const bool usingNHWC;
  const bool inputsUseNHWC;
  const bool useWideQKV;
  const bool useWideFFN;
  const bool useFusedResidual;
  const bool useRMSNormOpt;
  const bool useFusedQKRoPE;
  const bool usePrecomputedQKRoPE;
  const bool useQKVRoPEGemm;
  const int ropeBatchGroup;
  const bool useFlashAttention;
  const bool useDualGemmSwiGLU;
  const bool useLinear2Gemm;
  const bool useOutProjGemm;
  const bool usePreConvGemm;
  const bool usePostConvGemm;
  const bool usePostConvBNSilu;
  const bool useLinear2PostBNSilu;
  const bool usePersistingL2Trunk;
  const float persistingL2TrunkHitRatio;
  const bool usePersistingL2Inner;
  const float persistingL2InnerHitRatio;
  const bool useScaleBiasSiluVec8;
  const bool useScaleBiasSiluVec4C384;
  const bool useInitialConvFrontend;
  const bool useInitialGlobalMatMulAdd;
  const bool useFusedPolicyP1;
  const bool useHeadBNHalfToFloat;
  const bool useWideHeadProjection;
  const bool useExactMaskElision;
  const bool useFusedValueTerminal;
  const bool shareModelWeights;
  Sm89Ctx ctx;
  Sm89Scratch scratch;
  Sm89Trunk trunk;
  Sm89WideHeadProjection wideHeadProjection;
  Sm89PolicyHead policyHead;
  Sm89ValueHead valueHead;
  void* convWorkspace;
  size_t convWorkspaceBytes;

  Impl(
    const ModelDesc* desc,
    int maxBatchSize_,
    int nnXLen_,
    int nnYLen_,
    bool inputsUseNHWC_,
    bool useFP16,
    bool useNHWC,
    bool useWideQKV_,
    bool useWideFFN_,
    bool useFusedResidual_,
    bool useRMSNormOpt_,
    bool useFusedQKRoPE_,
    bool usePrecomputedQKRoPE_,
    bool useQKVRoPEGemm_,
    int ropeBatchGroup_,
    bool useFlashAttention_,
    bool useDualGemmSwiGLU_,
    bool useLinear2Gemm_,
    bool useOutProjGemm_,
    bool usePreConvGemm_,
    bool usePostConvGemm_,
    bool usePostConvBNSilu_,
    bool useLinear2PostBNSilu_,
    bool usePersistingL2Trunk_,
    float persistingL2TrunkHitRatio_,
    bool usePersistingL2Inner_,
    float persistingL2InnerHitRatio_,
    bool useScaleBiasSiluVec8_,
    bool useScaleBiasSiluVec4C384_,
    bool useInitialConvFrontend_,
    bool useInitialGlobalMatMulAdd_,
    bool useFusedPolicyP1_,
    bool useHeadBNHalfToFloat_,
    bool useWideHeadProjection_,
    bool useExactMaskElision_,
    bool useFusedValueTerminal_,
    bool shareModelWeights_
  )
    : maxBatchSize(maxBatchSize_),
      nnXLen(nnXLen_),
      nnYLen(nnYLen_),
      numInputChannels(desc->numInputChannels),
      usingFP16(useFP16),
      usingNHWC(useNHWC),
      inputsUseNHWC(inputsUseNHWC_),
      useWideQKV(useWideQKV_),
      useWideFFN(useWideFFN_),
      useFusedResidual(useFusedResidual_),
      useRMSNormOpt(useRMSNormOpt_),
      useFusedQKRoPE(useFusedQKRoPE_),
      usePrecomputedQKRoPE(usePrecomputedQKRoPE_),
      useQKVRoPEGemm(useQKVRoPEGemm_),
      ropeBatchGroup(ropeBatchGroup_),
      useFlashAttention(useFlashAttention_),
      useDualGemmSwiGLU(useDualGemmSwiGLU_),
      useLinear2Gemm(useLinear2Gemm_),
      useOutProjGemm(useOutProjGemm_),
      usePreConvGemm(usePreConvGemm_),
      usePostConvGemm(usePostConvGemm_),
      usePostConvBNSilu(usePostConvBNSilu_),
      useLinear2PostBNSilu(useLinear2PostBNSilu_),
      usePersistingL2Trunk(usePersistingL2Trunk_),
      persistingL2TrunkHitRatio(persistingL2TrunkHitRatio_),
      usePersistingL2Inner(usePersistingL2Inner_),
      persistingL2InnerHitRatio(persistingL2InnerHitRatio_),
      useScaleBiasSiluVec8(useScaleBiasSiluVec8_),
      useScaleBiasSiluVec4C384(useScaleBiasSiluVec4C384_),
      useInitialConvFrontend(useInitialConvFrontend_),
      useInitialGlobalMatMulAdd(useInitialGlobalMatMulAdd_),
      useFusedPolicyP1(useFusedPolicyP1_),
      useHeadBNHalfToFloat(useHeadBNHalfToFloat_),
      useWideHeadProjection(useWideHeadProjection_),
      useExactMaskElision(useExactMaskElision_),
      useFusedValueTerminal(useFusedValueTerminal_),
      shareModelWeights(shareModelWeights_),
      ctx(),
      scratch(
        useFP16, useExactMaskElision_, maxBatchSize_, nnXLen_ * nnYLen_
      ),
      trunk(&ctx, &desc->trunk, maxBatchSize_, nnXLen_, nnYLen_, useFP16, useNHWC,
        useWideQKV_, useWideFFN_, useFusedResidual_, useRMSNormOpt_, useFusedQKRoPE_,
        usePrecomputedQKRoPE_, useQKVRoPEGemm_, ropeBatchGroup_,
        useFlashAttention_, useDualGemmSwiGLU_, useLinear2Gemm_, useOutProjGemm_,
        usePreConvGemm_, usePostConvGemm_, usePostConvBNSilu_,
        useLinear2PostBNSilu_,
        usePersistingL2Trunk_,
        persistingL2TrunkHitRatio_, usePersistingL2Inner_,
        persistingL2InnerHitRatio_, useScaleBiasSiluVec8_,
        useScaleBiasSiluVec4C384_,
        useInitialConvFrontend_, useInitialGlobalMatMulAdd_, shareModelWeights_),
      wideHeadProjection(
        &desc->policyHead, &desc->valueHead, useFP16,
        useWideHeadProjection_ && useFusedPolicyP1_
      ),
      policyHead(&ctx, &desc->policyHead, maxBatchSize_, nnXLen_, nnYLen_, useFP16, useNHWC, useFusedPolicyP1_, useHeadBNHalfToFloat_, shareModelWeights_),
      valueHead(&ctx, &desc->valueHead, maxBatchSize_, nnXLen_, nnYLen_, useFP16, useNHWC, useHeadBNHalfToFloat_, useFusedValueTerminal_, shareModelWeights_),
      convWorkspace(NULL),
      convWorkspaceBytes(64 * 1024 * 1024)
  {
    CUDA_ERR("Sm89Forward",cudaMalloc(&convWorkspace, convWorkspaceBytes));
  }

  ~Impl() {
    if(convWorkspace != NULL)
      cudaFree(convWorkspace);
  }

  void apply(
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
    (void)inputMetaBuf;
    (void)workspaceBuf;
    (void)workspaceBytes;
    int xySize = nnXLen * nnYLen;

    std::unique_ptr<SizedBuf<void*>> maskBuf;
    std::unique_ptr<SizedBuf<void*>> maskFloatBuf;
    std::unique_ptr<SizedBuf<void*>> maskSumBuf;
    SizedBuf<void*> trunkBuf(&scratch.allocator, scratch.getBufSizeXY(trunk.trunkNumChannels, maxBatchSize, xySize, usingFP16));

    void* mask = NULL;
    float* maskFloat = NULL;
    float* maskSum = NULL;
    const bool elideExactMask =
      useExactMaskElision && requireExactNNLen &&
      scratch.fullBoardMaskSumBuf != NULL;
    if(elideExactMask) {
      maskSum = (float*)scratch.fullBoardMaskSumBuf;
    }
    else {
      maskBuf = std::make_unique<SizedBuf<void*>>(
        &scratch.allocator,
        scratch.getBufSize(1, maxBatchSize, usingFP16) * xySize
      );
      maskFloatBuf = std::make_unique<SizedBuf<void*>>(
        &scratch.allocator,
        scratch.getBufSizeFloat(1, maxBatchSize) * xySize
      );
      maskSumBuf = std::make_unique<SizedBuf<void*>>(
        &scratch.allocator, scratch.getBufSizeFloat(1, maxBatchSize)
      );

      if(!usingFP16) {
        if(inputsUseNHWC)
          customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf->buf, batchSize, xySize, numInputChannels);
        else
          customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf->buf, batchSize, numInputChannels, xySize);
        customCudaPoolRowsSumNCHW((const float*)maskBuf->buf, (float*)maskSumBuf->buf, batchSize, 1, xySize, 1.0);
      }
      else {
        if(inputsUseNHWC)
          customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf->buf, batchSize, xySize, numInputChannels);
        else
          customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf->buf, batchSize, numInputChannels, xySize);
        customCudaCopyFromHalf((const half*)maskBuf->buf, (float*)maskFloatBuf->buf, batchSize * xySize);
        customCudaPoolRowsSumNCHW((const float*)maskFloatBuf->buf, (float*)maskSumBuf->buf, batchSize, 1, xySize, 1.0);
      }
      CUDA_ERR("Sm89Forward",cudaPeekAtLastError());

      mask = requireExactNNLen ? NULL : maskBuf->buf;
      maskFloat = requireExactNNLen ? NULL : (float*)maskFloatBuf->buf;
      maskSum = (float*)maskSumBuf->buf;
    }

    trunk.apply(&ctx, &scratch, batchSize, inputBuf, inputGlobalBuf, mask, trunkBuf.buf, convWorkspace, convWorkspaceBytes);
    std::unique_ptr<SizedBuf<void*>> wideHeadBuf;
    const half* wideHead = nullptr;
    if(requireExactNNLen && usingFP16 && usingNHWC && useWideHeadProjection) {
      wideHeadBuf = std::make_unique<SizedBuf<void*>>(
        &scratch.allocator, scratch.getBufSizeXY(384, maxBatchSize, xySize, true)
      );
      if(wideHeadProjection.apply(
        (const half*)trunkBuf.buf, (half*)wideHeadBuf->buf,
        batchSize, xySize, ctx.stream
      ))
        wideHead = (const half*)wideHeadBuf->buf;
    }
    policyHead.apply(&ctx, &scratch, batchSize, mask, maskFloat, maskSum, trunkBuf.buf, wideHead, policyPassBuf, policyBuf, convWorkspace, convWorkspaceBytes);
    valueHead.apply(&ctx, &scratch, batchSize, mask, maskSum, trunkBuf.buf, wideHead, valueBuf, scoreValueBuf, ownershipBuf, convWorkspace, convWorkspaceBytes);
  }

};

bool Sm89Forward::supports(const ModelDesc& desc, bool useFP16, bool useNHWC) {
  if(!useFP16 || !useNHWC)
    return false;
  if(desc.metaEncoderVersion > 0)
    return false;
  if(desc.trunk.trunkNormKind != TRUNK_NORM_KIND_STANDARD)
    return false;
  for(size_t i = 0; i < desc.trunk.blocks.size(); i++) {
    if(desc.trunk.blocks[i].first != NESTED_BOTTLENECK_BLOCK_KIND)
      return false;
    const NestedBottleneckResidualBlockDesc* b =
      (const NestedBottleneckResidualBlockDesc*)desc.trunk.blocks[i].second.get();
    for(size_t j = 0; j < b->blocks.size(); j++) {
      int kind = b->blocks[j].first;
      if(kind != TRANSFORMER_ATTENTION_BLOCK_KIND && kind != TRANSFORMER_FFN_BLOCK_KIND)
        return false;
    }
  }
  return true;
}

Sm89Forward::Sm89Forward(
  const ModelDesc* desc,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool inputsUseNHWC,
  bool useFP16,
  bool useNHWC,
  bool useWideQKV,
  bool useWideFFN,
  bool useFusedResidual,
  bool useRMSNormOpt,
  bool useFusedQKRoPE,
  bool usePrecomputedQKRoPE,
  bool useQKVRoPEGemm,
  int ropeBatchGroup,
  bool useFlashAttention,
  bool useDualGemmSwiGLU,
  bool useLinear2Gemm,
  bool useOutProjGemm,
  bool usePreConvGemm,
  bool usePostConvGemm,
  bool usePostConvBNSilu,
  bool useLinear2PostBNSilu,
  bool usePersistingL2Trunk,
  float persistingL2TrunkHitRatio,
  bool usePersistingL2Inner,
  float persistingL2InnerHitRatio,
  bool useScaleBiasSiluVec8,
  bool useScaleBiasSiluVec4C384,
  bool useInitialConvFrontend,
  bool useInitialGlobalMatMulAdd,
  bool useFusedPolicyP1,
  bool useHeadBNHalfToFloat,
  bool useWideHeadProjection,
  bool useExactMaskElision,
  bool useFusedValueTerminal,
  bool shareModelWeights
)
  : impl(std::make_unique<Impl>(desc, maxBatchSize, nnXLen, nnYLen, inputsUseNHWC,
      useFP16, useNHWC, useWideQKV, useWideFFN, useFusedResidual, useRMSNormOpt,
      useFusedQKRoPE, usePrecomputedQKRoPE, useQKVRoPEGemm,
      ropeBatchGroup, useFlashAttention,
      useDualGemmSwiGLU, useLinear2Gemm,
      useOutProjGemm, usePreConvGemm, usePostConvGemm, usePostConvBNSilu,
      useLinear2PostBNSilu,
      usePersistingL2Trunk,
      persistingL2TrunkHitRatio, usePersistingL2Inner,
      persistingL2InnerHitRatio, useScaleBiasSiluVec8,
      useScaleBiasSiluVec4C384,
      useInitialConvFrontend, useInitialGlobalMatMulAdd,
      useFusedPolicyP1, useHeadBNHalfToFloat, useWideHeadProjection,
      useExactMaskElision,
      useFusedValueTerminal,
      shareModelWeights))
{}

Sm89Forward::~Sm89Forward() = default;

void Sm89Forward::apply(
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
  impl->apply(
    batchSize, requireExactNNLen,
    inputBuf, inputGlobalBuf, inputMetaBuf,
    policyPassBuf, policyBuf, valueBuf, scoreValueBuf, ownershipBuf,
    workspaceBuf, workspaceBytes
  );
}

} // namespace Sm89Backend
