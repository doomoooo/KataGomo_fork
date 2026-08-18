#ifndef KATAGO_CUDA_BACKEND_SM103_H
#define KATAGO_CUDA_BACKEND_SM103_H

#include "../core/config_parser.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <map>
#include <string>
#include <utility>

// B300 / SM103 runtime adapter. All hooks are architecture-exact and explicit;
// the default remains the official CUDA path. The adapter may reuse only the
// architecture-neutral SM120 operator hooks that pass the SM103 validator.

struct CudaHandles;    // defined in cudabackend.cpp
struct ScratchBuffers; // defined in cudabackend.cpp
struct Logger;         // defined in core/logger.h
struct KatagoCudnnOssB29Context;
struct KatagoFa4Sm103B29Context;

namespace Sm103Backend {

typedef void (*OfficialApplyFn)(
  void* ctx,
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
);

// SM103-owned exact-shape FFN hook. This is separate from the portable SM120
// callback set because the linked AOT image is architecture- and batch-exact.
typedef bool (*Sm103FusedFFNFn)(
  void* ctx,
  const void* linear1Weights,
  const void* linearGateWeights,
  const void* input,
  void* ab12Scratch,
  void* output,
  int matBatchSize,
  int inputChannels,
  int ffnChannels,
  bool usingFP16,
  cudaStream_t stream
);

// Exact B29 planar-QKV FA4 forward hook. Keeping this separate from the SM120
// attention callback prevents an SM120 artifact from being installed on CC10.3.
typedef bool (*Sm103AttentionFn)(
  void* ctx,
  const void* q,
  const void* k,
  const void* v,
  bool packedQKV,
  const void* mask,
  void* output,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFP16,
  cudaStream_t stream
);

struct Options {
  // Default-off so the official CUDA baseline on B300 is unchanged.
  bool enabled = false;
  // Reuse only the architecture-neutral operator hooks owned by
  // Sm120Model. The adapter rejects every SM120 AOT/FA/CUTLASS option before
  // that owner is constructed, so an SM120 cubin can never be selected on
  // CC10.3. The existing cuda*Sm120 operator switches remain the per-tactic
  // controls; this flag is the additional architecture opt-in.
  bool reusePortableTactics = false;
  // Exact candidate identifier, or "disabled". Kept independent of the
  // portable tactic bundle and default-off.
  std::string dualFfnTactic = "disabled";
  // Exact native FA4/CuTe SM103a candidate identifier, or "disabled".
  std::string attentionTactic = "disabled";
  // Wiring tests may opt into an official-forward-only scaffold. Enabling the
  // SM103 backend without this or reusePortableTactics is a startup error.
  bool allowOfficialForwardScaffold = false;
};

bool isSm103Arch(int majorComputeCapability, int minorComputeCapability);
Options parseOptions(ConfigParser& cfg);

class Sm103Model {
 public:
  Sm103Model(
    void* officialApplyContext,
    OfficialApplyFn officialApply,
    CudaHandles* cudaHandles,
    int maxBatchSize,
    int nnXLen,
    int nnYLen,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC,
    int majorComputeCapability,
    int minorComputeCapability,
    const Options& options
  );
  ~Sm103Model();

  void setLogger(Logger* logger);

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
  );

  bool fusedFFN(
    const void* linear1Weights,
    const void* linearGateWeights,
    const void* input,
    void* ab12Scratch,
    void* output,
    int matBatchSize,
    int inputChannels,
    int ffnChannels,
    bool usingFP16,
    cudaStream_t stream
  );

  bool attention(
    const void* q,
    const void* k,
    const void* v,
    bool packedQKV,
    const void* mask,
    void* output,
    int batchSize,
    int seqLen,
    int numHeads,
    int numKVHeads,
    int qHeadDim,
    int vHeadDim,
    bool usingFP16,
    cudaStream_t stream
  );

 private:
  void* officialApplyContext;
  OfficialApplyFn officialApply;
  Options options;
  Logger* logger;
  bool loggedScaffold;
  bool loggedCudnnOssFfn;
  bool loggedFa4Attention;
  int device;
  KatagoCudnnOssB29Context* cudnnOssB29Context;
  KatagoFa4Sm103B29Context* fa4Sm103B29Context;
  std::map<std::pair<const void*,const void*>,void*> packedFfnWeights;
  bool hasLaunchedFfn;
  cudaStream_t lastFfnStream;
  bool hasLaunchedAttention;
  cudaStream_t lastAttentionStream;
};

bool applyFusedFFN(
  void* context,
  const void* linear1Weights,
  const void* linearGateWeights,
  const void* input,
  void* ab12Scratch,
  void* output,
  int matBatchSize,
  int inputChannels,
  int ffnChannels,
  bool usingFP16,
  cudaStream_t stream
);

bool applyAttention(
  void* context,
  const void* q,
  const void* k,
  const void* v,
  bool packedQKV,
  const void* mask,
  void* output,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool usingFP16,
  cudaStream_t stream
);

} // namespace Sm103Backend

#endif
