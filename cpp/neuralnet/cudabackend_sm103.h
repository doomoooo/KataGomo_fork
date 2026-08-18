#ifndef KATAGO_CUDA_BACKEND_SM103_H
#define KATAGO_CUDA_BACKEND_SM103_H

#include "../core/config_parser.h"
#include <cstddef>

// B300 / SM103 runtime adapter. All hooks are architecture-exact and explicit;
// the default remains the official CUDA path. The adapter may reuse only the
// architecture-neutral SM120 operator hooks that pass the SM103 validator.

struct CudaHandles;    // defined in cudabackend.cpp
struct ScratchBuffers; // defined in cudabackend.cpp
struct Logger;         // defined in core/logger.h

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

struct Options {
  // Default-off so the official CUDA baseline on B300 is unchanged.
  bool enabled = false;
  // Reuse only the architecture-neutral operator hooks owned by
  // Sm120Model. The adapter rejects every SM120 AOT/FA/CUTLASS option before
  // that owner is constructed, so an SM120 cubin can never be selected on
  // CC10.3. The existing cuda*Sm120 operator switches remain the per-tactic
  // controls; this flag is the additional architecture opt-in.
  bool reusePortableTactics = false;
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
    int nnXLen,
    int nnYLen,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC,
    int majorComputeCapability,
    int minorComputeCapability,
    const Options& options
  );

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

 private:
  void* officialApplyContext;
  OfficialApplyFn officialApply;
  Options options;
  Logger* logger;
  bool loggedScaffold;
};

} // namespace Sm103Backend

#endif
