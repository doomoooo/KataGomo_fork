#ifndef NEURALNET_CUDATACTICPLAN_H_
#define NEURALNET_CUDATACTICPLAN_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../core/config_parser.h"
#include "../core/logger.h"

namespace CudaTacticPlan {

struct DeviceRequirements {
  std::string name;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiprocessor;
  int regsPerMultiprocessor;
  int sharedMemPerBlockOptin;
  int sharedMemPerMultiprocessor;
  int l2CacheSize;
  int memoryBusWidth;
  int asyncEngineCount;
  bool concurrentKernels;
  uint64_t totalGlobalMem;
};

struct Plan {
  std::string path;
  std::string fileSha256;
  std::string planId;
  std::string planSha256;
  std::string architecture;
  std::string gpuClass;
  std::string modelSha256;
  int batchSize;
  int streamsPerDevice;
  DeviceRequirements device;
  std::map<std::string,std::string> tacticOverrides;
};

// Loads and applies cudaTacticPlanFile when configured. The selected plan is
// authoritative for its tactic keys, exact batch shape, precision, and stream
// count per device. An identical plan may be replicated across matching GPUs;
// device ordinals remain receiver-local and are never copied from the producer.
std::unique_ptr<Plan> loadAndApply(
  ConfigParser& cfg,
  Logger& logger,
  int nnXLen,
  int nnYLen,
  bool& requireExactNNLen
);

// Validates every receiver device used by the evaluator. A plan produced for
// one ordinal may be applied to another ordinal only when tactic-relevant CUDA
// capabilities match.
void validateDevices(
  const Plan& plan,
  const std::vector<int>& gpuIdxByServerThread
);

}

#endif  // NEURALNET_CUDATACTICPLAN_H_
