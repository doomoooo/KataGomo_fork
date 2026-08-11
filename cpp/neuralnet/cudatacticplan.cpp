#include "../neuralnet/cudatacticplan.h"

#include "../core/fileutils.h"
#include "../external/nlohmann_json/json.hpp"

#if defined(USE_CUDA_BACKEND)
#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/cudaerrorcheck.h"
#endif

#include <set>

using namespace std;
using json = nlohmann::json;

namespace CudaTacticPlan {

namespace {

int requiredInt(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_number_integer())
    throw StringError("CUDA tactic plan " + context + " requires integer " + key);
  return obj[key].get<int>();
}

string requiredString(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_string())
    throw StringError("CUDA tactic plan " + context + " requires string " + key);
  return obj[key].get<string>();
}

bool requiredBool(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_boolean())
    throw StringError("CUDA tactic plan " + context + " requires boolean " + key);
  return obj[key].get<bool>();
}

uint64_t requiredUint64(const json& obj, const string& key, const string& context) {
  if(!obj.is_object() || !obj.contains(key) || !obj[key].is_number_unsigned())
    throw StringError("CUDA tactic plan " + context + " requires unsigned integer " + key);
  return obj[key].get<uint64_t>();
}

void requireConfigBool(ConfigParser& cfg, const string& key, bool required) {
  if(cfg.contains(key) && cfg.getBool(key) != required)
    throw StringError(
      "CUDA tactic plan requires " + key + "=" + (required ? "true" : "false")
    );
}

void requireConfigInt(ConfigParser& cfg, const string& key, int required) {
  if(cfg.contains(key) && cfg.getInt(key) != required)
    throw StringError(
      "CUDA tactic plan requires " + key + "=" + Global::intToString(required)
    );
}

DeviceRequirements parseDeviceRequirements(const json& target) {
  if(!target.contains("cuda_device_capabilities_at_scan") ||
     !target["cuda_device_capabilities_at_scan"].is_array() ||
     target["cuda_device_capabilities_at_scan"].empty())
    throw StringError("CUDA tactic plan has no producer device capabilities");
  const json& device = target["cuda_device_capabilities_at_scan"][0];
  DeviceRequirements result;
  result.name = requiredString(device,"name","device target");
  result.computeCapabilityMajor = requiredInt(device,"computeCapabilityMajor","device target");
  result.computeCapabilityMinor = requiredInt(device,"computeCapabilityMinor","device target");
  result.multiProcessorCount = requiredInt(device,"multiProcessorCount","device target");
  result.maxThreadsPerBlock = requiredInt(device,"maxThreadsPerBlock","device target");
  result.maxThreadsPerMultiprocessor = requiredInt(device,"maxThreadsPerMultiProcessor","device target");
  result.regsPerMultiprocessor = requiredInt(device,"regsPerMultiprocessor","device target");
  result.sharedMemPerBlockOptin = requiredInt(device,"sharedMemPerBlockOptin","device target");
  result.sharedMemPerMultiprocessor = requiredInt(device,"sharedMemPerMultiprocessor","device target");
  result.l2CacheSize = requiredInt(device,"l2CacheSize","device target");
  result.memoryBusWidth = requiredInt(device,"memoryBusWidth","device target");
  result.asyncEngineCount = requiredInt(device,"asyncEngineCount","device target");
  result.concurrentKernels = requiredBool(device,"concurrentKernels","device target");
  result.totalGlobalMem = requiredUint64(device,"totalGlobalMem","device target");
  return result;
}

int selectBatch(ConfigParser& cfg, const json& payload) {
  if(!payload.contains("batches") || !payload["batches"].is_array() || payload["batches"].empty())
    throw StringError("CUDA tactic plan has no certified batches");
  set<int> batches;
  for(const json& item : payload["batches"]) {
    if(!item.is_number_integer())
      throw StringError("CUDA tactic plan batches must be integers");
    batches.insert(item.get<int>());
  }

  int batch = -1;
  if(cfg.contains("cudaTacticPlanBatch"))
    batch = cfg.getInt("cudaTacticPlanBatch",1,65536);
  else if(cfg.contains("nnMaxBatchSize"))
    batch = cfg.getInt("nnMaxBatchSize",1,65536);
  else if(batches.size() == 1)
    batch = *batches.begin();
  else
    throw StringError(
      "CUDA tactic plan contains multiple batches; set cudaTacticPlanBatch explicitly"
    );
  if(batches.find(batch) == batches.end())
    throw StringError(
      "CUDA tactic plan has no certified B" + Global::intToString(batch) + " selection"
    );
  return batch;
}

void validateCertifiedBatch(const json& payload, int batch) {
  const string batchKey = Global::intToString(batch);
  if(!payload.contains("final_joint") || !payload["final_joint"].is_object() ||
     !payload["final_joint"].contains(batchKey) ||
     !payload["final_joint"][batchKey].is_object())
    throw StringError("CUDA tactic plan has no final joint B" + batchKey + " evidence");
  const json& joint = payload["final_joint"][batchKey];
  if(requiredString(joint,"measurement_kind","final joint") != "long_stable")
    throw StringError("CUDA tactic plan B" + batchKey + " is not long-stable");
  if(!joint.contains("correctness") || !joint["correctness"].is_object() ||
     requiredString(joint["correctness"],"status","correctness") != "passed")
    throw StringError("CUDA tactic plan B" + batchKey + " lacks passed correctness evidence");
}

map<string,string> parseTacticOverrides(const json& payload, int batch) {
  const string batchKey = Global::intToString(batch);
  if(!payload.contains("apply") || !payload["apply"].is_object() ||
     !payload["apply"].contains("per_batch_tactic_overrides") ||
     !payload["apply"]["per_batch_tactic_overrides"].is_object() ||
     !payload["apply"]["per_batch_tactic_overrides"].contains(batchKey))
    throw StringError("CUDA tactic plan has no B" + batchKey + " apply mapping");
  const json& value = payload["apply"]["per_batch_tactic_overrides"][batchKey];
  if(!value.is_string())
    throw StringError("CUDA tactic plan B" + batchKey + " apply mapping is not a string");
  map<string,string> overrides = ConfigParser::parseCommaSeparated(value.get<string>());
  if(overrides.empty())
    throw StringError("CUDA tactic plan B" + batchKey + " apply mapping is empty");
  for(const auto& item : overrides) {
    if(item.first.compare(0,4,"cuda") != 0)
      throw StringError("CUDA tactic plan apply mapping contains non-CUDA key " + item.first);
    if(item.first.find("DeviceToUse") != string::npos ||
       item.first.find("GpuToUse") != string::npos)
      throw StringError("CUDA tactic plan apply mapping must not contain a producer device ordinal");
  }
  return overrides;
}

void requireSameInt(const string& label, int expected, int actual, int deviceIdx) {
  if(actual != expected) {
    throw StringError(
      "CUDA tactic plan device " + Global::intToString(deviceIdx) + " " + label +
      " mismatch: expected " + Global::intToString(expected) +
      ", got " + Global::intToString(actual)
    );
  }
}

void requireSameUint64(const string& label, uint64_t expected, uint64_t actual, int deviceIdx) {
  if(actual != expected) {
    throw StringError(
      "CUDA tactic plan device " + Global::intToString(deviceIdx) + " " + label +
      " mismatch: expected " + Global::uint64ToString(expected) +
      ", got " + Global::uint64ToString(actual)
    );
  }
}

void requireSameString(const string& label, const string& expected, const string& actual, int deviceIdx) {
  if(actual != expected) {
    throw StringError(
      "CUDA tactic plan device " + Global::intToString(deviceIdx) + " " + label +
      " mismatch: expected " + expected + ", got " + actual
    );
  }
}

}

unique_ptr<Plan> loadAndApply(
  ConfigParser& cfg,
  Logger& logger,
  int nnXLen,
  int nnYLen,
  bool& requireExactNNLen
) {
  if(!cfg.contains("cudaTacticPlanFile"))
    return nullptr;
#if !defined(USE_CUDA_BACKEND)
  (void)logger;
  (void)nnXLen;
  (void)nnYLen;
  (void)requireExactNNLen;
  throw StringError("cudaTacticPlanFile requires a CUDA backend build");
#else
  const string path = cfg.getString("cudaTacticPlanFile");
  string contents;
  string fileSha256;
  FileUtils::loadFileIntoString(path,"",contents,&fileSha256);

  json payload;
  try {
    payload = json::parse(contents);
  }
  catch(const nlohmann::detail::exception& e) {
    throw StringError("Could not parse CUDA tactic plan " + path + ": " + e.what());
  }
  if(!payload.is_object() || requiredInt(payload,"schema","root") != 1 ||
     requiredString(payload,"kind","root") != "cuda-tactic-plan")
    throw StringError("Unsupported CUDA tactic plan schema in " + path);
  if(!requiredBool(payload,"ready_for_scan_bypass","root") ||
     !requiredBool(payload,"production_ready","root") ||
     requiredString(payload,"status","root") != "complete_long_stable")
    throw StringError("CUDA tactic plan is not production-ready: " + path);
  if(!payload.contains("positive_history_closure") ||
     !payload["positive_history_closure"].is_object() ||
     !requiredBool(payload["positive_history_closure"],"complete","history closure"))
    throw StringError("CUDA tactic plan has incomplete positive-history closure");

  if(!payload.contains("target") || !payload["target"].is_object())
    throw StringError("CUDA tactic plan has no target");
  const json& target = payload["target"];
  const string architecture = requiredString(target,"architecture","target");
  if(architecture != "sm89" && architecture != "sm120")
    throw StringError("Unsupported CUDA tactic plan architecture " + architecture);
  if(!target.contains("fixed_board") || target["fixed_board"] != json::array({19,19}))
    throw StringError("CUDA tactic plans currently require an exact 19x19 board");
  if(nnXLen != 19 || nnYLen != 19)
    throw StringError("CUDA tactic plan requires an exact 19x19 evaluator");
  if(requiredString(target,"precision","target") != "FP16/NHWC")
    throw StringError("CUDA tactic plan requires FP16/NHWC precision");

  const int batch = selectBatch(cfg,payload);
  const int streams = requiredInt(target,"streams","target");
  if(streams <= 0)
    throw StringError("CUDA tactic plan target stream count must be positive");
  const int evaluatorThreads = cfg.contains("numNNServerThreadsPerModel") ?
    cfg.getInt("numNNServerThreadsPerModel",1,65536) : streams;
  if(evaluatorThreads % streams != 0)
    throw StringError(
      "CUDA tactic plan requires exactly " + Global::intToString(streams) +
      " NN server threads per device"
    );
  validateCertifiedBatch(payload,batch);
  map<string,string> tacticOverrides = parseTacticOverrides(payload,batch);

  requireConfigInt(cfg,"nnMaxBatchSize",batch);
  requireConfigBool(cfg,"requireMaxBoardSize",true);
  requireConfigBool(cfg,"nnBatchAwareDispatch",true);
  requireConfigBool(cfg,"cudaWarmupOnlyMaxBatchSize",true);
  if(cfg.contains("useFP16") && cfg.getEnabled("useFP16") == enabled_t::False)
    throw StringError("CUDA tactic plan requires useFP16=true");
  if(cfg.contains("cudaUseNHWC") && cfg.getEnabled("cudaUseNHWC") == enabled_t::False)
    throw StringError("CUDA tactic plan requires cudaUseNHWC=true");
  requireExactNNLen = true;

  map<string,string> runtimeOverrides = tacticOverrides;
  runtimeOverrides["nnMaxBatchSize"] = Global::intToString(batch);
  runtimeOverrides["numNNServerThreadsPerModel"] = Global::intToString(evaluatorThreads);
  runtimeOverrides["nnBatchAwareDispatch"] = "true";
  runtimeOverrides["cudaWarmupOnlyMaxBatchSize"] = "true";
  runtimeOverrides["useFP16"] = "true";
  runtimeOverrides["cudaUseNHWC"] = "true";
  if(architecture == "sm89") {
    runtimeOverrides["cudaSm89Backend"] = "true";
    runtimeOverrides["cudaSm89Forward"] = "true";
  }
  else
    runtimeOverrides["cudaSm120Backend"] = "true";
  cfg.overrideKeys(runtimeOverrides);

  unique_ptr<Plan> plan = make_unique<Plan>();
  plan->path = path;
  plan->fileSha256 = fileSha256;
  plan->planId = requiredString(payload,"plan_id","root");
  plan->planSha256 = requiredString(payload,"plan_sha256","root");
  plan->architecture = architecture;
  plan->gpuClass = requiredString(target,"gpu_class","target");
  plan->modelSha256 = requiredString(target,"model_sha256","target");
  if(plan->modelSha256.size() != 64)
    throw StringError("CUDA tactic plan target model SHA-256 is malformed");
  plan->batchSize = batch;
  plan->streamsPerDevice = streams;
  plan->device = parseDeviceRequirements(target);
  plan->tacticOverrides = tacticOverrides;

  logger.write(
    "Loaded CUDA tactic plan " + plan->planId + " from " + path +
    " fileSha256=" + fileSha256 + " B" + Global::intToString(batch) +
    " streamsPerDevice=" + Global::intToString(streams) +
    " evaluatorThreads=" + Global::intToString(evaluatorThreads)
  );
  return plan;
#endif
}

void validateDevices(const Plan& plan, const vector<int>& gpuIdxByServerThread) {
#if !defined(USE_CUDA_BACKEND)
  (void)plan;
  (void)gpuIdxByServerThread;
  throw StringError("CUDA tactic plan device validation requires a CUDA backend build");
#else
  map<int,int> streamsByDevice;
  for(int deviceIdx : gpuIdxByServerThread)
    streamsByDevice[deviceIdx < 0 ? 0 : deviceIdx] += 1;
  for(const auto& item : streamsByDevice) {
    const int deviceIdx = item.first;
    if(item.second != plan.streamsPerDevice) {
      throw StringError(
        "CUDA tactic plan requires " + Global::intToString(plan.streamsPerDevice) +
        " streams on device " + Global::intToString(deviceIdx) + ", got " +
        Global::intToString(item.second)
      );
    }
    cudaDeviceProp prop;
    CUDA_ERR("CUDA tactic plan",cudaGetDeviceProperties(&prop,deviceIdx));
    requireSameString("GPU name",plan.device.name,string(prop.name),deviceIdx);
    requireSameInt("compute capability major",plan.device.computeCapabilityMajor,prop.major,deviceIdx);
    requireSameInt("compute capability minor",plan.device.computeCapabilityMinor,prop.minor,deviceIdx);
    requireSameInt("SM count",plan.device.multiProcessorCount,prop.multiProcessorCount,deviceIdx);
    requireSameInt("max threads/block",plan.device.maxThreadsPerBlock,prop.maxThreadsPerBlock,deviceIdx);
    requireSameInt("max threads/SM",plan.device.maxThreadsPerMultiprocessor,prop.maxThreadsPerMultiProcessor,deviceIdx);
    requireSameInt("registers/SM",plan.device.regsPerMultiprocessor,prop.regsPerMultiprocessor,deviceIdx);
    requireSameInt("opt-in shared memory/block",plan.device.sharedMemPerBlockOptin,(int)prop.sharedMemPerBlockOptin,deviceIdx);
    requireSameInt("shared memory/SM",plan.device.sharedMemPerMultiprocessor,(int)prop.sharedMemPerMultiprocessor,deviceIdx);
    requireSameInt("L2 size",plan.device.l2CacheSize,prop.l2CacheSize,deviceIdx);
    requireSameInt("memory bus width",plan.device.memoryBusWidth,prop.memoryBusWidth,deviceIdx);
    requireSameInt("async engine count",plan.device.asyncEngineCount,prop.asyncEngineCount,deviceIdx);
    requireSameInt("concurrent kernels",plan.device.concurrentKernels ? 1 : 0,prop.concurrentKernels ? 1 : 0,deviceIdx);
    requireSameUint64("global memory",plan.device.totalGlobalMem,(uint64_t)prop.totalGlobalMem,deviceIdx);
  }
#endif
}

}
