#include "../neuralnet/cudabackend_sm89_tactic_kernels.h"

#include <cstring>

namespace Sm89Backend {

namespace {

bool matchesTopology(int entryStreams, int streams) {
  return entryStreams == 0 || entryStreams == streams;
}

} // namespace

const FusedFFNAotTactic* findSm89DualFfnTactic(
  int batchSize, int streams, const char* requestedId,
  bool& requestedIdKnown
) {
  requestedIdKnown = false;
  if(requestedId == nullptr || requestedId[0] == '\0' ||
     std::strcmp(requestedId, "disabled") == 0)
    return nullptr;
  std::size_t count = 0;
  const FusedFFNAotTactic* values = getSm89SearchDualFfnFatTactics(count);
  for(std::size_t i = 0; i < count; i++) {
    if(matchesTopology(values[i].streams, streams) &&
       std::strcmp(values[i].id, requestedId) == 0) {
      requestedIdKnown = true;
      if(values[i].batchSize == batchSize)
        return &values[i];
    }
  }
  return nullptr;
}

const ResidualGemmAotTactic* findSm89Linear2Tactic(
  int batchSize, int streams, int inputChannels,
  const char* requestedId, bool& requestedIdKnown
) {
  requestedIdKnown = false;
  if(requestedId == nullptr || requestedId[0] == '\0' ||
     std::strcmp(requestedId, "disabled") == 0)
    return nullptr;
  std::size_t count = 0;
  const ResidualGemmAotTactic* values = getSm89SearchLinear2FatTactics(count);
  for(std::size_t i = 0; i < count; i++) {
    if(values[i].inputChannels == inputChannels &&
       matchesTopology(values[i].streams, streams) &&
       std::strcmp(values[i].id, requestedId) == 0) {
      requestedIdKnown = true;
      if(values[i].batchSize == batchSize)
        return &values[i];
    }
  }
  return nullptr;
}

} // namespace Sm89Backend
