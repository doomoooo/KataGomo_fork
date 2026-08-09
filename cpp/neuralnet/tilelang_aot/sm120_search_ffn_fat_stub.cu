#include "../cudabackend_sm120_kernels.h"

namespace Sm120Backend {

const FusedFFNAotTactic* getSm120SearchFfnFatTactics(std::size_t& count) {
  count = 0;
  return nullptr;
}

} // namespace Sm120Backend
