#include "../cudabackend_sm120_kernels.h"

namespace Sm120Backend {

const WideQKVRopeAotTactic* getSm120SearchQkvRopeFatTactics(
  std::size_t& count
) {
  count = 0;
  return nullptr;
}

} // namespace Sm120Backend
