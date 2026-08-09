#include "../cudabackend_sm120_kernels.h"

namespace Sm120Backend {

const WideQKVAotTactic* getSm120SearchQkvFatTactics(std::size_t& count) {
  count = 0;
  return nullptr;
}

} // namespace Sm120Backend
