#include "../cudabackend_sm120_kernels.h"

namespace Sm120Backend {

const FA4AotTactic* getSm120SearchFA4FatTactics(std::size_t& count) {
  count = 0;
  return nullptr;
}

} // namespace Sm120Backend
