#include "../cudabackend_sm120_kernels.h"

namespace Sm120Backend {

const ResidualGemmAotTactic* getSm120SearchLinear2FatTactics(std::size_t& count) {
  count = 0;
  return nullptr;
}

} // namespace Sm120Backend
