/******************************************************************************
 * Fixed-board launcher derived from Dao-AILab/flash-attention, BSD-3-Clause.
 * Upstream commit: 5835c733e7e9c07606b045255768e8a7e9e851bd
 * CUTLASS submodule: 7127592069c2fe01b041e174ba4345ef9b279671
 ******************************************************************************/

#include "../neuralnet/cudabackend_sm89_flash.h"

#include "flash_fwd_launch_template.h"

#include <cmath>

namespace Sm89Backend {
namespace {

constexpr int S = 361;
constexpr int H = 12;
constexpr int D = 32;

Flash_fwd_params makeParams(
  const half* q,
  const half* k,
  const half* v,
  half* output,
  float* lse,
  int batchSize,
  int numSms
) {
  Flash_fwd_params p{};
  p.q_ptr = const_cast<half*>(q);
  p.k_ptr = const_cast<half*>(k);
  p.v_ptr = const_cast<half*>(v);
  p.o_ptr = output;
  p.softmax_lse_ptr = lse;

  constexpr int64_t rowStride = H * D;
  constexpr int64_t batchStride = S * rowStride;
  p.q_batch_stride = batchStride;
  p.k_batch_stride = batchStride;
  p.v_batch_stride = batchStride;
  p.o_batch_stride = batchStride;
  p.q_row_stride = rowStride;
  p.k_row_stride = rowStride;
  p.v_row_stride = rowStride;
  p.o_row_stride = rowStride;
  p.q_head_stride = D;
  p.k_head_stride = D;
  p.v_head_stride = D;
  p.o_head_stride = D;
  p.v_dim_stride = 1;

  p.b = batchSize;
  p.b_k = batchSize;
  p.h = H;
  p.h_k = H;
  p.seqlen_q = S;
  p.seqlen_k = S;
  p.seqlen_q_rounded = 384;
  p.seqlen_k_rounded = 384;
  p.d = D;
  p.dv = D;
  p.d_rounded = D;
  p.dv_rounded = D;
  p.total_q = batchSize * S;
  p.total_k = batchSize * S;

  p.scale_softmax = 1.0f / std::sqrt((float)D);
  p.p_dropout = 1.0f;
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.0f;
  p.window_size_left = S - 1;
  p.window_size_right = S - 1;
  p.num_splits = 1;
  p.pack_gqa = false;
  p.arch = 89;
  p.num_sm = numSms;
  return p;
}

} // namespace

size_t sm89FlashAttentionLseBytesD32(int batchSize) {
  return (size_t)batchSize * H * S * sizeof(float);
}

bool sm89FlashAttentionD32(
  const half* q,
  const half* k,
  const half* v,
  half* output,
  float* lse,
  int batchSize,
  int seqLen,
  int numHeads,
  int numKVHeads,
  int qHeadDim,
  int vHeadDim,
  bool useBoth16Accum,
  int numSms,
  cudaStream_t stream
) {
  if(batchSize < 1 || seqLen != S || numHeads != H || numKVHeads != H ||
     qHeadDim != D || vHeadDim != D || q == nullptr || k == nullptr ||
     v == nullptr || output == nullptr || lse == nullptr || numSms <= 0)
    return false;

  Flash_fwd_params p = makeParams(q, k, v, output, lse, batchSize, numSms);
  if(useBoth16Accum) {
    run_flash_fwd<
      86, D, D, 1,
      cutlass::half_t, cutlass::half_t,
      false, false, false, false, false, false, false, false, false, false,
      cutlass::half_t
    >(p, stream);
  }
  else {
    run_flash_fwd<
      86, D, D, 1,
      cutlass::half_t, cutlass::half_t,
      false, false, false, false, false, false, false, false, false, false
    >(p, stream);
  }
  return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace Sm89Backend
