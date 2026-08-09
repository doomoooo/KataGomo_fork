
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
__device__ __forceinline__ cutlass::half_t tilelang_htanh_approx(
    cutlass::half_t value) {
  return cutlass::half_t(htanh_approx(value.to_half()));
}
__device__ __forceinline__ unsigned tilelang_h2tanh_approx(unsigned bits) {
  union PackedHalf2 {
    unsigned bits;
    __half2 value;
  } input, output;
  input.bits = bits;
  output.value = h2tanh_approx(input.value);
  return output.bits;
}
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void katago_ffn_tilelang_sm120_b1_s361_kernel(const half_t* __restrict__ data, const half_t* __restrict__ gate_weight, const half_t* __restrict__ linear_weight, half_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 3) katago_ffn_tilelang_sm120_b1_s361_kernel(const half_t* __restrict__ data, const half_t* __restrict__ gate_weight, const half_t* __restrict__ linear_weight, half_t* __restrict__ output) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* data_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* output_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* linear_weight_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* gate_weight_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  half_t linear_local[64];
  half_t gate_local[64];
  half_t data_local[32];
  half_t linear_weight_local[16];
  half_t gate_weight_local[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    *(uint2*)(linear_local + (i * 4)) = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
    *(uint2*)(gate_local + (i_1 * 4)) = make_uint2(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)data_shared)[((((i_2 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(data[((((((int)blockIdx.y) * 49152) + (i_2 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + ((((int)threadIdx.x) & 3) * 8))])), (((((((int)blockIdx.y) * 128) + (i_2 * 32)) + (((int)threadIdx.x) >> 2)) < 361) && ((((((int)blockIdx.y) * 128) + (i_2 * 32)) + (((int)threadIdx.x) >> 2)) < 361)));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)linear_weight_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(linear_weight[((((i_3 * 18432) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)gate_weight_shared)[(((((i_4 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(gate_weight[((((i_4 * 18432) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  for (int tile_k = 0; tile_k < 11; ++tile_k) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)data_shared)[(((((((tile_k + 1) & 1) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(data[((((((((int)blockIdx.y) * 49152) + (i_5 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + (tile_k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])), (((((((int)blockIdx.y) * 128) + (i_5 * 32)) + (((int)threadIdx.x) >> 2)) < 361) && ((((((int)blockIdx.y) * 128) + (i_5 * 32)) + (((int)threadIdx.x) >> 2)) < 361)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)linear_weight_shared)[((((((((tile_k + 1) & 1) * 2048) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(linear_weight[((((((tile_k * 36864) + (i_6 * 18432)) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 36864)])));
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      tl::cp_async_gs<16>((&(((half_t*)gate_weight_shared)[((((((((tile_k + 1) & 1) * 2048) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(gate_weight[((((((tile_k * 36864) + (i_7 * 18432)) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 36864)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)data_shared)[(((((((tile_k & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_8 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(data_local[(i_8 * 8)])));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 2; ++i_9) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)linear_weight_shared)[(((((tile_k & 1) * 2048) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_9) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(linear_weight_local[(i_9 * 8)])));
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 2; ++i_10) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)gate_weight_shared)[(((((tile_k & 1) * 2048) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_10) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(gate_weight_local[(i_10 * 8)])));
      }
      for (int atom_m = 0; atom_m < 4; ++atom_m) {
        for (int atom_n = 0; atom_n < 2; ++atom_n) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + ((atom_m * 16) + (atom_n * 8))), reinterpret_cast<const unsigned*>(data_local + (atom_m * 8)), reinterpret_cast<const unsigned*>(linear_weight_local + (atom_n * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + (((atom_m * 16) + (atom_n * 8)) + 4)), reinterpret_cast<const unsigned*>(data_local + (atom_m * 8)), reinterpret_cast<const unsigned*>(linear_weight_local + ((atom_n * 8) + 4)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + ((atom_m * 16) + (atom_n * 8))), reinterpret_cast<const unsigned*>(data_local + (atom_m * 8)), reinterpret_cast<const unsigned*>(gate_weight_local + (atom_n * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + (((atom_m * 16) + (atom_n * 8)) + 4)), reinterpret_cast<const unsigned*>(data_local + (atom_m * 8)), reinterpret_cast<const unsigned*>(gate_weight_local + ((atom_n * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      tl::ptx_ldmatrix_x4((&(((half_t*)data_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_11 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(data_local[(i_11 * 8)])));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)linear_weight_shared)[((((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_12) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(linear_weight_local[(i_12 * 8)])));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)gate_weight_shared)[((((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(gate_weight_local[(i_13 * 8)])));
    }
    for (int atom_m_1 = 0; atom_m_1 < 4; ++atom_m_1) {
      for (int atom_n_1 = 0; atom_n_1 < 2; ++atom_n_1) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + ((atom_m_1 * 16) + (atom_n_1 * 8))), reinterpret_cast<const unsigned*>(data_local + (atom_m_1 * 8)), reinterpret_cast<const unsigned*>(linear_weight_local + (atom_n_1 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + (((atom_m_1 * 16) + (atom_n_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(data_local + (atom_m_1 * 8)), reinterpret_cast<const unsigned*>(linear_weight_local + ((atom_n_1 * 8) + 4)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + ((atom_m_1 * 16) + (atom_n_1 * 8))), reinterpret_cast<const unsigned*>(data_local + (atom_m_1 * 8)), reinterpret_cast<const unsigned*>(gate_weight_local + (atom_n_1 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + (((atom_m_1 * 16) + (atom_n_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(data_local + (atom_m_1 * 8)), reinterpret_cast<const unsigned*>(gate_weight_local + ((atom_n_1 * 8) + 4)));
      }
    }
  }
  uint1 half2 = uint1{tl::pack_half2(half_t(0x1p-1f/*5.000000e-01*/), half_t(0x1p-1f/*5.000000e-01*/))};
  #pragma unroll
  for (int pair_index = 0; pair_index < 32; ++pair_index) {
    uint1 linear_pair = uint1{tl::pack_half2(linear_local[(pair_index * 2)], linear_local[((pair_index * 2) + 1)])};
    uint1 gate_pair = uint1{tl::pack_half2(gate_local[(pair_index * 2)], gate_local[((pair_index * 2) + 1)])};
    uint1 v_ = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(linear_pair), tl::from_uint1<__half2>(uint1{tl::pack_half2(half_t(0x1p-1f/*5.000000e-01*/), half_t(0x1p-1f/*5.000000e-01*/))})));
    uint v__1 = tilelang_h2tanh_approx((*(uint *)(&(v_))));
    uint1 tanh_pair = (*(uint1 *)(&(v__1)));
    uint1 sigmoid_pair = tl::to_uint1(tl::fma2(tl::from_uint1<__half2>(tanh_pair), tl::from_uint1<__half2>(uint1{tl::pack_half2(half_t(0x1p-1f/*5.000000e-01*/), half_t(0x1p-1f/*5.000000e-01*/))}), tl::from_uint1<__half2>(uint1{tl::pack_half2(half_t(0x1p-1f/*5.000000e-01*/), half_t(0x1p-1f/*5.000000e-01*/))})));
    uint1 v__2 = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(linear_pair), tl::from_uint1<__half2>(tl::to_uint1(tl::fma2(tl::from_uint1<__half2>(tanh_pair), tl::from_uint1<__half2>(uint1{tl::pack_half2(half_t(0x1p-1f/*5.000000e-01*/), half_t(0x1p-1f/*5.000000e-01*/))}), tl::from_uint1<__half2>(uint1{tl::pack_half2(half_t(0x1p-1f/*5.000000e-01*/), half_t(0x1p-1f/*5.000000e-01*/))}))))))), tl::from_uint1<__half2>(gate_pair)));
    uint result_bits = (*(uint *)(&(v__2)));
    ushort v__3 = (ushort)((*(uint *)(&(v__2))) & (uint)65535);
    linear_local[(pair_index * 2)] = (*(half_t *)(&(v__3)));
    ushort v__4 = (ushort)((*(uint *)(&(v__2))) >> (uint)16);
    linear_local[((pair_index * 2) + 1)] = (*(half_t *)(&(v__4)));
  }
  __syncthreads();
  for (int i_14 = 0; i_14 < 4; ++i_14) {
    for (int j = 0; j < 2; ++j) {
      for (int local_id_o = 0; local_id_o < 4; ++local_id_o) {
        *(uint1*)(((half_t*)output_shared) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_14 * 1024)) + ((local_id_o & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + j) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (local_id_o >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(linear_local + (((i_14 * 16) + (j * 8)) + (local_id_o * 2)));
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_15 = 0; i_15 < 8; ++i_15) {
    if ((((((int)blockIdx.y) * 128) + (i_15 * 16)) + (((int)threadIdx.x) >> 3)) < 361) {
      *(uint4*)(output + (((((((int)blockIdx.y) * 147456) + (i_15 * 18432)) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((half_t*)output_shared) + (((((i_15 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}

