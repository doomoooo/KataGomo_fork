#include "../cudabackend_sm120_kernels.h"

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
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void fused_ffn_candidate_a_reuse_kernel(const half_t* __restrict__ gate_weights, const half_t* __restrict__ input_tensor, const half_t* __restrict__ linear_weights, half_t* __restrict__ output);
extern "C" __global__ void __launch_bounds__(128, 3) fused_ffn_candidate_a_reuse_kernel(const half_t* __restrict__ gate_weights, const half_t* __restrict__ input_tensor, const half_t* __restrict__ linear_weights, half_t* __restrict__ output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* linear_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* input_shared = ((void*)((char*)buf_dyn_shmem + 8192));
  void* gate_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  half_t linear_local[64];
  half_t gate_local[64];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
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
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)linear_shared)[(((((i_2 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(linear_weights[((((i_2 * 18432) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((int)blockIdx.y) * 49152) + (i_3 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + ((((int)threadIdx.x) & 3) * 8))])), (((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)gate_shared)[(((((i_4 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(gate_weights[((((i_4 * 18432) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)linear_shared)[((((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(linear_weights[(((((i_5 * 18432) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 36864)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((i_6 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(input_tensor[(((((((int)blockIdx.y) * 49152) + (i_6 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])), (((((((int)blockIdx.y) * 128) + (i_6 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_6 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    tl::cp_async_gs<16>((&(((half_t*)gate_shared)[((((((i_7 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 2048)])), (&(gate_weights[(((((i_7 * 18432) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 36864)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 10; ++ko) {
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[16];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[(((((((ko & 1) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_8 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_8 * 8)])));
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)linear_shared)[(((((ko & 1) * 2048) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_9) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_9 * 8)])));
        }
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + ((i_10 * 16) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + (((i_10 * 16) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_10 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
        #pragma unroll
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)gate_shared)[(((((ko & 1) * 2048) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_13 * 8)])));
        }
        for (int i_14 = 0; i_14 < 4; ++i_14) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + ((i_14 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local + (i_14 * 8)), reinterpret_cast<const unsigned*>(B_local + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + (((i_14 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_14 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      tl::cp_async_gs<16>((&(((half_t*)linear_shared)[(((((((ko & 1) * 2048) + (i_11 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(linear_weights[((((((ko * 36864) + (i_11 * 18432)) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 73728)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[((((((ko & 1) * 4096) + (i_15 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((((int)blockIdx.y) * 49152) + (i_15 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])), (((((((int)blockIdx.y) * 128) + (i_15 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_15 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      tl::cp_async_gs<16>((&(((half_t*)gate_shared)[(((((((ko & 1) * 2048) + (i_16 * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(gate_weights[((((((ko * 36864) + (i_16 * 18432)) + ((((int)threadIdx.x) >> 3) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 73728)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t A_local_2[32];
    half_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_17 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local_2[(i_17 * 8)])));
      }
      #pragma unroll
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)linear_shared)[(((ki_2 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_18) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_18 * 8)])));
      }
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + ((i_19 * 16) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_19 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + (((i_19 * 16) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_19 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 2; ++i_21) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)gate_shared)[(((ki_2 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_21) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_2[(i_21 * 8)])));
      }
      for (int i_22 = 0; i_22 < 4; ++i_22) {
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + ((i_22 * 16) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + (((i_22 * 16) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_4[32];
    half_t B_local_4[16];
    for (int ki_4 = 0; ki_4 < 2; ++ki_4) {
      #pragma unroll
      for (int i_23 = 0; i_23 < 4; ++i_23) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_23 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_4) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_4[(i_23 * 8)])));
      }
      #pragma unroll
      for (int i_24 = 0; i_24 < 2; ++i_24) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)linear_shared)[((((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_24) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(B_local_4[(i_24 * 8)])));
      }
      for (int i_25 = 0; i_25 < 4; ++i_25) {
        for (int j_4 = 0; j_4 < 2; ++j_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + ((i_25 * 16) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_25 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(linear_local + (((i_25 * 16) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_25 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
        }
      }
      #pragma unroll
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)gate_shared)[((((ki_4 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_27) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 2048)])), (&(B_local_4[(i_27 * 8)])));
      }
      for (int i_28 = 0; i_28 < 4; ++i_28) {
        for (int j_5 = 0; j_5 < 2; ++j_5) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + ((i_28 * 16) + (j_5 * 8))), reinterpret_cast<const unsigned*>(A_local_4 + (i_28 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_5 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(gate_local + (((i_28 * 16) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + (i_28 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_5 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 16; ++i_29) {
    float broadcast_var_2 = 0x1p+0f/*1.000000e+00*/;
    half_t broadcast_var_3 = half_t(-0x1p+0f/*-1.000000e+00*/);
    uint2 __1;
    float4 __2;
      float4 __3;
        float4 __4;
        uint2 v_ = *(uint2*)(linear_local + (i_29 * 4));
        ((float2*)(&__4))[0] = __half22float2(((half2*)(&v_))[0]);
        ((float2*)(&__4))[1] = __half22float2(((half2*)(&v_))[1]);
        float4 __5;
          float4 v__1 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
          float4 __6;
          uint2 __7;
          uint2 __8;
            uint2 v__2 = make_uint2(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
            *(uint1*)(&(__8.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__2.x)))));
            *(uint1*)(&(__8.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(v_.y))), tl::from_uint1<__half2>(*(uint1*)(&(v__2.y)))));
          ((half2*)(&(__7.x)))->x = hexp(((half2*)(&(__8.x)))->x);
          ((half2*)(&(__7.x)))->y = hexp(((half2*)(&(__8.x)))->y);
          ((half2*)(&(__7.y)))->x = hexp(((half2*)(&(__8.y)))->x);
          ((half2*)(&(__7.y)))->y = hexp(((half2*)(&(__8.y)))->y);
          ((float2*)(&__6))[0] = __half22float2(((half2*)(&__7))[0]);
          ((float2*)(&__6))[1] = __half22float2(((half2*)(&__7))[1]);
          *(float2*)(&(__5.x)) = tl::add2(*(float2*)(&(v__1.x)), *(float2*)(&(__6.x)));
          *(float2*)(&(__5.z)) = tl::add2(*(float2*)(&(v__1.z)), *(float2*)(&(__6.z)));
        __3.x = (__4.x/__5.x);
        __3.y = (__4.y/__5.y);
        __3.z = (__4.z/__5.z);
        __3.w = (__4.w/__5.w);
      float4 __9;
      uint2 v__3 = *(uint2*)(gate_local + (i_29 * 4));
      ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__3))[0]);
      ((float2*)(&__9))[1] = __half22float2(((half2*)(&v__3))[1]);
      *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(__9.x)));
      *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(__9.z)));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
    *(uint2*)(linear_local + (i_29 * 4)) = __1;
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 32; ++i_30) {
    if ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + ((i_30 >> 3) * 16)) + ((i_30 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 4693) {
      *(uint1*)(output + (((((((((((int)blockIdx.y) * 147456) + (((((int)threadIdx.x) & 63) >> 5) * 73728)) + ((i_30 >> 3) * 18432)) + ((i_30 & 1) * 9216)) + (((((int)threadIdx.x) & 31) >> 2) * 1152)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (((i_30 & 7) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(linear_local + (i_30 * 2));
    }
  }
#endif
}

namespace Sm120Backend {

void launchFusedFFNB13CandidateAReuse(
  const half* input,
  const half* linearWeights,
  const half* gateWeights,
  half* output,
  cudaStream_t stream
) {
  dim3 grid(18, 37, 1);
  fused_ffn_candidate_a_reuse_kernel<<<grid, 128, 32768, stream>>>(
    reinterpret_cast<const half_t*>(gateWeights),
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<const half_t*>(linearWeights),
    reinterpret_cast<half_t*>(output));
}

} // namespace Sm120Backend
