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

extern "C" __global__ void linear2_residual_kernel(const half_t* __restrict__ input_tensor, half_t* __restrict__ output, const half_t* __restrict__ residual, const half_t* __restrict__ weights);
extern "C" __global__ void __launch_bounds__(128, 3) linear2_residual_kernel(const half_t* __restrict__ input_tensor, half_t* __restrict__ output, const half_t* __restrict__ residual, const half_t* __restrict__ weights) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  half_t output_local[128];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    *(uint2*)(output_local + (i * 4)) = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((int)blockIdx.y) * 147456) + (i_1 * 36864)) + ((((int)threadIdx.x) >> 2) * 1152)) + ((((int)threadIdx.x) & 3) * 8))])), (((((((int)blockIdx.y) * 128) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)weight_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(weights[((((i_2 * 3072) + ((((int)threadIdx.x) >> 4) * 384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(input_tensor[(((((((int)blockIdx.y) * 147456) + (i_3 * 36864)) + ((((int)threadIdx.x) >> 2) * 1152)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])), (((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)weight_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(weights[(((((i_4 * 3072) + ((((int)threadIdx.x) >> 4) * 384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 12288)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((i_5 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(input_tensor[(((((((int)blockIdx.y) * 147456) + (i_5 * 36864)) + ((((int)threadIdx.x) >> 2) * 1152)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])), (((((((int)blockIdx.y) * 128) + (i_5 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_5 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 4; ++i_6) {
    tl::cp_async_gs<16>((&(((half_t*)weight_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(weights[(((((i_6 * 3072) + ((((int)threadIdx.x) >> 4) * 384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 24576)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 33; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((((ko + 3) & 3) * 4096) + (i_7 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((((int)blockIdx.y) * 147456) + (i_7 * 36864)) + ((((int)threadIdx.x) >> 2) * 1152)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 96)])), (((((((int)blockIdx.y) * 128) + (i_7 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_7 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      tl::cp_async_gs<16>((&(((half_t*)weight_shared)[(((((((((ko + 3) & 3) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(weights[((((((ko * 12288) + (i_8 * 3072)) + ((((int)threadIdx.x) >> 4) * 384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 36864)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[(((((((ko & 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_9 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_9 * 8)])));
        }
        #pragma unroll
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((ko & 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_10 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_10 * 8)])));
        }
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_11 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_11 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_11 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t A_local_1[32];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_12 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_1[(i_12 * 8)])));
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_13 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])), (&(B_local_1[(i_13 * 8)])));
      }
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_14 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_14 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_14 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_14 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_2[32];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_15 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 8192)])), (&(A_local_2[(i_15 * 8)])));
      }
      #pragma unroll
      for (int i_16 = 0; i_16 < 4; ++i_16) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_2[(i_16 * 8)])));
      }
      for (int i_17 = 0; i_17 < 4; ++i_17) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_17 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_17 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_3[32];
    half_t B_local_3[32];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 4; ++i_18) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_18 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 12288)])), (&(A_local_3[(i_18 * 8)])));
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 4; ++i_19) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_19 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_19 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])), (&(B_local_3[(i_19 * 8)])));
      }
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_20 * 32) + (j_3 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_20 * 32) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_20 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 64; ++i_21) {
    if ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + ((i_21 >> 4) * 16)) + ((i_21 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 4693) {
      uint1 __1;
        uint1 v_ = *(uint1*)(output_local + (i_21 * 2));
        uint1 v__1 = *(uint1*)(residual + (((((((((((int)blockIdx.y) * 49152) + (((((int)threadIdx.x) & 63) >> 5) * 24576)) + ((i_21 >> 4) * 6144)) + ((i_21 & 1) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_21 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
      *(uint1*)(output_local + (i_21 * 2)) = __1;
    }
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 64; ++i_22) {
    if ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + ((i_22 >> 4) * 16)) + ((i_22 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 4693) {
      *(uint1*)(output + (((((((((((int)blockIdx.y) * 49152) + (((((int)threadIdx.x) & 63) >> 5) * 24576)) + ((i_22 >> 4) * 6144)) + ((i_22 & 1) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 384)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_22 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(output_local + (i_22 * 2));
    }
  }
#endif
}

namespace Sm120Backend {

cudaError_t launchLinear2ResidualB13(
  const half* input,
  const half* weights,
  half* residual,
  cudaStream_t stream
) {
  static const cudaError_t attributeStatus = cudaFuncSetAttribute(
    linear2_residual_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
  if(attributeStatus != cudaSuccess)
    return attributeStatus;
  dim3 grid(3, 37, 1);
  linear2_residual_kernel<<<grid, 128, 65536, stream>>>(
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<half_t*>(residual),
    reinterpret_cast<const half_t*>(residual),
    reinterpret_cast<const half_t*>(weights));
  return cudaPeekAtLastError();
}

} // namespace Sm120Backend
