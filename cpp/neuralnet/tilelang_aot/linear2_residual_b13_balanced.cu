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

extern "C" __global__ void linear2_residual_balanced_kernel(const half_t* __restrict__ input_tensor, half_t* __restrict__ output, const half_t* __restrict__ residual, const half_t* __restrict__ weights);
extern "C" __global__ void __launch_bounds__(256, 2) linear2_residual_balanced_kernel(const half_t* __restrict__ input_tensor, half_t* __restrict__ output, const half_t* __restrict__ residual, const half_t* __restrict__ weights) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_shared = ((void*)((char*)buf_dyn_shmem + 65536));
  half_t output_local[64];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    *(uint2*)(output_local + (i * 4)) = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((int)blockIdx.y) * 294912) + (i_1 * 36864)) + ((((int)threadIdx.x) >> 3) * 1152)) + ((((int)threadIdx.x) & 7) * 8))])), (((((((int)blockIdx.y) * 256) + (i_1 * 32)) + (((int)threadIdx.x) >> 3)) < 4693) && ((((((int)blockIdx.y) * 256) + (i_1 * 32)) + (((int)threadIdx.x) >> 3)) < 4693)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)weight_shared)[(((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(weights[((((i_2 * 12288) + ((((int)threadIdx.x) >> 3) * 384)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8))])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 17; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[((((((((ko + 1) & 1) * 16384) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((((int)blockIdx.y) * 294912) + (i_3 * 36864)) + ((((int)threadIdx.x) >> 3) * 1152)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 64)])), (((((((int)blockIdx.y) * 256) + (i_3 * 32)) + (((int)threadIdx.x) >> 3)) < 4693) && ((((((int)blockIdx.y) * 256) + (i_3 * 32)) + (((int)threadIdx.x) >> 3)) < 4693)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      tl::cp_async_gs<16>((&(((half_t*)weight_shared)[((((((((ko + 1) & 1) * 4096) + (i_4 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(weights[((((((ko * 24576) + (i_4 * 12288)) + ((((int)threadIdx.x) >> 3) * 384)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 24576)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[64];
      half_t B_local[8];
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 8; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((ko & 1) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + (i_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[(i_5 * 8)])));
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[(((((ko & 1) * 4096) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[0])));
        for (int i_6 = 0; i_6 < 8; ++i_6) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 8)), reinterpret_cast<const unsigned*>(B_local + 4));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_1[64];
    half_t B_local_1[8];
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[(((((((((int)threadIdx.x) & 63) >> 5) * 8192) + (i_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])), (&(A_local_1[(i_7 * 8)])));
      }
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((ki_1 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])), (&(B_local_1[0])));
      for (int i_8 = 0; i_8 < 8; ++i_8) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_8 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    if ((((((((int)blockIdx.y) * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + ((i_9 >> 2) * 16)) + ((i_9 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 4693) {
      uint1 __1;
        uint1 v_ = *(uint1*)(output_local + (i_9 * 2));
        uint1 v__1 = *(uint1*)(residual + (((((((((((int)blockIdx.y) * 98304) + (((((int)threadIdx.x) & 63) >> 5) * 49152)) + ((i_9 >> 2) * 6144)) + ((i_9 & 1) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 384)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_9 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
        *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
      *(uint1*)(output_local + (i_9 * 2)) = __1;
    }
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 32; ++i_10) {
    if ((((((((int)blockIdx.y) * 256) + (((((int)threadIdx.x) & 63) >> 5) * 128)) + ((i_10 >> 2) * 16)) + ((i_10 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 4693) {
      *(uint1*)(output + (((((((((((int)blockIdx.y) * 98304) + (((((int)threadIdx.x) & 63) >> 5) * 49152)) + ((i_10 >> 2) * 6144)) + ((i_10 & 1) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 384)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(output_local + (i_10 * 2));
    }
  }
#endif
}

namespace Sm120Backend {

cudaError_t launchLinear2ResidualB13Balanced(
  const half* input,
  const half* weights,
  half* residual,
  cudaStream_t stream
) {
  static const cudaError_t attributeStatus = cudaFuncSetAttribute(
    linear2_residual_balanced_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 81920);
  if(attributeStatus != cudaSuccess)
    return attributeStatus;
  dim3 grid(6, 19, 1);
  linear2_residual_balanced_kernel<<<grid, 256, 81920, stream>>>(
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<half_t*>(residual),
    reinterpret_cast<const half_t*>(residual),
    reinterpret_cast<const half_t*>(weights));
  return cudaPeekAtLastError();
}

} // namespace Sm120Backend
