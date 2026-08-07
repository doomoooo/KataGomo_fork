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

extern "C" __global__ void wide_qkv_s1_kernel(const half_t* __restrict__ input_tensor, half_t* __restrict__ output, const half_t* __restrict__ weights);
extern "C" __global__ void __launch_bounds__(128, 3) wide_qkv_s1_kernel(const half_t* __restrict__ input_tensor, half_t* __restrict__ output, const half_t* __restrict__ weights) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* input_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* weight_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  half_t output_local[128];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    *(uint2*)(output_local + (i * 4)) = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((int)blockIdx.y) * 49152) + (i_1 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + ((((int)threadIdx.x) & 3) * 8))])), (((((((int)blockIdx.y) * 128) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)weight_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(weights[((((i_2 * 9216) + ((((int)threadIdx.x) >> 4) * 1152)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((i_3 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(input_tensor[(((((((int)blockIdx.y) * 49152) + (i_3 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])), (((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)weight_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)])), (&(weights[(((((i_4 * 9216) + ((((int)threadIdx.x) >> 4) * 1152)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 36864)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 10; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)input_shared)[(((((((ko + 2) % 3) * 4096) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(input_tensor[((((((((int)blockIdx.y) * 49152) + (i_5 * 12288)) + ((((int)threadIdx.x) >> 2) * 384)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])), (((((((int)blockIdx.y) * 128) + (i_5 * 32)) + (((int)threadIdx.x) >> 2)) < 4693) && ((((((int)blockIdx.y) * 128) + (i_5 * 32)) + (((int)threadIdx.x) >> 2)) < 4693)));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)weight_shared)[(((((((((ko + 2) % 3) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(weights[((((((ko * 36864) + (i_6 * 9216)) + ((((int)threadIdx.x) >> 4) * 1152)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 73728)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      half_t A_local[32];
      half_t B_local[32];
      for (int ki = 0; ki < 2; ++ki) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[(((((((ko % 3) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (i_7 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8))])), (&(A_local[(i_7 * 8)])));
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((ko % 3) * 4096) + ((((int)threadIdx.x) >> 6) * 2048)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_8 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local[(i_8 * 8)])));
        }
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_9 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_9 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_1[32];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_10 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 4096)])), (&(A_local_1[(i_10 * 8)])));
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 4; ++i_11) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_11 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 4096)])), (&(B_local_1[(i_11 * 8)])));
      }
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_12 * 32) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_12 * 32) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_2[32];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((half_t*)input_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_13 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_2) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 8192)])), (&(A_local_2[(i_13 * 8)])));
      }
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)weight_shared)[((((((((int)threadIdx.x) >> 6) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_14 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_14 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(B_local_2[(i_14 * 8)])));
      }
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + ((i_15 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat16, 16, 8, 16, false, true>(reinterpret_cast<unsigned*>(output_local + (((i_15 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_15 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 64; ++i_16) {
    if ((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 63) >> 5) * 64)) + ((i_16 >> 4) * 16)) + ((i_16 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 4693) {
      *(uint1*)(output + (((((((((((((int)blockIdx.x) / 3) * 1802112) + (((int)blockIdx.y) * 49152)) + (((((int)threadIdx.x) & 63) >> 5) * 24576)) + ((i_16 >> 4) * 6144)) + ((i_16 & 1) * 3072)) + (((((int)threadIdx.x) & 31) >> 2) * 384)) + ((((int)blockIdx.x) % 3) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_16 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(output_local + (i_16 * 2));
    }
  }
#endif
}

namespace Sm120Backend {

cudaError_t launchWideQKVB13S1(
  const half* input,
  const half* weights,
  half* output,
  cudaStream_t stream
) {
  static const cudaError_t attributeStatus = cudaFuncSetAttribute(
    wide_qkv_s1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
  if(attributeStatus != cudaSuccess)
    return attributeStatus;
  dim3 grid(9, 37, 1);
  wide_qkv_s1_kernel<<<grid, 128, 49152, stream>>>(
    reinterpret_cast<const half_t*>(input),
    reinterpret_cast<half_t*>(output),
    reinterpret_cast<const half_t*>(weights));
  return cudaPeekAtLastError();
}

} // namespace Sm120Backend
