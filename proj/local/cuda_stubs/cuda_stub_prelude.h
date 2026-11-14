// Minimal stand-ins so Apple clang can syntax-check M11's CUDA source (-fsyntax-only, -nocudainc).
// Only declarations; nothing here is compiled for a GPU. Signatures follow the CUDA 13 headers.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <initializer_list>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __align__(n) __attribute__((aligned(n)))
#include <__clang_cuda_builtin_vars.h>
struct dim3 { unsigned x, y, z; __host__ __device__ dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
extern "C" __host__ __device__ int cudaConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, cudaStream_t stream = 0);
extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
struct uint4 { unsigned x, y, z, w; };
__device__ uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w);
__device__ uint4 __ldcs(const uint4* p);
__device__ unsigned short __ldcs(const unsigned short* p);
__device__ uint4 __ldg(const uint4* p);
__device__ unsigned __byte_perm(unsigned x, unsigned y, unsigned s);
__device__ unsigned __ballot_sync(unsigned mask, int pred);
__device__ int __popc(unsigned x);
__device__ int atomicAdd(int* address, int val);
__device__ void __syncthreads();
__device__ float __expf(float x);
__device__ int min(int a, int b);
struct cudaFuncAttributes { size_t sharedSizeBytes, constSizeBytes, localSizeBytes; int maxThreadsPerBlock, numRegs; };
template <class T> cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, T* entry);
__device__ size_t __cvta_generic_to_shared(const void* p);
__device__ void __syncwarp(unsigned mask = 0xffffffffu);
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8, cudaFuncAttributeNonPortableClusterSizeAllowed = 11 };
template <class T> cudaError_t cudaFuncSetAttribute(T* entry, cudaFuncAttribute attr, int value);
__device__ void __trap();
// Added for proj (PE3 microbenchmarks): cluster dims attribute and the 16-B vector type.
#define __cluster_dims__(...)  // nvcc defines it (host_defines.h); clang ignores the attribute
#define __restrict__ __restrict
// Added for proj P9: extended launch API (cluster dimension attribute), per the CUDA 13 runtime.
enum cudaLaunchAttributeID { cudaLaunchAttributeClusterDimension = 4 };
struct cudaLaunchAttributeValue { struct { unsigned x, y, z; } clusterDim; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned numAttrs; };
template <class... ExpTypes, class... ActTypes>
cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args);
// Added for proj P9: launch error checks (c10/cuda/CUDAException.h, CUDA runtime).
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetLastError();
#define cudaSuccess 0
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do { (void)cudaGetLastError(); } while (0)
// Added for proj P22: device intrinsics used by the F12 corrections (CUDA math API).
__device__ float __uint_as_float(unsigned x);
__device__ float fmaf(float a, float b, float c);
// Added for proj F12V (CUDA math API / warp shuffle, CUDA 13 headers).
__device__ unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift);
__device__ float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32);
// Added for proj WG (CUDA 13 runtime: versioned driver entry points; grid-constant kernel params).
enum cudaDriverEntryPointQueryResult { cudaDriverEntryPointSuccess = 0, cudaDriverEntryPointSymbolNotFound = 1 };
#define cudaEnableDefault 0x0
cudaError_t cudaGetDriverEntryPointByVersion(const char* symbol, void** funcPtr, unsigned int cudaVersion, unsigned long long flags,
                                             cudaDriverEntryPointQueryResult* driverStatus = 0);
#define __grid_constant__
__device__ float ldexpf(float x, int e);
struct float4 { float x, y, z, w; };

// added for proj F12K
__device__ float __ldcg(const float* p);
__device__ void __stcg(float* p, float v);
__device__ void __threadfence();
template <class T> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n, T* fn, int block, size_t smem);
