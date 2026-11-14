// (attention copy of moe/local/cuda_stubs, extended) Minimal stand-ins so Apple clang can syntax-check CUDA source (-fsyntax-only, -nocudainc).
// Only declarations; nothing here is compiled for a GPU. Signatures follow the CUDA 13 headers.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <initializer_list>
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __align__(n) __attribute__((aligned(n)))
#define __cluster_dims__(...)
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#include <__clang_cuda_builtin_vars.h>
struct dim3 { unsigned x, y, z; __host__ __device__ dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
extern "C" __host__ __device__ int cudaConfigureCall(dim3 gridSize, dim3 blockSize, size_t sharedSize = 0, cudaStream_t stream = 0);
extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
struct uint4 { unsigned x, y, z, w; };
struct uint2 { unsigned x, y; };
struct float2 { float x, y; };
__device__ float2 make_float2(float x, float y);
struct float4 { float x, y, z, w; };   // vector_types.h (16 B aligned there)
__device__ float4 make_float4(float x, float y, float z, float w);
__device__ uint2 __ldg(const uint2* p);
__device__ long __shfl_sync(unsigned mask, long var, int src);
__device__ float __shfl_xor_sync(unsigned mask, float var, int lane);
__device__ float __uint_as_float(unsigned x);
__device__ float fmaxf(float a, float b);
__device__ float fmaf(float a, float b, float c);
__device__ int max(int a, int b);
__device__ uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w);
__device__ uint4 __ldcs(const uint4* p);
__device__ unsigned short __ldcs(const unsigned short* p);
__device__ uint4 __ldg(const uint4* p);
__device__ int __ldg(const int* p);
__device__ unsigned __byte_perm(unsigned x, unsigned y, unsigned s);
__device__ unsigned __ballot_sync(unsigned mask, int pred);
__device__ int __popc(unsigned x);
__device__ int atomicAdd(int* address, int val);
__device__ void __syncthreads();
__device__ float __expf(float x);
__device__ int min(int a, int b);
struct cudaFuncAttributes { size_t sharedSizeBytes, constSizeBytes, localSizeBytes; int maxThreadsPerBlock, numRegs; };
template <class T> cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, T* entry);
// CUDA runtime symbol copies (cuda_runtime_api.h; used by W3d's lab watchdog getter)
template <class T> cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0, int kind = 2);
template <class T> cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0, int kind = 1);

__device__ size_t __cvta_generic_to_shared(const void* p);
__device__ int atomicExch(int* address, int val);
__device__ void __syncwarp(unsigned mask = 0xffffffffu);
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8, cudaFuncAttributeNonPortableClusterSizeAllowed = 9 };
template <class T> cudaError_t cudaFuncSetAttribute(T* entry, cudaFuncAttribute attr, int value);
__device__ float __ldcg(const float* p);
__device__ void __threadfence();
__device__ float rsqrtf(float x);
__device__ void sincosf(float x, float* s, float* c);
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t e);
enum { cudaSuccess = 0 };
// PDL launches (cuda_runtime.h, CUDA 12+; same declarations as moe/local/cuda_stubs).
enum cudaLaunchAttributeID { cudaLaunchAttributeProgrammaticStreamSerialization = 5 };
union cudaLaunchAttributeValue { char pad[64]; int programmaticStreamSerializationAllowed; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; char pad[8 - sizeof(cudaLaunchAttributeID)]; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned int numAttrs; };
template <typename... ExpTypes, typename... ActTypes>
cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args) {
  // as cuda_runtime.h: the arguments are coerced to the kernel's parameter types (count and types must match)
  return [&](ExpTypes... coerced) -> cudaError_t { void* p[] = {&coerced...}; (void)p; (void)config; (void)kernel; return 0; }(
      static_cast<ActTypes&&>(args)...);
}
// CUDA runtime occupancy queries (cuda_runtime.h C++ templates; SGLang's cluster_probe.cuh uses the cluster one)
template <class T> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize);
template <class T> cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, T* func, const cudaLaunchConfig_t* config);
