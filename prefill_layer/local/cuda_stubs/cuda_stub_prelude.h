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
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
template <class T> cudaError_t cudaFuncSetAttribute(T* entry, cudaFuncAttribute attr, int value);
__device__ void __trap();
// Added for prefill R14 (prefill session's copy of MoE's stubs).
struct int4 { int x, y, z, w; };
__host__ __device__ int4 make_int4(int x, int y, int z, int w);
__device__ float expf(float x);
// Added for prefill R38 (v6): cluster launch API (CUDA 13 driver_types.h / cuda_runtime.h shapes).
enum cudaLaunchAttributeID { cudaLaunchAttributeClusterDimension = 4, cudaLaunchAttributeProgrammaticStreamSerialization = 5 };
union cudaLaunchAttributeValue { char pad[64]; int programmaticStreamSerializationAllowed; struct { unsigned int x, y, z; } clusterDim; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; char pad[8 - sizeof(cudaLaunchAttributeID)]; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned int numAttrs; };
template <typename... ExpTypes, typename... ActTypes>
cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args);
template <class T> cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, T* func, const cudaLaunchConfig_t* config);
extern "C" cudaError_t cudaGetLastError();
extern "C" const char* cudaGetErrorString(cudaError_t error);
constexpr cudaError_t cudaSuccess = 0;
__device__ long long clock64();
__device__ unsigned long long atomicAdd(unsigned long long* address, unsigned long long val);
__device__ int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32);
__device__ unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift);
