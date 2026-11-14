// decode_layer/local/cuda_stubs/cuda_stub_prelude.h + the cluster launch attribute (for dgrs.py's syntax check)
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
__device__ unsigned atomicOr(unsigned* address, unsigned val);
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
// M13 additions (CUDA 13 runtime / device function signatures).
struct uint2 { unsigned x, y; };
__device__ uint2 __ldcs(const uint2* p);
__device__ unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift);
__device__ unsigned __match_any_sync(unsigned mask, int value);
__device__ int __shfl_sync(unsigned mask, int var, int src, int width = 32);
__device__ int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32);
__device__ float __shfl_xor_sync(unsigned mask, float var, int lane_mask, int width = 32);
__device__ float __ldcg(const float* p);
__device__ void __stcg(float* p, float v);
__device__ float atomicAdd(float* address, float val);
__device__ float atomicExch(float* address, float val);  // v16 (MoE E19's split-K scratch: read and clear)
__device__ int atomicExch(int* address, int val);
__device__ void __threadfence();
__device__ float __uint_as_float(unsigned x);
__device__ float rsqrtf(float x);
enum cudaLaunchAttributeID { cudaLaunchAttributeClusterDimension = 4, cudaLaunchAttributeProgrammaticStreamSerialization = 5 };
union cudaLaunchAttributeValue { char pad[64]; int programmaticStreamSerializationAllowed; struct { unsigned int x, y, z; } clusterDim; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; char pad[8 - sizeof(cudaLaunchAttributeID)]; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned int numAttrs; };
enum { cudaSuccess = 0 };
const char* cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetLastError(void);  // v16 (MoE E19's M13 launcher)
template <typename... ExpTypes, typename... ActTypes>
cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args) {
  // as cuda_runtime.h: the arguments are coerced to the kernel's parameter types (count and types must match)
  return [&](ExpTypes... coerced) -> cudaError_t { void* p[] = {&coerced...}; (void)p; (void)config; (void)kernel; return 0; }(
      static_cast<ActTypes&&>(args)...);
}
template <class T> cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize);
// decode_layer additions (X5): signatures follow the CUDA 13 headers.
__device__ long long clock64();
__device__ unsigned __ldg(const unsigned* p);
__device__ uint2 __ldg(const uint2* p);  // v17 (MoE E25's M11b scale loads)
// CUDA runtime: cudaOccupancyMaxActiveClusters(int* numClusters, const void* func, const cudaLaunchConfig_t* config)
template <class T> cudaError_t cudaOccupancyMaxActiveClusters(int* numClusters, T* func, const cudaLaunchConfig_t* config);
