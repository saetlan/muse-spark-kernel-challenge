#pragma once
// Stand-in for the CUDA 13 driver API header: only the tensor-map types proj's WG kernel uses
// (cuda.h, CUDA 13.0: CUtensorMap is 128 bytes, 64-byte aligned).
#include "cuda_stub_prelude.h"
typedef uint64_t cuuint64_t;
typedef uint32_t cuuint32_t;
typedef enum cudaError_enum { CUDA_SUCCESS = 0 } CUresult;
typedef struct CUtensorMap_st { alignas(64) unsigned long long opaque[16]; } CUtensorMap;
typedef enum CUtensorMapDataType_enum { CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9 } CUtensorMapDataType;
typedef enum CUtensorMapInterleave_enum { CU_TENSOR_MAP_INTERLEAVE_NONE = 0 } CUtensorMapInterleave;
typedef enum CUtensorMapSwizzle_enum { CU_TENSOR_MAP_SWIZZLE_128B = 3 } CUtensorMapSwizzle;
typedef enum CUtensorMapL2promotion_enum { CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3 } CUtensorMapL2promotion;
typedef enum CUtensorMapFloatOOBfill_enum { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0 } CUtensorMapFloatOOBfill;
CUresult cuTensorMapEncodeTiled(CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank,
                                void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides,
                                const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave,
                                CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill);
