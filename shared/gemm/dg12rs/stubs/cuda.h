// Syntax-check stub of the CUDA driver API pieces dgrs.py uses (signatures as in cuda.h of CUDA 12/13).
#pragma once
typedef unsigned int cuuint32_t;
typedef unsigned long long cuuint64_t;
typedef int CUresult;
#define CUDA_SUCCESS 0
typedef struct alignas(64) CUtensorMap_st { unsigned long long opaque[16]; } CUtensorMap;
typedef enum { CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9 } CUtensorMapDataType;
typedef enum { CU_TENSOR_MAP_INTERLEAVE_NONE = 0 } CUtensorMapInterleave;
typedef enum { CU_TENSOR_MAP_SWIZZLE_NONE = 0, CU_TENSOR_MAP_SWIZZLE_128B = 3 } CUtensorMapSwizzle;
typedef enum { CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3 } CUtensorMapL2promotion;
typedef enum { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0 } CUtensorMapFloatOOBfill;
