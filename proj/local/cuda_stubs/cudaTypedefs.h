#pragma once
// Stand-in for cudaTypedefs.h (CUDA 13): the versioned PFN type of cuTensorMapEncodeTiled.
#include "cuda.h"
typedef CUresult (*PFN_cuTensorMapEncodeTiled_v12000)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                                       const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                                       CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
