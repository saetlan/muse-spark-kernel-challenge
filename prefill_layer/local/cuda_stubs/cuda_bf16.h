#pragma once
#include "cuda_stub_prelude.h"
struct __nv_bfloat16 { unsigned short x; };
__device__ __nv_bfloat16 __float2bfloat16_rn(float f);
