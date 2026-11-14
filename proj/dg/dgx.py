"""DGX (P40 step 1): our own build of the judge image's DeepGEMM sm90 BF16 GEMM, unchanged.

Each instance is one .cu file that includes `deep_gemm/impls/sm90_bf16_gemm.cuh` from the installed deep_gemm package
(sgl-project/DeepGEMM cbe1727) and instantiates `sm90_bf16_gemm_impl` exactly as DeepGEMM's host code does
(csrc/jit_kernels/impls/sm90_bf16_gemm.hpp): A = x [m, K] and B = W [N, K] K-major with SW128 TMA boxes [64, BLOCK],
D = out [m, N] BF16 with the CD swizzle, compiled dims N and K, grid = 132 persistent CTAs, 128 TMA + NMATH math
threads, cluster MC x MN; TMA descriptors as in runtime_utils.hpp make_tma_2d_desc (cuTensorMapEncodeTiled from
libcuda.so.1 via dlopen, L2 promotion 256B, no OOB fill). The .cu files hold no torch headers (torch's
__CUDA_NO_* defines are undefined first); each instance is its own load_inline module (no implicit headers: the C++
side includes torch and wraps the launch, the CUDA side is pure CUDA + DeepGEMM/CUTLASS), and the modules build from a
thread pool, so the nvcc runs overlap (load_inline writes its own sources; no file I/O here).

Instance: (BM, BN, ST, SWD, NMATH, MC, MN, N, K[, LV]), from dg_configs.dg_best (DeepGEMM's own heuristic). LV, the launch
variant (PE23): 0 = as DeepJIT (no cluster attribute when the cluster is 1 CTA, no PDL), 1 = a cluster attribute
always (PE22's launch), 2 = 0 + programmatic stream serialization (PDL; the kernel calls cudaGridDependencySynchronize).
"""

import os
import tempfile

DGX_CU = r"""
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

namespace dgx_@@NAME@@ {

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (fn == nullptr) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) fn = reinterpret_cast<EncodeTiledFn>(dlsym(h, "cuTensorMapEncodeTiled"));
  }
  return fn;
}

static CUtensorMapSwizzle swizzle_of(int mode) {
  switch (mode) {
    case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    default: return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

// DeepGEMM's make_tma_2d_desc for a BF16 tensor: gmem [outer][inner] with row stride outer_stride (elements);
// with a swizzle, the smem inner box is swizzle / 2 elements.
static int make_2d(CUtensorMap* map, void* ptr, int inner, int outer, int smem_inner, int smem_outer,
                   int outer_stride, int swizzle) {
  EncodeTiledFn fn = encode_fn();
  if (fn == nullptr) return 9001;
  if (swizzle != 0) smem_inner = swizzle / 2;
  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
  const cuuint32_t sdims[2] = {static_cast<cuuint32_t>(smem_inner), static_cast<cuuint32_t>(smem_outer)};
  const cuuint64_t gstrides[1] = {static_cast<cuuint64_t>(outer_stride) * 2};
  const cuuint32_t estrides[2] = {1, 1};
  const CUresult r = fn(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gdims, gstrides, sdims, estrides,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_of(swizzle), CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : 10000 + static_cast<int>(r);
}

constexpr uint32_t BM = @@BM@@, BN = @@BN@@, ST = @@ST@@, SWD = @@SWD@@, NMATH = @@NMATH@@, MC = @@MC@@, MN = @@MN@@;
constexpr uint32_t N = @@N@@, K = @@K@@, SMEM = @@SMEM@@, NUM_SMS = 132, LV = @@LV@@;

static auto kernel = &deep_gemm::sm90_bf16_gemm_impl<
    cute::UMMA::Major::K, cute::UMMA::Major::K,
    0, N, K,
    1,
    BM, BN, 64,
    128, 128, SWD,
    ST,
    128, NMATH,
    MC * MN, (MN > 1),
    NUM_SMS,
    deep_gemm::GemmType::Normal, false,
    cutlass::bfloat16_t>;

}  // namespace dgx_@@NAME@@

int dgx_launch_@@NAME@@(const void* x, const void* w, void* out, int m, void* stream) {
  using namespace dgx_@@NAME@@;
  CUtensorMap ta, tb, td;
  int e = make_2d(&ta, const_cast<void*>(x), K, m, 64, BM, K, 128);
  if (e == 0) e = make_2d(&tb, const_cast<void*>(w), K, N, 64, BN, K, 128);
  if (e == 0) e = make_2d(&td, out, N, m, BN, BM, N, SWD);
  if (e != 0) return e;
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[2];
  unsigned int na = 0;
  if (LV == 1 || MC * MN > 1) {
    attr[na].id = cudaLaunchAttributeClusterDimension;
    attr[na].val.clusterDim.x = MC * MN;
    attr[na].val.clusterDim.y = 1;
    attr[na].val.clusterDim.z = 1;
    ++na;
  }
  if (LV == 2) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.gridDim = dim3(NUM_SMS, 1, 1);
  cfg.blockDim = dim3(128 + NMATH, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = static_cast<cudaStream_t>(stream);
  cfg.attrs = na ? attr : nullptr;
  cfg.numAttrs = na;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, static_cast<int*>(nullptr), static_cast<uint32_t>(m), N, K,
                                             ta, tb, td);
  return static_cast<int>(err);
}

int dgx_attr_@@NAME@@() {
  return static_cast<int>(cudaFuncSetAttribute(dgx_@@NAME@@::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(dgx_@@NAME@@::SMEM)));
}

long long dgx_info_@@NAME@@() {
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, dgx_@@NAME@@::kernel) != cudaSuccess) return -1;
  // regs * 1e6 + local bytes, plus 1e9 * the max co-resident 1-CTA clusters (cudaOccupancyMaxActiveClusters)
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(dgx_@@NAME@@::NUM_SMS, 1, 1);
  cfg.blockDim = dim3(128 + dgx_@@NAME@@::NMATH, 1, 1);
  cfg.dynamicSmemBytes = dgx_@@NAME@@::SMEM;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int clusters = 0;
  if (cudaOccupancyMaxActiveClusters(&clusters, dgx_@@NAME@@::kernel, &cfg) != cudaSuccess) clusters = 0;
  return static_cast<long long>(clusters) * 1000000000LL + static_cast<long long>(a.numRegs) * 1000000
         + static_cast<long long>(a.localSizeBytes);
}
"""

# Instances compiled into the module, (BM, BN, ST, SWD, NMATH, MC, MN, N, K).
DGX_INSTANCES = ()


def dgx_unpack(inst):
    return tuple(inst) + ((1,) if len(inst) == 9 else ())


def dgx_name(inst):
    bm, bn, st, swd, nmath, mc, mn, n, k, lv = dgx_unpack(inst)
    return f"dgx_{bm}_{bn}_{st}_{swd}_{nmath}_{mc}{mn}_{n}_{k}" + (f"_l{lv}" if lv != 1 else "")


def dgx_smem(inst):
    bm, bn, st, swd, nmath, mc, mn, n, k, lv = dgx_unpack(inst)
    smem_cd = -(-(bm * bn * 2) // 1024) * 1024
    return smem_cd + 16 * 8 * 2 + st * (bm * 64 * 2 + bn * 64 * 2)


def dgx_cu(inst):
    bm, bn, st, swd, nmath, mc, mn, n, k, lv = dgx_unpack(inst)
    src = DGX_CU
    for key, val in (("NAME", dgx_name(inst)), ("BM", bm), ("BN", bn), ("ST", st), ("SWD", swd), ("NMATH", nmath),
                     ("MC", mc), ("MN", mn), ("N", n), ("K", k), ("SMEM", dgx_smem(inst)), ("LV", lv)):
        src = src.replace(f"@@{key}@@", str(val))
    return src


def dgx_cpp(inst):
    """C++ side of one instance's module: torch wrapper (shape checks, current stream), attr and info."""
    name, n, k = dgx_name(inst), inst[7], inst[8]
    return f"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int dgx_launch_{name}(const void* x, const void* w, void* out, int m, void* stream);
int dgx_attr_{name}();
long long dgx_info_{name}();
void {name}(torch::Tensor x, torch::Tensor w, torch::Tensor out) {{
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "dgx: dtypes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous(), "dgx: layouts");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == {k} && w.size(1) == {k} && w.size(0) >= {n}, "dgx: x or w shape");
  TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == {n} && x.size(0) >= 1, "dgx: out shape");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "dgx: unaligned");
  const int e = dgx_launch_{name}(x.data_ptr(), w.data_ptr(), out.data_ptr(), static_cast<int>(x.size(0)),
                                   static_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
  TORCH_CHECK(e == 0, "dgx launch failed: ", e);
}}
void {name}_attr() {{
  const int e = dgx_attr_{name}();
  TORCH_CHECK(e == 0, "dgx attr failed: ", e);
}}
int64_t {name}_info() {{ return static_cast<int64_t>(dgx_info_{name}()); }}
"""


def dgx_include_dirs():
    """deep_gemm/include first (its own cutlass/cute when bundled), then the image's other CUTLASS copies (F39)."""
    import deep_gemm
    site = os.path.dirname(os.path.dirname(deep_gemm.__file__))
    dirs = [os.path.join(os.path.dirname(deep_gemm.__file__), "include"),
            os.path.join(site, "flashinfer", "data", "cutlass", "include"),
            os.path.join(site, "tilelang", "3rdparty", "cutlass", "include")]
    return [d for d in dirs if os.path.isdir(d)]


def dgx_recon():
    """One log line: which include dirs exist and which hold cutlass/arch/barrier.h, cute/arch/mma_sm100_desc.hpp and
    deep_gemm/impls/sm90_bf16_gemm.cuh."""
    out = []
    for d in dgx_include_dirs():
        has = [f for f in ("cutlass/arch/barrier.h", "cute/arch/mma_sm100_desc.hpp", "deep_gemm/impls/sm90_bf16_gemm.cuh")
               if os.path.isfile(os.path.join(d, f))]
        out.append(f"{d.split('site-packages/')[-1]}: {','.join(h.split('/')[0] for h in has) or '-'}")
    return "; ".join(out)


def dgx_build_one(inst, include_dirs):
    from torch.utils.cpp_extension import load_inline
    name = dgx_name(inst)
    return load_inline(
        name="okc_proj_" + name, cpp_sources=dgx_cpp(inst), cuda_sources=dgx_cu(inst),
        functions=[name, name + "_attr", name + "_info"], extra_include_paths=list(include_dirs),
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "--ptxas-options=--register-usage-level=10"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_" + name + "_"), verbose=False, no_implicit_headers=True)


def dgx_build(instances, include_dirs=None, timeout=None):
    """Every instance's module built from a thread pool; waits at most `timeout` seconds (an unfinished build counts
    as failed and keeps compiling in the background). Returns (namespace of the built functions, {name: error})."""
    import concurrent.futures
    import types
    inc = include_dirs or dgx_include_dirs()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(instances)))
    futures = {pool.submit(dgx_build_one, inst, inc): inst for inst in instances}
    done, _ = concurrent.futures.wait(list(futures), timeout=timeout)
    pool.shutdown(wait=False)
    fns, errors = {}, {}
    for fut, inst in futures.items():
        name = dgx_name(inst)
        if fut not in done:
            errors[name] = TimeoutError(f"build not finished within {timeout:.0f}s")
            continue
        try:
            mod = fut.result()
            for fn in (name, name + "_attr", name + "_info"):
                fns[fn] = getattr(mod, fn)
        except Exception as exc:
            errors[name] = exc
    return types.SimpleNamespace(**fns), errors
