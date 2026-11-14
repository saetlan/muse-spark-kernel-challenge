"""Generates attention/local/cuda_anat.py: instrumented copies of D3 (cuda_dec4.py), UD2
(cuda_unified2.py) and S2c (cuda_ver3.py) for the L15 anatomy lab, plus a micro module (pointer
chase for cold dependent-load latency, empty kernels for launch cost).

The copies are the verified kernels with only these changes, each at an anchor that must occur
exactly once (asserted):
- namespace okc_X -> okc_an_X; host entry points X -> an_X (separate modules, no clashes);
- one extra kernel argument `unsigned long long* an_buf` (32 slots per CTA) and host argument
  `torch::Tensor an_buf`;
- stamps: one thread per role stores %clock64 (and %globaltimer / %smid at entry and exit) at
  phase boundaries. Stamps are inline asm volatile, so they keep their order against the
  kernel's own asm volatile (cp.async, waits, mma); stamps that must follow a register value
  store that value first (the store waits for it). Wait times inside loops are accumulated
  in registers and stored after the loop.
Slots (D3): 0 gt entry, 1 clk entry, 2 page entries issued (seqlens arrived), 3 page entries
landed, 4 first K/V chunk landed (warp 0), 5 loop end, 6 accumulated K/V wait clocks, 7 warp
merge visible, 8 warp 0's chunk count, 28 clk end, 29 gt end, 30 smid.
Slots (UD2): 0, 1, 2, 3 as D3; 4 RoPE table done; 5 fold done (query fragments ready); 6 first
K/V landed; 7 loop end; 8 accumulated wait; 9 warp merge visible; 10 CTA partial in smem;
11 after cluster barrier 1; 12 after cluster barrier 2; 13 chunk count; 28, 29, 30 as D3.
Slots (S2c): consumer warp 0 (thread 0): 0, 1 entry; 2 query fragments ready; 3 first stage
landed; 4 loop end; 5 accumulated full-barrier wait clocks; 6 stages; 28, 29 end; producer
(thread 128): 16 start (after the CTA barrier); 17 page entries issued; 18 landed; 19 loop end;
20 after wait_all; 21 accumulated empty-barrier wait; 22 accumulated page-entry wait; 23 gt end;
30 smid.

Run: .venv/bin/python attention/local/make_anatomy.py [--check]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_anat.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var], ns[var.replace("_SRC", "_DECL")]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor found {n} times: {old[:80]!r}"
    return src.replace(old, new)


HELPERS = r"""
// ---- L15 anatomy stamps (lab only) ----
__device__ __forceinline__ unsigned long long an_clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long an_clk_m() {   // also ordered against memory operations
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
#define AN_PUT(k, v) do { if (threadIdx.x == 0) an_buf[an_base + (k)] = (v); } while (0)
#define AN_PP(k, v) do { if (threadIdx.x == 128) an_buf[an_base + (k)] = (v); } while (0)
#define AN_DEP(k, an_v) do { if (threadIdx.x == 0) { an_buf[an_base + 31] = static_cast<unsigned long long>(an_v); \
    an_buf[an_base + (k)] = an_clk_m(); } } while (0)
#define AN_END() do { if (threadIdx.x == 0) { an_buf[an_base + 28] = an_clk_m(); an_buf[an_base + 29] = an_gt(); } } while (0)
"""

ENTRY = ("\n  const long an_base = (static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x) * 32;\n"
         "  if (threadIdx.x == 0) { an_buf[an_base + 0] = an_gt(); an_buf[an_base + 1] = an_clk(); "
         "an_buf[an_base + 30] = an_smid(); }\n")

PT_PROLOGUE = ("  for (int j = 0; j < STAGES; ++j) issue_pt(j);\n"
               "  asm volatile(\"cp.async.commit_group;\");\n"
               "  asm volatile(\"cp.async.wait_group 0;\");\n"
               "  __syncwarp();\n")
LOOP_WAIT = ("  for (int i = 0; i < mine; ++i) {\n"
             "    issue(i + STAGES - 1);\n"
             "    asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));\n"
             "    __syncwarp();\n")
MERGE_SYNC = ("      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])"
              "   /* static indices: keeps acc in registers */;\n  __syncthreads();\n")


def common(src, decl, ns, old_name, new_name, sig_old, args_macro_tail, host_sig_tail, decl_sig_tail):
    """Renames, helpers, the extra argument and the entry stamps."""
    src = src.replace(f"namespace {ns} {{", f"namespace okc_an_{ns[4:]} {{\n" + HELPERS)
    src = src.replace(f"{ns}::", f"okc_an_{ns[4:]}::").replace(f"}}  // namespace {ns}", f"}}  // namespace okc_an_{ns[4:]}")
    src = re.sub(rf"\b{old_name}", new_name, src)
    decl = re.sub(rf"\b{old_name}", new_name, decl)
    src = sub1(src, sig_old, sig_old[:-3] + ", unsigned long long* __restrict__ an_buf) {" + ENTRY, f"{new_name} signature")
    src = sub1(src, args_macro_tail, args_macro_tail + ", reinterpret_cast<unsigned long long*>(an_buf.data_ptr())",
               f"{new_name} launch args")
    src = sub1(src, host_sig_tail, host_sig_tail[:-3] + ", torch::Tensor an_buf) {", f"{new_name} host signature")
    decl = sub1(decl, decl_sig_tail, decl_sig_tail[:-2] + ", torch::Tensor an_buf);", f"{new_name} decl")
    assert ns not in src.replace(f"okc_an_{ns[4:]}", ""), f"{ns} left in the source"
    return src, decl


def make_dec4():
    src, decl = load("cuda_dec4.py", "CUDA_DEC4_SRC")
    src, decl = common(src, decl, "okc_dec4", "attend_dec4", "an_dec4",
                       "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {",
                       "static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)",
                       "int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages) {",
                       "int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages);")
    src = sub1(src, PT_PROLOGUE, PT_PROLOGUE.replace("  asm volatile(\"cp.async.wait_group 0;\");\n",
               "  AN_PUT(2, an_clk());\n  asm volatile(\"cp.async.wait_group 0;\");\n")
               + "  AN_PUT(3, an_clk());\n  AN_PUT(8, mine);\n", "D3 page-entry prologue")
    src = sub1(src, LOOP_WAIT,
               "  unsigned long long an_wacc = 0;\n" + LOOP_WAIT.replace(
                   "    asm volatile(\"cp.async.wait_group %0;\"", "    const unsigned long long an_w0 = an_clk();\n"
                   "    asm volatile(\"cp.async.wait_group %0;\"")
               + "    const unsigned long long an_w1 = an_clk();\n    an_wacc += an_w1 - an_w0;\n"
                 "    if (i == 0) AN_PUT(4, an_w1);\n", "D3 loop wait")
    src = sub1(src, "  asm volatile(\"cp.async.wait_all;\");\n  __syncthreads();                     // every warp",
               "  AN_PUT(5, an_clk_m());\n  AN_PUT(6, an_wacc);\n"
               "  asm volatile(\"cp.async.wait_all;\");\n  __syncthreads();                     // every warp", "D3 loop end")
    src = sub1(src, MERGE_SYNC, MERGE_SYNC + "  AN_PUT(7, an_clk_m());\n", "D3 merge")
    src = sub1(src, "      l_part[part] = l;\n    }\n  }\n}\n", "      l_part[part] = l;\n    }\n  }\n  AN_END();\n}\n",
               "D3 end")
    return src, decl


def make_ud2():
    src, decl = load("cuda_unified2.py", "CUDA_UD2_SRC")
    src, decl = common(src, decl, "okc_ud2", "ud2", "an_ud2",
                       "    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {",
                       "static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)",
                       "int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs) {",
                       "double sm_scale, int64_t warps, int64_t stages, int64_t cs);")
    src = sub1(src, PT_PROLOGUE, PT_PROLOGUE.replace("  asm volatile(\"cp.async.wait_group 0;\");\n",
               "  AN_PUT(2, an_clk());\n  asm volatile(\"cp.async.wait_group 0;\");\n")
               + "  AN_PUT(3, an_clk());\n  AN_PUT(13, mine);\n", "UD2 page-entry prologue")
    src = sub1(src, "    __syncthreads();\n  }\n\n  // ---- new key/value",
               "    __syncthreads();\n  }\n  AN_PUT(4, an_clk_m());\n\n  // ---- new key/value", "UD2 rope")
    src = sub1(src, "  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp\n",
               "  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp\n"
               "  AN_DEP(5, qa[7][0]);\n", "UD2 fold")
    src = sub1(src, LOOP_WAIT,
               "  unsigned long long an_wacc = 0;\n" + LOOP_WAIT.replace(
                   "    asm volatile(\"cp.async.wait_group %0;\"", "    const unsigned long long an_w0 = an_clk();\n"
                   "    asm volatile(\"cp.async.wait_group %0;\"")
               + "    const unsigned long long an_w1 = an_clk();\n    an_wacc += an_w1 - an_w0;\n"
                 "    if (i == 0) AN_PUT(6, an_w1);\n", "UD2 loop wait")
    src = sub1(src, "  asm volatile(\"cp.async.wait_all;\");\n  __syncthreads();                     // every warp",
               "  AN_PUT(7, an_clk_m());\n  AN_PUT(8, an_wacc);\n"
               "  asm volatile(\"cp.async.wait_all;\");\n  __syncthreads();                     // every warp", "UD2 loop end")
    src = sub1(src, MERGE_SYNC, MERGE_SYNC + "  AN_PUT(9, an_clk_m());\n", "UD2 merge")
    src = sub1(src, "  // ---- A42 cluster merge", "  AN_PUT(10, an_clk_m());\n  // ---- A42 cluster merge", "UD2 partial")
    src = sub1(src, "  cluster_sync();                                   // every CTA's partial written and visible\n",
               "  cluster_sync();                                   // every CTA's partial written and visible\n"
               "  AN_PUT(11, an_clk_m());\n", "UD2 cluster barrier 1")
    src = sub1(src, "  cluster_sync();                                   // the leader is done reading: peers may exit\n"
               "  if (rank != 0) return;\n",
               "  cluster_sync();                                   // the leader is done reading: peers may exit\n"
               "  AN_PUT(12, an_clk_m());\n  if (rank != 0) { AN_END(); return; }\n", "UD2 cluster barrier 2")
    src = sub1(src, "    for (int k2 = 0; k2 < RPW; ++k2) finalize(warp + k2 * WARPS, fo0[k2], fo1[k2], fl[k2]);\n    return;\n",
               "    for (int k2 = 0; k2 < RPW; ++k2) finalize(warp + k2 * WARPS, fo0[k2], fo1[k2], fl[k2]);\n"
               "    AN_END();\n    return;\n", "UD2 single-cluster end")
    src = sub1(src, "  if (!last_flag) return;\n", "  if (!last_flag) { AN_END(); return; }\n", "UD2 ticket end")
    src = sub1(src, "  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;    // ready for the next call (and replay)\n}\n",
               "  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;    // ready for the next call (and replay)\n  AN_END();\n}\n",
               "UD2 end")
    return src, decl


def make_ver3():
    src, decl = load("cuda_ver3.py", "CUDA_VER3_SRC")
    src, decl = common(src, decl, "okc_ver3", "attend_ver3", "an_ver3",
                       "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {",
                       "static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)",
                       "int64_t page, int64_t window, double sm_scale, int64_t stages) {",
                       "int64_t page, int64_t window, double sm_scale, int64_t stages);")
    src = sub1(src, "  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  const int nst",
               "  AN_DEP(2, qa[1][3][3]);\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  const int nst",
               "S2c query fragments")
    src = sub1(src, "  if (warp == 4) {\n", "  if (warp == 4) {\n    AN_PP(16, an_clk_m());\n", "S2c producer start")
    src = sub1(src, "    for (int j = 0; j < STAGES; ++j) issue_pt(j);   // the one exposed page-table round trip\n"
               "    asm volatile(\"cp.async.commit_group;\");\n    asm volatile(\"cp.async.wait_group 0;\");\n    __syncwarp();\n",
               "    for (int j = 0; j < STAGES; ++j) issue_pt(j);   // the one exposed page-table round trip\n"
               "    asm volatile(\"cp.async.commit_group;\");\n    AN_PP(17, an_clk());\n"
               "    asm volatile(\"cp.async.wait_group 0;\");\n    __syncwarp();\n    AN_PP(18, an_clk());\n"
               "    unsigned long long an_eacc = 0, an_pacc = 0;\n", "S2c producer prologue")
    src = sub1(src, "      if (i >= STAGES) {                              // wait until the consumers released this stage\n",
               "      const unsigned long long an_e0 = an_clk();\n"
               "      if (i >= STAGES) {                              // wait until the consumers released this stage\n",
               "S2c producer empty wait start")
    src = sub1(src, "                       : \"=r\"(done) : \"r\"(smem_u32(&empty_bar[st])), \"r\"(parity) : \"memory\");\n      }\n",
               "                       : \"=r\"(done) : \"r\"(smem_u32(&empty_bar[st])), \"r\"(parity) : \"memory\");\n      }\n"
               "      an_eacc += an_clk() - an_e0;\n", "S2c producer empty wait end")
    src = sub1(src, "      asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));   // stage i+1's entries landed\n"
               "      __syncwarp();\n",
               "      const unsigned long long an_p0 = an_clk();\n"
               "      asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));   // stage i+1's entries landed\n"
               "      __syncwarp();\n      an_pacc += an_clk() - an_p0;\n", "S2c producer entry wait")
    src = sub1(src, "    asm volatile(\"cp.async.wait_all;\");\n    return;                                           // the producer",
               "    AN_PP(19, an_clk_m());\n    asm volatile(\"cp.async.wait_all;\");\n    AN_PP(20, an_clk_m());\n"
               "    AN_PP(21, an_eacc);\n    AN_PP(22, an_pacc);\n    AN_PP(23, an_gt());\n"
               "    return;                                           // the producer", "S2c producer end")
    src = sub1(src, "  for (int i = 0; i < nst; ++i) {\n    const int st = i % STAGES;\n"
               "    const uint32_t parity = static_cast<uint32_t>((i / STAGES) & 1);\n    uint32_t done = 0;\n",
               "  unsigned long long an_facc = 0;\n"
               "  for (int i = 0; i < nst; ++i) {\n    const int st = i % STAGES;\n"
               "    const uint32_t parity = static_cast<uint32_t>((i / STAGES) & 1);\n"
               "    const unsigned long long an_f0 = an_clk();\n    uint32_t done = 0;\n", "S2c consumer wait start")
    src = sub1(src, "                   : \"=r\"(done) : \"r\"(smem_u32(&full_bar[st])), \"r\"(parity) : \"memory\");\n",
               "                   : \"=r\"(done) : \"r\"(smem_u32(&full_bar[st])), \"r\"(parity) : \"memory\");\n"
               "    {\n      const unsigned long long an_f1 = an_clk();\n      an_facc += an_f1 - an_f0;\n"
               "      if (i == 0) AN_PUT(3, an_f1);\n    }\n", "S2c consumer wait end")
    src = sub1(src, "\n  // Partials: rows (token 2*mt + r",
               "\n  AN_PUT(4, an_clk_m());\n  AN_PUT(5, an_facc);\n  AN_PUT(6, nst);\n  // Partials: rows (token 2*mt + r",
               "S2c consumer loop end")
    src = sub1(src, "        l_part[part] = l;\n      }\n    }\n}\n", "        l_part[part] = l;\n      }\n    }\n  AN_END();\n}\n",
               "S2c end")
    return src, decl


MICRO_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

namespace okc_an_micro {

// One thread follows a chain of dependent loads (ld.global.cg: L2, no L1); clock64 around it.
__global__ void chase_kernel(const int* __restrict__ next, int start, int steps, unsigned long long* __restrict__ out) {
  if (threadIdx.x != 0) return;
  int idx = start;
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
  for (int s = 0; s < steps; ++s) {
    int v;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(next + idx) : "memory");
    idx = v;
  }
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
  out[0] = t1 - t0;
  out[1] = static_cast<unsigned long long>(idx);
}

__global__ void null_kernel(int* __restrict__ sink) {
  if (sink == nullptr) return;
}

// One cold load and one store per CTA (thread 0): a kernel with one exposed DRAM round trip.
__global__ void one_trip_kernel(const int* __restrict__ src, int* __restrict__ dst) {
  if (threadIdx.x == 0) dst[blockIdx.x] = src[static_cast<long>(blockIdx.x) * 4096];
}

__global__ void __cluster_dims__(8, 1, 1) null_cluster_kernel(int* __restrict__ sink) {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (sink == nullptr) return;
}

// Throughput: each thread runs `iters` x 8 independent operations of one kind; clock64 per CTA.
// KIND 0: mma.sync m16n8k16 f16 -> f32; 1: mma.sync m16n8k32 e4m3 -> f32; 2: ex2.approx.f32;
// 3: ex2.approx.f16x2; 4: cvt.rn.f16x2.e4m3x2. Values stay bounded; a never-true test keeps them live.
template <int KIND>
__global__ void bench_kernel(int iters, unsigned long long* __restrict__ out, float* __restrict__ sink) {
  const unsigned lane = threadIdx.x & 31;
  const uint32_t a0 = 0x3c003c00u ^ lane, a1 = a0 ^ 1u, a2 = a0 ^ 2u, a3 = a0 ^ 3u, b0 = 0x3c003c00u, b1 = b0 ^ 7u;
  float c[8][4];
  float e[8];
  uint32_t h[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
    e[j] = -static_cast<float>(lane + j) / 64.f;
    h[j] = 0xb800b800u ^ (lane << 3) ^ static_cast<uint32_t>(j);
  }
  __syncthreads();
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (KIND == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if (KIND == 1) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if (KIND == 2) {
        float y;
        asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(e[j]));
        e[j] = y - 1.0f;
      } else if (KIND == 3) {
        uint32_t y;
        asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(h[j]));
        h[j] = y ^ 0x80008000u;
      } else {
        uint32_t y;
        asm volatile("{\n\t.reg .b16 l, u;\n\tmov.b32 {l, u}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(y) : "r"(h[j]));
        h[j] = y ^ (h[j] >> 16);
      }
    }
  }
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3] + e[j] + static_cast<float>(h[j] & 0xffu);
  if (s == 1.2345e30f) sink[0] = s;
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

}  // namespace okc_an_micro

void an_bench(int64_t kind, int64_t ctas, int64_t threads, int64_t iters, torch::Tensor out, torch::Tensor sink) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto o = reinterpret_cast<unsigned long long*>(out.data_ptr());
  const dim3 grid(static_cast<unsigned>(ctas)), block(static_cast<unsigned>(threads));
  const int n = static_cast<int>(iters);
  float* sk = sink.data_ptr<float>();
  if (kind == 0) okc_an_micro::bench_kernel<0><<<grid, block, 0, stream>>>(n, o, sk);
  else if (kind == 1) okc_an_micro::bench_kernel<1><<<grid, block, 0, stream>>>(n, o, sk);
  else if (kind == 2) okc_an_micro::bench_kernel<2><<<grid, block, 0, stream>>>(n, o, sk);
  else if (kind == 3) okc_an_micro::bench_kernel<3><<<grid, block, 0, stream>>>(n, o, sk);
  else okc_an_micro::bench_kernel<4><<<grid, block, 0, stream>>>(n, o, sk);
}

void an_chase(torch::Tensor next, int64_t start, int64_t steps, torch::Tensor out) {
  okc_an_micro::chase_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      next.data_ptr<int>(), static_cast<int>(start), static_cast<int>(steps),
      reinterpret_cast<unsigned long long*>(out.data_ptr()));
}

// kind 0: empty kernel; 1: one cold load per CTA; 2: empty 8-CTA clusters with one cluster barrier.
// pdl: launch with programmatic stream serialization (the previous kernel in the stream may overlap).
void an_null(int64_t kind, int64_t ctas, int64_t threads, int64_t smem, int64_t pdl, torch::Tensor src, torch::Tensor dst) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(ctas), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaError_t e;
  if (kind == 1) e = cudaLaunchKernelEx(&cfg, okc_an_micro::one_trip_kernel, static_cast<const int*>(src.data_ptr<int>()), dst.data_ptr<int>());
  else if (kind == 2) e = cudaLaunchKernelEx(&cfg, okc_an_micro::null_cluster_kernel, dst.data_ptr<int>());
  else e = cudaLaunchKernelEx(&cfg, okc_an_micro::null_kernel, dst.data_ptr<int>());
  TORCH_CHECK(e == cudaSuccess, "an_null launch failed: ", cudaGetErrorString(e));
}

int64_t an_null_setup(int64_t smem) {   // once: allow `smem` bytes of dynamic shared memory
  int err = 0;
  err |= cudaFuncSetAttribute(okc_an_micro::null_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  err |= cudaFuncSetAttribute(okc_an_micro::null_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  (void)cudaGetLastError();
  return err;
}
"""

MICRO_DECL = ("void an_chase(torch::Tensor next, int64_t start, int64_t steps, torch::Tensor out);\n"
              "void an_null(int64_t kind, int64_t ctas, int64_t threads, int64_t smem, int64_t pdl, torch::Tensor src, "
              "torch::Tensor dst);\n"
              "int64_t an_null_setup(int64_t smem);\n"
              "void an_bench(int64_t kind, int64_t ctas, int64_t threads, int64_t iters, torch::Tensor out, "
              "torch::Tensor sink);")


def render():
    parts = ['"""Generated by make_anatomy.py (do not edit): instrumented D3 / UD2 / S2c and the micro module\n'
             'for the L15 anatomy lab; see make_anatomy.py for the slots."""\n']
    for var, (src, decl) in (("CUDA_AN_DEC4", make_dec4()), ("CUDA_AN_UD2", make_ud2()), ("CUDA_AN_VER3", make_ver3()),
                             ("CUDA_AN_MICRO", (MICRO_SRC, MICRO_DECL))):
        parts.append(f"\n{var}_SRC = r\"\"\"{src}\"\"\"\n\n{var}_DECL = {decl!r}\n")
    return "".join(parts)


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_anat.py up to date" if same else "cuda_anat.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
