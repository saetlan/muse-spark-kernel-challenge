"""Derive parts/m11_pf.py from parts/m11_e3c.py (the verified snapshot of the MoE session's M11): the same
kernels plus L8, a next-task L2 prefetch. Every change is an exact, counted replacement, so the diff is
the whole story.

L8: M11 gives task `task + k * stride` to the same warp group next (stride = gridDim.x * GROUPS). With
pf = ahead | mode << 4 and ahead > 0, each warp, at the top of a task, asks L2 for its own K slice
(codes and scales, contiguous) of the task `ahead` strides on (on its first task, of every task 1..ahead
strides on). By the time the warp gets there (a whole task later, several us) the slice is in L2, so the
task's pipeline fill and loads see L2 latency instead of DRAM's. pf = 0 is M11 as before (bit-identical
results; the prefetch changes no data). Modes: 0 bulk (cp.async.bulk.prefetch.L2 by lane 0), 1 per-lane
prefetch.global.L2 of 128-B lines, 2 the same with L2::evict_last (PTX ISA 9.4, sm_90).

Run: .venv/bin/python decode_layer/local/make_m11pf.py
"""

from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]

HELPER = r'''// L8: L2 prefetch of one warp's K slice of a tile, kgw 64-wide chunks: 512 code bytes and 64 scale bytes
// per chunk, both contiguous and 16-B aligned. mode 0: two bulk prefetches by lane 0
// (cp.async.bulk.prefetch.L2.global, PTX ISA 9.4 9.7.10.28.4.3); 1: each lane prefetches 128-B lines
// (prefetch.global.L2, 9.7.10.16); 2: the same with .L2::evict_last. Hints only: no data changes.
__device__ __forceinline__ void m11_prefetch(const uint4* wp, const uint16_t* sp, int kgw, int mode, int lane) {
  const int wbytes = kgw * 512, sbytes = kgw * 64;
  if (mode == 0) {
    if (lane == 0) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(wp), "r"(wbytes));
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(sp), "r"(sbytes));
    }
    return;
  }
  const char* wc = reinterpret_cast<const char*>(wp);
  const char* sc = reinterpret_cast<const char*>(sp);
  for (int off = lane * 128; off < wbytes; off += 32 * 128) {
    if (mode == 1) asm volatile("prefetch.global.L2 [%0];" ::"l"(wc + off));
    else asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(wc + off));
  }
  for (int off = lane * 128; off < sbytes; off += 32 * 128) {
    if (mode == 1) asm volatile("prefetch.global.L2 [%0];" ::"l"(sc + off));
    else asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(sc + off));
  }
}

'''

PREFETCH = r'''    if (pf & 15) {  // L8: this warp's K slice of the task `ahead` strides on (on the first task: 1..ahead)
      const int ahead = pf & 15, stride = gridDim.x * GROUPS;
      for (int k = task < stride ? 1 : ahead; k <= ahead; ++k) {
        const int nxt = task + k * stride;
        if (nxt >= n_tasks) break;
        const int e2 = s_active[nxt / per_expert], tile2 = nxt % per_expert;
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          const int rt2 = GATED ? tile2 + i * (rt_total / 2) : tile2 * TT + i;
          const size_t b2 = (static_cast<size_t>(e2) * rt_total + rt2) * KG * 32 + static_cast<size_t>(kpart) * KGW * 32;
          m11_prefetch(w + b2, sf + b2, KGW, pf >> 4, lane);
        }
      }
    }
'''

LOOP = "  for (int task = blockIdx.x * GROUPS + grp; task < n_tasks; task += gridDim.x * GROUPS) {\n"

REPLACEMENTS = (
    # the kernel template: a prefetch helper before it, and the pf argument
    ("template <int TT, int NT, int D, int KS, bool GATED>\n__global__ void __launch_bounds__(128) gemm_kernel(\n",
     HELPER + "template <int TT, int NT, int D, int KS, bool GATED>\n__global__ void __launch_bounds__(128) gemm_kernel(\n"),
    ("    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {\n"
     "  constexpr int GROUPS = 4 / KS;  // tasks a CTA works on at once\n"
     "  constexpr int KGW = KG / KS;    // K chunks per warp\n"
     "  static_assert(KGW % D == 0",
     "    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int pf) {\n"
     "  constexpr int GROUPS = 4 / KS;  // tasks a CTA works on at once\n"
     "  constexpr int KGW = KG / KS;    // K chunks per warp\n"
     "  static_assert(KGW % D == 0"),
    # the prefetch at the top of each task (gemm_kernel's loop only: tma_kernel's differs)
    (LOOP, LOOP + PREFETCH),
    # M11_CASE passes pf (the M12_CASE launch keeps its arguments)
    ("    m11::gemm_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 0, stream>>>(                      \\\n"
     "        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \\\n"
     "        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \\\n"
     "        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \\\n"
     "        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \\\n"
     "        static_cast<int>(out_cols));                                                                     \\\n",
     "    m11::gemm_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 0, stream>>>(                      \\\n"
     "        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \\\n"
     "        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \\\n"
     "        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \\\n"
     "        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \\\n"
     "        static_cast<int>(out_cols), static_cast<int>(pf));                                               \\\n"),
    # the host function and its declaration
    ("                       int64_t ks, int64_t grid) {\n  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n  @@CASES@@\n",
     "                       int64_t ks, int64_t grid, int64_t pf) {\n  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n  @@CASES@@\n"),
    ("                       int64_t ks, int64_t grid);\ntorch::Tensor m11_attrs(",
     "                       int64_t ks, int64_t grid, int64_t pf);\ntorch::Tensor m11_attrs("),
    # the Python wrappers pass the config's pf (absent = 0)
    ('        state["m11_ext"].m11_gemm(*args, s1["d"], s1["ks"], s1["c"] * state["sms"])\n',
     '        state["m11_ext"].m11_gemm(*args, s1["d"], s1["ks"], s1["c"] * state["sms"], s1.get("pf", 0))\n'),
    ('        state["m11_ext"].m11_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"])\n',
     '        state["m11_ext"].m11_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"], s2.get("pf", 0))\n'),
    # its own extension name, so it can never be confused with the plain M11 module
    ('        name="m11_ext", cpp_sources=_M11_CPP,', '        name="m11pf_ext", cpp_sources=_M11_CPP,'),
)


def build(src=DECODE / "parts" / "m11_e3c.py", out=DECODE / "parts" / "m11_pf.py"):
    text = src.read_text()
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(f"expected exactly one {old.strip()[:70]!r} in {src.name}, found {text.count(old)}")
        text = text.replace(old, new)
    head = ("# parts/m11_pf.py: generated by local/make_m11pf.py from parts/m11_e3c.py; M11 + L8 (next-task L2\n"
            "# prefetch, runtime pf = ahead | mode << 4; pf = 0 is M11 unchanged). Do not edit by hand.\n")
    out.write_text(head + text)
    print(f"wrote {out} ({len(text)} chars, {len(REPLACEMENTS)} replacements)")


if __name__ == "__main__":
    build()
