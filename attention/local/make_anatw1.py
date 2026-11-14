"""Generates attention/local/cuda_anatw1.py: an instrumented copy of W1-C (cuda_w1c.py: W1 with wgmma pairs in one
asm statement and memory clobbers on its cp.async waits, make_w1m.py) for an anatomy lab (A64; V5's method, make_anat5.py), to split W1's ~1.18 us 64-key stage (L20:
~2,200 cycles vs a ~400-cycle softmax floor, AF47, and a ~870-cycle memory floor) between waiting for data, the
tensor-core pairs, the softmax, the copier and the transposers, before designing W2 (A61).
Each change sits at an anchor that must occur exactly once (asserted): namespace okc_w1 -> okc_aw1, attend_w1 ->
an_w1 (one extra argument an_buf: 64 x u64 per CTA); %clock64 stamps in asm volatile; a phase end is read by an asm
that takes the phase's last result as an input, so in-order issue makes it wait for that value.
Slots per CTA: consumer warp w (0-7), lane 0, 6w + k: 0 waiting on full_bar, 1 Q.K pair (until the scores are
usable), 2 masks + max + exp + sums + P pack, 3 P.V pair, 4 empty arrive + O update, 5 entry -> loop end;
copier lane 0: 48 waiting on empty_bar, 49 waiting on its cp.async groups (page entries / data), 50 entry -> loop end;
transposer warp 9 / 10 lane 0: 52 / 54 waiting on land_bar, 53 / 55 entry -> loop end; thread 0: 56 clock at entry,
57 clock at exit, 58 SM id, 59 stages of the slice.
Run: .venv/bin/python attention/local/make_anatw1.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_anatw1.py")


def load(fname):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns["CUDA_W1C_SRC"], ns["CUDA_W1C_DECL"]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor found {n} times: {old[:90]!r}"
    return src.replace(old, new)


STAMPS = r"""
// ---- anatomy stamps (lab only) ----
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
__device__ __forceinline__ unsigned long long an_dep(float x) {   // issues once x is available
  unsigned long long t;
  asm volatile("{\n\t.reg .b32 an_t;\n\tmov.b32 an_t, %1;\n\tmov.u64 %0, %%clock64;\n\t}" : "=l"(t) : "f"(x));
  return t;
}
__device__ __forceinline__ unsigned long long an_dep_u(uint32_t x) {
  unsigned long long t;
  asm volatile("{\n\t.reg .b32 an_t;\n\tmov.b32 an_t, %1;\n\tmov.u64 %0, %%clock64;\n\t}" : "=l"(t) : "r"(x));
  return t;
}
__device__ __forceinline__ unsigned long long an_smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

"""


def render():
    src, decl = load("cuda_w1c.py")
    src = src.replace("okc_w1c", "okc_aw1").replace("attend_w1c", "an_w1").replace("OKC_W1C_", "OKC_AW1_")
    decl = decl.replace("attend_w1c", "an_w1")
    src = sub1(src, "template <int STAGES, int SBOV, int WF>\n__global__", STAMPS + "template <int STAGES, int SBOV, int WF>\n__global__", "stamps")
    src = sub1(src, "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n"
                    "  constexpr int PT_SLOTS = 2 * STAGES;",
               "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale,\n"
               "    unsigned long long* __restrict__ an_buf) {\n"
               "  const long an_base = (static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x) * 64;\n"
               "  const unsigned long long an_t0 = an_clk();\n"
               "  constexpr int PT_SLOTS = 2 * STAGES;", "kernel args")
    # copier
    src = sub1(src, "    for (int i = 0; i < nst; ++i) {\n      const int st = i % STAGES;\n"
                    "      if (i >= STAGES) bar_wait(&empty_bar[st], static_cast<uint32_t>((i / STAGES - 1) & 1));\n",
               "    unsigned long long an_we = 0, an_wp = 0;\n"
               "    for (int i = 0; i < nst; ++i) {\n      const int st = i % STAGES;\n"
               "      if (i >= STAGES) {\n        const unsigned long long an_e0 = an_clk_m();\n"
               "        bar_wait(&empty_bar[st], static_cast<uint32_t>((i / STAGES - 1) & 1));\n"
               "        an_we += an_clk_m() - an_e0;\n      }\n", "copier empty wait")
    src = sub1(src, "        asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1) : \"memory\");   // stage i+1's entries landed\n",
               "        const unsigned long long an_p0 = an_clk_m();\n"
               "        asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1) : \"memory\");   // stage i+1's entries landed\n"
               "        an_wp += an_clk_m() - an_p0;\n", "copier group wait")
    src = sub1(src, "    asm volatile(\"cp.async.wait_all;\" ::: \"memory\");\n    return;\n  }\n",
               "    asm volatile(\"cp.async.wait_all;\" ::: \"memory\");\n"
               "    if (lane == 0) {\n      an_buf[an_base + 48] = an_we;\n      an_buf[an_base + 49] = an_wp;\n"
               "      an_buf[an_base + 50] = an_clk() - an_t0;\n    }\n    return;\n  }\n", "copier store")
    # transposers
    src = sub1(src, "    for (int j = 0; j < nst; ++j) {\n      const int st = j % STAGES;\n"
                    "      bar_wait(&land_bar[st], static_cast<uint32_t>((j / STAGES) & 1));\n",
               "    unsigned long long an_wl = 0;\n"
               "    for (int j = 0; j < nst; ++j) {\n      const int st = j % STAGES;\n"
               "      const unsigned long long an_l0 = an_clk_m();\n"
               "      bar_wait(&land_bar[st], static_cast<uint32_t>((j / STAGES) & 1));\n"
               "      an_wl += an_clk_m() - an_l0;\n", "transposer land wait")
    src = sub1(src, "      if (lane == 0) bar_arrive(&full_bar[st]);\n    }\n    return;\n  }\n",
               "      if (lane == 0) bar_arrive(&full_bar[st]);\n    }\n"
               "    if (lane == 0) {\n      an_buf[an_base + 52 + 2 * (warp - 9)] = an_wl;\n"
               "      an_buf[an_base + 53 + 2 * (warp - 9)] = an_clk() - an_t0;\n    }\n    return;\n  }\n", "transposer store")
    # consumers
    src = sub1(src, "  for (int i = 0; i < nst; ++i) {\n    const int st = i % STAGES;\n    bar_wait(&full_bar[st], static_cast<uint32_t>((i / STAGES) & 1));\n",
               "  unsigned long long an_aw = 0, an_aq = 0, an_as = 0, an_ap = 0, an_ao = 0;\n"
               "  for (int i = 0; i < nst; ++i) {\n    const int st = i % STAGES;\n"
               "    const unsigned long long an_c0 = an_clk_m();\n"
               "    bar_wait(&full_bar[st], static_cast<uint32_t>((i / STAGES) & 1));\n"
               "    const unsigned long long an_c1 = an_clk_m();\n    an_aw += an_c1 - an_c0;\n", "full wait")
    src = sub1(src, "    wgmma2_e4m3_sync(sc, qa[0], qa[1], gdesc(kc_addr, LBO_K, SBO_K), gdesc(kc_addr + 2 * LBO_K, LBO_K, SBO_K));\n"
                    "    fence_regs(sc);\n    fence_a(qa[0]);\n    fence_a(qa[1]);\n",
               "    wgmma2_e4m3_sync(sc, qa[0], qa[1], gdesc(kc_addr, LBO_K, SBO_K), gdesc(kc_addr + 2 * LBO_K, LBO_K, SBO_K));\n"
               "    fence_regs(sc);\n    fence_a(qa[0]);\n    fence_a(qa[1]);\n"
               "    const unsigned long long an_c2 = an_dep(sc[31] + sc[0] + sc[17]);\n    an_aq += an_c2 - an_c1;\n", "qk end")
    src = sub1(src, "    fence_regs(opv);\n    fence_a(pa[0]);\n    fence_a(pa[1]);\n    wgmma2_e4m3_sync(opv",
               "    const unsigned long long an_c3 = an_dep_u(pa[1][3] ^ pa[0][0] ^ pa[1][1]);\n    an_as += an_c3 - an_c2;\n"
               "    fence_regs(opv);\n    fence_a(pa[0]);\n    fence_a(pa[1]);\n    wgmma2_e4m3_sync(opv", "softmax end")
    src = sub1(src, "    wgmma2_e4m3_sync(opv, pa[0], pa[1], gdesc(vt_addr, LBO_V, SBOV), gdesc(vt_addr + 2 * LBO_V, LBO_V, SBOV));\n"
                    "    fence_regs(opv);\n    fence_a(pa[0]);\n    fence_a(pa[1]);\n",
               "    wgmma2_e4m3_sync(opv, pa[0], pa[1], gdesc(vt_addr, LBO_V, SBOV), gdesc(vt_addr + 2 * LBO_V, LBO_V, SBOV));\n"
               "    fence_regs(opv);\n    fence_a(pa[0]);\n    fence_a(pa[1]);\n"
               "    const unsigned long long an_c4 = an_dep(opv[31] + opv[0] + opv[13]);\n    an_ap += an_c4 - an_c3;\n", "pv end")
    src = sub1(src, "      o[4 * j + 3] = o[4 * j + 3] * alB + opv[4 * j + 3];\n    }\n  }\n",
               "      o[4 * j + 3] = o[4 * j + 3] * alB + opv[4 * j + 3];\n    }\n"
               "    an_ao += an_dep(o[31] + o[0] + o[14]) - an_c4;\n  }\n"
               "  if (lane == 0) {\n    an_buf[an_base + 6 * warp + 0] = an_aw;\n    an_buf[an_base + 6 * warp + 1] = an_aq;\n"
               "    an_buf[an_base + 6 * warp + 2] = an_as;\n    an_buf[an_base + 6 * warp + 3] = an_ap;\n"
               "    an_buf[an_base + 6 * warp + 4] = an_ao;\n    an_buf[an_base + 6 * warp + 5] = an_clk() - an_t0;\n  }\n",
               "o update end")
    src = sub1(src, "      l_part[part] = l;\n    }\n  }\n}\n",
               "      l_part[part] = l;\n    }\n  }\n"
               "  if (threadIdx.x == 0) {\n    an_buf[an_base + 56] = an_t0;\n    an_buf[an_base + 57] = an_clk();\n"
               "    an_buf[an_base + 58] = an_smid();\n    an_buf[an_base + 59] = static_cast<unsigned long long>(nst);\n  }\n}\n", "exit")
    # host: extra tensor argument
    src = sub1(src, "double sm_scale, int64_t stages, int64_t sbov, int64_t wf) {",
               "double sm_scale, int64_t stages, int64_t sbov, int64_t wf, torch::Tensor an_buf) {", "host sig")
    src = sub1(src, "static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale));",
               "static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale),                  \\\n"
               "        reinterpret_cast<unsigned long long*>(an_buf.data_ptr()));", "host launch")
    decl = sub1(decl, "double sm_scale, int64_t stages, int64_t sbov, int64_t wf);",
                "double sm_scale, int64_t stages, int64_t sbov, int64_t wf, torch::Tensor an_buf);", "decl")
    assert "attend_w1" not in src and "okc_w1c" not in src
    return ('"""Generated by make_anatw1.py (do not edit): the fixed W1 with anatomy stamps (see make_anatw1.py)."""\n\n'
            f"CUDA_AW1_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_AW1_DECL = {decl!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_anatw1.py up to date" if same else "cuda_anatw1.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
