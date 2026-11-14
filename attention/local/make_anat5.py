"""Generates attention/local/cuda_anat5.py: an instrumented copy of V5 (cuda_ver5.py) for an anatomy lab
(L15's method, make_anatomy.py), to split a 64-key stage's time between waiting for data and the
consumer's phases, and to tell a consumer-bound kernel from a producer-bound one (L17's L2-hot probe
cannot: both are cache-independent).
Changes, each at an anchor that must occur exactly once (asserted): namespace okc_ver5 -> okc_an5,
attend_ver5 -> an_ver5; one extra kernel / host argument an_buf (64 x u64 per CTA); stamps
(%clock64 in asm volatile; a phase end is read by an asm that takes the phase's last result as an
input, so in-order issue makes it wait for that value):
consumer warp w, lane 0 (slots 6w + k): 0 cycles waiting on the full barrier, 1 Q.K + mask + max +
shuffles, 2 exponent + l + P packing + V fragments, 3 P.V + rescale, 4 stages, 5 entry -> loop end;
producer lane 0: 48 cycles waiting on the empty barrier, 49 waiting on page entries
(cp.async.wait_group), 50 entry -> loop end; thread 0: 56 clock at entry, 57 clock at exit, 58 SM id.
Run: .venv/bin/python attention/local/make_anat5.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_anat5.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var], ns[var.replace("_SRC", "_DECL")]


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
    src, decl = load("cuda_ver5.py", "CUDA_VER5_SRC")
    src = src.replace("okc_ver5", "okc_an5").replace("attend_ver5", "an_ver5").replace("OKC_VER5", "OKC_AN5")
    decl = decl.replace("attend_ver5", "an_ver5")
    src = sub1(src, "template <int STAGES, int MINB>\n", STAMPS + "template <int STAGES, int MINB>\n", "stamps")
    src = sub1(src, "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n"
                    "  constexpr int MT = 1;",
               "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale,\n"
               "    unsigned long long* __restrict__ an_buf) {\n"
               "  const long an_base = (static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x) * 64;\n"
               "  const unsigned long long an_t0 = an_clk();\n"
               "  constexpr int MT = 1;", "kernel args")
    # producer
    src = sub1(src, "    for (int i = 0; i < nst; ++i) {\n      const int st = i % STAGES;\n      if (i >= STAGES) {",
               "    unsigned long long an_we = 0, an_wp = 0;\n"
               "    for (int i = 0; i < nst; ++i) {\n      const int st = i % STAGES;\n      if (i >= STAGES) {", "producer acc")
    src = sub1(src, "        uint32_t done = 0;\n        while (!done)\n",
               "        uint32_t done = 0;\n        const unsigned long long an_e0 = an_clk_m();\n        while (!done)\n", "empty wait 0")
    src = sub1(src, ": \"=r\"(done) : \"r\"(smem_u32(&empty_bar[st])), \"r\"(parity) : \"memory\");\n      }\n",
               ": \"=r\"(done) : \"r\"(smem_u32(&empty_bar[st])), \"r\"(parity) : \"memory\");\n"
               "        an_we += an_clk_m() - an_e0;\n      }\n", "empty wait 1")
    src = sub1(src, "      asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));   // stage i+1's entries landed\n",
               "      const unsigned long long an_p0 = an_clk_m();\n"
               "      asm volatile(\"cp.async.wait_group %0;\" ::\"n\"(STAGES - 1));   // stage i+1's entries landed\n"
               "      an_wp += an_clk_m() - an_p0;\n", "page wait")
    src = sub1(src, "    asm volatile(\"cp.async.wait_all;\");\n    return;",
               "    asm volatile(\"cp.async.wait_all;\");\n"
               "    if (lane == 0) {\n      an_buf[an_base + 48] = an_we;\n      an_buf[an_base + 49] = an_wp;\n"
               "      an_buf[an_base + 50] = an_clk() - an_t0;\n    }\n    return;", "producer store")
    # consumers
    src = sub1(src, "  // V5: one softmax round per 64-key stage\n",
               "  unsigned long long an_aw = 0, an_aq = 0, an_as = 0, an_ap = 0;\n  // V5: one softmax round per 64-key stage\n",
               "consumer acc")
    src = sub1(src, "    uint32_t done = 0;\n    while (!done)\n",
               "    uint32_t done = 0;\n    const unsigned long long an_c0 = an_clk_m();\n    while (!done)\n", "full wait 0")
    src = sub1(src, ": \"=r\"(done) : \"r\"(smem_u32(&full_bar[st])), \"r\"(parity) : \"memory\");\n    const uint8_t* sk",
               ": \"=r\"(done) : \"r\"(smem_u32(&full_bar[st])), \"r\"(parity) : \"memory\");\n"
               "    const unsigned long long an_c1 = an_clk_m();\n    an_aw += an_c1 - an_c0;\n    const uint8_t* sk", "full wait 1")
    src = sub1(src, "    mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));\n",
               "    mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));\n"
               "    const unsigned long long an_c2 = an_dep(mxA + mxB);\n    an_aq += an_c2 - an_c1;\n", "qk end")
    src = sub1(src, "    __syncwarp();          // every lane has read this stage's K and V rows\n",
               "    const unsigned long long an_c3 = an_dep_u(pa[1][3] ^ vb1[1][7] ^ vb0[0][0]);\n    an_as += an_c3 - an_c2;\n"
               "    __syncwarp();          // every lane has read this stage's K and V rows\n", "softmax end")
    src = sub1(src, "      acc[0][dt][3] = acc[0][dt][3] * alB + pv[3];\n    }\n  }\n",
               "      acc[0][dt][3] = acc[0][dt][3] * alB + pv[3];\n    }\n"
               "    an_ap += an_dep(acc[0][7][3] + acc[0][0][0] + acc[0][3][1]) - an_c3;\n  }\n"
               "  if (lane == 0) {\n    an_buf[an_base + 6 * warp + 0] = an_aw;\n    an_buf[an_base + 6 * warp + 1] = an_aq;\n"
               "    an_buf[an_base + 6 * warp + 2] = an_as;\n    an_buf[an_base + 6 * warp + 3] = an_ap;\n"
               "    an_buf[an_base + 6 * warp + 4] = static_cast<unsigned long long>(nst);\n"
               "    an_buf[an_base + 6 * warp + 5] = an_clk() - an_t0;\n  }\n", "pv end")
    src = sub1(src, "        l_part[part] = l;\n      }\n    }\n}\n",
               "        l_part[part] = l;\n      }\n    }\n"
               "  if (threadIdx.x == 0) {\n    an_buf[an_base + 56] = an_t0;\n    an_buf[an_base + 57] = an_clk();\n"
               "    an_buf[an_base + 58] = an_smid();\n  }\n}\n", "exit")
    # host: extra tensor argument
    src = sub1(src, "int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t minb) {",
               "int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t minb, torch::Tensor an_buf) {", "host sig")
    src = sub1(src, "static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale));",
               "static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale),                  \\\n"
               "        reinterpret_cast<unsigned long long*>(an_buf.data_ptr()));", "host launch")
    decl = sub1(decl, "double sm_scale, int64_t stages, int64_t minb);", "double sm_scale, int64_t stages, int64_t minb, torch::Tensor an_buf);",
                "decl")
    assert "attend_ver5" not in src and "okc_ver5" not in src
    return ('"""Generated by make_anat5.py (do not edit): V5 with anatomy stamps (see make_anat5.py)."""\n\n'
            f"CUDA_AN5_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_AN5_DECL = {decl!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_anat5.py up to date" if same else "cuda_anat5.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
