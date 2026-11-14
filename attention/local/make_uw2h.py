"""Generates attention/local/cuda_uw2h.py: UW2h = UW2 (cuda_uw2.py) with a half-size push. L36's anatomy: UW2's push takes
1.95 us at cs16 (Sv1) but 1.03 us at cs4 (Sv16) for the same bytes per CTA, so it scales with the cluster's total DSMEM
traffic, not with store instructions (L48: 16-byte stores changed nothing). Here each pushed row carries o / l as FP16
pairs (one 4-byte st.shared::cluster per column pair instead of 8 bytes) plus m and l in FP32: rows of 36 words (32 words
of f16x2 + m + l + 2 pad) in the receive buffer. Normalizing by l first keeps |o / l| <= max |v| (FP8 values), far
inside FP16's range; an empty slice (l = 0) pushes zeros with m = -1e30 (weight 0). The owner merges
l_tot = sum w_s l_s, o = sum w_s l_s (o / l)_s. Only the push and the owner's merge change (asserted anchors).
Host: uw2h(... as uw2 ...), uw2h_setup(), uw2h_attrs().
Run: .venv/bin/python attention/local/make_uw2h.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_uw2h.py")

PUSH_OLD = """#pragma unroll
      for (int j = 0; j < 8; ++j) st_cluster2(row + (8 * j + 2 * t4) * 4, o[4 * j + 2 * half], o[4 * j + 2 * half + 1]);
      if (t4 == 0) {
        st_cluster(row + 64 * 4, m_run[half]);
        st_cluster(row + 65 * 4, l);
      }
"""
PUSH_NEW = """      const float inv = l > 0.f ? 1.f / l : 0.f;        // UW2h: o / l as FP16 pairs, m and l in FP32
#pragma unroll
      for (int j = 0; j < 8; ++j)
        st_cluster_u32(row + (4 * j + t4) * 4, pack_h2(o[4 * j + 2 * half] * inv, o[4 * j + 2 * half + 1] * inv));
      if (t4 == 0) {
        st_cluster(row + 32 * 4, m_run[half]);
        st_cluster(row + 33 * 4, l);
      }
"""
ROW_OLD = """      const uint32_t row = map_rank(smem_u32(rbuf), static_cast<uint32_t>(R / RPC)) +
                           static_cast<uint32_t>((static_cast<int>(rank) * RPC + R % RPC) * LBP * 4);
#pragma unroll
      for (int j = 0; j < 8; ++j) st_cluster2("""
ROW_NEW = """      const uint32_t row = map_rank(smem_u32(rbuf), static_cast<uint32_t>(R / RPC)) +
                           static_cast<uint32_t>((static_cast<int>(rank) * RPC + R % RPC) * LBPH * 4);
#pragma unroll
      for (int j = 0; j < 8; ++j) st_cluster2("""
MERGE_OLD = """    for (int src = 0; src < CS; ++src) mm = fmaxf(mm, rbuf[(src * RPC + slot) * LBP + 64]);
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int src = 0; src < CS; ++src) {
      const float* pr = rbuf + (src * RPC + slot) * LBP;
      const float wt = ex2(pr[64] - mm);
      l += wt * pr[65];
      o0 += wt * pr[lane];
      o1 += wt * pr[32 + lane];
    }
"""
MERGE_NEW = """    for (int src = 0; src < CS; ++src) mm = fmaxf(mm, rbuf[(src * RPC + slot) * LBPH + 32]);
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int src = 0; src < CS; ++src) {   // UW2h: rows of LBPH words: f16x2 columns, m, l
      const float* pr = rbuf + (src * RPC + slot) * LBPH;
      const float wl = ex2(pr[32] - mm) * pr[33];
      l += wl;
      const uint32_t* pw = reinterpret_cast<const uint32_t*>(pr);
      o0 += wl * h2f(pw[lane >> 1], lane & 1);
      o1 += wl * h2f(pw[16 + (lane >> 1)], lane & 1);
    }
"""
HELPERS = """__device__ __forceinline__ void st_cluster_u32(uint32_t addr, uint32_t v) {   // UW2h
  asm volatile("st.shared::cluster.b32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {   // UW2h: {lo, hi} as f16x2 (lo in bits 0..15)
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float h2f(uint32_t w, int hi) {   // UW2h: half `hi` of an f16x2 word as FP32
  const uint16_t h = static_cast<uint16_t>(hi ? (w >> 16) : (w & 0xffffu));
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
constexpr int LBPH = 36;    // UW2h: receive row: 32 words of f16x2 (64 o / l), m, l, 2 pad
"""


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor x{n}: {old[:90]!r}"
    return src.replace(old, new)


def render():
    text = open(os.path.join(HERE, "cuda_uw2.py")).read()
    for pat, rep in ((r"\bokc_uw2", "okc_uw2h"), (r"\bOKC_UW2_", "OKC_UW2H_"), (r"\buw2\(", "uw2h("), (r"\buw2_setup\b", "uw2h_setup"),
                     (r"\buw2_attrs\b", "uw2h_attrs"), (r"\buw2_kernel\b", "uw2h_kernel"), (r"\bCUDA_UW2_", "CUDA_UW2H_"),
                     (r'"uw2: ', '"uw2h: '), (r'"uw2 ', '"uw2h ')):
        text = re.sub(pat, rep, text)
    text = sub1(text, ROW_OLD, ROW_NEW, "push row base")
    text = sub1(text, PUSH_OLD, PUSH_NEW, "push")
    text = sub1(text, MERGE_OLD, MERGE_NEW, "owner merge")
    text = sub1(text, "__device__ __forceinline__ void st_cluster2(uint32_t addr, float a, float b) {\n",
                HELPERS + "__device__ __forceinline__ void st_cluster2(uint32_t addr, float a, float b) {\n", "helpers")
    assert "LBP * 4);\n#pragma unroll\n      for (int j = 0; j < 8; ++j) st_cluster2(" not in text
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_uw2h.py from cuda_uw2.py (do not edit): UW2h, UW2 with a half-size FP16 push (see\n'
            'make_uw2h.py). Original docstring follows.\n' + text[3:i] + text[i:])


def sim():
    """Push / merge model: random per-lane o, l, m for CS sources; FP16 rounding of o / l; the owner's merge from the
    source's word / half formulas equals the FP32 direct merge to FP16 precision for every column; empty slices ignored;
    the cvt operand order puts lo in bits 0..15 (PTX: d = {a -> upper, b -> lower} for cvt.rn.f16x2.f32 d, a, b)."""
    import numpy as np
    t = render()
    for frag in ("st_cluster_u32(row + (4 * j + t4) * 4, pack_h2(o[4 * j + 2 * half] * inv, o[4 * j + 2 * half + 1] * inv));",
                 'asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));', "o0 += wl * h2f(pw[lane >> 1], lane & 1);",
                 "o1 += wl * h2f(pw[16 + (lane >> 1)], lane & 1);", "st_cluster(row + 32 * 4, m_run[half]);", "st_cluster(row + 33 * 4, l);",
                 "const float wl = ex2(pr[32] - mm) * pr[33];"):
        assert t.count(frag) == 1, frag
    rng = np.random.default_rng(7)
    for CS in (4, 16):
        o = rng.normal(0, 1, (CS, 64)) * rng.uniform(1, 3e4, (CS, 1))
        l = rng.uniform(1, 3e4, CS)
        m = rng.uniform(-20, 5, CS)
        l[1] = 0.0
        o[1] = 0.0
        m[1] = -1e30
        words = np.zeros((CS, 32), dtype=np.uint32)
        for s in range(CS):
            inv = 1.0 / l[s] if l[s] > 0 else 0.0
            for t4 in range(4):
                for j in range(8):
                    lo, hi = np.float16(o[s, 8 * j + 2 * t4] * inv), np.float16(o[s, 8 * j + 2 * t4 + 1] * inv)
                    words[s, 4 * j + t4] = int(lo.view(np.uint16)) | (int(hi.view(np.uint16)) << 16)
        mm = m.max()
        for lane in range(32):
            ltot = o0 = o1 = 0.0
            for s in range(CS):
                wl = 2.0 ** (m[s] - mm) * l[s]
                ltot += wl
                w0, w1 = words[s, lane >> 1], words[s, 16 + (lane >> 1)]
                h0 = np.uint16((w0 >> 16) if lane & 1 else (w0 & 0xffff)).view(np.float16)
                h1 = np.uint16((w1 >> 16) if lane & 1 else (w1 & 0xffff)).view(np.float16)
                o0 += wl * float(h0)
                o1 += wl * float(h1)
            w = 2.0 ** (m - mm)
            ref0 = (w * o[:, lane]).sum() / (w * l).sum()
            ref1 = (w * o[:, 32 + lane]).sum() / (w * l).sum()
            assert abs(o0 / ltot - ref0) <= 2e-3 * (abs(ref0) + 1e-3) + 1e-4, (CS, lane, o0 / ltot, ref0)
            assert abs(o1 / ltot - ref1) <= 2e-3 * (abs(ref1) + 1e-3) + 1e-4, (CS, lane, o1 / ltot, ref1)
    return True


if __name__ == "__main__":
    sim()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_uw2h.py up to date" if same else "cuda_uw2h.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} (push / merge model ok)")
