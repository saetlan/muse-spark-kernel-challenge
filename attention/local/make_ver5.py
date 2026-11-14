"""Generates attention/local/cuda_ver5.py: V5, V4's verify kernel with a leaner consumer loop, built for
the bottleneck L17 measured (AF37/AF39: consumer-bound at ~1 warp-instruction per clock per SM; an
L2-resident cache changed nothing). V4's verified pieces are copied byte for byte by anchors that
must occur exactly once (asserted): helpers, kernel setup + producer warp + query fragments (MT = 1),
and the partial stores. The consumer loop is new:
1. one softmax round per 64-key stage (V4: two 32-key rounds): 16 independent Q.K MMAs, one max /
   shuffle / rescale per stage, then P.V over the stage's two 32-key chunks into one fresh
   accumulator per output tile (64 FP8 products per MMA pair, as V4's 32 per MMA, then FMA into O);
2. masks only on stages a row's window start, the block's causal tail or the slice end reaches
   (V4 masked every sliding sub-chunk);
3. the score scale fused into the exponent's FMA (the max is taken on raw scores: qk_scale > 0),
   ex2.approx.ftz;
4. __launch_bounds__(288, MINB): MINB 2 caps registers at 112 so two CTAs (16 consumer warps) share
   an SM (V4 MT1: 141 registers, one CTA).
Chunk c of a stage is V4's sub-chunk sc = c: P fragments packed from n-tiles 4c..4c+3, V rows 32c + ...
Variants (stages, min blocks): (3, 1), (4, 1), (3, 2), (4, 2).
Run: .venv/bin/python attention/local/make_ver5.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_ver5.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def between(src, a, b, what, include_b=False):
    assert src.count(a) == 1, f"{what}: start anchor x{src.count(a)}"
    i = src.index(a)
    j = src.index(b, i)
    return src[i:j + (len(b) if include_b else 0)]


def patch(src, old, new, what):
    assert src.count(old) == 1, f"{what}: x{src.count(old)}"
    return src.replace(old, new)


V4 = load("cuda_ver4.py", "CUDA_VER4_SRC")
HELPERS = between(V4, "#include <torch/extension.h>", "template <int STAGES, int MT, int MINB>", "helpers")
PRE = between(V4, "template <int STAGES, int MT, int MINB>", "  float m_run[MT][2], l_run[MT][2], acc[MT][8][4];\n", "pre")
POST = between(V4, "  // Partials: rows (token 2*mt + r, head (kv0+h)*8 + g), d = 8*(2*t4 + jj) + dt.\n",
               "}  // namespace okc_ver4\n", "post", include_b=True)
HELPERS = patch(HELPERS, "namespace okc_ver4 {", "namespace okc_ver5 {", "namespace")
PRE = patch(PRE, "template <int STAGES, int MT, int MINB>\n__global__ void __launch_bounds__((8 / MT + 1) * 32, MINB) attend_ver4_kernel(",
            "template <int STAGES, int MINB>\n__global__ void __launch_bounds__(9 * 32, MINB) attend_ver5_kernel(", "signature")
PRE = patch(PRE, "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n",
            "    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n"
            "  constexpr int MT = 1;                               // V5: one m-tile per consumer warp, 8 consumer warps\n",
            "MT")
POST = patch(POST, "}  // namespace okc_ver4\n", "}  // namespace okc_ver5\n", "namespace end")
for piece in (HELPERS, PRE, POST):
    assert "okc_ver4" not in piece and "attend_ver4" not in piece

LOOP = r"""  float m_run[MT][2], l_run[MT][2], acc[MT][8][4];
  m_run[0][0] = m_run[0][1] = -1.0e30f;
  l_run[0][0] = l_run[0][1] = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[0][j][0] = acc[0][j][1] = acc[0][j][2] = acc[0][j][3] = 0.f;
  const int posA = n - L + 2 * mt0, posB = posA + 1;     // rows g / g+8

  // V5: one softmax round per 64-key stage
  for (int i = 0; i < nst; ++i) {
    const int st = i % STAGES;
    const uint32_t parity = static_cast<uint32_t>((i / STAGES) & 1);
    uint32_t done = 0;
    while (!done)
      asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                   : "=r"(done) : "r"(smem_u32(&full_bar[st])), "r"(parity) : "memory");
    const uint8_t* sk = ring + st * STAGE_BYTES;
    const uint8_t* sv = sk + SKEYS * KP;
    const int s0 = start + i * SKEYS;
    // Q.K: n-tile nt = keys s0 + 8*nt .. +7 (column 2*t4 + jj of the C fragment = key 8*nt + 2*t4 + jj)
    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const uint4 kb = *reinterpret_cast<const uint4*>(sk + (8 * nt + g) * KP + 64 * h + 16 * t4);
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
      mma_e4m3(s[nt], qa[0][0][0], qa[0][0][1], qa[0][0][2], qa[0][0][3], kb.x, kb.y);
      mma_e4m3(s[nt], qa[0][1][0], qa[0][1][1], qa[0][1][2], qa[0][1][3], kb.z, kb.w);
    }
    // masks only where a row's window start (keys < lo + L), the causal tail (keys > n - L) or the
    // slice end falls inside this stage
    if (s0 + SKEYS > end || s0 + SKEYS > n - L || (window > 0 && s0 < lo + L)) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int key = s0 + 8 * nt + 2 * t4 + jj;
          const bool ok = key < end;
          const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);
          const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);
          s[nt][jj] = okA ? s[nt][jj] : -INFINITY;
          s[nt][2 + jj] = okB ? s[nt][2 + jj] : -INFINITY;
        }
    }
    float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mxA = fmaxf(mxA, fmaxf(s[nt][0], s[nt][1]));
      mxB = fmaxf(mxB, fmaxf(s[nt][2], s[nt][3]));
    }
    mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));
    mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));
    mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));
    mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));
    // raw-score max times qk_scale (> 0) = max of the scaled scores
    const float mnA = fmaxf(m_run[0][0], mxA * qk_scale), mnB = fmaxf(m_run[0][1], mxB * qk_scale);
    const float alA = ex2f(m_run[0][0] - mnA), alB = ex2f(m_run[0][1] - mnB);
    const float oA = P_OFF - mnA, oB = P_OFF - mnB;
    float sumA = 0.f, sumB = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s[nt][0] = ex2f(fmaf(s[nt][0], qk_scale, oA));
      s[nt][1] = ex2f(fmaf(s[nt][1], qk_scale, oA));
      s[nt][2] = ex2f(fmaf(s[nt][2], qk_scale, oB));
      s[nt][3] = ex2f(fmaf(s[nt][3], qk_scale, oB));
      sumA += s[nt][0] + s[nt][1];
      sumB += s[nt][2] + s[nt][3];
    }
    l_run[0][0] = l_run[0][0] * alA + sumA;
    l_run[0][1] = l_run[0][1] * alB + sumB;
    m_run[0][0] = mnA;
    m_run[0][1] = mnB;
    // chunk c (keys 32c .. 32c+31) = V4's sub-chunk sc = c: P from n-tiles 4c .. 4c+3, V rows 32c + ...
    uint32_t pa[2][4], vb0[2][8], vb1[2][8];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      pa[c][0] = e4m3x4(s[4 * c][0], s[4 * c][1], s[4 * c + 1][0], s[4 * c + 1][1]);         // row g
      pa[c][1] = e4m3x4(s[4 * c][2], s[4 * c][3], s[4 * c + 1][2], s[4 * c + 1][3]);         // row g+8
      pa[c][2] = e4m3x4(s[4 * c + 2][0], s[4 * c + 2][1], s[4 * c + 3][0], s[4 * c + 3][1]); // row g
      pa[c][3] = e4m3x4(s[4 * c + 2][2], s[4 * c + 2][3], s[4 * c + 3][2], s[4 * c + 3][3]); // row g+8
      const uint8_t* vr = sv + (32 * c) * VP + 64 * h + 8 * g;
      const uint2 r0 = *reinterpret_cast<const uint2*>(vr + (2 * t4) * VP);
      const uint2 r1 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 1) * VP);
      const uint2 r2 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 8) * VP);
      const uint2 r3 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 9) * VP);
      const uint2 r4 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 16) * VP);
      const uint2 r5 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 17) * VP);
      const uint2 r6 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 24) * VP);
      const uint2 r7 = *reinterpret_cast<const uint2*>(vr + (2 * t4 + 25) * VP);
      uint32_t t[4];
      tr4(r0.x, r1.x, r2.x, r3.x, t);
      vb0[c][0] = t[0]; vb0[c][1] = t[1]; vb0[c][2] = t[2]; vb0[c][3] = t[3];
      tr4(r0.y, r1.y, r2.y, r3.y, t);
      vb0[c][4] = t[0]; vb0[c][5] = t[1]; vb0[c][6] = t[2]; vb0[c][7] = t[3];
      tr4(r4.x, r5.x, r6.x, r7.x, t);
      vb1[c][0] = t[0]; vb1[c][1] = t[1]; vb1[c][2] = t[2]; vb1[c][3] = t[3];
      tr4(r4.y, r5.y, r6.y, r7.y, t);
      vb1[c][4] = t[0]; vb1[c][5] = t[1]; vb1[c][6] = t[2]; vb1[c][7] = t[3];
    }
    __syncwarp();          // every lane has read this stage's K and V rows
    if (lane == 0) {
      uint64_t state;
      asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(smem_u32(&empty_bar[st])) : "memory");
      (void)state;
    }
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      float pv[4] = {0.f, 0.f, 0.f, 0.f};
      mma_e4m3(pv, pa[0][0], pa[0][1], pa[0][2], pa[0][3], vb0[0][dt], vb1[0][dt]);
      mma_e4m3(pv, pa[1][0], pa[1][1], pa[1][2], pa[1][3], vb0[1][dt], vb1[1][dt]);
      acc[0][dt][0] = acc[0][dt][0] * alA + pv[0];   // contracted to FMA
      acc[0][dt][1] = acc[0][dt][1] * alA + pv[1];
      acc[0][dt][2] = acc[0][dt][2] * alB + pv[2];
      acc[0][dt][3] = acc[0][dt][3] * alB + pv[3];
    }
  }

"""

EX2F = r"""__device__ __forceinline__ float ex2f(float x) {   // V5: flush-to-zero exp2 (P below 2^-126 is 0 in E4M3 anyway)
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

"""

HOST = r"""
#define OKC_VER5_VARIANTS(X) X(3, 1) X(4, 1) X(3, 2) X(4, 2)

void attend_ver5(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t minb) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(stages) * okc_ver5::STAGE_BYTES;
  bool launched = false;
#define OKC_VER5_LAUNCH(S, B)                                                                                     \
  if (!launched && stages == S && minb == B) {                                                                    \
    okc_ver5::attend_ver5_kernel<S, B><<<grid, 9 * 32, smem, stream>>>(                                           \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),         \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),  \
        seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),  \
        static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),      \
        static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale));                          \
    launched = true;                                                                                              \
  }
  OKC_VER5_VARIANTS(OKC_VER5_LAUNCH)
#undef OKC_VER5_LAUNCH
  TORCH_CHECK(launched, "attend_ver5: no variant stages=", stages, " minb=", minb);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_ver5 launch failed: ", cudaGetErrorString(e));
}

int64_t attend_ver5_setup() {   // once, outside any capture: each variant's ring
  int err = 0;
#define OKC_VER5_SMEM(S, B) \
  err |= cudaFuncSetAttribute(okc_ver5::attend_ver5_kernel<S, B>, cudaFuncAttributeMaxDynamicSharedMemorySize, S * okc_ver5::STAGE_BYTES);
  OKC_VER5_VARIANTS(OKC_VER5_SMEM)
#undef OKC_VER5_SMEM
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> attend_ver5_attrs(int64_t stages, int64_t minb) {
  cudaFuncAttributes a;
  a.numRegs = -1;
  a.localSizeBytes = 0;
  a.sharedSizeBytes = 0;
#define OKC_VER5_ATTR(S, B) \
  if (stages == S && minb == B) cudaFuncGetAttributes(&a, okc_ver5::attend_ver5_kernel<S, B>);
  OKC_VER5_VARIANTS(OKC_VER5_ATTR)
#undef OKC_VER5_ATTR
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

DECL = ("void attend_ver5(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
        "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
        "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
        "int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t minb);\n"
        "int64_t attend_ver5_setup();\n"
        "std::vector<int64_t> attend_ver5_attrs(int64_t stages, int64_t minb);")


def render():
    src = HELPERS + EX2F + PRE + LOOP + POST + HOST
    for name in ("issue_pt", "pt_slot", "full_bar", "empty_bar", "qa[mi][s][0]", "attend_ver5_kernel<S, B>"):
        assert name in src, name
    return ('"""Generated by make_ver5.py (do not edit): V5, V4 with a 64-key softmax round per stage, edge-only '
            'masks and a register cap (see make_ver5.py)."""\n\n'
            f"CUDA_VER5_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_VER5_DECL = {DECL!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ver5.py up to date" if same else "cuda_ver5.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
