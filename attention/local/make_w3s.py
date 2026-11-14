"""Generates attention/local/cuda_w3s.py: W3s = W3b (cuda_w3b.py) with a length-aware split that spans requests (AF65).
W3b gives every request the same number of CTAs in one wave (8 / 4 at bs16 / bs32: 128 of 132 SMs), so full verify waits
on the longest request: 1.100x / 1.104x the ideal on the judge's lengths (simulation), 4-5% measured (L38 / L39 tail).
W3s: every request's keys become 64-key stages (stage k of request b = keys 64k .. min(64k + 64, n_b)); the flattened
list of all requests' stages (request-major, S_total stages) is cut into gridDim.x contiguous ranges of
Q = ceil(S_total / G) stages, one per CTA (grid (G, KV-head pairs)). A CTA's range covers one or more requests: segments.
Thread 0 builds the segment table (request, first key, stage offset, stage count, partial slot) before the CTA barrier;
the copier and transposers run over the CTA's stages exactly as W3b (keys and page-table row per stage from the table),
so the ring and its barriers are untouched. The consumers loop over segments: per segment the request's Q tile is
re-written (after a warpgroup barrier: every wgmma on the old tile has been waited), O / m / l reset, the pair's stages of
the segment (global stage parity, as W3) run W3's FA3-order code (pp 0), and the segment's partial goes to slot
x - first_cta(b) of its rows: partial index = row * (G * NP) + slot * NP + par, first_cta(b) = P_b / Q (P_b = stages
before request b). _combine_s (lab / v-block Triton) recomputes Q, P_b and the request's slot count from seqlens and
merges exactly those slots. Full layers only (window 0); batch <= 32 (segment table size). Barrier watchdog as W3d.
Host: attend_w3s(q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages, G, page, window,
sm_scale, stages, pp, nwg, rc) with o_part / m_part / l_part of tokens x hq x G x nwg/2 rows; attend_w3s_setup(),
attend_w3s_attrs(...), attend_w3s_hang().
Run: .venv/bin/python attention/local/make_w3s.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w3s.py")
VARIANTS = [(6, 0, 4, 104, 56), (7, 0, 4, 104, 56)]
MAXSEG = 32


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:100]!r}"
    return src.replace(old, new)


def cut(src, a, b, new, what):
    assert src.count(a) == 1 and src.count(b) >= 1, (what, src.count(a), src.count(b))
    i = src.index(a)
    j = src.index(b, i)
    return src[:i] + new + src[j:]


HEADER = r'''  const int pairs = hkv >> 1;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int x = blockIdx.x;                           // W3s: this CTA's range of the flattened 64-key stage list
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;
  // segment table (thread 0, published by the CTA barrier below): request, its keys n, first key, stage offset in this
  // CTA, stage count, partial slot (this CTA's index among the CTAs covering the request)
  __shared__ int seg_b[MAXSEG], seg_n[MAXSEG], seg_k0[MAXSEG], seg_off[MAXSEG], seg_cnt[MAXSEG], seg_slot[MAXSEG];
  __shared__ int nseg_sh, nst_sh;
  if (tid == 0) {
    int stot = 0;
    for (int bb = 0; bb < batch; ++bb) stot += (seqlens[bb] + SKEYS - 1) / SKEYS;
    const int Qs = max((stot + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x), 1);
    const int c0 = x * Qs, c1 = min(c0 + Qs, stot);
    int pfx = 0, ns = 0, off = 0;
    for (int bb = 0; bb < batch && pfx < c1; ++bb) {
      const int nb = seqlens[bb], sb = (nb + SKEYS - 1) / SKEYS;
      const int a = max(pfx, c0), e = min(pfx + sb, c1);
      if (a < e && ns < MAXSEG) {
        seg_b[ns] = bb;
        seg_n[ns] = nb;
        seg_k0[ns] = (a - pfx) * SKEYS;
        seg_off[ns] = off;
        seg_cnt[ns] = e - a;
        seg_slot[ns] = x - pfx / Qs;
        off += e - a;
        ++ns;
      }
      pfx += sb;
    }
    nseg_sh = ns;
    nst_sh = off;
  }
'''

AFTER_SYNC = r'''  __syncthreads();
  const int nst = nst_sh, nseg = nseg_sh;              // W3s: this CTA's stages (all segments), segments
  auto seg_of = [&](int j) -> int {                   // the segment holding CTA stage j
    int s = 0;
    while (s + 1 < nseg && j >= seg_off[s + 1]) ++s;
    return s;
  };
'''

CONSUMER = r'''  // ---- W3s consumers: W3's warpgroups (cw = warp >> 2: head h = cw & 1, pair par = cw >> 1), one pass per segment
  const int cw = warp >> 2, h = cw & 1, par = cw >> 1, w = warp & 3;
  uint8_t* qt = q_tiles + cw * 4096;
  uint8_t* ptl = p_tiles + cw * 4096;
  auto load_q = [&](int bq) {                         // request bq's Q tile (W3's rows: token 2w / 2w+1, head (kv0+h)*8+g)
    const uint16_t* qA = q + ((static_cast<long>(bq) * L + 2 * w) * hq + (kv0 + h) * 8 + g) * 64;
    const uint16_t* qB = qA + static_cast<long>(hq) * 64;
    const int mA = 16 * w + g, mB = mA + 8;
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      const int o = 32 * s + 4 * t4;
      *reinterpret_cast<uint32_t*>(qt + tile_off(mA, o)) = e4m3x4(bf16f(qA[o]), bf16f(qA[o + 1]), bf16f(qA[o + 2]), bf16f(qA[o + 3]));
      *reinterpret_cast<uint32_t*>(qt + tile_off(mB, o)) = e4m3x4(bf16f(qB[o]), bf16f(qB[o + 1]), bf16f(qB[o + 2]), bf16f(qB[o + 3]));
      *reinterpret_cast<uint32_t*>(qt + tile_off(mA, o + 16)) = e4m3x4(bf16f(qA[o + 16]), bf16f(qA[o + 17]), bf16f(qA[o + 18]), bf16f(qA[o + 19]));
      *reinterpret_cast<uint32_t*>(qt + tile_off(mB, o + 16)) = e4m3x4(bf16f(qB[o + 16]), bf16f(qB[o + 17]), bf16f(qB[o + 18]), bf16f(qB[o + 19]));
    }
    proxy_fence();                                    // generic Q-tile stores -> wgmma (async proxy) reads
    wg_bar(1 + cw);
  };
  const uint32_t qt_addr = smem_u32(qt), pt_addr = smem_u32(ptl);
  const uint64_t qd0 = gdesc(qt_addr, 128, 512), qd1 = gdesc(qt_addr + 256, 128, 512);
  const uint64_t pd0 = gdesc(pt_addr, 128, 512), pd1 = gdesc(pt_addr + 256, 128, 512);
  float o[32];
  float m_run[2], l_run[2];
  int posA = 0, posB = 0, s_k0 = 0, s_end = 0, s_off = 0, s_n = 0;   // the current segment

  auto wait_full = [&](int j) {
    bar_wait(&full_bar[j % STAGES], static_cast<uint32_t>((j / STAGES) & 1));
    proxy_fence();
  };
  auto issue_qk = [&](float (&acc)[32], int j) {     // stage j's Q.K into acc (async; full(j) waited)
    const uint32_t kc_addr = smem_u32(sm + (j % STAGES) * SLOTB) + h * KC_HEAD;
    fence_regs(acc);
    wgmma2_ss_issue(acc, qd0, qd1, gdesc(kc_addr, LBO_K, SBO_K), gdesc(kc_addr + 2 * LBO_K, LBO_K, SBO_K));
  };
  auto issue_pv = [&](float (&acc)[32], int j) {     // stage j's P.V (P tile, stage j's Vt) into acc (async)
    const uint32_t vt_addr = smem_u32(sm + (j % STAGES) * SLOTB + OFF_VT) + h * VTH;
    fence_regs(acc);
    wgmma2_ss_acc(acc, pd0, pd1, gdesc(vt_addr, LBO_V, SBOV), gdesc(vt_addr + 2 * LBO_V, LBO_V, SBOV));
  };
  auto release = [&](int j) {                         // stage j's K / Vt reads are complete (its P.V waited)
    __syncwarp();
    if (lane == 0) bar_arrive(&empty_bar[j % STAGES]);
  };
  // softmax of CTA stage j on sc (W1's: edge-only masks, fused scale, ftz exp2) in the current segment
  auto softmax = [&](float (&sc)[32], int j, float& alA, float& alB) {
    const int s0 = s_k0 + (j - s_off) * SKEYS;
    if (s0 + SKEYS > s_end || s0 + SKEYS > s_n - L) {
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int key = s0 + 8 * jj + 2 * t4 + c;
          const bool ok = key < s_end;
          const bool okA = ok && key <= posA;
          const bool okB = ok && key <= posB;
          sc[4 * jj + c] = okA ? sc[4 * jj + c] : -INFINITY;
          sc[4 * jj + 2 + c] = okB ? sc[4 * jj + 2 + c] : -INFINITY;
        }
    }
    float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      mxA = fmaxf(mxA, fmaxf(sc[4 * jj], sc[4 * jj + 1]));
      mxB = fmaxf(mxB, fmaxf(sc[4 * jj + 2], sc[4 * jj + 3]));
    }
    mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));
    mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));
    mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));
    mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));
    const float mnA = fmaxf(m_run[0], mxA * qk_scale), mnB = fmaxf(m_run[1], mxB * qk_scale);
    alA = ex2f(m_run[0] - mnA);
    alB = ex2f(m_run[1] - mnB);
    const float oA = P_OFF - mnA, oB = P_OFF - mnB;
    float sumA = 0.f, sumB = 0.f;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      sc[4 * jj] = ex2f(fmaf(sc[4 * jj], qk_scale, oA));
      sc[4 * jj + 1] = ex2f(fmaf(sc[4 * jj + 1], qk_scale, oA));
      sc[4 * jj + 2] = ex2f(fmaf(sc[4 * jj + 2], qk_scale, oB));
      sc[4 * jj + 3] = ex2f(fmaf(sc[4 * jj + 3], qk_scale, oB));
      sumA += sc[4 * jj] + sc[4 * jj + 1];
      sumB += sc[4 * jj + 2] + sc[4 * jj + 3];
    }
    l_run[0] = l_run[0] * alA + sumA;
    l_run[1] = l_run[1] * alB + sumB;
    m_run[0] = mnA;
    m_run[1] = mnB;
  };
  auto o_rescale = [&](float alA, float alB) {
    fence_regs(o);
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      o[4 * jj] *= alA;
      o[4 * jj + 1] *= alA;
      o[4 * jj + 2] *= alB;
      o[4 * jj + 3] *= alB;
    }
  };
  auto store_p = [&](float (&sc)[32]) {               // P -> P tile (W1's pack), warpgroup barrier
    const int mA = 16 * w + g, mB = mA + 8;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int k0 = 32 * c + 4 * t4;
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mA, k0)) = e4m3x4(sc[16 * c + 0], sc[16 * c + 1], sc[16 * c + 4], sc[16 * c + 5]);
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mB, k0)) = e4m3x4(sc[16 * c + 2], sc[16 * c + 3], sc[16 * c + 6], sc[16 * c + 7]);
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mA, k0 + 16)) = e4m3x4(sc[16 * c + 8], sc[16 * c + 9], sc[16 * c + 12], sc[16 * c + 13]);
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mB, k0 + 16)) = e4m3x4(sc[16 * c + 10], sc[16 * c + 11], sc[16 * c + 14], sc[16 * c + 15]);
    }
    proxy_fence();
    wg_bar(1 + cw);
  };

  for (int sg = 0; sg < nseg; ++sg) {
    const int bq = seg_b[sg];
    s_n = seg_n[sg];
    s_k0 = seg_k0[sg];
    s_off = seg_off[sg];
    s_end = min(s_n, s_k0 + seg_cnt[sg] * SKEYS);
    posA = s_n - L + 2 * w;
    posB = posA + 1;
    wg_bar(1 + cw);                                   // every warp of the warpgroup is past the last wgmma on the old Q tile
    load_q(bq);
#pragma unroll
    for (int k = 0; k < 32; ++k) o[k] = 0.f;
    m_run[0] = m_run[1] = -1.0e30f;
    l_run[0] = l_run[1] = 0.f;
    const int first = s_off + ((par - s_off % NP) % NP + NP) % NP;   // this pair's first stage of the segment
    const int segend = s_off + seg_cnt[sg];
    const int nloc = first < segend ? (segend - first + NP - 1) / NP : 0;
    auto gst = [&](int i) { return first + NP * i; };
    if (nloc > 0) {
      {
        float s0[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) s0[k] = 0.f;
        wait_full(gst(0));
        issue_qk(s0, gst(0));
        wg_wait0();
        fence_regs(s0);
        float alA, alB;
        softmax(s0, gst(0), alA, alB);
        store_p(s0);
      }
      for (int i = 1; i < nloc; ++i) {
        float sc[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) sc[k] = 0.f;
        wait_full(gst(i));
        issue_qk(sc, gst(i));
        issue_pv(o, gst(i - 1));
        wg_wait1();
        fence_regs(sc);
        float alA, alB;
        softmax(sc, gst(i), alA, alB);
        wg_wait0();
        o_rescale(alA, alB);
        release(gst(i - 1));
        store_p(sc);
      }
      issue_pv(o, gst(nloc - 1));
      wg_wait0();
      fence_regs(o);
      release(gst(nloc - 1));
    }
    // the segment's partial: row (token 2w + half, head (kv0+h)*8 + g), slot seg_slot, pair par
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float l = l_run[half];
      l += __shfl_xor_sync(FULL, l, 1);
      l += __shfl_xor_sync(FULL, l, 2);
      const int tok = 2 * w + half;
      const long part = ((static_cast<long>(bq) * L + tok) * hq + (kv0 + h) * 8 + g) * (static_cast<long>(gridDim.x) * NP) +
                        seg_slot[sg] * NP + par;
      float* op = o_part + part * 64 + 2 * t4;
#pragma unroll
      for (int jj = 0; jj < 8; ++jj)
        *reinterpret_cast<float2*>(op + 8 * jj) = make_float2(o[4 * jj + 2 * half], o[4 * jj + 2 * half + 1]);
      if (t4 == 0) {
        m_part[part] = m_run[half];
        l_part[part] = l;
      }
    }
  }
}
'''


def render():
    text = open(os.path.join(HERE, "cuda_w3b.py")).read()
    for pat, rep in ((r"\bokc_w3b", "okc_w3s"), (r"\bOKC_W3B_", "OKC_W3S_"), (r"\battend_w3b", "attend_w3s"),
                     (r"\bCUDA_W3B_", "CUDA_W3S_"), (r"\bW3B_VARIANTS", "W3S_VARIANTS")):
        text = re.sub(pat, rep, text)
    text = sub1(text, "int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n",
                "int hq, int hkv, int max_pages, int batch, int page, int window, float sm_scale) {\n"
                f"  constexpr int MAXSEG = {MAXSEG};                    // W3s: segments per CTA at most (batch <= MAXSEG)\n", "signature")
    text = sub1(text, "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n",
                "  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n" + HEADER, "header in")
    text = cut(text, "  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n", "  if (tid == 0) {\n#pragma unroll\n    for (int s = 0; s < STAGES; ++s) {",
               "", "old header out")
    text = sub1(text, "    asm volatile(\"fence.mbarrier_init.release.cluster;\");\n  }\n  __syncthreads();\n",
                "    asm volatile(\"fence.mbarrier_init.release.cluster;\");\n  }\n" + AFTER_SYNC, "after sync")
    # copier: keys and page-table row per stage from the segment table
    text = sub1(text, "    auto tok = [&](int j, int e) -> int { return min(start + j * SKEYS + e, end - 1); };\n",
                "    auto tok = [&](int j, int e) -> int {           // W3s: key of entry e of CTA stage j in its request\n"
                "      const int s = seg_of(j);\n"
                "      return min(seg_k0[s] + (j - seg_off[s]) * SKEYS + e, min(seg_n[s], seg_k0[s] + seg_cnt[s] * SKEYS) - 1);\n"
                "    };\n"
                "    auto ptab = [&](int j) -> const int* { return page_table + static_cast<long>(seg_b[seg_of(j)]) * max_pages; };\n",
                "copier tok")
    text = sub1(text, "cp4(ptr_ring + (j % PT_SLOTS) * 64 + 32 * cw_ + lane, pt + tok(j, 32 * cw_ + lane) / page);",
                "cp4(ptr_ring + (j % PT_SLOTS) * 64 + 32 * cw_ + lane, ptab(j) + tok(j, 32 * cw_ + lane) / page);", "copier entries")
    # consumers: segment loop
    a = text.index("  // ---- W2 consumer warpgroup h")
    b = text.index("}  // namespace okc_w3s")
    text = text[:a] + CONSUMER + "\n" + text[b:]
    # host: grid (G, pairs); the nsplit argument carries G (slot stride); window 0; batch <= MAXSEG; batch passed to the kernel
    text = sub1(text, "  const int64_t rows = batch * 8 * hq * nsplit * (nwg / 2);\n",
                "  TORCH_CHECK(window == 0 && batch <= " + str(MAXSEG) + ", \"attend_w3s: full layers, batch <= " + str(MAXSEG) + "\");\n"
                "  const int64_t rows = batch * 8 * hq * nsplit * (nwg / 2);   // nsplit = G: slot stride\n", "host check")
    text = sub1(text, "static_cast<int>(max_pages),                       \\\n        static_cast<int>(nsplit),",
                "static_cast<int>(max_pages),                       \\\n        static_cast<int>(batch),", "host batch arg")
    text = sub1(text, "  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n",
                "  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(hkv / 2));   // W3s: G CTAs per pair cover all requests\n",
                "host grid")
    m = re.findall(r"#define OKC_W3S_VARIANTS\(X\) [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "#define OKC_W3S_VARIANTS(X) " + " ".join(f"X({s}, {p}, {n}, {rc}, {rp})" for s, p, n, rc, rp in VARIANTS) + "\n")
    m = re.findall(r"\nW3S_VARIANTS = [^\n]*\n", text)
    assert len(m) == 1
    text = text.replace(m[0], "\nW3S_VARIANTS = " + repr(VARIANTS) + "\n")
    # watchdog (as make_w3d.py)
    bw = ('__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {\n  uint32_t done = 0;\n  while (!done)\n'
          '    asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n'
          '                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n')
    assert text.count(bw) == 1
    text = text.replace(bw, '__device__ unsigned int okc_w3s_hang = 0;   // watchdog flag (make_w3s.py)\n'
                        + bw.replace('  uint32_t done = 0;\n  while (!done)\n', '  uint32_t done = 0, spins = 0;\n  while (!done) {\n')
                            .replace(' : "memory");\n}\n', ' : "memory");\n'
                                     '    if (!done && ++spins > (1u << 20)) {\n      asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(&okc_w3s_hang), "r"(1u) : "memory");\n      break;\n    }\n  }\n}\n'))
    host = ('\nint64_t attend_w3s_hang() {   // lab: 1 when a W3s barrier wait gave up (watchdog); reads and clears the flag\n'
            '  unsigned int h = 0, z = 0;\n'
            '  cudaMemcpyFromSymbol(&h, okc_w3s::okc_w3s_hang, sizeof(h));\n'
            '  cudaMemcpyToSymbol(okc_w3s::okc_w3s_hang, &z, sizeof(z));\n'
            '  return h;\n}\n')
    k = text.index('"""\n\nCUDA_W3S_DECL = ')
    text = text[:k] + host + text[k:]
    text = text.replace("std::vector<int64_t> attend_w3s_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);'",
                        "std::vector<int64_t> attend_w3s_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);\\nint64_t attend_w3s_hang();'")
    assert "int64_t attend_w3s_hang();" in text
    for gone in ("const int b = blockIdx.y", "const int split = blockIdx.x", "pt + tok(", " start + j * SKEYS", "(nsplit * NP)"):
        assert gone not in text, f"left over: {gone}"
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_w3s.py from cuda_w3b.py (do not edit): W3s, W3b with a length-aware split across requests\n'
            '(see make_w3s.py). Original docstring follows.\n' + text[3:i] + text[i:])


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w3s.py up to date" if same else "cuda_w3s.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")
