"""Generates attention/local/cuda_w4.py: W4, full / sliding verify where consumer pairs copy their own stages (AF68).
Every W2c-family design (dedicated copier warps feeding one shared ring, transposer hand-off, consumers on mbarriers)
holds ~2.2-2.3 TB/s however its copier is written; D3 reaches ~3.1 TB/s with warp-private rings. W4 has no producer warps:
a CTA holds NPAIR consumer pairs (pair = 2 warpgroups = the KV-head pair's two heads, 256 threads); each pair takes its
own split (split = blockIdx.x * NPAIR + pair of nsplit = gridDim.x * NPAIR) with its own STAGES-slot ring and runs:
  prologue: Q tiles (W3b's code), stages 0 .. S-2 copied (each warp wp: keys 8wp .. 8wp+7 of a stage, K and V whole
  rows, 4 cp.async 16 B per lane; page entries loaded one stage ahead into registers);
  stage i: wait its own groups (stage i landed), fence generic -> async proxy, pair barrier; warps 0-3 re-lay V chunk wp
  (W3b's transposer body), fence, pair barrier; each warpgroup: Q.K(i), P.V(i-1) into O in place, wait 1, softmax(i),
  wait 0, O rescale, P(i) -> its P tile (W3's FA3 order, no ping-pong); pair barrier (both warpgroups done with stage i-1);
  copies of stage i+S-1 into stage i-1's slot.
  end: the last P.V, the partial (W3b's store, slot = split).
Barriers: pair p uses named barrier 1 + p (256), P-tile barriers 3 + cw (128). No mbarriers, nothing to deadlock on
across warps but the pair barriers every thread of a pair reaches the same number of times (nst is per pair).
Variants (STAGES, NPAIR, MINB): (3, 2, 1) 512 threads; (3, 1, 2) and (6, 1, 1) 256 threads.
Host: attend_w4(q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit_total, page,
window, sm_scale, stages, npair, minb); partials as W2 with nsplit_total slots per row; attend_w4_setup(), attend_w4_attrs().
Run: .venv/bin/python attention/local/make_w4.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w4.py")
VARIANTS = [(3, 2, 1), (3, 1, 2), (6, 1, 1)]

KERNEL = r'''
template <int STAGES, int NPAIR, int MINB>
__global__ void __launch_bounds__(256 * NPAIR, MINB) attend_w4_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  constexpr int SBOV = 528;
  constexpr int VTH = 8 * SBOV;
  constexpr int SLOTB = OFF_VT + 2 * VTH;
  extern __shared__ __align__(128) uint8_t sm[];     // [NPAIR][STAGES] slots (Kc | Vs | Vt) | Q tiles [2 NPAIR] | P tiles
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int pr = warp >> 3, wp = warp & 7;           // pair, warp within the pair
  const int cw = warp >> 2, h = cw & 1, w = warp & 3;   // warpgroup in the CTA, head (warpgroup in the pair), warp in it
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x * NPAIR + pr;         // this pair's split of nsplit
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;

  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l
  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const int nst = (nchunks + 3) / 4;                  // 64-key stages of this pair's slice
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;
  const int* pt = page_table + static_cast<long>(b) * max_pages;
  uint8_t* ring = sm + pr * STAGES * SLOTB;
  uint8_t* qt = sm + NPAIR * STAGES * SLOTB + cw * 4096;
  uint8_t* ptl = sm + NPAIR * STAGES * SLOTB + 2 * NPAIR * 4096 + cw * 4096;
  auto pbar = [&]() { asm volatile("bar.sync %0, 256;" ::"r"(1 + pr) : "memory"); };   // the pair's 256 threads

  // ---- copies: warp wp, instruction it (0, 1): key 8 wp + 4 it + (lane >> 3), chunk lane & 7 (whole 128 B rows)
  auto tok = [&](int j, int it) -> int { return min(start + j * SKEYS + 8 * wp + 4 * it + (lane >> 3), end - 1); };
  auto entry = [&](int j, int it) -> int { return j < nst ? pt[tok(j, it) / page] : 0; };
  int e0 = 0, e1 = 0;                                 // page entries of the next stage to copy (keys of it 0 / 1)
  auto issue = [&](int j) {                           // stage j's rows into slot j % STAGES (entries e0 / e1), one group
    if (j < nst) {
      uint8_t* slot = ring + (j % STAGES) * SLOTB;
#pragma unroll
      for (int it = 0; it < 2; ++it) {
        const int key = 8 * wp + 4 * it + (lane >> 3), ch = lane & 7;
        const long row = static_cast<long>(it ? e1 : e0) * page + tok(j, it) % page;
        cp16(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16,
             kc + row * row_bytes + poff + 16 * ch);
        cp16(slot + OFF_VS + key * VS_PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    asm volatile("cp.async.commit_group;");
  };
  for (int j = 0; j < STAGES - 1; ++j) {
    e0 = entry(j, 0);
    e1 = entry(j, 1);
    issue(j);
  }
  e0 = entry(STAGES - 1, 0);
  e1 = entry(STAGES - 1, 1);

  // ---- Q tile of this warpgroup (W3b's rows: token 2w / 2w+1, head (kv0+h)*8+g)
  {
    const uint16_t* qA = q + ((static_cast<long>(b) * L + 2 * w) * hq + (kv0 + h) * 8 + g) * 64;
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
  }
  proxy_fence();
  wg_bar(3 + cw);
  const uint32_t qt_addr = smem_u32(qt), pt_addr = smem_u32(ptl);
  const uint64_t qd0 = gdesc(qt_addr, 128, 512), qd1 = gdesc(qt_addr + 256, 128, 512);
  const uint64_t pd0 = gdesc(pt_addr, 128, 512), pd1 = gdesc(pt_addr + 256, 128, 512);
  float o[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) o[k] = 0.f;
  float m_run[2] = {-1.0e30f, -1.0e30f}, l_run[2] = {0.f, 0.f};
  const int posA = n - L + 2 * w, posB = posA + 1;

  auto issue_qk = [&](float (&acc)[32], int j) {
    const uint32_t kc_addr = smem_u32(ring + (j % STAGES) * SLOTB) + h * KC_HEAD;
    fence_regs(acc);
    wgmma2_ss_issue(acc, qd0, qd1, gdesc(kc_addr, LBO_K, SBO_K), gdesc(kc_addr + 2 * LBO_K, LBO_K, SBO_K));
  };
  auto issue_pv = [&](float (&acc)[32], int j) {
    const uint32_t vt_addr = smem_u32(ring + (j % STAGES) * SLOTB + OFF_VT) + h * VTH;
    fence_regs(acc);
    wgmma2_ss_acc(acc, pd0, pd1, gdesc(vt_addr, LBO_V, SBOV), gdesc(vt_addr + 2 * LBO_V, LBO_V, SBOV));
  };
SOFTMAX_RESCALE_STOREP
  auto relay = [&](int j) {                           // warps 0-3 of the pair: V chunk wp of stage j -> Vt (W3b's body)
    if (wp < 4) {
      const int hh = lane >> 4, qd = lane & 15;
      const uint8_t* vs = ring + (j % STAGES) * SLOTB + OFF_VS + 64 * hh + 4 * qd;
      uint8_t* vt = ring + (j % STAGES) * SLOTB + OFF_VT + hh * VTH;
      const int p = wp;
      uint32_t wv[16];
#pragma unroll
      for (int kp = 0; kp < 16; ++kp) {
        const int key = 16 * p + 8 * ((kp & 3) >> 1) + 2 * (kp >> 2) + (kp & 1);
        wv[kp] = *reinterpret_cast<const uint32_t*>(vs + key * VS_PITCH);
      }
      uint32_t t0[4], t1[4], t2[4], t3[4];
      tr4(wv[0], wv[1], wv[2], wv[3], t0);
      tr4(wv[4], wv[5], wv[6], wv[7], t1);
      tr4(wv[8], wv[9], wv[10], wv[11], t2);
      tr4(wv[12], wv[13], wv[14], wv[15], t3);
#pragma unroll
      for (int jd = 0; jd < 4; ++jd) {
        const int dn = 4 * qd + jd;
        *reinterpret_cast<uint4*>(vt + (dn & 7) * 16 + (dn >> 3) * SBOV + p * LBO_V) = make_uint4(t0[jd], t1[jd], t2[jd], t3[jd]);
      }
    }
  };

  for (int i = 0; i < nst; ++i) {
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));   // this thread's copies of stage i landed
    proxy_fence();                                   // cp.async-written K (and Vs) -> generic / async readers
    pbar();                                          // the pair's copies of stage i are all in
    relay(i);
    proxy_fence();                                   // Vt stores -> wgmma
    pbar();
    float sc[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) sc[k] = 0.f;
    issue_qk(sc, i);
    if (i > 0) issue_pv(o, i - 1);
    if (i > 0) wg_wait1(); else wg_wait0();
    fence_regs(sc);
    float alA, alB;
    softmax(sc, i, alA, alB);
    wg_wait0();
    if (i > 0) o_rescale(alA, alB);
    store_p(sc);
    pbar();                                          // both warpgroups are past P.V(i-1): stage i-1's slot is free
    issue(i + STAGES - 1);
    e0 = entry(i + STAGES, 0);
    e1 = entry(i + STAGES, 1);
  }
  if (nst > 0) {
    issue_pv(o, nst - 1);
    wg_wait0();
    fence_regs(o);
  }
  asm volatile("cp.async.wait_all;");

  // partials: row (token 2w + half, head (kv0+h)*8 + g), slot split (W2's layout with nsplit slots)
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float l = l_run[half];
    l += __shfl_xor_sync(FULL, l, 1);
    l += __shfl_xor_sync(FULL, l, 2);
    const int tk = 2 * w + half;
    const long part = ((static_cast<long>(b) * L + tk) * hq + (kv0 + h) * 8 + g) * nsplit + split;
    float* op = o_part + part * 64 + 2 * t4;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      *reinterpret_cast<float2*>(op + 8 * j) = make_float2(o[4 * j + 2 * half], o[4 * j + 2 * half + 1]);
    if (t4 == 0) {
      m_part[part] = m_run[half];
      l_part[part] = l;
    }
  }
}

}  // namespace okc_w4
'''

HOST = r'''
#define OKC_W4_VARIANTS(X) VARLIST

inline size_t okc_w4_smem(int64_t stages, int64_t npair) {
  return static_cast<size_t>(npair) * stages * (okc_w4::OFF_VT + 2 * 8 * 528) + static_cast<size_t>(4 * npair) * 4096;
}

void attend_w4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table,
               torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch,
               int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale,
               int64_t stages, int64_t npair, int64_t minb) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0, "attend_w4: 8 query heads per KV head, KV heads in pairs");
  TORCH_CHECK(nsplit % npair == 0, "attend_w4: nsplit must be a multiple of npair");
  const int64_t rows = batch * 8 * hq * nsplit;
  TORCH_CHECK(m_part.numel() >= rows && l_part.numel() >= rows && o_part.numel() >= rows * 64, "attend_w4: partial sizes");
  dim3 grid(static_cast<unsigned>(nsplit / npair), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = okc_w4_smem(stages, npair);
  bool launched = false;
#define OKC_W4_LAUNCH(S, P, M)                                                                                    \
  if (!launched && stages == S && npair == P && minb == M) {                                                      \
    okc_w4::attend_w4_kernel<S, P, M><<<grid, 256 * P, smem, stream>>>(                                           \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),         \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),  \
        seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),  \
        static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),      \
        static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale));                         \
    launched = true;                                                                                              \
  }
  OKC_W4_VARIANTS(OKC_W4_LAUNCH)
#undef OKC_W4_LAUNCH
  TORCH_CHECK(launched, "attend_w4: no variant stages=", stages, " npair=", npair, " minb=", minb);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_w4 launch failed: ", cudaGetErrorString(e));
}

int64_t attend_w4_setup() {
  int err = 0;
#define OKC_W4_SMEM(S, P, M) \
  err |= cudaFuncSetAttribute(okc_w4::attend_w4_kernel<S, P, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(okc_w4_smem(S, P)));
  OKC_W4_VARIANTS(OKC_W4_SMEM)
#undef OKC_W4_SMEM
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> attend_w4_attrs(int64_t stages, int64_t npair, int64_t minb) {
  cudaFuncAttributes a;
  a.numRegs = -1;
  a.localSizeBytes = 0;
  a.sharedSizeBytes = 0;
#define OKC_W4_ATTR(S, P, M) \
  if (stages == S && npair == P && minb == M) cudaFuncGetAttributes(&a, okc_w4::attend_w4_kernel<S, P, M>);
  OKC_W4_VARIANTS(OKC_W4_ATTR)
#undef OKC_W4_ATTR
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
'''


def render():
    ns = {}
    exec(open(os.path.join(HERE, "cuda_w3b.py")).read(), ns)
    w3b = ns["CUDA_W3B_SRC"]
    head = w3b[:w3b.index("template <int STAGES, int PP, int NWG, int RC, int RP, int SBOV, int WF>")]
    head = head.replace("okc_w3b", "okc_w4")
    # the acc pair (scale-d 1) from W3 helpers is in the head already (wgmma2_ss_acc)
    assert "wgmma2_ss_acc" in head and "wgmma2_ss_issue" in head and "wg_wait1" in head and "tile_off" in head
    # softmax / o_rescale / store_p lambdas from W3b's consumer, byte for byte except the P-tile barrier id and store_p's
    # signature (no alphas)
    a = w3b.index("  // softmax(i) on sc (W1's: edge-only masks, fused scale, ftz exp2); returns stage i's alphas\n")
    b = w3b.index("  // FA3's intra-warpgroup order (mainloop_fwd_sm90 IntraWGOverlap)")
    lam = w3b[a:b]
    assert lam.count("wg_bar(1 + cw);") == 1
    lam = lam.replace("wg_bar(1 + cw);", "wg_bar(3 + cw);")
    lam = lam.replace("  auto store_p = [&](float (&sc)[32], float alA, float alB) {", "  auto store_p = [&](float (&sc)[32]) {")
    lam = lam.replace("    (void)alA;\n    (void)alB;\n", "")
    assert "const int s0 = start + i * SKEYS;" in lam
    kernel = KERNEL.replace("SOFTMAX_RESCALE_STOREP\n", lam)
    host = HOST.replace("VARLIST", " ".join(f"X({s}, {p}, {m})" for s, p, m in VARIANTS))
    src = head + kernel + host
    for s, p, m in VARIANTS:
        smem = p * s * (2 * 4096 + 64 * 144 + 2 * 8 * 528) + 4 * p * 4096
        assert m * (smem + 1024) <= 228 * 1024, (s, p, m, smem)
        assert s >= 2
    decl = ("void attend_w4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, "
            "torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, "
            "int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t stages, "
            "int64_t npair, int64_t minb);\nint64_t attend_w4_setup();\n"
            "std::vector<int64_t> attend_w4_attrs(int64_t stages, int64_t npair, int64_t minb);")
    return ('"""Generated by make_w4.py (do not edit): W4, consumer pairs copying their own stages (see make_w4.py)."""\n'
            'CUDA_W4_SRC = r"""\n' + src + '"""\n\nCUDA_W4_DECL = ' + repr(decl) + "\nW4_VARIANTS = " + repr(VARIANTS) + "\n")


def sim():
    """Per pair: copy groups (one per issue, prologue S-1) and the wait S-2 make stage i's rows land before its pair
    barrier; a slot is rewritten only after both warpgroups passed P.V of the stage it held; every key of a stage is
    copied once over the pair's 8 warps (keys 8 wp + 4 it + lane / 8, chunk lane % 8)."""
    for S in (2, 3, 4, 6, 7):
        for nst in range(0, 12):
            committed = 0
            done = set()
            for j in range(S - 1):
                committed += 1                      # group j = stage j (or empty if j >= nst)
            pv_done = -1
            slot_owner = {j % S: j for j in range(min(S - 1, nst))}
            for i in range(nst):
                # wait_group S-2: groups <= committed - 1 - (S - 2) complete
                for gidx in range(committed - (S - 2)):
                    done.add(gidx)
                assert i in done, (S, nst, i)
                # compute stage i: P.V(i-1) completes in this iteration
                pv_done = i - 1
                # issue stage i + S - 1 into slot (i + S - 1) % S = (i - 1) % S, whose previous stage is i - 1
                j = i + S - 1
                prev = slot_owner.get(j % S)
                if j < nst:
                    assert prev is None or prev <= pv_done, (S, nst, i, prev, pv_done)
                    slot_owner[j % S] = j
                committed += 1
    seen = {}
    for wp in range(8):
        for it in range(2):
            for lane in range(32):
                k, ch = 8 * wp + 4 * it + (lane >> 3), lane & 7
                seen[(k, ch)] = seen.get((k, ch), 0) + 1
    assert sorted(seen) == [(k, c) for k in range(64) for c in range(8)] and set(seen.values()) == {1}


if __name__ == "__main__":
    sim()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w4.py up to date" if same else "cuda_w4.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"sim ok; wrote {OUT}")
