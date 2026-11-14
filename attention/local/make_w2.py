"""Generates attention/local/cuda_w2.py: W2, W1 with its stage chain overlapped (A61) and no register A operands (AF49).

L25 anatomy (AF50): a W1 stage is one serialized chain per warpgroup (wait 199, Q.K pair 375, softmax + pack 835,
P.V pair 263, O update 104 cycles; both warpgroups in phase). L25 bisection (AF49): W1 is right only with a dead
branch that reads the Q A-registers after Q.K, so W2 keeps no long-lived A-operand registers:
- Q in shared memory: each consumer warpgroup writes its 64 x 64 E4M3 Q tile once (K-major canonical, LBO 128,
  SBO 512, as the K tile) and Q.K is an SS wgmma pair (A and B descriptors; CUTLASS MMA_64x64x32_F32E4M3E4M3_SS_TN);
- P in shared memory: each stage's P (E4M3, W1's pack and key order) goes to a 4 KB P tile per warpgroup (same
  layout, k = W1's k-position), then a named barrier per warpgroup, and P.V is an SS wgmma pair on the Vt tile;
- overlap in FA3's intra-warpgroup order (L26 / L27: windows whose count depended on runtime branches, or that crossed
  the loop back-edge, made ptxas serialize every wgmma, C7515 / C7517 / C7514): stage i >= 1 issues Q.K(i) into a fresh
  score fragment and P.V(i-1) (P tile, stage i-1's Vt), waits 1 (Q.K(i) done), runs softmax(i) while P.V(i-1) runs,
  waits 0, updates O, releases stage i-1 and writes P(i) to the tile (named barrier); stage 0's Q.K and the last P.V
  are their own phases. Plain waits followed by operand fences; every issue fences.
- template PP: 1 ping-pongs the two warpgroups (FA3's warp_scheduler_barrier): named barriers 3 / 4 hand the turn to
  issue GEMMs from one warpgroup to the other, so one warpgroup's softmax overlaps the other's wgmmas; warpgroup 0
  goes first, warpgroup 1 skips its last hand-over (nst + 1 phases each, no barrier left half-arrived).
- copier and transposers: W1's (cuda_w1.py) byte for byte (Vt stride 528, no writer fence).
Shared memory: STAGES slots (Kc | Vs | Vt) | page-entry ring | Q tiles (2 x 4 KB) | P tiles (2 x 4 KB).
Host: attend_w2(q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, page,
window, sm_scale, stages, pp), attend_w2_setup(), attend_w2_attrs(stages, pp). Variants (stages, pp): (6,0) (6,1) (8,1)
(4,1) (8 stages = 227,328 B of shared memory). Partials: W1's rows and dims, merged by _combine.
Run: .venv/bin/python attention/local/make_w2.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w2.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def between(src, a, b, what):
    assert src.count(a) == 1, f"{what}: start anchor x{src.count(a)}"
    i = src.index(a)
    return src[i:src.index(b, i)]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor x{n}: {old[:80]!r}"
    return src.replace(old, new)


W1 = load("cuda_w1.py", "CUDA_W1_SRC")
HEAD = W1[:W1.index("template <int STAGES, int SBOV, int WF>\n__global__")]
ROLES = between(W1, "template <int STAGES, int SBOV, int WF>\n__global__", "  // ---- consumer warpgroup h", "kernel head + roles")
HEAD = HEAD.replace("namespace okc_w1 {", "namespace okc_w2 {")
assert "okc_w1" not in HEAD

ACC = ", ".join(f"%{i}" for i in range(32))
OUTS = ",\n        ".join(", ".join(f'"+f"(d[{i}])' for i in range(r, r + 8)) for r in range(0, 32, 8))

HELPERS = r'''// ---- W2 helpers: SS wgmma pairs issued asynchronously (fence, two wgmma, commit; no wait), waits that bind the
// accumulators they complete, named barriers per warpgroup
__device__ __forceinline__ void wgmma2_ss_issue(float (&d)[32], uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) {
  asm volatile(
      "{\n"
      ".reg .pred p0, p1;\n"
      "setp.ne.b32 p0, %36, 0;\n"
      "setp.ne.b32 p1, %37, 0;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {''' + ACC + r'''}, %32, %34, p0, 1, 1;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {''' + ACC + r'''}, %33, %35, p1, 1, 1;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : ''' + OUTS + r'''
      : "l"(a0), "l"(a1), "l"(b0), "l"(b1), "r"(0), "r"(1)
      : "memory");
}
__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }
// (L26: waits that named the accumulators as outputs, plus runtime-dependent wait counts, made ptxas serialize every
// wgmma (C7515 / C7517); CUTLASS's form: a plain wait, then an operand fence on the completed accumulators)
__device__ __forceinline__ void wg_bar(int id) {   // the 128 threads of one consumer warpgroup
  asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}
// K-major canonical tile (as the K tile): element (row m, k) at (m % 8)*16 + (m / 8)*512 + (k % 16) + (k / 16)*128
__device__ __forceinline__ int tile_off(int m, int k) { return (m & 7) * 16 + (m >> 3) * 512 + (k & 15) + (k >> 4) * 128; }

'''

ROLES = sub1(ROLES, "template <int STAGES, int SBOV, int WF>\n__global__ void __launch_bounds__(352, 1) attend_w1_kernel(",
             "template <int STAGES, int PP, int SBOV, int WF, int ABL = 0>\n__global__ void __launch_bounds__(352, 1) attend_w2_kernel(", "template")
# ablations (L29, attend_w2abl only): 1 consumers skip all compute (wait full, release), 2 the copier skips the K/V
# copies (page entries only), 3 the transposers skip the transpose (wait land, fence, arrive)
ROLES = sub1(ROLES, """        cp16(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16,
             kc + row * row_bytes + poff + 16 * ch);
        cp16(slot + OFF_VS + key * VS_PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
""", """        if (ABL != 2) {
          cp16(slot + (ch >> 2) * KC_HEAD + (key >> 3) * SBO_K + (ch & 3) * LBO_K + (key & 7) * 16,
               kc + row * row_bytes + poff + 16 * ch);
          cp16(slot + OFF_VS + key * VS_PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
        }
""", "abl copier")
ROLES = sub1(ROLES, "      for (int pp = 0; pp < 2; ++pp) {\n", "      for (int pp = 0; pp < (ABL == 3 ? 0 : 2); ++pp) {\n", "abl transposer")
ROLES = sub1(ROLES, "float* __restrict__ l_part,\n    int hq,", "float* __restrict__ l_part, float* __restrict__ dbg,\n    int hq,", "dbg arg")
ROLES = sub1(ROLES, "  int* ptr_ring = reinterpret_cast<int*>(sm + STAGES * SLOTB);   // [PT_SLOTS][64]\n",
             "  int* ptr_ring = reinterpret_cast<int*>(sm + STAGES * SLOTB);   // [PT_SLOTS][64]\n"
             "  uint8_t* q_tiles = sm + STAGES * SLOTB + PT_SLOTS * 64 * 4;   // W2: 2 x 4 KB Q tiles, then 2 x 4 KB P tiles\n"
             "  uint8_t* p_tiles = q_tiles + 2 * 4096;\n", "tiles")

CONSUMER = r'''  // ---- W2 consumer warpgroup h (warps 4h .. 4h+3); warp w: rows g (token 2w), g+8 (token 2w+1); row m = 16w + g + 8*half
  const int h = warp >> 2, w = warp & 3;
  uint8_t* qt = q_tiles + h * 4096;
  uint8_t* ptl = p_tiles + h * 4096;
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
  proxy_fence();                                      // generic Q-tile stores -> wgmma (async proxy) reads
  wg_bar(1 + h);
  const uint32_t qt_addr = smem_u32(qt), pt_addr = smem_u32(ptl);
  const uint64_t qd0 = gdesc(qt_addr, 128, 512), qd1 = gdesc(qt_addr + 256, 128, 512);
  const uint64_t pd0 = gdesc(pt_addr, 128, 512), pd1 = gdesc(pt_addr + 256, 128, 512);
  float o[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) o[k] = 0.f;
  float m_run[2] = {-1.0e30f, -1.0e30f}, l_run[2] = {0.f, 0.f};
  float alA_prev = 1.f, alB_prev = 1.f;               // alpha of the stage whose P.V is issued next
  const int posA = n - L + 2 * w, posB = posA + 1;

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
    wgmma2_ss_issue(acc, pd0, pd1, gdesc(vt_addr, LBO_V, SBOV), gdesc(vt_addr + 2 * LBO_V, LBO_V, SBOV));
  };
  auto release = [&](int j) {                         // stage j's K / Vt reads are complete (its P.V waited)
    __syncwarp();
    if (lane == 0) bar_arrive(&empty_bar[j % STAGES]);
  };
  // ping-pong (PP, FA3's warp_scheduler_barrier): the two warpgroups take turns to issue their GEMMs, so one's
  // softmax runs while the other's wgmmas do; named barriers 3 / 4 (128 waiting + 128 arriving threads);
  // phases per warpgroup: stage 0's Q.K, each later stage's (Q.K(i), P.V(i-1)), the last P.V = nst + 1
  const int phases = nst + 1;
  auto turn_wait = [&]() {
    if (PP) asm volatile("bar.sync %0, 256;" ::"r"(3 + h) : "memory");
  };
  auto turn_pass = [&](int k) {                       // warpgroup 1 passes only if warpgroup 0 has a phase k+1
    if (PP && (h == 0 || k + 1 < phases)) asm volatile("bar.arrive %0, 256;" ::"r"(4 - h) : "memory");
  };
  // softmax(i) on sc (W1's: edge-only masks, fused scale, ftz exp2); returns stage i's alphas
  auto softmax = [&](float (&sc)[32], int i, float& alA, float& alB) {
    const int s0 = start + i * SKEYS;
    if (s0 + SKEYS > end || s0 + SKEYS > n - L || (window > 0 && s0 < lo + L)) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const int key = s0 + 8 * j + 2 * t4 + c;
          const bool ok = key < end;
          const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);
          const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);
          sc[4 * j + c] = okA ? sc[4 * j + c] : -INFINITY;
          sc[4 * j + 2 + c] = okB ? sc[4 * j + 2 + c] : -INFINITY;
        }
    }
    float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      mxA = fmaxf(mxA, fmaxf(sc[4 * j], sc[4 * j + 1]));
      mxB = fmaxf(mxB, fmaxf(sc[4 * j + 2], sc[4 * j + 3]));
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
    for (int j = 0; j < 8; ++j) {
      sc[4 * j] = ex2f(fmaf(sc[4 * j], qk_scale, oA));
      sc[4 * j + 1] = ex2f(fmaf(sc[4 * j + 1], qk_scale, oA));
      sc[4 * j + 2] = ex2f(fmaf(sc[4 * j + 2], qk_scale, oB));
      sc[4 * j + 3] = ex2f(fmaf(sc[4 * j + 3], qk_scale, oB));
      sumA += sc[4 * j] + sc[4 * j + 1];
      sumB += sc[4 * j + 2] + sc[4 * j + 3];
    }
    l_run[0] = l_run[0] * alA + sumA;
    l_run[1] = l_run[1] * alB + sumB;
    m_run[0] = mnA;
    m_run[1] = mnB;
  };
  auto o_update = [&](float (&pv)[32]) {             // O = O * alpha(that stage) + its P.V
    fence_regs(pv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[4 * j] = o[4 * j] * alA_prev + pv[4 * j];
      o[4 * j + 1] = o[4 * j + 1] * alA_prev + pv[4 * j + 1];
      o[4 * j + 2] = o[4 * j + 2] * alB_prev + pv[4 * j + 2];
      o[4 * j + 3] = o[4 * j + 3] * alB_prev + pv[4 * j + 3];
    }
  };
  // P(i) -> P tile (W1's pack: k-position 32c + 4*t4 + byte <- n-blocks 4c, 4c+1; +16 <- 4c+2, 4c+3), barrier
  auto store_p = [&](float (&sc)[32], float alA, float alB) {
    const int mA = 16 * w + g, mB = mA + 8;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int k0 = 32 * c + 4 * t4;
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mA, k0)) = e4m3x4(sc[16 * c + 0], sc[16 * c + 1], sc[16 * c + 4], sc[16 * c + 5]);
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mB, k0)) = e4m3x4(sc[16 * c + 2], sc[16 * c + 3], sc[16 * c + 6], sc[16 * c + 7]);
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mA, k0 + 16)) = e4m3x4(sc[16 * c + 8], sc[16 * c + 9], sc[16 * c + 12], sc[16 * c + 13]);
      *reinterpret_cast<uint32_t*>(ptl + tile_off(mB, k0 + 16)) = e4m3x4(sc[16 * c + 10], sc[16 * c + 11], sc[16 * c + 14], sc[16 * c + 15]);
    }
    proxy_fence();                                    // generic P-tile stores -> wgmma reads
    wg_bar(1 + h);                                    // the whole warpgroup's P is in the tile
    alA_prev = alA;
    alB_prev = alB;
  };

  // FA3's intra-warpgroup order (mainloop_fwd_sm90 IntraWGOverlap): every wgmma issued in an iteration is waited in
  // that iteration (L27: windows across the loop back-edge left ptxas serializing, C7514), score and P.V fragments
  // fresh per iteration. Stage i (i >= 1): issue Q.K(i), issue P.V(i-1), wait 1 (Q.K(i) done), softmax(i) while
  // P.V(i-1) runs, wait 0, O update, release stage i-1, P(i) -> tile.
  if (ABL == 1) {                                     // ablation: consumers only cycle the ring
    for (int i = 0; i < nst; ++i) {
      wait_full(i);
      release(i);
    }
  } else if (nst > 0) {
    if (PP && h == 1) asm volatile("bar.arrive 3, 256;" ::: "memory");   // warpgroup 0 takes the first turn
    {
      float s0[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) s0[k] = 0.f;
      wait_full(0);
      turn_wait();
      issue_qk(s0, 0);
      turn_pass(0);
      wg_wait0();
      fence_regs(s0);
      float alA, alB;
      softmax(s0, 0, alA, alB);
      store_p(s0, alA, alB);
    }
    for (int i = 1; i < nst; ++i) {
      float sc[32], pv[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) sc[k] = pv[k] = 0.f;
      wait_full(i);
      turn_wait();
      issue_qk(sc, i);
      issue_pv(pv, i - 1);
      turn_pass(i);
      wg_wait1();                                     // Q.K(i) complete; P.V(i-1) runs during the softmax
      fence_regs(sc);
      float alA, alB;
      softmax(sc, i, alA, alB);
      wg_wait0();                                     // P.V(i-1) complete
      o_update(pv);
      release(i - 1);
      store_p(sc, alA, alB);
    }
    {
      float pv[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) pv[k] = 0.f;
      turn_wait();
      issue_pv(pv, nst - 1);
      turn_pass(nst);
      wg_wait0();
      o_update(pv);
      release(nst - 1);
    }
  }

  // partials: row (token 2w + half, head (kv0+h)*8 + g), dims 8j + 2*t4 + c, V4's row order
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float l = l_run[half];
    l += __shfl_xor_sync(FULL, l, 1);
    l += __shfl_xor_sync(FULL, l, 2);
    const int tok = 2 * w + half;
    const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;
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

}  // namespace okc_w2

#define OKC_W2_VARIANTS(X) X(6, 0) X(6, 1) X(8, 1) X(4, 1)

inline size_t okc_w2_smem(int64_t stages) {
  return static_cast<size_t>(stages) * (okc_w2::OFF_VT + 2 * 8 * 528) + static_cast<size_t>(2 * stages) * 64 * 4 + 4 * 4096;
}

void attend_w2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
               torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
               torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
               int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t pp) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0, "attend_w2: 8 query heads per KV head, KV heads in pairs");
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = okc_w2_smem(stages);
  bool launched = false;
#define OKC_W2_LAUNCH(S, D)                                                                                       \
  if (!launched && stages == S && pp == D) {                                                                      \
    okc_w2::attend_w2_kernel<S, D, 528, 0><<<grid, 352, smem, stream>>>(                                          \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),         \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),  \
        seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),  \
        nullptr, static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages),                       \
        static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)); \
    launched = true;                                                                                              \
  }
  OKC_W2_VARIANTS(OKC_W2_LAUNCH)
#undef OKC_W2_LAUNCH
  TORCH_CHECK(launched, "attend_w2: no variant stages=", stages, " pp=", pp);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_w2 launch failed: ", cudaGetErrorString(e));
}

#define OKC_W2ABL_VARIANTS(X) X(1) X(2) X(3)

void attend_w2abl(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                  torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                  torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                  int64_t page, int64_t window, double sm_scale, int64_t abl) {   // 4 stages, ping-pong, ablation abl
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0, "attend_w2abl: 8 query heads per KV head, KV heads in pairs");
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = okc_w2_smem(4);
  bool launched = false;
#define OKC_W2ABL_LAUNCH(A)                                                                                       \
  if (!launched && abl == A) {                                                                                    \
    okc_w2::attend_w2_kernel<4, 1, 528, 0, A><<<grid, 352, smem, stream>>>(                                       \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),         \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),  \
        seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),  \
        nullptr, static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages),                       \
        static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)); \
    launched = true;                                                                                              \
  }
  OKC_W2ABL_VARIANTS(OKC_W2ABL_LAUNCH)
#undef OKC_W2ABL_LAUNCH
  TORCH_CHECK(launched, "attend_w2abl: no variant abl=", abl);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_w2abl launch failed: ", cudaGetErrorString(e));
}

int64_t attend_w2abl_setup() {
  int err = 0;
#define OKC_W2ABL_SMEM(A) \
  err |= cudaFuncSetAttribute(okc_w2::attend_w2_kernel<4, 1, 528, 0, A>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(okc_w2_smem(4)));
  OKC_W2ABL_VARIANTS(OKC_W2ABL_SMEM)
#undef OKC_W2ABL_SMEM
  (void)cudaGetLastError();
  return err;
}

int64_t attend_w2_setup() {   // once, outside any capture
  int err = 0;
#define OKC_W2_SMEM(S, D) \
  err |= cudaFuncSetAttribute(okc_w2::attend_w2_kernel<S, D, 528, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(okc_w2_smem(S)));
  OKC_W2_VARIANTS(OKC_W2_SMEM)
#undef OKC_W2_SMEM
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> attend_w2_attrs(int64_t stages, int64_t pp) {
  cudaFuncAttributes a;
  a.numRegs = -1;
  a.localSizeBytes = 0;
  a.sharedSizeBytes = 0;
#define OKC_W2_ATTR(S, D) \
  if (stages == S && pp == D) cudaFuncGetAttributes(&a, okc_w2::attend_w2_kernel<S, D, 528, 0>);
  OKC_W2_VARIANTS(OKC_W2_ATTR)
#undef OKC_W2_ATTR
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
'''

DECL = ("void attend_w2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
        "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
        "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
        "int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t pp);\n"
        "int64_t attend_w2_setup();\nstd::vector<int64_t> attend_w2_attrs(int64_t stages, int64_t pp);\n"
        "void attend_w2abl(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
        "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
        "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
        "int64_t page, int64_t window, double sm_scale, int64_t abl);\nint64_t attend_w2abl_setup();")


def render():
    src = HEAD + HELPERS + ROLES + CONSUMER
    assert "attend_w1" not in src and "okc_w1" not in src
    return ('"""Generated by make_w2.py (do not edit): W2, overlapped W1 with Q and P in shared memory (see make_w2.py)."""\n\n'
            f"CUDA_W2_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_W2_DECL = {DECL!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w2.py up to date" if same else "cuda_w2.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
