"""Generates attention/local/cuda_w3.py: W3 = W2c (cuda_w2.py) with more consumer chains and copy issue per SM (A66).

L29 (AF55): full verify on W2c has two ~1 us-per-64-key-stage limits, the memory side (one copier warp, 4 stages in
flight per SM) and the consumer side (one serialized chain per warpgroup, 2 warpgroups per SM). W3 changes, by
asserted text substitutions on W2c's source (everything else byte for byte):
- P.V accumulates into O in place (FA3's mainloop: both k32 wgmmas with scale-d 1, then after softmax(i) and the
  wait for P.V(i-1), O is rescaled by alpha(i) before P.V(i) adds to it). Drops W2c's 32-register P.V fragment.
- a producer warpgroup of 4 warps: 2 copiers (copier c issues keys 32c .. 32c+31 of each stage and their page
  entries; land_bar counts 64 lanes) and W2c's 2 transposers.
- NWG consumer warpgroups: 2 (one per KV head, as W2c) or 4 (two pairs; pair `par` takes global stages
  par, par + 2, ...; each pair keeps its own O, m, l and writes its own partial at split index split * NP + par, so
  _combine merges nsplit * NP partials per row). Per-warpgroup Q and P tiles; P-tile barriers 1 + cw, ping-pong turn
  barriers 5 + 2 par + h within a pair.
- RC > 0: `setmaxnreg` (PTX 9.7.19.?, sm_90a): consumers raise to RC registers, producers lower to RP; launch bounds
  (128 NWG + 128, 1). The host launches a variant only when the compiled register count leaves a pool large enough
  for 128 NWG x RC + 128 x RP (setmaxnreg.inc would otherwise wait forever) and RP <= regs <= RC.
Shared memory: STAGES slots (Kc | Vs | Vt) | page-entry ring | NWG Q tiles | NWG P tiles.
Host: attend_w3(q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, page,
window, sm_scale, stages, pp, nwg, rc): o_part / m_part / l_part hold tokens x hq x nsplit x (nwg / 2) rows;
attend_w3_setup() (smem attributes + register-pool checks; bit v set when variant v is refused),
attend_w3_attrs(stages, pp, nwg, rc) -> (regs, local bytes, static smem, launchable).
Run: .venv/bin/python attention/local/make_w3.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w3.py")
VARIANTS = [(4, 1, 2, 0, 0), (6, 1, 2, 0, 0), (6, 1, 4, 104, 56), (7, 1, 4, 104, 56), (6, 0, 4, 104, 56),
            (6, 1, 4, 112, 32)]


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:90]!r}"
    return src.replace(old, new)


W2 = load("cuda_w2.py", "CUDA_W2_SRC")
end_ns = "}  // namespace okc_w2\n"
S = W2[:W2.index(end_ns) + len(end_ns)]
S = S.replace("okc_w2", "okc_w3")
assert "attend_w2abl" not in S

# ---- helper: the P.V pair accumulating into D (scale-d 1 on both k32 steps)
issue_start = S.index("__device__ __forceinline__ void wgmma2_ss_issue(")
issue_end = S.index("}\n", S.index('"memory");', issue_start)) + 2
acc_fn = S[issue_start:issue_end]
acc_fn = sub1(acc_fn, "void wgmma2_ss_issue(", "void wgmma2_ss_acc(", "acc name")
acc_fn = sub1(acc_fn, '"r"(0), "r"(1)', '"r"(1), "r"(1)', "acc scale-d")
S = S[:issue_end] + "// W3: the same pair with scale-d 1 on both steps: D += A * B (P.V into O in place, FA3)\n" + acc_fn + S[issue_end:]

# ---- kernel head
S = sub1(S, "template <int STAGES, int PP, int SBOV, int WF, int ABL = 0>\n__global__ void __launch_bounds__(352, 1) attend_w2_kernel("
         .replace("okc_w2", "okc_w3"),
         "template <int STAGES, int PP, int NWG, int RC, int RP, int SBOV, int WF>\n"
         "__global__ void __launch_bounds__(128 * NWG + 128, 1) attend_w3_kernel(", "template")
S = sub1(S, "  constexpr int PT_SLOTS = 2 * STAGES;\n",
         "  constexpr int PT_SLOTS = 2 * STAGES;\n  constexpr int ABL = 0;\n"
         "  constexpr int PW = 4 * NWG;                         // producer warps: copiers PW, PW + 1; transposers PW + 2, PW + 3\n"
         "  constexpr int NP = NWG / 2;                         // stage parities (consumer pairs)\n", "consts")
S = sub1(S, "  uint8_t* p_tiles = q_tiles + 2 * 4096;\n", "  uint8_t* p_tiles = q_tiles + NWG * 4096;   // W3: one Q and one P tile per consumer warpgroup\n", "tiles")
S = sub1(S, 'asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&land_bar[s])));   // copier lanes',
         'asm volatile("mbarrier.init.shared::cta.b64 [%0], 64;" ::"r"(smem_u32(&land_bar[s])));   // both copiers\' lanes', "land count")

# ---- roles: setmaxnreg, two copiers
S = sub1(S, "  if (warp == 8) {\n",
         "  if constexpr (RC > 0) {                            // whole warpgroups: producers give registers, consumers take them\n"
         "    if (warp >= PW) asm volatile(\"setmaxnreg.dec.sync.aligned.u32 %0;\" ::\"n\"(RP));\n"
         "    else asm volatile(\"setmaxnreg.inc.sync.aligned.u32 %0;\" ::\"n\"(RC));\n"
         "  }\n"
         "  if (warp == PW || warp == PW + 1) {\n"
         "    const int cw_ = warp - PW;                        // copier cw_: keys 32 cw_ .. 32 cw_ + 31 of every stage\n", "copier role")
S = sub1(S, """        cp4(ptr_ring + (j % PT_SLOTS) * 64 + lane, pt + tok(j, lane) / page);
        cp4(ptr_ring + (j % PT_SLOTS) * 64 + 32 + lane, pt + tok(j, 32 + lane) / page);
""", """        cp4(ptr_ring + (j % PT_SLOTS) * 64 + 32 * cw_ + lane, pt + tok(j, 32 * cw_ + lane) / page);
""", "copier entries")
S = sub1(S, """      const long r0 = static_cast<long>(pe[lane]) * page + tok(i, lane) % page;
      const long r1 = static_cast<long>(pe[32 + lane]) * page + tok(i, 32 + lane) % page;
""", """      const long r0 = static_cast<long>(pe[32 * cw_ + lane]) * page + tok(i, 32 * cw_ + lane) % page;
""", "copier rows")
S = sub1(S, """      for (int it = 0; it < 16; ++it) {
        const int key = 8 * (it >> 1) + (lane & 7), ch = 4 * (it & 1) + (lane >> 3);
        const long row = __shfl_sync(FULL, key < 32 ? r0 : r1, key & 31);
""", """      for (int i2 = 0; i2 < 8; ++i2) {
        const int it = 8 * cw_ + i2;                    // W2c's iterations 8 cw_ .. 8 cw_ + 7: keys 32 cw_ + (0 .. 31)
        const int key = 8 * (it >> 1) + (lane & 7), ch = 4 * (it & 1) + (lane >> 3);
        const long row = __shfl_sync(FULL, r0, key & 31);
""", "copier loop")
S = sub1(S, "  if (warp >= 9) {\n", "  if (warp >= PW + 2) {\n", "transposer role")
S = sub1(S, "        const int p = 2 * (warp - 9) + pp;\n", "        const int p = 2 * (warp - PW - 2) + pp;\n", "transposer chunks")

# ---- consumers
S = sub1(S, "  const int h = warp >> 2, w = warp & 3;\n  uint8_t* qt = q_tiles + h * 4096;\n  uint8_t* ptl = p_tiles + h * 4096;\n",
         "  const int cw = warp >> 2, h = cw & 1, par = cw >> 1, w = warp & 3;   // W3: warpgroup cw = head h of pair par\n"
         "  uint8_t* qt = q_tiles + cw * 4096;\n  uint8_t* ptl = p_tiles + cw * 4096;\n", "consumer ids")
S = sub1(S, "wg_bar(1 + h);", "wg_bar(1 + cw);", "wg bars", count=2)
S = sub1(S, "  float alA_prev = 1.f, alB_prev = 1.f;               // alpha of the stage whose P.V is issued next\n", "", "alpha prev")
S = sub1(S, "    wgmma2_ss_issue(acc, pd0, pd1,", "    wgmma2_ss_acc(acc, pd0, pd1,", "pv accumulate")
S = sub1(S, '  const int phases = nst + 1;\n',
         '  const int nloc = (nst - par + NP - 1) / NP;          // this pair\'s stages: global stage gst(i) = NP i + par\n'
         '  auto gst = [&](int i) { return NP * i + par; };\n  const int phases = nloc + 1;\n', "phases")
S = sub1(S, 'if (PP) asm volatile("bar.sync %0, 256;" ::"r"(3 + h) : "memory");',
         'if (PP) asm volatile("bar.sync %0, 256;" ::"r"(5 + 2 * par + h) : "memory");', "turn wait")
S = sub1(S, 'if (PP && (h == 0 || k + 1 < phases)) asm volatile("bar.arrive %0, 256;" ::"r"(4 - h) : "memory");',
         'if (PP && (h == 0 || k + 1 < phases)) asm volatile("bar.arrive %0, 256;" ::"r"(6 + 2 * par - h) : "memory");', "turn pass")
ou_start = S.index("  auto o_update = [&](float (&pv)[32]) {")
ou_end = S.index("  };\n", ou_start) + len("  };\n")
S = S[:ou_start] + """  auto o_rescale = [&](float alA, float alB) {       // O (P.V up to the previous stage, complete) to this stage's max
    fence_regs(o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[4 * j] *= alA;
      o[4 * j + 1] *= alA;
      o[4 * j + 2] *= alB;
      o[4 * j + 3] *= alB;
    }
  };
""" + S[ou_end:]
S = sub1(S, "    wg_bar(1 + cw);                                    // the whole warpgroup's P is in the tile\n    alA_prev = alA;\n    alB_prev = alB;\n",
         "    wg_bar(1 + cw);                                    // the whole warpgroup's P is in the tile\n    (void)alA;\n    (void)alB;\n", "store_p alphas")
ml_start = S.index("  } else if (nst > 0) {\n")
ml_end = S.index("  // partials: row (token 2w + half")
S = S[:ml_start] + """  } else if (nloc > 0) {
    if (PP && h == 1) asm volatile("bar.arrive %0, 256;" ::"r"(5 + 2 * par) : "memory");   // head 0 of the pair goes first
    {
      float s0[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) s0[k] = 0.f;
      wait_full(gst(0));
      turn_wait();
      issue_qk(s0, gst(0));
      turn_pass(0);
      wg_wait0();
      fence_regs(s0);
      float alA, alB;
      softmax(s0, gst(0), alA, alB);
      store_p(s0, alA, alB);                          // O is still zero: nothing to rescale
    }
    for (int i = 1; i < nloc; ++i) {
      float sc[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) sc[k] = 0.f;
      wait_full(gst(i));
      turn_wait();
      issue_qk(sc, gst(i));
      issue_pv(o, gst(i - 1));                        // P(i-1).V(i-1) accumulates into O in place (FA3)
      turn_pass(i);
      wg_wait1();                                     // Q.K(i) complete; P.V(i-1) runs during the softmax
      fence_regs(sc);
      float alA, alB;
      softmax(sc, gst(i), alA, alB);
      wg_wait0();                                     // P.V(i-1) complete
      o_rescale(alA, alB);                            // O to stage i's running max before P.V(i) adds to it
      release(gst(i - 1));
      store_p(sc, alA, alB);
    }
    {
      turn_wait();
      issue_pv(o, gst(nloc - 1));
      turn_pass(nloc);
      wg_wait0();
      fence_regs(o);
      release(gst(nloc - 1));
    }
  }

""" + S[ml_end:]
S = sub1(S, "    const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;\n",
         "    const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * (nsplit * NP) + split * NP + par;\n",
         "partial index")
for gone in ("o_update", "alA_prev", "float sc[32], pv[32]", "(3 + h)", "(4 - h)", "warp == 8", "warp >= 9", "key < 32 ? r0 : r1"):
    assert gone not in S, f"left over: {gone}"
assert S.count("wgmma2_ss_acc(") == 2 and S.count("issue_pv(o, ") == 2

vlist = " ".join(f"X({s}, {p}, {n}, {rc}, {rp})" for s, p, n, rc, rp in VARIANTS)
HOST = r'''
#define OKC_W3_VARIANTS(X) ''' + vlist + r'''

inline size_t okc_w3_smem(int64_t stages, int64_t nwg) {
  return static_cast<size_t>(stages) * (okc_w3::OFF_VT + 2 * 8 * 528) + static_cast<size_t>(2 * stages) * 64 * 4 +
         static_cast<size_t>(2 * nwg) * 4096;
}

static int okc_w3_refused = -1;   // bit v: variant v may not launch (register pool); -1 until attend_w3_setup ran

// a setmaxnreg variant runs only if the CTA's register pool (compiled regs per thread, allocated per warp in units of
// 256) covers 128 NWG x RC + 128 x RP, and RP <= regs <= RC (else setmaxnreg.inc waits forever or is invalid)
static bool okc_w3_pool_ok(int regs, int nwg, int rc, int rp) {
  if (rc == 0) return regs > 0;
  const long per_warp = (static_cast<long>(regs) * 32 + 255) / 256 * 256;
  const long pool = per_warp * (4 * nwg + 4);
  return regs <= rc && rp <= regs && pool >= 128L * nwg * rc + 128L * rp;
}

void attend_w3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
               torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
               torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
               int64_t page, int64_t window, double sm_scale, int64_t stages, int64_t pp, int64_t nwg, int64_t rc) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0, "attend_w3: 8 query heads per KV head, KV heads in pairs");
  TORCH_CHECK(okc_w3_refused >= 0, "attend_w3: call attend_w3_setup first");
  const int64_t rows = batch * 8 * hq * nsplit * (nwg / 2);
  TORCH_CHECK(m_part.numel() >= rows && l_part.numel() >= rows && o_part.numel() >= rows * 64,
              "attend_w3: partial buffers need tokens x hq x nsplit x nwg/2 rows");
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = okc_w3_smem(stages, nwg);
  bool launched = false;
  int v = 0;
#define OKC_W3_LAUNCH(S, D, N, RC, RP)                                                                            \
  if (!launched && stages == S && pp == D && nwg == N && rc == RC) {                                              \
    TORCH_CHECK(!((okc_w3_refused >> v) & 1), "attend_w3: variant refused (register pool)");                     \
    okc_w3::attend_w3_kernel<S, D, N, RC, RP, 528, 0><<<grid, 128 * N + 128, smem, stream>>>(                     \
        reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),         \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(),  \
        seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(),  \
        nullptr, static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages),                       \
        static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)); \
    launched = true;                                                                                              \
  }                                                                                                               \
  ++v;
  OKC_W3_VARIANTS(OKC_W3_LAUNCH)
#undef OKC_W3_LAUNCH
  TORCH_CHECK(launched, "attend_w3: no variant stages=", stages, " pp=", pp, " nwg=", nwg, " rc=", rc);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_w3 launch failed: ", cudaGetErrorString(e));
}

int64_t attend_w3_setup() {   // once, outside any capture: smem limits, register-pool checks
  int refused = 0, v = 0;
#define OKC_W3_SETUP(S, D, N, RC, RP)                                                                              \
  {                                                                                                                \
    cudaFuncAttributes a;                                                                                          \
    a.numRegs = -1;                                                                                                \
    const bool ok = cudaFuncSetAttribute(okc_w3::attend_w3_kernel<S, D, N, RC, RP, 528, 0>,                       \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
                                         static_cast<int>(okc_w3_smem(S, N))) == cudaSuccess &&                   \
                    cudaFuncGetAttributes(&a, okc_w3::attend_w3_kernel<S, D, N, RC, RP, 528, 0>) == cudaSuccess && \
                    okc_w3_pool_ok(a.numRegs, N, RC, RP);                                                          \
    if (!ok) refused |= 1 << v;                                                                                    \
    ++v;                                                                                                           \
  }
  OKC_W3_VARIANTS(OKC_W3_SETUP)
#undef OKC_W3_SETUP
  (void)cudaGetLastError();
  okc_w3_refused = refused;
  return refused;
}

std::vector<int64_t> attend_w3_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc) {
  cudaFuncAttributes a;
  a.numRegs = -1;
  a.localSizeBytes = 0;
  a.sharedSizeBytes = 0;
  int64_t launchable = 0;
#define OKC_W3_ATTR(S, D, N, RC, RP)                                                                \
  if (stages == S && pp == D && nwg == N && rc == RC) {                                             \
    cudaFuncGetAttributes(&a, okc_w3::attend_w3_kernel<S, D, N, RC, RP, 528, 0>);                   \
    launchable = okc_w3_pool_ok(a.numRegs, N, RC, RP);                                              \
  }
  OKC_W3_VARIANTS(OKC_W3_ATTR)
#undef OKC_W3_ATTR
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes), launchable};
}
'''
for s, p, n, rc, rp in VARIANTS:
    smem = s * (2 * 4096 + 64 * 144 + 2 * 8 * 528) + 2 * s * 64 * 4 + 2 * n * 4096
    assert smem <= 227 * 1024, f"variant {(s, p, n, rc, rp)}: {smem} B of shared memory"
    assert rc == 0 or (rc % 8 == 0 and rp % 8 == 0 and 24 <= rp <= rc <= 256), (s, p, n, rc, rp)
    if rc:   # the pool at the launch-bounds register cap must cover the split
        cap = 65536 // (128 * n + 128) // 8 * 8
        assert (128 * n + 128) * cap >= 128 * n * rc + 128 * rp, f"variant {(s, p, n, rc, rp)}: pool {cap} x threads too small"

SRC = S + HOST
DECL = ("void attend_w3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, "
        "torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, "
        "int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t stages, "
        "int64_t pp, int64_t nwg, int64_t rc);\nint64_t attend_w3_setup();\n"
        "std::vector<int64_t> attend_w3_attrs(int64_t stages, int64_t pp, int64_t nwg, int64_t rc);")
text = ('"""Generated by make_w3.py (do not edit): W3, W2c with P.V into O in place, two copiers and 2 or 4 consumer '
        'warpgroups (see make_w3.py)."""\nCUDA_W3_SRC = r"""\n' + SRC + '"""\n\nCUDA_W3_DECL = ' + repr(DECL) + "\n"
        + "W3_VARIANTS = " + repr(VARIANTS) + "\n")
if "--check" in sys.argv:
    old = open(OUT).read() if os.path.exists(OUT) else ""
    print("cuda_w3.py up to date" if old == text else "cuda_w3.py STALE")
    sys.exit(0 if old == text else 1)
open(OUT, "w").write(text)
print(f"wrote {OUT}: {len(SRC)} chars, variants {VARIANTS}")
