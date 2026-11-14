"""Generates attention/local/cuda_uw3.py: UW3 = UW2 (cuda_uw2.py, one launch per sliding verify layer) with W3's second
consumer pair (AF62: at Sv16 UW2's loop is 8.5 us of a 16.4 us span, 9 stages x 0.95 us of serial consumer chain;
W3's pair interleave took full verify from 188 to 166 us). By asserted substitutions on UW2 (rest byte for byte):
- 640 threads: consumer warps 0-15 (warpgroup cw = warp >> 2: head h = cw & 1, pair par = cw >> 1), copier warp 16,
  transposers 17-18 (UW2's copier / transposers + fresh-row patch, warp ids moved), warp 19 idle (completes the producer
  warpgroup for setmaxnreg); `__launch_bounds__(640, 1)` (96 registers), setmaxnreg 104 consumers / 56 producers.
- prologue (gate rows, queries, RoPE table, query fold into the Q tiles, the block's new rows, fresh_bar) by pair 0
  (warps 0-7, UW2's code with WC = 8); pair 1 skips it; both pairs share the Q tiles (read-only after named barrier 9).
- consumers: W3's loop (pair par takes global stages par, par + 2, ...; P.V into O in place, O rescaled after the
  softmax; P tiles per warpgroup, P-tile barriers 1 + cw, ping-pong turn barriers 11 + 2 par + h).
- after the loop (named barrier 10: both pairs done, the ring is free) pair 1 writes (o, m, l) per thread to ring
  slots 0-1, named barrier 10, pair 0 merges them into
  its own (exact softmax merge), then pair 0 pushes and finalizes as UW2 (pair 1 joins the cluster barriers only).
Host: uw3(... same as uw2 ...); uw3_setup() also refuses a variant whose register pool cannot cover the setmaxnreg split.
Run: .venv/bin/python attention/local/make_uw3.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_uw3.py")
VARIANTS = [(4, 1, 4), (4, 1, 8), (6, 0, 16), (4, 1, 16)]


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var], ns[var.replace("_SRC", "_DECL")]


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:100]!r}"
    return src.replace(old, new)


def render():
    S, D = load("cuda_uw2.py", "CUDA_UW2_SRC")
    for pat, rep in ((r"\bokc_uw2", "okc_uw3"), (r"\bOKC_UW2_", "OKC_UW3_"), (r"\buw2\(", "uw3("), (r"\buw2_setup\b", "uw3_setup"),
                     (r"\buw2_attrs\b", "uw3_attrs"), (r"\buw2_kernel\b", "uw3_kernel"), (r'"uw2: ', '"uw3: '), (r'"uw2 ', '"uw3 ')):
        S = re.sub(pat, rep, S)
        D = re.sub(pat, rep, D)
    # ---- the P.V pair accumulating into O (W3)
    i0 = S.index("__device__ __forceinline__ void wgmma2_ss_issue(")
    i1 = S.index("}\n", S.index('"memory");', i0)) + 2
    acc = sub1(sub1(S[i0:i1], "void wgmma2_ss_issue(", "void wgmma2_ss_acc(", "acc name"), '"r"(0), "r"(1)', '"r"(1), "r"(1)', "acc scale")
    S = S[:i1] + "// UW3 (W3): the same pair with scale-d 1 on both steps: D += A * B\n" + acc + S[i1:]
    # ---- kernel head
    S = sub1(S, "__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(352, 1) uw3_kernel(",
             "__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(640, 1) uw3_kernel(", "launch bounds")
    S = sub1(S, "  constexpr int SBOV = 528, WF = 0, ABL = 0, MT = 1, WC = 8;\n",
             "  constexpr int SBOV = 528, WF = 0, ABL = 0, MT = 1, WC = 8;\n"
             "  constexpr int PW = 16, NP = 2, RC = 104, RP = 56;   // UW3: producer warps 16-19, 2 consumer pairs, setmaxnreg split\n",
             "constants")
    S = sub1(S, "  float* rbuf = reinterpret_cast<float*>(p_tiles + 2 * 4096);   // [CS][RPC][LBP]\n",
             "  float* rbuf = reinterpret_cast<float*>(p_tiles + 4 * 4096);   // [CS][RPC][LBP]; UW3: 4 P tiles\n", "rbuf")
    S = sub1(S, "  uint8_t* ptl = p_tiles + h * 4096;\n",
             "  const int cw = warp >> 2, par = (warp >> 3) & 1;   // UW3: consumer warpgroup / pair\n"
             "  uint8_t* ptl = p_tiles + (cw & 3) * 4096;\n", "ptl")
    S = sub1(S, "  if (warp == 8) {   // ---- W2's copier",
             "  if (warp >= PW) asm volatile(\"setmaxnreg.dec.sync.aligned.u32 %0;\" ::\"n\"(RP));\n"
             "  else asm volatile(\"setmaxnreg.inc.sync.aligned.u32 %0;\" ::\"n\"(RC));\n"
             "  if (warp == PW) {   // ---- W2's copier", "roles head")
    S = sub1(S, "  else if (warp >= 9) {   // ---- W2's transposers + the fresh-row patch\n",
             "  else if (warp == PW + 3) {   // UW3: idle producer warp (completes the warpgroup for setmaxnreg)\n"
             "    asm volatile(\"barrier.cluster.wait.aligned;\" ::: \"memory\");\n  }\n"
             "  else if (warp >= PW + 1) {   // ---- W2's transposers + the fresh-row patch\n", "transposer role")
    S = sub1(S, "          const int r = 32 * (warp - 9) + lane, key = min(s0 + r, end - 1);\n",
             "          const int r = 32 * (warp - PW - 1) + lane, key = min(s0 + r, end - 1);\n", "patch rows")
    S = sub1(S, "        const int p = 2 * (warp - 9) + pp;\n", "        const int p = 2 * (warp - PW - 1) + pp;\n", "transpose chunk")
    # ---- prologue by pair 0 only
    S = sub1(S, "  else {   // ---- consumers (warps 0-7): UV4's prologue, then W2c's loop, then the push\n",
             "  else {   // ---- consumers (warps 0-15): UV4's prologue (pair 0), then W3's loop, then the push (pair 0)\n"
             "   if (warp < WC) {\n", "prologue open")
    S = sub1(S, "    if (tid == 0) bar_arrive(&fresh_bar);\n", "    if (tid == 0) bar_arrive(&fresh_bar);\n   }\n", "prologue close")
    S = sub1(S, "  proxy_fence();                                      // generic Q-tile stores -> wgmma (async proxy) reads\n  wg_bar(1 + h);\n",
             "  proxy_fence();                                      // generic Q-tile stores -> wgmma (async proxy) reads\n"
             "  asm volatile(\"bar.sync 9, 512;\" ::: \"memory\");   // UW3: both pairs, after pair 0's Q tiles\n", "q ready")
    # ---- W3's consumer loop (make_w3.py's edits on W2c's consumer text)
    S = sub1(S, "    wg_bar(1 + h);                                    // the whole warpgroup's P is in the tile\n    alA_prev = alA;\n    alB_prev = alB;\n",
             "    wg_bar(1 + cw);                                   // the whole warpgroup's P is in the tile\n    (void)alA;\n    (void)alB;\n", "store_p")
    S = sub1(S, "  float alA_prev = 1.f, alB_prev = 1.f;               // alpha of the stage whose P.V is issued next\n", "", "alpha prev")
    S = sub1(S, "    wgmma2_ss_issue(acc, pd0, pd1,", "    wgmma2_ss_acc(acc, pd0, pd1,", "pv accumulate")
    S = sub1(S, "  const int phases = nst + 1;\n",
             "  const int nloc = (nst - par + NP - 1) / NP;          // UW3: this pair's stages: global stage NP i + par\n"
             "  auto gst = [&](int i) { return NP * i + par; };\n  const int phases = nloc + 1;\n", "phases")
    S = sub1(S, 'if (PP) asm volatile("bar.sync %0, 256;" ::"r"(3 + h) : "memory");',
             'if (PP) asm volatile("bar.sync %0, 256;" ::"r"(11 + 2 * par + h) : "memory");', "turn wait")
    S = sub1(S, 'if (PP && (h == 0 || k + 1 < phases)) asm volatile("bar.arrive %0, 256;" ::"r"(4 - h) : "memory");',
             'if (PP && (h == 0 || k + 1 < phases)) asm volatile("bar.arrive %0, 256;" ::"r"(12 + 2 * par - h) : "memory");', "turn pass")
    a = S.index("  auto o_update = [&](float (&pv)[32]) {")
    b = S.index("  };\n", a) + len("  };\n")
    S = S[:a] + """  auto o_rescale = [&](float alA, float alB) {       // UW3 (W3): O to this stage's max before the next P.V adds
    fence_regs(o);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[4 * j] *= alA;
      o[4 * j + 1] *= alA;
      o[4 * j + 2] *= alB;
      o[4 * j + 3] *= alB;
    }
  };
""" + S[b:]
    a = S.index("  } else if (nst > 0) {\n")
    b = S.index("    asm volatile(\"cp.async.wait_all;\");               // this thread's gate pieces landed")
    S = S[:a] + """  } else if (nloc > 0) {
    if (PP && h == 1) asm volatile("bar.arrive %0, 256;" ::"r"(11 + 2 * par) : "memory");   // head 0 of the pair first
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
      store_p(s0, alA, alB);
    }
    for (int i = 1; i < nloc; ++i) {
      float sc[32];
#pragma unroll
      for (int k = 0; k < 32; ++k) sc[k] = 0.f;
      wait_full(gst(i));
      turn_wait();
      issue_qk(sc, gst(i));
      issue_pv(o, gst(i - 1));
      turn_pass(i);
      wg_wait1();
      fence_regs(sc);
      float alA, alB;
      softmax(sc, gst(i), alA, alB);
      wg_wait0();
      o_rescale(alA, alB);
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
  {   // UW3: pair 1's (o, m, l) to the free ring slots (every stage consumed), pair 0 merges them into its own
    float* mrg = reinterpret_cast<float*>(sm) + ((warp & 7) * 32 + lane) * 36;
    asm volatile("bar.sync 10, 512;" ::: "memory");   // both pairs done with every stage: the ring is free
    if (warp >= WC) {
#pragma unroll
      for (int k = 0; k < 32; ++k) mrg[k] = o[k];
      mrg[32] = m_run[0];
      mrg[33] = m_run[1];
      mrg[34] = l_run[0];
      mrg[35] = l_run[1];
    }
    asm volatile("bar.sync 10, 512;" ::: "memory");
    if (warp < WC) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float mo = mrg[32 + half], mn = fmaxf(m_run[half], mo);
        const float a0 = ex2f(m_run[half] - mn), a1 = ex2f(mo - mn);
        l_run[half] = l_run[half] * a0 + mrg[34 + half] * a1;
        m_run[half] = mn;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          o[4 * j + 2 * half] = o[4 * j + 2 * half] * a0 + mrg[4 * j + 2 * half] * a1;
          o[4 * j + 2 * half + 1] = o[4 * j + 2 * half + 1] * a0 + mrg[4 * j + 2 * half + 1] * a1;
        }
      }
    }
  }
  if (warp >= WC) {
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // pair 1 pushes nothing
  } else {

""" + S[b:]
    S = sub1(S, "        st_cluster(row + 65 * 4, l);\n      }\n    }\n  }\n  cluster_sync();",
             "        st_cluster(row + 65 * 4, l);\n      }\n    }\n  }\n  }\n  cluster_sync();", "push close")
    # ---- barrier watchdog (as make_w3d.py, after L35's hang): a wait that spins 2^20 times sets okc_uw3_hang and gives up
    bw = ('__device__ __forceinline__ void bar_wait(uint64_t* bar, uint32_t parity) {\n  uint32_t done = 0;\n  while (!done)\n'
          '    asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n'
          '                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n')
    assert S.count(bw) == 1, S.count(bw)
    S = S.replace(bw, '__device__ unsigned int okc_uw3_hang = 0;   // watchdog flag (make_uw3.py)\n'
                  + bw.replace('  uint32_t done = 0;\n  while (!done)\n', '  uint32_t done = 0, spins = 0;\n  while (!done) {\n')
                      .replace(' : "memory");\n}\n', ' : "memory");\n'
                               '    if (!done && ++spins > (1u << 20)) {\n      asm volatile("st.global.relaxed.gpu.u32 [%0], %1;" ::"l"(&okc_uw3_hang), "r"(1u) : "memory");\n      break;\n    }\n  }\n}\n'))
    S = S + ('\nint64_t uw3_hang() {   // lab: 1 when a UW3 barrier wait gave up (watchdog); reads and clears the flag\n'
             '  unsigned int h = 0, z = 0;\n'
             '  cudaMemcpyFromSymbol(&h, okc_uw3::okc_uw3_hang, sizeof(h));\n'
             '  cudaMemcpyToSymbol(okc_uw3::okc_uw3_hang, &z, sizeof(z));\n'
             '  return h;\n}\n')
    D = D + "\nint64_t uw3_hang();"
    for gone in ("o_update", "alA_prev", "(3 + h)", "(4 - h)", "warp == 8", "warp >= 9", "warp - 9"):
        assert gone not in S, f"left over: {gone}"
    # ---- host: 640 threads, 6 tiles, register-pool check
    S = sub1(S, "+ 4 * 4096 +\n         static_cast<size_t>(okc_uw3::ROWS) * okc_uw3::LBP * 4;",
             "+ 6 * 4096 +\n         static_cast<size_t>(okc_uw3::ROWS) * okc_uw3::LBP * 4;", "host smem")
    m = re.findall(r"<<<grid, 352, smem, stream>>>", S)
    assert len(m) == 1, len(m)
    S = S.replace("<<<grid, 352, smem, stream>>>", "<<<grid, 640, smem, stream>>>")
    m = re.findall(r"#define OKC_UW3_VARIANTS\(X\) [^\n]*\n", S)
    assert len(m) == 1
    S = S.replace(m[0], "#define OKC_UW3_VARIANTS(X) " + " ".join(f"X({s}, {p}, {c})" for s, p, c in VARIANTS) + "\n")
    i = S.index("int64_t uw3_setup() {")
    j = S.index("  (void)cudaGetLastError();\n", i)
    S = S[:j] + """#define OKC_UW3_POOL(S, P, C)                                                                          \\
  {                                                                                                     \\
    cudaFuncAttributes a;                                                                               \\
    a.numRegs = 999;                                                                                    \\
    cudaFuncGetAttributes(&a, okc_uw3::uw3_kernel<S, P, C>);                                            \\
    const long pool = (static_cast<long>(a.numRegs) * 32 + 255) / 256 * 256 * 20;                       \\
    if (!(a.numRegs <= 104 && 56 <= a.numRegs && pool >= 512L * 104 + 128L * 56)) err |= 1 << 30;     \\
  }
  OKC_UW3_VARIANTS(OKC_UW3_POOL)
#undef OKC_UW3_POOL
""" + S[j:]
    for s, p, c in VARIANTS:
        smem = s * (2 * 4096 + 64 * 144 + 2 * 8 * 528) + 2 * s * 64 * 4 + 6 * 4096 + 128 * 68 * 4
        assert smem <= 227 * 1024, (s, smem)
        assert s * (2 * 4096 + 64 * 144 + 2 * 8 * 528) >= 256 * 36 * 4, "pair-merge scratch fits the ring"
    i = S.index('"""', 3) + 3 if S.startswith('"""') else 0
    return ('"""Generated by make_uw3.py from cuda_uw2.py (do not edit): UW3, UW2 with W3\'s second consumer pair (see\n'
            'make_uw3.py)."""\n\nCUDA_UW3_SRC = r"""' + S + '"""\n\nCUDA_UW3_DECL = ' + repr(D) + "\n"
            + "UW3_VARIANTS = " + repr(VARIANTS) + "\n")


def merge_check():
    """Pair 0 merging pair 1's (o, m, l) by the kernel's formula equals a single softmax over both pairs' keys."""
    import random
    rnd = random.Random(7)
    for _ in range(200):
        m0, m1 = rnd.uniform(-20, 20), rnd.choice([rnd.uniform(-20, 20), -1.0e30])
        l0, l1 = rnd.uniform(0.1, 9), (0.0 if m1 == -1.0e30 else rnd.uniform(0.1, 9))
        o0, o1 = rnd.uniform(-5, 5), (0.0 if m1 == -1.0e30 else rnd.uniform(-5, 5))
        mn = max(m0, m1)
        a0, a1 = 2.0 ** (m0 - mn), 2.0 ** (m1 - mn)
        l, o = l0 * a0 + l1 * a1, o0 * a0 + o1 * a1
        want = (o0 * 2.0 ** (m0 - mn) + o1 * 2.0 ** (m1 - mn)) / (l0 * 2.0 ** (m0 - mn) + l1 * 2.0 ** (m1 - mn))
        assert abs(o / l - want) <= 1e-12 * max(1.0, abs(want))


if __name__ == "__main__":
    merge_check()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_uw3.py up to date" if same else "cuda_uw3.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"merge check ok; wrote {OUT}")
