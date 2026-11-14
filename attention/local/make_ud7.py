"""Generates attention/local/cuda_ud7.py: UD7 = UD6 (make_ud6.py) with multi-row owners, so clusters of 1 or 2 work with
4-warp CTAs: 16 / CS rows per CTA, NOW = min(16 / CS, WARPS) owner warps, RK = (16 / CS) / NOW rows per owner warp:
rows hr = rank + CS * (warp + WARPS * k), k < RK. Gates per (warp, k) at entry; the finalize merges each owned row's
CS partials, writes all RK rows to gpart before the single per-CTA ticket, and the last CTA merges each row's groups
partials (loops unrolled by 8 so the __ldcg loads overlap). Why: UD6 at Fd16 / Fd32 needs D3's occupancy (3 x 4-warp
CTAs per SM, many slices per request), which clusters >= 4 cannot give (L50: cluster capacity 92 at cs4); with CS = 1
UD7 is D3's loop + in-kernel prologue + ticketed merge: one launch instead of prologue + D3 + combine.
Host: ud7(... as ud6 ...), ud7_setup(), ud7_attrs(warps, stages, cs, grid_y).
Run: .venv/bin/python attention/local/make_ud7.py [--check]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_ud6  # noqa: E402

OUT = os.path.join(HERE, "cuda_ud7.py")
VARIANTS = [(4, 3, 1), (4, 4, 1), (4, 3, 2), (8, 2, 1), (4, 3, 16), (4, 3, 4), (4, 3, 8), (4, 4, 2)]
sub1 = make_ud6.sub1

FIN_OLD_HEAD = """  if (warp >= ROWS) return;

  // ---- 9. UD4 row owner: warp w merges row hr = rank + CS * w from its own buffer, RMSNorm x sigmoid(gate), BF16 out
  {
    const int hr = static_cast<int>(rank) + CS * warp;
    float mm = -1.0e30f;"""


def render(variants=None):
    vs = variants or VARIANTS
    text = make_ud6.render(vs, multirow=True)
    for old, new in (("okc_ud6", "okc_ud7"), ("OKC_UD6_", "OKC_UD7_"), ("CUDA_UD6_", "CUDA_UD7_"), ("UD6_VARIANTS", "UD7_VARIANTS"),
                     ("ud6(", "ud7("), ("ud6_setup", "ud7_setup"), ("ud6_attrs", "ud7_attrs"), ("ud6_kernel", "ud7_kernel"),
                     ('"ud6', '"ud7'), ("ud6_last", "ud7_last")):
        text = text.replace(old, new)
    # row ownership
    text = sub1(text, '  static_assert(16 % CS == 0 && ROWS <= WARPS, "row owners: one warp per owned row");\n',
                "  constexpr int NOW = ROWS < WARPS ? ROWS : WARPS;       // UD7: owner warps\n"
                "  constexpr int RK = ROWS / NOW;                         // UD7: rows per owner warp: hr = rank + CS * (warp + WARPS * k)\n"
                '  static_assert(16 % CS == 0 && ROWS % NOW == 0, "row owners: RK rows per owner warp");\n', "rows")
    text = sub1(text, """  float gate0 = 0.f, gate1 = 0.f;                       // gate values of this warp's owned row
  if (warp < ROWS) {
    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + (kv0 * 8 + static_cast<int>(rank) + CS * warp) * 64;
    gate0 = bf16f(gp[lane]);
    gate1 = bf16f(gp[32 + lane]);
  }
""", """  float gate0[RK], gate1[RK];                            // UD7: gate values of this warp's owned rows
#pragma unroll
  for (int k = 0; k < RK; ++k) {
    gate0[k] = gate1[k] = 0.f;
    if (warp < NOW) {
      const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + (kv0 * 8 + static_cast<int>(rank) + CS * (warp + WARPS * k)) * 64;
      gate0[k] = bf16f(gp[lane]);
      gate1[k] = bf16f(gp[32 + lane]);
    }
  }
""", "gates")
    a = text.index(FIN_OLD_HEAD)
    b = text.index("}  // namespace okc_ud7")
    fin = """  if (warp >= NOW) return;

  // ---- 9. UD7 row owners: warp w merges rows hr = rank + CS * (w + WARPS * k) from its own buffer; second level over
  // the request's groups (one ticket per CTA after all its rows are in gpart); RMSNorm x sigmoid(gate), BF16 out
  float fo0[RK], fo1[RK], fm[RK], fl[RK];
#pragma unroll
  for (int k = 0; k < RK; ++k) {
    const int hr = static_cast<int>(rank) + CS * (warp + WARPS * k);
    float mm = -1.0e30f;
#pragma unroll
    for (int r = 0; r < CS; ++r) mm = fmaxf(mm, lbuf[(r * 16 + hr) * LBP + 64]);
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
    for (int r = 0; r < CS; ++r) {
      const float* pr = lbuf + (r * 16 + hr) * LBP;
      const float wt = ex2(pr[64] - mm);
      l += wt * pr[65];
      o0 += wt * pr[lane];
      o1 += wt * pr[32 + lane];
    }
    fo0[k] = o0, fo1[k] = o1, fm[k] = mm, fl[k] = l;
  }
  if (groups > 1) {
#pragma unroll
    for (int k = 0; k < RK; ++k) {
      const int hr = static_cast<int>(rank) + CS * (warp + WARPS * k);
      float* gr = gpart + ((static_cast<long>(bpi) * groups + grp) * 16 + hr) * LBP;
      gr[lane] = fo0[k];
      gr[32 + lane] = fo1[k];
      if (lane == 0) {
        gr[64] = fm[k];
        gr[65] = fl[k];
      }
    }
    __threadfence();
    asm volatile("bar.sync 1, %0;" ::"n"(NOW * 32) : "memory");
    if (threadIdx.x == 0) ud7_last = atomicAdd(&tickets[bpi * CS + static_cast<int>(rank)], 1) == groups - 1;
    asm volatile("bar.sync 1, %0;" ::"n"(NOW * 32) : "memory");
    if (!ud7_last) return;
    __threadfence();
#pragma unroll
    for (int k = 0; k < RK; ++k) {
      const int hr = static_cast<int>(rank) + CS * (warp + WARPS * k);
      const float* g0 = gpart + (static_cast<long>(bpi) * groups * 16 + hr) * LBP;
      float m2 = -1.0e30f;
#pragma unroll 8
      for (int c = 0; c < groups; ++c) m2 = fmaxf(m2, __ldcg(g0 + c * 16 * LBP + 64));
      float l2 = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll 8
      for (int c = 0; c < groups; ++c) {
        const float* pr = g0 + c * 16 * LBP;
        const float wt = ex2(__ldcg(pr + 64) - m2);
        l2 += wt * __ldcg(pr + 65);
        a0 += wt * __ldcg(pr + lane);
        a1 += wt * __ldcg(pr + 32 + lane);
      }
      fo0[k] = a0, fo1[k] = a1, fl[k] = l2;
    }
    if (threadIdx.x == 0) tickets[bpi * CS + static_cast<int>(rank)] = 0;   // ready for the next launch
  }
#pragma unroll
  for (int k = 0; k < RK; ++k) {
    const int hr = static_cast<int>(rank) + CS * (warp + WARPS * k);
    const float scale = vsc / fl[k];
    const float o0 = fo0[k] * scale, o1 = fo1[k] * scale;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float rn = rsqrtf(ss / 64.f + 1e-5f);
    const int head = kv0 * 8 + hr;
    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;
    op[lane] = f2bf16(o0 * rn / (1.f + __expf(-gate0[k])));
    op[32 + lane] = f2bf16(o1 * rn / (1.f + __expf(-gate1[k])));
  }
}

"""
    text = text[:a] + fin + text[b:]
    # cluster size 1 needs no cluster attribute; >8 keeps the non-portable attribute (UD4c's setup)
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_ud7.py from make_ud6.render (do not edit): UD7, UD6 with multi-row owners (clusters 1..16;\n'
            'see make_ud7.py). Original docstring follows.\n' + text[3:i] + text[i:])


def sim():
    """Row ownership: for every (CS, WARPS) variant every hr in 0..15 is owned exactly once across the cluster's CTAs,
    by an owner warp (< NOW), and its gate row index is the row's; formulas asserted present."""
    src = render()
    for frag in ("const int hr = static_cast<int>(rank) + CS * (warp + WARPS * k);",
                 "(kv0 * 8 + static_cast<int>(rank) + CS * (warp + WARPS * k)) * 64",
                 "constexpr int NOW = ROWS < WARPS ? ROWS : WARPS;", "constexpr int RK = ROWS / NOW;", "if (warp >= NOW) return;",
                 'asm volatile("bar.sync 1, %0;" ::"n"(NOW * 32) : "memory");'):
        assert src.count(frag) >= 1, frag
    assert "ROWS * 32" not in src and "if (warp >= ROWS) return;" not in src
    for w, s, cs in VARIANTS:
        rows = 16 // cs
        now = min(rows, w)
        rk = rows // now
        assert rows % now == 0
        owned = {}
        for rank in range(cs):
            for warp in range(now):
                for k in range(rk):
                    hr = rank + cs * (warp + w * k)
                    assert 0 <= hr < 16 and hr not in owned, (w, cs, hr)
                    owned[hr] = (rank, warp, k)
                    assert hr % cs == rank   # the push sends row hr to owner hr % CS
        assert len(owned) == 16, (w, cs)
    return True


if __name__ == "__main__":
    sim()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ud7.py up to date" if same else "cuda_ud7.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} (row ownership model ok)")
