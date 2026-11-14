"""Generates attention/local/cuda_ud6.py: UD6 = UD5 (make_ud5.py) for any context length: `groups` clusters of CS CTAs per
(request, KV-head pair), split index grp * CS + rank over nsplit = groups * CS slices. Each cluster merges its CS
partials at the row owners in shared memory as UD5; with groups > 1 each owner warp then writes its row's cluster partial
(o unnormalized, m, l in the exp2 domain) to gpart[bp][grp][row][LBP], fences, and one atomicAdd per CTA on
tickets[bp * CS + rank] picks the last of the `groups` CTAs holding that rank's rows; it merges the groups partials
(__ldcg), resets the ticket for the next launch (graph replay) and finalizes as UD5 (its gates were prefetched at entry:
they depend on rank only). One launch per decode layer, full or sliding: no prologue / combine kernels (v14's full
decode runs prologue + D3 + combine). Pattern as UV4's second level (v14 source) and CUDA's threadFenceReduction.
groups == 1 is UD5 exactly (no global traffic). The window check is gone (full layers: window 0, no RoPE).
Host: ud6(... as ud5 ..., warps, stages, cs, groups, gpart, tickets), ud6_setup(), ud6_attrs(warps, stages, cs, grid_y).
Run: .venv/bin/python attention/local/make_ud6.py [--check]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_ud5  # noqa: E402

OUT = os.path.join(HERE, "cuda_ud6.py")
VARIANTS = [(4, 3, 16), (8, 2, 16), (4, 3, 8), (8, 3, 8), (4, 3, 4), (4, 4, 4), (8, 3, 4)]
sub1 = make_ud5.sub1

SECOND = """    if (groups > 1) {   // UD6: second level over the request's `groups` clusters (ticket per (bp, rank))
      float* gr = gpart + ((static_cast<long>(bpi) * groups + grp) * 16 + hr) * LBP;
      gr[lane] = o0;
      gr[32 + lane] = o1;
      if (lane == 0) {
        gr[64] = mm;
        gr[65] = l;
      }
      __threadfence();
      asm volatile("bar.sync 1, %0;" ::"n"(ROWS * 32) : "memory");
      if (threadIdx.x == 0) ud6_last = atomicAdd(&tickets[bpi * CS + static_cast<int>(rank)], 1) == groups - 1;
      asm volatile("bar.sync 1, %0;" ::"n"(ROWS * 32) : "memory");
      if (!ud6_last) return;
      __threadfence();
      const float* g0 = gpart + (static_cast<long>(bpi) * groups * 16 + hr) * LBP;
      float m2 = -1.0e30f;
      for (int c = 0; c < groups; ++c) m2 = fmaxf(m2, __ldcg(g0 + c * 16 * LBP + 64));
      float l2 = 0.f, a0 = 0.f, a1 = 0.f;
      for (int c = 0; c < groups; ++c) {
        const float* pr = g0 + c * 16 * LBP;
        const float wt = ex2(__ldcg(pr + 64) - m2);
        l2 += wt * __ldcg(pr + 65);
        a0 += wt * __ldcg(pr + lane);
        a1 += wt * __ldcg(pr + 32 + lane);
      }
      o0 = a0;
      o1 = a1;
      l = l2;
      if (threadIdx.x == 0) tickets[bpi * CS + static_cast<int>(rank)] = 0;   // ready for the next launch
    }
    const float scale = vsc / l;
"""


def render(variants=None, multirow=False):
    text = make_ud5.render(variants or VARIANTS, multirow)
    for old, new in (("okc_ud5", "okc_ud6"), ("OKC_UD5_", "OKC_UD6_"), ("CUDA_UD5_", "CUDA_UD6_"), ("UD5_VARIANTS", "UD6_VARIANTS")):
        text = text.replace(old, new)
    for old, new in (("ud5(", "ud6("), ("ud5_setup", "ud6_setup"), ("ud5_attrs", "ud6_attrs"), ("ud5_kernel", "ud6_kernel"),
                     ('"ud5', '"ud6')):
        text = text.replace(old, new)
    assert "ud5" not in text.split('"""', 2)[2].replace("make_ud5", ""), "leftover ud5 name"
    # kernel parameters and block indices
    text = sub1(text, "    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {\n",
                "    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale,\n"
                "    int groups, float* __restrict__ gpart, int* __restrict__ tickets) {\n"
                "  __shared__ int ud6_last;\n", "params")
    text = sub1(text, "  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n",
                "  const int bpi = blockIdx.y / groups, grp = blockIdx.y % groups;   // UD6: (request, pair) and cluster group\n"
                "  const int b = bpi / pairs;\n  const int kv0 = (bpi % pairs) * 2;\n"
                "  const int split = blockIdx.x + CS * grp;             // UD6: slice of nsplit = groups * CS\n", "indices")
    text = sub1(text, "    const float scale = vsc / l;\n", SECOND, "second level")
    # host
    text = sub1(text, "int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs) {\n",
                "int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs,\n"
                "         int64_t groups, torch::Tensor gpart, torch::Tensor tickets) {\n", "host signature")
    text = sub1(text, '  TORCH_CHECK(window > 0, "ud3 serves sliding layers (one cluster per request and KV-head pair)");\n'
                      "  dim3 grid(static_cast<unsigned>(cs), static_cast<unsigned>(batch * (hkv / 2)));\n",
                '  TORCH_CHECK(groups >= 1 && hq == 8 * hkv && hkv % 2 == 0, "ud6: 8 query heads per KV head, KV heads in pairs");\n'
                "  TORCH_CHECK(groups == 1 || (gpart.numel() >= batch * (hkv / 2) * groups * 16 * okc_ud6::LBP &&\n"
                "                              tickets.numel() >= batch * (hkv / 2) * cs), \"ud6: gpart / tickets too small\");\n"
                "  dim3 grid(static_cast<unsigned>(cs), static_cast<unsigned>(batch * (hkv / 2) * groups));\n", "host grid")
    text = sub1(text, "      static_cast<int>(max_pages), static_cast<int>(cs), static_cast<int>(page), static_cast<int>(window),    \\\n"
                      "      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)\n",
                "      static_cast<int>(max_pages), static_cast<int>(cs * groups), static_cast<int>(page), static_cast<int>(window), \\\n"
                "      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale), static_cast<int>(groups), \\\n"
                "      gpart.data_ptr<float>(), tickets.data_ptr<int>()\n", "host args")
    text = sub1(text, "int64_t warps, int64_t stages, int64_t cs);\\nint64_t ud6_setup();",
                "int64_t warps, int64_t stages, int64_t cs, int64_t groups, torch::Tensor gpart, torch::Tensor tickets);\\nint64_t ud6_setup();",
                "decl")
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_ud6.py from make_ud5.render (do not edit): UD6, UD5 with `groups` clusters per request and a\n'
            'ticketed second-level merge (see make_ud6.py). Original docstring follows.\n' + text[3:i] + text[i:])


def sim():
    """Second-level merge model: random per-CTA partials, cluster merge at owners (UD5), then groups partials through
    gpart with the source's index formulas, merged by the last ticket holder per rank; equals a direct merge of all
    nsplit partials for every (bp, row). Formulas asserted present in the generated source. Ticket order random."""
    import math
    import random
    src = render()
    for frag in ("const int split = blockIdx.x + CS * grp;", "const int bpi = blockIdx.y / groups, grp = blockIdx.y % groups;",
                 "float* gr = gpart + ((static_cast<long>(bpi) * groups + grp) * 16 + hr) * LBP;",
                 "const float* g0 = gpart + (static_cast<long>(bpi) * groups * 16 + hr) * LBP;",
                 "const float* pr = g0 + c * 16 * LBP;", "for (int c = 0; c < groups; ++c) m2 = fmaxf(m2, __ldcg(g0 + c * 16 * LBP + 64));",
                 "atomicAdd(&tickets[bpi * CS + static_cast<int>(rank)], 1) == groups - 1",
                 "const int hr = static_cast<int>(rank) + CS * warp;", "static_cast<int>(cs * groups)",
                 "static_cast<unsigned>(batch * (hkv / 2) * groups)"):
        assert src.count(frag) == 1, frag
    LBP = 68
    rnd = random.Random(6)
    for CS, G, BP in ((16, 8, 1), (8, 2, 3), (4, 4, 2), (4, 1, 2)):
        nsplit = G * CS
        part = {(bp, s, hr): (rnd.uniform(-20, 5), rnd.uniform(0.5, 2), [rnd.gauss(0, 1) for _ in range(4)])
                for bp in range(BP) for s in range(nsplit) for hr in range(16)}
        gpart = {}
        tickets = {}
        outs = {}
        order = [(bp, grp, rank) for bp in range(BP) for grp in range(G) for rank in range(CS)]
        rnd.shuffle(order)
        for blocky in range(BP * G):
            bpi, grp = blocky // G, blocky % G
            assert blocky == bpi * G + grp
        for bpi, grp, rank in order:
            for w in range(16 // CS):
                hr = rank + CS * w
                # cluster merge over its CS slices: split = rank' + CS * grp
                ms = [part[(bpi, r2 + CS * grp, hr)] for r2 in range(CS)]
                mm = max(m for m, _, _ in ms)
                l = sum(2 ** (m - mm) * li for m, li, _ in ms)
                o = [sum(2 ** (m - mm) * oi[k] for m, _, oi in ms) for k in range(4)]
                gpart[((bpi * G + grp) * 16 + hr) * LBP] = (mm, l, o)
            t = tickets.get(bpi * CS + rank, 0)
            tickets[bpi * CS + rank] = t + 1
            if t == G - 1 or G == 1:
                for w in range(16 // CS):
                    hr = rank + CS * w
                    if G == 1:
                        mm, l, o = gpart[((bpi * G + 0) * 16 + hr) * LBP]
                    else:
                        g0 = (bpi * G * 16 + hr) * LBP
                        ps = [gpart[g0 + c * 16 * LBP] for c in range(G)]
                        m2 = max(p[0] for p in ps)
                        l = sum(2 ** (p[0] - m2) * p[1] for p in ps)
                        o = [sum(2 ** (p[0] - m2) * p[2][k] for p in ps) for k in range(4)]
                    assert (bpi, hr) not in outs
                    outs[(bpi, hr)] = [x / l for x in o]
                tickets[bpi * CS + rank] = 0
        assert len(outs) == BP * 16 and all(v == 0 for v in tickets.values())
        for bp in range(BP):
            for hr in range(16):
                ps = [part[(bp, s, hr)] for s in range(nsplit)]
                M = max(p[0] for p in ps)
                L = sum(2 ** (p[0] - M) * p[1] for p in ps)
                ref = [sum(2 ** (p[0] - M) * p[2][k] for p in ps) / L for k in range(4)]
                assert all(math.isclose(a, r, rel_tol=1e-9, abs_tol=1e-12) for a, r in zip(outs[(bp, hr)], ref)), (CS, G, bp, hr)
    return True


if __name__ == "__main__":
    sim()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ud6.py up to date" if same else "cuda_ud6.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} (second-level merge model ok)")
