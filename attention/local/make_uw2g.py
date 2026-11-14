"""Generates attention/local/cuda_uw2g.py: UW2g = UW2 (cuda_uw2.py, one-launch sliding verify) for any context length,
with UV4's second level ported verbatim (cuda_uv4.py step 8, v14 source): `ncl` clusters of CS CTAs per (request, KV-head
pair), slice cl * CS + rank of nsplit = CS * ncl (cluster-major, as UV4); each CTA's merged rows go to gpart[bp][cl][R][LBP],
`__threadfence`, one atomicAdd per CTA on tickets[bp * CS + rank]; the last cluster's CTA merges the ncl partials with
__ldcg, resets the ticket and finalizes. ncl == 1 is UW2 exactly. The window check is gone (full verify: window 0, no
RoPE). One launch per full verify layer (v14's Fv1: prologue + W2 + combine).
Host: uw2g(... as uw2 ..., stages, pp, cs, ncl, gpart, tickets), uw2g_setup(), uw2g_attrs().
Run: .venv/bin/python attention/local/make_uw2g.py [--check]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_uw2g.py")


def sub1(src, old, new, what, count=1):
    n = src.count(old)
    assert n == count, f"{what}: anchor x{n} (want {count}): {old[:90]!r}"
    return src.replace(old, new)


def uv4_second_level():
    ns = {}
    exec(open(os.path.join(HERE, "cuda_uv4.py")).read(), ns)
    src = ns["CUDA_UV4_SRC"]
    a = src.index("  if (ncl > 1) {\n    // ---- 8. second level")
    b = src.index("  // ---- 9. RMSNorm x sigmoid(gate), BF16 out", a)
    return src[a:b]


def render():
    text = open(os.path.join(HERE, "cuda_uw2.py")).read()
    for pat, rep in ((r"\bokc_uw2", "okc_uw2g"), (r"\bOKC_UW2_", "OKC_UW2G_"), (r"\buw2\(", "uw2g("), (r"\buw2_setup\b", "uw2g_setup"),
                     (r"\buw2_attrs\b", "uw2g_attrs"), (r"\buw2_kernel\b", "uw2g_kernel"), (r"\bCUDA_UW2_", "CUDA_UW2G_"),
                     (r'"uw2: ', '"uw2g: '), (r'"uw2 ', '"uw2g ')):
        text = re.sub(pat, rep, text)
    second = uv4_second_level()
    text = sub1(text, "    int hq, int hkv, int max_pages, int page, int window, int rope, float qk_factor, float sm_scale) {\n",
                "    int hq, int hkv, int max_pages, int page, int window, int rope, float qk_factor, float sm_scale,\n"
                "    int ncl, float* __restrict__ gpart, int* __restrict__ tickets) {\n"
                "  __shared__ int s_last;                              // UW2g (UV4): this CTA's cluster finished last\n", "params")
    text = sub1(text, "  const int bp = blockIdx.y;\n", "  const int bp = blockIdx.y / ncl, cl = blockIdx.y % ncl;   // UW2g: (request, KV-head pair), its cluster\n", "bp")
    text = sub1(text, "  const int nsplit = CS;\n  const int split = blockIdx.x;                       // grid x = CS = cluster x: blockIdx.x == cluster rank\n",
                "  const int nsplit = CS * ncl;                        // UW2g: slices of the request's keys, cluster-major\n"
                "  const int split = cl * CS + blockIdx.x;             // grid x = CS = cluster x: blockIdx.x == cluster rank\n", "split")
    text = sub1(text, "    fo0[j] = o0, fo1[j] = o1, fm[j] = mm, fl[j] = l;\n  }\n  // ---- 9. RMSNorm x sigmoid(gate), BF16 out\n",
                "    fo0[j] = o0, fo1[j] = o1, fm[j] = mm, fl[j] = l;\n  }\n" + second + "  // ---- 9. RMSNorm x sigmoid(gate), BF16 out\n",
                "second level")
    # host
    text = sub1(text, "         int64_t rope, double qk_factor, double sm_scale, int64_t stages, int64_t pp, int64_t cs) {\n",
                "         int64_t rope, double qk_factor, double sm_scale, int64_t stages, int64_t pp, int64_t cs, int64_t ncl,\n"
                "         torch::Tensor gpart, torch::Tensor tickets) {\n", "host signature")
    text = sub1(text, '  TORCH_CHECK(window > 0, "uw2g serves sliding layers (one cluster per request and KV-head pair)");\n'
                      "  dim3 grid(static_cast<unsigned>(cs), static_cast<unsigned>(batch * (hkv / 2)));\n",
                "  TORCH_CHECK(ncl >= 1 && (ncl == 1 || (gpart.numel() >= batch * (hkv / 2) * ncl * okc_uw2g::ROWS * okc_uw2g::LBP &&\n"
                "              tickets.numel() >= batch * (hkv / 2) * cs)), \"uw2g: gpart / tickets too small\");\n"
                "  dim3 grid(static_cast<unsigned>(cs), static_cast<unsigned>(batch * (hkv / 2) * ncl));\n", "host grid")
    text = sub1(text, "        static_cast<float>(qk_factor), static_cast<float>(sm_scale));                                           \\\n",
                "        static_cast<float>(qk_factor), static_cast<float>(sm_scale), static_cast<int>(ncl), gpart.data_ptr<float>(), \\\n"
                "        tickets.data_ptr<int>());                                                                               \\\n", "host args")
    text = sub1(text, "int64_t stages, int64_t pp, int64_t cs);\\nint64_t uw2g_setup();",
                "int64_t stages, int64_t pp, int64_t cs, int64_t ncl, torch::Tensor gpart, torch::Tensor tickets);\\nint64_t uw2g_setup();", "decl")
    text = sub1(text, "#define OKC_UW2G_VARIANTS(X) X(4, 1, 4) X(4, 1, 8) X(4, 1, 16) X(6, 0, 16)\n",
                "#define OKC_UW2G_VARIANTS(X) X(4, 1, 4) X(4, 1, 8) X(4, 1, 16) X(6, 0, 16) X(6, 0, 8) X(6, 1, 8) X(6, 1, 16)\n", "variants")
    i = text.index('"""', 3) + 3
    return ('"""Generated by make_uw2g.py from cuda_uw2.py (do not edit): UW2g, UW2 with UV4\'s second-level merge (see\n'
            'make_uw2g.py). Original docstring follows.\n' + text[3:i] + text[i:])


def check():
    """UV4's second level reads only names UW2g defines with UV4's meaning: bp, cl, ncl, ROWS, RPC, RPW, WC, LBP, rank, tid,
    lane, warp, fo0/fo1/fm/fl, gpart, tickets, s_last; ROWS / LBP equal in both kernels; UV4's split is cluster-major."""
    ns = {}
    exec(open(os.path.join(HERE, "cuda_uv4.py")).read(), ns)
    uv4 = ns["CUDA_UV4_SRC"]
    t = render()
    for frag in ("const int bp = blockIdx.y / ncl, cl = blockIdx.y % ncl;", "const int nsplit = CS * ncl;"):
        assert uv4.count(frag) == 1 and t.count(frag) == 1, frag
    assert re.search(r"const int split = cl \* CS \+ blockIdx\.x;", uv4) or re.search(r"const int split = blockIdx\.x \+ CS \* cl;", uv4) \
        or "cl * CS + rank" in uv4 or "split = cl * CS" in uv4, "UV4 split formula"
    for k in ("ROWS", "LBP"):
        a = re.search(rf"constexpr int {k} = (\d+);", uv4).group(1)
        b = re.search(rf"constexpr int {k} = (\d+);", t).group(1)
        assert a == b, (k, a, b)
    for k in ("RPC = ROWS / CS", "RPW = RPC / WC", "WC = 8"):
        assert k in uv4 and k in t, k
    return True


if __name__ == "__main__":
    check()
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_uw2g.py up to date" if same else "cuda_uw2g.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")
