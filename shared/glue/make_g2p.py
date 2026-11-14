"""Build shared/glue/g2p.py: M11's GEMM2 fused with the post-expert norm weighted sum (TRACKER shared item 5,
for muse_decode_layer; MoE owns M11).

Source: decode_layer/parts/m11_pdl.py's _M11_CUDA (read with ast, never executed). The GEMM kernel is copied byte for
byte and turned into `gemm_post_kernel` by exact-anchor edits only:
  1. name and signature (+ nw, tw, pout, cnt, eps);
  2. after the PDL wait: CTA b < tokens writes token b's output row as zeros when b has no local route;
  3. after each task (an expert's TT row tiles for all its pairs): the last-arriver epilogue G2P_TASK_DONE.
Everything else (scan, task split, decode, MMA, K split, y epilogue, PDL) is M11's text.

Last arriver (graph-safe, self-resetting counters cnt[0..31] per local expert, cnt[32 + t] per token, all zero
between calls): warp kpart 0 of the task's group fences its y stores and adds 1 to cnt[e]; the group that brings it
to per_expert owns expert e: it resets cnt[e], then for each pair p of e adds 1 to cnt[32 + p / 8]; whoever brings
a token's counter to its number of local routes resets it and runs the post-norm of that token (M13's math, two
passes over the token's y rows: sum of squares, then the weighted sum) from y read with ld.global.cg.

Run: .venv/bin/python shared/glue/make_g2p.py   (writes shared/glue/g2p.py)
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "decode_layer" / "parts" / "m11_pdl.py"
OUT = Path(__file__).resolve().parent / "g2p.py"
# MoE variant: the same edits on MoE's own live M11 (its gemm_kernel = decode_layer's minus L8's pf argument/block).
MOE_SRC = ROOT / "moe" / "submission_m13t_e22.py"
MOE_OUT = Path(__file__).resolve().parent / "g2p_moe.py"
MOE_INSTANCES = ((1, 1, 4, 4, 1), (2, 1, 4, 4, 1), (2, 1, 4, 4, 0))  # MoE's live bs1-16 GEMM2 stages

# Fused instances (tt, nt, depth, ks, pdl): decode_layer's V3_M11_CONFIGS GEMM2 stages at 1-32 tokens.
G2P_INSTANCES = ((1, 1, 4, 4, 1), (2, 1, 4, 4, 1), (2, 1, 4, 4, 0), (2, 2, 2, 4, 0))


def m11_cuda(src=SRC):
    tree = ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "_M11_CUDA":
            return node.value.value
    raise SystemExit("make_g2p: _M11_CUDA not found in " + str(src))


def edit(text, old, new):
    assert text.count(old) == 1, ("anchor must occur exactly once", old[:120], text.count(old))
    return text.replace(old, new)


HELPERS_END = "// PDL (GEMM2 only): launched early with programmatic stream serialization;"
KERNEL_END = "\n\n\n// ------------------------------------------------------------------------------------------------\n// M12:"

SIG_OLD = """    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int pf) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");"""
SIG_NEW = """    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int pf,
    const __nv_bfloat16* __restrict__ nw, const float* __restrict__ tw, __nv_bfloat16* __restrict__ pout,
    int* __restrict__ cnt, float eps) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  static_assert(!GATED, "G2P: GEMM2 only");
  __shared__ int s_fin[4 / KS];
  __shared__ int s_ndone[4 / KS];
  __shared__ int s_done[4 / KS][MAXP];
  __shared__ float s_pss[4 / KS][KS][TOP_K];
  __shared__ float s_rs[4 / KS][TOP_K];"""

WAIT_OLD = """  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)
"""
WAIT_NEW = WAIT_OLD + """  for (int zt = blockIdx.x; zt < n_pairs / TOP_K; zt += gridDim.x) {  // G2P: tokens without a local route: zero rows
    int any = 0;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      const int ze = ids[zt * TOP_K + k] - local_start;
      any |= (ze >= 0 && ze < LOCAL) ? 1 : 0;
    }
    if (!any)
      for (int i = tid; i < K / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(pout + static_cast<size_t>(zt) * K)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
"""

TAIL_OLD = """
    }
  }
}"""
TAIL_NEW = """
    }
    G2P_TASK_DONE
  }
}"""

TASK_DONE = r"""{  // G2P: this task's y tiles are written (warp kpart 0 of the group stored them)
      const int tig = kpart * 32 + lane;  // thread in group
      if (kpart == 0) {
        __threadfence();  // release: this lane's y stores before the counter
        __syncwarp();
        int fin = 0;
        if (lane == 0) {
          fin = atomicAdd(cnt + e, 1) == per_expert - 1;
          if (fin) {
            __threadfence();  // acquire: every other task's y stores of expert e
            atomicExch(cnt + e, 0);  // self-reset: no other task of this call touches cnt[e] again
          }
        }
        fin = __shfl_sync(0xffffffffu, fin, 0);
        if (lane == 0) {
          s_fin[grp] = fin;
          s_ndone[grp] = 0;
        }
      }
      g2p_sync(grp, KS);
      if (s_fin[grp]) {  // this group finished expert e: count its pairs' tokens
        for (int i = tig; i < total; i += KS * 32) {
          const int tk = s_list[e][i] / TOP_K;
          int nl = 0;
#pragma unroll
          for (int k = 0; k < TOP_K; ++k) {
            const int e2 = ids[tk * TOP_K + k] - local_start;
            nl += (e2 >= 0 && e2 < LOCAL) ? 1 : 0;
          }
          __threadfence();  // release by this thread: expert e's y rows (acquired by the group above) before the token counter
          if (atomicAdd(cnt + LOCAL + tk, 1) == nl - 1) {
            __threadfence();  // acquire: the other experts' y rows of token tk
            atomicExch(cnt + LOCAL + tk, 0);
            s_done[grp][atomicAdd(&s_ndone[grp], 1)] = tk;
          }
        }
        g2p_sync(grp, KS);
        const int nd = s_ndone[grp];
        for (int d = 0; d < nd; ++d)
          g2p_token<KS>(s_done[grp][d], tig, grp, ids, local_start, a_y(out), nw, tw, pout, eps, s_pss, s_rs);
      }
      g2p_sync(grp, KS);  // s_fin / s_done / s_ndone are rewritten by the next task
    }"""

DEVICE = r"""
// ------------------------------------------------------------------------------------------------ G2P
// Group barrier of KS warps (M11's group_sync ids 1-4), or a warp sync for a one-warp group.
__device__ __forceinline__ void g2p_sync(int grp, int ks) {
  if (ks == 1) __syncwarp();
  else group_sync(grp + 1, ks * 32);
}

__device__ __forceinline__ const __nv_bfloat16* a_y(const __nv_bfloat16* p) { return p; }

// M13's post-norm for one token by the KS * 32 threads of a group: v = y * nw (y bf16, nw bf16, fp32 products),
// rs[k] = rsqrt(sum v^2 / K + eps), out = bf16_rn(sum over local k of (tw * rs[k]) * v). Two passes over the rows
// (sum of squares, then the weighted sum) keep registers low; thread tig owns uint4 columns tig + i * KS * 32.
template <int KS>
__device__ __forceinline__ void g2p_token(int tk, int tig, int grp, const int* __restrict__ ids, int local_start,
                                          const __nv_bfloat16* y, const __nv_bfloat16* __restrict__ nw,
                                          const float* __restrict__ tw, __nv_bfloat16* __restrict__ pout, float eps,
                                          float (*s_pss)[KS][TOP_K], float (*s_rs)[TOP_K]) {
  constexpr int NT_ = KS * 32;
  constexpr int PER = (K / 8) / NT_;  // uint4 columns per thread
  const int lane = tig & 31, wq = tig >> 5;
  bool loc[TOP_K];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    const int e = ids[tk * TOP_K + k] - local_start;
    loc[k] = e >= 0 && e < LOCAL;
    ss[k] = 0.0f;
  }
#pragma unroll
  for (int c = 0; c < PER; ++c) {
    const int col = tig + c * NT_;
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + col);
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (!loc[k]) continue;
      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t u = word(r, j), wu = word(wraw, j);
        const float v0 = __uint_as_float(u << 16) * __uint_as_float(wu << 16);
        const float v1 = __uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u);
        ss[k] += v0 * v0 + v1 * v1;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  if (lane == 0)
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) s_pss[grp][wq][k] = ss[k];
  g2p_sync(grp, KS);
  if (tig < TOP_K) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < KS; ++i) s += s_pss[grp][i][tig];
    s_rs[grp][tig] = rsqrtf(s / static_cast<float>(K) + eps);
  }
  g2p_sync(grp, KS);
  float f[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) f[k] = loc[k] ? tw[tk * TOP_K + k] * s_rs[grp][k] : 0.0f;
#pragma unroll
  for (int c = 0; c < PER; ++c) {
    const int col = tig + c * NT_;
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + col);
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = 0.0f;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (!loc[k]) continue;
      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t u = word(r, j), wu = word(wraw, j);
        o[2 * j] += f[k] * (__uint_as_float(u << 16) * __uint_as_float(wu << 16));
        o[2 * j + 1] += f[k] * (__uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u));
      }
    }
    uint32_t pk[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j]));
      const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j + 1]));
      pk[j] = lo | (hi << 16);
    }
    reinterpret_cast<uint4*>(pout + static_cast<size_t>(tk) * K)[col] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  g2p_sync(grp, KS);  // s_rs is rewritten by the next token
}
"""

HOST = r"""
}  // namespace m11

template <typename... Exp, typename... Act>
static void g2p_launch(void (*kernel)(Exp...), int grid, int pdl, Act... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, "g2p launch failed: ", cudaGetErrorString(err));
}

#define G2P_CASE(TT, NT, D, KS, P)                                                                          \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS) && pdl == (P)) {                               \
    g2p_launch(m11::gemm_post_kernel<TT, NT, D, KS, false, (P) != 0>, static_cast<int>(grid), (P),           \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total), 1,             \
               static_cast<int>(out_cols), 0, reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()),           \
               static_cast<const float*>(tw.data_ptr<float>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
               cnt.data_ptr<int>(), static_cast<float>(eps));                                                 \
    return out;                                                                                              \
  }

#define G2P_ATTR(TT, NT, D, KS, P)                                                                           \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS) && pdl == (P)) {                                \
    cudaFuncAttributes attr;                                                                                 \
    cudaFuncGetAttributes(&attr, m11::gemm_post_kernel<TT, NT, D, KS, false, (P) != 0>);                     \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes),    \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                      \
  }

// M11's GEMM2 (y = act @ w2^T for every local pair, y bf16 [n_pairs, 4096]) + the post-norm weighted sum into out
// [n_pairs / 8, 4096]; cnt int32 [32 + tokens] must be zero on entry and is zero again on exit.
torch::Tensor g2p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor y, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t out_cols,
                       int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid, int64_t pdl, torch::Tensor nw,
                       torch::Tensor tw, torch::Tensor out, torch::Tensor cnt, double eps) {
  TORCH_CHECK(out_cols == m11::K && nw.numel() == m11::K && out.size(1) == m11::K && out.size(0) * 8 == n_pairs,
              "g2p_gemm: shapes");
  TORCH_CHECK(cnt.numel() >= m11::LOCAL + out.size(0) && out.size(0) <= m11::MAXP, "g2p_gemm: counters");
  @@CASES@@
  TORCH_CHECK(false, "g2p_gemm: no instance for tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks, " pdl=", pdl);
  return out;
}

torch::Tensor g2p_attrs(int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t pdl) {
  @@ATTRS@@
  TORCH_CHECK(false, "g2p_attrs: no instance");
  return torch::Tensor();
}
"""

CPP = """
torch::Tensor g2p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor y, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t out_cols,
                       int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid, int64_t pdl, torch::Tensor nw,
                       torch::Tensor tw, torch::Tensor out, torch::Tensor cnt, double eps);
torch::Tensor g2p_attrs(int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t pdl);
"""

PY = '''

# G2P_INSTANCES: (tt, nt, depth, ks, pdl) compiled.
G2P_INSTANCES = @@INSTANCES@@


def g2p_source(instances=None):
    """The CUDA text with one case per instance; instances default to G2P_INSTANCES read at call time (so a submission
    may reassign the module-level tuple to build fewer instances)."""
    instances = G2P_INSTANCES if instances is None else instances
    args = [f"{tt}, {nt}, {d}, {ks}, {p}" for tt, nt, d, ks, p in instances]
    return (_G2P_CUDA.replace("@@CASES@@", "\\n  ".join(f"G2P_CASE({a})" for a in args))
            .replace("@@ATTRS@@", "\\n  ".join(f"G2P_ATTR({a})" for a in args))
            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))


def g2p_build():
    """Compile the fused kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="g2p_ext", cpp_sources=_G2P_CPP, cuda_sources=g2p_source(), functions=["g2p_gemm", "g2p_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="g2p_"), verbose=False)


def g2p_setup(state):
    """After m11_setup: the extension and the self-resetting counters (32 experts + up to M11_MAX_TOKENS tokens)."""
    state["g2p_ext"] = g2p_build()
    state["g2p_cnt"] = torch.zeros(32 + M11_MAX_TOKENS, dtype=torch.int32, device=state["norm_w"].device)


def m11_gemm2_post(act, ids, tw, state, tokens, s2):
    """m11_post_norm(m11_gemm2(act, ids, state, s2), ids, tw, state, tokens) in one kernel: act bf16 [pairs, I],
    ids int32 [pairs] (global expert ids), tw float32 [tokens, 8] contiguous -> bf16 [tokens, 4096]. s2 = cfg["s2"]
    (tt, nt, d, ks, c, optional pdl) of an instance in G2P_INSTANCES."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    return state["g2p_ext"].g2p_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"],
                                     bank["rt_w2"], bank["hidden"], s2["tt"], s2["nt"], s2["d"], s2["ks"],
                                     s2["c"] * state["sms"], int(s2.get("pdl", 0)), state["norm_w"], tw, out,
                                     state["g2p_cnt"], POST_NORM_EPS)


def g2p_ok(s2):
    """True when GEMM2 stage s2 has a fused instance."""
    return (s2.get("kind") != "m12" and (s2["tt"], s2["nt"], s2["d"], s2["ks"], int(s2.get("pdl", 0)))
            in G2P_INSTANCES)


def g2p_m11_run(x, topk_ids, topk_weights, state, cfg):
    """m11_run, with cfg["g2p"] taking the fused GEMM2 + post-norm (cfg["post"] is then unused)."""
    ids = topk_ids.reshape(-1).contiguous()
    act = m11_gemm1(x, ids, state, cfg["s1"])
    if cfg.get("g2p"):
        return m11_gemm2_post(act, ids, topk_weights.contiguous(), state, x.shape[0], cfg["s2"])
    y = m11_gemm2(act, ids, state, cfg["s2"])
    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0], cfg.get("post"))


def g2p_install(state, picks):
    """state["m11_configs"] with cfg["g2p"] = picks[below] for the classes in picks (new dicts; classes whose s2 has
    no fused instance stay off); the configs before the first install are kept in state["g2p_base_configs"]."""
    base = state.setdefault("g2p_base_configs", state["m11_configs"])
    state["m11_configs"] = tuple(
        (below, dict(cfg, g2p=bool(picks.get(below)) and g2p_ok(cfg["s2"])) if cfg is not None else cfg)
        for below, cfg in base)
    state["g2p_picks"] = {b: bool(v) for b, v in picks.items()}


def g2p_warm(state):
    """Run every fused instance the current configs can use once (8 tokens), before any graph capture."""
    for _below, cfg in state["m11_configs"]:
        if cfg is not None and g2p_ok(cfg["s2"]):
            dev = state["norm_w"].device
            x = torch.zeros((8, state["m11"]["hidden"]), dtype=torch.bfloat16, device=dev)
            ids = (torch.arange(64, dtype=torch.int32, device=dev) % state["m11"]["count"]).view(8, 8)
            tw = torch.full((8, 8), 0.125, dtype=torch.float32, device=dev)
            g2p_m11_run(x, ids + state["m11"]["start"], tw, state, dict(cfg, g2p=True))
    torch.cuda.synchronize()


def g2p_check(state, token_counts=(1, 4, 8, 16, 32), seed=5):
    """Worst relative error of the fused path vs GEMM2 + post-norm (the same cfg) over random routings with
    ~1/4 local routes, tokens without local routes included; and whether the counters are zero afterwards."""
    dev = state["norm_w"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    worst = 0.0
    for tokens in token_counts:
        cfg = None
        for below, c in state["m11_configs"]:
            if tokens < below:
                cfg = c
                break
        if cfg is None or not g2p_ok(cfg["s2"]):
            continue
        x = (torch.randn((tokens, state["m11"]["hidden"]), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        scores = torch.rand((tokens, 256), generator=gen, device=dev)
        scores[:, :32] += 0.3
        ids = scores.topk(8, dim=-1).indices.to(torch.int32) + state["m11"]["start"]
        ids = torch.where(ids >= 256, ids - 256, ids)
        tw = torch.softmax(torch.randn((tokens, 8), generator=gen, device=dev), -1)
        ref = g2p_m11_run(x, ids, tw, state, dict(cfg, g2p=False)).float()
        out = g2p_m11_run(x, ids, tw, state, dict(cfg, g2p=True)).float()
        err = float((out - ref).norm() / ref.norm().clamp(min=1e-30)) if bool(torch.isfinite(out).all()) else float("inf")
        worst = max(worst, err)
    torch.cuda.synchronize()
    return worst, int(state["g2p_cnt"].abs().sum().item()) == 0
'''


def device_part(src, has_pf=True):
    """M11's helpers and the edited kernel (from the start of src's CUDA through the kernel; namespace m11 open)."""
    i0 = src.index("namespace m11 {")
    i1 = src.index(HELPERS_END)
    i2 = src.index("\n}\n", src.index("__global__ void __launch_bounds__(128) gemm_kernel(", i1)) + 2  # its closing brace
    head = src[:i0] + src[i0:i1]
    kernel = src[i1:i2]
    kernel = edit(kernel, "__global__ void __launch_bounds__(128) gemm_kernel(",
                  "__global__ void __launch_bounds__(128) gemm_post_kernel(")
    if has_pf:
        kernel = edit(kernel, SIG_OLD, SIG_NEW)
    else:
        kernel = edit(kernel, SIG_OLD.replace(", int pf) {", ") {"), SIG_NEW.replace(", int pf,\n", ",\n"))
    kernel = edit(kernel, WAIT_OLD, WAIT_NEW)
    assert kernel.endswith(TAIL_OLD), kernel[-200:]
    kernel = kernel[: -len(TAIL_OLD)] + TAIL_NEW
    kernel = edit(kernel, "G2P_TASK_DONE", TASK_DONE)
    # Self-check: undoing the edits gives the source's gemm_kernel byte for byte (the GEMM itself is M11's).
    back = kernel.replace(TASK_DONE, "G2P_TASK_DONE").replace(TAIL_NEW, TAIL_OLD).replace(WAIT_NEW, WAIT_OLD)
    back = back.replace(SIG_NEW if has_pf else SIG_NEW.replace(", int pf,\n", ",\n"),
                        SIG_OLD if has_pf else SIG_OLD.replace(", int pf) {", ") {"))
    back = back.replace("gemm_post_kernel(", "gemm_kernel(")
    assert back == src[i1:i2], "make_g2p: the fused kernel's GEMM part differs from the source's gemm_kernel"
    # G2P's device helpers go before the kernel (they use M11's word, ld_cg4, group_sync)
    return head + DEVICE + "\n" + kernel + "\n"


def build():
    cuda = device_part(m11_cuda()) + HOST
    cuda = cuda.replace("constexpr int MAXP = @@MAXP@@;", "constexpr int MAXP = @@MAXP@@;", 1)
    text = ('"""G2P: M11 GEMM2 + post-expert norm weighted sum in one kernel (TRACKER shared item 5; built by\n'
            'shared/glue/make_g2p.py from decode_layer/parts/m11_pdl.py, whose GEMM it copies byte for byte). Do not\n'
            'edit by hand. Needs the submission namespace: torch, load_inline, tempfile, M11_MAX_TOKENS, POST_NORM_EPS."""\n\n'
            f"_G2P_CUDA = r'''{cuda}'''\n\n_G2P_CPP = '''{CPP}'''\n"
            + PY.replace("@@INSTANCES@@", repr(G2P_INSTANCES)))
    OUT.write_text(text)
    return text


MOE_HOST = r"""
}  // namespace m11f

// G2P for MoE: M11's GEMM2 (y_scratch = act @ w2^T for every local pair) + the post-norm weighted sum into out
// [tokens, hidden], one kernel, launched with MoE's m13_launch (128 threads, no dynamic shared memory; PDL when pdl).
// ctr: int32 [32 + tokens] (allocate 32 + M11_MAX_TOKENS once), zero on entry and zero again on exit.
torch::Tensor m11f_gemm2_post(torch::Tensor w2, torch::Tensor w2_s, torch::Tensor c2, torch::Tensor ids,
                              torch::Tensor act, torch::Tensor y_scratch, torch::Tensor topk_w, torch::Tensor norm_w,
                              torch::Tensor out, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                              int64_t rt_w2, int64_t hidden, int64_t tokens, double eps, int64_t tt, int64_t nt,
                              int64_t depth, int64_t ks, int64_t grid, int64_t pdl) {
  TORCH_CHECK(hidden == m11f::K && norm_w.numel() == m11f::K && out.size(1) == m11f::K && tokens * 8 == n_pairs &&
              out.size(0) == tokens && tokens <= m11f::MAXP && ctr.numel() >= m11f::LOCAL + tokens,
              "m11f_gemm2_post: shapes / counters");
  @@CASES@@
  TORCH_CHECK(false, "m11f_gemm2_post: no instance for tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks, " pdl=", pdl);
  return out;
}
"""

MOE_CASE = r"""#define M11F_CASE(TT, NT, D, KS, P)                                                                         \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS) && pdl == (P)) {                                \
    m13_launch(m11f::gemm_post_kernel<TT, NT, D, KS, false, (P) != 0>, static_cast<int>(grid), 128, 0, (P),   \
               reinterpret_cast<const uint4*>(w2.data_ptr()), reinterpret_cast<const uint16_t*>(w2_s.data_ptr()), \
               static_cast<const float*>(c2.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()),  \
               reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(y_scratch.data_ptr()), static_cast<int>(n_pairs),           \
               static_cast<int>(local_start), static_cast<int>(rt_w2), 1, static_cast<int>(hidden),         \
               reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),                                    \
               static_cast<const float*>(topk_w.data_ptr<float>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
               ctr.data_ptr<int>(), static_cast<float>(eps));                                                  \
    return out;                                                                                               \
  }
"""

MOE_CPP = """
torch::Tensor m11f_gemm2_post(torch::Tensor w2, torch::Tensor w2_s, torch::Tensor c2, torch::Tensor ids,
                              torch::Tensor act, torch::Tensor y_scratch, torch::Tensor topk_w, torch::Tensor norm_w,
                              torch::Tensor out, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                              int64_t rt_w2, int64_t hidden, int64_t tokens, double eps, int64_t tt, int64_t nt,
                              int64_t depth, int64_t ks, int64_t grid, int64_t pdl);
"""

# For check_g2p_syntax.py only: MoE's m13_launch is defined in MoE's module; a declaration stands in for it.
MOE_TEST_PRELUDE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
template <typename... Exp, typename... Act>
static void m13_launch(void (*kernel)(Exp...), int grid, int threads, size_t smem, int pdl, Act... args) {}
"""

MOE_PY = '''

# MoE variant of G2P (TRACKER shared item 5). Splice _M11F_CUDA (device code in namespace m11f + the host function
# m11f_gemm2_post, which calls MoE's m13_launch) into MoE's CUDA module after m13_launch's definition, add _M11F_CPP's
# prototype to its cpp sources and "m11f_gemm2_post" to its functions.
M11F_INSTANCES = @@INSTANCES@@


def g2p_moe_source(instances=None):
    """The device + host text with one launch case per (tt, nt, depth, ks, pdl) instance; instances default to
    M11F_INSTANCES read at call time."""
    instances = M11F_INSTANCES if instances is None else instances
    cases = "\\n  ".join(f"M11F_CASE({tt}, {nt}, {d}, {ks}, {p})" for tt, nt, d, ks, p in instances)
    return _M11F_CUDA.replace("@@CASES@@", cases).replace("@@MAXP@@", str(M11_MAX_TOKENS))


def m11f_counters(device):
    """The counters m11f_gemm2_post needs: int32 [32 + M11_MAX_TOKENS], zero; allocate once in prepare."""
    return torch.zeros(32 + M11_MAX_TOKENS, dtype=torch.int32, device=device)


def m11f_gemm2_post_py(act, ids, topk_w, state, tokens, s2, ext, ctr, sms):
    """act bf16 [pairs, I], ids int32 [pairs] (global ids), topk_w float32 [tokens, 8] contiguous -> bf16 [tokens, H]
    = M13's post-norm of M11's GEMM2; s2 = dict(tt, nt, d, ks, c, optional pdl) of an instance in M11F_INSTANCES."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    return ext.m11f_gemm2_post(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, topk_w, state["norm_w"], out, ctr,
                               n_pairs, bank["start"], bank["rt_w2"], bank["hidden"], tokens, POST_NORM_EPS,
                               s2["tt"], s2["nt"], s2["d"], s2["ks"], s2["c"] * sms, int(s2.get("pdl", 0)))
'''


def build_moe():
    src = m11_cuda(MOE_SRC)
    dev = device_part(src, has_pf=False)
    dev = dev[dev.index("namespace m11 {"):]  # no includes: spliced into MoE's module
    dev = dev.replace("namespace m11 {", "namespace m11f {", 1)
    cuda = dev + MOE_CASE + MOE_HOST
    text = ('"""G2P for MoE (namespace m11f): M11 GEMM2 + post-expert norm weighted sum in one kernel; built by\n'
            'shared/glue/make_g2p.py from moe/' + MOE_SRC.name + '\'s _M11_CUDA (its gemm_kernel copied byte for byte).\n'
            'Do not edit by hand. Needs MoE\'s namespace: torch, M11_MAX_TOKENS, POST_NORM_EPS, and m13_launch in CUDA."""\n\n'
            f"_M11F_CUDA = r'''{cuda}'''\n\n_M11F_CPP = '''{MOE_CPP}'''\n\n"
            f"G2P_MOE_TEST_PRELUDE = r'''{MOE_TEST_PRELUDE}'''\n"
            + MOE_PY.replace("@@INSTANCES@@", repr(MOE_INSTANCES)))
    MOE_OUT.write_text(text)
    return text


if __name__ == "__main__":
    t = build()
    print(f"wrote {OUT.relative_to(ROOT)} ({len(t)} chars)")
    t = build_moe()
    print(f"wrote {MOE_OUT.relative_to(ROOT)} ({len(t)} chars)")
