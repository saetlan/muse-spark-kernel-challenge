"""Derive parts/m11_pdl.py from parts/m11_pf.py (v7 / v9's M11): the same kernels plus the MoE session's E15 / E16
PDL path at the smallest batches (moe/scored/m11_e16.py and submission_m13t_e16.py, judged 1.455x / 1.459x on
muse_moe_w4a16; bs1 38.8 -> 35.8 us, MF16): every M11 kernel signals `griddepcontrol.launch_dependents` first
thing (a no-op unless the next kernel is PDL-launched); GEMM2 gets a PDL instance (`PDL = true`) that scans the
routing, then waits (`griddepcontrol.wait`) before its activation loads, which go through coherent loads that
depend on the wait's zero token (MoE's MF8 / E15 fix: no `__ldg` behind the wait); and M13's CUDA post-norm
(SGLang's fused post-expert norm + weighted sum, as the Triton `_m11_post_norm_kernel`), PDL-launched when asked,
waiting before it reads y. Configs opt in per stage: s2 {"pdl": 1} launches GEMM2 through `m11p_gemm`; a
config's "post" {"pdl": 0 / 1} runs the CUDA post-norm (absent: the Triton kernel as before). Every change is an
exact, counted replacement, copied from MoE's E16 sources (helpers, kernel lines, launch helper, post-norm
kernel), so the diff is the whole story.

Run: .venv/bin/python decode_layer/local/make_m11pdl.py
"""

from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]

PDL_HELPERS = r'''// PDL (MoE E15, as M13's): the next grid may launch once every CTA of this one has signalled; a PDL-launched grid
// waits before reading the previous grid's output, through loads that depend on the wait's (zero) token.
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t pdl_wait_token() {
  uint32_t z;
  asm volatile("griddepcontrol.wait;\n\tmov.u32 %0, 0;" : "=r"(z) : : "memory");
  return z;
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// PDL (GEMM2 only): launched early with programmatic stream serialization; scans the routing, then every CTA waits
// and reads the activations with coherent token-ordered loads (__ldg's non-coherent path isn't safe behind the wait).
'''

POST_NORM = r'''
// MoE M13's post-norm (E15 / E16): out[t] = sum over local slots k of tw[t, k] * rmsnorm(y[t, k] * nw), as SGLang's
// fused kernel; y and nw bf16, H = 4096 = 512 threads x 8. With pdl it reads the routing first, then waits for GEMM2.
__global__ void __launch_bounds__(512) post_norm_kernel(
    const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ tw,
    const int* __restrict__ ids, __nv_bfloat16* __restrict__ out, int local_start, float eps, int pdl) {
  __shared__ float s_part[16][TOP_K];
  __shared__ float s_rs[TOP_K];
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + tid);
  float wv[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t u = word(wraw, j);
    wv[2 * j] = __uint_as_float(u << 16);
    wv[2 * j + 1] = __uint_as_float(u & 0xFFFF0000u);
  }
  bool loc[TOP_K];
  float gk[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    const int e = ids[tok * TOP_K + k] - local_start;
    loc[k] = e >= 0 && e < LOCAL;
    gk[k] = tw[tok * TOP_K + k];
  }
  const uint32_t ptok = pdl ? pdl_wait_token() : 0u;
  float v[TOP_K][8];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    ss[k] = 0.0f;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (loc[k]) r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tok * TOP_K + k) * K) + tid + ptok);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t u = word(r, j);
      v[k][2 * j] = __uint_as_float(u << 16) * wv[2 * j];
      v[k][2 * j + 1] = __uint_as_float(u & 0xFFFF0000u) * wv[2 * j + 1];
      ss[k] += v[k][2 * j] * v[k][2 * j] + v[k][2 * j + 1] * v[k][2 * j + 1];
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) s_part[warp][k] = ss[k];
  }
  __syncthreads();
  if (tid < TOP_K) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) s += s_part[i][tid];
    s_rs[tid] = rsqrtf(s / static_cast<float>(K) + eps);
  }
  __syncthreads();
  float o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = 0.0f;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    if (loc[k]) {
      const float f = gk[k] * s_rs[k];
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] += f * v[k][j];
    }
  }
  uint4 res;
  uint32_t pk[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j]));
    const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j + 1]));
    pk[j] = lo | (hi << 16);
  }
  res = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  reinterpret_cast<uint4*>(out + static_cast<size_t>(tok) * K)[tid] = res;
}

}  // namespace m11
'''

HOST = r'''// Launch with programmatic stream serialization when pdl (MoE's m13_launch): the kernel may then start before the
// previous kernel in the stream completes; it calls griddepcontrol.wait before reading that kernel's outputs.
template <typename... Exp, typename... Act>
static void m11_launch(void (*kernel)(Exp...), int grid, int threads, size_t smem, int pdl, Act... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, "m11 launch failed: ", cudaGetErrorString(err));
}

#define M11P_CASE(TT, NT, D, KS)                                                                            \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS)) {                                            \
    m11_launch(m11::gemm_kernel<TT, NT, D, KS, false, true>, static_cast<int>(grid), 128, 0, 1,             \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                \
               static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),     \
               static_cast<int>(out_cols), 0);                                                              \
    return out;                                                                                            \
  }

// M11's GEMM2 launched with PDL (MoE E15): m11_gemm's arguments; it scans the routing while GEMM1 finishes.
torch::Tensor m11p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                        int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                        int64_t ks, int64_t grid) {
  TORCH_CHECK(gated == 0, "m11p_gemm: GEMM2 only");
  @@M11P_CASES@@
  TORCH_CHECK(false, "m11p_gemm: no instance for tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return out;
}

torch::Tensor m11_post_norm_cuda(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids,
                                 torch::Tensor out, int64_t tokens, int64_t local_start, double eps, int64_t pdl) {
  TORCH_CHECK(y.size(1) == m11::K && nw.numel() == m11::K && out.size(1) == m11::K, "m11_post_norm_cuda: H != 4096");
  m11_launch(m11::post_norm_kernel, static_cast<int>(tokens), 512, 0, static_cast<int>(pdl),
             reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()),
             static_cast<const float*>(tw.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(local_start), static_cast<float>(eps),
             static_cast<int>(pdl));
  return out;
}

torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks) {
'''

CPP_EXTRA = '''torch::Tensor m11p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                        int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                        int64_t ks, int64_t grid);
torch::Tensor m11_post_norm_cuda(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids,
                                 torch::Tensor out, int64_t tokens, int64_t local_start, double eps, int64_t pdl);
'''

GEMM2_PY_OLD = '''    if s2.get("kind") == "m12":
        state["m11_ext"].m12_gemm(*args, s2["ks"], s2["ch"], s2["r"], s2["c"] * state["sms"])
    else:
        state["m11_ext"].m11_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"], s2.get("pf", 0))
    return y
'''
GEMM2_PY_NEW = '''    if s2.get("kind") == "m12":
        state["m11_ext"].m12_gemm(*args, s2["ks"], s2["ch"], s2["r"], s2["c"] * state["sms"])
    elif s2.get("pdl"):  # MoE E15: GEMM2 PDL-launched (an M11P instance), scanning the routing while GEMM1 finishes
        state["m11_ext"].m11p_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"])
    else:
        state["m11_ext"].m11_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"], s2.get("pf", 0))
    return y
'''

POST_PY_OLD = '''def m11_post_norm(y, ids, topk_weights, state, tokens):
    bank = state["m11"]
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=y.device)
'''
POST_PY_NEW = '''def m11_post_norm(y, ids, topk_weights, state, tokens, post=None):
    bank = state["m11"]
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=y.device)
    if post is not None:  # MoE M13's CUDA post-norm, PDL-launched when post["pdl"] (it waits before reading y)
        return state["m11_ext"].m11_post_norm_cuda(y, state["norm_w"], topk_weights, ids, out, tokens, bank["start"],
                                                   POST_NORM_EPS, int(post.get("pdl", 0)))
'''

GEMM_HEAD = ("    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int pf) {\n"
             "  constexpr int GROUPS = 4 / KS;  // tasks a CTA works on at once\n"
             "  constexpr int KGW = KG / KS;    // K chunks per warp\n"
             "  static_assert(KGW % D == 0, \"depth must divide each warp's K chunks\");\n"
             "  static_assert(!GATED || TT == 2, \"a gated task is one gate tile and its up tile\");\n"
             "  __shared__ int s_cnt[LOCAL];\n"
             "  __shared__ int s_active[LOCAL];\n"
             "  __shared__ int s_nactive;\n"
             "  __shared__ short s_list[LOCAL][MAXP];\n"
             "  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];\n"
             "  const int tid = threadIdx.x;\n")
SCAN_END = ("    if (tid == 0) s_nactive = __popc(m);\n  }\n  __syncthreads();\n\n"
            "  const int lane = tid & 31, g = lane >> 2, t = lane & 3;\n"
            "  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;\n"
            "  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;\n"
            "  const int n_tasks = s_nactive * per_expert;\n"
            "  for (int task = blockIdx.x * GROUPS + grp; task < n_tasks; task += gridDim.x * GROUPS) {\n"
            "    if (pf & 15) {")

REPLACEMENTS = (
    # helpers, then the PDL template flag
    ("template <int TT, int NT, int D, int KS, bool GATED>\n__global__ void __launch_bounds__(128) gemm_kernel(\n",
     PDL_HELPERS + "template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>\n"
     "__global__ void __launch_bounds__(128) gemm_kernel(\n"),
    # GEMM1 can't be PDL; signal dependents first thing (anchored on gemm_kernel's signature: tma_kernel shares the body)
    (GEMM_HEAD, GEMM_HEAD.replace("int pf) {\n", "int pf) {\n  static_assert(!(GATED && PDL), \"GEMM1 is the first kernel of a call\");\n")
     + "  if (tid == 0) pdl_launch();  // a no-op unless the next kernel is PDL-launched\n"),
    # wait after the routing scan, before any activation load (anchored on gemm_kernel's task loop + L8 prefetch)
    (SCAN_END, SCAN_END.replace("  __syncthreads();\n\n",
                                "  __syncthreads();\n"
                                "  uint32_t tok = 0u;  // zero; with PDL it carries the dependency on the wait into the activation loads\n"
                                "  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)\n\n")),
    # the activation loads: coherent, token-ordered under PDL
    ("            ra[s][n][0] = __ldg(ap[n] + kg * 8);\n            ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);\n",
     "            if constexpr (PDL) {\n"
     "              ra[s][n][0] = ld_cg4(ap[n] + kg * 8 + tok);\n"
     "              ra[s][n][1] = ld_cg4(ap[n] + kg * 8 + 1 + tok);\n"
     "            } else {\n"
     "              ra[s][n][0] = __ldg(ap[n] + kg * 8);\n"
     "              ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);\n"
     "            }\n"),
    # the post-norm kernel at the end of the namespace
    ("\n}  // namespace m11\n", POST_NORM),
    # host: launch helper, M11P cases, m11p_gemm, the post-norm launcher (before m11_attrs's definition)
    ("torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks) {\n", HOST),
    ("torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks);\n",
     "torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks);\n" + CPP_EXTRA),
    # the M11P instances (GEMM2 configs used with PDL) and their dispatch cases
    ("def m11_source(instances=M11_INSTANCES, m12_instances=M12_INSTANCES):\n",
     "# GEMM2 instances (tt, nt, depth, ks) compiled with PDL = true: the configs that opt in (s2 {\"pdl\": 1}).\n"
     "M11P_INSTANCES = ((1, 1, 4, 4), (2, 1, 4, 4))\n\n\n"
     "def m11_source(instances=M11_INSTANCES, m12_instances=M12_INSTANCES, m11p_instances=None):\n"),
    ('            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))\n',
     '            .replace("@@M11P_CASES@@", "\\n  ".join(f"M11P_CASE({tt}, {nt}, {d}, {ks})" for tt, nt, d, ks in\n'
     '                                                   (M11P_INSTANCES if m11p_instances is None else m11p_instances)))\n'
     '            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))\n'),
    ('        name="m11pf_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(), functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup"],',
     '        name="m11pdl_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(),\n'
     '        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m11p_gemm", "m11_post_norm_cuda"],'),
    # Python: GEMM2 through m11p_gemm when s2["pdl"]; the post-norm through the CUDA kernel when cfg has "post"
    (GEMM2_PY_OLD, GEMM2_PY_NEW),
    (POST_PY_OLD, POST_PY_NEW),
    ('    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0])\n',
     '    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0], cfg.get("post"))\n'),
)


def build(src=DECODE / "parts" / "m11_pf.py", out=DECODE / "parts" / "m11_pdl.py"):
    text = src.read_text()
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(f"expected exactly one {old.strip()[:70]!r} in {src.name}, found {text.count(old)}")
        text = text.replace(old, new)
    lead = ("# parts/m11_pdl.py, generated by local/make_m11pdl.py from parts/m11_pf.py: + the MoE session's E15 / E16 PDL\n"
            "# path (every kernel signals dependents; a PDL GEMM2 instance; M13's CUDA post-norm, PDL-launched on request).\n")
    out.write_text(lead + text)
    print(f"wrote {out} ({len(text)} chars, {len(REPLACEMENTS)} replacements)")


if __name__ == "__main__":
    build()
