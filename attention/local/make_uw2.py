"""Generates attention/local/cuda_uw2.py: UW2, one launch per sliding verify layer (q_len 8): UV4's in-kernel prologue and
cluster merge (make_uv4.py, verified) around W2c's data path and consumers (make_w2.py: W1's copier / transposers, Q and P
in shared memory, SS wgmma pairs in FA3's order, optional ping-pong). L29 / AF55: the sliding shapes (weight 46 per load)
spend 5-7 us in the separate merge and ~2 us in the prologue kernel, and UV4 lost only on V4's slow consumers (AF37 / AF41).
Composed by anchors that must each occur exactly once (asserted):
- helpers: cuda_w2.py's (W1's + W2's), UV4's cluster helpers (mapa, st.shared::cluster, cluster rank / sync, BF16 out);
- kernel head (written here): one cluster of CS CTAs per (request, KV-head pair), slice = split blockIdx.x of the window,
  barriers land / full / empty (W2's) + fresh_bar (1 arrival: the block's new K / V rows are in shared memory);
- warp 8 = W2's copier, byte for byte; it reads the cache as it stands, so the block's own rows may be stale;
- warps 9-10 = W2's transposers + the fresh-row patch: in a stage whose clamped keys reach n - L, they wait fresh_bar once,
  then lane r = 32 * (warp - 9) + lane (the row this warp transposes) overwrites row r of the K tile (both heads, 8 x 16 B
  at (r >> 3) * SBO_K + (r & 7) * 16 + head * KC_HEAD + chunk * LBO_K) and of V staging from new_k / new_v when its
  clamped key min(s0 + r, end - 1) >= n - L (AF36: clamped copies included);
- warps 0-7 = UV4's prologue sections 1-4 (gate rows by cp.async, query fold, RoPE table, the block's keys / values into the
  cache and new_k / new_v), the query fold now storing E4M3 bytes straight into W2's Q tile (row 16w + g + 8r, dims
  16*t4 .. +15: one 16 B store at tile_off(row, 16*t4)); named barrier 5 (256 consumer threads) replaces UV4's id 1 (W2 uses
  1-4); after the prologue thread 0 arrives on fresh_bar; then W2c's consumer loop byte for byte (its Q-tile build and
  partial stores removed); then the push of W2's partials (row R = 16 * token + 8h + g, dims 8j + 2*t4 + c, m, l) into
  the owner's receive buffer (UV4's layout: R -> owner R / RPC, slot rank * RPC + R % RPC, LBP 68 floats);
- every warp: barrier.cluster.wait (pairs with the entry arrive) before the first remote store / after its loop, then
  cluster_sync; warps 0-7 run UV4's finalize (merge of the CS partials, v_scale / l, RMSNorm x sigmoid(gate), BF16 out).
Shared memory: W2's slots + page-entry ring + Q / P tiles, then the receive buffer (128 x 68 floats); static: barriers,
new_k / new_v, gate rows, RoPE tables.
Host: uw2(qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, page_table, seqlens, out, batch, hq, hkv, max_pages,
page, window, rope, qk_factor, sm_scale, stages, pp, cs), uw2_setup(), uw2_attrs(stages, pp, cs). Variants (stages, pp, cs):
(4,1,4) (4,1,8) (4,1,16) (6,0,16). Sliding layers only (one cluster covers the window).
Run: .venv/bin/python attention/local/make_uw2.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_uw2.py")


def load(fname, var):
    ns = {"__name__": "make_uw2_load", "__file__": os.path.join(HERE, fname)}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def between(src, a, b, what, include_b=False):
    assert src.count(a) == 1, f"{what}: start anchor x{src.count(a)}"
    i = src.index(a)
    j = src.index(b, i)
    return src[i:j + (len(b) if include_b else 0)]


def sub1(src, old, new, what):
    n = src.count(old)
    assert n == 1, f"{what}: anchor x{n}: {old[:90]!r}"
    return src.replace(old, new)


W2 = load("cuda_w2.py", "CUDA_W2_SRC")
UV = load("make_uv4.py", "KERNEL")
HELP = W2[:W2.index("template <int STAGES, int PP, int SBOV, int WF, int ABL = 0>\n__global__")]
HELP = HELP.replace("namespace okc_w2 {", "namespace okc_uw2 {")
assert "okc_w2" not in HELP
UVHELP = between(UV, "constexpr int LBP = 68;", "template <int STAGES, int MT, int CS>", "uv4 helpers")
COPIER = between(W2, "  if (warp == 8) {\n", "  if (warp >= 9) {\n", "copier")
COPIER = sub1(COPIER, "    asm volatile(\"cp.async.wait_all;\");\n    return;\n  }\n",
              "    asm volatile(\"cp.async.wait_all;\");\n    asm volatile(\"barrier.cluster.wait.aligned;\" ::: \"memory\");\n  }\n",
              "copier end")
COPIER = sub1(COPIER, "  if (warp == 8) {\n", "  if (warp == 8) {   // ---- W2's copier (byte for byte but for the end)\n", "copier start")
TRANS = between(W2, "  if (warp >= 9) {\n", "  // ---- W2 consumer warpgroup h", "transposers")
TRANS = sub1(TRANS, "  if (warp >= 9) {\n", "  else if (warp >= 9) {   // ---- W2's transposers + the fresh-row patch\n    bool fresh_seen = false;\n", "trans start")
TRANS = sub1(TRANS, "      bar_wait(&land_bar[st], static_cast<uint32_t>((j / STAGES) & 1));\n",
             "      bar_wait(&land_bar[st], static_cast<uint32_t>((j / STAGES) & 1));\n"
             "      {   // the block's own rows (keys n-8 .. n-1, clamped copies included) from new_k / new_v (AF36)\n"
             "        const int s0 = start + j * SKEYS;\n"
             "        if (min(s0 + SKEYS - 1, end - 1) >= n - L) {\n"
             "          if (!fresh_seen) {\n"
             "            bar_wait(&fresh_bar, 0u);\n"
             "            fresh_seen = true;\n"
             "          }\n"
             "          const int r = 32 * (warp - 9) + lane, key = min(s0 + r, end - 1);\n"
             "          if (key >= n - L) {\n"
             "            const int nl = key - (n - L);\n"
             "            uint8_t* slot = sm + st * SLOTB;\n"
             "#pragma unroll\n"
             "            for (int c = 0; c < 8; ++c) {   // K: head c >> 2, dims 16 * (c & 3) ..; V: bytes 16c ..\n"
             "              *reinterpret_cast<uint4*>(slot + (c >> 2) * KC_HEAD + (r >> 3) * SBO_K + (c & 3) * LBO_K + (r & 7) * 16) =\n"
             "                  *reinterpret_cast<const uint4*>(new_k + nl * 128 + 16 * c);\n"
             "              *reinterpret_cast<uint4*>(slot + OFF_VS + r * VS_PITCH + 16 * c) =\n"
             "                  *reinterpret_cast<const uint4*>(new_v + nl * 128 + 16 * c);\n"
             "            }\n"
             "          }\n"
             "          __syncwarp();\n"
             "        }\n"
             "      }\n", "fresh patch")
TRANS = sub1(TRANS, "      if (lane == 0) bar_arrive(&full_bar[st]);\n    }\n    return;\n  }\n",
             "      if (lane == 0) bar_arrive(&full_bar[st]);\n    }\n    asm volatile(\"barrier.cluster.wait.aligned;\" ::: \"memory\");\n  }\n",
             "trans end")
CONS = between(W2, "  // ---- W2 consumer warpgroup h", "  // partials: row (token 2w + half", "consumer")
CONS = sub1(CONS, "  const int h = warp >> 2, w = warp & 3;\n  uint8_t* qt = q_tiles + h * 4096;\n  uint8_t* ptl = p_tiles + h * 4096;\n", "", "cons head")
qa = CONS.index("  {\n    const uint16_t* qA = q +")
qb = CONS.index("  proxy_fence();                                      // generic Q-tile stores -> wgmma (async proxy) reads")
CONS = CONS[:qa] + CONS[qb:]
assert "qA" not in CONS and "attend_w2" not in CONS
PRO = between(UV, "    // ---- 1. loads that do not depend on seqlens", "    // ---- 5. V4's consumer loop", "uv4 prologue")
PRO = PRO.replace('asm volatile("bar.sync 1, %0;" ::"n"(WC * 32) : "memory");', 'asm volatile("bar.sync 5, %0;" ::"n"(WC * 32) : "memory");')
assert PRO.count('bar.sync 5, %0;') == 2 and "bar.sync 1," not in PRO
qa0 = PRO.index("    uint32_t qa[MT][2][4];\n")
qa1 = PRO.index("    // ---- 4. the block's keys/values held by this CTA")
FOLD = PRO[qa0:qa1]
FOLD = sub1(FOLD, "    uint32_t qa[MT][2][4];\n", "", "qa decl")
FOLD = sub1(FOLD, """#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int o = 8 * s;
        qa[mi][s][0] = e4m3x4(x[0][o + 0], x[0][o + 1], x[0][o + 2], x[0][o + 3]);
        qa[mi][s][1] = e4m3x4(x[1][o + 0], x[1][o + 1], x[1][o + 2], x[1][o + 3]);
        qa[mi][s][2] = e4m3x4(x[0][o + 4], x[0][o + 5], x[0][o + 6], x[0][o + 7]);
        qa[mi][s][3] = e4m3x4(x[1][o + 4], x[1][o + 5], x[1][o + 6], x[1][o + 7]);
      }
""", """#pragma unroll
      for (int r = 0; r < 2; ++r)   // UW2: E4M3 straight into W2's Q tile: row 16w + g + 8r, dims 16*t4 .. +15 (one 16 B store)
        *reinterpret_cast<uint4*>(qt + tile_off(16 * w + g + 8 * r, 16 * t4)) =
            make_uint4(e4m3x4(x[r][0], x[r][1], x[r][2], x[r][3]), e4m3x4(x[r][4], x[r][5], x[r][6], x[r][7]),
                       e4m3x4(x[r][8], x[r][9], x[r][10], x[r][11]), e4m3x4(x[r][12], x[r][13], x[r][14], x[r][15]));
""", "qa store")
PRO = PRO[:qa0] + FOLD + PRO[qa1:]
FIN = between(UV, "  // ---- 7. this CTA's rows: merge the CS partials of its cluster", "}  // namespace okc_uv4", "finalize")
cut0 = FIN.index("  if (ncl > 1) {")
cut1 = FIN.index("  // ---- 9. RMSNorm x sigmoid(gate), BF16 out")
FIN = FIN[:cut0] + FIN[cut1:]
assert "ncl" not in FIN and "gpart" not in FIN and "tickets" not in FIN

HEAD = r'''__device__ __forceinline__ void st_cluster2(uint32_t addr, float a, float b) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(addr), "f"(a), "f"(b) : "memory");
}

template <int STAGES, int PP, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(352, 1) uw2_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int page, int window, int rope, float qk_factor, float sm_scale) {
  constexpr int SBOV = 528, WF = 0, ABL = 0, MT = 1, WC = 8;
  constexpr int PT_SLOTS = 2 * STAGES;
  constexpr int VTH = 8 * SBOV;
  constexpr int SLOTB = OFF_VT + 2 * VTH;
  constexpr int RPC = ROWS / CS;                      // output rows each CTA of the cluster finalizes
  constexpr int RPW = RPC / WC;                       // ... per consumer warp
  constexpr int LPW = L / WC;                         // block tokens per consumer warp (new rows)
  static_assert(RPC % WC == 0 && L % WC == 0, "rows and tokens split evenly over the consumer warps");
  extern __shared__ __align__(128) uint8_t sm[];     // slots (Kc | Vs | Vt) | page-entry ring | Q tiles | P tiles | RB
  int* ptr_ring = reinterpret_cast<int*>(sm + STAGES * SLOTB);
  uint8_t* q_tiles = sm + STAGES * SLOTB + PT_SLOTS * 64 * 4;
  uint8_t* p_tiles = q_tiles + 2 * 4096;
  float* rbuf = reinterpret_cast<float*>(p_tiles + 2 * 4096);   // [CS][RPC][LBP]
  __shared__ __align__(8) uint64_t land_bar[STAGES], full_bar[STAGES], empty_bar[STAGES], fresh_bar;
  __shared__ __align__(16) uint8_t new_k[L * 128];    // the block's keys / values (E4M3), token l at 128 * l,
  __shared__ __align__(16) uint8_t new_v[L * 128];    // byte 64 * h + d = head kv0 + h, dim d
  __shared__ __align__(16) uint16_t gsm[RPC * 64];    // gate rows of this CTA's output rows (BF16)
  __shared__ float rope_c[L][32], rope_s[L][32];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int h = (warp >> 2) & 1, w = warp & 3, mt0 = w;   // consumers (warps 0-7): UV4's head / m-tile = W2's h / w
  uint8_t* qt = q_tiles + h * 4096;
  uint8_t* ptl = p_tiles + h * 4096;
  const int pairs = hkv >> 1;
  const int bp = blockIdx.y;
  const int b = bp / pairs;
  const int kv0 = (bp % pairs) * 2;
  const int nsplit = CS;
  const int split = blockIdx.x;                       // grid x = CS = cluster x: blockIdx.x == cluster rank
  const uint32_t rank = cluster_rank();
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;
  // every CTA of the cluster must be running before any peer writes its shared memory: arrive now, wait just
  // before the first remote store (consumers) or after the loop (copier, transposers)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * L * qkv_row;   // block token l: qkv_b + l * qkv_row
  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l
  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const int nst = (nchunks + 3) / 4;                  // 64-key stages of this slice
  // queries are E4M3(normalized query); qk_scale_factor joins the score scale (UV4)
  const float qk_scale = sm_scale * qk_factor * (*k_scale_p) * 1.4426950408889634f;
  const int* pt = page_table + static_cast<long>(b) * max_pages;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&land_bar[s])));   // copier lanes
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" ::"r"(smem_u32(&full_bar[s])));    // transposer warps
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 8;" ::"r"(smem_u32(&empty_bar[s])));   // consumer warps
    }
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&fresh_bar)));        // consumer thread 0
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

'''

CONS_HEAD = r'''  else {   // ---- consumers (warps 0-7): UV4's prologue, then W2c's loop, then the push
'''
PUSH = r'''    // ---- the block's new rows are in new_k / new_v (bar.sync 5 above): release the transposers' patch
    if (tid == 0) bar_arrive(&fresh_bar);
@@CONS@@
    asm volatile("cp.async.wait_all;");               // this thread's gate pieces landed (read after cluster_sync)
    // ---- push: row R = 16 * token + 8h + g (token 2w + half) -> owner R / RPC, slot [rank][R % RPC]; W2's dims
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // every CTA of the cluster has started
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float l = l_run[half];
      l += __shfl_xor_sync(FULL, l, 1);
      l += __shfl_xor_sync(FULL, l, 2);
      const int R = (2 * w + half) * 16 + h * 8 + g;
      const uint32_t row = map_rank(smem_u32(rbuf), static_cast<uint32_t>(R / RPC)) +
                           static_cast<uint32_t>((static_cast<int>(rank) * RPC + R % RPC) * LBP * 4);
#pragma unroll
      for (int j = 0; j < 8; ++j) st_cluster2(row + (8 * j + 2 * t4) * 4, o[4 * j + 2 * half], o[4 * j + 2 * half + 1]);
      if (t4 == 0) {
        st_cluster(row + 64 * 4, m_run[half]);
        st_cluster(row + 65 * 4, l);
      }
    }
  }
  cluster_sync();                                     // every row's CS partials in its owner's buffer
  if (warp >= WC) return;

'''


HOST = r'''
}  // namespace okc_uw2

#define OKC_UW2_VARIANTS(X) X(4, 1, 4) X(4, 1, 8) X(4, 1, 16) X(6, 0, 16)

inline size_t okc_uw2_smem(int64_t stages) {
  return static_cast<size_t>(stages) * (okc_uw2::OFF_VT + 2 * 8 * 528) + static_cast<size_t>(2 * stages) * 64 * 4 + 4 * 4096 +
         static_cast<size_t>(okc_uw2::ROWS) * okc_uw2::LBP * 4;
}

void uw2(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc,
         torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens,
         torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t page, int64_t window,
         int64_t rope, double qk_factor, double sm_scale, int64_t stages, int64_t pp, int64_t cs) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(hq == 8 * hkv && hkv % 2 == 0, "uw2: 8 query heads per KV head, KV heads in pairs");
  TORCH_CHECK(window > 0, "uw2 serves sliding layers (one cluster per request and KV-head pair)");
  dim3 grid(static_cast<unsigned>(cs), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = okc_uw2_smem(stages);
  bool launched = false;
#define OKC_UW2_LAUNCH(S, P, C)                                                                                   \
  if (!launched && stages == S && pp == P && cs == C) {                                                           \
    okc_uw2::uw2_kernel<S, P, C><<<grid, 352, smem, stream>>>(                                                    \
        reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),  \
        reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                        \
        reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                   \
        k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(), seqlens.data_ptr<int>(), \
        reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq), static_cast<int>(hkv),               \
        static_cast<int>(max_pages), static_cast<int>(page), static_cast<int>(window), static_cast<int>(rope),   \
        static_cast<float>(qk_factor), static_cast<float>(sm_scale));                                           \
    launched = true;                                                                                              \
  }
  OKC_UW2_VARIANTS(OKC_UW2_LAUNCH)
#undef OKC_UW2_LAUNCH
  TORCH_CHECK(launched, "uw2: no variant stages=", stages, " pp=", pp, " cs=", cs);
  const cudaError_t e = cudaGetLastError();        // surface a refused (cluster) launch here, not later
  TORCH_CHECK(e == cudaSuccess, "uw2 launch failed: ", cudaGetErrorString(e));
}

int64_t uw2_setup() {   // once, outside any capture: shared memory per variant, 16-CTA clusters
  int err = 0;
#define OKC_UW2_SMEM(S, P, C)                                                                                     \
  err |= cudaFuncSetAttribute(okc_uw2::uw2_kernel<S, P, C>, cudaFuncAttributeMaxDynamicSharedMemorySize,         \
                              static_cast<int>(okc_uw2_smem(S)));                                                 \
  if (C > 8) err |= cudaFuncSetAttribute(okc_uw2::uw2_kernel<S, P, C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  OKC_UW2_VARIANTS(OKC_UW2_SMEM)
#undef OKC_UW2_SMEM
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> uw2_attrs(int64_t stages, int64_t pp, int64_t cs) {
  cudaFuncAttributes a;
  a.numRegs = -1;
  a.localSizeBytes = 0;
  a.sharedSizeBytes = 0;
#define OKC_UW2_ATTR(S, P, C) \
  if (stages == S && pp == P && cs == C) cudaFuncGetAttributes(&a, okc_uw2::uw2_kernel<S, P, C>);
  OKC_UW2_VARIANTS(OKC_UW2_ATTR)
#undef OKC_UW2_ATTR
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
'''

DECL = ("void uw2(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, "
        "torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, "
        "torch::Tensor seqlens, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t page, "
        "int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t stages, int64_t pp, int64_t cs);\n"
        "int64_t uw2_setup();\nstd::vector<int64_t> uw2_attrs(int64_t stages, int64_t pp, int64_t cs);")


def render():
    body = (HELP + UVHELP + HEAD + COPIER + TRANS + CONS_HEAD + PRO + PUSH.replace("@@CONS@@", CONS) + FIN.rstrip() + "\n" + HOST)
    for bad in ("okc_w2", "okc_uv4", "attend_w2", "uv4_kernel", "qa[", "o_part"):
        assert bad not in body, bad
    return ('"""Generated by make_uw2.py (do not edit): UW2, one launch per sliding verify layer (see make_uw2.py)."""\n\n'
            f"CUDA_UW2_SRC = r\"\"\"\n{body}\"\"\"\n\nCUDA_UW2_DECL = {DECL!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_uw2.py up to date" if same else "cuda_uw2.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
