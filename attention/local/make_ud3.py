"""Generates attention/local/cuda_ud3.py: UD3, the one-launch sliding-window decode kernel rebuilt from
UD2 (cuda_unified2.py) along L15's anatomy (AF32). Verified UD2 pieces are copied byte for byte
by anchors that must occur exactly once (asserted): helpers, the page-entry ring (tok_of,
pt_slot, issue_pt, issue), the new key/value block and the main loop. What changes:
1. every load that does not depend on seqlens is issued at entry, next to seqlens: positions,
   inv_freq, this lane's 32 query values (4 x 16 B instead of 32 scalar loads), the scales and
   (cluster leader only) the gate values of its output rows;
2. the RoPE table is built while the page entries are in flight, and the query fold and the new
   row while the first K/V stages are in flight (UD2 did both after the page-entry wait);
3. the CTA partial: per-(head, row) max and weights computed once by 16 threads (UD2 recomputed
   them for every output element);
4. cluster merge by push: every CTA stores its 16 x (64 o + m + l) partial straight into the
   leader's buffer (st.shared::cluster to mapa(rank 0)), one cluster barrier, then the leader
   merges from its own shared memory (UD2: leader loaded every peer's values over DSMEM with
   dependent loads, two barriers); a split barrier (arrive at entry, wait before the first remote
   store) guarantees every CTA of the cluster is running before its memory is written;
5. sliding layers only: one cluster of CS CTAs per (request, KV-head pair), no second level.
Leader buffer: dynamic shared memory after the ring, LB[CS][16][LBP] floats, LBP = 68 (64 o, m, l, 2 pad).
Run: .venv/bin/python attention/local/make_ud3.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_ud3.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


def between(src, a, b, what, include_b=False):
    assert src.count(a) == 1, f"{what}: start anchor x{src.count(a)}"
    i = src.index(a)
    j = src.index(b, i)
    assert src.count(b) >= 1
    return src[i:j + (len(b) if include_b else 0)]


UD2 = load("cuda_unified2.py", "CUDA_UD2_SRC")
HELPERS = between(UD2, "#include <torch/extension.h>", "template <int WARPS, int STAGES, int CS>", "helpers")
RING = between(UD2, "  const int* pt = page_table + static_cast<long>(b) * max_pages;\n",
               "  // Prologue: the first STAGES chunks' page entries", "ring")
NEWROW = between(UD2, "  // ---- new key/value of the pair (only the CTA holding position n-1), warp 0: 4 values per lane.\n",
                 "  // ---- queries: lane = head hl", "new row")
LOOP = between(UD2, "  float m_run[2] = {-1.0e30f, -1.0e30f};\n",
               "    __syncwarp();          // every lane done reading this stage before it is refilled\n  }\n", "loop",
               include_b=True)
HELPERS = HELPERS.replace("namespace okc_ud2 {", "namespace okc_ud3 {")
assert "okc_ud2" not in HELPERS

KERNEL = r"""constexpr int LBP = 68;     // leader buffer row: 64 outputs, m, l, 2 pad (floats)

__device__ __forceinline__ uint32_t map_rank(uint32_t local_addr, uint32_t rank) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(local_addr), "r"(rank));
  return a;
}
__device__ __forceinline__ void st_cluster(uint32_t addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}

template <int WARPS, int STAGES, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(WARPS * 32) ud3_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {
  __shared__ __align__(16) uint8_t new_k[128];
  __shared__ __align__(16) uint8_t new_v[128];
  __shared__ float rope_c[32], rope_s[32];
  constexpr int RPW = 16 / WARPS;                   // (head, row)s per warp in the leader's finalize

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row
  const uint32_t rank = cluster_rank();
  // every CTA of the cluster must be running before any peer writes its shared memory: arrive
  // now, wait just before the first remote store (step 8)
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  // ---- 1. loads that do not depend on seqlens, issued with it
  const int n = seqlens[b];
  const float pos = static_cast<float>(positions[b]);
  const float invf = (rope && warp == 0) ? inv_freq[lane] : 0.f;
  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;
  uint4 qv[4];                                          // query g of head kv0+hl, dims 32*(t4&1) .. +31
  {
    const uint4* qp4 = reinterpret_cast<const uint4*>(qkv_b + static_cast<long>((kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1));
#pragma unroll
    for (int j = 0; j < 4; ++j) qv[j] = qp4[j];
  }
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;
  const float vsc = *v_scale_p;
  float gate0[RPW], gate1[RPW];                         // leader: gate values of its output rows
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    gate0[k2] = gate1[k2] = 0.f;
    if (rank == 0) {
      const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + (kv0 * 8 + warp + k2 * WARPS) * 64;
      gate0[k2] = bf16f(gp[lane]);
      gate1[k2] = bf16f(gp[32 + lane]);
    }
  }

  // ---- 2. the slice; page entries
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA's slice holds the new token

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  float* lbuf = reinterpret_cast<float*>(dsm + WARPS * STAGES * STAGE_BYTES);   // leader buffer [CS][16][LBP]
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Token of lane (lane & 15) in the warp's chunk j (c = warp + j*WARPS), clamped into the slice.
  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };
@@RING_TAIL@@
  for (int j = 0; j < STAGES; ++j) issue_pt(j);
  asm volatile("cp.async.commit_group;");

  // ---- 3. RoPE table while the page entries are in flight (one full-range sincosf per frequency)
  if (rope) {
    if (warp == 0) sincosf(pos * invf, &rope_s[lane], &rope_c[lane]);
    __syncthreads();
  }

  // ---- 4. page entries landed: the first STAGES - 1 stages of K/V rows
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i);

  // ---- 5. while they fly: the new row (CTA holding position n-1) and the query fold
@@NEWROW@@
  uint32_t qa[8][4];
  {
    float x[32];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t w[4] = {qv[j].x, qv[j].y, qv[j].z, qv[j].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[8 * j + 2 * e] = __uint_as_float(w[e] << 16);
        x[8 * j + 2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
      }
    }
#pragma unroll
    for (int j = 0; j < 32; ++j) ss += x[j] * x[j];
    ss += __shfl_xor_sync(FULL, ss, 1);                       // the other half of the head (lane t4 ^ 1)
    const float r = rsqrtf(ss / 64.f + 1e-5f) * qk_factor;
#pragma unroll
    for (int j = 0; j < 32; ++j) x[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        const float sn = rope_s[16 * (t4 & 1) + p], cs = rope_c[16 * (t4 & 1) + p];
        const float e = x[2 * p], o = x[2 * p + 1];
        x[2 * p] = e * cs - o * sn;
        x[2 * p + 1] = e * sn + o * cs;
      }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint32_t lo2 = pack_f16x2(x[4 * ks + 0], x[4 * ks + 1]);
      const uint32_t hi2 = pack_f16x2(x[4 * ks + 2], x[4 * ks + 3]);
      qa[ks][0] = hl == 0 ? lo2 : 0u;
      qa[ks][1] = hl == 1 ? lo2 : 0u;
      qa[ks][2] = hl == 0 ? hi2 : 0u;
      qa[ks][3] = hl == 1 ? hi2 : 0u;
    }
  }
  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp

  // ---- 6. the main loop (UD2's, byte for byte)
@@LOOP@@
  asm volatile("cp.async.wait_all;");
  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it
  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 + 16 * (WARPS + 2) * 4 <= WARPS * STAGES * STAGE_BYTES,
                "merge fits the ring");
  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);
  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);
  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);
  float* sm_w = reinterpret_cast<float*>(dsm + WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4);   // [16][WARPS]
  float* sm_mm = sm_w + 16 * WARPS;                                                                // [16]
  float* sm_lt = sm_mm + 16;                                                                       // [16]

  // ---- 7. the CTA partial: warps' (m, l, o) to smem; per-(head, row) max and weights once
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  if (threadIdx.x < 16) {                              // (head h, row) = (tid >> 3, tid & 7)
    const int h = threadIdx.x >> 3, row = threadIdx.x & 7;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float lt = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      sm_w[threadIdx.x * WARPS + w] = wt;
      lt += wt * sm_l[w][h][row];
    }
    sm_mm[threadIdx.x] = mm;
    sm_lt[threadIdx.x] = lt;
  }
  __syncthreads();

  // ---- 8. push the partial into the leader's buffer slot [rank][hr][.], one cluster barrier
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");   // every CTA of the cluster has started
  const uint32_t lead = map_rank(smem_u32(lbuf), 0) + rank * (16 * LBP * 4);
  for (int idx = threadIdx.x; idx < 16 * 64; idx += WARPS * 32) {
    const int hr = idx >> 6, d = idx & 63, h = hr >> 3, row = hr & 7;
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) o += sm_w[hr * WARPS + w] * sm_o[w][h][row][d];
    st_cluster(lead + (hr * LBP + d) * 4, o);
  }
  if (threadIdx.x < 16) {
    st_cluster(lead + (threadIdx.x * LBP + 64) * 4, sm_mm[threadIdx.x]);
    st_cluster(lead + (threadIdx.x * LBP + 65) * 4, sm_lt[threadIdx.x]);
  }
  cluster_sync();                                   // every partial in the leader's buffer
  if (rank != 0) return;

  // ---- 9. leader: merge the CS partials from its own buffer, RMSNorm x sigmoid(gate), BF16 out
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    const int hr = warp + k2 * WARPS;
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
    const float scale = vsc / l;
    o0 *= scale;
    o1 *= scale;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float rn = rsqrtf(ss / 64.f + 1e-5f);
    const int head = kv0 * 8 + hr;
    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;
    op[lane] = f2bf16(o0 * rn / (1.f + __expf(-gate0[k2])));
    op[32 + lane] = f2bf16(o1 * rn / (1.f + __expf(-gate1[k2])));
  }
}

}  // namespace okc_ud3

void ud3(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc,
         torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens,
         torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t page, int64_t window,
         int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(window > 0, "ud3 serves sliding layers (one cluster per request and KV-head pair)");
  dim3 grid(static_cast<unsigned>(cs), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_ud3::STAGE_BYTES + static_cast<size_t>(cs) * 16 * okc_ud3::LBP * 4;
#define OKC_UD3_ARGS                                                                                            \
  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),     \
      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                       \
      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                  \
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(), seqlens.data_ptr<int>(), \
      reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq), static_cast<int>(hkv),              \
      static_cast<int>(max_pages), static_cast<int>(cs), static_cast<int>(page), static_cast<int>(window),    \
      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)
  if (cs == 16) {
    if (warps == 8) okc_ud3::ud3_kernel<8, 2, 16><<<grid, 256, smem, stream>>>(OKC_UD3_ARGS);
    else okc_ud3::ud3_kernel<4, 3, 16><<<grid, 128, smem, stream>>>(OKC_UD3_ARGS);
  } else {
    if (warps == 8) okc_ud3::ud3_kernel<8, 2, 8><<<grid, 256, smem, stream>>>(OKC_UD3_ARGS);
    else okc_ud3::ud3_kernel<4, 3, 8><<<grid, 128, smem, stream>>>(OKC_UD3_ARGS);
  }
#undef OKC_UD3_ARGS
  const cudaError_t e = cudaGetLastError();        // surface a refused (cluster) launch here, not later
  TORCH_CHECK(e == cudaSuccess, "ud3 launch failed: ", cudaGetErrorString(e));
}

int64_t ud3_setup() {   // once, outside any capture: each variant's ring + leader buffer, and 16-CTA clusters
  int err = 0;
  using okc_ud3::STAGE_BYTES;
  using okc_ud3::LBP;
#define OKC_UD3_SMEM(W, S, C) \
  err |= cudaFuncSetAttribute(okc_ud3::ud3_kernel<W, S, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, W * S * STAGE_BYTES + C * 16 * LBP * 4)
  OKC_UD3_SMEM(4, 3, 8);
  OKC_UD3_SMEM(8, 2, 8);
  OKC_UD3_SMEM(4, 3, 16);
  OKC_UD3_SMEM(8, 2, 16);
#undef OKC_UD3_SMEM
  err |= cudaFuncSetAttribute(okc_ud3::ud3_kernel<4, 3, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  err |= cudaFuncSetAttribute(okc_ud3::ud3_kernel<8, 2, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> ud3_attrs(int64_t warps, int64_t cs) {
  cudaFuncAttributes a;
  if (cs == 16) {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud3::ud3_kernel<8, 2, 16>);
    else cudaFuncGetAttributes(&a, okc_ud3::ud3_kernel<4, 3, 16>);
  } else {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud3::ud3_kernel<8, 2, 8>);
    else cudaFuncGetAttributes(&a, okc_ud3::ud3_kernel<4, 3, 8>);
  }
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

DECL = ("void ud3(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, "
        "torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, "
        "torch::Tensor seqlens, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, "
        "int64_t page, int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, "
        "int64_t cs);\n"
        "int64_t ud3_setup();\n"
        "std::vector<int64_t> ud3_attrs(int64_t warps, int64_t cs);")


def render():
    # the ring code after tok_of (pt_slot, issue_pt, issue): UD2's, byte for byte
    ring_tail = RING[RING.index("  // Page entries ride the ring"):]
    src = (HELPERS + KERNEL.replace("@@RING_TAIL@@\n", ring_tail).replace("@@NEWROW@@\n", NEWROW)
           .replace("@@LOOP@@\n", LOOP))
    assert "@@" not in src
    # the UD2 pieces must use only names UD3 defines
    for name in ("pt_slot", "issue_pt", "issue(", "tok_of", "has_new", "new_k", "cache_loc", "qkv_b"):
        assert name in src
    return ('"""Generated by make_ud3.py (do not edit): UD3, one-launch sliding-window decode (see make_ud3.py)."""\n\n'
            f"CUDA_UD3_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_UD3_DECL = {DECL!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_ud3.py up to date" if same else "cuda_ud3.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
