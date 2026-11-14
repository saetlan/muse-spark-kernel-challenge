"""F12K (P43): stream-K F12M. The same 12-bit planes, decode and mma.sync.m16n8k16 math as F12M (proj/f12m/f12m.py), but
the grid is chosen by the caller (2-4 CTAs per SM) and every CTA takes an equal contiguous range of the (16-row tile,
64-k chunk) units, so the SMs carry the same bytes even when the tile count doesn't divide by the SM count (qkv: 144
tiles on 132 SMs; the unpooler slice at m <= 8: 64 tiles).

Work split (per call: n_tiles = n_rows / 16, chs = K / 64, units = n_tiles * chs, grid G):
  CTA b    units [b units / G, (b + 1) units / G): one or more segments, each a run of chunks of one tile.
  warp ks  of a segment's chunks c0 .. c0 + nseg - 1, those with (c - c0) % KSW == ks; a register ring keeps D chunks
           in flight; the next segment's first D chunks (and its row records) are loaded before this segment's epilogue.
  reduce   C fragments to part[ks][row][token] in shared memory, KSW parts summed in a fixed order.
  tile     covered by one CTA ("pieces" = 1): finalize at once (scale 2^(base - 112), the row's escapes, BF16 store).
           Covered by several: each piece stores its sum to slot [tile][piece] (FP32, L2), fence, one atomic on the
           tile's counter; the CTA that brings it to pieces finalizes from the slots summed in piece order (deterministic,
           independent of arrival order) and resets the counter to 0 for the next call. No CTA waits for another, so the
           grid needn't be co-resident.
Scratch (slots, counters) is allocated in prepare(), sized by f12k_plan; counters start at 0 and are back at 0 after every
call. check_f12k.py parses this file's helpers and emulates the kernel (partition, ring, fix-up) at the real sizes.
"""

import tempfile

F12K_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace f12k {

__device__ __forceinline__ void ld_w(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// Index helpers (check_f12k.py evaluates their return expressions): first unit of CTA b; the CTA owning unit `unit`;
// chunk of warp ks at step i of a segment starting at c0; steps of warp ks in a segment of nseg chunks; uint4 index of
// an F12 slot (F12M's tile-major planes), of an x half-chunk (k8 = K / 8); slot and word of step j's L words; C fragment
// row and token; record half; FP32 partial slot of (tile, piece, token, row).
__device__ __forceinline__ long long f12k_u0(long long b, long long units, long long grid) { return b * units / grid; }
__device__ __forceinline__ long long f12k_owner(long long unit, long long units, long long grid) { return ((unit + 1) * grid - 1) / units; }
__device__ __forceinline__ int f12k_wchunk(int c0, int ks, int i, int ksw) { return c0 + ks + ksw * i; }
__device__ __forceinline__ int f12k_nw(int nseg, int ks, int ksw) { return (nseg + ksw - 1 - ks) / ksw; }
__device__ __forceinline__ long long f12k_widx(long long tile, int chs, int c, int slot, int lane) { return ((tile * chs + c) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ long long f12k_xidx(long long tok, int k8, int c, int t, int h) { return tok * k8 + 8 * c + 2 * t + h; }
__device__ __forceinline__ int f12k_lslot(int j) { return 1 + (j >> 1); }
__device__ __forceinline__ int f12k_lword(int j, int h) { return 2 * (j & 1) + h; }
__device__ __forceinline__ int f12k_crow(int g, int r) { return g + 8 * (r >> 1); }
__device__ __forceinline__ int f12k_ctok(int n, int t, int r) { return 8 * n + 2 * t + (r & 1); }
__device__ __forceinline__ long long f12k_ridx(long long row, int h) { return row * 2 + h; }
__device__ __forceinline__ long long f12k_sidx(long long tile, int piece, int maxp, int mt, int tok, int row) { return ((tile * maxp + piece) * mt + tok) * 16 + row; }

// F12M's decode of one lane's A fragment for one k16 step: four BF16x2 registers scaled by 2^(112 - base) per row.
__device__ __forceinline__ uint4 f12k_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

__device__ __forceinline__ float f12k_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12k_wdec(uint32_t base) { return __uint_as_float(base << 23); }

template <int NT, int KSW, int D>
__global__ void __launch_bounds__(KSW * 32, (512 / (KSW * 32)) > 0 ? (512 / (KSW * 32)) : 1)
f12k_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const uint4* __restrict__ x,
            __nv_bfloat16* __restrict__ out, float* __restrict__ slots, int* __restrict__ cnt,
            int m, int n_tiles, int chs, int maxp) {
  static_assert(NT >= 1 && NT <= 4 && KSW >= 1 && KSW <= 16 && D >= 1 && D <= 8, "f12k config");
  constexpr int MT = 8 * NT;            // token slots
  constexpr int NTH = KSW * 32;
  __shared__ float part[KSW][16][MT];
  __shared__ uint4 recs[32];
  __shared__ int last_s;
  const int tid = threadIdx.x, lane = tid & 31, ks = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const long long units = static_cast<long long>(n_tiles) * chs, grid = gridDim.x, b = blockIdx.x;
  const long long u0 = f12k_u0(b, units, grid), u1 = f12k_u0(b + 1, units, grid);
  if (u0 >= u1) return;                 // only when grid > units (the wrapper refuses that); uniform per CTA
  const int k8 = 8 * chs;
  const long long n_rows = static_cast<long long>(n_tiles) * 16;
  const unsigned short* xh = reinterpret_cast<const unsigned short*>(x);
  const int mo = m < MT ? m : MT;

  bool tok_ok[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) tok_ok[n] = 8 * n + g < m;
  uint4 wv[D][3];
  uint4 xv[D][NT][2];
  auto loadw = [&](int slot, long long tl, int c) {
#pragma unroll
    for (int j = 0; j < 3; ++j) ld_w(wp + f12k_widx(tl, chs, c, j, lane), wv[slot][j]);
  };
  auto loadx = [&](int slot, int c) {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (tok_ok[n]) {
        ld_x(x + f12k_xidx(8 * n + g, k8, c, t, 0), xv[slot][n][0]);
        ld_x(x + f12k_xidx(8 * n + g, k8, c, t, 1), xv[slot][n][1]);
      } else {
        xv[slot][n][0] = make_uint4(0u, 0u, 0u, 0u);
        xv[slot][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  // first segment: tile, first chunk c0, its chunk count nseg, this warp's step count nw
  long long tile = u0 / chs;
  int c0 = static_cast<int>(u0 - tile * chs);
  const long long cend0 = u1 - tile * chs;
  int nseg = static_cast<int>((cend0 < chs ? cend0 : chs) - c0);
  int nw = f12k_nw(nseg, ks, KSW);
#pragma unroll
  for (int s = 0; s < D; ++s) {
    if (s < nw) {
      loadw(s, tile, f12k_wchunk(c0, ks, s, KSW));
      loadx(s, f12k_wchunk(c0, ks, s, KSW));
    }
  }
  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);
  if (tid < 16) {
    ld_x(rec + f12k_ridx(tile * 16 + tid, 0), rc0);
    ld_x(rec + f12k_ridx(tile * 16 + tid, 1), rc1);
  }

  while (true) {
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;

#pragma unroll 1
    for (int s0 = 0; s0 < nw; s0 += D) {
#pragma unroll
      for (int slot = 0; slot < D; ++slot) {
        const int s = s0 + slot;
        if (s < nw) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint4& lv = wv[slot][f12k_lslot(j)];
            const uint4 a = f12k_decode(word(wv[slot][0], j), word(lv, f12k_lword(j, 0)), word(lv, f12k_lword(j, 1)));
#pragma unroll
            for (int n = 0; n < NT; ++n) {
              const uint32_t b0 = word(xv[slot][n][j >> 1], (2 * j) & 3);
              const uint32_t b1 = word(xv[slot][n][j >> 1], (2 * j + 1) & 3);
              mma(acc[n], a, b0, b1);
            }
          }
          if (s + D < nw) {
            loadw(slot, tile, f12k_wchunk(c0, ks, s + D, KSW));
            loadx(slot, f12k_wchunk(c0, ks, s + D, KSW));
          }
        }
      }
    }

    // the next segment (chunk 0 of the next tile): its first loads go out before this segment's epilogue
    const long long ntile = tile + 1;
    const bool more = ntile * chs < u1;
    int nnseg = 0, nnw = 0;
    uint4 rn0 = make_uint4(0u, 0u, 0u, 0u), rn1 = make_uint4(0u, 0u, 0u, 0u);
    if (more) {
      const long long cend = u1 - ntile * chs;
      nnseg = static_cast<int>(cend < chs ? cend : chs);
      nnw = f12k_nw(nnseg, ks, KSW);
#pragma unroll
      for (int s = 0; s < D; ++s) {
        if (s < nnw) {
          loadw(s, ntile, f12k_wchunk(0, ks, s, KSW));
          loadx(s, f12k_wchunk(0, ks, s, KSW));
        }
      }
      if (tid < 16) {
        ld_x(rec + f12k_ridx(ntile * 16 + tid, 0), rn0);
        ld_x(rec + f12k_ridx(ntile * 16 + tid, 1), rn1);
      }
    }

    // this segment's epilogue
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) part[ks][f12k_crow(g, r)][f12k_ctok(n, t, r)] = acc[n][r];
    if (tid < 16) {
      recs[2 * tid] = rc0;
      recs[2 * tid + 1] = rc1;
    }
    __syncthreads();
    const long long first = f12k_owner(tile * chs, units, grid);
    const int pieces = static_cast<int>(f12k_owner(tile * chs + chs - 1, units, grid) - first + 1);
    bool fin = pieces == 1;
    if (!fin) {
      const int piece = static_cast<int>(b - first);
      for (int idx = tid; idx < 16 * mo; idx += NTH) {
        const int tok = idx / 16, row = idx % 16;
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];
        __stcg(slots + f12k_sidx(tile, piece, maxp, MT, tok, row), sum);
      }
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        __threadfence();
        const int old = atomicAdd(cnt + tile, 1);
        last_s = old == pieces - 1;
      }
      __syncthreads();
      fin = last_s != 0;
      if (fin) __threadfence();
    }
    if (fin) {
      for (int idx = tid; idx < 16 * mo; idx += NTH) {
        const int tok = idx / 16, row = idx % 16;
        float sum = 0.0f;
        if (pieces == 1) {
#pragma unroll
          for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];
        } else {
          for (int p = 0; p < pieces; ++p) sum += __ldcg(slots + f12k_sidx(tile, p, maxp, MT, tok, row));
        }
        const uint4 q0 = recs[2 * row], q1 = recs[2 * row + 1];
        const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;
        sum *= f12k_scale(base);
        const float wdec = f12k_wdec(base);
        const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int j = 0; j < 7; ++j) {
          if (j < count) {
            const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * k8) + (esc[j] & 0xFFFFu)]) << 16);
            sum = fmaf(-xk, wdec, sum);
            sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);
          }
        }
        out[static_cast<long long>(tok) * n_rows + tile * 16 + row] = __float2bfloat16_rn(sum);
      }
      if (pieces > 1 && tid == 0) cnt[tile] = 0;   // every piece has arrived: reset for the next call
    }
    __syncthreads();                    // part, recs and last_s are reused by the next segment
    if (!more) break;
    tile = ntile;
    c0 = 0;
    nseg = nnseg;
    nw = nnw;
    rc0 = rn0;
    rc1 = rn1;
  }
}

}  // namespace f12k

@@INSTANCES@@
"""

# (NT, KSW, D) instances compiled into the module.
F12K_INSTANCES = ()
# Rendered by the local clang syntax check.
F12K_CHECK_INSTANCES = ((1, 8, 4), (2, 8, 2), (1, 4, 2), (2, 4, 4), (1, 8, 1), (4, 16, 8))


def f12k_name(inst):
    nt, ksw, d = inst
    return f"f12k_run_{nt}_{ksw}_{d}"


def f12k_pieces(n_tiles, chs, grid):
    """Pieces (CTAs) per tile for a grid of `grid` CTAs over n_tiles x chs units: the host mirror of the kernel's
    f12k_owner arithmetic (check_f12k.py compares the two)."""
    units = n_tiles * chs
    assert 1 <= grid <= units, (n_tiles, chs, grid)
    owner = lambda u: ((u + 1) * grid - 1) // units  # noqa: E731
    return [owner(t * chs + chs - 1) - owner(t * chs) + 1 for t in range(n_tiles)]


def f12k_source(instances):
    wrappers = []
    for inst in instances:
        nt, ksw, d = inst
        mt, threads = 8 * nt, 32 * ksw
        tmpl = f"{nt}, {ksw}, {d}"
        wrappers.append(f"""
void {f12k_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor slots,
    torch::Tensor cnt, int64_t m, int64_t n_rows, int64_t grid, int64_t maxp) {{
  TORCH_CHECK(x.dim() == 2 && x.size(0) == m && x.size(1) % 64 == 0 && x.is_contiguous(), "f12k: x must be [m, K], K % 64 == 0");
  const int64_t chs = x.size(1) / 64, n_tiles = n_rows / 16;
  TORCH_CHECK(n_rows > 0 && n_rows % 16 == 0 && m >= 1 && m <= {mt}, "f12k: rows or tokens out of range");
  TORCH_CHECK(out.numel() == m * n_rows && out.scalar_type() == at::kBFloat16, "f12k: out has the wrong size or type");
  TORCH_CHECK(wp.numel() * wp.element_size() >= n_tiles * chs * 1536LL, "f12k: weight planes too small");
  TORCH_CHECK(rec.numel() * rec.element_size() >= n_rows * 32LL, "f12k: records too small");
  TORCH_CHECK(grid >= 1 && grid <= n_tiles * chs && grid < (1LL << 31), "f12k: grid out of range");
  int64_t need = 1;
  for (int64_t tl = 0; tl < n_tiles; ++tl) {{   // pieces per tile, as the kernel's f12k_owner computes them
    const int64_t units = n_tiles * chs;
    const int64_t a = ((tl * chs + 1) * grid - 1) / units, z = ((tl * chs + chs) * grid - 1) / units;
    need = need > z - a + 1 ? need : z - a + 1;
  }}
  TORCH_CHECK(maxp >= need, "f12k: maxp below the pieces per tile");
  TORCH_CHECK(slots.scalar_type() == at::kFloat && slots.numel() >= n_tiles * maxp * {mt} * 16, "f12k: slots too small");
  TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= n_tiles, "f12k: counters too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "f12k: unaligned");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  f12k::f12k_kernel<{tmpl}><<<static_cast<unsigned>(grid), {threads}, 0, stream>>>(
      reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()),
      reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      slots.data_ptr<float>(), cnt.data_ptr<int>(), static_cast<int>(m), static_cast<int>(n_tiles),
      static_cast<int>(chs), static_cast<int>(maxp));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
int64_t {f12k_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, f12k::f12k_kernel<{tmpl}>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f12k::f12k_kernel<{tmpl}>, {threads}, 0);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes) * 100 + occ;   // occ < 100
}}""")
    return F12K_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def f12k_decls(instances):
    out = []
    for inst in instances:
        name = f12k_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor slots, "
                   f"torch::Tensor cnt, int64_t m, int64_t n_rows, int64_t grid, int64_t maxp);")
        out.append(f"int64_t {name}_info();")
    return out


def f12k_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12k_decls(instances)
    return load_inline(
        name="okc_proj_f12k", cpp_sources="\n".join(decls), cuda_sources=f12k_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_f12k_"), verbose=False)
